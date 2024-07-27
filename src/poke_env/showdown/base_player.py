from __future__ import annotations

import json
import logging
from abc import abstractmethod

import numpy as np
import numpy.typing as npt
import requests

from poke_env.environment import Battle
from poke_env.showdown.client import Client, MessageType


class BasePlayer(Client):
    username: str
    password: str | None
    logger: logging.Logger
    room: str | None

    def __init__(self, username: str, password: str | None):
        super().__init__(username)
        self.password = password or ""

    @abstractmethod
    def get_action(self, battle: Battle) -> int | None:
        pass

    @abstractmethod
    def embed_battle(self, battle: Battle) -> npt.NDArray[np.float32]:
        pass

    def setup(self):
        self.room = None
        self.login()
        self.forfeit_games()

    def close(self):
        self.leave()
        self.logout()

    ###################################################################################
    # Commands to be used by Player when communicating with PokemonShowdown website

    def login(self):
        split_message = self.find_message(MessageType.LOGIN)
        client_id = split_message[2]
        challstr = split_message[3]
        response = requests.post(
            "https://play.pokemonshowdown.com/api/login",
            {
                "name": self.username,
                "pass": self.password,
                "challstr": f"{client_id}|{challstr}",
            },
        )
        response_json = json.loads(response.text[1:])
        assertion = response_json.get("assertion")
        self.send_message(f"/trn {self.username},0,{assertion}")

    def forfeit_games(self):
        # The first games message is always empty, so this is here to pass by that message.
        self.find_message(MessageType.GAMES)
        try:
            split_message = self.find_message(MessageType.GAMES, timeout=5)
        except TimeoutError:
            self.logger.info(
                "Second updatesearch message not received. This should mean the user just logged in."
            )
        else:
            games = json.loads(split_message[2])["games"]
            if games:
                Battlerooms = list(games.keys())
                prev_room = self.room
                for room in Battlerooms:
                    self.join(room)
                    self.send_message("/forfeit")
                    self.leave()
                if prev_room:
                    self.join(prev_room)

    def set_avatar(self, avatar: str):
        self.send_message(f"/avatar {avatar}")

    def challenge(
        self, opponent: BasePlayer, Battleformat: str, team: str | None = None
    ):
        self.send_message(f"/utm {team}")
        self.send_message(f"/challenge {opponent.username}, {Battleformat}")
        # Waiting for confirmation that challenge was sent
        self.find_message(MessageType.CHALLENGE)

    def cancel(self, opponent: BasePlayer):
        self.send_message(f"/cancelchallenge {opponent.username}")
        # Waiting for confirmation that challenge was cancelled
        self.find_message(MessageType.CANCEL)

    def accept(self, opponent: BasePlayer, team: str | None = None) -> str:
        # Waiting for confirmation that challenge was received
        self.find_message(MessageType.ACCEPT)
        self.send_message(f"/utm {team}")
        self.send_message(f"/accept {opponent.username}")
        # The first games message is always empty, so this is here to pass by that message.
        self.find_message(MessageType.GAMES)
        split_message = self.find_message(MessageType.GAMES)
        games = json.loads(split_message[2])["games"]
        room = list(games.keys())[0]
        return room

    def join(self, room: str):
        self.send_message(f"/join {room}")
        self.room = room

    def timer_on(self):
        self.send_message("/timer on")

    def observe(self, battle: Battle | None = None) -> Battle:
        split_message = self.find_message(MessageType.OBSERVE)
        if split_message[1] == "request":
            request = json.loads(split_message[2])
            protocol = self.find_message(MessageType.OBSERVE)
        else:
            request = None
            protocol = split_message
        if battle:
            battle.parse_message(protocol)
        else:
            battle_tag = "-".join(split_message)[1:]
            logger = logging.getLogger(battle_tag)
            battle = Battle(
                battle_tag=battle_tag,
                username=self.username,
                logger=logger,
                gen=9,
            )
            battle.parse_request(request or {})
            battle.parse_message(split_message)
        return battle

    def choose(self, action: int | None):
        action_space = (
            [f"switch {i}" for i in range(1, 7)]
            + [f"move {i}" for i in range(1, 5)]
            + [f"move {i} mega" for i in range(1, 5)]
            + [f"move {i} zmove" for i in range(1, 5)]
            + [f"move {i} max" for i in range(1, 5)]
            + [f"move {i} terastallize" for i in range(1, 5)]
        )
        if action is not None:
            self.send_message(f"/choose {action_space[action]}")

    def leave(self):
        if self.room:
            self.send_message(f"/leave {self.room}")
            # gets rid of all messages having to do with the room being left
            self.find_message(MessageType.LEAVE)
            self.room = None

    def logout(self):
        self.send_message("/logout")
