import asyncio
import logging
import time
from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt
from gymnasium import Env
from gymnasium.spaces import Discrete, Space

from poke_env.environment import Battle
from poke_env.showdown.base_player import BasePlayer
from poke_env.showdown.client import PopupError


class BaseEnv(Env[npt.NDArray[np.float32], int]):
    agent: BasePlayer
    agent_battle: Optional[Battle]
    agent_request: Any | None
    env_player: BasePlayer
    env_player_battle: Optional[Battle]
    env_player_request: Any | None
    logger: logging.Logger
    loop: asyncio.AbstractEventLoop

    def __init__(
        self,
        agent: BasePlayer,
        env_player: BasePlayer,
        battle_format: str,
    ):
        self.loop = asyncio.get_event_loop()
        self.loop.create_task(self._init_async(agent, env_player, battle_format))

    async def _init_async(
        self, agent: BasePlayer, env_player: BasePlayer, battle_format: str
    ):
        self.observation_space = self.describe_embedding()
        self.action_space = Discrete(26)  # type: ignore
        self.agent = agent
        await self.agent.setup()
        self.env_player = env_player
        await self.env_player.setup()
        self.battle_format = battle_format
        self.logger = logging.getLogger(f"{agent.username}-env")

    @abstractmethod
    def describe_embedding(self) -> Space[npt.NDArray[np.float32]]:
        pass

    @abstractmethod
    def get_reward(self, battle: Battle) -> float:
        pass

    def get_action_str(self, battle: Battle, action: Optional[int]) -> Optional[str]:
        if action is None:
            return None
        elif battle.available_moves and battle.available_moves[0].id == "recharge":
            return "move recharge"
        elif battle.available_moves and battle.available_moves[0].id == "struggle":
            return "move struggle"
        elif action < 6:
            return f"switch {list(battle.team.values())[action].species}"
        assert battle.active_pokemon is not None
        if action < 10:
            return f"move {list(battle.active_pokemon.moves.values())[action - 6].id}"
        elif action < 14:
            return f"move {list(battle.active_pokemon.moves.values())[action - 10].id} mega"
        elif action < 18:
            return f"move {list(battle.active_pokemon.moves.values())[action - 14].id} zmove"
        elif action < 22:
            return (
                f"move {list(battle.active_pokemon.moves.values())[action - 18].id} max"
            )
        else:
            return f"move {list(battle.active_pokemon.moves.values())[action - 22].id} tera"

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[npt.NDArray[np.float32], Dict[str, Any]]:
        return self.loop.run_until_complete(self._reset_async(seed, options))

    async def _reset_async(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[npt.NDArray[np.float32], Dict[str, Any]]:
        await self.env_player.leave()
        while True:
            try:
                await self.agent.challenge(self.env_player, self.battle_format)
                room = await self.env_player.accept(self.agent)
                break
            except PopupError as e1:
                self.logger.warning(e1)
                try:
                    await self.agent.cancel(self.env_player)
                except PopupError as e2:
                    self.logger.warning(e2)
                if (
                    "Due to spam from your internet provider, you can't challenge others right now."
                    in str(e1)
                ):
                    self.logger.info("Waiting for 5 hours to be allowed back in...")
                    time.sleep(5 * 60 * 60)
                else:
                    time.sleep(5)
        await self.agent.join(room)
        await self.env_player.join(room)
        self.agent_battle, self.agent_request = await self.agent.observe()
        self.env_player_battle, self.env_player_request = (
            await self.env_player.observe()
        )
        obs = self.agent.embed_battle(self.agent_battle)
        return obs, {}

    def step(
        self,
        action: int,
    ) -> Tuple[npt.NDArray[np.float32], float, bool, bool, Dict[str, Any]]:
        return self.loop.run_until_complete(self._async_step(action))

    async def _async_step(
        self,
        action: int,
    ) -> Tuple[npt.NDArray[np.float32], float, bool, bool, Dict[str, Any]]:
        if self.agent_battle is None or self.env_player_battle is None:
            raise LookupError()
        agent_action = (
            None
            if self.agent_request is None or "wait" in self.agent_request
            else action
        )
        agent_rqid = None if self.agent_request is None else self.agent_request["rqid"]
        await self.agent.choose(
            self.get_action_str(self.agent_battle, agent_action), agent_rqid
        )
        env_player_action = (
            None
            if self.env_player_request is None or "wait" in self.env_player_request
            else self.env_player.get_action(self.env_player_battle)
        )
        env_player_rqid = (
            None if self.env_player_request is None else self.env_player_request["rqid"]
        )
        await self.env_player.choose(
            self.get_action_str(self.env_player_battle, env_player_action),
            env_player_rqid,
        )
        self.agent_battle, self.agent_request = await self.agent.observe(
            self.agent_battle
        )
        self.env_player_battle, self.env_player_request = await self.env_player.observe(
            self.env_player_battle
        )
        next_obs = self.agent.embed_battle(self.agent_battle)
        reward = self.get_reward(self.agent_battle)
        terminated = self.agent_battle.finished
        return next_obs, reward, terminated, False, {}

    def close(self):
        self.loop.run_until_complete(self._async_close())

    async def _async_close(self):
        await self.agent.close()
        await self.env_player.close()
        # resetting logger
        for handler in logging.getLogger().handlers:
            logging.getLogger().removeHandler(handler)
