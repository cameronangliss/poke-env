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
    env_player: BasePlayer
    env_player_battle: Optional[Battle]
    logger: logging.Logger

    def __init__(
        self,
        agent: BasePlayer,
        env_player: BasePlayer,
        battle_format: str,
    ):
        self.observation_space = self.describe_embedding()
        self.action_space = Discrete(26)  # type: ignore
        self.agent = agent
        self.agent.setup()
        self.env_player = env_player
        self.env_player.setup()
        self.battle_format = battle_format
        self.logger = logging.getLogger(f"{agent.username}-env")

    @abstractmethod
    def describe_embedding(self) -> Space[npt.NDArray[np.float32]]:
        pass

    @abstractmethod
    def get_reward(self, battle: Battle) -> float:
        pass

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[npt.NDArray[np.float32], Dict[str, Any]]:
        self.env_player.leave()
        while True:
            try:
                self.agent.challenge(self.env_player, self.battle_format)
                room = self.env_player.accept(self.agent)
                break
            except PopupError as e1:
                self.logger.warning(e1)
                try:
                    self.agent.cancel(self.env_player)
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
        self.agent.join(room)
        self.env_player.join(room)
        self.agent_battle = self.agent.observe()
        self.env_player_battle = self.env_player.observe()
        obs = self.agent.embed_battle(self.agent_battle)
        return obs, {}

    def step(
        self,
        action: int,
    ) -> Tuple[npt.NDArray[np.float32], float, bool, bool, Dict[str, Any]]:
        if self.agent_battle is None or self.env_player_battle is None:
            raise LookupError()
        self.agent.choose(action)
        env_player_action = self.env_player.get_action(self.env_player_battle)
        self.env_player.choose(env_player_action)
        self.agent_battle = self.agent.observe(self.agent_battle)
        self.env_player_battle = self.env_player.observe(self.env_player_battle)
        next_obs = self.agent.embed_battle(self.agent_battle)
        reward = self.get_reward(self.agent_battle)
        terminated = self.agent_battle.finished
        return next_obs, reward, terminated, False, {}

    def close(self):
        self.agent.close()
        self.env_player.close()
        # resetting logger
        for handler in logging.getLogger().handlers:
            logging.getLogger().removeHandler(handler)
