import numpy as np
from gymnasium.spaces import Box
from pettingzoo.test.parallel_test import parallel_api_test

from poke_env import LocalhostServerConfiguration
from poke_env.player import PokeEnv


class TestEnv(PokeEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observation_spaces = {
            agent: Box(
                low=np.array([1, 1, float("-inf")]),
                high=np.array([2, 4, float("+inf")]),
                dtype=np.float64,
            )
            for agent in self.possible_agents
        }

    def embed_battle(self, battle):
        return np.array([1.0, 2.0, 3.0], dtype=np.float64)

    def calc_reward(self, battle):
        return 0.25


def gymnasium_api():
    gymnasium_env = TestEnv(
        battle_format="gen8randombattle",
        server_configuration=LocalhostServerConfiguration,
        start_challenging=True,
    )
    parallel_api_test(gymnasium_env)
    gymnasium_env.close()


if __name__ == "__main__":
    gymnasium_api()