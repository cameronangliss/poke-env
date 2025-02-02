import numpy as np
import pytest
from gymnasium.spaces import Box
from pettingzoo.test.parallel_test import parallel_api_test
from gymnasium.utils.env_checker import check_env

from poke_env.player import SinglesEnv, SingleAgentWrapper, RandomPlayer


class SinglesTestEnv(SinglesEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observation_spaces = {
            agent: Box(np.array([0]), np.array([1]), dtype=np.int64)
            for agent in self.possible_agents
        }

    def calc_reward(self, battle) -> float:
        return 0.0

    def embed_battle(self, battle):
        return np.array([0])


def play_function(env, n_battles):
    for _ in range(n_battles):
        done = False
        env.reset()
        while not done:
            actions = {name: env.action_space(name).sample() for name in env.agents}
            _, _, terminated, truncated, _ = env.step(actions)
            done = any(terminated.values()) or any(truncated.values())


def single_agent_play_function(env: SingleAgentWrapper, n_battles: int):
    for _ in range(n_battles):
        done = False
        env.reset()
        while not done:
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated


@pytest.mark.timeout(60)
def test_env_run():
    for gen in range(4, 10):
        env = SinglesTestEnv(
            battle_format=f"gen{gen}randombattle", log_level=25, start_challenging=False
        )
        env.start_challenging(3)
        play_function(env, 3)
        env.close()
        env = SingleAgentWrapper(env, RandomPlayer())
        env.env.start_challenging(3)
        single_agent_play_function(env, 3)
    env = SinglesTestEnv(
        battle_format="gen8randombattle", log_level=25, start_challenging=False
    )
    env.start_challenging(2)
    play_function(env, 2)
    env.start_challenging(2)
    play_function(env, 2)
    env.close()
    env = SinglesTestEnv(
        battle_format="gen9randombattle", log_level=25, start_challenging=False
    )
    env.start_challenging(2)
    play_function(env, 2)
    env.start_challenging(2)
    play_function(env, 2)
    env.close()


@pytest.mark.timeout(60)
def test_env_api():
    for gen in range(4, 10):
        env = SinglesTestEnv(
            battle_format=f"gen{gen}randombattle", log_level=25, start_challenging=True
        )
        parallel_api_test(env)
        env.close()
        env = SingleAgentWrapper(env, RandomPlayer())
        check_env(env)
        env.close()
