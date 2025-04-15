import asyncio
import sys
from io import StringIO

import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Discrete

from poke_env import AccountConfiguration, ServerConfiguration
from poke_env.concurrency import POKE_LOOP
from poke_env.environment import (
    AbstractBattle,
    Battle,
    Move,
    Pokemon,
    PokemonType,
    Status,
)
from poke_env.player import BattleOrder, DefaultBattleOrder, ForfeitBattleOrder, Player, PokeEnv, SinglesEnv
from poke_env.player.env import _AsyncQueue, _EnvPlayer

account_configuration1 = AccountConfiguration("username1", "password1")
account_configuration2 = AccountConfiguration("username2", "password2")
server_configuration = ServerConfiguration("server.url", "auth.url")


class CustomEnv(SinglesEnv[npt.NDArray[np.float32]]):
    def calc_reward(self, battle: AbstractBattle) -> float:
        return 69.42

    def embed_battle(self, battle: AbstractBattle) -> npt.NDArray[np.float32]:
        return np.array([0, 1, 2])


def test_init_queue():
    q = _AsyncQueue(asyncio.Queue())
    assert isinstance(q, _AsyncQueue)


def test_queue():
    q = _AsyncQueue(asyncio.Queue())
    assert q.empty()
    q.put(1)
    assert q.queue.qsize() == 1
    asyncio.get_event_loop().run_until_complete(q.async_put(2))
    assert q.queue.qsize() == 2
    item = q.get()
    q.queue.task_done()
    assert q.queue.qsize() == 1
    assert item == 1
    item = asyncio.get_event_loop().run_until_complete(q.async_get())
    q.queue.task_done()
    assert q.empty()
    assert item == 2
    asyncio.get_event_loop().run_until_complete(q.async_join())
    q.join()


def test_async_player():
    def embed_battle(battle):
        return "battle"

    player = _EnvPlayer(start_listening=False, username="usr")
    battle = Battle("bat1", player.username, player.logger, gen=8)
    player.order_queue.put(ForfeitBattleOrder())
    order = asyncio.get_event_loop().run_until_complete(player._env_move(battle))
    assert isinstance(order, ForfeitBattleOrder)
    assert embed_battle(player.battle_queue.get()) == "battle"


def test_env_reset_and_step():
    # Create a CustomEnv instance.
    env = CustomEnv(
        account_configuration1=account_configuration1,
        account_configuration2=account_configuration2,
        battle_format="gen8randombattles",
        server_configuration=server_configuration,
        start_listening=False,
        strict=False,
    )
    # --- Part 1: Test reset() ---
    # Pre-populate each agent's battle_queue with a new battle.
    battle_new1 = Battle("new_battle1", env.agent1.username, env.agent1.logger, gen=8)
    battle_new2 = Battle("new_battle2", env.agent2.username, env.agent2.logger, gen=8)
    env.agent1.battle_queue.put(battle_new1)
    env.agent2.battle_queue.put(battle_new2)
    env.agent1.battle = battle_new1
    env.agent2.battle = battle_new2

    # Call reset().
    obs, add_info = env.reset()

    # Verify that the environment's battles have been updated.
    assert env.battle1.battle_tag == "new_battle1"
    assert env.battle2.battle_tag == "new_battle2"
    # Verify that embed_battle returns the expected observations.
    np.testing.assert_array_equal(obs[env.agents[0]], np.array([0, 1, 2]))
    np.testing.assert_array_equal(obs[env.agents[1]], np.array([0, 1, 2]))
    assert add_info == {env.agents[0]: {}, env.agents[1]: {}}

    # --- Part 2: Test step() ---
    # Pre-fill the battle queues again to simulate battle updates.
    env.agent1.battle_queue.put(env.battle1)
    env.agent2.battle_queue.put(env.battle2)

    # Prepare dummy actions. Here, we use an integer action that should be
    # converted to an order via env.action_to_order (calc_reward and embed_battle
    # are defined in CustomEnv).
    actions = {env.agents[0]: np.int64(6), env.agents[1]: np.int64(6)}
    obs_step, rew, term, trunc, add_info_step = env.step(actions)
    assert not env.agent1.order_queue.empty()
    assert not env.agent2.order_queue.empty()
    env.agent1.order_queue.get()
    env.agent2.order_queue.get()

    # Check that observations are as expected.
    np.testing.assert_array_equal(obs_step[env.agents[0]], np.array([0, 1, 2]))
    np.testing.assert_array_equal(obs_step[env.agents[1]], np.array([0, 1, 2]))
    # Check that rewards match CustomEnv.calc_reward.
    assert rew[env.agents[0]] == 69.42
    assert rew[env.agents[1]] == 69.42
    # Termination and truncation flags should be False.
    assert not term[env.agents[0]]
    assert not term[env.agents[1]]
    assert not trunc[env.agents[0]]
    assert not trunc[env.agents[1]]
    # Additional info should be empty.
    assert add_info_step == {env.agents[0]: {}, env.agents[1]: {}}

    # --- Part 3: Additional Cycle: reset, step again, and then close ---
    # Simulate a new cycle by preparing new battles.
    cycle_battle1 = Battle(
        "cycle_battle1", env.agent1.username, env.agent1.logger, gen=8
    )
    cycle_battle2 = Battle(
        "cycle_battle2", env.agent2.username, env.agent2.logger, gen=8
    )
    env.agent1.battle_queue.put(cycle_battle1)
    env.agent2.battle_queue.put(cycle_battle2)

    # Call reset() again.
    obs_cycle, add_info_cycle = env.reset()
    assert not env.agent1.order_queue.empty()
    assert not env.agent2.order_queue.empty()
    order1 = env.agent1.order_queue.get()
    order2 = env.agent2.order_queue.get()
    env.agent1.battle = cycle_battle1
    env.agent2.battle = cycle_battle2

    # Verify that the environment's battles have been updated and prior battle was forfeited
    assert env.battle1.battle_tag == "cycle_battle1"
    assert env.battle2.battle_tag == "cycle_battle2"
    assert isinstance(order1, ForfeitBattleOrder)
    assert isinstance(order2, DefaultBattleOrder)
    np.testing.assert_array_equal(obs_cycle[env.agents[0]], np.array([0, 1, 2]))
    np.testing.assert_array_equal(obs_cycle[env.agents[1]], np.array([0, 1, 2]))
    assert add_info_cycle == {env.agents[0]: {}, env.agents[1]: {}}

    # Pre-fill battle queues for the next step.
    env.agent1.battle_queue.put(env.battle1)
    env.agent2.battle_queue.put(env.battle2)

    # Prepare dummy actions for the new step.
    new_actions = {env.agents[0]: np.int64(6), env.agents[1]: np.int64(6)}
    obs_cycle_step, rew_cycle, term_cycle, trunc_cycle, add_info_cycle_step = env.step(
        new_actions
    )
    assert not env.agent1.order_queue.empty()
    assert not env.agent2.order_queue.empty()
    env.agent1.order_queue.get()
    env.agent2.order_queue.get()

    # Verify the step outcome.
    np.testing.assert_array_equal(obs_cycle_step[env.agents[0]], np.array([0, 1, 2]))
    np.testing.assert_array_equal(obs_cycle_step[env.agents[1]], np.array([0, 1, 2]))
    assert rew_cycle[env.agents[0]] == 69.42
    assert rew_cycle[env.agents[1]] == 69.42
    assert not term_cycle[env.agents[0]]
    assert not term_cycle[env.agents[1]]
    assert not trunc_cycle[env.agents[0]]
    assert not trunc_cycle[env.agents[1]]
    assert add_info_cycle_step == {env.agents[0]: {}, env.agents[1]: {}}


def render(battle):
    player = CustomEnv(start_listening=False)
    captured_output = StringIO()
    sys.stdout = captured_output
    player.battle1 = battle
    player.render()
    sys.stdout = sys.__stdout__
    return captured_output.getvalue()


def test_render():
    battle = Battle("bat1", "usr", None, gen=8)
    battle._turn = 3
    active_mon = Pokemon(species="charizard", gen=8)
    active_mon._active = True
    battle._team = {"1": active_mon}
    opponent_mon = Pokemon(species="pikachu", gen=8)
    opponent_mon._active = True
    battle._opponent_team = {"1": opponent_mon}
    expected = "  Turn    3. | [●][  0/  0hp]  charizard -    pikachu [  0%hp][●]\r"
    assert render(battle) == expected
    active_mon._max_hp = 120
    active_mon._current_hp = 60
    expected = "  Turn    3. | [●][ 60/120hp]  charizard -    pikachu [  0%hp][●]\r"
    assert render(battle) == expected
    opponent_mon._current_hp = 20
    expected = "  Turn    3. | [●][ 60/120hp]  charizard -    pikachu [ 20%hp][●]\r"
    assert render(battle) == expected
    other_mon = Pokemon(species="pichu", gen=8)
    battle._team["2"] = other_mon
    expected = "  Turn    3. | [●●][ 60/120hp]  charizard -    pikachu [ 20%hp][●]\r"
    assert render(battle) == expected


def test_init():
    gymnasium_env = CustomEnv(
        account_configuration1=account_configuration1,
        account_configuration2=account_configuration2,
        server_configuration=server_configuration,
        start_listening=False,
        battle_format="gen7randombattles",
    )
    player = gymnasium_env.agent1
    assert isinstance(gymnasium_env, CustomEnv)
    assert isinstance(player, _EnvPlayer)


async def run_test_choose_move():
    player = CustomEnv(
        account_configuration1=account_configuration1,
        account_configuration2=account_configuration2,
        server_configuration=server_configuration,
        start_listening=False,
        battle_format="gen7randombattles",
        start_challenging=False,
    )
    # Create a mock battle and moves
    battle = Battle("bat1", player.agent1.username, player.agent1.logger, gen=8)
    battle._available_moves = [Move("flamethrower", gen=8)]
    # Test choosing a move
    message = await player.agent1.choose_move(battle)
    order = player.action_to_order(np.int64(6), battle)
    player.agent1.order_queue.put(order)
    assert message.message == "/choose move flamethrower"
    # Test switching Pokémon
    battle._available_switches = [Pokemon(species="charizard", gen=8)]
    message = await player.agent1.choose_move(battle)
    order = player.action_to_order(np.int64(0), battle)
    player.agent1.order_queue.put(order)
    assert message.message == "/choose switch charizard"


def test_choose_move():
    asyncio.run_coroutine_threadsafe(run_test_choose_move(), POKE_LOOP)


def test_reward_computing_helper():
    player = CustomEnv(
        account_configuration1=account_configuration1,
        account_configuration2=account_configuration2,
        server_configuration=server_configuration,
        start_listening=False,
        battle_format="gen7randombattles",
    )
    battle_1 = Battle("bat1", player.agent1.username, player.agent1.logger, gen=8)
    battle_2 = Battle("bat2", player.agent1.username, player.agent1.logger, gen=8)
    battle_3 = Battle("bat3", player.agent1.username, player.agent1.logger, gen=8)
    battle_4 = Battle("bat4", player.agent1.username, player.agent1.logger, gen=8)

    assert (
        player.reward_computing_helper(
            battle_1,
            fainted_value=0,
            hp_value=0,
            number_of_pokemons=4,
            starting_value=0,
            status_value=0,
            victory_value=1,
        )
        == 0
    )

    battle_1._won = True
    assert (
        player.reward_computing_helper(
            battle_1,
            fainted_value=0,
            hp_value=0,
            number_of_pokemons=4,
            starting_value=0,
            status_value=0,
            victory_value=1,
        )
        == 1
    )

    assert (
        player.reward_computing_helper(
            battle_2,
            fainted_value=0,
            hp_value=0,
            number_of_pokemons=4,
            starting_value=0.5,
            status_value=0,
            victory_value=5,
        )
        == -0.5
    )

    battle_2._won = False
    assert (
        player.reward_computing_helper(
            battle_2,
            fainted_value=0,
            hp_value=0,
            number_of_pokemons=4,
            starting_value=0,
            status_value=0,
            victory_value=5,
        )
        == -5
    )

    battle_3._team = {i: Pokemon(species="slaking", gen=8) for i in range(4)}
    battle_3._opponent_team = {i: Pokemon(species="slowbro", gen=8) for i in range(3)}

    battle_3._team[0].status = Status["FRZ"]
    battle_3._team[1]._current_hp = 100
    battle_3._team[1]._max_hp = 200
    battle_3._opponent_team[0].status = Status["FNT"]
    battle_3._opponent_team[1].status = Status["FNT"]

    # Opponent: two fainted, one full hp opponent
    # You: one half hp mon, one frozen mon
    assert (
        player.reward_computing_helper(
            battle_3,
            fainted_value=2,
            hp_value=3,
            number_of_pokemons=4,
            starting_value=0,
            status_value=0.25,
            victory_value=100,
        )
        == 2.25
    )

    battle_3._won = True
    assert (
        player.reward_computing_helper(
            battle_3,
            fainted_value=2,
            hp_value=3,
            number_of_pokemons=4,
            starting_value=0,
            status_value=0.25,
            victory_value=100,
        )
        == 100
    )

    battle_4._team, battle_4._opponent_team = (
        battle_3._opponent_team,
        battle_3._team,
    )
    assert (
        player.reward_computing_helper(
            battle_4,
            fainted_value=2,
            hp_value=3,
            number_of_pokemons=4,
            starting_value=0,
            status_value=0.25,
            victory_value=100,
        )
        == -2.25
    )


def test_action_space():
    player = CustomEnv(battle_format="gen7randombattle", start_listening=False)
    assert player.action_space(player.possible_agents[0]) == Discrete(18)

    for gen, (has_megas, has_z_moves, has_dynamax) in enumerate(
        [
            (False, False, False),
            (False, False, False),
            (True, False, False),
            (True, True, False),
            (True, True, True),
        ],
        start=4,
    ):
        p = SinglesEnv(
            battle_format=f"gen{gen}randombattle",
            start_listening=False,
            start_challenging=False,
        )
        assert p.action_space(p.possible_agents[0]) == Discrete(
            4 * sum([1, has_megas, has_z_moves, has_dynamax]) + 6
        )


def test_singles_action_order_conversions():
    for gen, (has_megas, has_z_moves, has_dynamax, has_tera) in enumerate(
        [
            (False, False, False, False),
            (False, False, False, False),
            (True, False, False, False),
            (True, True, False, False),
            (True, True, True, False),
            (True, True, True, True),
        ],
        start=4,
    ):
        p = SinglesEnv(
            battle_format=f"gen{gen}randombattle",
            start_listening=False,
            start_challenging=False,
        )
        battle = Battle("bat1", p.agent1.username, p.agent1.logger, gen=gen)
        active_pokemon = Pokemon(species="charizard", gen=gen)
        move = Move("flamethrower", gen=gen)
        active_pokemon._moves = {move.id: move}
        active_pokemon._active = True
        active_pokemon._item = "firiumz"
        battle._team = {"charizard": active_pokemon}
        assert p.action_to_order(np.int64(-1), battle).message == "/forfeit"
        check_action_order_roundtrip(p, ForfeitBattleOrder(), battle)
        battle._available_moves = [move]
        assert (
            p.action_to_order(np.int64(6), battle).message
            == "/choose move flamethrower"
        )
        check_action_order_roundtrip(p, Player.create_order(move), battle)
        battle._available_switches = [active_pokemon]
        assert (
            p.action_to_order(np.int64(0), battle).message == "/choose switch charizard"
        )
        check_action_order_roundtrip(p, Player.create_order(active_pokemon), battle)
        battle._available_switches = []
        assert (
            p.action_to_order(np.int64(9), battle, strict=False).message
            == "/choose default"
        )
        if has_megas:
            battle._can_mega_evolve = True
            assert (
                p.action_to_order(np.int64(6 + 4), battle).message
                == "/choose move flamethrower mega"
            )
            check_action_order_roundtrip(
                p, Player.create_order(move, mega=True), battle
            )
        if has_z_moves:
            battle._can_z_move = True
            assert (
                p.action_to_order(np.int64(6 + 4 + 4), battle).message
                == "/choose move flamethrower zmove"
            )
            check_action_order_roundtrip(
                p, Player.create_order(move, z_move=True), battle
            )
        if has_dynamax:
            battle._can_dynamax = True
            assert (
                p.action_to_order(np.int64(6 + 4 + 8), battle).message
                == "/choose move flamethrower dynamax"
            )
            check_action_order_roundtrip(
                p, Player.create_order(move, dynamax=True), battle
            )
        if has_tera:
            battle._can_tera = PokemonType.FIRE
            assert (
                p.action_to_order(np.int64(6 + 4 + 12), battle).message
                == "/choose move flamethrower terastallize"
            )
            check_action_order_roundtrip(
                p, Player.create_order(move, terastallize=True), battle
            )


def check_action_order_roundtrip(
    env: PokeEnv, order: BattleOrder, battle: AbstractBattle
):
    a = env.order_to_action(order, battle)
    o = env.action_to_order(a, battle)
    assert order.message == o.message
