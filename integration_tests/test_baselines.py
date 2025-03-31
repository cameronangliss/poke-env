import asyncio

import pytest

from poke_env.player import (
    MaxBasePowerPlayer,
    RandomPlayer,
    SimpleHeuristicsPlayer,
    cross_evaluate,
)


async def simple_cross_evaluation(n_battles, players):
    await cross_evaluate(players, n_challenges=n_battles)

    for player in players:
        player.reset_battles()
        await player.ps_client.stop_listening()


@pytest.mark.asyncio
async def test_random_players():
    players = [RandomPlayer(), RandomPlayer()]
    await asyncio.wait_for(
        simple_cross_evaluation(5, players=players),
        timeout=5,
    )


@pytest.mark.asyncio
async def test_random_players_in_doubles(showdown_format_teams):
    players = [
        RandomPlayer(battle_format="gen9vgc2025regg", team=showdown_format_teams["gen9vgc2025regg"][0]),
        RandomPlayer(battle_format="gen9vgc2025regg", team=showdown_format_teams["gen9vgc2025regg"][1]),
    ]
    await asyncio.wait_for(simple_cross_evaluation(100, players=players), timeout=100)


@pytest.mark.asyncio
async def test_shp():
    players = [RandomPlayer(), SimpleHeuristicsPlayer()]
    await asyncio.wait_for(simple_cross_evaluation(5, players=players), timeout=5)


@pytest.mark.asyncio
async def test_max_base_power():
    players = [RandomPlayer(), MaxBasePowerPlayer()]
    await asyncio.wait_for(simple_cross_evaluation(5, players=players), timeout=5)


@pytest.mark.asyncio
async def test_max_base_power_in_doubles():
    players = [
        RandomPlayer(battle_format="gen9randomdoublesbattle"),
        MaxBasePowerPlayer(battle_format="gen9randomdoublesbattle"),
    ]
    await asyncio.wait_for(simple_cross_evaluation(5, players=players), timeout=5)
