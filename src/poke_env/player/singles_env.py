from typing import Optional, Union

import numpy as np
from bidict import bidict
from gymnasium.spaces import Discrete

from poke_env.environment import Battle, Pokemon
from poke_env.player.battle_order import BattleOrder, ForfeitBattleOrder
from poke_env.player.gymnasium_api import ObsType, PokeEnv
from poke_env.player.player import Player
from poke_env.ps_client import (
    AccountConfiguration,
    LocalhostServerConfiguration,
    ServerConfiguration,
)
from poke_env.teambuilder import Teambuilder


class SinglesEnv(PokeEnv[ObsType, np.int64]):
    def __init__(
        self,
        account_configuration1: Optional[AccountConfiguration] = None,
        account_configuration2: Optional[AccountConfiguration] = None,
        avatar: Optional[int] = None,
        battle_format: str = "gen8randombattle",
        log_level: Optional[int] = None,
        save_replays: Union[bool, str] = False,
        server_configuration: Optional[
            ServerConfiguration
        ] = LocalhostServerConfiguration,
        accept_open_team_sheet: Optional[bool] = False,
        start_timer_on_battle_start: bool = False,
        start_listening: bool = True,
        open_timeout: Optional[float] = 10.0,
        ping_interval: Optional[float] = 20.0,
        ping_timeout: Optional[float] = 20.0,
        team: Optional[Union[str, Teambuilder]] = None,
        start_challenging: bool = False,
    ):
        super().__init__(
            account_configuration1,
            account_configuration2,
            avatar=avatar,
            battle_format=battle_format,
            log_level=log_level,
            save_replays=save_replays,
            server_configuration=server_configuration,
            accept_open_team_sheet=accept_open_team_sheet,
            start_timer_on_battle_start=start_timer_on_battle_start,
            start_listening=start_listening,
            open_timeout=open_timeout,
            ping_interval=ping_interval,
            ping_timeout=ping_timeout,
            team=team,
            start_challenging=start_challenging,
        )
        num_switches = 6
        num_moves = 4
        if battle_format.startswith("gen6"):
            num_gimmicks = 1
        elif battle_format.startswith("gen7"):
            num_gimmicks = 2
        elif battle_format.startswith("gen8"):
            num_gimmicks = 3
        elif battle_format.startswith("gen9"):
            num_gimmicks = 4
        else:
            num_gimmicks = 0
        act_size = num_switches + num_moves * (num_gimmicks + 1)
        self.action_spaces = {
            agent: Discrete(act_size) for agent in self.possible_agents
        }

    @staticmethod
    def action_to_order(action: np.int64, battle: Battle) -> BattleOrder:
        """
        Returns the BattleOrder relative to the given action.

        The action mapping is as follows:
        action = -1: forfeit
        0 <= action <= 5: switch
        6 <= action <= 9: move
        10 <= action <= 13: move and mega evolve
        14 <= action <= 17: move and z-move
        18 <= action <= 21: move and dynamax
        22 <= action <= 25: move and terastallize

        :param action: The action to take.
        :type action: int64
        :param battle: The current battle state
        :type battle: AbstractBattle

        :return: The battle order for the given action in context of the current battle.
        :rtype: BattleOrder
        """
        try:
            action_order_map = SinglesEnv.get_action_order_map(battle)
            order = action_order_map[action]
            if SinglesEnv.is_valid_order(order, battle):
                return order
            else:
                return Player.choose_random_move(battle)
        except IndexError:
            return Player.choose_random_move(battle)

    @staticmethod
    def order_to_action(order: BattleOrder, battle: Battle) -> np.int64:
        """
        Returns the action relative to the given BattleOrder.

        :param order: The order to take.
        :type order: BattleOrder
        :param battle: The current battle state
        :type battle: AbstractBattle

        :return: The action for the given battle order in context of the current battle.
        :rtype: int64
        """
        assert SinglesEnv.is_valid_order(order, battle), "invalid choice"
        action_order_map = SinglesEnv.get_action_order_map(battle)
        return action_order_map.inv[order]

    @staticmethod
    def get_action_order_map(
        battle: Battle,
    ) -> bidict[np.int64, BattleOrder]:
        num_switches = 6
        num_moves = 4
        num_gimmicks = 5
        action_order_map: bidict[np.int64, BattleOrder] = bidict()
        action_order_map[np.int64(-1)] = ForfeitBattleOrder()
        for switch in range(num_switches):
            action_order_map[np.int64(switch)] = Player.create_order(
                list(battle.team.values())[switch]
            )
        active_mon = battle.active_pokemon
        if active_mon is not None:
            mvs = (
                battle.available_moves
                if len(battle.available_moves) == 1
                and battle.available_moves[0].id in ["struggle", "recharge"]
                else list(active_mon.moves.values())
            )
            for gimmick in range(num_gimmicks):
                for move in range(num_moves):
                    action_order_map[
                        np.int64(num_switches + num_moves * gimmick + move)
                    ] = Player.create_order(
                        mvs[move],
                        mega=gimmick == 1,
                        z_move=gimmick == 2,
                        dynamax=gimmick == 3,
                        terastallize=gimmick == 4,
                    )
        return action_order_map

    @staticmethod
    def is_valid_order(order: BattleOrder, battle: Battle) -> bool:
        active_mon = battle.active_pokemon
        if order.order is None:
            return False
        elif isinstance(order.order, Pokemon):
            return order.order in battle.available_switches
        else:
            return (
                not battle.force_switch
                and active_mon is not None
                and order.order.id in [m.id for m in battle.available_moves]
                and (not order.mega or battle.can_mega_evolve)
                and (not order.z_move or battle.can_z_move)
                and (not order.dynamax or battle.can_dynamax)
                and (not order.terastallize or battle.can_tera is not False)
            )
