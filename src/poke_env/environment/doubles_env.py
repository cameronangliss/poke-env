from typing import Optional, Union

import numpy as np
import numpy.typing as npt
from gymnasium.spaces import MultiDiscrete

from poke_env.battle.double_battle import DoubleBattle
from poke_env.battle.pokemon import Pokemon
from poke_env.environment.env import ObsType, PokeEnv
from poke_env.player import (
    BattleOrder,
    DefaultBattleOrder,
    DoubleBattleOrder,
    ForfeitBattleOrder,
    SingleBattleOrder,
)
from poke_env.player.player import Player
from poke_env.ps_client import (
    AccountConfiguration,
    LocalhostServerConfiguration,
    ServerConfiguration,
)
from poke_env.teambuilder import Teambuilder


class DoublesEnv(PokeEnv[ObsType, npt.NDArray[np.int64]]):
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
        fake: bool = False,
        strict: bool = True,
    ):
        super().__init__(
            account_configuration1=account_configuration1,
            account_configuration2=account_configuration2,
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
            fake=fake,
            strict=strict,
        )
        num_switches = 6
        num_moves = 4
        num_targets = 5
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
        act_size = 1 + num_switches + num_moves * num_targets * (num_gimmicks + 1)
        self.action_spaces = {
            agent: MultiDiscrete([act_size, act_size]) for agent in self.possible_agents
        }

    @staticmethod
    def action_to_order(
        action: npt.NDArray[np.int64],
        battle: DoubleBattle,
        fake: bool = False,
        strict: bool = True,
    ) -> BattleOrder:
        """
        Returns the BattleOrder relative to the given action.

        The action is a list in doubles, and the individual action mapping is as follows:
        element = -2: default
        element = -1: forfeit
        element = 0: pass
        1 <= element <= 6: switch
        7 <= element <= 11: move 1
        12 <= element <= 16: move 2
        17 <= element <= 21: move 3
        22 <= element <= 26: move 4
        27 <= element <= 31: move 1 and mega evolve
        32 <= element <= 36: move 2 and mega evolve
        37 <= element <= 41: move 3 and mega evolve
        42 <= element <= 46: move 4 and mega evolve
        47 <= element <= 51: move 1 and z-move
        52 <= element <= 56: move 2 and z-move
        57 <= element <= 61: move 3 and z-move
        62 <= element <= 66: move 4 and z-move
        67 <= element <= 71: move 1 and dynamax
        72 <= element <= 76: move 2 and dynamax
        77 <= element <= 81: move 3 and dynamax
        82 <= element <= 86: move 4 and dynamax
        87 <= element <= 91: move 1 and terastallize
        92 <= element <= 96: move 2 and terastallize
        97 <= element <= 101: move 3 and terastallize
        102 <= element <= 106: move 4 and terastallize

        :param action: The action to take.
        :type action: ndarray[int64]
        :param battle: The current battle state
        :type battle: AbstractBattle

        :return: The battle order for the given action in context of the current battle.
        :rtype: BattleOrder

        """
        if action[0] == -1 or action[1] == -1:
            return ForfeitBattleOrder()
        if not fake and (
            len(battle.available_switches[0]) == 1
            and battle.force_switch == [True, True]
            and 1 <= action[0] <= 6
            and 1 <= action[1] <= 6
        ):
            if strict:
                raise ValueError()
            else:
                return DefaultBattleOrder()
        try:
            order1 = DoublesEnv._action_to_order_individual(action[0], battle, fake, 0)
            order2 = DoublesEnv._action_to_order_individual(action[1], battle, fake, 1)
        except ValueError as e:
            if strict:
                raise e
            else:
                return DefaultBattleOrder()
        joined_orders = DoubleBattleOrder.join_orders(
            [order1] if order1 is not None else [],
            [order2] if order2 is not None else [],
        )
        if not fake and not joined_orders:
            if strict:
                raise ValueError(
                    f"Invalid action {action} from player {battle.player_username} "
                    f"in battle {battle.battle_tag} - converted orders {order1} "
                    f"and {order2} are incompatible!"
                )
            else:
                return DefaultBattleOrder()
        return joined_orders[0]

    @staticmethod
    def _action_to_order_individual(
        action: np.int64, battle: DoubleBattle, fake: bool, pos: int
    ) -> Optional[SingleBattleOrder]:
        if action == -2:
            return DefaultBattleOrder()
        elif action == 0:
            order = None
        elif action < 7:
            order = Player.create_order(list(battle.team.values())[action - 1])
        else:
            active_mon = battle.active_pokemon[pos]
            if not fake and battle.force_switch[pos]:
                raise ValueError(
                    f"Invalid action {action} from player {battle.player_username} "
                    f"in battle {battle.battle_tag} at position {pos} - action "
                    f"specifies a move, but battle.force_switch[pos] is True!"
                )
            if active_mon is None:
                raise ValueError(
                    f"Invalid order from player {battle.player_username} "
                    f"in battle {battle.battle_tag} at position {pos} - action "
                    f"specifies a move, but battle.active_pokemon is None!"
                )
            mvs = (
                battle.available_moves[pos]
                if len(battle.available_moves[pos]) == 1
                and battle.available_moves[pos][0].id in ["struggle", "recharge"]
                else list(active_mon.moves.values())
            )
            if (action - 7) % 20 // 5 not in range(len(mvs)):
                raise ValueError(
                    f"Invalid action {action} from player {battle.player_username} "
                    f"in battle {battle.battle_tag} at position {pos} - action "
                    f"specifies a move but the move index {(action - 7) % 20 // 5} "
                    f"is out of bounds for available moves {mvs}!"
                )
            order = Player.create_order(
                mvs[(action - 7) % 20 // 5],
                move_target=(action.item() - 7) % 5 - 2,
                mega=(action - 7) // 20 == 1,
                z_move=(action - 7) // 20 == 2,
                dynamax=(action - 7) // 20 == 3,
                terastallize=(action - 7) // 20 == 4,
            )
        print(battle.player_username, battle.valid_orders[pos])
        if not fake and order not in battle.valid_orders[pos]:
            raise ValueError(
                f"Invalid action {action} from player {battle.player_username} "
                f"in battle {battle.battle_tag} at position {pos} - "
                f"order {order} not in action space {battle.valid_orders[pos]}!"
            )
        return order

    @staticmethod
    def order_to_action(
        order: BattleOrder,
        battle: DoubleBattle,
        fake: bool = False,
        strict: bool = True,
    ) -> npt.NDArray[np.int64]:
        """
        Returns the action relative to the given BattleOrder.

        :param order: The order to take.
        :type order: BattleOrder
        :param battle: The current battle state
        :type battle: AbstractBattle

        :return: The action for the given battle order in context of the current battle.
        :rtype: ndarray[int64]
        """
        if isinstance(order, DefaultBattleOrder):
            return np.array([-2, -2])
        elif isinstance(order, ForfeitBattleOrder):
            return np.array([-1, -1])
        assert isinstance(order, DoubleBattleOrder)
        if not fake and (
            len(battle.available_switches[0]) == 1
            and battle.force_switch == [True, True]
            and order.first_order is not None
            and isinstance(order.first_order.order, Pokemon)
            and order.second_order is not None
            and isinstance(order.second_order.order, Pokemon)
        ):
            if strict:
                raise ValueError()
            else:
                return np.array([-2, -2])
        try:
            action1 = DoublesEnv._order_to_action_individual(
                order.first_order, battle, fake, 0
            )
            action2 = DoublesEnv._order_to_action_individual(
                order.second_order, battle, fake, 1
            )
            return np.array([action1, action2])
        except ValueError as e:
            if strict:
                raise e
            else:
                return np.array([-2, -2])

    @staticmethod
    def _order_to_action_individual(
        order: Optional[SingleBattleOrder], battle: DoubleBattle, fake: bool, pos: int
    ) -> np.int64:
        if order is None:
            action = 0
        elif isinstance(order, DefaultBattleOrder):
            return np.int64(-2)
        else:
            assert isinstance(order, SingleBattleOrder)
            assert not isinstance(order.order, str), "invalid order"
            if isinstance(order.order, Pokemon):
                action = [p.base_species for p in battle.team.values()].index(
                    order.order.base_species
                ) + 1
            else:
                active_mon = battle.active_pokemon[pos]
                if not fake and battle.force_switch[pos]:
                    raise ValueError(
                        f"Invalid order {order} from player {battle.player_username} "
                        f"in battle {battle.battle_tag} at position {pos} - order "
                        f"specifies a move, but battle.force_switch[pos] is True!"
                    )
                if active_mon is None:
                    raise ValueError(
                        f"Invalid order {order} from player {battle.player_username} "
                        f"in battle {battle.battle_tag} at position {pos} - type of "
                        f"order.order is Move, but battle.active_pokemon[pos] is None!"
                    )
                mvs = (
                    battle.available_moves[pos]
                    if len(battle.available_moves[pos]) == 1
                    and battle.available_moves[pos][0].id in ["struggle", "recharge"]
                    else list(active_mon.moves.values())
                )
                if order.order.id not in [m.id for m in mvs]:
                    raise ValueError(
                        f"Invalid order {order} from player {battle.player_username} "
                        f"in battle {battle.battle_tag} at position {pos} - order "
                        f"specifies a move but the move {order.order.id} is not in "
                        f"available moves {mvs}!"
                    )
                action = [m.id for m in mvs].index(order.order.id)
                target = order.move_target + 2
                if order.mega:
                    gimmick = 1
                elif order.z_move:
                    gimmick = 2
                elif order.dynamax:
                    gimmick = 3
                elif order.terastallize:
                    gimmick = 4
                else:
                    gimmick = 0
                action = 1 + 6 + 5 * action + target + 20 * gimmick
        if not fake and order not in battle.valid_orders[pos]:
            raise ValueError(
                f"Invalid order from player {battle.player_username} "
                f"in battle {battle.battle_tag} at position {pos} - order "
                f"{order} not in action space {battle.valid_orders[pos]}!"
            )
        return np.int64(action)
