from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from poke_env.battle.effect import Effect
from poke_env.battle.field import Field
from poke_env.battle.move import Move
from poke_env.battle.move_category import MoveCategory
from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.status import Status

if TYPE_CHECKING:
    from poke_env.battle.abstract_battle import AbstractBattle
    from poke_env.battle.pokemon import Pokemon

STAT_NAMES = ("hp", "atk", "def", "spa", "spd", "spe")
SPEED_STAGE_MULTIPLIERS = {
    -6: (2, 8),
    -5: (2, 7),
    -4: (2, 6),
    -3: (2, 5),
    -2: (2, 4),
    -1: (2, 3),
    0: (1, 1),
    1: (3, 2),
    2: (4, 2),
    3: (5, 2),
    4: (6, 2),
    5: (7, 2),
    6: (8, 2),
}
POWER_ITEMS = {
    "poweranklet",
    "powerband",
    "powerbelt",
    "powerbracer",
    "powerlens",
    "powerweight",
}
SPEED_DEPENDENT_DAMAGE_MOVES = {
    "boltbeak",
    "electroball",
    "fishiousrend",
    "gyroball",
    "payback",
}
SPECIAL_ORDER_ITEMS = {"custapberry", "fullincense", "laggingtail", "quickclaw"}
SPECIAL_ORDER_ABILITIES = {"stall"}
TURN_WIDE_SPECIAL_ORDER_MOVES = {"afteryou", "quash"}
UNSUPPORTED_DAMAGE_INFERENCE_MOVES = {
    "beatup",
    "crushgrip",
    "endeavor",
    "finalgambit",
    "lastrespects",
    "naturesmadness",
    "nightshade",
    "ragefist",
    "ruination",
    "seismictoss",
    "shedtail",
    "substitute",
    "superfang",
    "terastarstorm",
    "wringout",
}
FULL_HP_DEFENSIVE_ABILITIES = {"multiscale", "shadowshield", "terashell"}


@dataclass
class _MoveContext:
    attacker_identifier: str
    move_id: str
    priority: Optional[int]
    counts_for_order: bool
    special_order: bool
    crit_targets: set[str] = field(default_factory=set)


class BattleStatInference:
    def __init__(self):
        self._current_move: Optional[_MoveContext] = None
        self._turn_actions: list[_MoveContext] = []
        self._turn_has_special_order_override = False

    def reset_turn(self):
        self._current_move = None
        self._turn_actions = []
        self._turn_has_special_order_override = False

    def on_move(
        self,
        battle: AbstractBattle,
        attacker_identifier: str,
        move_name: str,
        *,
        counts_for_order: bool,
    ):
        if not battle_supports_stat_inference(battle):
            return

        try:
            move = Move(Move.retrieve_id(move_name), gen=battle.gen)
        except KeyError:
            return

        attacker = battle.get_pokemon(attacker_identifier)
        context = _MoveContext(
            attacker_identifier=attacker_identifier,
            move_id=move.id,
            priority=_effective_priority(attacker, move),
            counts_for_order=counts_for_order,
            special_order=(
                self._turn_has_special_order_override
                or _has_special_move_order(attacker, move)
            ),
        )
        self._current_move = context

        if move.id in TURN_WIDE_SPECIAL_ORDER_MOVES:
            self._turn_has_special_order_override = True
            context.special_order = True

        if not counts_for_order or context.priority is None or context.special_order:
            return

        for previous in self._turn_actions:
            if previous.priority == context.priority and not previous.special_order:
                _apply_speed_order_constraint(
                    battle, previous.attacker_identifier, attacker_identifier
                )

        self._turn_actions.append(context)

    def on_critical_hit(self, battle: AbstractBattle, target_identifier: str):
        if battle_supports_stat_inference(battle) and self._current_move is not None:
            self._current_move.crit_targets.add(target_identifier)

    def on_damage(
        self,
        battle: AbstractBattle,
        event: list[str],
        *,
        previous_hp: int,
        previous_hp_scale: int,
    ):
        if (
            not battle_supports_stat_inference(battle)
            or self._current_move is None
            or len(event) != 4
        ):
            return

        try:
            move = Move(self._current_move.move_id, gen=battle.gen)
        except KeyError:
            return

        if _damage_inference_is_unsupported(
            battle, self._current_move.attacker_identifier, event[2], move, previous_hp, previous_hp_scale
        ):
            return

        attacker = battle.get_pokemon(self._current_move.attacker_identifier)
        defender = battle.get_pokemon(event[2])
        if _damage_depends_on_hidden_speed(attacker, move):
            return

        attack_source_identifier = (
            event[2] if move.id == "foulplay" else self._current_move.attacker_identifier
        )
        attack_stat = "def" if move.id == "bodypress" else _attack_stat(move)
        defense_stat = _defense_stat(move)

        attack_source = battle.get_pokemon(attack_source_identifier)

        if not (
            attack_source.has_hidden_stats or defender.has_hidden_stats
        ):
            return

        # Replay-only validation can leave both sides with hidden stat candidates
        # because OTS showteam data omits EVs, IVs, and nature. Exhaustively
        # crossing attacker, HP, and defense candidates in that setting is both
        # expensive and underconstrained, while live battle inference typically has
        # one side exact from requests/teambuilder data.
        if attack_source.has_hidden_stats and defender.has_hidden_stats:
            return

        attack_values = attack_source.stat_candidates(attack_stat)
        hp_values = defender.stat_candidates("hp")
        defense_values = defender.stat_candidates(defense_stat)
        if not attack_values or not hp_values or not defense_values:
            return

        valid_attack_values: set[int] = set()
        valid_hp_values: set[int] = set()
        valid_defense_values: set[int] = set()
        critical = event[2] in self._current_move.crit_targets

        for attack_value in attack_values:
            for hp_value in hp_values:
                for defense_value in defense_values:
                    attacker_overrides = {}
                    defender_overrides = {"hp": hp_value, defense_stat: defense_value}

                    if attack_source_identifier == self._current_move.attacker_identifier:
                        attacker_overrides[attack_stat] = attack_value
                    else:
                        defender_overrides[attack_stat] = attack_value

                    damage_range = _calculate_damage_with_overrides(
                        battle,
                        self._current_move.attacker_identifier,
                        event[2],
                        move,
                        critical,
                        attacker_overrides,
                        defender_overrides,
                    )
                    if damage_range is None:
                        continue

                    if defender.has_hidden_stats:
                        observed = _displayed_damage_matches(
                            previous_hp,
                            previous_hp_scale,
                            defender.current_hp,
                            defender.max_hp,
                            hp_value,
                            damage_range[0],
                            damage_range[1],
                        )
                    else:
                        observed_damage = previous_hp - defender.current_hp
                        observed = damage_range[0] <= observed_damage <= damage_range[1]

                    if observed:
                        valid_attack_values.add(attack_value)
                        valid_hp_values.add(hp_value)
                        valid_defense_values.add(defense_value)

        if attack_source.has_hidden_stats and valid_attack_values:
            attack_source.filter_stat_candidates(
                attack_stat, lambda value: value in valid_attack_values
            )
        if defender.has_hidden_stats and valid_hp_values:
            defender.filter_stat_candidates("hp", lambda value: value in valid_hp_values)
        if defender.has_hidden_stats and valid_defense_values:
            defender.filter_stat_candidates(
                defense_stat, lambda value: value in valid_defense_values
            )


def battle_supports_stat_inference(battle: AbstractBattle) -> bool:
    return (
        battle.gen == 9
        and battle.player_role is not None
        and battle.opponent_role is not None
        and hasattr(battle, "_opponent_active_pokemon")
        and battle._opponent_packed_team is not None
        and (battle.format is None or "vgc" in battle.format)
    )


def _apply_speed_order_constraint(
    battle: AbstractBattle, first_identifier: str, second_identifier: str
):
    first = battle.get_pokemon(first_identifier)
    second = battle.get_pokemon(second_identifier)
    first_values = first.stat_candidates("spe")
    second_values = second.stat_candidates("spe")
    if not first_values or not second_values:
        return

    trick_room = Field.TRICK_ROOM in battle.fields
    first_effective = {
        raw_speed: _effective_speed(first, battle, raw_speed) for raw_speed in first_values
    }
    second_effective = {
        raw_speed: _effective_speed(second, battle, raw_speed)
        for raw_speed in second_values
    }

    if trick_room:
        first_valid = {
            raw_speed
            for raw_speed, effective_speed in first_effective.items()
            if any(effective_speed <= other for other in second_effective.values())
        }
        second_valid = {
            raw_speed
            for raw_speed, effective_speed in second_effective.items()
            if any(other <= effective_speed for other in first_effective.values())
        }
    else:
        first_valid = {
            raw_speed
            for raw_speed, effective_speed in first_effective.items()
            if any(effective_speed >= other for other in second_effective.values())
        }
        second_valid = {
            raw_speed
            for raw_speed, effective_speed in second_effective.items()
            if any(other >= effective_speed for other in first_effective.values())
        }

    if first.has_hidden_stats and first_valid:
        first.filter_stat_candidates("spe", lambda value: value in first_valid)
    if second.has_hidden_stats and second_valid:
        second.filter_stat_candidates("spe", lambda value: value in second_valid)


def _attack_stat(move: Move) -> str:
    return "atk" if move.category == MoveCategory.PHYSICAL else "spa"


def _defense_stat(move: Move) -> str:
    hits_physical = (
        move.category == MoveCategory.PHYSICAL
        or move.entry.get("overrideDefensiveStat", "") == "def"
    )
    return "def" if hits_physical else "spd"


def _effective_priority(mon: Pokemon, move: Move) -> int:
    priority = move.priority
    if mon.ability == "prankster" and move.category == MoveCategory.STATUS:
        priority += 1
    if mon.ability == "triage" and "heal" in move.flags:
        priority += 3
    if (
        mon.ability == "galewings"
        and move.type == PokemonType.FLYING
        and mon.current_hp == mon.max_hp
    ):
        priority += 1
    return priority


def _has_special_move_order(mon: Pokemon, move: Move) -> bool:
    return (
        mon.ability in SPECIAL_ORDER_ABILITIES
        or (mon.ability == "myceliummight" and move.category == MoveCategory.STATUS)
        or mon.item in SPECIAL_ORDER_ITEMS
    )


def _effective_speed(mon: Pokemon, battle: AbstractBattle, raw_speed: int) -> int:
    speed = raw_speed
    stage_num, stage_den = SPEED_STAGE_MULTIPLIERS[mon.boosts["spe"]]
    speed = math.floor(speed * stage_num / stage_den)

    if Effect.SLOW_START in mon.effects:
        speed = math.floor(speed / 2)
    if mon.item == "choicescarf":
        speed = math.floor(speed * 3 / 2)
    if mon.item in POWER_ITEMS or mon.item == "ironball":
        speed = math.floor(speed / 2)
    if mon.base_species == "ditto" and mon.item == "quickpowder":
        speed *= 2

    if mon.status == Status.PAR and mon.ability != "quickfeet":
        speed = math.floor(speed / 2)
    elif mon.ability == "quickfeet" and mon.status is not None:
        speed = math.floor(speed * 3 / 2)

    side_conditions = (
        battle.side_conditions
        if mon in battle.team.values()
        else battle.opponent_side_conditions
    )
    if SideCondition.TAILWIND in side_conditions:
        speed *= 2

    if mon.ability == "chlorophyll" and any(
        weather.name in {"SUNNYDAY", "DESOLATELAND"} for weather in battle.weather
    ):
        speed *= 2
    elif mon.ability == "swiftswim" and any(
        weather.name in {"RAINDANCE", "PRIMORDIALSEA"} for weather in battle.weather
    ):
        speed *= 2
    elif mon.ability == "sandrush" and any(
        weather.name == "SANDSTORM" for weather in battle.weather
    ):
        speed *= 2
    elif mon.ability == "slushrush" and any(
        weather.name in {"HAIL", "SNOW"} for weather in battle.weather
    ):
        speed *= 2
    elif mon.ability == "surgesurfer" and Field.ELECTRIC_TERRAIN in battle.fields:
        speed *= 2

    if (
        Effect.PROTOSYNTHESISSPE in mon.effects
        or Effect.QUARKDRIVESPE in mon.effects
    ):
        speed = math.floor(speed * 3 / 2)

    return speed


def _damage_depends_on_hidden_speed(attacker: Pokemon, move: Move) -> bool:
    return move.id in SPEED_DEPENDENT_DAMAGE_MOVES or attacker.ability == "analytic"


def _damage_inference_is_unsupported(
    battle: AbstractBattle,
    attacker_identifier: str,
    defender_identifier: str,
    move: Move,
    previous_hp: int,
    previous_hp_scale: int,
) -> bool:
    defender = battle.get_pokemon(defender_identifier)
    if move.category == MoveCategory.STATUS or move.base_power <= 0:
        return True
    if move.id in UNSUPPORTED_DAMAGE_INFERENCE_MOVES:
        return True
    if move.n_hit != (1, 1):
        return True
    if defender.current_hp == 1:
        return True
    return (
        previous_hp == previous_hp_scale
        and defender.ability in FULL_HP_DEFENSIVE_ABILITIES
    )


def _calculate_damage_with_overrides(
    battle: AbstractBattle,
    attacker_identifier: str,
    defender_identifier: str,
    move: Move,
    critical: bool,
    attacker_overrides: dict[str, int],
    defender_overrides: dict[str, int],
) -> Optional[tuple[int, int]]:
    from poke_env.calc.damage_calc_gen9 import calculate_damage

    attacker = battle.get_pokemon(attacker_identifier)
    defender = battle.get_pokemon(defender_identifier)
    attacker_previous = _apply_stat_overrides(attacker, attacker_overrides)
    defender_previous = _apply_stat_overrides(defender, defender_overrides)
    try:
        return calculate_damage(
            attacker_identifier, defender_identifier, move, battle, critical
        )
    except AssertionError:
        return None
    finally:
        _restore_stats(attacker, attacker_previous)
        _restore_stats(defender, defender_previous)


def _apply_stat_overrides(mon: Pokemon, overrides: dict[str, int]) -> dict[str, Optional[int]]:
    previous: dict[str, Optional[int]] = {}
    for stat in STAT_NAMES:
        if stat in overrides or mon.stats[stat] is None:
            previous[stat] = mon.stats[stat]
            if stat in overrides:
                mon._stats[stat] = overrides[stat]
            else:
                candidates = mon.stat_candidates(stat)
                if not candidates:
                    continue
                mon._stats[stat] = candidates[0]
    return previous


def _restore_stats(mon: Pokemon, previous: dict[str, Optional[int]]):
    for stat, value in previous.items():
        mon._stats[stat] = value


def _displayed_damage_matches(
    previous_hp: int,
    previous_scale: int,
    current_hp: int,
    current_scale: int,
    actual_max_hp: int,
    damage_min: int,
    damage_max: int,
) -> bool:
    if previous_scale != current_scale:
        return False

    old_band = _hp_display_band(previous_hp, previous_scale, actual_max_hp)
    new_band = _hp_display_band(current_hp, current_scale, actual_max_hp)
    if old_band is None or new_band is None:
        return False
    if current_hp == 0:
        return old_band[0] <= damage_max

    min_pre_hp = max(old_band[0], new_band[0] + damage_min)
    max_pre_hp = min(old_band[1], new_band[1] + damage_max)
    return min_pre_hp <= max_pre_hp


def _hp_display_band(
    displayed_hp: int, displayed_scale: int, actual_max_hp: int
) -> Optional[tuple[int, int]]:
    if displayed_hp == 0:
        return (0, 0)
    if displayed_scale <= 0:
        return None
    lower = math.floor(actual_max_hp * (displayed_hp - 1) / displayed_scale) + 1
    upper = math.floor(actual_max_hp * displayed_hp / displayed_scale)
    lower = max(1, lower)
    upper = min(actual_max_hp, upper)
    if lower > upper:
        return None
    return lower, upper
