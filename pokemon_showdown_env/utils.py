# -*- coding: utf-8 -*-
"""Utils file

This file contains utility functions and objects.

Functions:
    compute_type_chart: Returns a dictionnary representing the Pokemon type chart,
        loaded from a json file in `data`. This dictionnary is computed in this file as
        TYPE_CHART.

Attributes:
    TYPE_CHART (Dict[str, Dict[str, float]]): A dictionnary representing the Pokemon
        type chart. Each key is a string representing a type (corresponding to `Type`
        names), and each value is a dictionnary whose keys are string representation of
        types, and whose values are floats. TYPE_CHART[type_1][type_2] corresponds to
        the damage multiplier of an attack of type_1 on a Pokemon of type_2. This
        dictionnary is computed using the `compute_type_chart` function.
"""

import json

from pokemon_showdown_env.constants import TYPE_CHART_PATH
from typing import Dict


def compute_type_chart() -> Dict[str, Dict[str, float]]:
    """Returns the pokemon type_chart.

    Returns a dictionnary representing the Pokemon type chart, loaded from a json file
    in `data`. This dictionnary is computed in this file as TYPE_CHART.

    Returns:
        type_chart (Dict[str, Dict[str, float]]): the pokemon type chart
    """
    with open(TYPE_CHART_PATH) as chart:
        json_chart = json.load(chart)

    types = [entry["name"].upper() for entry in json_chart]

    type_chart = {type_1: {type_2: 1 for type_2 in types} for type_1 in types}

    for entry in json_chart:
        type_ = entry["name"].upper()
        for immunity in entry["immunes"]:
            type_chart[type_][immunity.upper()] = 0
        for weakness in entry["weaknesses"]:
            type_chart[type_][weakness.upper()] = 0.5
        for strength in entry["strengths"]:
            type_chart[type_][strength.upper()] = 2

    return type_chart


TYPE_CHART = compute_type_chart()
"""(Dict[str, Dict[str, float]]): A dictionnary representing the Pokemon type chart.

Each key is a string representing a type (corresponding to `Type` names), and each value
is a dictionnary whose keys are string representation of types, and whose values are
floats.

TYPE_CHART[type_1][type_2] corresponds to the damage multiplier of an attack of type_1
on a Pokemon of type_2. This dictionnary isncomputed using the `compute_type_chart`
function.
"""
