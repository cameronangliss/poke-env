"""This module contains objects related to player configuration."""

from __future__ import annotations

import random
import string
from typing import NamedTuple, Optional


class AccountConfiguration(NamedTuple):
    """Player configuration object. Represented with a tuple with two entries: username and
    password."""

    username: str
    password: Optional[str]
    char_space: str = string.ascii_lowercase + string.digits

    @classmethod
    def generate_config(cls, length: int) -> AccountConfiguration:
        username = "".join(random.choices(cls.char_space, k=length))
        return AccountConfiguration(username, None)
