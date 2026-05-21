# =============================================================================
#    Copyright (C) 2026  Nate MacFadden
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
# =============================================================================
#
# -----------------------------------------------------------------------------
# Description:  CLI wrapper around `dualgnn.training.train`. Flags are
#               auto-derived from `TrainConfig` fields.
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path

# local imports
from dualgnn.training import train, TrainConfig


csv_ints = lambda s: [int(x) for x in s.split(",") if x.strip()]

CLI_TYPES = {
    "int":              int,
    "float":            float,
    "str":              str,
    "Path":             Path,
    "float | None":     float,
    "list[int] | None": csv_ints,
}

def add_cfg_args(parser, cfg_cls):
    """Auto-add one CLI flag per dataclass field on `cfg_cls`."""
    for f in fields(cfg_cls):
        flag = f"--{f.name.replace('_', '-')}"
        if f.type == "bool | None":
            parser.add_argument(
                flag, action=argparse.BooleanOptionalAction,
                dest=f.name, default=argparse.SUPPRESS,
            )
        else:
            parser.add_argument(
                flag, type=CLI_TYPES[f.type], dest=f.name,
                default=argparse.SUPPRESS,
            )

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Multi-polygon DualGNN trainer with curriculum + "
                    "exploration.",
    )
    add_cfg_args(p, TrainConfig)
    train(**vars(p.parse_args()))
