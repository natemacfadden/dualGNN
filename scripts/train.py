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
# Description:  CLI wrapper around `dualgnn.training.train`; flags auto-
#               derived from `TrainConfig` fields.
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

# Per-field --help text. Only fields with non-obvious behavior (e.g. auto-
# derived defaults) need an entry; the rest fall back to no help string.
HELP = {
    "poly_ids":
        "comma-separated polygon ids to restrict training to "
        "(default: all polygons in --src-polygons). Affects the auto-"
        "defaults of --explore-new-frac and --compile-model below.",
    "explore_new_frac":
        "probability per explore round of introducing a brand-new random "
        "polygon (auto: 0.0 if --poly-ids is set, else 0.20).",
    "compile_model":
        "wrap the model with torch.compile(dynamic=True) "
        "(auto: enabled iff --poly-ids has exactly one id, since a single "
        "polygon means a fixed graph shape that compiles well).",
}

def add_cfg_args(parser, cfg_cls):
    """Auto-add one CLI flag per dataclass field on `cfg_cls`."""
    for f in fields(cfg_cls):
        flag = f"--{f.name.replace('_', '-')}"
        kwargs = {"dest": f.name, "default": argparse.SUPPRESS}
        if f.name in HELP:
            kwargs["help"] = HELP[f.name]
        if f.type == "bool | None":
            parser.add_argument(
                flag, action=argparse.BooleanOptionalAction, **kwargs,
            )
        else:
            if f.type not in CLI_TYPES:
                raise TypeError(
                    f"TrainConfig.{f.name}: type {f.type!r} has no CLI_TYPES "
                    f"entry in scripts/train.py. Add one (or change the field "
                    f"to a supported type)."
                )
            parser.add_argument(
                flag, type=CLI_TYPES[f.type], **kwargs,
            )

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Multi-polygon DualGNN trainer with curriculum + "
                    "exploration.",
    )
    add_cfg_args(p, TrainConfig)
    train(**vars(p.parse_args()))
