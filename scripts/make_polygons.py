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
# Description:  CLI wrapper around
#               `dualgnn.training.polygons.write_random_polygons`.
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
from pathlib import Path

# local imports
from dualgnn.training.polygons import (
    write_random_polygons,
    NPOLYS_PER_BUCKET, NPTS_MIN, NPTS_MAX, MAX_ATTEMPTS,
)
from dualgnn.training.hparams  import VAL_POLY_FRAC


if __name__ == "__main__":
    # CLI
    p = argparse.ArgumentParser(
        description="Generate a random sample of 2D lattice polygons.",
    )
    p.add_argument("--Npolys-per",         type=int,   default=NPOLYS_PER_BUCKET,
                   dest="Npolys_per_bucket")
    p.add_argument("--Npts-min",         type=int,   default=NPTS_MIN)
    p.add_argument("--Npts-max",         type=int,   default=NPTS_MAX)
    p.add_argument("--max-attempts",  type=int,   default=MAX_ATTEMPTS)
    p.add_argument("--val-poly-frac", type=float, default=VAL_POLY_FRAC)
    p.add_argument("--seed",          type=int,   default=0)
    p.add_argument("--out",           type=Path,  default=Path("polygons.parquet"))
    p.add_argument("--verbose",       action=argparse.BooleanOptionalAction,
                   default=True,
                   help="print QHull / degenerate-polygon warnings emitted "
                        "when random draws are rejected (default: on; pass "
                        "--no-verbose to silence)")
    p.add_argument("--force",         action="store_true",
                   help="overwrite --out if it already exists "
                        "(default: refuse, to avoid desyncing the dataset "
                        "from any harvested fts/poly_XXXX.parquet files)")

    # generate random polygons
    write_random_polygons(**vars(p.parse_args()))
