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
# Description:  CLI: harvest one polygon's FRTs from `polygons.parquet`.
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
from pathlib import Path

# local imports
from dualgnn.training.harvest import (
    MAX_NPTS_FULL_ENUM, GROW2D_TARGET, bootstrap_fts,
)
from dualgnn.training.io      import fts_path, load_polygon
from dualgnn.training.hparams import VAL_FRAC


if __name__ == "__main__":
    # CLI
    p = argparse.ArgumentParser(
        description="Harvest one polygon's FRTs from polygons.parquet.",
    )
    p.add_argument("--poly-id",       type=int,   required=True)
    p.add_argument("--polygons",      type=Path,  default=Path("polygons.parquet"),
                   dest="polygons_parquet")
    p.add_argument("--fts-dir",       type=Path,  default=Path("fts"))
    p.add_argument("--max-npts-full-enum", type=int, default=MAX_NPTS_FULL_ENUM)
    p.add_argument("--grow2d-target",      type=int, default=GROW2D_TARGET)
    p.add_argument("--val-frac",      type=float, default=VAL_FRAC)
    args = p.parse_args()

    # harvest FRTs
    pts, role    = load_polygon(args.polygons_parquet, args.poly_id)
    parquet_path = fts_path(args.poly_id, args.fts_dir)
    print(f"[harvest] poly_id={args.poly_id}  n_pts={len(pts)}  role={role}  "
          f"out={parquet_path}")
    val_frac = 1.0 if role == "val" else args.val_frac
    bootstrap_fts(
        pts, parquet_path,
        max_npts_full_enum=args.max_npts_full_enum,
        grow2d_target=args.grow2d_target,
        val_frac=val_frac, split_seed=args.poly_id,
    )
