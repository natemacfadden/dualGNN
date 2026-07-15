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
# Description:  CLI wrapper around `dualgnn.training.reinforce`.
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib  import Path

# local imports
# (training pulls in CYTools/polars/tensorboard... fail with guidance)
try:
    from dualgnn.training import reinforce
except ImportError as e:
    import sys
    sys.exit(
        "dualGNN training requires the training extras (CYTools, polars, "
        "tensorboard). Install them with:\n"
        "    pip install dualgnn[train]\n"
        "(a bare pip install is inference-only.)\n"
        f"[original import error: {e}]"
    )


DEFAULT_RUN_PATH = (
    Path("runs") / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
)


if __name__ == "__main__":
    # CLI
    p = argparse.ArgumentParser(
        description="REINFORCE fine-tune a dualgnn AR sampler.",
    )
    p.add_argument("--init-ckpt",       type=Path,  required=True,
                   help="warm-start ckpt to initialize from (its parent dir "
                        "provides polygons.parquet and fts/)")
    p.add_argument("--run-path",        type=Path,  default=DEFAULT_RUN_PATH,
                   help="output dir for ckpts + TB logs")
    p.add_argument("--steps",           type=int,   default=10000)
    p.add_argument("--batch",           type=int,   default=4)
    p.add_argument("--lr",              type=float, default=3e-5)
    p.add_argument("--invalid-reward",  type=float, default=-2.0)
    p.add_argument("--val-every",       type=int,   default=200)
    p.add_argument("--val-Ntriangs",    type=int,   default=512)
    p.add_argument("--val-Npolys",      type=int,   default=4,
                   help="number of val polys to sample per validation pass")
    p.add_argument("--ckpt-every",      type=int,   default=500)
    p.add_argument("--device",          type=str,   default=None)
    p.add_argument("--seed",            type=int,   default=0)

    # run REINFORCE
    reinforce(**vars(p.parse_args()))
