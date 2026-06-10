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
# Description:  Generate polygons.parquet: random 2D lattice polygons with
#               Npts in [Npts_min, Npts_max].
# -----------------------------------------------------------------------------

# external imports
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import polars as pl

# local imports
from ..geometry import random_lattice_polygon
from .hparams   import VAL_POLY_FRAC


# defaults
# ========
NPOLYS_PER_BUCKET = 3
NPTS_MIN          = 5
NPTS_MAX          = 40
MAX_ATTEMPTS      = 50_000


# main method
# ===========
def write_random_polygons(
    *,
    Npolys_per_bucket: int   = NPOLYS_PER_BUCKET,
    Npts_min:          int   = NPTS_MIN,
    Npts_max:          int   = NPTS_MAX,
    max_attempts:      int   = MAX_ATTEMPTS,
    val_poly_frac:     float = VAL_POLY_FRAC,
    seed:              int   = 0,
    out:               Path  = Path("polygons.parquet"),
    verbose:           bool  = True,
    force:             bool  = False,
):
    """
    Generate `polygons.parquet` with `Npolys_per_bucket` polygons per `Npts`
    bucket in `[Npts_min, Npts_max]`, assigning each a "train" or "val" role.

    Parameters
    ----------
    Npolys_per_bucket : int, optional
        Polygons per `Npts` bucket. Default 3.
    Npts_min, Npts_max : int, optional
        Inclusive `Npts` range to cover. Defaults 5, 40.
    max_attempts : int, optional
        Cap on random-vertex draws across all buckets. Default 50,000.
    val_poly_frac : float, optional
        Fraction of polygons reserved as held-out (role `"val"`). Default
        `VAL_POLY_FRAC`.
    seed : int, optional
        Seed for the RNG. Default 0.
    out : Path, optional
        Output parquet path. Default `polygons.parquet` (CWD-relative).
    verbose : bool, optional
        If False, suppress the QHull / degenerate-polygon warnings emitted
        when random draws are rejected. Default True.
    force : bool, optional
        If True, overwrite `out` if it already exists. Default False
        (refuse and exit, to avoid silently desyncing an existing
        polygons.parquet from any harvested `fts/poly_XXXX.parquet` files
        that index into it).
    """
    # refuse to clobber an existing polygons.parquet (FRT pools index into
    # it by id; a silent overwrite can desync the on-disk dataset).
    out = Path(out)
    if out.exists() and not force:
        sys.exit(f"{out} already exists; pass --force to overwrite "
                 f"(or --out PATH to write elsewhere)")

    # prep
    rng          = np.random.default_rng(seed)
    buckets      = {n: [] for n in range(Npts_min, Npts_max + 1)} # for Npts
    seen         = set()
    target_total = Npolys_per_bucket * (Npts_max - Npts_min + 1)
    attempts     = 0
    found        = 0

    # sample buckets
    while True:
        underfilled = [n for n, b in buckets.items() if len(b) < Npolys_per_bucket]
        if not underfilled:
            break
        if attempts >= max_attempts:
            print(f"[warn] exhausted {max_attempts} attempts; only "
                  f"{found}/{target_total} polygons found", flush=True)
            break

        # weight bucket pick by Npts: large polygons are both the dualGNN
        # target regime and the hardest to land via random hull draws
        weights  = np.asarray(underfilled, dtype=float)
        weights /= weights.sum()
        target_Npts = int(rng.choice(underfilled, p=weights))

        # generate a random polygon
        pts = random_lattice_polygon(
            rng, target_Npts=target_Npts,
            Npts_min=Npts_min, Npts_max=Npts_max,
            verbose=verbose,
        )
        attempts += 1

        # skip if unacceptable
        if pts is None:
            continue
        Npts = len(pts)
        if len(buckets[Npts]) >= Npolys_per_bucket:
            continue
        key = pts.tobytes()
        if key in seen:
            continue

        # accept
        seen.add(key)
        buckets[Npts].append(pts)
        found += 1
        if found % 25 == 0:
            print(f"  found={found}/{target_total}  attempts={attempts}",
                  flush=True)

    # flatten buckets to columnar rows
    rows = []
    for n in sorted(buckets):
        for pts in buckets[n]:
            role = "val" if rng.random() < val_poly_frac else "train"
            rows.append({
                "id":    len(rows),
                "n_pts": n,
                "role":  role,
                "pts":   pts.astype(np.int32).tolist(),
            })

    df = pl.DataFrame(rows, schema={
        "id":    pl.Int32,
        "n_pts": pl.Int32,
        "role":  pl.String,
        "pts":   pl.List(pl.List(pl.Int32)),
    })
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)

    n_val = int((df["role"] == "val").sum())
    print(f"\nwrote {df.height} polygons to {out}  "
          f"({n_val} val, {df.height - n_val} train)")
    print("\nNpts -> count")
    for n in sorted(buckets):
        c = len(buckets[n])
        print(f"  Npts={n:3d}: {c}  {'#' * c}")
