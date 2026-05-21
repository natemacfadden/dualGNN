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
# Description:  Harvest (i.e., generate) fine triangulations (FTs) of a polygon.
#               Small polygons enumerate exactly via CYTools; large polygons
#               sample via grow2d.
# -----------------------------------------------------------------------------

# external imports
from __future__ import annotations

from pathlib import Path

import numpy as np
from cytools import Polytope
# pplpy (loaded transitively by CYTools) leaves the FPU in FE_UPWARD,
# which corrupts float math globally (e.g. AdamW underflow -> OverflowError).
import ctypes; ctypes.CDLL(None).fesetround(0)  # FE_TONEAREST

# local imports
from ..          import grow2d
from ..geometry  import canonical_simps, is_regular
from .io         import load_fts, save_fts
from .hparams    import VAL_FRAC

# defaults
# ========
MAX_NPTS_FULL_ENUM = 17       # CYTools-full above this -> grow2d
GROW2D_TARGET      = 10_000

# main methods
# ============
def harvest_fts(
    pts: np.ndarray,
    *,
    max_npts_full_enum: int = MAX_NPTS_FULL_ENUM,
    grow2d_target:      int = GROW2D_TARGET,
) -> np.ndarray:
    """
    Run the full harvest for one polygon. Returns only FRTs.

    The CYTools branch enumerates FRTs directly. The grow2d branch samples fine
    FTs (regular + irregular) and filters to FRTs post-hoc.

    Parameters
    ----------
    pts : ndarray
        `(Npts, 2)` int. Lattice points of the polygon.

    max_npts_full_enum : int, optional
        `len(pts) <= max_npts_full_enum` -> CYTools full enumeration; else
        grow2d sampling. Default `MAX_NPTS_FULL_ENUM`.
    grow2d_target : int, optional
        Number of unique fine FTs to gather in the grow2d branch before
        filtering to FRTs. Default `GROW2D_TARGET`.

    Returns
    -------
    simps : ndarray
        `(Nft, N_simps_per_ft, 3)` int8. FRTs only.
    """
    if len(pts) <= max_npts_full_enum:
        return harvest_full(pts)
    all_simps = harvest_grow2d(pts, grow2d_target)
    is_reg    = _classify_regularity(pts, all_simps)
    return all_simps[is_reg]

def bootstrap_fts(
    pts:          np.ndarray,
    parquet_path: Path,
    *,
    max_npts_full_enum: int   = MAX_NPTS_FULL_ENUM,
    grow2d_target:      int   = GROW2D_TARGET,
    val_frac:           float = VAL_FRAC,
    split_seed:         int   = 0,
    verbose:            bool  = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load FRTs from `parquet_path` if it exists; otherwise harvest, save, and
    return.

    Parameters
    ----------
    pts : ndarray
        `(Npts, 2)` int. Lattice points of the polygon.
    parquet_path : Path
        Per-polygon FRT parquet. Loaded if it exists, else written.

    max_npts_full_enum : int, optional
        CYTools/grow2d cutoff (see `harvest_fts`). Default `MAX_NPTS_FULL_ENUM`.
    grow2d_target : int, optional
        Number of unique FTs in the grow2d branch. Default `GROW2D_TARGET`.
    val_frac : float, optional
        Fraction reserved for `"val"`. Pass `1.0` to hold the entire polygon out
        (every FRT labeled `"val"`). Default `VAL_FRAC`.
    split_seed : int, optional
        Seed for the split RNG. Default 0.
    verbose : bool, optional
        Print progress. Default True.

    Returns
    -------
    simps : ndarray
        `(Nft, N_simps_per_ft, 3)` int8. FRTs only.
    split : ndarray
        `(Nft,)` object. Per-FRT `"train"` / `"val"`.
    """
    # check if we can simply load and return
    if parquet_path.exists():
        if verbose:
            print(f"[bootstrap] loading FRTs from {parquet_path}")
        return load_fts(parquet_path)

    # have to actually do some work...
    if verbose:
        print(f"[bootstrap] harvesting FRTs for polygon "
              f"(n_pts={len(pts)}, val_frac={val_frac})...")
    
    simps = harvest_fts(
        pts,
        max_npts_full_enum=max_npts_full_enum,
        grow2d_target=grow2d_target,
    )

    # split by train/val
    rng    = np.random.default_rng(split_seed)
    split  = np.where(rng.random(len(simps)) < val_frac,
                      "val", "train").astype(object)

    # save it
    save_fts(simps, split, parquet_path)

    # verbosity/printing
    if verbose:
        method = ("cytools_full" if len(pts) <= max_npts_full_enum
                                 else f"grow2d_n{len(simps)}")

        Nval = int((split == "val").sum())
        print(f"[bootstrap] {len(simps)} FRTs "
              f"({Nval} val, {len(simps) - Nval} train) via {method} "
              f"-> {parquet_path}")

    # return
    return simps, split

# harvest backends
# ================
def harvest_full(pts_input: np.ndarray) -> np.ndarray:
    """
    Enumerate every fine FT of `pts_input` via CYTools.

    Parameters
    ----------
    pts_input : ndarray
        `(Npts, 2)` int. Lattice points of the polygon.

    Returns
    -------
    simps : ndarray
        `(Nft, N_simps_per_ft, 3)` int8. Simp point indices remapped to the
        input `pts_input` ordering (CYTools reorders points).
    """
    poly = Polytope(pts_input)
    pts  = np.asarray(poly.points(), dtype=np.int64)

    # reindexing maps
    coord_to_idx = {tuple(p.tolist()): i for i, p in enumerate(pts_input)}
    pts_to_input = np.array(
        [coord_to_idx[tuple(p.tolist())] for p in pts],
        dtype=np.int8,
    )

    # generate the triangulations
    simps_list = []
    for s in poly.all_triangulations(
        only_fine=True, only_regular=True, only_star=False,
        include_points_interior_to_facets=True, raw_output=True,
    ):
        arr = np.asarray(s, dtype=np.int8)
        arr = pts_to_input[arr]      # remap to our pts ordering
        simps_list.append(canonical_simps(arr))

    # return
    if not simps_list:
        return np.empty((0, 0, 3), dtype=np.int8)
    return np.stack(simps_list)

def harvest_grow2d(
    pts:      np.ndarray,
    N_target: int,
    *,
    seed:                int = 0,
    max_attempts_factor: int = 10,
) -> np.ndarray:
    """
    Sample fine FTs of `pts` via grow2d.

    Samples until `N_target` unique deduped FTs are found, or
    `max_attempts_factor * N_target` draws are exhausted.

    Parameters
    ----------
    pts : ndarray
        `(Npts, 2)` int. Lattice points of the polygon.
    N_target : int
        Number of unique deduped FTs to find.

    seed : int, optional
        Seed for the RNG. Default 0.
    max_attempts_factor : int, optional
        Caps draws at `max_attempts_factor * N_target`. Default 10.

    Returns
    -------
    simps : ndarray
        `(Nft, N_simps_per_ft, 3)` int8. Each row is a canonicalized FT.
    """
    rng = np.random.default_rng(seed)

    # generate the triangulations
    seen = {}

    attempts     = 0
    max_attempts = max_attempts_factor * N_target

    while len(seen) < N_target and attempts < max_attempts:
        simps_raw, status = grow2d(pts, seed=int(rng.integers(2**31 - 1)))
        attempts += 1

        if status != 0:
            continue # skip bad generations

        # canonicalize and save
        canon = canonical_simps(np.asarray(simps_raw, dtype=np.int8))
        key   = canon.tobytes()
        if key in seen:
            continue
        seen[key] = canon

        # printing
        if len(seen) % 100 == 0:
            print(f"  grow2d: seen={len(seen):>6}/{N_target}  "
                  f"attempts={attempts}", end="\r", flush=True)

    print(f"  grow2d: seen={len(seen):>6}/{N_target}  "
          f"attempts={attempts}", flush=True)

    # return
    if not seen:
        return np.empty((0, 0, 3), dtype=np.int8)
    return np.stack(list(seen.values()))

# regularity checking
# ===================
def _classify_regularity(
    pts:   np.ndarray,
    simps: np.ndarray,
) -> np.ndarray:
    """
    Per-FT `is_regular` flag. Used to filter the grow2d output to regulars.

    Parameters
    ----------
    pts : ndarray
        `(Npts, 2)` int. Lattice points of the polygon.
    simps : ndarray
        `(Nft, N_simps_per_ft, 3)` int. Each row is one FT.

    Returns
    -------
    flags : ndarray
        `(Nft,)` bool. `flags[t]` True iff FT `t` is regular.
    """
    if len(simps) == 0:
        return np.zeros(0, dtype=bool)
    return np.array([bool(is_regular(pts, s)) for s in simps], dtype=bool)
