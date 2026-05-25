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
# Description:  Storage layer for per-polygon FRT pools.
# -----------------------------------------------------------------------------

# external imports
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import polars as pl
import torch


# local imports
from .hparams import POLYGONS_PARQUET, FTS_DIR

# polygon
# =======
def load_polygons(
    parquet: Path = POLYGONS_PARQUET, id: int | None = None,
) -> tuple[np.ndarray, str] | tuple[pl.DataFrame, pl.DataFrame]:
    """
    Load `polygons.parquet`.

    If `id` is given, returns that single polygon's `(pts, role)`. Else
    returns the full table split by role: `(train_df, val_df)`.

    Parameters
    ----------
    parquet : Path, optional
        Path to the polygons parquet. Default `POLYGONS_PARQUET`.
    id : int or None, optional
        If given, return only this polygon. Default `None`.

    Returns
    -------
    pts, role : ndarray, str
        Only if `id` is given. `pts` is `(Npts, 2)` int64; `role` is
        `"train"` or `"val"`.
    train, val : pl.DataFrame, pl.DataFrame
        Only if `id` is `None`. Rows split by `role`.
    """
    df = pl.read_parquet(parquet)

    # return particular polygon
    if id is not None:
        row = df.filter(pl.col("id") == id)
        if row.is_empty():
            raise KeyError(f"polygon id={id} not in {parquet}")

        pts  = np.asarray(row["pts"][0].to_list(), dtype=np.int64)
        role = str(row["role"][0])
        return pts, role

    # return all data
    train = df.filter(pl.col("role") == "train")
    val   = df.filter(pl.col("role") == "val")
    return train, val

# triangulation
# =============
def fts_path(poly_id: int, fts_dir: Path = FTS_DIR) -> Path:
    """
    Conventional path for one polygon's FRT parquet.
    """
    return fts_dir / f"poly_{poly_id:04d}.parquet"

def save_fts(
    simps: np.ndarray,
    split: np.ndarray,
    path:  Path,
) -> None:
    """
    Save `(simps, split)` to per-polygon parquet. All entries are FRTs
    (irregular FTs are dropped upstream at harvest time).

    Parameters
    ----------
    simps : ndarray
        `(Nft, N_simps_per_ft, 3)` int. Each row is one FRT.
    split : ndarray
        `(Nft,)` object. Per-FRT split: `"train"` or `"val"`.
    path : Path
        Destination parquet file.

    Notes
    -----
    Output schema:
        simps  List<List<Int32>>  (N_simps_per_ft, 3) point indices
        split  String             ("train" | "val")
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(
        {
            "simps": [s.astype(np.int32).tolist() for s in simps],
            "split": [str(s) for s in split],
        },
        schema={
            "simps": pl.List(pl.List(pl.Int32)),
            "split": pl.String,
        },
    )
    df.write_parquet(path)

def load_fts(
    path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load `(simps, split)` from per-polygon parquet.

    Parameters
    ----------
    path : Path
        Source parquet file (as produced by `save_fts`).

    Returns
    -------
    simps : ndarray
        `(Nft, N_simps_per_ft, 3)` int8.
    split : ndarray
        `(Nft,)` object. Per-FRT `"train"` / `"val"`.
    """
    df         = pl.read_parquet(path)
    simps_list = [np.asarray(s, dtype=np.int8) for s in df["simps"].to_list()]
    split      = df["split"].to_numpy()

    if not simps_list:
        return np.empty((0, 0, 3), dtype=np.int8), split
    return np.stack(simps_list), split

# checkpoints
# -----------
def save_ckpt(path: Path, *, net, step: int, optim, hparams) -> None:
    """
    Save a training checkpoint to `path`. Persists model + optimizer state
    and torch / Python / numpy *global* RNG state. On resume that makes the
    AR-rollout draws (`torch.multinomial`) replay bit-for-bit, but the
    per-purpose `np.random.Generator`s on `Trainer` (polygon pick / batch /
    eval / explore) are re-seeded from `cfg.seed` at `__init__` and do NOT
    replay their post-ckpt stream.
    """
    torch.save({
        "state_dict":       net.state_dict(),
        "step":             step,
        "optim_state":      optim.state_dict(),
        "rng_state":        torch.get_rng_state(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state":  np.random.get_state(),
        "hparams":          hparams,
    }, path)

def load_ckpt(path: Path, device: str) -> dict:
    """
    Load a training checkpoint produced by `save_ckpt`.
    """
    return torch.load(path, map_location=device, weights_only=False)
