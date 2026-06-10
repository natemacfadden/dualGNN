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

from pathlib import Path
import pickle
import random

import numpy as np
import polars as pl
import torch


# local imports
from .hparams import POLYGONS_PARQUET, FTS_DIR

# polygon
# =======
def load_polygon(
    parquet: Path, poly_id: int,
) -> tuple[np.ndarray, str]:
    """
    Load one polygon from `polygons.parquet`.

    Parameters
    ----------
    parquet : Path
        Path to the polygons parquet.
    poly_id : int
        Polygon ID to load.

    Returns
    -------
    pts : ndarray
        `(Npts, 2)` int64 lattice points.
    role : str
        `"train"` or `"val"`.
    """
    df  = pl.read_parquet(parquet)
    row = df.filter(pl.col("id") == poly_id)
    if row.is_empty():
        raise KeyError(f"polygon id={poly_id} not in {parquet}")

    pts  = np.asarray(row["pts"][0].to_list(), dtype=np.int64)
    role = str(row["role"][0])
    return pts, role

def load_polygons(
    parquet: Path = POLYGONS_PARQUET,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Load `polygons.parquet`, split by role.

    Parameters
    ----------
    parquet : Path, optional
        Path to the polygons parquet. Default `POLYGONS_PARQUET`.

    Returns
    -------
    train, val : pl.DataFrame, pl.DataFrame
        Rows with `role == "train"` / `role == "val"`.
    """
    df    = pl.read_parquet(parquet)
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
def _to_safe(ckpt: dict) -> dict:
    """Coerce a ckpt payload to types `torch.load(weights_only=True)`
    accepts: `Path` hparams -> str, numpy's uint32 key array -> list."""
    out = dict(ckpt)
    out["hparams"] = {k: str(v) if isinstance(v, Path) else v
                      for k, v in ckpt["hparams"].items()}
    state = ckpt.get("numpy_rng_state")
    if state is not None and isinstance(state[1], np.ndarray):
        out["numpy_rng_state"] = (state[0], state[1].tolist()) + state[2:]
    return out

def save_ckpt(path: Path, *, net, step: int, optim, hparams) -> None:
    """
    Save a training checkpoint to `path`, loadable with
    `torch.load(weights_only=True)`. Persists model + optimizer state
    and torch / Python / numpy *global* RNG state. On resume that makes the
    AR-rollout draws (`torch.multinomial`) replay bit-for-bit, but the
    per-purpose `np.random.Generator`s on `Trainer` (polygon pick / batch /
    eval / explore) are re-seeded from `cfg.seed` at `__init__` and do NOT
    replay their post-ckpt stream.
    """
    torch.save(_to_safe({
        "state_dict":       net.state_dict(),
        "step":             step,
        "optim_state":      optim.state_dict(),
        "rng_state":        torch.get_rng_state(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state":  np.random.get_state(),
        "hparams":          hparams,
    }), path)

def load_ckpt(path: Path, device: str) -> dict:
    """
    Load a training checkpoint produced by `save_ckpt`. Prefers safe
    weights-only loading; ckpts predating the weights_only-safe format
    fall back to full unpickling (fine for your own runs -- never load
    untrusted files this way; see `resave_ckpt_safe`).
    """
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except pickle.UnpicklingError:
        print(f"[load_ckpt] {path} predates the weights_only-safe format; "
              f"falling back to weights_only=False (consider "
              f"resave_ckpt_safe)", flush=True)
        ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("numpy_rng_state")
    if state is not None and isinstance(state[1], list):
        ckpt["numpy_rng_state"] = (
            (state[0], np.asarray(state[1], dtype=np.uint32)) + state[2:]
        )
    return ckpt

def resave_ckpt_safe(path: Path) -> None:
    """
    Rewrite a pre-weights_only-format ckpt in the safe format, in place.
    Fully unpickles the file -- only run it on ckpts you trust.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    torch.save(_to_safe(ckpt), path)
