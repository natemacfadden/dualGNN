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
# Description:  Per-polygon training state shared by SFT and REINFORCE.
# -----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

# local imports
from ..dualgraph         import DualGraph
from .target_conditional import SimpConditional


@dataclass
class PolyState:
    """
    Per-polygon training state.

    Bundles the polygon's `DualGraph`, the canonical-key set of its FRT
    pool, and per-polygon graph tensors pre-uploaded to the training
    device. SFT-only fields (FRT pool split into train/val, per-split
    `SimpConditional`s, etc.) are optional -- they are populated for
    supervised training in `load_poly_state` and left as `None` by
    REINFORCE's `build_rl_poly_state`.

    Attributes
    ----------
    poly_id : int
        Polygon ID in the run-local `polygons.parquet`.
    cmplx : DualGraph
        Dual-graph candidate complex for this polygon.
    pool_keys : set[bytes]
        Canonical-form keys of every FRT in the full (train + val) pool.
        Used for novel-FRT deduplication during SFT exploration and for
        the KL-vs-uniform validation metric in REINFORCE.
    circ_features_t, edge_indices_t, simp_compat_t : torch.Tensor
        Per-polygon graph tensors pre-uploaded to the training device.

    role : str or None, optional
        `"train"` or `"val"` for SFT; `None` for REINFORCE.
    pts : ndarray or None, optional
        `(Npts, 2)` int64 lattice points. SFT only.
    N_simps_per_ft : int or None, optional
        Number of simps in any fine triangulation of this polygon. SFT only.
    train_triangs, val_triangs : ndarray or None, optional
        `(Ntrain|Nval, N_simps_per_ft, 3)` int64. SFT only.
    simp_cond_train, simp_cond_val : SimpConditional or None, optional
        Empirical next-simp conditionals over the train / val pools.
        SFT only.
    """
    # core (always populated)
    poly_id:         int
    cmplx:           DualGraph
    pool_keys:       set[bytes]
    circ_features_t: torch.Tensor
    edge_indices_t:  torch.Tensor
    simp_compat_t:   torch.Tensor

    # SFT-only (None for REINFORCE)
    role:            str | None             = None
    pts:             np.ndarray | None      = None
    N_simps_per_ft:  int | None             = None
    train_triangs:   np.ndarray | None      = None
    val_triangs:     np.ndarray | None      = None
    simp_cond_train: SimpConditional | None = None
    simp_cond_val:   SimpConditional | None = None
