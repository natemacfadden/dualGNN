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
# Description:  Empirical next-simp conditional P(sigma | T_partial) built
#               from a harvested FRT pool.
# -----------------------------------------------------------------------------

# external imports
from __future__ import annotations

import numpy as np
import torch


# empirical simplex probabilities
# ===============================
class SimpConditional:
    """
    Answers next-simp conditional probability queries via a bitmap pool.

    Parameters
    ----------
    pool_simps : ndarray
        `(Ntriangs, N_simps_per_ft, 3)` int. Each row is one triangulation's
        simplices (as point indices). `Ntriangs` is the number of triangulations
        in the pool.
    dualgraph_simps : ndarray
        `(Nsimps, 3)` int. The DualGraph's full candidate simp set (as point
        indices).

    Attributes
    ----------
    bm : ndarray
        `(Nft, Nsimps)` bool bitmap pool.
    """
    def __init__(
        self,
        pool_simps:      np.ndarray,
        dualgraph_simps: np.ndarray,
    ):
        # (simps_to_bitmaps is separate since we use it elsewhere)
        self.bm = simps_to_bitmaps(pool_simps, dualgraph_simps)
        self._bm_gpu: torch.Tensor | None = None

    def conditional_batch(
        self,
        t_partials: torch.Tensor,
    ) -> torch.Tensor:
        """
        Batched next-simp conditional probabilities (computed on GPU).

        Parameters
        ----------
        t_partials : Tensor
            `(batch_size, Nsimps)` bool on the target device.

        Returns
        -------
        target : Tensor
            `(batch_size, Nsimps)` float, rows of probabilities (sums to 1) for
            valid entries, else zero (if no FRT in pool extends triang).
        """
        device       = t_partials.device
        pool_f       = self._gpu_pool(device)
        placed_f     = t_partials.float()
        placed_count = placed_f.sum(dim=1, keepdim=True)

        keep   = (pool_f @ placed_f.T) == placed_count.T
        counts = keep.float().T @ pool_f
        counts = counts.masked_fill(t_partials, 0)
        tot    = counts.sum(dim=1, keepdim=True)
        return counts / tot.clamp(min=1)

    def _gpu_pool(self, device: str | torch.device) -> torch.Tensor:
        """Move the bitmap to the device on first call, then cache"""
        if self._bm_gpu is None:
            self._bm_gpu = torch.from_numpy(self.bm).to(
                device=device, dtype=torch.float32,
            )
        return self._bm_gpu

# bitmap encoding
# ===============
def simps_to_bitmaps(
    pool_simps:      np.ndarray,
    dualgraph_simps: np.ndarray,
) -> np.ndarray:
    """
    Encode a pool of triangulations as bitmaps over the simps of a DualGraph.

    Parameters
    ----------
    pool_simps : ndarray
        `(Ntriangs, N_simps_per_ft, 3)` int. Each row is one triangulation's
        simplices (as point indices). `Ntriangs` is the number of triangulations
        in the pool.
    dualgraph_simps : ndarray
        `(Nsimps, 3)` int. The DualGraph's full candidate simp set.

    Returns
    -------
    bitmap : ndarray
        `(Nft, Nsimps)` bool. `bm[T, sigma]` True iff simp `sigma` is in
        triangulation `T`.
    """
    # all candidate simplices
    Nsimps     = dualgraph_simps.shape[0]
    key_to_idx = {k.tobytes(): i
                  for i, k in enumerate(np.sort(dualgraph_simps, axis=1))}

    # the pool simplices
    pool_keys = np.sort(pool_simps, axis=2).astype(dualgraph_simps.dtype)
    Nft, S, _ = pool_keys.shape

    # build the bitmap
    bitmap = np.zeros((Nft, Nsimps), dtype=bool)
    for ft_idx in range(Nft):
        for simp_pos in range(S):
            simp_idx = key_to_idx[pool_keys[ft_idx, simp_pos].tobytes()]
            bitmap[ft_idx, simp_idx] = True
    return bitmap
