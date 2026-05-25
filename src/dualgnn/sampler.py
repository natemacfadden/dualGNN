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
# Description:  Methods to call dualgnn to actually sample FRTs.
# -----------------------------------------------------------------------------

# external imports
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

# local imports
from .dualgraph import DualGraph
from .geometry  import canonical_simps
from .model     import DualGNN


# main method
# ===========
def sample(
    net:       DualGNN,
    dualgraph: DualGraph,
    Ntriangs:  int,
    *,
    device:     str | torch.device | None = None,
    batch_size: int = 256,
    beta:       float = 1.0,
    seed:       int | None = None,
    verbose:    bool = True,
    compile:    bool = False,
) -> np.ndarray:
    """
    Generate `Ntriangs` triangulations from the dualGNN sampler.

    Parameters
    ----------
    net : DualGNN
        Trained model.
    dualgraph : DualGraph
        Graph to apply the model to (set by the polygon).
    Ntriangs : int
        Total number of triangulations to sample.

    device : str | torch.device, optional
        Where to run the model and tensors. Default `None` infers from the
        model's parameters (whatever device `net` lives on).
    batch_size : int, optional
        Number of independent rollouts batched into one GPU forward pass. Larger
        values use more memory but better GPU utilization. Default `256`.
    beta : float, optional
        Inverse temperature: `softmax(beta * logits)`. Default `1.0` gives
        uniform-over-FRT sampling. `beta=0` is uniform over legal simps (i.e.,
        grow2d); `beta<0` samples model-unlikely triangulations.
    seed : int, optional
        Seed for the random number generator.
    verbose : bool, optional
        Print a warning when `beta != 1.0`. Default True.
    compile : bool, optional
        If True, wrap `net` with `torch.compile(net, dynamic=True)`. ~1.9x
        speedup on Npts=64 polygons but pays ~10s compile on first call. For
        repeated calls, compile once externally instead. Default False.

    Returns
    -------
    out : ndarray
        `(Ntriangs, N_simps_per_ft, 3)` int8. Each FRT in canonical form:
        each simp's three point indices sorted ascending, simps lex-sorted
        within the FRT.
    """
    # check/set inputs
    # ----------------
    if device is None:
        device = next(net.parameters()).device

    if verbose and beta != 1.0:
        print(f"beta = {beta}... set to 1 for uniform sampling...")
        print( "disable warning with verbose=False")

    if seed is None:
        gen = None
    else:
        gen = torch.Generator(device=device).manual_seed(seed)

    # read from DualGraph
    # -------------------
    N_simps_per_ft = dualgraph.N_simps_per_ft
    Nsimps         = dualgraph.simps.shape[0]
    circ_features  = torch.from_numpy(dualgraph.circ_features).float().to(device)
    edge_indices   = torch.from_numpy(dualgraph.edges).to(device)
    compat         = torch.from_numpy(dualgraph.simp_compat).to(device)

    # sample
    # ------
    dtype     = np.int8 if dualgraph.pts.shape[0] <= 128 else np.int16
    canon     = np.empty((Ntriangs, N_simps_per_ft, 3), dtype=dtype)
    simp_keys = dualgraph.simps.astype(dtype)
    N_out     = 0

    net.eval() # turn training mode off
    # `dynamic=True` handles the trailing partial batch when batch_size does
    # not divide Ntriangs; idempotent if `net` is already a compiled module.
    if compile and not hasattr(net, "_orig_mod"):
        net = torch.compile(net, dynamic=True)
    while N_out < Ntriangs:
        n = min(batch_size, Ntriangs - N_out)   # triangs in this batch
        with torch.no_grad():
            placed, _ = ar_rollout_batch(
                net,
                batch          = n,
                N_simps_per_ft = N_simps_per_ft,
                circ_features  = circ_features,
                edge_indices   = edge_indices,
                compat         = compat,
                device         = device,
                beta           = beta,
                generator      = gen,
            )

        # convert `placed` simplices to canonical-form FRTs
        placed_np = placed.cpu().numpy()
        simp_idxs = np.where(placed_np)[1].reshape(n, N_simps_per_ft)
        for j in range(n):
            canon[N_out + j] = canonical_simps(simp_keys[simp_idxs[j]])
        N_out += n

    return canon

# legal next-simp mask
# ====================
def compute_legal(
    placed: torch.Tensor,
    compat: torch.Tensor,
) -> torch.Tensor:
    """
    'Legal' next-simp mask for a batch of partial triangulations. Batched and
    device-agnostic (CPU / MPS / CUDA).

    Simp `i` is legal iff no placed simp `j` is incompatible with it. Because
    `compat` has `False` on the diagonal, a placed simp is incompatible with
    itself and gets filtered out automatically.

    Parameters
    ----------
    placed : Tensor
        `(batch_size, Nsimps)` bool. `placed[b, i]` True iff simp `i` is in the
        partial triangulation of batch element `b`.
    compat : Tensor
        `(Nsimps, Nsimps)` bool, symmetric, diagonal False. `compat[i, j]` True
        iff simps `i` and `j` have disjoint interiors.

    Returns
    -------
    legal : Tensor
        `(batch_size, Nsimps)` bool. `legal[b, i]` True iff `i` is unplaced AND
        pairwise-compatible with every placed simp in row `b`.
    """
    B, Nsimps = placed.shape

    # fast path: faster for small Nsimps. Crossover ~3k; above that the
    # chunked path below wins by skipping the (~compat).float() materialization
    # (and is required to stay under ~2 GB once Nsimps > 22360).
    if Nsimps <= 3000:
        placed_f    = placed.float()         # (batch_size, Nsimps)
        incompat    = (~compat).float()      # (Nsimps, Nsimps)
        N_conflicts = placed_f @ incompat    # (batch_size, Nsimps): #conflicts
        return N_conflicts < 0.5 # (0.5 for floating point tolerances)

    # use placed @ (~compat) = |placed| - placed @ compat to skip the
    # (~compat).float() materialization; stream column chunks for big Nsimps
    placed_f  = placed.float()
    n_placed  = placed_f.sum(dim=1, keepdim=True)
    chunk_len = max(1, 2_000_000_000 // (4 * Nsimps))
    legal     = torch.empty(B, Nsimps, dtype=torch.bool, device=placed.device)
    for c0 in range(0, Nsimps, chunk_len):
        c1 = min(c0 + chunk_len, Nsimps)
        N_compat = placed_f @ compat[:, c0:c1].float()
        legal[:, c0:c1] = (n_placed - N_compat) < 0.5
    return legal

# AR rollout
# ==========
def ar_rollout_batch(
    net: DualGNN,
    *,
    batch:          int,
    N_simps_per_ft: int,
    circ_features:  torch.Tensor,
    edge_indices:   torch.Tensor,
    compat:         torch.Tensor,
    device:         str | torch.device,
    beta:           float = 1.0,
    generator:      torch.Generator | None = None,
    track_log_probs: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    One batch of grad-aware AR rollout.

    Used by both `sample` (with `torch.no_grad()` around the call) and
    REINFORCE's `_train_step` (with `track_log_probs=True` to keep the per-step
    selected-action log-probs for the policy-gradient update).

    Parameters
    ----------
    net : DualGNN
        Model. Caller controls eval/train mode and any `no_grad` context.

    batch : int
        Number of trajectories to roll out in parallel.
    N_simps_per_ft : int
        Number of placement steps per trajectory.
    circ_features : Tensor
        `(Nedges, Dedge)` float, on `device`.
    edge_indices : Tensor
        `(2, Nedges)` long, on `device`.
    compat : Tensor
        `(Nsimps, Nsimps)` bool, on `device`. Symmetric with diagonal False.
    device : str or torch.device
        Target device for the rollout tensors.
    beta : float, optional
        Inverse temperature: `softmax(beta * logits)`. Default 1.0 (uniform
        sampling over FRTs). `beta=0` -> uniform over legal simps (grow2d).
    generator : torch.Generator, optional
        Seeded RNG for reproducible draws. Default `None` (torch global RNG).
    track_log_probs : bool, optional
        If True, also return the per-trajectory sum of the selected-action
        log-probs (with grad), for policy-gradient updates. Default False.

    Returns
    -------
    placed : Tensor
        `(batch, Nsimps)` bool. Final placement mask of the trajectories.
    log_probs_sum : Tensor or None
        `(batch,)` float with grad if `track_log_probs`, else `None`. Sum of
        selected-action log-probs per trajectory.
    """
    Nsimps = compat.shape[0]
    placed = torch.zeros(batch, Nsimps, dtype=torch.bool, device=device)
    log_probs_sum = (
        torch.zeros(batch, device=device) if track_log_probs else None
    )

    for step in range(N_simps_per_ft):
        legal  = compute_legal(placed, compat)
        logits = net(circ_features, edge_indices, placed, legal) * beta
        logits = logits.masked_fill(~legal, float("-inf"))
        log_p  = F.log_softmax(logits, dim=-1)
        picks  = torch.multinomial(log_p.exp(), 1,
                                   generator=generator).squeeze(-1)
        if track_log_probs:
            log_probs_sum += log_p.gather(1, picks.unsqueeze(-1)).squeeze(-1)
        placed.scatter_(1, picks.unsqueeze(-1), True)

    return placed, log_probs_sum
