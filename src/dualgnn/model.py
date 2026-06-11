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
# Description:  dualGNN model: per-node logit for next-simp prediction.
# -----------------------------------------------------------------------------

# external imports
from __future__ import annotations

from importlib import resources
from pathlib import Path
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

# local imports
from .device import default_device


# main model class
# ================
class _Layer(nn.Module):
    """
    One round of message passing on dualGNN's graph.

    Each forward pass:
        1) build a per-edge message `(circ, LayerNorm(f))` from each sender
        2) scatter messages to receivers under reductions `agg = (sum,min,max)`
        3) MLP on `(LayerNorm(f), agg, metadata)` for
           `metadata = (placed, legal)`
        4) residual add: `f_new = f + delta`

    Note: `agg` summarizes messages from a node's neighbors, so it does NOT
    include its own feature vector. Step 4 therefore passes its own `f`
    separately, alongside `agg` and `metadata`, to preserve self-state across
    layers.

    The per-node `metadata` (placed, legal) is re-injected at every layer
    rather than only at init, so every layer can directly see the current
    placement/legality state.

    Implementation notes (the math matches the description above exactly;
    the layout below avoids the memory traffic that used to dominate):
        - messages reduce elementwise, so the circ columns of `agg` are
          state-independent and computed once per forward by the caller
          (`DualGNN._aggregate_circ`), not per layer and batch element
        - on CUDA inference the `f` columns reduce via `segment_reduce`
          over dst-sorted edges (contiguous segments, no atomics); under
          grad they keep the original `scatter_reduce`
        - on CPU inference the whole forward runs in `(Nsimps, batch, D)`
          layout (see `transposed` below): a dim-0 scatter of an
          `(E, batch * D)` view hits torch's fast contiguous-2D CPU path
          (vectorized row streaming, ~19x per-core over the generic
          strided dim-1 path; measured single-thread 631 -> 34 ms on the
          [0,6]^2 trio, with threading a minor factor on the fast form)
        - `mlp[0]` is applied blockwise (one weight slice per input block),
          so the wide `(f, agg, metadata)` concat is never materialized

    Parameters
    ----------
    D : int
        Hidden state dimension.
    """

    def __init__(self, D):
        super().__init__()

        # dimensions
        self.D         = D
        self.Dmetadata = 2 # (placed, legal)
        self.Dedge     = 4 # for 2D polytopes

        msg_dim = self.Dedge + D # circuit + feature vector
        agg_dim = 3 * msg_dim    # (sum, min, max)

        # layer pieces
        self.norm = nn.LayerNorm(D)
        self.mlp  = nn.Sequential(
            nn.Linear(D + agg_dim + self.Dmetadata, D), # +D to also pass self f
            nn.GELU(),
            nn.Linear(D, D),
        )

    def forward(self, f, circ_agg, routing, metadata, *, transposed=False):
        """
        Apply one message-passing layer.

        Parameters
        ----------
        f : Tensor
            `(batch_size, Nsimps, D)` float per-node hidden state --
            `(Nsimps, batch_size, D)` when `transposed`.
        circ_agg : Tensor
            `(1, Nsimps, 3 * Dedge)` float per-node `(sum, min, max)` of
            incoming-edge circuit features, shared across the batch
            (`DualGNN._aggregate_circ`) -- `(Nsimps, 1, 3 * Dedge)` when
            `transposed`.
        routing : tuple
            `(src_sorted, dst_sorted, seg_lengths, isolated)` from
            `DualGNN._routing`: sender / receiver simp index per dst-sorted
            edge `(Nedges,)`, incoming-edge count per node `(Nsimps,)`, and
            a `(Nsimps, 1)` bool mask of nodes with no incoming edges
            (`None` if there are none).
        metadata : Tensor
            `(batch_size, Nsimps, Dmetadata)` float per-node state features
            (placed, legal) -- `(Nsimps, batch_size, Dmetadata)` when
            `transposed`.
        transposed : bool, optional
            Node-major layout (set by `DualGNN.forward` for CPU inference;
            see the class docstring). Default False.

        Returns
        -------
        f_new : Tensor
            Updated hidden state, same shape/layout as `f`.
        """
        src_sorted, dst_sorted, seg_lengths, isolated = routing
        Nedges = src_sorted.shape[0]

        # aggregate each node's incoming messages: per-edge sender features,
        # reduced per node. Nodes with no incoming edges get 0
        f_norm = self.norm(f)
        if transposed:
            # CPU inference, node-major: dim-0 gather + dim-0 scatters of an
            # (Nedges, batch * D) view, which hits the fast contiguous-2D
            # scatter path (the batch-major dim-1 scatter takes the generic
            # strided path, ~19x slower per core)
            Nsimps, batch_size, D = f.shape
            f_sender = f_norm[src_sorted].view(Nedges, batch_size * D)
            idx   = dst_sorted.view(Nedges, 1).expand(Nedges, batch_size * D)
            f_sum = torch.zeros(Nsimps, batch_size * D,
                                device=f.device, dtype=f_sender.dtype)
            f_min = torch.zeros_like(f_sum)
            f_max = torch.zeros_like(f_sum)
            f_sum.scatter_reduce_(0, idx, f_sender, reduce="sum",
                                  include_self=False)
            f_min.scatter_reduce_(0, idx, f_sender, reduce="amin",
                                  include_self=False)
            f_max.scatter_reduce_(0, idx, f_sender, reduce="amax",
                                  include_self=False)
            f_sum = f_sum.view(Nsimps, batch_size, D)
            f_min = f_min.view(Nsimps, batch_size, D)
            f_max = f_max.view(Nsimps, batch_size, D)
        elif torch.is_grad_enabled() or f.device.type != "cuda":
            # scatter_reduce under grad: its backward distributes min/max
            # gradients across ties exactly as before (ties are common --
            # symmetric nodes share hidden states -- and segment_reduce's
            # backward picks a different subgradient at them)
            batch_size, Nsimps, D = f.shape
            f_sender = f_norm[:, src_sorted, :]
            idx   = dst_sorted.view(1, Nedges, 1).expand(batch_size,
                                                         Nedges, D)
            f_sum = torch.zeros(batch_size, Nsimps, D,
                                device=f.device, dtype=f_sender.dtype)
            f_min = torch.zeros_like(f_sum)
            f_max = torch.zeros_like(f_sum)
            f_sum.scatter_reduce_(1, idx, f_sender, reduce="sum",
                                  include_self=False)
            f_min.scatter_reduce_(1, idx, f_sender, reduce="amin",
                                  include_self=False)
            f_max.scatter_reduce_(1, idx, f_sender, reduce="amax",
                                  include_self=False)
        else:
            # CUDA inference: contiguous segment reductions over the
            # dst-sorted edges, with no atomics. ~1.6x faster than scatter
            batch_size, Nsimps, D = f.shape
            f_sender = f_norm[:, src_sorted, :]
            lengths = seg_lengths.view(1, Nsimps).expand(batch_size, Nsimps)
            f_sum = torch.segment_reduce(f_sender, "sum", lengths=lengths,
                                         axis=1, unsafe=True, initial=0.0)
            f_min = torch.segment_reduce(f_sender, "min", lengths=lengths,
                                         axis=1, unsafe=True,
                                         initial=float("inf"))
            f_max = torch.segment_reduce(f_sender, "max", lengths=lengths,
                                         axis=1, unsafe=True,
                                         initial=float("-inf"))
            if isolated is not None:
                f_min = f_min.masked_fill(isolated, 0.0)
                f_max = f_max.masked_fill(isolated, 0.0)

        # apply mlp[0] blockwise. Its weight columns follow the layout
        #   [f | circ_sum, f_sum | circ_min, f_min | circ_max, f_max | meta]
        # so slicing them lets the batch-independent circ blocks fold into a
        # single bias-like term, and spares materializing the wide concat
        De = self.Dedge
        Wf, Wcs, Ws, Wcm, Wm, Wcx, Wx, Wmeta = torch.split(
            self.mlp[0].weight,
            [D, De, D, De, D, De, D, self.Dmetadata], dim=1,
        )
        cs, cm, cx = circ_agg.split(De, dim=-1)
        const = (F.linear(cs, Wcs) + F.linear(cm, Wcm) + F.linear(cx, Wcx)
                 + self.mlp[0].bias)
        h = (F.linear(torch.cat([f_norm, f_sum, f_min, f_max], dim=-1),
                      torch.cat([Wf, Ws, Wm, Wx], dim=1))
             + F.linear(metadata, Wmeta) + const)
        return f + self.mlp[2](self.mlp[1](h))

class DualGNN(nn.Module):
    """
    The dualGNN model: AR sampler of simplices for uniform sampling of
    fine regular triangulations. Analogous to a Pointer Network.

    Stack of `K` `_Layer`s with a shared (per-polygon) dual-graph structure.

    Output is one logit per candidate simp; downstream code applies log-softmax
    over the candidate axis to get a distribution over which simp to place next.

    Parameters
    ----------
    D : int, optional
        Hidden state dimension. Default 32.
    K : int, optional
        Number of `_Layer` rounds. Default 16.
    """

    def __init__(self, *, D=32, K=16):
        super().__init__()
        # hyperparameters
        self.D = D
        self.K = K

        # unchanging config
        self.Dedge     = 4 # (for 2D polytopes)
        self.Dmetadata = 2 # (placed, legal)

        # for setting initial feature vector data... one hot
        self.init_mlp = nn.Sequential(
            nn.Linear(self.Dmetadata, D),
            nn.GELU(),
            nn.Linear(D, D),
        )

        # message passing/simplex selection data
        self.layers = nn.ModuleList(
            [_Layer(D) for _ in range(K)]
        )
        self.norm = nn.LayerNorm(D)
        self.head = nn.Linear(D, 1)

    # default() cache, keyed by str(device)
    _default_nets: dict[str, "DualGNN"] = {}

    @classmethod
    def default(
        cls,
        device: str | torch.device | None = None,
    ) -> "DualGNN":
        """
        Load the shipped default model: the `D=32`, `K=16` dualGNN after
        REINFORCE fine-tuning (`dualgnn/ckpts/reinforce.pt`, packaged as
        data, so this works from any install -- no repo checkout needed).
        Cached per device; repeated calls return the same instance.

        Parameters
        ----------
        device : str or torch.device, optional
            Target device. Default `None` -> best available (CUDA, else
            MPS, else CPU). CPU-only environments are fully supported.

        Returns
        -------
        net : DualGNN
        """
        if device is None:
            device = default_device()
        key = str(device)
        if key not in cls._default_nets:
            ref = resources.files("dualgnn") / "ckpts" / "reinforce.pt"
            with resources.as_file(ref) as path:
                cls._default_nets[key] = cls.from_ckpt(path, device=device)
        return cls._default_nets[key]

    @classmethod
    def from_ckpt(
        cls,
        path:   str | Path,
        device: str | torch.device | None = None,
    ) -> "DualGNN":
        """
        Load a `DualGNN` from a checkpoint produced by `Trainer` or
        `reinforce`. Reads `D` and `K` from the ckpt's hparams, strips the
        `_orig_mod.` prefix left by `torch.compile`, and returns the model
        in eval mode on `device`.

        Parameters
        ----------
        path : str or Path
            Checkpoint file.
        device : str or torch.device, optional
            Target device. Default `None` -> best available (CUDA, else MPS,
            else CPU).

        Returns
        -------
        net : DualGNN
        """
        if device is None:
            device = default_device()
        try:
            ckpt = torch.load(path, map_location=device, weights_only=True)
        except pickle.UnpicklingError as e:
            raise RuntimeError(
                f"{path} is not loadable with weights_only=True (it predates "
                f"dualgnn's weights_only-safe ckpt format). If you trust the "
                f"file, rewrite it in place with:\n"
                f"    from dualgnn.training.io import resave_ckpt_safe\n"
                f"    resave_ckpt_safe({str(path)!r})"
            ) from e
        hp   = ckpt["hparams"]
        net  = cls(D=hp["d_model"], K=hp["k_rounds"]).to(device).eval()
        net.load_state_dict({
            k.replace("_orig_mod.", ""): v
            for k, v in ckpt["state_dict"].items()
        })
        return net

    @staticmethod
    def _routing(edge_indices, *, Nsimps):
        """
        Static per-graph message routing shared by every `_Layer` call:
        sender / receiver indices in dst-sorted order, per-node incoming-edge
        counts (the segment lengths), and a mask of isolated nodes (`None`
        when every node has an incoming edge, the usual case).

        Returns
        -------
        src_sorted, dst_sorted : Tensor
            `(Nedges,)` long.
        seg_lengths : Tensor
            `(Nsimps,)` long.
        isolated : Tensor or None
            `(Nsimps, 1)` bool, or `None`.
        """
        order       = torch.argsort(edge_indices[1])
        src_sorted  = edge_indices[0][order]
        dst_sorted  = edge_indices[1][order]
        seg_lengths = torch.bincount(edge_indices[1], minlength=Nsimps)
        isolated    = (seg_lengths == 0).view(Nsimps, 1)
        return (src_sorted, dst_sorted, seg_lengths,
                isolated if isolated.any() else None)

    def _aggregate_circ(self, circ_features, edge_indices, *, Nsimps):
        """
        Per-node `(sum, min, max)` of incoming-edge circuit features: the
        state-independent columns of every `_Layer`'s aggregation, computed
        once per forward instead of per layer and batch element.

        Returns
        -------
        circ_agg : Tensor
            `(1, Nsimps, 3 * Dedge)` float.
        """
        Nedges = edge_indices.shape[1]
        idx    = edge_indices[1].view(Nedges, 1).expand(Nedges, self.Dedge)
        aggs   = []
        for reduce in ("sum", "amin", "amax"):
            agg = circ_features.new_zeros(Nsimps, self.Dedge)
            agg.scatter_reduce_(0, idx, circ_features, reduce=reduce,
                                include_self=False)
            aggs.append(agg)
        return torch.cat(aggs, dim=-1).unsqueeze(0)

    def forward(self, circ_features, edge_indices, placed, legal):
        """
        Score the next simp to place for each batch element. Does so via
            1) initializing hidden states via `init_mlp(metadata)`,
            2) running `K` `_Layer` message passing rounds,
            3) projecting to a single logit per simp, and then
            4) masking placed / illegal entries to `-inf`.

        Parameters
        ----------
        circ_features : Tensor
            `(Nedges, Dedge)` float circuit features for each edge.
        edge_indices : Tensor
            `(2, Nedges)` long. Row 0 = sender simp, row 1 = receiver.
        placed : Tensor
            `(batch_size, Nsimps)` bool or float. `1` where simp `i` is
            in the current partial triangulation of batch element `b`.
        legal : Tensor
            `(batch_size, Nsimps)` bool or float. `1` where simp `i` is
            unplaced AND pairwise-compatible with every placed simp in
            row `b` (see `sampler.compute_legal`).

        Returns
        -------
        logits : Tensor
            `(batch_size, Nsimps)` float. `-inf` for placed or non-legal
            entries. Apply `log_softmax(dim=-1)` for the predicted distribution
            over which simp to place next.
        """
        metadata = torch.cat([
            placed.float().unsqueeze(-1),
            legal.float().unsqueeze(-1),
        ], dim=-1)
        circ_agg = self._aggregate_circ(circ_features, edge_indices,
                                        Nsimps=placed.shape[-1])
        routing  = self._routing(edge_indices, Nsimps=placed.shape[-1])

        # CPU inference runs node-major (Nsimps, batch, D): every per-node
        # op acts on the last dim either way, and the layers' scatters
        # parallelize in this layout (see the _Layer docstring)
        transposed = (placed.device.type != "cuda"
                      and not torch.is_grad_enabled())
        if transposed:
            metadata = metadata.transpose(0, 1).contiguous()
            circ_agg = circ_agg.transpose(0, 1)

        f = self.init_mlp(metadata)

        # K message-passing rounds
        for layer in self.layers:
            f = layer(f, circ_agg, routing, metadata, transposed=transposed)

        # project to logits
        f      = self.norm(f)
        logits = self.head(f).squeeze(-1)
        if transposed:
            logits = logits.transpose(0, 1)

        # mask and return
        mask = placed.bool() | (~legal.bool())
        return logits.masked_fill(mask, float("-inf"))
