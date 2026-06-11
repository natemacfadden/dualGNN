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
# Description:  Interactive AR-rollout viewer.
#                 left  panel: polygon, placed simps as filled triangles
#                 right panel: dual graph, nodes colored by model probability
#               Initial polygon is a random draw in [0, 8]^2; bottom strip has
#               |x|<= / |y|<= bounds + [Random]. Click in the polygon panel
#               to add/remove a hull vertex.
#
# Usage:    python scripts/visualize.py
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
import re
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.widgets import Button, TextBox
from scipy.spatial import ConvexHull

# local imports
from dualgnn.device    import default_device
from dualgnn.dualgraph import DualGraph
from dualgnn.geometry  import enum_lattice_pts
from dualgnn.model     import DualGNN
from dualgnn.sampler   import compute_legal

# matplotlib steals "s" for save-figure; turn that off so it can be our key.
plt.rcParams["keymap.save"] = []


# layout
# ======
_LAYOUT_KINDS = ("random", "centroid", "spectral", "spring")

def graph_layout(dualgraph: DualGraph,
                 kind: str = "random",
                 rng: np.random.Generator | None = None) -> np.ndarray:
    """2D dual-graph layout normalized to ~[-1, 1]^2. `kind` is one of:
      - "random":   uniform in [-1, 1]^2; instant, no structure
      - "centroid": triangle centroid in polygon coords + collision spread; O(N)
      - "spectral": bottom non-trivial eigvecs of normalized graph Laplacian,
                    percentile-clipped to suppress spires; O(N^3) dense eigh
      - "spring":   vectorized Fruchterman-Reingold, warm-started from
                    centroid; O(iters * N^2) in pure numpy
    `rng` only matters for "random"."""
    if kind == "random":
        rng = rng or np.random.default_rng()
        return rng.uniform(-1, 1, size=(dualgraph.simps.shape[0], 2))
    if kind == "centroid":
        return _centroid_layout(dualgraph)
    if kind == "spectral":
        return _spectral_layout(dualgraph)
    if kind == "spring":
        return _spring_layout(dualgraph)
    raise ValueError(f"unknown layout kind: {kind!r}")


def _centroid_layout(dualgraph: DualGraph) -> np.ndarray:
    pts   = dualgraph.pts
    simps = dualgraph.simps
    cents = pts[simps].mean(axis=1).astype(np.float64)
    # Two simps share a centroid iff their integer vertex sums match.
    sums = pts[simps].sum(axis=1)
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i in range(len(simps)):
        groups[(int(sums[i, 0]), int(sums[i, 1]))].append(i)
    coords = cents.copy()
    radius = 0.3
    for members in groups.values():
        n = len(members)
        if n < 2:
            continue
        for j, idx in enumerate(members):
            theta = 2.0 * np.pi * j / n
            coords[idx, 0] += radius * np.cos(theta)
            coords[idx, 1] += radius * np.sin(theta)
    return _normalize_layout(coords)


def _spectral_layout(dualgraph: DualGraph) -> np.ndarray:
    N = dualgraph.simps.shape[0]
    if N < 3:
        return np.zeros((N, 2))
    src, dst = dualgraph.edges[0], dualgraph.edges[1]
    A = np.zeros((N, N))
    A[src, dst] = 1.0
    A[dst, src] = 1.0
    deg = A.sum(axis=1)
    deg_safe = np.where(deg > 0, deg, 1.0)
    d_inv_sqrt = 1.0 / np.sqrt(deg_safe)
    L = np.eye(N) - (d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :])
    _, vecs = np.linalg.eigh(L)
    coords = vecs[:, 1:3].copy()
    # outlier-clip each axis so a single spire doesn't dominate the window
    for d in range(2):
        lo, hi = np.percentile(coords[:, d], [3.0, 97.0])
        coords[:, d] = np.clip(coords[:, d], lo, hi)
    return _normalize_layout(coords)


def _spring_layout(dualgraph: DualGraph, iters: int = 80) -> np.ndarray:
    N = dualgraph.simps.shape[0]
    if N < 2:
        return np.zeros((N, 2))
    pos = _centroid_layout(dualgraph).copy()
    src, dst = dualgraph.edges[0], dualgraph.edges[1]
    keep = src < dst
    src, dst = src[keep], dst[keep]
    k = 2.0 / np.sqrt(N)
    t = 0.2
    cooling = t / iters
    eye = np.eye(N, dtype=bool)
    for _ in range(iters):
        delta = pos[:, None, :] - pos[None, :, :]      # (N, N, 2)
        dist2 = (delta * delta).sum(axis=2)
        dist2[eye] = 1.0                               # avoid 0/0 on diag
        dist2 = np.maximum(dist2, 1e-3)
        rep = (k * k / dist2)[:, :, None] * delta
        rep[eye] = 0.0
        force = rep.sum(axis=1)
        d_edge = pos[src] - pos[dst]
        d = np.sqrt((d_edge * d_edge).sum(axis=1, keepdims=True))
        d_safe = np.maximum(d, 1e-6)
        att = (d * d / k) * (d_edge / d_safe)
        np.add.at(force, src, -att)
        np.add.at(force, dst, att)
        f = np.sqrt((force * force).sum(axis=1, keepdims=True))
        f_safe = np.maximum(f, 1e-9)
        pos = pos + (force / f_safe) * np.minimum(f, t)
        t -= cooling
    return _normalize_layout(pos)


def _normalize_layout(coords: np.ndarray) -> np.ndarray:
    out  = np.asarray(coords, dtype=np.float64).copy()
    lo, hi = out.min(axis=0), out.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    return (out - (hi + lo) / 2.0) * (1.8 / span.max())

# polygon helpers
# ===============
def random_polygon(xmax: int, ymax: int,
                   rng: np.random.Generator,
                   max_tries: int = 50) -> np.ndarray | None:
    """Random convex lattice polygon in [0, xmax] x [0, ymax]: pick 3-7
    random integer vertices, convex-hull-fill. `None` if no non-degenerate
    polygon was produced in `max_tries` attempts."""
    for _ in range(max_tries):
        n     = int(rng.integers(3, 8))
        verts = rng.integers(0, [xmax + 1, ymax + 1], size=(n, 2))
        pts   = enum_lattice_pts(verts)
        if pts is not None:
            return pts
    return None

# main visualizer
# ===============
class Visualizer:
    """
    Interactive 2-panel AR-rollout viewer with a random-polygon button.

    Left panel : polygon, placed simps as filled triangles.
    Right panel: dual graph, random layout, nodes colored by the model's
                 next-simp probability over legal candidates.
    Bottom     : `|x| <=`, `|y| <=` text boxes and a `[Random]` button.

    Parameters
    ----------
    net : DualGNN
        Trained model.
    device : str
        Where the model and tensors live (`"cuda"` / `"mps"` / `"cpu"`).
    initial_pts : ndarray
        `(Npts, 2)` int. Polygon to load first.
    """
    def __init__(self, net: DualGNN, device: str,
                 initial_pts: np.ndarray,
                 layout_kind: str = "centroid"):
        self.net         = net
        self.device      = device
        self.rng         = np.random.default_rng()
        self.layout_kind = layout_kind

        # figure + widgets
        self.fig, (self.ax_poly, self.ax_dual) = plt.subplots(
            1, 2, figsize=(13, 7.0),
        )
        self.fig.subplots_adjust(bottom=0.18, top=0.97)

        # bottom strip widgets
        ax_xmax   = self.fig.add_axes([0.13, 0.04, 0.05, 0.05])
        ax_ymax   = self.fig.add_axes([0.27, 0.04, 0.05, 0.05])
        ax_btn    = self.fig.add_axes([0.13, 0.105, 0.19, 0.04])
        ax_layout = self.fig.add_axes([0.40, 0.105, 0.22, 0.04])
        ax_beta   = self.fig.add_axes([0.40, 0.04, 0.05, 0.05])
        ax_turbo  = self.fig.add_axes([0.52, 0.04, 0.10, 0.05])

        # default bounds = 6, but grow to fit the initial polygon
        init_xmax = max(8, int(initial_pts[:, 0].max()))
        init_ymax = max(8, int(initial_pts[:, 1].max()))
        self.tb_xmax = TextBox(ax_xmax, "0 <= x_i <= ", initial=str(init_xmax))
        self.tb_ymax = TextBox(ax_ymax, "0 <= y_i <= ", initial=str(init_ymax))
        self.tb_beta = TextBox(ax_beta, "beta=1/T=", initial="1.0")
        self.tb_beta.on_submit(self._on_beta_change)
        # numeric-only filters: any non-matching keystroke reverts to the
        # last valid value
        self._suppress_validation = False
        self._last_valid_xmax = str(init_xmax)
        self._last_valid_ymax = str(init_ymax)
        self._last_valid_beta = "1.0"
        self.tb_xmax.on_text_change(
            self._validator(self.tb_xmax, "_last_valid_xmax", self._INT_RE))
        self.tb_ymax.on_text_change(
            self._validator(self.tb_ymax, "_last_valid_ymax", self._INT_RE))
        self.tb_beta.on_text_change(
            self._validator(self.tb_beta, "_last_valid_beta", self._FLOAT_RE))
        self.btn_rand = Button(ax_btn, "random")
        self.btn_rand.on_clicked(self._on_random)
        self.btn_layout = Button(ax_layout, f"layout: {self.layout_kind}")
        self.btn_layout.on_clicked(self._on_layout)
        self.btn_turbo = Button(ax_turbo, "turbo: OFF")
        self.btn_turbo.on_clicked(self._on_turbo)
        self.beta  = 1.0
        self.turbo = False

        # colorbar: log scale so small probabilities are visible
        self._norm = LogNorm(vmin=1e-3, vmax=1.0)
        self._sm   = ScalarMappable(norm=self._norm, cmap=plt.cm.viridis)
        self._cb   = self.fig.colorbar(self._sm, ax=self.ax_dual,
                                       fraction=0.04, pad=0.02,
                                       extend="min")
        self._cb.set_label("model probability (legal simps, log scale)")

        # event handlers (attached once; live across polygon swaps)
        self.fig.canvas.mpl_connect("button_press_event",   self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event",  self._on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("key_press_event",      self._on_key)
        self.fig.canvas.mpl_connect("key_release_event",    self._on_key_release)
        self.fig.text(
            0.66, 0.17,
            "click: place/unplace simp\n"
            "drag:  move node\n"
            "[n]:   step  (turbo: hold)\n"
            "[r]:   reset\n"
            "[q]:   quit",
            family="monospace", fontsize=10,
            verticalalignment="top",
        )

        # set on first _load_polygon
        self._artists_placed = []
        self._artists_nodes  = None
        self._dual_edge_lc   = None

        self._drag_idx     = None   # node being grabbed (None = no drag)
        self._drag_started = False  # True once we've moved past threshold

        # held-step state (see `_on_key` for the X11 auto-repeat dance)
        self._step_held         = False
        self._step_timer        = None
        self._release_timer     = None
        self._last_release_time = 0.0

        self._load_polygon(initial_pts)

    # polygon (re-)load
    # -----------------
    _WARN_NPTS  = 45     # print a heads-up at this size
    _BLOCK_NPTS = 100    # refuse to load anything bigger than this

    def _check_size(self, pts: np.ndarray) -> bool:
        """Warn at >`_WARN_NPTS`, refuse at >`_BLOCK_NPTS`; False if blocked"""
        n = len(pts)
        if n > self._BLOCK_NPTS:
            print(f"[visualize] REFUSING: Npts={n} > {self._BLOCK_NPTS}; "
                  "DualGraph construction and model eval would be impractical "
                  "at this scale. Pick smaller bounds.",
                  flush=True)
            return False
        if n > self._WARN_NPTS:
            print(f"[visualize] warning: Npts={n} > {self._WARN_NPTS}; "
                  "DualGraph construction and model eval may take a few seconds.",
                  flush=True)
        return True

    def _load_polygon(self, pts: np.ndarray) -> None:
        """Rebuild DualGraph + layout + tensors, reset placed, redraw both
        panels. No-op (with a message) if `_check_size` blocks."""
        if not self._check_size(pts):
            return
        self.dualgraph = DualGraph(pts)
        self.N         = self.dualgraph.simps.shape[0]
        self.placed    = np.zeros(self.N, dtype=bool)

        self.ef_t  = torch.from_numpy(self.dualgraph.circ_features) \
                          .float().to(self.device)
        self.ei_t  = torch.from_numpy(self.dualgraph.edges).to(self.device)
        self.cmp_t = torch.from_numpy(self.dualgraph.simp_compat).to(self.device)

        t0 = time.perf_counter()
        self.layout = graph_layout(
            self.dualgraph, kind=self.layout_kind, rng=self.rng,
        )
        print(f"[layout] {self.layout_kind} "
              f"({(time.perf_counter() - t0) * 1000:.1f} ms)", flush=True)

        # clear panels (artists from previous polygon)
        self._artists_placed = []
        self._artists_nodes  = None
        self.ax_poly.clear()
        self.ax_dual.clear()
        self._setup_axes()
        self._draw_static()
        self._refresh()
        print(f"[visualize] Npts={len(pts)}  N_simps={self.N}  "
              f"N_simps_per_ft={self.dualgraph.N_simps_per_ft}", flush=True)

    # setup
    # -----
    def _setup_axes(self):
        """Polygon panel: bounds from the |x|/|y| text boxes, lattice grid.
        Dual panel: fixed [-1.1, 1.1]^2 window (matches graph_layout norm)."""
        try:
            xmax = max(int(self.tb_xmax.text), 1)
            ymax = max(int(self.tb_ymax.text), 1)
        except ValueError:
            xmax = int(self.dualgraph.pts[:, 0].max())
            ymax = int(self.dualgraph.pts[:, 1].max())
        self.ax_poly.set_xlim(-0.5, xmax + 0.5)
        self.ax_poly.set_ylim(-0.5, ymax + 0.5)
        self.ax_poly.set_aspect("equal")
        self.ax_poly.set_xticks(np.arange(0, xmax + 1))
        self.ax_poly.set_yticks(np.arange(0, ymax + 1))
        self.ax_poly.tick_params(labelbottom=False, labelleft=False,
                                 length=0)
        self.ax_poly.grid(True, color="gray", alpha=0.3, linewidth=0.5)
        self.ax_poly.set_axisbelow(True)

        # dual panel: fixed window because graph_layout normalizes to ~[-1,1]
        self.ax_dual.set_xlim(-1.1, 1.1)
        self.ax_dual.set_ylim(-1.1, 1.1)
        self.ax_dual.set_aspect("equal")
        self.ax_dual.set_xticks([]); self.ax_dual.set_yticks([])

    def _draw_static(self):
        """Polygon hull + lattice scatter (left) and dual-edge LineCollection
        (right). Drawn once per `_load_polygon`; persists until next swap."""
        pts  = self.dualgraph.pts
        hull = pts[ConvexHull(pts).vertices]
        self.ax_poly.add_patch(MplPolygon(
            hull, closed=True, fill=False, edgecolor="black", linewidth=1.5,
        ))
        self.ax_poly.scatter(pts[:, 0], pts[:, 1], c="black", s=18, zorder=3)

        edges    = self.dualgraph.edges
        src, dst = edges[0], edges[1]
        self._dual_edge_keep = src < dst   # for update during drag
        seg = np.stack([
            self.layout[src[self._dual_edge_keep]],
            self.layout[dst[self._dual_edge_keep]],
        ], axis=1)
        self._dual_edge_lc = LineCollection(
            seg, colors="lightgray", linewidths=0.5, zorder=1,
        )
        self.ax_dual.add_collection(self._dual_edge_lc)

    # model
    # -----
    def _model_probs(self):
        """
        Run one model forward at the current `placed` state and return the
        per-simp next-step distribution and legal mask. Probs come from
        `softmax(beta * logits)` with illegal simps masked to `-inf`. Returns
        `(probs, legal)`, both `(N,)` numpy arrays.
        """
        placed = torch.from_numpy(self.placed[None, :]).to(self.device)
        legal  = compute_legal(placed, self.cmp_t)
        with torch.no_grad():
            logits = self.net(self.ef_t, self.ei_t, placed, legal)
        logits = logits * self.beta
        # re-mask after multiply (handles beta=0: -inf*0 = nan otherwise)
        logits = logits.masked_fill(~legal, float("-inf"))
        probs  = torch.softmax(logits, dim=-1)
        return probs[0].cpu().numpy(), legal[0].cpu().numpy()

    # redraw
    # ------
    def _add_placed_patch(self, i: int):
        tri = self.dualgraph.pts[self.dualgraph.simps[i]]
        patch = MplPolygon(
            tri, closed=True, facecolor="#4c72b0", edgecolor="black",
            alpha=0.65, linewidth=0.8, zorder=2,
        )
        self.ax_poly.add_patch(patch)
        self._artists_placed.append(patch)

    def _refresh(self, *, incremental_add=None):
        if incremental_add is not None:
            self._add_placed_patch(incremental_add)   # rapid path
        else:
            for a in self._artists_placed:
                a.remove()
            self._artists_placed = []
            for i in np.where(self.placed)[0]:
                self._add_placed_patch(i)
        if self._artists_nodes is not None:
            self._artists_nodes.remove()

        probs, legal = self._model_probs()
        colors = np.empty((self.N, 4))
        colors[:] = [0.88, 0.88, 0.88, 0.35]                  # faded default
        legal_active = legal & ~self.placed
        if legal_active.any():
            # clamp to vmin so tiny probs render at the bottom of cmap
            colors[legal_active] = plt.cm.viridis(
                self._norm(np.maximum(probs[legal_active], 1e-3))
            )
        colors[self.placed] = [0.15, 0.15, 0.15, 1.0]
        sizes      = np.where(self.placed | legal, 80, 12)
        linewidths = np.where(self.placed | legal, 0.6, 0.0)
        self._artists_nodes = self.ax_dual.scatter(
            self.layout[:, 0], self.layout[:, 1],
            c=colors, s=sizes, edgecolors="black", linewidths=linewidths,
            zorder=4,
        )

        n_placed = int(self.placed.sum())
        n_legal  = int(legal.sum())
        target   = self.dualgraph.N_simps_per_ft
        self.ax_poly.set_xlabel(
            f"placed: {n_placed}/{target}   legal next: {n_legal}",
        )
        self.fig.canvas.draw_idle()

    # events
    # ------
    # press grabs the nearest node (if within threshold); a quick release
    # without movement = place; release after movement = finalize a drag.
    _GRAB_RADIUS  = 0.08    # data-coord radius to grab a node on press
    _DRAG_TRIGGER = 0.04    # min motion (data coords) to start dragging

    def _on_press(self, event):
        # double-click on an entry field clears it (typing replaces)
        if event.dblclick:
            for tb in (self.tb_xmax, self.tb_ymax, self.tb_beta):
                if event.inaxes is tb.ax:
                    self._suppress_validation = True
                    tb.set_val("")
                    self._suppress_validation = False
                    return
        # polygon panel: hull-vertex toggle
        if event.inaxes is self.ax_poly:
            if event.xdata is None or event.ydata is None:
                return
            x = int(round(event.xdata))
            y = int(round(event.ydata))
            if (abs(event.xdata - x) > 0.4
                    or abs(event.ydata - y) > 0.4):
                return
            self._toggle_hull_vertex(np.array([x, y], dtype=np.int64))
            return

        # dual panel: drag / place
        if event.inaxes is not self.ax_dual:
            return
        if event.xdata is None or event.ydata is None:
            return
        d2 = ((self.layout[:, 0] - event.xdata) ** 2
              + (self.layout[:, 1] - event.ydata) ** 2)
        i = int(np.argmin(d2))
        if d2[i] > self._GRAB_RADIUS ** 2:
            return
        self._drag_idx     = i
        self._drag_started = False

    def _toggle_hull_vertex(self, click_xy: np.ndarray):
        """Add `click_xy` to the hull seed if new, remove if it's currently a
        hull vertex; interior / facet-interior points are no-ops."""
        pts   = self.dualgraph.pts
        match = np.where(
            (pts[:, 0] == click_xy[0]) & (pts[:, 1] == click_xy[1])
        )[0]
        if len(match) == 0:
            seed = np.concatenate([pts, click_xy[None, :]], axis=0)
            print(f"[edit] add vertex {tuple(click_xy.tolist())}",
                  flush=True)
        else:
            idx = int(match[0])
            try:
                hull_idxs = list(ConvexHull(pts).vertices)
            except Exception:
                return
            if idx not in hull_idxs:
                print(f"[edit] {tuple(click_xy.tolist())} is interior; "
                      f"no-op", flush=True)
                return
            kept = np.delete(pts, idx, axis=0)
            if len(kept) < 3:
                print("[edit] removing would leave <3 vertices; no-op",
                      flush=True)
                return
            print(f"[edit] remove vertex {tuple(click_xy.tolist())}",
                  flush=True)
            seed = kept
        new_pts = enum_lattice_pts(seed)
        if new_pts is None:
            print("[edit] degenerate polygon; no-op", flush=True)
            return
        self._load_polygon(new_pts)

    def _on_motion(self, event):
        if self._drag_idx is None:
            return
        if event.inaxes is not self.ax_dual:
            return
        if event.xdata is None or event.ydata is None:
            return
        i = self._drag_idx
        dx = event.xdata - self.layout[i, 0]
        dy = event.ydata - self.layout[i, 1]
        if not self._drag_started and (dx*dx + dy*dy) < self._DRAG_TRIGGER ** 2:
            return
        self._drag_started = True
        self.layout[i, 0] = event.xdata
        self.layout[i, 1] = event.ydata
        self._update_positions()

    def _on_release(self, event):
        if self._drag_idx is None:
            return
        i, started = self._drag_idx, self._drag_started
        self._drag_idx     = None
        self._drag_started = False
        if started:
            return        # finalize drag
        self._try_place(i)

    def _status(self, action: str, i: int, prefix: str = "  ") -> str:
        """`{prefix}{action} simp[i]  (n_placed/n_simps_per_ft)` log line"""
        return (f"{prefix}{action} simp[{i}]  "
                f"({int(self.placed.sum())}/{self.dualgraph.N_simps_per_ft})")

    def _try_place(self, i: int):
        if self.placed[i]:
            self.placed[i] = False
            print(self._status("removed", i), flush=True)
            self._refresh()
            return
        _, legal = self._model_probs()
        if not legal[i]:
            print(f"  simp[{i}] not legal", flush=True)
            return
        self.placed[i] = True
        print(self._status("placed", i), flush=True)
        self._refresh()

    def _update_positions(self):
        """Drag-time redraw: scatter offsets + edge segments only, no
        recolor / no model forward."""
        self._artists_nodes.set_offsets(self.layout)
        src, dst = self.dualgraph.edges
        keep     = self._dual_edge_keep
        seg = np.stack([
            self.layout[src[keep]],
            self.layout[dst[keep]],
        ], axis=1)
        self._dual_edge_lc.set_segments(seg)
        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        if event.key == "r":
            self.placed[:] = False
            print("[reset]", flush=True)
            self._refresh()
        elif event.key in ("n", " "):
            # A press with a release pending could be either (a) X11 auto-
            # repeat's fake release/press pair (within a few ms) or (b) a
            # real manual tap landing inside the 150 ms release-debounce.
            # Discriminate by elapsed time: <_REPEAT_GAP_S = auto-repeat
            # (keep cadence), otherwise finalize prior release so this tap
            # is treated as a fresh step.
            if self._release_timer is not None:
                self._release_timer.stop()
                self._release_timer = None
                if (time.perf_counter() - self._last_release_time
                        > self._REPEAT_GAP_S):
                    self._finalize_release()
            if self._step_held:
                return                    # ignore OS auto-repeat
            self._step_held = True
            self._do_step()
            if self.turbo:
                self._step_timer = self.fig.canvas.new_timer(interval=60)
                self._step_timer.add_callback(self._do_step)
                self._step_timer.start()
        elif event.key == "q":
            plt.close(self.fig)

    def _do_step(self):
        probs, legal = self._model_probs()
        if not legal.any():
            print("[step] no legal simps (triangulation complete)", flush=True)
            if self._step_timer is not None:
                self._step_timer.stop()
                self._step_timer = None
            return
        probs = probs * legal.astype(probs.dtype)
        i = int(self.rng.choice(self.N, p=probs / probs.sum()))
        self.placed[i] = True
        print(self._status("sampled", i, prefix="[step] "), flush=True)
        self._refresh(incremental_add=i)

    # Threshold used in _on_key to distinguish X11 auto-repeat's fake
    # release/press pairs (~few ms) from real manual taps (~>=100 ms).
    _REPEAT_GAP_S = 0.05

    def _on_key_release(self, event):
        if event.key in ("n", " "):
            # may be a fake release injected by X11 auto-repeat; confirm
            # after 150 ms. _on_key cancels this timer if a press lands first.
            self._last_release_time = time.perf_counter()
            if self._release_timer is not None:
                self._release_timer.stop()
            self._release_timer = self.fig.canvas.new_timer(interval=150)
            self._release_timer.single_shot = True
            self._release_timer.add_callback(self._finalize_release)
            self._release_timer.start()

    def _finalize_release(self):
        self._release_timer = None
        self._step_held = False
        if self._step_timer is not None:
            self._step_timer.stop()
            self._step_timer = None

    def _on_beta_change(self, text):
        try:
            self.beta = float(text)
        except ValueError:
            print(f"[beta] invalid: {text!r}", flush=True)
            return
        print(f"[beta] set to {self.beta}", flush=True)
        self._refresh()

    _INT_RE   = re.compile(r"^\d*$")
    _FLOAT_RE = re.compile(r"^-?\d*\.?\d*$")

    def _validator(self, textbox, attr, regex):
        """Build an on_text_change cb that accepts `regex`-matching input
        (stored in `attr`) and reverts any other keystroke."""
        def cb(text):
            if self._suppress_validation: return
            if regex.match(text):
                setattr(self, attr, text)
            else:
                self._suppress_validation = True
                textbox.set_val(getattr(self, attr))
                self._suppress_validation = False
        return cb

    def _on_turbo(self, event):
        self.turbo = not self.turbo
        self.btn_turbo.label.set_text("turbo: ON" if self.turbo else "turbo: OFF")
        self.fig.canvas.draw_idle()

    def _on_layout(self, event):
        kinds = list(_LAYOUT_KINDS)
        i = kinds.index(self.layout_kind)
        self.layout_kind = kinds[(i + 1) % len(kinds)]
        self.btn_layout.label.set_text(f"layout: {self.layout_kind}")
        t0 = time.perf_counter()
        self.layout = graph_layout(
            self.dualgraph, kind=self.layout_kind, rng=self.rng,
        )
        print(f"[layout] {self.layout_kind} "
              f"({(time.perf_counter() - t0) * 1000:.1f} ms)", flush=True)
        self._update_positions()

    def _on_random(self, event):
        try:
            xmax = int(self.tb_xmax.text)
            ymax = int(self.tb_ymax.text)
        except ValueError:
            print(f"[random] invalid bounds: x='{self.tb_xmax.text}' "
                  f"y='{self.tb_ymax.text}'", flush=True)
            return
        if xmax < 1 or ymax < 1:
            print("[random] bounds must be >= 1", flush=True)
            return
        pts = random_polygon(xmax, ymax, self.rng)
        if pts is None:
            print(f"[random] failed to find a polygon in "
                  f"[0,{xmax}]x[0,{ymax}]", flush=True)
            return
        print(f"[random] new polygon: |x|<={xmax} |y|<={ymax} "
              f"Npts={len(pts)}", flush=True)
        self._load_polygon(pts)


def main():
    p = argparse.ArgumentParser(
        description="Interactive dualGNN AR-rollout viewer: build a lattice "
                    "polygon (left), inspect its dual graph and the model's "
                    "next-simp distribution (right).",
    )
    p.add_argument("--ckpt", type=Path, default=None,
                   help="trained DualGNN checkpoint (default: the shipped "
                        "model, DualGNN.default())")
    p.add_argument("--device", type=str, default=None,
                   help="cuda|mps|cpu; autodetected if omitted")
    args = p.parse_args()

    device = args.device or default_device()
    rng = np.random.default_rng()
    pts = random_polygon(8, 8, rng)
    if pts is None:
        raise SystemExit("[visualize] failed to generate a random polygon")
    net = (DualGNN.default(device) if args.ckpt is None
           else DualGNN.from_ckpt(args.ckpt, device))
    print(f"[visualize] device={device}", flush=True)
    viz = Visualizer(net, device, pts)   # noqa: F841
    plt.show()


if __name__ == "__main__":
    main()
