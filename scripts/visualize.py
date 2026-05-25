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
# Description:  Interactive GUI for AR rollout.
#                 left  panel: polygon with placed simps as filled triangles.
#                 right panel: dual graph (random layout), nodes colored by the
#                              model's next-simp probability.
#               Click a legal node on the right to place that simp. The
#               bottom strip has |x|<= / |y|<= bounds and a [Random] button to
#               generate a fresh random polygon under those bounds.
#
# Usage:
#     python scripts/visualize.py --ckpt ckpts/reinforce.pt
#
# The initial polygon is a random draw in [0, 4] x [0, 4]; use the bottom
# strip's |x|<= / |y|<= bounds + [Random] button (or click in the polygon
# panel to add/remove a hull vertex) to change it.
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
from dualgnn.dualgraph import DualGraph
from dualgnn.geometry  import enum_lattice_pts, is_regular
from dualgnn.model     import DualGNN
from dualgnn.sampler   import compute_legal

# matplotlib steals "s" for save-figure; turn that off so it can be our key.
plt.rcParams["keymap.save"] = []


# CLI + helpers
# =============
def autodetect_device() -> str:
    if torch.cuda.is_available():           return "cuda"
    if torch.backends.mps.is_available():   return "mps"
    return "cpu"

# layout
# ======
_LAYOUT_KINDS = ("random", "centroid", "spectral", "spring")

def graph_layout(dualgraph: DualGraph,
                 kind: str = "random",
                 rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Compute a 2D layout for the dual graph; output normalized to ~`[-1, 1]^2`.

    Parameters
    ----------
    dualgraph : DualGraph
        Source graph.
    kind : str
        One of:
          - "random":   uniform in `[-1, 1]^2`. Instant, no structure.
          - "centroid": triangle centroid in polygon coords, with collision
                        spreading (simps sharing a centroid are placed on a
                        tiny ring around the shared point). `O(N)`.
          - "spectral": bottom non-trivial eigenvectors of the symmetric
                        normalized graph Laplacian, with percentile clipping
                        to suppress spires. `O(N^3)` (dense `eigh`).
          - "spring":   vectorized Fruchterman-Reingold, warm-started from
                        the centroid layout. `O(iters * N^2)` in pure numpy.
    rng : np.random.Generator, optional
        Source of randomness for "random". Other kinds are deterministic.

    Returns
    -------
    layout : ndarray
        `(Nsimps, 2)` float.
    """
    if rng is None:
        rng = np.random.default_rng()
    if kind == "random":
        return _random_layout(dualgraph, rng)
    if kind == "centroid":
        return _centroid_layout(dualgraph)
    if kind == "spectral":
        return _spectral_layout(dualgraph)
    if kind == "spring":
        return _spring_layout(dualgraph)
    raise ValueError(f"unknown layout kind: {kind!r}")


def _random_layout(dualgraph: DualGraph,
                   rng: np.random.Generator) -> np.ndarray:
    N = dualgraph.simps.shape[0]
    return rng.uniform(-1, 1, size=(N, 2))


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
    """
    Generate a random convex lattice polygon in `[0, xmax] x [0, ymax]`.

    Picks `n in [3, 7]` random integer vertices, then convex-hull-fills.

    Parameters
    ----------
    xmax, ymax : int
        Coord ranges: vertices in `[0, xmax] x [0, ymax]`.
    rng : np.random.Generator
        Source of randomness.
    max_tries : int, optional
        Bail after this many failed attempts. Default 50.

    Returns
    -------
    pts : ndarray or None
        `(Npts, 2)` int64 lattice points, or `None` if no non-degenerate
        polygon was produced in `max_tries` attempts.
    """
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
        init_xmax = max(6, int(initial_pts[:, 0].max()))
        init_ymax = max(6, int(initial_pts[:, 1].max()))
        self.tb_xmax = TextBox(ax_xmax, "0 <= x_i <= ", initial=str(init_xmax))
        self.tb_ymax = TextBox(ax_ymax, "0 <= y_i <= ", initial=str(init_ymax))
        self.tb_beta = TextBox(ax_beta, "beta=1/T=", initial="1.0")
        self.tb_beta.on_submit(self._on_beta_change)
        # numeric-only filters: int for xmax/ymax, float for beta. Reject any
        # other keystroke by reverting to the last valid value.
        self._suppress_validation = False
        self._last_valid_xmax = str(init_xmax)
        self._last_valid_ymax = str(init_ymax)
        self._last_valid_beta = "1.0"
        self.tb_xmax.on_text_change(self._validate_int_xmax)
        self.tb_ymax.on_text_change(self._validate_int_ymax)
        self.tb_beta.on_text_change(self._validate_float_beta)
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
            "[d]:   debug timings\n"
            "[q]:   quit",
            family="monospace", fontsize=10,
            verticalalignment="top",
        )

        # placeholders, set in _load_polygon
        self._artists_placed = []
        self._artists_nodes  = None
        self._dual_edge_lc   = None

        # drag state
        self._drag_idx     = None   # node being grabbed (None = no drag)
        self._drag_started = False  # True once we've moved past threshold

        # held-step state: our timer drives the step cadence while n/space is
        # held. _release_timer delays the take-effect of a release by 150 ms
        # so X11 auto-repeat's fake release/press pairs don't break cadence.
        # A press landing >_REPEAT_GAP_S after the release is treated as a
        # real new tap, not auto-repeat, so rapid manual tapping fires.
        self._step_held         = False
        self._step_timer        = None
        self._release_timer     = None
        self._last_release_time = 0.0

        # debug mode: collects per-step phase timings, emits a summary table
        # on key release. Toggled by [d].
        self._debug:         bool = False
        self._step_log:      list[dict] = []
        self._last_step_end: float | None = None
        self._refresh_phase: dict[str, float] = {}

        self._load_polygon(initial_pts)

    # polygon (re-)load
    # -----------------
    _WARN_NPTS  = 45     # print a heads-up at this size
    _BLOCK_NPTS = 100    # refuse to load anything bigger than this

    def _check_size(self, pts: np.ndarray) -> bool:
        """
        Print warning / block based on `Npts`. Returns False if blocked.
        """
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
        """
        Build a fresh `DualGraph` from `pts`, recompute layout + tensors, reset
        placed state, redraw both panels. No-op (with a message) if the polygon
        would be too large.
        """
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
        """
        Configure both panels: polygon-panel bounds from the current `|x| <=` /
        `|y| <=` text boxes (so polygons stay to-scale across reloads), and a
        fixed `[-1, 1]^2` window on the dual panel matching `graph_layout`'s
        normalization. Adds an integer lattice grid to the polygon panel.
        """
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
        """
        Draw the polygon's hull outline + lattice-point scatter (left panel)
        and the dual-graph edge collection (right panel). Called once per
        `_load_polygon`; the artists persist until the next polygon swap.
        """
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
    def _refresh(self, *, incremental_add=None):
        t = time.perf_counter()
        pts = self.dualgraph.pts
        if incremental_add is not None:
            # rapid path: append only the new patch
            tri = pts[self.dualgraph.simps[incremental_add]]
            patch = MplPolygon(
                tri, closed=True, facecolor="#4c72b0", edgecolor="black",
                alpha=0.65, linewidth=0.8, zorder=2,
            )
            self.ax_poly.add_patch(patch)
            self._artists_placed.append(patch)
        else:
            for a in self._artists_placed:
                a.remove()
            self._artists_placed = []
            for i in np.where(self.placed)[0]:
                tri = pts[self.dualgraph.simps[i]]
                patch = MplPolygon(
                    tri, closed=True, facecolor="#4c72b0", edgecolor="black",
                    alpha=0.65, linewidth=0.8, zorder=2,
                )
                self.ax_poly.add_patch(patch)
                self._artists_placed.append(patch)
        if self._artists_nodes is not None:
            self._artists_nodes.remove()
        self._refresh_phase["patches"] = time.perf_counter() - t

        # right: nodes colored by status + prob (log-scaled so small
        # probabilities are visible; absolute, so all legal probs sum to 1)
        t = time.perf_counter()
        probs, legal = self._model_probs()
        self._refresh_phase["forward2"] = time.perf_counter() - t

        t = time.perf_counter()
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

        # status
        n_placed = int(self.placed.sum())
        n_legal  = int(legal.sum())
        target   = self.dualgraph.N_simps_per_ft
        self.ax_poly.set_xlabel(
            f"placed: {n_placed}/{target}   legal next: {n_legal}",
        )
        self._refresh_phase["scatter"] = time.perf_counter() - t

        t = time.perf_counter()
        self.fig.canvas.draw_idle()
        self._refresh_phase["draw_idle"] = time.perf_counter() - t

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
        """
        Click in polygon panel: if `click_xy` is a current hull vertex,
        remove it from the hull seed; if it's a lattice point not yet in the
        polygon, add it. Interior / facet-interior points are no-ops.
        """
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

    def _try_place(self, i: int):
        if self.placed[i]:
            self.placed[i] = False
            print(f"  removed simp[{i}]  "
                  f"({int(self.placed.sum())}/{self.dualgraph.N_simps_per_ft})",
                  flush=True)
            self._refresh()
            return
        _, legal = self._model_probs()
        if not legal[i]:
            print(f"  simp[{i}] not legal", flush=True)
            return
        self.placed[i] = True
        print(f"  placed simp[{i}]  "
              f"({int(self.placed.sum())}/{self.dualgraph.N_simps_per_ft})",
              flush=True)
        self._announce_regularity()
        self._refresh()

    def _announce_regularity(self):
        """
        On a fully-placed FT, run a regularity check on the placed simps and
        print `REGULAR` or `IRREGULAR` to the terminal. No-op while the
        triangulation is still incomplete.
        """
        if int(self.placed.sum()) != self.dualgraph.N_simps_per_ft:
            return
        simps = self.dualgraph.simps[self.placed]
        tag   = "REGULAR" if is_regular(self.dualgraph.pts, simps) else "IRREGULAR"
        print(f"  [complete] {tag}", flush=True)

    def _update_positions(self):
        """
        Cheap redraw during a drag: update the node scatter's offsets and the
        dual-edge `LineCollection` segments to the current `self.layout`,
        without touching colors or recomputing model probs. Used by
        `_on_motion` for live cursor following.
        """
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
        elif event.key == "d":
            self._debug = not self._debug
            print(f"[debug] {'ON' if self._debug else 'OFF'}", flush=True)
        elif event.key == "q":
            plt.close(self.fig)

    def _do_step(self):
        t0 = time.perf_counter()
        probs, legal = self._model_probs()
        t1 = time.perf_counter()
        if not legal.any():
            print("[step] no legal simps (triangulation complete)",
                  flush=True)
            if self._step_timer is not None:
                self._step_timer.stop()
                self._step_timer = None
            return
        probs = probs * legal.astype(probs.dtype)
        i = int(self.rng.choice(self.N, p=probs / probs.sum()))
        self.placed[i] = True
        print(f"[step] sampled simp[{i}]  "
              f"({int(self.placed.sum())}/{self.dualgraph.N_simps_per_ft})",
              flush=True)
        self._announce_regularity()
        t2 = time.perf_counter()
        self._refresh_phase = {}
        self._refresh(incremental_add=i)
        t3 = time.perf_counter()
        if self._debug:
            rec = {
                "forward1": (t1 - t0) * 1000,
                "between":  (t2 - t1) * 1000,
                "refresh":  (t3 - t2) * 1000,
                "total":    (t3 - t0) * 1000,
                "placed":   int(self.placed.sum()),
                **{k: v * 1000 for k, v in self._refresh_phase.items()},
            }
            if self._last_step_end is not None:
                rec["gap"] = (t0 - self._last_step_end) * 1000
            self._last_step_end = t3
            self._step_log.append(rec)

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
        if self._debug:
            self._dump_step_log()
        else:
            self._step_log.clear()
            self._last_step_end = None

    def _dump_step_log(self):
        if not self._step_log:
            return
        n = len(self._step_log)
        rows = [
            ("forward1",  "model forward (sample next simp)"),
            ("between",   "selection + regularity check"),
            ("refresh",   "refresh subtotal"),
            ("patches",   "  artist update (1 patch)"),
            ("forward2",  "  model forward (recolor)"),
            ("scatter",   "  scatter colors + recreate"),
            ("draw_idle", "  draw_idle queue (paint is async)"),
            ("total",     "TOTAL work per step"),
            ("gap",       "idle gap before next step"),
        ]
        first = self._step_log[0].get("placed", "?")
        last  = self._step_log[-1].get("placed", "?")
        print(f"\n[debug] {n} steps this hold (simps {first} -> {last}, "
              f"times in ms)")
        print(f"  {'phase':36s} {'mean':>7s} {'p50':>7s} "
              f"{'p90':>7s} {'max':>7s}")
        print(f"  {'-' * 36} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7}")
        for key, label in rows:
            vals = sorted(r[key] for r in self._step_log if key in r)
            if not vals:
                continue
            mean = sum(vals) / len(vals)
            p50  = vals[len(vals) // 2]
            p90  = vals[min(len(vals) - 1, int(len(vals) * 0.9))]
            print(f"  {label:36s} {mean:>7.1f} {p50:>7.1f} "
                  f"{p90:>7.1f} {vals[-1]:>7.1f}")
        self._step_log.clear()
        self._last_step_end = None

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

    def _validate_int_xmax(self, text):
        if self._suppress_validation: return
        if self._INT_RE.match(text):
            self._last_valid_xmax = text
        else:
            self._suppress_validation = True
            self.tb_xmax.set_val(self._last_valid_xmax)
            self._suppress_validation = False

    def _validate_int_ymax(self, text):
        if self._suppress_validation: return
        if self._INT_RE.match(text):
            self._last_valid_ymax = text
        else:
            self._suppress_validation = True
            self.tb_ymax.set_val(self._last_valid_ymax)
            self._suppress_validation = False

    def _validate_float_beta(self, text):
        if self._suppress_validation: return
        if self._FLOAT_RE.match(text):
            self._last_valid_beta = text
        else:
            self._suppress_validation = True
            self.tb_beta.set_val(self._last_valid_beta)
            self._suppress_validation = False

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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, default=Path("ckpts/reinforce.pt"),
                   help="trained DualGNN checkpoint "
                        "(default: ckpts/reinforce.pt)")
    p.add_argument("--device", type=str, default=None,
                   help="cuda|mps|cpu; autodetected if omitted")
    args = p.parse_args()

    device = args.device or autodetect_device()
    rng = np.random.default_rng()
    pts = random_polygon(4, 4, rng)
    if pts is None:
        raise SystemExit("[visualize] failed to generate a random polygon")
    net = DualGNN.from_ckpt(args.ckpt, device)
    print(f"[visualize] device={device}", flush=True)
    viz = Visualizer(net, device, pts)   # noqa: F841
    plt.show()


if __name__ == "__main__":
    main()
