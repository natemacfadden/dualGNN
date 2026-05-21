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
#                 right panel: dual graph (spring layout), nodes colored by the
#                              model's next-simp probability.
#               Click a legal node on the right to place that simp. The
#               bottom strip has |x|<= / |y|<= bounds and a [Random] button to
#               generate a fresh random polygon under those bounds.
#
# Usage:
#     python scripts/visualize.py \
#         --pts "0,0;1,0;...;4,4" \
#         --ckpt ckpts/D32K16.pt
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
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
def parse_pts(s: str) -> np.ndarray:
    """
    Parse `'x0,y0;x1,y1;...'` into an (Npts, 2) int64 array.

    Parameters
    ----------
    s : str
        Semicolon-separated `"x,y"` pairs.

    Returns
    -------
    pts : ndarray
        `(Npts, 2)` int64.
    """
    rows = [r.strip() for r in s.split(";") if r.strip()]
    return np.asarray(
        [[int(c) for c in r.split(",")] for r in rows],
        dtype=np.int64,
    )

def autodetect_device() -> str:
    if torch.cuda.is_available():           return "cuda"
    if torch.backends.mps.is_available():   return "mps"
    return "cpu"

# layout
# ======
def graph_layout(dualgraph: DualGraph,
                 rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Random uniform layout in `[-1, 1]^2`. Each node gets an independent random
    position; no graph structure is used. Nearly instant and spreads nodes more
    evenly than a Gaussian (no center clumping). Drag-to-move is the intended
    way to organize.

    Parameters
    ----------
    dualgraph : DualGraph
        Source of `Nsimps`.
    rng : np.random.Generator, optional
        Source of randomness. Default: a fresh `default_rng()`.

    Returns
    -------
    layout : ndarray
        `(Nsimps, 2)` float. Per-node 2D positions in `[-1, 1]^2`.
    """
    N   = dualgraph.simps.shape[0]
    rng = rng if rng is not None else np.random.default_rng()
    return rng.uniform(-1, 1, size=(N, 2))

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
    Right panel: dual graph, spring layout, nodes colored by the model's
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
                 initial_pts: np.ndarray):
        self.net    = net
        self.device = device
        self.rng    = np.random.default_rng()

        # figure + widgets
        self.fig, (self.ax_poly, self.ax_dual) = plt.subplots(
            1, 2, figsize=(13, 7.0),
        )
        self.fig.subplots_adjust(bottom=0.18)

        # bottom strip widgets
        ax_xmax = self.fig.add_axes([0.13, 0.04, 0.05, 0.05])
        ax_ymax = self.fig.add_axes([0.27, 0.04, 0.05, 0.05])
        ax_beta = self.fig.add_axes([0.40, 0.04, 0.05, 0.05])
        ax_btn  = self.fig.add_axes([0.52, 0.04, 0.10, 0.05])

        # default bounds = 4, but grow to fit the initial polygon
        init_xmax = max(4, int(initial_pts[:, 0].max()))
        init_ymax = max(4, int(initial_pts[:, 1].max()))
        self.tb_xmax = TextBox(ax_xmax, "|x_i| <= ", initial=str(init_xmax))
        self.tb_ymax = TextBox(ax_ymax, "|y_i| <= ", initial=str(init_ymax))
        self.tb_beta = TextBox(ax_beta, "beta = ",   initial="1.0")
        self.tb_beta.on_submit(self._on_beta_change)
        self.btn_rand = Button(ax_btn, "Random")
        self.btn_rand.on_clicked(self._on_random)
        self.beta = 1.0

        # colorbar: log scale so small probabilities are visible
        self._norm = LogNorm(vmin=1e-3, vmax=1.0)
        self._sm   = ScalarMappable(norm=self._norm, cmap=plt.cm.viridis)
        self._cb   = self.fig.colorbar(self._sm, ax=self.ax_dual,
                                       fraction=0.04, pad=0.02)
        self._cb.set_label("model probability (legal simps, log scale)")

        # event handlers (attached once; live across polygon swaps)
        self.fig.canvas.mpl_connect("button_press_event",   self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event",  self._on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("key_press_event",      self._on_key)
        self.fig.suptitle(
            "click a legal node to place that simp; drag to move a node     "
            "[r] reset   [n] sample-step   [q] quit",
            fontsize=10,
        )

        # placeholders, set in _load_polygon
        self._artists_placed = []
        self._artists_nodes  = None
        self._dual_edge_lc   = None

        # drag state
        self._drag_idx     = None   # node being grabbed (None = no drag)
        self._drag_started = False  # True once we've moved past threshold

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
                  "spring layout, simp_compat allocation, and model eval "
                  "would be impractical at this scale. Pick smaller bounds.",
                  flush=True)
            return False
        if n > self._WARN_NPTS:
            print(f"[visualize] warning: Npts={n} > {self._WARN_NPTS}; "
                  "spring layout and model eval may take a few seconds.",
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

        self.layout = graph_layout(self.dualgraph, rng=self.rng)

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
        self.ax_poly.set_title("polygon (placed simps)")

        # dual panel: fixed window because graph_layout normalizes to ~[-1,1]
        self.ax_dual.set_xlim(-1.1, 1.1)
        self.ax_dual.set_ylim(-1.1, 1.1)
        self.ax_dual.set_aspect("equal")
        self.ax_dual.set_xticks([]); self.ax_dual.set_yticks([])
        self.ax_dual.set_title("dual graph (click to place)")

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
    def _refresh(self):
        for a in self._artists_placed:
            a.remove()
        self._artists_placed = []
        if self._artists_nodes is not None:
            self._artists_nodes.remove()

        # left: placed simps as filled triangles
        pts = self.dualgraph.pts
        for i in np.where(self.placed)[0]:
            tri = pts[self.dualgraph.simps[i]]
            patch = MplPolygon(
                tri, closed=True, facecolor="#4c72b0", edgecolor="black",
                alpha=0.65, linewidth=0.8, zorder=2,
            )
            self.ax_poly.add_patch(patch)
            self._artists_placed.append(patch)

        # right: nodes colored by status + prob (log-scaled so small
        # probabilities are visible; absolute, so all legal probs sum to 1)
        probs, legal = self._model_probs()

        colors = np.empty((self.N, 4))
        for i in range(self.N):
            if self.placed[i]:
                colors[i] = (0.15, 0.15, 0.15, 1.0)
            elif not legal[i]:
                colors[i] = (0.88, 0.88, 0.88, 0.35)   # faded
            else:
                # clamp to vmin so tiny probs render at the bottom of cmap
                colors[i] = plt.cm.viridis(self._norm(max(probs[i], 1e-3)))
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
        self.fig.canvas.draw_idle()

    # events
    # ------
    # press grabs the nearest node (if within threshold); a quick release
    # without movement = place; release after movement = finalize a drag.
    _GRAB_RADIUS  = 0.08    # data-coord radius to grab a node on press
    _DRAG_TRIGGER = 0.04    # min motion (data coords) to start dragging

    def _on_press(self, event):
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
            probs, legal = self._model_probs()
            if not legal.any():
                print("[step] no legal simps (triangulation complete)",
                      flush=True)
                return
            probs = probs * legal.astype(probs.dtype)
            i = int(self.rng.choice(self.N, p=probs / probs.sum()))
            self.placed[i] = True
            print(f"[step] sampled simp[{i}]  "
                  f"({int(self.placed.sum())}/{self.dualgraph.N_simps_per_ft})",
                  flush=True)
            self._announce_regularity()
            self._refresh()
        elif event.key == "q":
            plt.close(self.fig)

    def _on_beta_change(self, text):
        try:
            self.beta = float(text)
        except ValueError:
            print(f"[beta] invalid: {text!r}", flush=True)
            return
        print(f"[beta] set to {self.beta}", flush=True)
        self._refresh()

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
    p.add_argument("--pts",  type=str, default=None,
                   help="polygon points 'x,y;x,y;...'. The convex hull is "
                        "taken and filled with every interior lattice point, "
                        "so you can pass just hull vertices (e.g. "
                        "'0,0;4,0;0,6') or the full lattice point list. "
                        "If omitted, a random polygon in [0, 4] x [0, 4] is "
                        "drawn.")
    p.add_argument("--ckpt", type=Path, required=True,
                   help="trained DualGNN checkpoint")
    p.add_argument("--device", type=str, default=None,
                   help="cuda|mps|cpu; autodetected if omitted")
    args = p.parse_args()

    device = args.device or autodetect_device()
    if args.pts is None:
        rng = np.random.default_rng()
        pts = random_polygon(4, 4, rng)
        if pts is None:
            raise SystemExit("[visualize] failed to generate a random polygon")
    else:
        raw = parse_pts(args.pts)
        pts = enum_lattice_pts(raw)
        if pts is None:
            raise SystemExit(
                f"[visualize] --pts is degenerate (collinear or <3 unique "
                f"points): {args.pts}"
            )
        if len(pts) > Visualizer._BLOCK_NPTS:
            raise SystemExit(
                f"[visualize] initial polygon Npts={len(pts)} > "
                f"{Visualizer._BLOCK_NPTS}; pick a smaller polygon."
            )
    net = DualGNN.from_ckpt(args.ckpt, device)
    print(f"[visualize] device={device}", flush=True)
    viz = Visualizer(net, device, pts)   # noqa: F841
    plt.show()


if __name__ == "__main__":
    main()
