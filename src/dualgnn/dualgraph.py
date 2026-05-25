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
# Description:  DualGraph: nodes = candidate unimodular simplices, edges =
#               compatible pairs. Any fine triangulation's dual graph is
#               a subgraph.
# -----------------------------------------------------------------------------

# external imports
from __future__ import annotations

from collections import defaultdict

import numba
import numpy as np
from scipy.spatial import ConvexHull


# main data class
# ===============
class DualGraph:
    """
    (Static) data class holding the graph dual to the abstract simplicial
    complex over a polygon's 'candidate simplices'.

    I.e., those simplices which contain exactly 3 lattice points in their
    support. In 2D, these are just the unimodular simplices.

    Parameters
    ----------
    pts : ndarray
        `(Npts, 2)` int. All lattice points of the polygon. (Includes interior.)

    Also stores
    -----------
    simps : ndarray
        `(Nsimps, 3)` int. Vertices of candidate simplices (indices into `pts`).
    simp_compat : ndarray
        `(Nsimps, Nsimps)` bool, symmetric, diagonal False. `True` iff the two
        candidate simps have disjoint interiors.
    edges : ndarray
        `(2, Nedges)` int. Directed edges of dual-graph as `[src, dst]`.
    circ_features : ndarray
        `(Nedges, 4)` int. 4D circuit (affine dependency vector) per directed
        edge, in the sender's (= `src`) perspective.
    N_simps_per_ft : int
        Number of simps in any fine triangulation of this polygon
        (`= 2 * area(conv(pts))`).
    """
    def __init__(self, pts: np.ndarray) -> None:
        """
        Initialize a `DualGraph` from the points of a lattice polygon.

        Builds:
            1) the candidate simps (every unimodular triangle)
            2) the pairwise simp_compat matrix (overlap-free pairs)
            3) the simps-per-FT count (`= 2 * area(conv(pts))`)
            4) the directed edges: pairs `(si, sj)` of compatible simps that
               share a facet (= 1D face = 2 lattice points)
            5) the per-edge circuit-normal features (sender's perspective)

        Parameters
        ----------
        pts : ndarray
            `(Npts, 2)` int. Lattice points of the polygon. Includes interior.
        """
        if pts is None:
            raise ValueError(
                "DualGraph(pts) got pts=None. This usually means an upstream "
                "call to `enum_lattice_pts` saw a degenerate polygon "
                "(collinear vertices, <3 unique points, or <3 lattice points "
                "inside the hull)."
            )
        self.pts            = np.ascontiguousarray(pts, dtype=np.int64)
        self.simps          = _candidate_simps(pts)
        self.simp_compat    = _simp_compat(self.pts, self.simps)
        self.N_simps_per_ft = _N_simps_per_ft(self.pts)

        # directed edges (ordered pairs of facet-sharing compatible simps)
        # Index simps by their facets so we only consider pairs that already
        # share one, rather than scanning all N_simps^2 pairs
        edge_to_simps: dict[tuple[int, int], list[int]] = defaultdict(list)
        for si, s in enumerate(self.simps):
            a, b, c = int(s[0]), int(s[1]), int(s[2])
            edge_to_simps[(a, b)].append(si)
            edge_to_simps[(a, c)].append(si)
            edge_to_simps[(b, c)].append(si)

        src_list = []
        dst_list = []
        for simps_with_edge in edge_to_simps.values():
            n = len(simps_with_edge)
            for k1 in range(n):
                si = simps_with_edge[k1]
                for k2 in range(k1 + 1, n):
                    sj = simps_with_edge[k2]
                    if self.simp_compat[si, sj]:
                        src_list += [si, sj]
                        dst_list += [sj, si]

        self.edges = np.stack([
            np.asarray(src_list, dtype=np.int64),
            np.asarray(dst_list, dtype=np.int64),
        ])

        self.circ_features = _circ_features(self.pts, self.simps, self.edges)


# helpers
# =======
# construct various class parameters
def _N_simps_per_ft(pts: np.ndarray) -> int:
    """
    Number of simps in any fine triangulation of the lattice polygon `pts`.

    Every candidate simp is unimodular (area = 1/2), so an FT's simp count
    equals `2 * area(conv(pts))`. Computed via shoelace on the convex hull.
    """
    hull = pts[ConvexHull(pts).vertices]
    x, y = hull[:, 0], hull[:, 1]
    return int(abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))

@numba.njit(cache=True)
def _candidate_simps(pts: np.ndarray) -> np.ndarray:
    """
    Enumerate every unimodular triangle.

    A 2D lattice triangle is unimodular iff (twice) its area is 1. This is true
    iff its three points are the only lattice points in its support (i.e., it
    can be part of a fine triangulation).

    The method is brute force: iterate all `Npts choose 3` triples and keep
    the ones with `|2 * area| == 1`.

    Parameters
    ----------
    pts : ndarray
        `(Npts, 2)` int. Lattice points of the polygon.

    Returns
    -------
    out : ndarray
        `(Nsimps, 3)` int64. Each row is `(i, j, k)` with `i < j < k`, three
        indices into `pts` forming a unimodular triangle.
    """
    Npts = pts.shape[0]

    # output object
    max_N_cands = (Npts * (Npts - 1) * (Npts - 2)) // 6      # Npts choose 3
    out         = np.empty((max_N_cands, 3), dtype=np.int64)
    N_cands     = 0

    # build the output
    # (do so in a dumb way... iterate over all simplices and reject bad ones)
    for i in range(Npts):
        x0 = pts[i, 0]; y0 = pts[i, 1]

        for j in range(i + 1, Npts):
            x1 = pts[j, 0]; y1 = pts[j, 1]

            for k in range(j + 1, Npts):
                x2 = pts[k, 0]; y2 = pts[k, 1]

                # check unimodular (equivalent to fineness for 2D)
                area2x = x0*(y1 - y2) + x1*(y2 - y0) + x2*(y0 - y1)
                if area2x == 1 or area2x == -1:
                    out[N_cands, 0] = i
                    out[N_cands, 1] = j
                    out[N_cands, 2] = k
                    N_cands += 1

    return out[:N_cands]

@numba.njit(cache=True)
def _pair_compat(pts, simps, si, sj):
    """
    Two simplices are compatible iff their interiors do not overlap.

    Separating Axis Theorem: two convex polygons have disjoint interiors
    iff some line lies between them. For triangles it suffices to test
    axes perpendicular to each edge (6 axes total). Non-strict `<=`
    means triangles that share only an edge or vertex are correctly
    counted as non-overlapping.

    Note: `si` is incompatible with itself (returns False).

    Parameters
    ----------
    pts : ndarray
        `(Npts, 2)` int64 lattice points.
    simps : ndarray
        `(Nsimps, 3)` int64 simp point indices.
    si, sj : int
        Indices into `simps`.

    Returns
    -------
    compatible : bool
    """
    ax0 = pts[simps[si, 0], 0]; ay0 = pts[simps[si, 0], 1]
    ax1 = pts[simps[si, 1], 0]; ay1 = pts[simps[si, 1], 1]
    ax2 = pts[simps[si, 2], 0]; ay2 = pts[simps[si, 2], 1]
    bx0 = pts[simps[sj, 0], 0]; by0 = pts[simps[sj, 0], 1]
    bx1 = pts[simps[sj, 1], 0]; by1 = pts[simps[sj, 1], 1]
    bx2 = pts[simps[sj, 2], 0]; by2 = pts[simps[sj, 2], 1]

    for k in range(6):
        if   k == 0: nx = ay1 - ay0; ny = ax0 - ax1
        elif k == 1: nx = ay2 - ay1; ny = ax1 - ax2
        elif k == 2: nx = ay0 - ay2; ny = ax2 - ax0
        elif k == 3: nx = by1 - by0; ny = bx0 - bx1
        elif k == 4: nx = by2 - by1; ny = bx1 - bx2
        else:        nx = by0 - by2; ny = bx2 - bx0

        pa0 = ax0 * nx + ay0 * ny
        pa1 = ax1 * nx + ay1 * ny
        pa2 = ax2 * nx + ay2 * ny
        if pa0 < pa1:
            min_a = pa0; max_a = pa1
        else:
            min_a = pa1; max_a = pa0
        if pa2 < min_a: min_a = pa2
        if pa2 > max_a: max_a = pa2

        pb0 = bx0 * nx + by0 * ny
        pb1 = bx1 * nx + by1 * ny
        pb2 = bx2 * nx + by2 * ny
        if pb0 < pb1:
            min_b = pb0; max_b = pb1
        else:
            min_b = pb1; max_b = pb0
        if pb2 < min_b: min_b = pb2
        if pb2 > max_b: max_b = pb2

        if max_a <= min_b or max_b <= min_a:
            return True

    return False

@numba.njit(cache=True)
def _simp_compat(pts, simps):
    """
    For every pair of simplices, compute whether or not they are compatible.

    That is, whether or not they can appear together in a triangulation (their
    interiors do not overlap).

    Parameters
    ----------
    pts : ndarray
        `(Npts, 2)` int64 lattice points.
    simps : ndarray
        `(Nsimps, 3)` int64 candidate simps.

    Returns
    -------
    compat : ndarray
        `(Nsimps, Nsimps)` bool, symmetric, diagonal False.
    """
    N = simps.shape[0]

    # bbox pre-filter: disjoint axis-aligned bboxes => no shared point =>
    # trivially compatible. Skip the expensive geometric check for those
    x_lo = np.empty(N, dtype=np.int64)
    x_hi = np.empty(N, dtype=np.int64)
    y_lo = np.empty(N, dtype=np.int64)
    y_hi = np.empty(N, dtype=np.int64)
    for si in range(N):
        a, b, c = simps[si, 0], simps[si, 1], simps[si, 2]
        xa = pts[a, 0]; xb = pts[b, 0]; xc = pts[c, 0]
        ya = pts[a, 1]; yb = pts[b, 1]; yc = pts[c, 1]
        x_lo[si] = min(xa, min(xb, xc))
        x_hi[si] = max(xa, max(xb, xc))
        y_lo[si] = min(ya, min(yb, yc))
        y_hi[si] = max(ya, max(yb, yc))

    # build in place; ~np.eye(N) would peak at 2 * N^2 bytes
    compat = np.ones((N, N), dtype=np.bool_)
    for si in range(N):
        compat[si, si] = False
    for si in range(N):
        for sj in range(si + 1, N):
            if (x_hi[si] < x_lo[sj] or x_hi[sj] < x_lo[si]
                    or y_hi[si] < y_lo[sj] or y_hi[sj] < y_lo[si]):
                continue                              # disjoint bboxes
            if not _pair_compat(pts, simps, si, sj):
                compat[si, sj] = False
                compat[sj, si] = False
    return compat

def _circ_features(pts, simps, edges):
    """
    Compute the affine dependency vector for each directed edge `(s_src,s_dst)`.

    Order as in the paper:
        1) from the unique point of `s_src` to the unique point of `s_dst`,
           call the leftmost shared point `e0` and the rightmost `e1` and
        2) order `[my, your, e0, e1]`.

    Parameters
    ----------
    pts : ndarray
        `(Npts, 2)` int64 lattice points.
    simps : ndarray
        `(Nsimps, 3)` int64 simp point indices.
    edges : ndarray
        `(2, Nedges)` int64 directed dual edges `[src, dst]`.

    Returns
    -------
    out : ndarray
        `(Nedges, 4)` int32. One primitive 4D circuit per directed edge,
        in `[my, your, e0, e1]` order.
    """
    Nedges = edges.shape[1]
    src    = simps[edges[0]]   # (Nedges, 3)
    dst    = simps[edges[1]]   # (Nedges, 3)

    # match[e, i, j] = (src[e, i] == dst[e, j]): per-edge 3x3 table marking
    # which src/dst vertex pairs coincide. We use it once to peel off the
    # unique src ("my"), the unique dst ("your"), and the two shared verts
    match = (src[:, :, None] == dst[:, None, :])
    shared_src = match.any(axis=2)
    shared_dst = match.any(axis=1)

    my   = src[~shared_src]
    your = dst[~shared_dst]
    v01  = src[shared_src].reshape(Nedges, 2)
    v0, v1 = v01[:, 0], v01[:, 1]

    my_p, your_p = pts[my], pts[your]
    v0_p, v1_p   = pts[v0], pts[v1]

    # order the shared verts so that signed_area2(my, your, e0) > .(.. e1)
    a0   = _cross2_batch(my_p, your_p, v0_p)
    a1   = _cross2_batch(my_p, your_p, v1_p)
    swap = a0 <= a1
    e0   = np.where(swap, v1, v0)
    e1   = np.where(swap, v0, v1)
    e0_p = pts[e0]
    e1_p = pts[e1]

    # affine dependence via signed areas of the 3 other points (Cramer)
    n = np.stack([
         _cross2_batch(your_p, e0_p, e1_p),
        -_cross2_batch(my_p,   e0_p, e1_p),
         _cross2_batch(my_p,   your_p, e1_p),
        -_cross2_batch(my_p,   your_p, e0_p),
    ], axis=1).astype(np.int64)

    # canonicalize sign so n_my >= 0, then reduce by row-wise gcd
    flip = n[:, 0] < 0
    n    = np.where(flip[:, None], -n, n)
    g    = np.gcd.reduce(np.abs(n), axis=1)
    n   //= g[:, None]
    return n.astype(np.int32)


def _cross2_batch(a, b, c):
    """Batched 2D signed area: `(b - a) x (c - a)` over `(N, 2)` inputs."""
    return ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
          - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))

