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
# Description:  Dual graph of the simplicial complex relevant to dualGNN. I.e.,
#               the minimum graph containing the dual graph of every fine
#               triangulation.
# -----------------------------------------------------------------------------

# external imports
from __future__ import annotations

import numba
import numpy as np
from scipy.spatial import ConvexHull

# local imports
from .geometry import signed_area2


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
        src_list = []
        dst_list = []
        for si in range(len(self.simps)):
            for sj in range(si + 1, len(self.simps)):
                if not self.simp_compat[si, sj]:
                    continue

                if len(set(self.simps[si]).intersection(self.simps[sj])) == 2:
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
    Two simplices `si`, `sj` are compatible iff they can coexist in a
    triangulation. That is, their interiors do not overlap.

    For each pair of edges `(ei, ej)` from `si` and `sj`, this splits in cases:
        -) `ei == ej` (shared edge): incompatible iff both simps' apexes lie
           on the same side of the shared edge,
        -) shared endpoint but distinct edges: skip (always compatible), or
        -) fully disjoint: incompatible iff the two open segments cross.

    Note: `si` is incompatible with itself.

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
    for ki in range(3):
        ui = simps[si, (ki + 1) % 3]
        vi = simps[si, (ki + 2) % 3]

        for kj in range(3):
            uj = simps[sj, (kj + 1) % 3]
            vj = simps[sj, (kj + 2) % 3]

            same_edge = (ui == uj and vi == vj) or (ui == vj and vi == uj)
            if same_edge:
                # shared edge: incompat iff apexes lie on same side
                wi    = simps[si, ki]
                wj    = simps[sj, kj]
                sidei = signed_area2(pts[ui], pts[vi], pts[wi])
                sidej = signed_area2(pts[ui], pts[vi], pts[wj])
                if (sidei > 0) == (sidej > 0):
                    return False
                continue

            # different edges: skip if they share an endpoint
            if ui == uj or ui == vj or vi == uj or vi == vj:
                continue

            # proper crossing: c, d on opposite sides of ab AND
            # a, b on opposite sides of cd
            a = pts[ui]; b = pts[vi]
            c = pts[uj]; d = pts[vj]
            if  (signed_area2(a, c, d) > 0) != (signed_area2(b, c, d) > 0) \
            and (signed_area2(a, b, c) > 0) != (signed_area2(a, b, d) > 0):
                return False

    return True

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
    N      = simps.shape[0]
    compat = ~np.eye(N, dtype=np.bool_)
    for si in range(N):
        for sj in range(si + 1, N):
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

    # output object
    out = np.empty((Nedges, 4), dtype=np.int32)

    # build the output
    for e in range(Nedges):
        s_src = int(edges[0, e])
        s_dst = int(edges[1, e])

        # get the role of the points
        src_verts = {int(v) for v in simps[s_src]}
        dst_verts = {int(v) for v in simps[s_dst]}
        shared    = src_verts & dst_verts
        v0, v1    = shared

        my   = (src_verts - shared).pop()
        your = (dst_verts - shared).pop()

        # ensure we have the appropriate ordering
        area_v0 = signed_area2(pts[my], pts[your], pts[v0])
        area_v1 = signed_area2(pts[my], pts[your], pts[v1])
        if area_v0 > area_v1:
            e0, e1 = v0, v1
        else:
            e0, e1 = v1, v0

        # store the circuit
        out[e] = _circ_normal(
            pts[my], pts[your], pts[e0], pts[e1]
        )
    return out

def _circ_normal(p_my, p_your, p_e0, p_e1):
    """
    Primitive integer affine-dependence vector for 4 points in the plane.

    Any 4 points in R^2 are affinely dependent: there exist integers
    `n = [n_my, n_your, n_e0, n_e1]`, not all zero, with `sum(n) == 0` and
    `sum(n_i * p_i) == 0`. The vector is unique up to sign and scale.

    `n_i` is (up to sign) the signed area of the triangle on the other 3
    points; we use that identity directly (Cramer's rule). The result is
    then made primitive (divided by `gcd`) and canonicalized to `n_my >= 0`.

    Parameters
    ----------
    p_my, p_your, p_e0, p_e1 : ndarray
        Length-2 int arrays giving the 2D coordinates of the four points,
        in the named roles. Convention: `(p_my, p_e0, p_e1)` is one simp
        and `(p_your, p_e0, p_e1)` is its flip neighbor across edge
        `(e0, e1)`.

    Returns
    -------
    n : ndarray
        `(4,)` int64 primitive coefficients in role order `[my, your, e0, e1]`.
        Sums to zero (as an affine dependence).
    """
    n = np.array([
         signed_area2(p_your, p_e0,   p_e1),
        -signed_area2(p_my,   p_e0,   p_e1),
         signed_area2(p_my,   p_your, p_e1),
        -signed_area2(p_my,   p_your, p_e0),
    ], dtype=np.int64)

    if n[0] < 0:
        n = -n

    # reduce normal by gcd
    g = int(np.gcd.reduce(np.abs(n)))
    n //= g
    return n
