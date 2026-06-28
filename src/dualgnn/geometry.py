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
# Description:  Geometry helpers used by dualgnn.
# -----------------------------------------------------------------------------

# external imports
import itertools
import math
import warnings
from collections import defaultdict

import numpy as np
from scipy.spatial import ConvexHull


# polytope
# ========
def compute_bdry(pts: np.ndarray) -> np.ndarray:
    """
    Primitive boundary edges of a 2D convex lattice polygon. Consumed by
    `grow2d` (when called without an explicit `bdry`).

    Iterates over facets `(u,v)` of the polygon, directed ccw (e.g., the
    boundary iterates over `(u,v)`, `(v,w)`, `(w,z)`, ..., `(.,u)`). For edge
    `(u,v)`,
        1) compute the difference `d = v - u` and its gcd `g` and
        2) save the segment `u + [r, r+1]*(d//g)` for `r` in `[0,g)`.

    Parameters
    ----------
    pts : ndarray
        `(Nvert, 2)` int. All lattice points of the polygon.

    Returns
    -------
    out : ndarray
        `(Nedges, 2)` int64. Each row is a sorted `(i, j)` index pair into `pts`
        for one primitive boundary edge.
    """
    pts          = np.ascontiguousarray(pts, dtype=np.int64)
    coord_to_ind = {(int(x), int(y)): i for i, (x, y) in enumerate(pts)}
    hull_verts   = ConvexHull(pts).vertices.astype(np.int64)
    N_hull       = len(hull_verts)

    # build the output
    out = []
    for k in range(N_hull):
        u_idx = int(hull_verts[k])
        v_idx = int(hull_verts[(k + 1) % N_hull])

        u     = pts[u_idx]
        v     = pts[v_idx]

        d     = v - u
        g     = math.gcd(int(abs(d[0])), int(abs(d[1])))
        step  = d // g

        # build each segment
        prev = u_idx
        cur  = u
        for _ in range(g):
            cur     = cur + step
            cur_idx = coord_to_ind[(int(cur[0]), int(cur[1]))]
            a, b    = sorted((prev, cur_idx))
            out.append((a, b))
            prev = cur_idx

    if not out:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(out, dtype=np.int64)

# lattice point enumeration
# -------------------------
def enum_lattice_pts(verts: np.ndarray) -> np.ndarray | None:
    """
    Enumerate every lattice point in the convex hull of `verts`.

    Takes the convex hull of `verts` and then filters lattice points in the
    bounding box for those that lie inside the hull.

    There are much fancier methods to do this, but they aren't really needed
    here.

    Parameters
    ----------
    verts : ndarray
        `(K, 2)` int. Points to take the convex hull of.

    Returns
    -------
    pts : ndarray or None
        `(Npts, 2)` int64. `None` if `verts` is degenerate (low dimensional).
    """
    # compute the hull
    v = np.unique(verts, axis=0)
    if len(v) < 3:
        warnings.warn(
            f"enum_lattice_pts: only {len(v)} unique vertex(es); need >= 3 "
            f"to form a 2D polygon. Returning None."
        )
        return None
    try:
        hull = ConvexHull(v)
    except Exception as e:
        warnings.warn(
            f"enum_lattice_pts: ConvexHull failed ({type(e).__name__}: {e}); "
            f"input is likely collinear or otherwise degenerate. Returning None."
        )
        return None

    # ensure sufficiently high dimension (2D)
    hull_v = v[hull.vertices]
    if np.linalg.matrix_rank(hull_v - hull_v[0]) < 2:
        warnings.warn(
            "enum_lattice_pts: hull is rank-deficient (vertices lie on a "
            "common line). Returning None."
        )
        return None

    # filter the bounding box lattice points. c = (x, y) is inside the (CCW)
    # hull iff every directed edge (a, b) sees it on the non-negative side:
    # twice-signed-area (b-a) x (c-a) >= 0
    x_lo, y_lo = v.min(axis=0)
    x_hi, y_hi = v.max(axis=0)
    xs, ys = np.meshgrid(np.arange(x_lo, x_hi + 1, dtype=np.int64),
                         np.arange(y_lo, y_hi + 1, dtype=np.int64),
                         indexing="ij")
    cand   = np.stack([xs.ravel(), ys.ravel()], axis=1)    # (Nbox, 2)
    a      = hull_v                                        # (N_hull, 2)
    b      = np.roll(hull_v, -1, axis=0)
    cross  = ((b[:, 0] - a[:, 0]) * (cand[:, 1, None] - a[:, 1])
            - (b[:, 1] - a[:, 1]) * (cand[:, 0, None] - a[:, 0]))
    pts    = cand[(cross >= 0).all(axis=1)]

    if len(pts) < 3:
        warnings.warn(
            f"enum_lattice_pts: only {len(pts)} lattice point(s) inside the "
            f"hull of {len(hull_v)} vertices; need >= 3 to triangulate. "
            f"Returning None."
        )
        return None
    return pts

# random polygons
# ---------------
def random_lattice_polygon(
    rng:         np.random.Generator,
    target_Npts: int,
    *,
    Npts_min:    int | None = None,
    Npts_max:    int | None = None,
    max_coord:   int | None = None,
    verbose:     bool       = True,
) -> np.ndarray | None:
    """
    Sample one random convex lattice polygon aimed at `target_Npts` (but we
    allow any `Npts` within range `[Npts_min,Npts_max]`).

    Does so by
        1) picking 3 to 7 random integer vertices in `[0, max_coord]^2`,
        2) taking the convex hull of these vertices, and
        3) returning the lattice points in the hull if `Npts` is acceptable.
    If `max_coord` is not given, it's set via the heuristic
    `max(2, ceil(sqrt(target_Npts) * 1.3))`

    The points are translated so the min x and y are both 0. They are also
    lexicographically sorted. These are both fairly optional but useful for
    dedup.

    Parameters
    ----------
    rng : np.random.Generator
        Source of randomness.
    target_Npts : int
        Aimed-for `Npts`. Drives the default `max_coord`.

    Npts_min, Npts_max : int, optional
        Reject draws whose `len(pts)` is outside `[Npts_min, Npts_max]`.
    max_coord : int, optional
        Coord range for seed vertices: `[0, max_coord]`. Auto-derived from
        `target_Npts` if not given.
    verbose : bool, optional
        If False, suppress the `enum_lattice_pts` warnings emitted on
        degenerate draws (collinear / rank-deficient / too-few lattice
        points). Default True (warnings shown).

    Returns
    -------
    pts : ndarray or None
        `(Npts, 2)` int64 lattice points, or `None` if the draw was degenerate
        (collinear or < 3 unique vertices) or fell outside the requested range.
    """
    # defaults if no variables are given
    if max_coord is None:
        max_coord = max(2, int(np.ceil(np.sqrt(target_Npts) * 1.3)))

    # get the vertices, lattice points
    Nverts = int(rng.integers(3, 7+1))
    verts = rng.integers(0, max_coord + 1, size=(Nverts, 2))
    if verbose:
        pts = enum_lattice_pts(verts)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pts = enum_lattice_pts(verts)

    # reject for bad cases
    if pts is None:
        return None
    if Npts_min is not None and len(pts) < Npts_min:
        return None
    if Npts_max is not None and len(pts) > Npts_max:
        return None

    # canonicalize and return
    pts = pts - pts.min(axis=0)
    pts = pts[np.lexsort(pts.T[::-1])]
    return pts

# triangulation
# =============
def canonical_simps(simps: np.ndarray) -> np.ndarray:
    """Canonicalize a triangulation: sort each simp's indices, then lex-sort
    the rows. Call `.tobytes()` on the result for a dedup key."""
    s = np.sort(simps, axis=1)
    return s[np.lexsort(s.T[::-1])]

def _det3(m: list) -> int:
    """Exact integer determinant of a 3x3 matrix given as three rows."""
    a, b, c = m[0]
    d, e, f = m[1]
    g, h, i = m[2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def secondary_cone_ineqs_2d(pts: np.ndarray, simps: np.ndarray) -> np.ndarray:
    """
    Inward CPL inequality rows `H` of a 2D triangulation's secondary cone,
    i.e. the cone `{x : H x >= 0}` of height vectors `x` that induce this
    triangulation. The triangulation is regular iff this cone is
    full-dimensional (see `is_regular`).

    For each pair of triangles sharing an edge, the four involved points form
    a circuit whose inward normal is the 1D nullspace of their homogenized
    coordinate matrix; here that nullspace is computed exactly via integer 3x3
    minors (no floating point, no exact-CAS dependency).

    Extracted and de-classed from CYTools'
    `Triangulation._2d_frt_cone_ineqs` (originally written by Nate MacFadden);
    reimplemented numpy-only so the regularity check needs no
    secondary-fan/CAS library.

    Parameters
    ----------
    pts : ndarray
        `(Npts, 2)` int. Lattice points of the polygon.
    simps : ndarray
        `(Nsimps, 3)` int. Index triples into `pts` forming a triangulation
        (not checked).

    Returns
    -------
    H : ndarray
        `(Nineqs, Npts)` int. Each row is an inward-facing CPL hyperplane
        normal over the height coordinates.
    """
    pts = np.asarray(pts)
    simps = [list(map(int, s)) for s in simps]

    # for each point, the simplices containing it; then the points each pair
    # of simplices shares (a shared edge => exactly two shared points)
    pt_to_simps = defaultdict(list)
    for si, s in enumerate(simps):
        for v in s:
            pt_to_simps[v].append(si)
    pair_shared = defaultdict(set)
    for v, sis in pt_to_simps.items():
        for a, b in itertools.combinations(sis, 2):
            pair_shared[(a, b)].add(v)

    npts = len(pts)
    rows = []
    for (a, b), shared in pair_shared.items():
        if len(shared) < 2:                  # share an edge => a circuit
            continue
        s = list(shared)
        n_s = [v for v in simps[a] + simps[b] if v not in shared]
        order = [n_s[0], n_s[1], s[0], s[1]]
        cols = [[int(pts[k][0]), int(pts[k][1]), 1] for k in order]
        # 1D nullspace of the 3x4 [x; y; 1]: v_j = (-1)^j det(cols without j)
        v = []
        for j in range(4):
            sub = [cols[k] for k in range(4) if k != j]
            mat = [[sub[c][r] for c in range(3)] for r in range(3)]
            v.append(((-1) ** j) * _det3(mat))
        v = np.array(v, dtype=np.int64)
        if v[0] < 0:                         # sign: not-shared coefficient positive
            v = -v
        row = np.zeros(npts, dtype=np.int64)
        for k, vi in zip(order, v):
            row[k] += int(vi)
        rows.append(row)
    return (np.array(rows, dtype=np.int64) if rows
            else np.zeros((0, npts), dtype=np.int64))


def is_regular(pts: np.ndarray, simps: np.ndarray) -> bool:
    """
    Check whether a 2D triangulation is regular: i.e. whether its secondary
    cone has nonempty interior, equivalently whether some height vector `x`
    satisfies `H x >= 1` for the CPL inequalities `H` of
    `secondary_cone_ineqs_2d`. Decided by an LP feasibility check (HiGHS).

    Parameters
    ----------
    pts : ndarray
        `(Npts, 2)` int. Lattice points of the polygon.
    simps : ndarray
        `(Nsimps, 3)` int. Index triples into `pts` forming a triangulation
        (not checked).

    Returns
    -------
    regular : bool
        True iff the triangulation is regular.
    """
    H = secondary_cone_ineqs_2d(pts, simps)
    if H.shape[0] == 0:                      # no internal edges -> trivially regular
        return True

    import highspy                           # only the regularity path needs it

    nr, n = H.shape
    inf = highspy.kHighsInf
    h = highspy.Highs()
    h.silent()
    h.addVars(n, np.full(n, -inf), np.full(n, inf))
    starts = (np.arange(nr) * n).astype(np.int32)
    index = np.tile(np.arange(n, dtype=np.int32), nr)
    h.addRows(nr, np.ones(nr), np.full(nr, inf),
              nr * n, starts, index, H.ravel().astype(float))
    h.run()
    return h.getModelStatus() == highspy.HighsModelStatus.kOptimal
