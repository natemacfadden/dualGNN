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
import ctypes
import math
import signal
import warnings

import numba
import numpy as np
from regfans import VectorConfiguration
from scipy.spatial import ConvexHull


# ppl messes up rounding types... we have to fix them periodically :(
_libc = ctypes.CDLL(None)
_libc.fesetround(0)


# polytope
# ========
def compute_bdry(pts: np.ndarray) -> np.ndarray:
    """
    Primitive boundary edges of a 2D convex lattice polygon.

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

    # return
    if not out:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(out, dtype=np.int64)

# volumes/areas
# -------------
@numba.njit(cache=True)
def signed_area2(a, b, c):
    """
    Twice the signed area of the triangle `(a, b, c)` in 2D.

    Equivalent to `det([[ax, ay, 1], [bx, by, 1], [cx, cy, 1]])`.

    Positive iff `a, b, c` is counter-clockwise; zero iff collinear; magnitude
    is twice the absolute area (so unimodular triangles have
    `|signed_area2| == 1`).

    Parameters
    ----------
    a, b, c : ndarray
        Length-2 int arrays giving the 2D lattice coordinates of the three
        triangle points.

    Returns
    -------
    s : int
        Twice the signed area.
    """
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

# lattice point enumeration
# -------------------------
def enum_lattice_pts(verts: np.ndarray) -> np.ndarray | None:
    """
    Enumerate every lattice point in the convex hull of `verts`.

    Takes the convex hull of `verts` and then filters lattice points in the
    bounding box for those that lay lies inside the hull.

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
        return None
    try:
        hull = ConvexHull(v)
    except Exception:
        return None

    # ensure sufficiently high dimension (2D)
    hull_v = v[hull.vertices]
    if np.linalg.matrix_rank(hull_v - hull_v[0]) < 2:
        return None

    # filter the bounding box lattice points (in hull iff all CCW edges
    # see (x, y) on the non-negative side)
    x_lo, y_lo = v.min(axis=0)
    x_hi, y_hi = v.max(axis=0)
    N_hull = len(hull_v)
    pts = [
        (x, y)
        for x in range(int(x_lo), int(x_hi) + 1)
        for y in range(int(y_lo), int(y_hi) + 1)
        if all(signed_area2(hull_v[i], hull_v[(i + 1) % N_hull], (x, y)) >= 0
               for i in range(N_hull))
    ]

    # return
    if len(pts) < 3:
        return None
    return np.asarray(pts, dtype=np.int64)

# random polygons
# ---------------
def random_lattice_polygon(
    rng:         np.random.Generator,
    target_Npts: int,
    *,
    Npts_min:    int | None = None,
    Npts_max:    int | None = None,
    max_coord:   int | None = None,
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
    pts   = enum_lattice_pts(verts)

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
    """
    Canonical form of a triangulation.

    Sorts each simp's point indices, then lex-sorts the rows. Callers that
    need a dedup key call `.tobytes()` on the result.

    Parameters
    ----------
    simps : ndarray
        `(N_simps_per_ft, 3)` int. The simps of one FT (as indices).

    Returns
    -------
    canon : ndarray
        Same shape as `simps`, with columns sorted and rows lex-sorted.
    """
    s = np.sort(simps, axis=1)
    return s[np.lexsort(s.T[::-1])]

def is_regular(pts: np.ndarray, simps: np.ndarray) -> bool:
    """
    Check if a triangulation is regular via `regfans`. This requires
        1) homogenizing pts,
        2) building a Fan in regfans, and then
        3) checking the regularity of said Fan.
    A 60s SIGALRM timeout guards the call (regfans can hang on degenerate
    inputs); regfans warnings are promoted to errors. Both timeout and
    promoted-warning are reported and return False; any other exception
    propagates.

    Parameters
    ----------
    pts : ndarray
        `(Npts, 2)` int. Lattice points of the polygon.
    simps : ndarray
        `(Nsimps, 3)` int. Each row is a triple of indices into `pts`. Together
        they should cover the polygon (i.e., form a triangulation); the function
        does not check that.

    Returns
    -------
    regular : bool
        True iff the triangulation is regular. If regfans hangs >60s or emits a
        warning, this is set to False and logged.
    """
    if len(simps) == 1:
        return True

    # homogenize
    ones = np.ones((pts.shape[0], 1), dtype=pts.dtype)
    vecs = np.hstack([pts, ones])

    # make the fan
    vc   = VectorConfiguration(vecs, labels=list(range(len(pts))))
    fan  = vc.triangulate(cells=simps)

    # check regularity
    def raise_timeout(signum, frame):
        raise TimeoutError("is_regular hung")

    old = signal.signal(signal.SIGALRM, raise_timeout)
    signal.alarm(60)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", module="regfans")
            return fan.is_regular()
    except (TimeoutError, Warning) as e:
        print(f"[is_regular] {type(e).__name__}: {e} "
              f"(Npts={len(pts)}, Nsimps={len(simps)})", flush=True)
        return False
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)

        # fix ppl's rounding bug
        _libc.fesetround(0)
