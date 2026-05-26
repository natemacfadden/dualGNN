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
# Description:  Sample NTFEs of a reflexive polytope dualGNN
# -----------------------------------------------------------------------------

# external imports
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import multiprocessing as mp
import os
import time
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import ConvexHull

if TYPE_CHECKING:
    from cytools import Polytope

# local imports
from .dualgraph import DualGraph
from .model     import DualGNN
from .sampler   import sample


# main entry point
# ================
def sample_ntfes(
    poly:           Polytope,
    net:            DualGNN,
    N:              int,
    *,
    as_triangs:     bool       = False,
    N_face_triangs: int        = 10_000,
    batch_size:     int        = 256,
    beta:           float      = 1.0,
    seed:           int | None = None,
    n_workers:      int        = 1,
    verbose:        bool       = True,
):
    """
    Sample NTFEs of `poly` using dualGNN and https://arxiv.org/abs/2309.10855.

    Regularity: 2-face draws are FRTs (regular by construction). The 4D
    extension is regular iff `cone_of_permissible_heights` has an interior
    point -- this is the per-attempt check.

    Parameters
    ----------
    poly : cytools.Polytope
        Reflexive 4D polytope.
    net : DualGNN
        Trained 2D dualGNN sampler.
    N : int
        Number of NTFEs to return.

    as_triangs : bool, optional
        If True, return FRSTs (cytools `Triangulation` objects via
        `poly.triangulate(heights=h, make_star=True)`) instead of heights.
        Default `False`.
    N_face_triangs : int, optional
        Number of FRTs to draw per distinct 2-face geometry; sampled with
        replacement when constructing NTFEs. Default `10_000`.
    batch_size : int, optional
        dualGNN batch size during 2-face FRT pool construction. Default `256`.
    beta : float, optional
        dualGNN inverse temperature. Default `1.0` (uniform-over-FRT).
    seed : int, optional
        Seeds the dualGNN pool draws and per-attempt index draws.
    n_workers : int, optional
        Worker processes. Default `1` (serial). Pass `-1` to use as many
        as `os.cpu_count()`, capped at `N`.
    verbose : bool, optional
        Print progress. Default `True`.

    Returns
    -------
    out : ndarray or list of Triangulation
        `(N, npts)` float64 heights, or a length-`N` list of FRSTs if
        `as_triangs=True`.
    """
    ctx = _build_ctx(
        poly, net,
        N_face_triangs = N_face_triangs,
        batch_size     = batch_size,
        beta           = beta,
        seed           = seed,
        verbose        = verbose,
    )

    if n_workers < 0:
        n_workers = min(os.cpu_count() or 1, N)

    # serial
    if n_workers <= 1:
        return _sample(ctx,
                       N=N, as_triangs=as_triangs,
                       seed=seed, verbose=verbose)

    # parallel
    # spawn workers and split sampling across them
    per_worker = (N + n_workers - 1) // n_workers
    base_seed  = 0 if seed is None else seed
    if verbose:
        print(f"[ntfe] spawning {n_workers} parallel workers x {per_worker} "
              f"(target {N}, as_triangs={as_triangs})", flush=True)

    t0 = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers = n_workers,
        mp_context  = mp.get_context("spawn"),
    ) as ex:
        futures = [
            ex.submit(_sample, ctx,
                      N=per_worker, as_triangs=as_triangs,
                      seed=base_seed + 1 + a, verbose=False)
            for a in range(n_workers)
        ]
        chunks = [f.result() for f in futures]
    if verbose:
        dt = time.perf_counter() - t0
        print(f"[ntfe] sampling done in {dt:.0f}s "
              f"({N / max(dt, 1e-9):.2f}/s aggregate)", flush=True)

    # return
    if as_triangs:
        merged: list = []
        for c in chunks:
            merged.extend(c)
        return merged[:N]
    return np.concatenate(chunks, axis=0)[:N]


# context + helpers
# =================
@dataclass
class _NTFEContext:
    """Pickleable setup work, shared with worker processes."""
    poly:          Polytope
    npts:          int                 # number of lattice points of `poly`
    # one entry per distinct 2-face geometry
    pools:         list[np.ndarray]    # FRTs; each (N_face_triangs, n_simps, 3)
    # one entry per 2-face of `poly` (all three lists parallel)
    shape_keys:    list[int]           # which pool to draw from
    src_to_labels: list[np.ndarray]    # FRT vertex idx -> parent label
    face_polys:    list[Polytope]      # 2-face as Polytope; call .triangulate


def _build_ctx(
    poly:           Polytope,
    net:            DualGNN,
    *,
    N_face_triangs: int        = 10_000,
    batch_size:     int        = 256,
    beta:           float      = 1.0,
    seed:           int | None = None,
    verbose:        bool       = True,
) -> _NTFEContext:
    """Group 2-faces by GL(2,Z)+t geometry, sample one FRT pool per group."""
    shapes, shape_keys, src_to_labels, face_polys = _decompose_2faces(poly)
    if verbose:
        sizes = [s.shape[0] for s in shapes]
        print(f"[ntfe] {len(face_polys)} 2-faces -> {len(shapes)} distinct "
              f"shapes (n_pts per shape: {sizes})", flush=True)

    pools = []
    for k, rep in enumerate(shapes):
        dg = DualGraph(rep)
        if verbose:
            print(f"[ntfe] shape {k}: n_pts={rep.shape[0]}, "
                  f"candidate simps={dg.simps.shape[0]}, "
                  f"simps/FT={dg.N_simps_per_ft}, "
                  f"sampling {N_face_triangs} FRTs...", flush=True)
        t0   = time.perf_counter()
        pool = sample(
            net, dg,
            Ntriangs   = N_face_triangs,
            batch_size = batch_size,
            beta       = beta,
            seed       = (None if seed is None else seed + k),
            verbose    = False,
        )
        if verbose:
            print(f"[ntfe] shape {k}: pool ready "
                  f"({time.perf_counter() - t0:.1f}s)", flush=True)
        pools.append(pool)

    return _NTFEContext(
        poly          = poly,
        npts          = len(poly.labels),
        pools         = pools,
        shape_keys    = shape_keys,
        src_to_labels = src_to_labels,
        face_polys    = face_polys,
    )

def _sample(
    ctx:        _NTFEContext,
    *,
    N:          int,
    as_triangs: bool       = False,
    seed:       int | None = None,
    verbose:    bool       = True,
):
    """Sample candidate 2-face FRTs until `N` extend to NTFEs."""
    # lazy import: cytools.ntfe isn't needed for the 2D path
    from cytools.ntfe.ntfe import cone_of_permissible_heights

    rng     = np.random.default_rng(seed)
    heights = np.empty((N, ctx.npts), dtype=np.float64)
    ntfes: list = [] if as_triangs else None

    n_ok       = 0
    n_try      = 0
    t_start    = time.perf_counter()
    t_last_log = t_start
    while n_ok < N:
        n_try  += 1
        triangs = []
        for shape_key, src_to_label, face_poly in zip(
            ctx.shape_keys, ctx.src_to_labels, ctx.face_polys,
        ):
            pool      = ctx.pools[shape_key]
            simp_idxs = pool[rng.integers(pool.shape[0])]
            triangs.append(face_poly.triangulate(
                simplices = src_to_label[simp_idxs].tolist()))
        # cone of heights making the 2-face FRTs extend to a regular NTFE
        cone = cone_of_permissible_heights(
            triangs, npts=ctx.npts, poly=ctx.poly)
        h    = cone.find_interior_point()
        if h is None:
            continue
        h_arr         = np.asarray(h, dtype=np.float64)
        heights[n_ok] = h_arr
        if as_triangs:
            ntfes.append(ctx.poly.triangulate(
                heights=h_arr, make_star=True))
        n_ok += 1

        if verbose:
            now = time.perf_counter()
            if now - t_last_log > 5 or n_ok in (1, 10, 100, 1000):
                t_last_log = now
                dt     = now - t_start
                rate   = n_ok / max(dt, 1e-9)
                accept = n_ok / n_try
                eta    = (N - n_ok) / max(rate, 1e-9)
                print(f"  n_ok={n_ok:>6}/{N}  n_try={n_try:>7}  "
                      f"accept={accept:.3%}  rate={rate:.2f}/s  "
                      f"elapsed={dt:.0f}s  eta={eta/60:.1f}m", flush=True)

    return ntfes if as_triangs else heights


# 2-face decomposition
# ====================
def _decompose_2faces(poly):
    """
    For each 2-face of `poly`: identify its 2D-geometry class (so same-shape
    faces share an FRT pool) and precompute the FRT-vertex -> parent-label
    lookup so the sampling loop only does array indexing. Returns four parallel
    lists:

      shapes        : (n_pts, 2) int64 arrays, one per distinct geometry
      shape_keys    : per 2-face, the index into `shapes`
      src_to_labels : per 2-face, (n_pts,) lookup. The i-th entry is the
                      parent-polytope label of the i-th point of this
                      face's shape rep.
      face_polys    : per 2-face, the 2-face as a `cytools.Polytope`
    """
    shapes:        list[np.ndarray] = []
    shape_keys:    list[int]        = []
    src_to_labels: list[np.ndarray] = []
    face_polys:    list             = []

    for f in poly.faces(2):
        # this face's lattice points in an optimal 2D basis
        face_pts = np.asarray(f.points(optimal=True), dtype=np.int64)

        # find a GL(2,Z)+t iso from an existing shape rep to this face's
        # coords; on miss, this face becomes the rep for a new class (so
        # the iso is the identity)
        shape_key = -1
        A, t      = None, None
        for k, rep in enumerate(shapes):
            if rep.shape[0] != face_pts.shape[0]:
                continue
            A, t = _find_gl2z_iso(rep, face_pts)
            if A is not None:
                shape_key = k
                break
        if shape_key < 0:
            shapes.append(face_pts)
            shape_key = len(shapes) - 1
            A = np.eye(2, dtype=np.int64)
            t = np.zeros(2, dtype=np.int64)

        # precompute src_to_label: shape-rep index i -> parent label, via
        #   rep[i] -> A @ rep[i] + t (face coord) -> coord_to_idx -> labels
        rep          = shapes[shape_key]
        labels       = f.labels
        coord_to_idx = {tuple(p.tolist()): i for i, p in enumerate(face_pts)}
        src_to_label = np.empty(rep.shape[0], dtype=np.int64)
        for i in range(rep.shape[0]):
            face_coord      = tuple((A @ rep[i] + t).tolist())
            src_to_label[i] = labels[coord_to_idx[face_coord]]

        shape_keys.append(shape_key)
        src_to_labels.append(src_to_label)
        face_polys.append(f.as_poly())

    return shapes, shape_keys, src_to_labels, face_polys


def _find_gl2z_iso(
    P_src: np.ndarray,
    P_dst: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """
    Find unimodular `A` (2x2 int, |det| = 1) and translation `t` so that
    `{A @ p + t for p in P_src} == set(P_dst)`, or return `(None, None)`
    if no such (A, t) exists.

    Algorithm. A GL(2,Z)+t iso maps the convex hull of P_src to that of
    P_dst, so we enumerate candidates by aligning the two CCW hull-vertex
    sequences: k cyclic rotations x 2 orientations = 2k candidates, where
    k = #hull vertices. For each candidate, A is determined by two edge
    correspondences at the anchor vertex; we check it is integer-unimodular
    and verify it maps all of P_src onto P_dst.
    """
    P_src = np.asarray(P_src, dtype=np.int64)
    P_dst = np.asarray(P_dst, dtype=np.int64)
    if len(P_src) != len(P_dst):
        return None, None

    # CCW hull-vertex sequences (scipy returns them in CCW order for 2D)
    H_src = P_src[ConvexHull(P_src).vertices]
    H_dst = P_dst[ConvexHull(P_dst).vertices]
    if len(H_src) != len(H_dst):
        return None, None
    k = len(H_src)

    # Anchor on H_dst[0]. The two edges leaving it (to its CCW and CW
    # neighbors) form a basis of Z^2; every candidate iso must map a
    # corresponding pair of edges of P_src to these.
    w0        = H_dst[0]
    edges_dst = np.column_stack(
        [H_dst[1] - w0, H_dst[-1] - w0]).astype(np.int64)

    P_dst_set = frozenset(map(tuple, P_dst.tolist()))

    for reverse in (False, True):              # orientation
        H = H_src[::-1] if reverse else H_src
        for r in range(k):                     # cyclic rotation
            # candidate: v0 -> w0, with CCW and CW neighbors of v0
            # mapping to the CCW and CW neighbors of w0
            v0        = H[r]
            edges_src = np.column_stack(
                [H[(r + 1) % k] - v0, H[(r - 1) % k] - v0]
            ).astype(np.int64)

            # solve A @ edges_src == edges_dst via the adjugate, staying
            # in integers: edges_src @ adj(edges_src) == det * I, hence
            # det * A == edges_dst @ adj(edges_src). A is an integer
            # matrix iff det divides every entry of the numerator.
            det = int(edges_src[0,0] * edges_src[1,1]
                      - edges_src[0,1] * edges_src[1,0])
            if det == 0:
                continue
            adj = np.array(
                [[ edges_src[1,1], -edges_src[0,1]],
                 [-edges_src[1,0],  edges_src[0,0]]], dtype=np.int64)
            num = edges_dst @ adj
            if (num % det != 0).any():
                continue
            A = (num // det).astype(np.int64)

            # unimodular check (|det A| == 1)
            if abs(int(A[0,0] * A[1,1] - A[0,1] * A[1,0])) != 1:
                continue

            # full point-set check
            t      = w0 - A @ v0
            mapped = (P_src @ A.T) + t
            if frozenset(map(tuple, mapped.tolist())) == P_dst_set:
                return A, t
    return None, None
