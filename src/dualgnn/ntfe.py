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
from dataclasses import dataclass, field
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
    max_tries:      int | None = None,
    ctx:            NTFEContext | None = None,
    return_ctx:     bool       = False,
    verbose:        bool       = True,
) -> np.ndarray | list | tuple:
    """
    Sample NTFEs of `poly` using dualGNN and https://arxiv.org/abs/2309.10855.

    Regularity: 2-face draws are FRTs (regular by construction). The 4D
    extension is regular iff the stacked secondary-cone inequalities of
    the chosen 2-face FRTs admit a strict interior point -- this is the
    per-attempt check (run incrementally; see `_sample`).

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
    max_tries : int, optional
        Raise `RuntimeError` after this many extension attempts (per
        worker, in the parallel path). Default `None` (no cap).
    ctx : NTFEContext, optional
        Reuse the setup from an earlier `return_ctx=True` call instead
        of sampling fresh 2-face FRT pools. When given, `net`,
        `N_face_triangs`, `batch_size`, and `beta` are ignored (they
        only shape pool building), and `poly` must be the
        polytope the context was built from. Note the statistics: calls
        sharing a `ctx` draw from the SAME finite pools, so their NTFEs
        are not independent across calls in the way fresh-pool runs are.
        Default `None` (build fresh).
    return_ctx : bool, optional
        If True, also return the built (or passed) `NTFEContext`, for
        reuse via `ctx` in later calls. Default `False`.
    verbose : bool, optional
        Print progress. Default `True`.

    Returns
    -------
    out : ndarray or list of Triangulation
        `(N, npts)` float64 heights, or a length-`N` list of FRSTs if
        `as_triangs=True`. With `return_ctx=True`, the tuple
        `(out, ctx)` instead.
    """
    if ctx is None:
        ctx = build_ntfe_context(
            poly, net,
            N_face_triangs = N_face_triangs,
            batch_size     = batch_size,
            beta           = beta,
            seed           = seed,
            verbose        = verbose,
        )
    elif ctx.poly is not poly:
        raise ValueError(
            "sample_ntfes: `ctx` was built from a different Polytope "
            "object than `poly`; reuse a context only with the polytope "
            "it was built from."
        )

    if n_workers < 0:
        n_workers = min(os.cpu_count() or 1, N)

    # serial
    if n_workers <= 1:
        out = _sample(ctx,
                      N=N, as_triangs=as_triangs,
                      seed=seed, max_tries=max_tries, verbose=verbose)
        return (out, ctx) if return_ctx else out

    # parallel
    # spawn workers and split sampling across them. seed=None must stay
    # nondeterministic (as in the serial path), so draw a random base seed
    per_worker = (N + n_workers - 1) // n_workers
    if seed is None:
        base_seed = int(np.random.default_rng().integers(2**31 - 1))
    else:
        base_seed = seed
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
                      seed=base_seed + 1 + a, max_tries=max_tries,
                      verbose=False)
            for a in range(n_workers)
        ]
        chunks = [f.result() for f in futures]
    if verbose:
        dt = time.perf_counter() - t0
        print(f"[ntfe] sampling done in {dt:.0f}s "
              f"({N / max(dt, 1e-9):.2f}/s aggregate)", flush=True)

    if as_triangs:
        out: list = []
        for c in chunks:
            out.extend(c)
        out = out[:N]
    else:
        out = np.concatenate(chunks, axis=0)[:N]
    return (out, ctx) if return_ctx else out


# context + helpers
# =================
@dataclass
class NTFEContext:
    """
    The reusable setup for NTFE sampling: one FRT pool per distinct
    2-face geometry, plus the per-face lookups for assembling
    candidates. Identical across `sample_ntfes` calls on the same
    polytope -- build once (`build_ntfe_context`, or the `return_ctx`
    flag) and pass to later calls via `ctx`. Pickleable (shared with
    worker processes).
    """
    poly:          Polytope
    npts:          int                 # number of lattice points of `poly`
    # one entry per distinct 2-face geometry
    pools:         list[np.ndarray]    # FRTs; each (N_face_triangs, n_simps, 3)
    # one entry per 2-face of `poly` (all three lists parallel)
    shape_keys:    list[int]           # which pool to draw from
    src_to_labels: list[np.ndarray]    # FRT vertex idx -> parent label
    face_polys:    list[Polytope]      # 2-face as Polytope; call .triangulate
    # per-face memo of derived inequality rows (filled lazily by
    # _FaceIneqs; persists across calls so repeated sampling
    # does not re-derive them)
    ineq_rows:     dict = field(default_factory=dict)


def build_ntfe_context(
    poly:           Polytope,
    net:            DualGNN,
    *,
    N_face_triangs: int        = 10_000,
    batch_size:     int        = 256,
    beta:           float      = 1.0,
    seed:           int | None = None,
    verbose:        bool       = True,
) -> NTFEContext:
    """
    Build the reusable setup for `sample_ntfes`: group `poly`'s 2-faces
    by GL(2,Z)+translation geometry and sample one FRT pool per group
    with the dualGNN sampler. The result is identical across
    `sample_ntfes` calls on the same polytope, so build it once and
    pass it to repeated calls via `ctx` rather than resampling.

    Parameters are as in `sample_ntfes`.
    """
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

    return NTFEContext(
        poly          = poly,
        npts          = len(poly.labels),
        pools         = pools,
        shape_keys    = shape_keys,
        src_to_labels = src_to_labels,
        face_polys    = face_polys,
    )

def _sample(
    ctx:        NTFEContext,
    *,
    N:          int,
    as_triangs: bool       = False,
    seed:       int | None = None,
    max_tries:  int | None = None,
    verbose:    bool       = True,
):
    """
    Sample candidate 2-face FRT combinations until `N` extend to NTFEs.

    Rejection sampling: draw one FRT per 2-face independently, accept
    iff the stacked secondary-cone inequalities admit a strict interior
    point. The check runs face-by-face on one persistent warm LP,
    aborting an attempt at its first infeasible prefix -- valid because
    infeasibility is monotone in prefixes, so the accept/reject
    decision (and hence the sampled distribution) is identical to
    checking the full stack, while failed attempts only pay a few warm
    solves down to their first failure. Inequality rows per (face, pool
    entry) are derived once and memoized on `ctx`.
    """
    rng     = np.random.default_rng(seed)
    order   = _adjacency_order(ctx)
    ineqs   = [_FaceIneqs(ctx, j) for j in order]
    lp      = _IncrementalLP(ctx.npts)
    heights = np.empty((N, ctx.npts), dtype=np.float64)
    ntfes: list = [] if as_triangs else None

    n_ok       = 0
    n_try      = 0
    t_start    = time.perf_counter()
    t_last_log = t_start
    while n_ok < N:
        if max_tries is not None and n_try >= max_tries:
            raise RuntimeError(
                f"sample_ntfes: only {n_ok}/{N} extended after "
                f"{n_try} attempts (max_tries={max_tries})")
        n_try += 1
        ok, pushed = True, 0
        for q in ineqs:
            if lp.push(q[rng.integers(len(q))]):  # feasible: row kept
                pushed += 1
            else:
                ok = False  # infeasible: the push self-popped
                break
        if ok:
            h_arr         = lp.witness()
            heights[n_ok] = h_arr
            if as_triangs:
                ntfes.append(ctx.poly.triangulate(
                    heights=h_arr, make_star=True))
            n_ok += 1
        for _ in range(pushed):
            lp.pop()

        # log on milestones and every >5s of wall time, including dry
        # stretches with no accepted extension yet
        if verbose:
            now       = time.perf_counter()
            milestone = ok and n_ok in (1, 10, 100, 1000)
            if now - t_last_log > 5 or milestone:
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
    lookup so the sampling loop only does array indexing.

    Returns
    -------
    shapes : list[np.ndarray]
        (n_pts, 2) int64 arrays, one per distinct geometry.
    shape_keys : list[int]
        Per 2-face, the index into `shapes`.
    src_to_labels : list[np.ndarray]
        Per 2-face, an (n_pts,) lookup; the i-th entry is the parent-polytope
        label of the i-th point of this face's shape rep.
    face_polys : list[Polytope]
        Per 2-face, the 2-face as a `cytools.Polytope`.
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


# incremental feasibility
# =======================
class _FaceIneqs:
    """
    One 2-face's pool, as inequality rows: entry `k` is the secondary cone of
    the pool's k-th FRT, nonzero over only the points in this face. Rows are
    derived on first touch and memoized on `ctx.ineq_rows`, so repeated sampling
    never re-derives them.
    """

    def __init__(self, ctx: NTFEContext, face_idx: int) -> None:
        self.pool      = ctx.pools[ctx.shape_keys[face_idx]]
        self.s2l       = ctx.src_to_labels[face_idx]
        self.face_poly = ctx.face_polys[face_idx]
        self.npts      = ctx.npts
        self._memo: dict[int, np.ndarray] = \
            ctx.ineq_rows.setdefault(face_idx, {})

    def __len__(self) -> int:
        return len(self.pool)

    def __getitem__(self, k: int) -> np.ndarray:
        k = int(k)
        if k not in self._memo:
            # lazy import: cytools is not needed for the 2D path
            from cytools.ntfe.ntfe import _2d_frt_cone_ineqs
            triang = self.face_poly.triangulate(
                simplices=self.s2l[self.pool[k]].tolist(),
                include_points_interior_to_facets=True,
            )
            rows = _2d_frt_cone_ineqs(triang, self.npts)
            self._memo[k] = np.asarray(rows.dense(), dtype=np.float64)
        return self._memo[k]


def _adjacency_order(ctx: NTFEContext) -> list[int]:
    """
    Order the 2-faces so interacting ones are checked early: walk the facets so
    consecutive facets share a 2-face, emitting each facet's not-yet-seen
    2-faces in turn. Faces conflict only through shared points, so this
    front-loading roughly halves the depth at which an attempt's first
    infeasibility surfaces, and with it the number of solves a rejected attempt
    costs.
    """
    pts = [set(int(v) for v in s2l) for s2l in ctx.src_to_labels]
    facet_labels = [set(int(v) for v in f.labels)
                    for f in ctx.poly.facets()]
    facet_faces: list[list[int]] = [[] for _ in facet_labels]
    for j, face_pts in enumerate(pts):
        for i, fl in enumerate(facet_labels):
            if face_pts <= fl:
                facet_faces[i].append(j)

    # greedy facet walk: hop to an unvisited facet sharing a 2-face with
    # the current one; fall back to any unvisited facet
    n_facets   = len(facet_labels)
    visited    = [False] * n_facets
    cur        = 0
    facet_walk = [cur]
    visited[cur] = True
    while len(facet_walk) < n_facets:
        shared = [i for i in range(n_facets) if not visited[i]
                  and set(facet_faces[i]) & set(facet_faces[cur])]
        cur = shared[0] if shared else visited.index(False)
        visited[cur] = True
        facet_walk.append(cur)

    order: list[int] = []
    seen:  set[int]  = set()
    for i in facet_walk:
        for j in facet_faces[i]:
            if j not in seen:
                seen.add(j)
                order.append(j)
    assert len(order) == len(pts)
    return order


# incremental feasibility
# =======================
class _IncrementalLP:
    """
    One persistent LP model: strict feasibility of the stacked prefix `H x >= 1`
    over the height coordinates, with rows added face-by-face (`push`) and
    removed when an attempt ends (`pop`) so dual simplex re-solves from the
    previous basis instead of from scratch (~25x cheaper per solve than cold
    full-stack solves, measured). Free variables, zero objective.
    """

    def __init__(self, npts: int) -> None:
        # lazy: only the NTFE path needs highspy
        try:
            import highspy
        except ImportError as e:
            raise ImportError(
                "sample_ntfes needs the `highspy` LP solver: "
                "pip install highspy"
            ) from e
        self._highspy = highspy
        self.h = highspy.Highs()
        self.h.silent()
        inf = highspy.kHighsInf
        self.h.addVars(npts, np.full(npts, -inf), np.full(npts, inf))
        self.depth_rows: list[int] = []

    def push(self, rows: np.ndarray) -> bool:
        """Add one face's rows; return strict feasibility of the stack.
        Infeasible pushes are rolled back internally."""
        n = len(rows)
        if n:
            ncols  = rows.shape[1]
            starts = np.arange(n, dtype=np.int32) * ncols
            index  = np.tile(np.arange(ncols, dtype=np.int32), n)
            self.h.addRows(n, np.ones(n),
                           np.full(n, self._highspy.kHighsInf),
                           n * ncols, starts, index, rows.ravel())
            self.h.run()
            ok = (self.h.getModelStatus()
                  == self._highspy.HighsModelStatus.kOptimal)
        else:                              # elementary face: no rows
            ok = True
        self.depth_rows.append(n)
        if not ok:
            self.pop()
        return ok

    def pop(self) -> None:
        """Remove the most recent level's rows (backtrack)."""
        n = self.depth_rows.pop()
        if n:
            total = self.h.getNumRow()
            self.h.deleteRows(
                n, np.arange(total - n, total, dtype=np.int32))

    def witness(self) -> np.ndarray:
        """The current solve's interior point (the NTFE heights)."""
        sol = self.h.getSolution()
        # invariant: the heights strictly satisfy every stacked inequality
        # Hx>=1 (row_value is the activity Hx) - guards an LP/solver regression
        # from handing back a spurious witness
        rv = sol.row_value
        assert len(rv) == 0 or min(rv) >= 1.0 - 1e-6, \
            f"witness violates Hx>=1 (min activity {min(rv):.3g})"
        return np.array(sol.col_value)
