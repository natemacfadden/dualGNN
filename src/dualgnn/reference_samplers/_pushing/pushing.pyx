# pushing.pyx
# Cython wrapper for pushing.h

from libc.stdint cimport uint32_t, uint64_t
from libc.stdlib cimport malloc, free
import numpy as np
import time


# declare the external C function
# --------------------------------
cdef extern from "pushing.h":
    ctypedef struct PushingOpts:
        int      random
        int      fine
        uint64_t seed

    int c_pushing "pushing"(
        int         *vecs,
        int          dim,
        int          num_vecs,
        PushingOpts *opts,
        int          max_num_simps,
        uint32_t    *simps,
        int         *num_simps
    )


# Python-exposed wrapper
# ----------------------
def pushing(pts, seed=None, int max_num_simps=-1) -> tuple:
    """
    Generate a random fine pushing triangulation of a 2D lattice polygon.

    Parameters
    ----------
    pts : array-like of shape (n, 2), int
        Input lattice points.
    seed : int, optional
        RNG seed. Defaults to a time-based value (as in `grow2d`).
    max_num_simps : int, optional
        Maximum number of simplices to allocate. Defaults to 3 * n.

    Returns
    -------
    simps : ndarray of shape (num_simps, 3), dtype uint32
        Simplices of the triangulation, each row (i, j, k) of vertex indices
        into pts.
    status : int
        Status code:
             0: success
            -1: misconfigured options (fine requires random)
            -2: memory allocation problem
            -3: 0 vectors input
            -4: couldn't find initial simplex
            -5: deadlock - couldn't add a new simplex
            -6: constructed too many simplices
          -100: error splitting a cone
    """
    # normalise pts and homogenize (append a column of 1s)
    pts_np = np.ascontiguousarray(pts, dtype=np.int32)
    if pts_np.ndim != 2 or pts_np.shape[1] != 2:
        raise ValueError(f"pts must be shape (n, 2), got {pts_np.shape}")
    pts_hom = np.ascontiguousarray(
        np.hstack([pts_np, np.ones((pts_np.shape[0], 1), dtype=np.int32)]),
        dtype=np.int32,
    )

    cdef int num_vecs = pts_hom.shape[0]
    cdef int dim      = pts_hom.shape[1]  # 3 (homogenized 2D)

    # default allocation
    if max_num_simps < 0:
        max_num_simps = 3 * num_vecs

    # seed
    if seed is None:
        seed = time.time_ns() % (2**64)

    # options
    cdef PushingOpts opts
    opts.random = 1
    opts.fine   = 1
    opts.seed   = <uint64_t>seed

    # allocate output buffer
    cdef uint32_t *simps_buf = <uint32_t *>malloc(
        max_num_simps * dim * sizeof(uint32_t))
    if simps_buf == NULL:
        raise MemoryError("Failed to allocate simps buffer")

    # set up pointers
    cdef int[:, ::1] pts_view = pts_hom
    cdef int *pts_ptr = &pts_view[0, 0]

    # call C
    cdef int num_simps = 0
    cdef int status = c_pushing(
        pts_ptr, dim, num_vecs,
        &opts,
        max_num_simps,
        simps_buf,
        &num_simps
    )

    # copy to numpy
    out = np.empty((num_simps, dim), dtype=np.uint32)
    cdef uint32_t[:, ::1] out_view = out
    cdef int i, j
    for i in range(num_simps):
        for j in range(dim):
            out_view[i, j] = simps_buf[dim * i + j]

    free(simps_buf)
    return out, status
