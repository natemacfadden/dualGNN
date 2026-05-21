# grow2d.pyx
# Cython wrapper for grow2d.h

from libc.stdint cimport uint32_t, uint64_t
from libc.stdlib cimport malloc, free
import numpy as np
import time

# declare the external C function
# --------------------------------
cdef extern from "grow2d.h":
    int c_grow2d "grow2d"(
        int      *pts,
        int       num_pts,
        int      *bdry,
        int       num_bdry,
        uint64_t *seed,
        int       max_num_simps,
        uint32_t *simps,
        int      *num_simps
    )


# Python-exposed wrapper
# ----------------------
def grow2d(pts,
           bdry=None,
           seed=None,
           int max_num_simps=-1) -> tuple:
    """
    Grow a fine triangulation of a 2D lattice polygon (C backend).

    Parameters
    ----------
    pts : array-like of shape (n, 2), int
        Input lattice points.
    bdry : set of frozenset({i, j}), or (m, 2) int array, optional
        Boundary edges. Computed from pts via ConvexHull if not provided.
    seed : int, optional
        RNG seed. Defaults to a time-based value.
    max_num_simps : int, optional
        Maximum number of simplices to allocate. Defaults to 3 * n.

    Returns
    -------
    simps : ndarray of shape (num_simps, 3), dtype uint32
        Simplices of the fine triangulation, each row (i, j, k) with i < j < k.
    status : int
        Status code:
             0: success
            -1: 0 points input
            -2: memory allocation problem
            -3: could not find an initial unit-area triangle
            -4: failed to grow (no compatible vertex found; partial result returned)
            -5: exceeded max_num_simps
    """
    # normalise pts
    pts_np = np.ascontiguousarray(pts, dtype=np.int32)
    if pts_np.ndim != 2 or pts_np.shape[1] != 2:
        raise ValueError(f"pts must be shape (n, 2), got {pts_np.shape}")

    cdef int num_pts = pts_np.shape[0]

    # normalise bdry
    if bdry is None:
        from dualgnn.geometry import compute_bdry
        bdry = compute_bdry(pts_np)

    # accept either a set of frozensets or an array
    if not isinstance(bdry, np.ndarray):
        bdry_list = sorted([sorted(e) for e in bdry])
        bdry_np = np.array(bdry_list, dtype=np.int32) if bdry_list \
                  else np.empty((0, 2), dtype=np.int32)
    else:
        bdry_np = np.ascontiguousarray(bdry, dtype=np.int32)

    cdef int num_bdry = bdry_np.shape[0]

    # seed
    if seed is None:
        seed = time.time_ns() % (2**64)
    cdef uint64_t c_seed = <uint64_t>seed

    # default allocation
    if max_num_simps < 0:
        max_num_simps = 3 * num_pts

    # allocate output buffer
    cdef uint32_t *simps_buf = <uint32_t *>malloc(
        max_num_simps * 3 * sizeof(uint32_t))
    if simps_buf == NULL:
        raise MemoryError("Failed to allocate simps buffer")

    # set up pointers
    cdef int[:, ::1] pts_view = pts_np
    cdef int *pts_ptr = &pts_view[0, 0]

    cdef int[:, ::1] bdry_view
    cdef int *bdry_ptr
    if num_bdry > 0:
        bdry_view = bdry_np
        bdry_ptr  = &bdry_view[0, 0]
    else:
        bdry_ptr  = NULL

    # call C
    cdef int num_simps = 0
    cdef int status = c_grow2d(
        pts_ptr,
        num_pts,
        bdry_ptr,
        num_bdry,
        &c_seed,
        max_num_simps,
        simps_buf,
        &num_simps
    )

    # copy to numpy
    out = np.empty((num_simps, 3), dtype=np.uint32)
    cdef uint32_t[:, ::1] out_view = out
    cdef int i, j
    for i in range(num_simps):
        for j in range(3):
            out_view[i, j] = simps_buf[3*i + j]

    free(simps_buf)
    return out, status
