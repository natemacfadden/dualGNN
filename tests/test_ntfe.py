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
# Description:  End-to-end test of the 4D NTFE pipeline (sample_ntfes).
#               Needs CYTools (pip-installable; see the [test]/[train]
#               extras). Always runs -- a missing CYTools is a real failure
#               here, not a skip.
# -----------------------------------------------------------------------------

# external imports
import numpy as np
import pytest

import cytools

# local imports
from dualgnn.model import DualGNN
from dualgnn.ntfe  import build_ntfe_context, sample_ntfes


def test_sample_ntfes_end_to_end():
    """`sample_ntfes` on a small reflexive 4D polytope: heights come back
    with the documented shape/dtype and define fine star triangulations
    when round-tripped through CYTools."""
    # the quintic simplex: small (6 lattice points), so the 2-face pools
    # and the extendability check run in seconds
    verts = [[ 1,  0,  0,  0], [ 0,  1,  0,  0], [ 0,  0,  1,  0],
             [ 0,  0,  0,  1], [-1, -1, -1, -1]]
    poly = cytools.Polytope(np.array(verts, dtype=np.int64))
    net  = DualGNN.default(device="cpu")

    N       = 2
    heights = sample_ntfes(poly, net, N=N, N_face_triangs=10,
                           seed=0, max_tries=1000, verbose=False)

    npts = len(poly.labels)
    assert heights.shape == (N, npts)
    assert heights.dtype == np.float64
    assert np.isfinite(heights).all()

    # the heights must define fine star triangulations of the polytope
    for h in heights:
        t = poly.triangulate(heights=h, make_star=True)
        assert t.is_fine() and t.is_star()


def test_sample_ntfes_max_tries_raises():
    """An unreachable target must raise, not loop forever."""
    verts = [[ 1,  0,  0,  0], [ 0,  1,  0,  0], [ 0,  0,  1,  0],
             [ 0,  0,  0,  1], [-1, -1, -1, -1]]
    poly = cytools.Polytope(np.array(verts, dtype=np.int64))
    net  = DualGNN.default(device="cpu")

    # max_tries=1 with N=2: at most one attempt can succeed, so the cap
    # must fire before the second
    with pytest.raises(RuntimeError, match="max_tries"):
        sample_ntfes(poly, net, N=2, N_face_triangs=10,
                     seed=0, max_tries=1, verbose=False)


def test_sample_ntfes_ctx_reuse():
    """`return_ctx=True` hands back the setup; reusing it reproduces the
    fresh-build call exactly and rejects a mismatched polytope."""
    verts = [[ 1,  0,  0,  0], [ 0,  1,  0,  0], [ 0,  0,  1,  0],
             [ 0,  0,  0,  1], [-1, -1, -1, -1]]
    poly = cytools.Polytope(np.array(verts, dtype=np.int64))
    net  = DualGNN.default(device="cpu")

    fresh, ctx = sample_ntfes(poly, net, N=2, N_face_triangs=10, seed=0,
                              max_tries=1000, return_ctx=True,
                              verbose=False)
    via_ctx = sample_ntfes(poly, net, N=2, seed=0, max_tries=1000,
                           ctx=ctx, verbose=False)
    assert np.array_equal(fresh, via_ctx)

    # the explicit builder yields the same context (same-seed pools)
    built = build_ntfe_context(poly, net, N_face_triangs=10, seed=0,
                               verbose=False)
    assert all(np.array_equal(a, b)
               for a, b in zip(ctx.pools, built.pools))

    # a second reuse with another seed draws from the same pools
    more = sample_ntfes(poly, net, N=2, seed=1, max_tries=1000,
                        ctx=ctx, verbose=False)
    assert more.shape == fresh.shape

    other = cytools.Polytope(np.array(verts, dtype=np.int64))
    with pytest.raises(ValueError, match="different Polytope"):
        sample_ntfes(other, net, N=2, ctx=ctx, verbose=False)
