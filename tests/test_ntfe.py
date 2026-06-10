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
#               Needs CYTools, which is conda-only, so the whole module
#               skips under a pip-only install (and in CI); run it from the
#               conda env (`conda env create -f environment.yml`).
# -----------------------------------------------------------------------------

# external imports
import numpy as np
import pytest

cytools = pytest.importorskip("cytools")

# local imports
from dualgnn.model import DualGNN
from dualgnn.ntfe  import sample_ntfes


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
