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
# Description:  Test that the sampler produces valid fine triangulations.
# -----------------------------------------------------------------------------

# external imports
from collections import Counter
import os

import numpy as np

# local imports
from dualgnn import DualGraph, sample
from dualgnn.model import DualGNN

_CKPT = os.path.join(os.path.dirname(__file__), "..", "ckpts", "reinforce.pt")


def _two_area(tri):
    """Twice the signed area of an integer triangle (3, 2)."""
    (x0, y0), (x1, y1), (x2, y2) = tri
    return x0 * (y1 - y2) + x1 * (y2 - y0) + x2 * (y0 - y1)

def test_sample_outputs_are_valid_fine_triangulations():
    """
    Every sampled triangulation is a valid fine triangulation by construction.

    The autoregressive masking guarantees fineness (unimodular simps that tile
    the polygon, using every lattice point). Regularity is NOT enforced by
    construction; it is learned, so it is deliberately not asserted here.
    """
    pts = np.array([[x, y] for x in range(4) for y in range(4)],   # [0,3]^2
                   dtype=np.int64)
    g   = DualGraph(pts)
    net = DualGNN.from_ckpt(_CKPT)
    fts = sample(net, g, Ntriangs=4, seed=0)

    assert fts.shape == (4, g.N_simps_per_ft, 3)
    npts = len(pts)

    for ft in fts:
        simps = np.asarray(ft, dtype=np.int64)

        # exactly N_simps_per_ft (= 2 * polygon area) simps
        assert len(simps) == g.N_simps_per_ft

        # every simp is unimodular (a fine candidate simp)
        assert all(abs(_two_area(pts[s])) == 1 for s in simps)

        # fine: every lattice point is used as a vertex
        assert {int(i) for i in simps.ravel()} == set(range(npts))

        # valid partition: no undirected edge is shared by more than 2 simps
        edges = Counter()
        for s in simps:
            a, b, c = sorted(int(v) for v in s)
            edges[(a, b)] += 1
            edges[(a, c)] += 1
            edges[(b, c)] += 1
        assert max(edges.values()) <= 2

if __name__ == "__main__":
    test_sample_outputs_are_valid_fine_triangulations()
    print("ok  test_sample_outputs_are_valid_fine_triangulations\n\n1 passed")
