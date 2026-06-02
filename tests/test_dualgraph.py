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
# Description:  Tests for DualGraph construction.
# -----------------------------------------------------------------------------

# external imports
import numpy as np

# local imports
from dualgnn import DualGraph
from dualgnn.dualgraph import _pair_compat, _simp_compat


def test_square():
    # unit square; points: 0=(0,0) 1=(1,0) 2=(0,1) 3=(1,1)
    g = DualGraph(np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.int64))

    # the four unimodular simps, in i<j<k enumeration order
    assert np.array_equal(g.simps, [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])

    # area = 1, so a fine triangulation has 2 simps
    assert g.N_simps_per_ft == 2

    # only the two diagonal splits tile without overlap: {0,3} and {1,2}
    assert g.simp_compat.dtype == bool
    assert np.array_equal(g.simp_compat, g.simp_compat.T)        # symmetric
    assert np.array_equal(g.simp_compat, np.array([[0, 0, 0, 1],
                                                   [0, 0, 1, 0],
                                                   [0, 1, 0, 0],
                                                   [1, 0, 0, 0]], dtype=bool))

    # 2 undirected facet-sharing compatible pairs -> 4 directed edges,
    # stored as [sources; sinks] (row 0 -> row 1)
    assert np.array_equal(g.edges, [[0, 3, 1, 2],     # source simps
                                    [3, 0, 2, 1]])    # sink simps

    # edge 0->3: my=(0,0) your=(1,1) e0,e1={(1,0),(0,1)}; my+your=(1,1)=e0+e1,
    # so the dependency is my + your - e0 - e1 = 0, i.e. [1, 1, -1, -1]. both
    # shared coefficients are -1, so swapping e0/e1 leaves it unchanged: the
    # signed-area ordering can't surface here (test_triangle exercises it)
    assert np.array_equal(g.circ_features, [[1, 1, -1, -1]] * 4)


def test_triangle():
    # conv{(0,0),(0,2),(1,1)}; labels 0=(0,0) 1=(0,1) 2=(0,2) 3=(1,1).
    # chosen because the candidate filter must reject the collinear triple
    # (0,1,2) and the area-1 triple (0,2,3), and the two surviving simps share
    # a non-parallelogram edge, so the circuits are asymmetric (unlike a square)
    g = DualGraph(np.array([[0, 0], [0, 1], [0, 2], [1, 1]], dtype=np.int64))

    assert np.array_equal(g.simps, [[0, 1, 3], [1, 2, 3]])
    assert g.N_simps_per_ft == 2
    assert np.array_equal(g.simp_compat, g.simp_compat.T)        # symmetric
    assert np.array_equal(g.simp_compat, np.array([[0, 1],
                                                   [1, 0]], dtype=bool))
    assert np.array_equal(g.edges, [[0, 1], [1, 0]])

    # edge 0->1: my=(0,0) your=(0,2) e0=(0,1) e1=(1,1); my+your=(0,2)=2*e0, so
    # the dependency is my + your - 2*e0 = 0, i.e. [1, 1, -2, 0]; the reverse
    # edge swaps the shared-vertex order, giving [1, 1, 0, -2]
    assert np.array_equal(g.circ_features, [[1, 1, -2, 0], [1, 1, 0, -2]])


def test_pair_compat_known_pairs():
    pts   = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.int64)
    simps = np.array([[0, 1, 2], [1, 2, 3], [0, 1, 3]], dtype=np.int64)
    assert _pair_compat(pts, simps, 0, 1)        # tile the square -> compatible
    assert not _pair_compat(pts, simps, 0, 2)    # interiors overlap
    assert not _pair_compat(pts, simps, 0, 0)    # never compatible with itself


def test_simp_compat_disjoint_bbox():
    # two unit triangles far apart: the bbox pre-filter should short-circuit
    # them to compatible without the geometric test
    pts   = np.array([[0, 0], [1, 0], [0, 1],
                      [10, 0], [11, 0], [10, 1]], dtype=np.int64)
    simps = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    assert np.array_equal(_simp_compat(pts, simps), np.array([[0, 1],
                                                              [1, 0]], dtype=bool))


# allow running without pytest
# ============================
if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
