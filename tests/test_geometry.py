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
# Description:  Tests for the geometry helpers.
# -----------------------------------------------------------------------------

# external imports
import numpy as np

# local imports
import dualgnn.geometry as geom
from dualgnn.geometry import canonical_simps, enum_lattice_pts


def test_canonical_simps_permutation_invariant():
    # a triangulation is the same regardless of row order or within-simp vertex
    # order, so canonical_simps must collapse all of those to one form (this is
    # the dedup key the uniformity diagnostics rely on)
    canon = canonical_simps(np.array([[0, 1, 2], [1, 2, 3]]))
    assert np.array_equal(canon, [[0, 1, 2], [1, 2, 3]])
    variants = [
        np.array([[1, 2, 3], [0, 1, 2]]),     # rows swapped
        np.array([[2, 0, 1], [3, 1, 2]]),     # vertices permuted within rows
        np.array([[2, 1, 0], [2, 3, 1]]),     # both
    ]
    for v in variants:
        c = canonical_simps(v)
        assert np.array_equal(c, canon)
        assert c.tobytes() == canon.tobytes()

def test_canonical_simps_distinguishes():
    # two genuinely different triangulations must not collapse to one key
    a = canonical_simps(np.array([[0, 1, 2], [1, 2, 3]]))
    b = canonical_simps(np.array([[0, 1, 3], [0, 2, 3]]))
    assert a.tobytes() != b.tobytes()

def test_enum_lattice_pts():
    def pts_set(verts):
        out = enum_lattice_pts(np.array(verts, dtype=np.int64))
        return {tuple(p) for p in out.tolist()}

    # unit square: just the 4 corners
    assert pts_set([[0, 0], [1, 0], [1, 1], [0, 1]]) == \
        {(0, 0), (1, 0), (1, 1), (0, 1)}

    # [0,2]^2: the full 3x3 grid of lattice points
    assert pts_set([[0, 0], [2, 0], [2, 2], [0, 2]]) == \
        {(x, y) for x in range(3) for y in range(3)}

    # conv{(0,0),(0,4),(6,0)}: 19 lattice points (Pick: area 12, B 12, I 7)
    tri = enum_lattice_pts(np.array([[0, 0], [0, 4], [6, 0]], dtype=np.int64))
    assert len(tri) == 19

def test_enum_lattice_pts_degenerate():
    # collinear vertices are not a valid polygon -> None
    assert enum_lattice_pts(
        np.array([[0, 0], [1, 1], [2, 2]], dtype=np.int64)) is None


# --- is_regular: real triangulations; verdicts independently verified with --
# --- CYTools. P_{4,4} is the 5x5 lattice grid; point index = x*5 + y. --------
_PTS_P44 = np.array([[x, y] for x in range(5) for y in range(5)], dtype=np.int64)

# Santos' fine-but-NON-regular triangulation (Kaibel & Ziegler 2003) -- the
# canonical smallest non-regular fine triangulation. CYTools: fine, irregular.
_SANTOS_IRREGULAR = np.array([
    [0, 1, 5], [1, 5, 6], [5, 6, 10], [6, 10, 11], [1, 2, 6], [2, 6, 11],
    [2, 7, 11], [7, 11, 12], [2, 3, 8], [2, 7, 8], [3, 4, 9], [3, 8, 9],
    [7, 12, 13], [7, 13, 14], [7, 8, 14], [8, 9, 14], [10, 15, 16],
    [10, 16, 17], [10, 11, 17], [11, 12, 17], [15, 20, 21], [15, 16, 21],
    [16, 21, 22], [16, 17, 22], [12, 13, 17], [13, 17, 22], [13, 18, 22],
    [18, 22, 23], [13, 14, 18], [14, 18, 19], [18, 19, 23], [19, 23, 24],
], dtype=np.int64)

# Three distinct fine REGULAR triangulations of P_{4,4} (CYTools is_regular ==
# True), sampled via cytools.random_triangulations_fast.
_REGULAR = [
    np.array([
        [6, 7, 12], [2, 6, 7], [6, 11, 12], [6, 10, 11], [1, 2, 6], [1, 5, 6],
        [5, 6, 10], [7, 8, 12], [2, 7, 8], [8, 12, 13], [8, 9, 13], [3, 4, 8],
        [4, 8, 9], [2, 3, 8], [11, 12, 16], [11, 15, 16], [10, 11, 15],
        [12, 13, 18], [12, 16, 17], [12, 17, 18], [13, 14, 18], [9, 13, 14],
        [16, 17, 21], [15, 16, 21], [17, 18, 22], [17, 21, 22], [18, 19, 24],
        [18, 23, 24], [14, 18, 19], [18, 22, 23], [0, 1, 5], [15, 20, 21],
    ], dtype=np.int64),
    np.array([
        [6, 7, 12], [2, 6, 7], [6, 11, 12], [5, 6, 11], [1, 2, 6], [1, 5, 6],
        [7, 8, 12], [3, 7, 8], [2, 3, 7], [8, 12, 13], [8, 9, 13], [3, 8, 9],
        [11, 12, 16], [11, 15, 16], [5, 10, 11], [10, 11, 15], [12, 13, 18],
        [12, 16, 17], [12, 17, 18], [13, 18, 19], [9, 13, 14], [13, 14, 19],
        [16, 17, 22], [15, 16, 21], [16, 21, 22], [17, 18, 22], [18, 19, 23],
        [18, 22, 23], [0, 1, 5], [3, 4, 9], [15, 20, 21], [19, 23, 24],
    ], dtype=np.int64),
    np.array([
        [6, 7, 11], [2, 6, 7], [5, 6, 11], [1, 2, 6], [1, 5, 6], [7, 8, 13],
        [2, 7, 8], [7, 11, 12], [7, 12, 13], [8, 9, 13], [2, 3, 8], [3, 8, 9],
        [11, 12, 17], [11, 16, 17], [11, 15, 16], [5, 10, 11], [10, 11, 15],
        [12, 13, 17], [13, 17, 18], [13, 18, 19], [9, 13, 14], [13, 14, 19],
        [16, 17, 22], [15, 16, 21], [16, 21, 22], [17, 18, 22], [18, 19, 23],
        [18, 22, 23], [0, 1, 5], [3, 4, 9], [15, 20, 21], [19, 23, 24],
    ], dtype=np.int64),
]


def test_is_regular_matches_cytools():
    # is_regular must reproduce CYTools' verdicts on real triangulations:
    # the Santos triangulation is irregular, the three samples are regular.
    assert geom.is_regular(_PTS_P44, _SANTOS_IRREGULAR) is False
    for i, simps in enumerate(_REGULAR):
        assert geom.is_regular(_PTS_P44, simps) is True, f"regular_{i} regular"


def test_is_regular_single_simplex_is_trivially_regular():
    pts = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.int64)
    assert geom.is_regular(pts, np.array([[0, 1, 2]], dtype=np.int64)) is True


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
