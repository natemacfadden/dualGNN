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
import warnings

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


# --- is_regular: stub regfans to exercise control flow deterministically -----
class _FakeFan:
    """Stand-in for a regfans Fan with a controllable is_regular()."""
    def __init__(self, mode):
        self.mode = mode

    def is_regular(self):
        if self.mode == "regular":   return True
        if self.mode == "irregular": return False
        if self.mode == "timeout":   raise TimeoutError("simulated regfans hang")
        if self.mode == "warning":   raise UserWarning("simulated regfans warning")
        raise AssertionError("bad mode")


def _stub_vc(mode):
    class _VC:
        def __init__(self, *a, **k): pass
        def triangulate(self, cells=None): return _FakeFan(mode)
    return _VC


def test_is_regular_outcomes():
    # exercise is_regular's control flow without a real triangulation or a 60s hang
    pts   = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.int64)
    simps = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
    orig  = geom.VectorConfiguration
    cases = [("regular", True), ("irregular", False),
             ("timeout", None), ("warning", None)]  # undetermined -> None, not False
    try:
        for mode, expected in cases:
            geom.VectorConfiguration = _stub_vc(mode)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                got = geom.is_regular(pts, simps)
            assert got is expected, f"{mode}: expected {expected!r}, got {got!r}"
    finally:
        geom.VectorConfiguration = orig


def test_is_regular_single_simplex_is_trivially_regular():
    pts = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.int64)
    assert geom.is_regular(pts, np.array([[0, 1, 2]], dtype=np.int64)) is True


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
