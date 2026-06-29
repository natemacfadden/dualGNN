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

import numpy as np
import pytest

# local imports
from dualgnn import DualGraph, sample
from dualgnn.geometry import canonical_simps
from dualgnn.model import DualGNN


def _two_area(tri):
    """Twice the signed area of an integer triangle (3, 2)."""
    (x0, y0), (x1, y1), (x2, y2) = tri
    return x0 * (y1 - y2) + x1 * (y2 - y0) + x2 * (y0 - y1)

def _proj_overlaps(p, q):
    """Do integer ranges [min p, max p] and [min q, max q] overlap in positive
    length? Touching at a single point does not count."""
    return min(max(p), max(q)) > max(min(p), min(q))

def _interiors_overlap(t1, t2):
    """True if two integer triangles share positive-area interior. Independent
    separating-axis test (not the sampler's own compatibility code); a shared
    edge or vertex is a touch, not an overlap."""
    for tri in (t1, t2):
        for i in range(3):
            ax, ay = tri[i]
            bx, by = tri[(i + 1) % 3]
            nx, ny = -(int(by) - int(ay)), int(bx) - int(ax)   # edge normal
            p1 = [nx * int(x) + ny * int(y) for x, y in t1]
            p2 = [nx * int(x) + ny * int(y) for x, y in t2]
            if not _proj_overlaps(p1, p2):
                return False                                   # axis separates
    return True

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
    net = DualGNN.default()
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

def test_sample_outputs_tile_without_gaps_or_overlaps():
    """
    Stronger than the necessary-but-not-sufficient "edge in <=2 simps" check:
    confirm each sampled triangulation (a) has total simplex area equal to the
    polygon area (shoelace) and (b) has no two simplices overlapping in
    positive-area interior - together a genuine gap-free, overlap-free tiling.
    """
    pts = np.array([[x, y] for x in range(4) for y in range(4)],   # [0,3]^2
                   dtype=np.int64)
    poly_two_area = 18   # [0,3]^2 is a 3x3 square: area 9, twice-area 18
    g   = DualGraph(pts)
    net = DualGNN.default()
    fts = sample(net, g, Ntriangs=4, seed=0)

    for ft in fts:
        tris = [pts[s] for s in np.asarray(ft, dtype=np.int64)]

        # (a) areas sum to the polygon area - no net gap or overlap
        assert sum(abs(_two_area(t)) for t in tris) == poly_two_area

        # (b) no pair shares positive-area interior
        for i in range(len(tris)):
            for j in range(i + 1, len(tris)):
                assert not _interiors_overlap(tris[i], tris[j]), \
                    f"simps {i},{j} overlap"

def _enumerate_frts(g):
    """Every fine triangulation of g's polygon, as a set of canonical_simps
    keys. A fine triangulation is exactly N_simps_per_ft pairwise-compatible
    candidate simps: non-overlapping unimodular simps whose areas then sum to
    the polygon area must tile it. Independent of the model and of CYTools."""
    simps, compat, target = g.simps, g.simp_compat, g.N_simps_per_ft
    n = len(simps)
    out, clique = set(), []
    def rec(start):
        if len(clique) == target:
            out.add(canonical_simps(simps[clique]).tobytes())
            return
        for i in range(start, n):
            if all(compat[i][j] for j in clique):
                clique.append(i)
                rec(i + 1)
                clique.pop()
    rec(0)
    return out

def test_sampler_is_uniform_over_frts():
    """
    The headline claim - the trained sampler draws uniformly over fine
    triangulations - as a CI gate (the review noted it was paper-only). On a
    small polygon we enumerate every FRT independently, draw many samples with
    the shipped REINFORCE-trained model at beta=1 (uniform-over-FRT), and
    chi-square the empirical counts against uniform. A non-uniform sampler
    drives the p-value down; the seed is fixed so the verdict is deterministic.
    """
    chisquare = pytest.importorskip("scipy.stats").chisquare

    # 5-FRT triangle conv{(0,0),(0,2),(3,0)} - small enough to fully enumerate
    pts = np.array([(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0), (3, 0)],
                   dtype=np.int64)
    g    = DualGraph(pts)
    frts = _enumerate_frts(g)
    assert len(frts) == 5            # known FRT count for this polygon

    net = DualGNN.default()                          # shipped trained model
    M   = 3000
    fts = sample(net, g, Ntriangs=M, seed=0)         # beta=1 -> uniform-over-FRT

    counts = Counter(
        canonical_simps(np.asarray(ft, dtype=np.int64)).tobytes() for ft in fts
    )
    assert set(counts) == frts                       # every FRT hit, none extra

    observed = np.array([counts[k] for k in frts])
    _, p = chisquare(observed)                        # null: uniform over N bins
    assert p > 0.01, f"sampler looks non-uniform: chi-square p={p:.4f}"

if __name__ == "__main__":
    test_sample_outputs_are_valid_fine_triangulations()
    print("ok  test_sample_outputs_are_valid_fine_triangulations")
    test_sample_outputs_tile_without_gaps_or_overlaps()
    print("ok  test_sample_outputs_tile_without_gaps_or_overlaps")
    test_sampler_is_uniform_over_frts()
    print("ok  test_sampler_is_uniform_over_frts\n\n3 passed")
