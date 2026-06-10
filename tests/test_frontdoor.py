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
# Description:  Pin the stable contract that downstream consumers (CYTools)
#               rely on: DualGNN.default(), sample_frts, the row-order index
#               convention, and canonical fine output.
# -----------------------------------------------------------------------------

# external imports
import numpy as np

# local imports
from dualgnn       import DualGraph, sample, sample_frts
from dualgnn.model import DualGNN

PTS = np.array([[x, y] for x in range(4) for y in range(3)],   # [0,3]x[0,2]
               dtype=np.int64)


def test_default_loads_and_is_cached():
    """`DualGNN.default()` loads the packaged checkpoint (no repo paths)
    and returns the same instance on repeated calls. CPU must work."""
    net = DualGNN.default(device="cpu")
    assert net is DualGNN.default(device="cpu")
    assert next(net.parameters()).device.type == "cpu"
    assert (net.D, net.K) == (32, 16)


def test_sample_frts_contract():
    """The front-door output: deduplicated canonical fine triangulations
    whose simplices index into the caller's `pts` rows."""
    out = sample_frts(PTS, 64, seed=0)

    n_simps_per_ft = DualGraph(PTS).N_simps_per_ft
    assert out.ndim == 3 and out.shape[1:] == (n_simps_per_ft, 3)
    assert 1 <= out.shape[0] <= 64

    # deduplicated (canonical form makes exact duplicates compare equal)
    assert len(np.unique(out, axis=0)) == out.shape[0]

    for ft in out:
        # canonical: vertex indices sorted within simps, simps lex-sorted
        assert (np.sort(ft, axis=1) == ft).all()
        s = np.asarray(ft)
        assert (s[np.lexsort(s.T[::-1])] == s).all()
        # indexes the caller's rows, and is fine (uses every lattice point)
        assert {int(i) for i in ft.ravel()} == set(range(len(PTS)))


def test_sample_frts_matches_low_level_api():
    """`sample_frts` is `sample` + dedup -- same draws for the same seed."""
    net  = DualGNN.default(device="cpu")
    low  = sample(net, DualGraph(PTS), Ntriangs=16, seed=3, verbose=False)
    high = sample_frts(PTS, 16, model=net, seed=3)
    assert np.array_equal(high, np.unique(low, axis=0))


def test_sample_frts_only_regular():
    """`only_regular=True` returns a subset of the unfiltered draws."""
    net = DualGNN.default(device="cpu")
    all_ = sample_frts(PTS, 16, model=net, seed=5)
    reg  = sample_frts(PTS, 16, model=net, seed=5, only_regular=True)
    keys = {ft.tobytes() for ft in all_}
    assert 1 <= reg.shape[0] <= all_.shape[0]
    assert all(ft.tobytes() in keys for ft in reg)
