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
# Description:  Contract-violating inputs to DualGraph must raise clear
#               ValueErrors, not corrupt outputs or leak internal errors.
# -----------------------------------------------------------------------------

# external imports
import numpy as np
import pytest

# local imports
from dualgnn import DualGraph

SQUARE = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.int64)


@pytest.mark.parametrize("bad, match", [
    (None,                                       "pts=None"),
    (np.zeros((4, 3), dtype=np.int64),           r"shape \(Npts, 2\)"),
    (np.array([[0.5, 0.0], [1, 0], [0, 1]]),     "non-integral"),
    (np.array([[0, 0], [1, 0]]),                 ">= 3 lattice points"),
    (np.array([[0, 0], [1, 0], [0, 1], [1, 0]]), "duplicate"),
    (np.array([[0, 0], [1, 0], [2, 0], [3, 0]]), "collinear"),
    (np.array([[0, 0], [2, 0], [0, 2]]),         "all lattice points"),
])
def test_bad_inputs_raise_value_error(bad, match):
    with pytest.raises(ValueError, match=match):
        DualGraph(bad)


def test_valid_inputs_still_accepted():
    # canonical int64 input
    assert DualGraph(SQUARE).N_simps_per_ft == 2
    # integral floats are accepted (cast, not rejected)
    assert DualGraph(SQUARE.astype(np.float64)).N_simps_per_ft == 2
    # plain lists are accepted
    assert DualGraph(SQUARE.tolist()).N_simps_per_ft == 2
