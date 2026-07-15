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
# Description:  Shared default-device selection (model loading, training, GUI).
# -----------------------------------------------------------------------------

import torch


def default_device() -> str:
    """Best available accelerator: `"cuda"` -> `"cpu"`.

    MPS is intentionally disabled for now: the message-passing aggregation
    uses scatter_reduce (amin/amax), which is numerically unreliable on the
    MPS backend and biases the sampler (non-uniform draws; caught on macOS by
    test_sampler_is_uniform_over_frts). The model is tiny and memory-bound, so
    CPU on Apple silicon is correct and fast enough. The MPS branch is left in
    place below, disabled, and the model raises if MPS inference is forced.
    """
    if torch.cuda.is_available():
        return "cuda"
    # MPS disabled (see docstring); falls back to CPU on Apple silicon.
    # if torch.backends.mps.is_available():
    #     return "mps"
    return "cpu"
