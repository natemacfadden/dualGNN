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
# Description:  Cross-module training hparams (>1 caller). Library-local
#               defaults stay in their owning module.
# -----------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path


# data layout
# ===========
POLYGONS_PARQUET = Path("polygons.parquet")  # CWD-relative
FTS_DIR          = Path("fts")               # CWD-relative

# splits
# ======
VAL_FRAC      = 0.15   # per-FRT val fraction (for "train" polygons)
VAL_POLY_FRAC = 0.15   # per-polygon fraction held out as role="val"
