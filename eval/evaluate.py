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
# Description:  Uniformity-evaluation protocol for FRT samplers on the paper's
#               benchmark polygons. Evaluate any sampler's draws against the
#               uniform reference (and against dualGNN); see eval/README.md.
# -----------------------------------------------------------------------------

# external imports
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

# local imports
from dualgnn.geometry import canonical_simps, is_regular

_HERE = Path(__file__).resolve().parent


# polygon specs
# =============
def load_polygons() -> dict[str, dict]:
    """`eval/polygons.json` as `{name: spec}`. Each spec carries the
    polygon's `vertices`, its full `lattice_points` list (the row order
    your sampler's simplices must index into), and the exact FRT count
    `n_frt`."""
    specs = json.loads((_HERE / "polygons.json").read_text())
    return {s["name"]: s for s in specs}


# the protocol
# ============
def evaluate(
    simps: np.ndarray,
    spec:  dict,
    *,
    seed:  int = 0,
) -> dict:
    """
    Score one sampler's draws on one benchmark polygon.

    Draws are regularity-filtered (the population is FRTs), then compared
    to `M_reg` uniform draws from the polygon's `n_frt` triangulations:
        - `n_unique` / `n_collisions` vs the uniform expectation
          `N * (1 - (1 - 1/N)^M)`,
        - `kl`: KL(empirical || uniform) in nats, vs `kl_noise_floor`,
          the same statistic for actually-uniform draws of the same size
          (the floor is > 0 at finite sample size; matching it -- not 0 --
          is the target).

    Parameters
    ----------
    simps : ndarray
        `(M, n_simps, 3)` int. Each row one sampled fine triangulation,
        simplices indexing the spec's `lattice_points` rows. Draws are
        with replacement; do NOT deduplicate before evaluating.
    spec : dict
        One entry of `load_polygons()`.
    seed : int, optional
        Seed for the uniform-reference simulation. Default 0.

    Returns
    -------
    metrics : dict
        Keys: `n_samples`, `regular_frac`, `n_unique`, `n_collisions`,
        `uniform_unique`, `kl`, `kl_noise_floor`.
    """
    pts = np.asarray(spec["lattice_points"], dtype=np.int64)
    N   = spec["n_frt"]
    M   = len(simps)

    reg   = [s for s in simps if is_regular(pts, np.asarray(s))]
    M_reg = len(reg)
    keys  = [canonical_simps(np.asarray(s, dtype=np.int32)).tobytes()
             for s in reg]
    counts = np.fromiter(Counter(keys).values(), dtype=float)

    # uniform reference at the same (post-filter) sample size
    rng        = np.random.default_rng(seed)
    ref_counts = np.fromiter(
        Counter(rng.integers(0, N, size=M_reg).tolist()).values(),
        dtype=float,
    )

    def kl(c):
        p = c / c.sum()
        return float(np.sum(p * np.log(p * N)))

    return {
        "n_samples":      M,
        "regular_frac":   M_reg / M if M else float("nan"),
        "n_unique":       int(len(counts)),
        "n_collisions":   int(M_reg - len(counts)),
        "uniform_unique": float(N * (1.0 - (1.0 - 1.0 / N) ** M_reg)),
        "kl":             kl(counts) if M_reg else float("nan"),
        "kl_noise_floor": kl(ref_counts) if M_reg else float("nan"),
    }


def _print_report(name: str, m: dict) -> None:
    print(f"\n=== {name} ===")
    print(f"  samples          {m['n_samples']:,}")
    print(f"  regular frac     {m['regular_frac']:.4f}")
    print(f"  unique           {m['n_unique']:,}   "
          f"(uniform expects ~{m['uniform_unique']:,.1f})")
    print(f"  collisions       {m['n_collisions']:,}")
    print(f"  KL(emp||unif)    {m['kl']:.4f} nats   "
          f"(noise floor {m['kl_noise_floor']:.4f})")
    print(f"  -> excess KL     {m['kl'] - m['kl_noise_floor']:+.4f} "
          f"(0 = indistinguishable from uniform at this sample size)")


# CLI
# ===
def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate FRT-sampler draws against the uniform "
                    "reference on the paper's benchmark polygons.",
    )
    p.add_argument("--polygon", required=True,
                   help="polygon name from eval/polygons.json "
                        "(e.g. fig11_00, 4x4sq); 'list' to list all")
    p.add_argument("--samples", type=Path, default=None,
                   help=".npy file, (M, n_simps, 3) int simplices indexing "
                        "the polygon's lattice_points rows. Omit to draw "
                        "from dualGNN instead (the baseline to beat)")
    p.add_argument("--n", type=int, default=10_000,
                   help="draws for the dualGNN baseline (default 10000)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    polys = load_polygons()
    if args.polygon == "list":
        for name, s in polys.items():
            print(f"{name:<10} n_pts={s['n_pts']:>3}  n_frt={s['n_frt']:,}")
        return
    spec = polys[args.polygon]

    if args.samples is not None:
        simps = np.load(args.samples)
        label = str(args.samples)
    else:
        from dualgnn import DualGraph, sample
        from dualgnn.model import DualGNN
        pts   = np.asarray(spec["lattice_points"], dtype=np.int64)
        simps = sample(DualGNN.default(), DualGraph(pts),
                       Ntriangs=args.n, seed=args.seed, verbose=False)
        label = f"dualGNN reinforce.pt ({args.n:,} draws)"

    _print_report(f"{args.polygon}: {label}", evaluate(simps, spec,
                                                       seed=args.seed))


if __name__ == "__main__":
    main()
