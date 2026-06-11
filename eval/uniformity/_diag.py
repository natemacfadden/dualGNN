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
# Description:  Shared diagnostics for the uniformity checks (see README.md):
#               KL, collisions, rank-frequency, and a fairness-vs-time scatter.
# -----------------------------------------------------------------------------

# external imports
from collections import Counter
import os
import time

import matplotlib
matplotlib.use("Agg")          # headless-safe; we only save figures
import matplotlib.pyplot as plt
import numpy as np

# local imports
from dualgnn          import DualGraph, sample, grow2d, pushing
from dualgnn.geometry import enum_lattice_pts, canonical_simps, is_regular
from dualgnn.model    import DualGNN

# config
# ======
_HERE        = os.path.dirname(__file__)
OUT_DIR      = _HERE
SEED         = 0
CACHE_MAX_MB = 50              # skip the sample cache if it would exceed this

# per-sampler colors/markers, shared across the plots
# dualGNN color = dark end of a truncated viridis_r, matching the paper figures
STYLE = {
    "dualGNN":      dict(color="#453781", marker="o"),
    "grow2d":       dict(color="#800020", marker="D"),
    "pushing":      dict(color="#2ca02c", marker="s"),
    "uniform(1/N)": dict(color="black",   marker="x"),
}


# sampling
# ========
def _draw_regular(name, net, pts, M, seed):
    """
    Draw M triangulations from one sampler and keep the regular (FRT) ones.

    Parameters
    ----------
    name : str
        Sampler to use: "dualGNN", "grow2d", or "pushing".
    net : DualGNN
        Loaded model (used only by "dualGNN").
    pts : np.ndarray
        Lattice points of the polygon, shape (Npts, 2).
    M : int
        Number of triangulations to draw.
    seed : int
        RNG seed.

    Returns
    -------
    reg : list of np.ndarray
        Regular triangulations, in draw order.
    frac : float
        Fraction of the draws that were regular.
    t_per : float
        Per-draw sampling time in seconds (excludes the regularity filter).
    """
    t0 = time.perf_counter()
    if name == "dualGNN":
        # returns numpy, so the gpu work is synced by the time it returns
        batch   = sample(net, DualGraph(pts), Ntriangs=M, seed=seed)
        triangs = [batch[i] for i in range(batch.shape[0])]
    else:
        fn      = {"grow2d": grow2d, "pushing": pushing}[name]
        triangs = [s for s, status in (fn(pts, seed=seed + i) for i in range(M))
                   if status == 0]
    t_per = (time.perf_counter() - t0) / max(1, M)
    if name == "pushing":                                # always regular
        return triangs, 1.0, t_per
    reg = [t for t in triangs if is_regular(pts, t)]
    return reg, (len(reg) / len(triangs) if triangs else float("nan")), t_per


# diagnostics
# ===========
def _diagnostics(counts, N):
    """
    Fairness diagnostics from an array of per-FRT sample counts.

    Parameters
    ----------
    counts : np.ndarray
        Number of times each distinct sampled FRT was drawn.
    N : int
        Total number of FRTs (the uniform universe size).

    Returns
    -------
    dict
        With keys "unique", "collisions", "kl" (empirical-to-uniform, nats),
        and "ranks" (counts sorted descending, for the rank-frequency plot).
    """
    M = counts.sum()
    p = counts / M
    return dict(unique=len(counts), collisions=int(M - len(counts)),
                kl=float(np.sum(p * np.log(p * N))),
                ranks=np.sort(counts)[::-1])


# plotting / saving
# =================
def _save_cache(tag, cache):
    """
    Save the per-sampler FRT draws to a compressed npz, unless too large.
    """
    total = sum(a.nbytes for a in cache.values())
    out   = os.path.join(OUT_DIR, f"samples_{tag}.npz")
    if total > CACHE_MAX_MB * 1e6:
        print(f"sample cache ~{total/1e6:.0f} MB > {CACHE_MAX_MB} MB "
              f"-- not saving")
        return
    np.savez_compressed(out, **cache)
    print(f"cached {sum(len(a) for a in cache.values()):,} sampled FRTs "
          f"({total/1e6:.1f} MB) -> {out}")


def _save(fig, kind, tag):
    """
    Save a figure to uniformity/<kind>_<tag>.png and close it.
    """
    out = os.path.join(OUT_DIR, f"{kind}_{tag}.png")
    fig.savefig(out, dpi=150); plt.close(fig); print(f"saved {out}")


# entry point
# ===========
def _estimate_runtime(net, pts, samples, probe=50):
    """
    Probe a few draws per sampler to print an up-front wall-time estimate.
    """
    print("  estimating runtime ...", flush=True)
    per = {}
    for name in ("dualGNN", "grow2d", "pushing"):
        t0 = time.perf_counter()
        _draw_regular(name, net, pts, probe, seed=SEED)
        per[name] = (time.perf_counter() - t0) / probe
    est = samples * sum(per.values())
    print(f"  estimated runtime ~{est/60:.1f} min "
          f"(dualGNN ~{1.0/per['dualGNN']:.0f} draws/s, "
          f"SAMPLES={samples:,})", flush=True)


def run(poly, samples):
    """
    Run the uniformity diagnostics for one polygon.

    Prints the fairness table, caches the draws, and saves the rank-frequency
    and KL-vs-time plots to uniformity/.

    Parameters
    ----------
    poly : dict
        Polygon spec with keys "name", "tag", "verts", and "N_FRT".
    samples : int
        Number of draws per sampler.
    """
    net = DualGNN.default()
    pts = np.asarray(enum_lattice_pts(np.array(poly["verts"], dtype=np.int64)),
                     dtype=np.int64)
    N, tag = poly["N_FRT"], poly["tag"]
    if samples * (samples - 1) / (2.0 * N) < 1.0:
        print(f"  note: ~{samples:,} draws is far below the ~sqrt(N) "
              f"needed to see collisions at N={N:,} -- collision/KL "
              f"signal will be weak (see README)")
    print(f"\n=== {poly['name']}: {pts.shape[0]} lattice points, N_FRT = {N:,} "
          f"(from paper) | device {next(net.parameters()).device} ===")
    print(f"    {samples} draws per sampler "
          f"(regular-filtered, vs the 1/N reference)")
    _estimate_runtime(net, pts, samples)
    print()

    learned  = ["dualGNN", "grow2d", "pushing"]
    rows     = {}
    rankfreq = {}
    cache    = {}

    for name in learned:
        reg, frac, t_per = _draw_regular(name, net, pts, samples, seed=SEED)
        canon  = [canonical_simps(np.asarray(x)) for x in reg]
        keys   = (c.astype(np.int32).tobytes() for c in canon)
        counts = np.fromiter(Counter(keys).values(), dtype=float)
        d = _diagnostics(counts, N)
        rankfreq[name] = d["ranks"]
        cache[name]    = np.stack([c.astype(np.int8) for c in canon])
        rows[name] = dict(regf=frac, uniq=d["unique"], coll=d["collisions"],
                          kl=d["kl"], time=t_per)

    keys   = np.random.default_rng(SEED).integers(0, N, size=samples).tolist()
    counts = np.fromiter(Counter(keys).values(), dtype=float)
    d = _diagnostics(counts, N)
    rankfreq["uniform(1/N)"] = d["ranks"]
    rows["uniform(1/N)"] = dict(regf=1.0, uniq=d["unique"],
                                coll=d["collisions"], kl=d["kl"],
                                time=float("nan"))

    # table, rows in order of increasing fairness (descending KL), as in
    # the paper
    order = sorted(learned, key=lambda n: -rows[n]["kl"])
    hdr = (f"{'sampler':<14}{'reg.frac':>9}{'#unique':>12}{'#collide':>10}"
           f"{'KL':>10}{'sec/draw':>11}")
    print(hdr); print("-" * len(hdr))
    for name in order + ["uniform(1/N)"]:
        r = rows[name]
        tcol = f"{r['time']:.2e}" if r["time"] == r["time"] else "--"
        print(f"{name:<14}{r['regf']:>9.3f}{r['uniq']:>12.0f}{r['coll']:>10.0f}"
              f"{r['kl']:>10.4f}{tcol:>11}")
    print("\n(lower KL + uniques/collisions matching the 1/N reference "
          "=> more uniform)")

    _save_cache(tag, cache)

    # rank-frequency of sampled FRTs (flat => uniform-like)
    fig, ax = plt.subplots(figsize=(7, 4))
    for name in order + ["uniform(1/N)"]:
        c = rankfreq[name]
        ax.plot(np.arange(1, len(c) + 1), c,
                color=STYLE[name]["color"], lw=1.5, label=name)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("rank"); ax.set_ylabel("count")
    ax.set_title(f"rank-frequency: {poly['name']}  (M={samples})")
    ax.legend(loc="upper right"); fig.tight_layout()
    _save(fig, "rankfreq", tag)

    # fairness vs time: KL vs per-draw time (bottom-left = fast + fair)
    fig, ax = plt.subplots(figsize=(7, 4))
    for name in order:
        s = STYLE[name]
        ax.scatter(rows[name]["time"], rows[name]["kl"], s=80,
                   marker=s["marker"], color=s["color"], edgecolor="black",
                   linewidth=0.6, zorder=3, label=name)
    ax.axhline(rows["uniform(1/N)"]["kl"], color="black", lw=2.0, ls="--",
               zorder=1, label="1/N")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("sample time [sec]")
    ax.set_ylabel("KL(empirical || uniform) [nats]")
    ax.set_title(f"fairness vs time: {poly['name']}  (M={samples})")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(loc="upper right"); fig.tight_layout()
    _save(fig, "kl_vs_time", tag)
