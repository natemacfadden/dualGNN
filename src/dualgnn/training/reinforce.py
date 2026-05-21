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
# Description:  REINFORCE fine-tune of a DualGNN AR sampler. Each step rolls
#               out grad-tracked trajectories and updates with
#               reward = -log P (valid in pool) or INVALID_REWARD (otherwise).
# -----------------------------------------------------------------------------

# external imports
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib     import Path

import numpy as np
import polars as pl
import torch
from torch.utils.tensorboard import SummaryWriter

# local imports
from ..dualgraph import DualGraph
from ..geometry  import canonical_simps, is_regular
from ..model     import DualGNN
from ..sampler   import ar_rollout_batch, sample
from .io         import load_polygons, save_ckpt


# training-loop internals (not user-facing)
# =========================================
EMA_LOG_BETA   = 0.9        # smoothing for printed reward/regular EMA
GRAD_CLIP_NORM = 1.0        # gradient clipping
STD_CLAMP_MIN  = 1e-6       # advantage normalization
LOG_FREQ       = 20         # log every N steps

# main
# ====
def reinforce(
    *,
    init_ckpt:      Path,
    run_path:       Path,
    steps:          int,
    batch:          int,
    lr:             float,
    val_every:      int,
    val_Ntriangs:   int,
    val_Npolys:     int,
    ckpt_every:     int,
    device:         str | None,
    seed:           int,
    beta:           float =  1.0,
    invalid_reward: float = -2.0,
):
    """
    REINFORCE fine-tune of a dualgnn AR sampler.

    Picks a train polygon per step, rolls out `batch` grad-tracked trajectories,
    and updates with REINFORCE (`reward = -log P` if valid, else
    `invalid_reward`).

    Parameters
    ----------
    init_ckpt : Path
        Warm-start ckpt to initialize from. Its parent directory provides the
        training pool (`polygons.parquet`, `fts/`).
    run_path : Path
        Output directory for ckpts and TB logs. Separated from the supervised
        training starting point.
    steps : int
        Number of REINFORCE update steps.
    batch : int
        Trajectories per gradient step.
    lr : float
        AdamW learning rate.
    val_every : int
        Validate every N steps.
    val_Ntriangs : int
        Triangulations drawn per val polygon.
    val_Npolys : int
        Val polygons sampled per validation pass.
    ckpt_every : int
        Save a checkpoint every N steps (final ckpt always saved).
    device : str or None
        Target device. `None` -> cuda if available, else cpu.
    seed : int
        Seed for torch + numpy.
    beta : float, optional
        Inverse temperature: `softmax(beta * logits)`. Default 1.0 (uniform-
        over-FRTs sampling). Matches the `beta` arg of `sampler.sample`.
    invalid_reward : float, optional
        Reward assigned to invalid (non-regular) draws. Default -2.0.
    """
    device  = device or ("cuda" if torch.cuda.is_available() else "cpu")
    src_run = init_ckpt.parent
    print(f"[reinforce] device={device}", flush=True)

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # load polygons + per-polygon state
    # ---------------------------------
    train_df, val_df = load_polygons(src_run / "polygons.parquet")
    print(f"[reinforce] {train_df.height} train polys, "
          f"{val_df.height} val polys", flush=True)

    def load_states(df: pl.DataFrame) -> list[RLPolyState]:
        states = (build_rl_poly_state(r, src_run)
                  for r in df.iter_rows(named=True))
        return [s for s in states if len(s.key_to_idx) > 0]

    train_states = load_states(train_df)
    val_states   = load_states(val_df)
    print(f"[reinforce] {len(train_states)} train polys, "
          f"{len(val_states)} val polys with non-empty reg pool", flush=True)

    # init model
    # ----------
    ckpt = torch.load(init_ckpt, map_location=device, weights_only=False)
    D    = ckpt["hparams"]["d_model"]
    K    = ckpt["hparams"]["k_rounds"]
    net  = DualGNN(D=D, K=K).to(device)
    sd   = {k.replace("_orig_mod.", ""): v
            for k, v in ckpt["state_dict"].items()}
    net.load_state_dict(sd)
    print(f"[reinforce] init from {init_ckpt} (D={D}, K={K})", flush=True)

    # run dir + writer
    # ----------------
    run_path.mkdir(parents=True, exist_ok=True)
    hparams_log = {
        "init_ckpt":      str(init_ckpt),
        "run_path":       str(run_path),
        "steps":          steps,
        "batch":          batch,
        "lr":             lr,
        "beta":           beta,
        "invalid_reward": invalid_reward,
        "val_every":      val_every,
        "val_Ntriangs":   val_Ntriangs,
        "val_Npolys":     val_Npolys,
        "ckpt_every":     ckpt_every,
        "device":         device,
        "seed":           seed,
    }
    (run_path / "hparams.json").write_text(json.dumps(hparams_log, indent=2))
    writer = SummaryWriter(log_dir=str(run_path))
    print(f"[reinforce] run dir: {run_path}", flush=True)

    # training loop
    # -------------
    optim = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.0)

    print("[reinforce] training ...", flush=True)
    t_start    = time.time()
    t_step     = time.perf_counter()
    ema_reward = 0.0
    ema_reg    = 0.0
    for step in range(steps):
        # one REINFORCE step
        p_idx = int(rng.integers(len(train_states)))
        state = train_states[p_idx]
        loss, mean_reward, regular_rate = _train_step(
            net, optim, state,
            batch          = batch,
            beta           = beta,
            invalid_reward = invalid_reward,
            device         = device,
        )

        ema_reward = EMA_LOG_BETA*ema_reward + (1-EMA_LOG_BETA)*mean_reward
        ema_reg    = EMA_LOG_BETA*ema_reg    + (1-EMA_LOG_BETA)*regular_rate

        # log
        if step % LOG_FREQ == 0:
            dt = time.perf_counter() - t_step
            t_step = time.perf_counter()
            writer.add_scalar("train/loss",         loss,         step)
            writer.add_scalar("train/mean_reward",  mean_reward,  step)
            writer.add_scalar("train/ema_reward",   ema_reward,   step)
            writer.add_scalar("train/regular_rate", regular_rate, step)
            writer.add_scalar("train/ema_regular",  ema_reg,      step)
            print(f"step {step:>6}  pid={state.pid:>3}  "
                  f"loss {loss:+.4f}  "
                  f"reward_ema {ema_reward:+.3f}  "
                  f"regular_ema {ema_reg:.3f}  "
                  f"dt {dt:.2f}s", flush=True)

        # validate
        if step > 0 and step % val_every == 0:
            _eval_pass(
                net, val_states,
                Ntriangs=val_Ntriangs, Npolys=val_Npolys,
                rng=rng, writer=writer, step=step, device=device,
            )

        # in-loop ckpt
        if step > 0 and step % ckpt_every == 0:
            ckpt_path = run_path / f"ckpt_{step:07d}.pt"
            save_ckpt(ckpt_path,
                      net=net, optim=optim,
                      step=step, hparams=ckpt["hparams"])
            print(f"  [ckpt] {ckpt_path}", flush=True)

    # always save final
    # -----------------
    final_path = run_path / f"ckpt_{steps:07d}.pt"
    save_ckpt(final_path,
              net=net, optim=optim,
              step=steps, hparams=ckpt["hparams"])
    print(f"  [ckpt] {final_path}", flush=True)
    print(f"\n[reinforce] done ({time.time() - t_start:.0f}s)", flush=True)
    writer.close()

# RL polygon state
# ================
@dataclass
class RLPolyState:
    """
    Per-polygon state for REINFORCE. Wraps a `DualGraph` (the candidate complex)
    with:

    1) a canonical-FRT -> pool-index lookup (`key_to_idx`) so AR draws can
       be matched against the polygon's FRT pool in O(1) for the KL-vs-
       uniform validation metric; and
    2) torch-tensor copies of the graph arrays so the inner training loop
       can move them to device with one `.to(device)` per step instead of
       a fresh `from_numpy` round-trip.

    Pool membership (`key_to_idx`) is the only thing not derivable from
    `cmplx`; everything else is convenience pre-bake.
    """
    pid:           int
    cmplx:         DualGraph
    key_to_idx:    dict[bytes, int]
    circ_features: torch.Tensor
    edge_indices:  torch.Tensor
    compat:        torch.Tensor

def build_rl_poly_state(row, src_run) -> RLPolyState:
    """
    Build an `RLPolyState` for one polygon: derive its `DualGraph` from
    `row["pts"]`, build the canonical-FRT -> pool-index lookup from the
    polygon's FRT parquet, and pre-bake the torch-tensor copies of the graph
    arrays.

    Parameters
    ----------
    row : dict-like
        One row of `polygons.parquet`, with keys `"id"` and `"pts"`.
    src_run : Path
        Run directory containing `fts/poly_{id:04d}.parquet`.
    """
    # read inputs
    pid   = int(row["id"])
    pts   = np.asarray(row["pts"], dtype=np.int64)
    fts   = pl.read_parquet(str(src_run / "fts" / f"poly_{pid:04d}.parquet"))

    # construct RLPolyState
    cmplx = DualGraph(pts)
    
    key_to_idx: dict[bytes, int] = {}
    for i, s in enumerate(fts["simps"]):
        key = canonical_simps(
            np.asarray(s.to_list(), dtype=np.int8)
        ).tobytes()
        key_to_idx[key] = i

    return RLPolyState(
        pid           = pid,
        cmplx         = cmplx,
        key_to_idx    = key_to_idx,
        circ_features = torch.from_numpy(cmplx.circ_features).float(),
        edge_indices  = torch.from_numpy(cmplx.edges),
        compat        = torch.from_numpy(cmplx.simp_compat),
    )

# training helpers
# ================
def _train_step(
    net,
    optim,
    state: RLPolyState,
    *,
    batch:          int,
    beta:           float,
    invalid_reward: float,
    device:         str,
) -> tuple[float, float, float]:
    """
    One REINFORCE update step on polygon `state`.

    Parameters
    ----------
    net : DualGNN
        Model in train mode; updated in place.
    optim : torch.optim.Optimizer
        Optimizer holding `net.parameters()`.
    state : RLPolyState
        Per-polygon state (graph tensors, FRT pool lookup).

    batch : int
        Trajectories per gradient step.
    beta : float
        Inverse temperature for the rollout (see `ar_rollout_batch`).
    invalid_reward : float
        Reward assigned to non-regular draws.
    device : str
        Target device.

    Returns
    -------
    loss : float
    mean_reward : float
    regular_rate : float
        Fraction of `batch` rollouts that were regular.
    """
    net.train()
    ef = state.circ_features.to(device, non_blocking=True)
    ei = state.edge_indices.to(device, non_blocking=True)
    cm = state.compat.to(device, non_blocking=True)
    placed, log_probs_sum = ar_rollout_batch(
        net,
        batch           = batch,
        N_simps_per_ft  = state.cmplx.N_simps_per_ft,
        circ_features   = ef,
        edge_indices    = ei,
        compat          = cm,
        device          = device,
        beta            = beta,
        track_log_probs = True,
    )

    # regularity check
    placed_np    = placed.detach().cpu().numpy()
    regular_mask = np.zeros(batch, dtype=bool)
    for b in range(batch):
        simp_idxs = np.where(placed_np[b])[0]
        regular_mask[b] = is_regular(state.cmplx.pts,
                                     state.cmplx.simps[simp_idxs])

    # reward: -log P if valid, else INVALID_REWARD
    log_pT    = log_probs_sum.detach()
    regular_t = torch.from_numpy(regular_mask).to(device)
    rewards_t = torch.where(
        regular_t, -log_pT,
        torch.full_like(log_pT, invalid_reward),
    )

    # normalize advantage
    adv = rewards_t - rewards_t.mean()
    adv = adv / adv.std().clamp_min(STD_CLAMP_MIN)

    # REINFORCE step
    loss = -(adv.detach() * log_probs_sum).mean()
    optim.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=GRAD_CLIP_NORM)
    optim.step()

    return (
        float(loss.item()),
        float(rewards_t.mean().item()),
        float(regular_mask.mean()),
    )

# diagnostics
# ===========
def _eval_pass(
    net, val_states,
    *,
    Ntriangs: int,
    Npolys:   int,
    rng,
    writer:   SummaryWriter,
    step:     int,
    device:   str,
):
    """
    One validation pass during REINFORCE training.

    Picks `Npolys` random val polygons, calls `kl_per_poly` on each, prints a
    summary, and logs per-polygon `val/poly_{pid}/kl` + `val/poly_{pid}/regular`
    scalars (and `val/mean_kl` across the polys) to TensorBoard.

    Parameters
    ----------
    net : DualGNN
        Model in eval mode; not modified.
    val_states : list[RLPolyState]
        Per-polygon eval states.

    Ntriangs : int
        AR draws per val polygon.
    Npolys : int
        Number of val polygons to evaluate (capped at `len(val_states)`).
    rng : np.random.Generator
        Source of randomness for the polygon-subset pick.
    writer : SummaryWriter
        TB writer.
    step : int
        Current training step, used as the x-axis for the scalars.
    device : str
        Target device.
    """
    net.eval()
    n_pick = min(Npolys, len(val_states))
    picks  = rng.choice(len(val_states), size=n_pick, replace=False)
    kl_list = []
    for j in picks:
        state = val_states[int(j)]
        kl, vrate, n_uniq = kl_per_poly(
            net, state, Ntriangs=Ntriangs, device=device,
        )
        if not np.isnan(kl):
            kl_list.append(kl)
            writer.add_scalar(f"val/poly_{state.pid}/kl",      kl,    step)
            writer.add_scalar(f"val/poly_{state.pid}/regular", vrate, step)
        print(f"  [val pid={state.pid:>3}] regular={vrate:.3f}  "
              f"KL={kl:.4f}  unique={n_uniq}/{Ntriangs}",
              flush=True)
    if kl_list:
        writer.add_scalar("val/mean_kl", float(np.mean(kl_list)), step)

def kl_per_poly(net, state: RLPolyState, Ntriangs, device):
    """
    Per-polygon KL of the AR-sampled distribution vs uniform over the pool.

    Draws `Ntriangs` from the AR sampler, keeps the regular ones, tallies how
    many fall in the polygon's FRT pool (by canonical key, multiplicities
    preserved), and computes the KL divergence of the resulting empirical
    distribution against uniform over the pool.

    Parameters
    ----------
    net : DualGNN
        Model (eval mode).
    state : RLPolyState
        Per-polygon state.

    Ntriangs : int
        Number of triangulations to draw.
    device : str
        Target device.

    Returns
    -------
    kl : float
        KL of observed-vs-uniform; `nan` if no in-pool draws were produced.
    reg_rate : float
        Fraction of draws that were regular.
    n_unique : int
        Number of distinct in-pool FRTs seen.
    """
    out = sample(net, state.cmplx, Ntriangs=Ntriangs, device=device)
    seen_pool: dict[bytes, int] = {}
    n_reg  = 0
    n_pool = 0
    for i in range(Ntriangs):
        if not is_regular(state.cmplx.pts, out[i]):
            continue
        n_reg += 1
        k = out[i].tobytes()
        if k in state.key_to_idx:
            n_pool += 1
            seen_pool[k] = seen_pool.get(k, 0) + 1
    reg_rate = n_reg / Ntriangs
    if n_pool == 0:
        return float("nan"), reg_rate, 0
    counts = np.array(list(seen_pool.values()), dtype=np.float64)
    p      = counts / n_pool
    kl     = float(np.sum(p * np.log(p * len(state.key_to_idx))))
    return kl, reg_rate, len(seen_pool)
