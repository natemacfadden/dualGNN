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
# Description:  Multi-polygon DualGNN trainer with curriculum + exploration.
#               Each step trains the next-simp conditional on a random
#               polygon / FRT / partial-subset, with k_min annealed from
#               `n_simps - gap_start` down to 0. Periodically explores via AR
#               sampling to add novel FRTs to the pool.
#
# Sections (in order):
#   1) TrainConfig + train() entry point
#   2) Trainer class (the main loop, eval, explore, ckpt resume/save)
#   3) Per-polygon state: PolyState + load_poly_state
#   4) Run-local data staging: setup_local_run_data + save_polygon
#   5) Batch construction: make_batch
#   6) Exploration: explore_polygon
#   7) Schedules: lr_schedule + k_min_schedule
#   8) Loss: cross_entropy + target_entropy
# -----------------------------------------------------------------------------

# external imports
from __future__ import annotations

import json
import math
import random
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# local imports
from ..dualgraph         import DualGraph
from ..geometry          import (
    canonical_simps, is_regular, random_lattice_polygon,
)
from ..model             import DualGNN
from ..sampler           import sample, compute_legal
from .harvest            import bootstrap_fts
from .io                 import (
    fts_path, save_fts, load_polygons, save_ckpt, load_ckpt,
)
from .hparams            import (
    POLYGONS_PARQUET, FTS_DIR, VAL_FRAC, VAL_POLY_FRAC,
)
from .target_conditional import SimpConditional


# small constants (not user-facing config)
# ========================================
NPTS_MIN = 5   # range for fresh polygon generation during explore
NPTS_MAX = 40

def _default_device() -> str:
    """Best available accelerator: `"cuda"` -> `"mps"` -> `"cpu"`."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# =============================================================================
# 1) TrainConfig + train() entry point
# =============================================================================
@dataclass
class TrainConfig:
    """
    All user-facing training hyperparameters. Single source of truth.
    """

    # data
    run_dir:        str = "runs/default"
    poly_ids:       list[int] | None = None
    src_polygons:   Path = field(default_factory=lambda: POLYGONS_PARQUET)
    src_fts_dir:    Path = field(default_factory=lambda: FTS_DIR)

    # training
    n_steps:        int   = 30_000
    batch_size:     int   = 32
    lr:             float = 1e-3
    warmup_steps:   int   = 200
    weight_decay:   float = 0.01
    grad_clip:      float = 1.0

    # eval / ckpt
    eval_every:     int = 250
    eval_batch:     int = 256
    ckpt_every:     int = 2500
    log_every:      int = 10     # per-step TB scalars; 0 disables them

    # model
    d_model:        int = 32
    k_rounds:       int = 16

    # curriculum + first-time pool
    gap_start:      int = 2      # start with k_min = N_simps - 2
    grow2d_target:  int = 10_000

    # explore
    explore_every:              int   = 500         # 0 disables
    explore_per_round:          int   = 100
    explore_beta:               float = 1.0
    explore_new_frac:           float | None = None # auto: 0.0 if poly_ids
    explore_new_grow2d_target:  int   = 2000
    explore_new_max_npts_full_enum: int   = 0       # 0 forces grow2d

    # split
    val_frac:       float = VAL_FRAC

    # runtime
    seed:           int  = 0
    device:         str  = field(default_factory=_default_device)
    compile_model:  bool | None = None  # auto: True iff len(poly_ids)==1

    def __post_init__(self):
        if self.explore_new_frac is None:
            self.explore_new_frac = 0.0 if self.poly_ids is not None else 0.20

        if self.compile_model is None:
            self.compile_model = (self.poly_ids is not None
                                  and len(self.poly_ids) == 1)


def train(cfg: TrainConfig | None = None, **kwargs):
    """
    Thin wrapper around `Trainer`. Either pass a fully-built `TrainConfig`
    as `cfg`, or pass field overrides as kwargs (forwarded to a default
    `TrainConfig` constructor).

    Parameters
    ----------
    cfg : TrainConfig, optional
        Full training configuration. Default `None`, in which case `kwargs`
        is used to build one.
    **kwargs : Any
        Field overrides for the default `TrainConfig`. Ignored when `cfg` is
        given.
    """
    Trainer(cfg or TrainConfig(**kwargs)).run()


# =============================================================================
# 2) Trainer class
# =============================================================================
class Trainer:
    """
    Multi-polygon DualGNN trainer.

    Each step picks a polygon, samples a partial FRT, and trains the model to
    match the empirical next-simp conditional. Curriculum lowers `k_min` from
    `N_simps - gap_start` to 0 over training. Periodically explores via AR
    sampling to add novel FRTs to the pool.

    Configuration via `TrainConfig` (see that class for the full list).
    """
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg

        if cfg.compile_model:
            print("[run] compile_model enabled "
                  "(pass --no-compile to disable)")
        if cfg.poly_ids is not None:
            print(f"[run] poly_ids={cfg.poly_ids}  "
                  f"explore_new_frac={cfg.explore_new_frac}")

        # rng seeds
        # ---------
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)

        # data
        # ----
        self.run_path = Path(cfg.run_dir)
        self.run_path.mkdir(parents=True, exist_ok=True)

        # localize data into run dir
        self.run_polygons, self.run_fts_dir = setup_local_run_data(
            run_dir=self.run_path, poly_ids=cfg.poly_ids,
            src_polygons=cfg.src_polygons, src_fts_dir=cfg.src_fts_dir,
        )

        # working-set IDs (train-eligible: role='train')
        df = pl.read_parquet(self.run_polygons)
        self.train_ids    = df.filter(pl.col("role") == "train")["id"].to_list()
        self.val_ids = (df.filter(pl.col("role") == "val")
                          ["id"].to_list())
        self.all_ids = df["id"].to_list()
        print(f"[run] {len(self.all_ids)} polygons total  "
              f"({len(self.train_ids)} train, "
              f"{len(self.val_ids)} val)")
        if not self.train_ids:
            sys.exit("no train-eligible polygons in run set (all val?)")

        # model + optim
        # -------------
        self.net = DualGNN(D=cfg.d_model, K=cfg.k_rounds).to(cfg.device)
        if cfg.compile_model:
            self.net = torch.compile(self.net, dynamic=True)

        self.optim = torch.optim.AdamW(
            self.net.parameters(), lr=cfg.lr,
            weight_decay=cfg.weight_decay, betas=(0.9, 0.95),
        )

        self.start_step = self._resume_from_latest_ckpt()
        print(f"[model] params={sum(p.numel() for p in self.net.parameters())}"
              f"  device={cfg.device}")

        # hparams + writer
        # ----------------
        self.hparams = {**asdict(cfg), "run_dir": str(self.run_path)}
        (self.run_path / "hparams.json").write_text(
            json.dumps(self.hparams, indent=2, default=str)
        )
        self.writer = SummaryWriter(log_dir=str(self.run_path))

        # runtime state
        # -------------
        self.step        = self.start_step
        self.states: dict[int, PolyState] = {}
        self.rng_pick    = np.random.default_rng(cfg.seed + 17)
        self.rng_train   = np.random.default_rng(cfg.seed + 7)
        self.rng_eval    = np.random.default_rng(cfg.seed + 11)
        self.rng_explore = np.random.default_rng(cfg.seed + 23)

    def run(self):
        """Main training loop. Iterates `cfg.n_steps` gradient steps, with
        periodic validation, exploration, and checkpointing."""
        self.step = self.start_step
        while self.step < self.cfg.n_steps:
            result = self._train_step()
            if result is not None:
                st, pid, loss_item = result
                if (self.step % self.cfg.eval_every == 0
                        or self.step == self.cfg.n_steps - 1):
                    self._eval_pass(st, pid, loss_item)
                if (self.cfg.explore_every > 0 and self.step > 0
                        and self.step % self.cfg.explore_every == 0):
                    self._explore_round()
                if (self.step > self.start_step
                        and self.step % self.cfg.ckpt_every == 0):
                    self._save_ckpt()
            self.step += 1
        self._save_ckpt()
        self.writer.close()

    def _get_state(self, poly_id: int) -> "PolyState | None":
        """Load (and cache) per-polygon state."""
        if poly_id not in self.states:
            st = load_poly_state(
                poly_id, self.run_polygons, self.run_fts_dir,
                device=self.cfg.device,
                grow2d_target=self.cfg.grow2d_target,
            )
            if st is None:
                return None
            self.states[poly_id] = st
        return self.states[poly_id]

    def _train_step(self):
        """
        One gradient step on a randomly-picked train polygon.

        Returns
        -------
        result : tuple or None
            `(st, pid, loss_item)` where `st` is the `PolyState`, `pid` is
            its polygon id, and `loss_item` is the train loss for the step.
            `None` if the batch could not be sampled (e.g. empty FRT pool).
        """
        cur_lr = lr_schedule(self.step, warmup=self.cfg.warmup_steps,
                             total=self.cfg.n_steps, base_lr=self.cfg.lr)
        for pg in self.optim.param_groups:
            pg["lr"] = cur_lr

        # pick a polygon, ensure state loaded
        for _ in range(20):
            pid = int(self.rng_pick.choice(self.train_ids))
            st  = self._get_state(pid)
            if st is not None:
                break
        else:
            sys.exit("could not load any train polygon")

        self.net.train()
        self.optim.zero_grad(set_to_none=True)

        cur_k_min = k_min_schedule(
            self.step, total=self.cfg.n_steps,
            N_simps_per_ft=st.N_simps_per_ft, gap_start=self.cfg.gap_start,
        )
        batch = make_batch(
            cmplx=st.cmplx, simp_compat_t=st.simp_compat_t,
            simp_cond=st.simp_cond_train, rng=self.rng_train,
            batch_size=self.cfg.batch_size, device=self.cfg.device,
            k_min=cur_k_min,
        )
        if batch is None:
            return None
        placed, legal, target = batch

        logits = self.net(st.circ_features_t, st.edge_indices_t,
                          placed, legal)
        loss   = cross_entropy(logits, target)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.net.parameters(), self.cfg.grad_clip,
        )
        self.optim.step()

        # per-step TB scalars + diagnostic computes -- throttled by log_every
        loss_item = loss.item()
        if (self.cfg.log_every > 0
                and self.step % self.cfg.log_every == 0):
            with torch.no_grad():
                h_target = target_entropy(target).item()
            w, s = self.writer, self.step
            w.add_scalar("train/loss",      loss_item,              s)
            w.add_scalar("train/kl",        loss_item - h_target,   s)
            w.add_scalar("train/H_target",  h_target,               s)
            w.add_scalar("train/lr",        cur_lr,                 s)
            w.add_scalar("train/grad_norm", float(grad_norm),       s)
            w.add_scalar("train/k_min",     cur_k_min,              s)
            w.add_scalar("train/poly_id",   pid,                    s)
        return st, pid, loss_item

    @torch.no_grad()
    def _eval_pass(self, train_st, train_pid, train_loss):
        """One validation pass on a randomly-picked polygon. Skips silently
        if the eval polygon fails to load or its val batch is empty."""
        self.net.eval()
        eval_pid = int(self.rng_eval.choice(self.all_ids))
        est = self._get_state(eval_pid)
        if est is None:
            return
        ebatch = make_batch(
            cmplx=est.cmplx, simp_compat_t=est.simp_compat_t,
            simp_cond=est.simp_cond_val, rng=self.rng_eval,
            batch_size=self.cfg.eval_batch, device=self.cfg.device,
            k_min=0,
        )
        if ebatch is None:
            return
        placed_v, legal_v, target_v = ebatch
        logits_v = self.net(est.circ_features_t, est.edge_indices_t,
                            placed_v, legal_v)
        loss_v   = cross_entropy(logits_v, target_v).item()
        h_v      = target_entropy(target_v).item()
        top1_acc = (logits_v.argmax(dim=-1)
                    == target_v.argmax(dim=-1)).float().mean().item()
        w, s = self.writer, self.step
        w.add_scalar("val/loss",     loss_v,        s)
        w.add_scalar("val/kl",       loss_v - h_v,  s)
        w.add_scalar("val/top1_acc", top1_acc,      s)
        print(
            f"step {self.step:>5}  poly={train_pid:>3} "
            f"(eval={eval_pid:>3} role={est.role:<8}) "
            f"loss={train_loss:.4f}  val_loss={loss_v:.4f}  "
            f"top1={top1_acc:.3f}"
        )

    def _explore_round(self):
        """One explore round: AR-sample, classify, add novels to pool."""
        self.net.eval()

        # pick a polygon (maybe a fresh one with prob explore_new_frac)
        est: "PolyState | None" = None
        if (self.cfg.explore_new_frac > 0
                and self.rng_explore.random() < self.cfg.explore_new_frac):
            new_id = self._generate_new_polygon()
            if new_id is not None:
                # explore-time bootstrap: force grow2d with modest target so
                # we don't stall on CYTools enum of a fresh polygon
                est = load_poly_state(
                    new_id, self.run_polygons, self.run_fts_dir,
                    device=self.cfg.device,
                    grow2d_target=self.cfg.explore_new_grow2d_target,
                    max_npts_full_enum=self.cfg.explore_new_max_npts_full_enum,
                )
                if est is not None:
                    self.states[new_id] = est
                    (self.train_ids if est.role == "train"
                     else self.val_ids).append(new_id)
                    self.all_ids.append(new_id)
        if est is None:
            pid = int(self.rng_explore.choice(self.all_ids))
            est = self._get_state(pid)
            if est is None:
                return

        ex_stats = explore_polygon(
            est, self.net,
            n_per_round=self.cfg.explore_per_round, beta=self.cfg.explore_beta,
            device=self.cfg.device, rng=self.rng_explore,
            val_frac=self.cfg.val_frac,
            run_fts_dir=self.run_fts_dir,
        )

        w, s = self.writer, self.step
        w.add_scalar("explore/success", ex_stats["success_rate"], s)
        w.add_scalar("explore/unique",  ex_stats["n_unique"],     s)
        w.add_scalar("explore/novel",   ex_stats["n_novel"],      s)
        w.add_scalar("explore/pool_size", len(est.pool_keys),     s)
        w.add_scalar("explore/n_polys",   len(self.states),       s)
        print(
            f"[explore step={s} poly={est.poly_id} role={est.role}] "
            f"succ={ex_stats['success_rate']:.2f} "
            f"uniq={ex_stats['n_unique']:>3} "
            f"novel={ex_stats['n_novel']:>3}  "
            f"pool={len(est.pool_keys)}",
            flush=True,
        )

    def _generate_new_polygon(self) -> int | None:
        """
        Sample a fresh lattice polygon (same criteria as
        `write_random_polygons`), dedup against the run-local
        polygons.parquet, append, return its new id. `None` on failure.
        """
        df   = pl.read_parquet(self.run_polygons)
        seen = {np.asarray(p.to_list(), dtype=np.int64).tobytes()
                for p in df["pts"]}
        for _ in range(200):
            target_Npts = int(self.rng_explore.integers(
                NPTS_MIN, NPTS_MAX + 1,
            ))
            pts = random_lattice_polygon(
                self.rng_explore, target_Npts=target_Npts,
                Npts_min=NPTS_MIN, Npts_max=NPTS_MAX,
            )
            if pts is None or pts.tobytes() in seen:
                continue
            return save_polygon(
                pts, val=(self.rng_explore.random() < VAL_POLY_FRAC),
                run_polygons=self.run_polygons,
            )
        return None

    def _save_ckpt(self):
        """Save ckpt at `self.step` if not already on disk."""
        ckpt_path = self.run_path / f"ckpt_{self.step:07d}.pt"
        if not ckpt_path.exists():
            save_ckpt(ckpt_path, net=self.net, step=self.step,
                      optim=self.optim, hparams=self.hparams)
            print(f"[ckpt] saved {ckpt_path}")

    def _resume_from_latest_ckpt(self) -> int:
        """Restore net/optim/RNGs from the latest ckpt under `run_path`.
        Returns the resumed step (0 if no ckpt found)."""
        ckpts = sorted(self.run_path.glob("ckpt_*.pt"))
        if not ckpts:
            return 0
        loaded = load_ckpt(ckpts[-1], self.cfg.device)
        self.net.load_state_dict(loaded["state_dict"])
        self.optim.load_state_dict(loaded["optim_state"])
        if loaded.get("rng_state") is not None:
            torch.set_rng_state(loaded["rng_state"].cpu())
        if loaded.get("python_rng_state") is not None:
            random.setstate(loaded["python_rng_state"])
        if loaded.get("numpy_rng_state") is not None:
            np.random.set_state(loaded["numpy_rng_state"])
        step = int(loaded["step"])
        print(f"[resume] from {ckpts[-1]}  step={step}")
        return step


# =============================================================================
# 3) Per-polygon state: PolyState + load_poly_state
# =============================================================================
@dataclass
class PolyState:
    """
    Per-polygon training state. Bundles the polygon's points, its `DualGraph`,
    the FRT pool (split into train and val), per-split `SimpConditional`s, and
    pre-uploaded edge tensors. Initialized via `load_poly_state` and cached in
    `Trainer.states`; subsequently mutated by `explore_polygon` each time it
    adds novel FRTs (the pool grows; `simp_cond_*` are rebuilt).

    Attributes
    ----------
    poly_id : int
        Polygon ID in the run-local `polygons.parquet`.
    role : str
        `"train"` or `"val"`.
    pts : ndarray
        `(Npts, 2)` int64 lattice points.
    cmplx : DualGraph
        Dual-graph candidate complex for this polygon.
    N_simps_per_ft : int
        Number of simps in any FRT of this polygon.
    train_simps : ndarray
        `(Ntrain, N_simps_per_ft, 3)` int64. Train-split FRTs.
    val_simps : ndarray
        `(Nval, N_simps_per_ft, 3)` int64. Val-split FRTs.
    simp_cond_train, simp_cond_val : SimpConditional
        Empirical next-simp conditionals over the train / val pools.
    circ_features_t, edge_indices_t, simp_compat_t : torch.Tensor
        Per-polygon graph tensors pre-uploaded to the training device.
    pool_keys : set[bytes]
        Canonical-form keys of every FRT in the full (train + val) pool.
        Used for novel-FRT deduplication during exploration.
    """
    poly_id:         int
    role:            str
    pts:             np.ndarray
    cmplx:           DualGraph
    N_simps_per_ft:  int
    train_simps:     np.ndarray
    val_simps:       np.ndarray
    simp_cond_train: SimpConditional
    simp_cond_val:   SimpConditional
    circ_features_t: torch.Tensor
    edge_indices_t:  torch.Tensor
    simp_compat_t:   torch.Tensor
    pool_keys:       set = field(default_factory=set)


def load_poly_state(
    poly_id: int, run_polygons: Path, run_fts_dir: Path,
    *, device: str, grow2d_target: int,
    max_npts_full_enum: int | None = None,
) -> PolyState | None:
    """
    Bootstrap (or load) the polygon's FRT pool from the run-local parquets,
    build cmplx + SimpConditionals, and pre-upload edge tensors.

    Parameters
    ----------
    poly_id : int
        Polygon ID in `run_polygons`.
    run_polygons : Path
        Run-local polygons parquet.
    run_fts_dir : Path
        Run-local FRTs directory.

    device : str
        Target device for the edge tensors.
    grow2d_target : int
        Default grow2d target (overridable by callers via the bootstrap
        kwarg).
    max_npts_full_enum : int or None, optional
        `None` -> use the harvest default; pass a small int (e.g. `0`) to
        force grow2d sampling regardless of polygon size -- useful for
        explore-time bootstrap where a fast modest sample beats a multi-minute
        CYTools enumeration. Default `None`.

    Returns
    -------
    state : PolyState or None
        Per-polygon state, or `None` if the polygon has no FRTs at all.
    """
    pts, role = load_polygons(run_polygons, id=poly_id)
    bootstrap_kwargs = dict(
        grow2d_target=grow2d_target, split_seed=poly_id,
    )
    if role == "val":
        bootstrap_kwargs["val_frac"] = 1.0
    if max_npts_full_enum is not None:
        bootstrap_kwargs["max_npts_full_enum"] = max_npts_full_enum
    all_simps, split = bootstrap_fts(
        pts, fts_path(poly_id, run_fts_dir), **bootstrap_kwargs,
    )
    if len(all_simps) == 0:
        print(f"[skip] poly {poly_id}: no FRTs harvested")
        return None

    train_simps = all_simps[split == "train"].astype(np.int64)
    val_simps   = all_simps[split == "val"  ].astype(np.int64)

    cmplx = DualGraph(pts)
    simp_cond_train = SimpConditional(train_simps, cmplx.simps)
    simp_cond_val   = SimpConditional(val_simps,   cmplx.simps)

    pool_keys = {canonical_simps(s).tobytes() for s in all_simps}

    circ_features_t = torch.from_numpy(cmplx.circ_features).float().to(device)
    edge_indices_t  = torch.from_numpy(cmplx.edges).to(device)
    simp_compat_t   = torch.from_numpy(cmplx.simp_compat).to(device)

    return PolyState(
        poly_id=poly_id, role=role, pts=pts, cmplx=cmplx,
        N_simps_per_ft=all_simps.shape[1],
        train_simps=train_simps, val_simps=val_simps,
        simp_cond_train=simp_cond_train, simp_cond_val=simp_cond_val,
        circ_features_t=circ_features_t, edge_indices_t=edge_indices_t,
        simp_compat_t=simp_compat_t,
        pool_keys=pool_keys,
    )


# =============================================================================
# 4) Run-local data staging: setup_local_run_data + save_polygon
# =============================================================================
def setup_local_run_data(
    *, run_dir: Path, poly_ids: list[int] | None,
    src_polygons: Path = POLYGONS_PARQUET, src_fts_dir: Path = FTS_DIR,
):
    """
    Materialize a run-local copy of the global polygons + FRTs.

    If `run_dir/polygons.parquet` is missing, create it from the global set
    filtered by `poly_ids` (`None` = all). Copy any matching FRT parquets
    into `run_dir/fts/`.

    Parameters
    ----------
    run_dir : Path
        Run directory (created if missing).
    poly_ids : list[int] or None
        Subset of polygon IDs to copy in; `None` -> all.
    src_polygons : Path, optional
        Global polygons parquet. Default `POLYGONS_PARQUET`.
    src_fts_dir : Path, optional
        Global FRTs directory. Default `FTS_DIR`.

    Returns
    -------
    run_polygons : Path
        Path to the run-local `polygons.parquet`.
    run_fts_dir : Path
        Path to the run-local `fts/` directory.
    """
    run_dir         = Path(run_dir)
    run_polygons    = run_dir / "polygons.parquet"
    run_fts_dir     = run_dir / "fts"
    run_fts_dir.mkdir(parents=True, exist_ok=True)
    if run_polygons.exists():
        print(f"[run] using existing run-local data at {run_dir}")
        return run_polygons, run_fts_dir

    df = pl.read_parquet(src_polygons)
    if poly_ids is not None:
        df = df.filter(pl.col("id").is_in(poly_ids))
        if df.height == 0:
            raise SystemExit(f"no polygons match poly_ids={poly_ids}")
    df.write_parquet(run_polygons)

    # copy FRT parquets that already exist in src
    for poly_id in df["id"].to_list():
        src_ft = src_fts_dir / f"poly_{poly_id:04d}.parquet"
        if src_ft.exists():
            shutil.copy2(src_ft, run_fts_dir / src_ft.name)
    print(f"[run] staged {df.height} polygons -> {run_dir}")
    return run_polygons, run_fts_dir


def save_polygon(
    pts: np.ndarray, *, val: bool, run_polygons: Path,
) -> int:
    """
    Append a new polygon to the run-local polygons.parquet.

    Parameters
    ----------
    pts : ndarray
        `(Npts, 2)` int. Lattice points of the polygon.

    val : bool
        If True, the polygon is held out for validation (`role = "val"` in
        the parquet); else marked `role = "train"`.
    run_polygons : Path
        Run-local polygons parquet (must exist).

    Returns
    -------
    new_id : int
        ID assigned to the appended row.
    """
    df = pl.read_parquet(run_polygons)
    new_id = int(df["id"].max()) + 1 if df.height > 0 else 0
    new_row = pl.DataFrame(
        {
            "id":    [new_id],
            "n_pts": [len(pts)],
            "role":  ["val" if val else "train"],
            "pts":   [pts.astype(np.int32).tolist()],
        },
        schema={
            "id":    pl.Int32,
            "n_pts": pl.Int32,
            "role":  pl.String,
            "pts":   pl.List(pl.List(pl.Int32)),
        },
    )
    pl.concat([df, new_row]).write_parquet(run_polygons)
    return new_id


# =============================================================================
# 5) Batch construction: make_batch
# =============================================================================
def make_batch(*, cmplx, simp_compat_t, simp_cond, rng, batch_size, device,
               k_min):
    """
    Build one batch from one polygon's pool.

    For each batch element: random FRT -> random k-subset -> next-simp
    conditional target. Conditional is computed on GPU in a single batched
    matmul (see `SimpConditional.conditional_batch`).

    Parameters
    ----------
    cmplx : DualGraph
        Candidate complex for this polygon.
    simp_compat_t : Tensor
        `(Nsimps, Nsimps)` bool on `device`.
    simp_cond : SimpConditional
        Pool of FRTs to draw from.
    rng : np.random.Generator
        Source of randomness.
    batch_size : int
        Target batch size.
    device : str
        Target device.
    k_min : int
        Minimum `k` (number of placed simps) per partial.

    Returns
    -------
    batch : tuple or None
        `(placed, legal, target)` ready for the model, or `None` if no
        valid examples could be sampled.
    """
    Nsimps = cmplx.simps.shape[0]
    pool   = simp_cond.bm
    if len(pool) == 0:
        return None

    placed_rows: list[np.ndarray] = []
    for _ in range(batch_size):
        ft_idx       = int(rng.integers(len(pool)))
        ft_simp_idxs = np.where(pool[ft_idx])[0]
        n_simps      = len(ft_simp_idxs)
        k_lo      = min(k_min, n_simps - 1)
        k         = int(rng.integers(k_lo, n_simps))
        subset    = rng.choice(n_simps, k, replace=False)
        T_partial = np.zeros(Nsimps, dtype=bool)
        T_partial[ft_simp_idxs[subset]] = True
        placed_rows.append(T_partial)

    placed_np = np.stack(placed_rows)
    placed    = torch.from_numpy(placed_np).to(device)
    target    = simp_cond.conditional_batch(placed)
    legal     = compute_legal(placed, simp_compat_t)
    return placed, legal, target


# =============================================================================
# 6) Exploration: explore_polygon
# =============================================================================
def explore_polygon(
    state: PolyState, net, *,
    n_per_round: int, beta: float, device: str,
    rng: np.random.Generator, val_frac: float,
    run_fts_dir: Path,
):
    """
    Run one explore round on `state`'s polygon: AR-sample, drop irregular draws,
    dedup novels against `state.pool_keys`, append novels into`state`'s pools +
    persist the full pool to disk.

    Parameters
    ----------
    state : PolyState
        Per-polygon state (mutated: `train_simps` / `val_simps` /
        `simp_cond_*` / `pool_keys` are updated in place if novels appear).
    net : DualGNN
        Trained-or-training model used for AR rollouts.

    n_per_round : int
        Number of AR draws to make this round.
    beta : float
        Inverse temperature for AR sampling (see `sampler.sample`).
    device : str
        Target device for AR sampling.
    rng : np.random.Generator
        Source of randomness for the per-novel train/val split.
    val_frac : float
        Fraction of novels labeled `"val"` for `role == "train"` polygons.
        For `role == "val"` polygons, all novels are labeled `"val"`.
    run_fts_dir : Path
        Run-local FTs directory; novels are persisted to its per-polygon
        parquet.

    Returns
    -------
    stats : dict
        `{"success_rate": float, "n_unique": int, "n_novel": int}`:
        regular-fraction across `n_per_round` AR draws, unique canonical
        FRTs among the regular draws, and novel FRTs (unique AND not yet
        in `state.pool_keys`).
    """
    try:
        canon = list(sample(
            net, state.cmplx, Ntriangs=n_per_round,
            device=device, beta=beta,
        ))
    except Exception as exc:
        print(f"  [explore] AR sampling failed: {exc}")
        return {"success_rate": 0.0, "n_unique": 0, "n_novel": 0}

    regs  = np.array([is_regular(state.pts, s) for s in canon], dtype=bool)
    n_reg = int(regs.sum())

    new_simps: list[np.ndarray] = []
    new_split: list[str]        = []
    unique_keys: set[bytes]     = set()
    for i, s in enumerate(canon):
        if not regs[i]:
            continue
        key = s.tobytes()
        if key in unique_keys:
            continue
        unique_keys.add(key)
        if key in state.pool_keys:
            continue
        state.pool_keys.add(key)
        new_simps.append(s)
        eff_frac = 1.0 if state.role == "val" else val_frac
        new_split.append("val" if rng.random() < eff_frac else "train")

    stats = {
        "success_rate": n_reg / n_per_round,
        "n_unique":     len(unique_keys),
        "n_novel":      len(new_simps),
    }
    if not new_simps:
        return stats

    new_simps_arr = np.stack(new_simps).astype(np.int8)
    new_split_arr = np.array(new_split, dtype=object)

    state.train_simps = np.concatenate([
        state.train_simps,
        new_simps_arr[new_split_arr == "train"].astype(np.int64),
    ])
    state.val_simps = np.concatenate([
        state.val_simps,
        new_simps_arr[new_split_arr == "val"].astype(np.int64),
    ])
    state.simp_cond_train = SimpConditional(state.train_simps,
                                            state.cmplx.simps)
    state.simp_cond_val   = SimpConditional(state.val_simps,
                                            state.cmplx.simps)

    full_simps = np.concatenate([state.train_simps, state.val_simps])
    full_split = np.array(
        ["train"] * len(state.train_simps) +
        ["val"]   * len(state.val_simps),
        dtype=object,
    )
    save_fts(full_simps.astype(np.int8), full_split,
             fts_path(state.poly_id, run_fts_dir))
    return stats


# =============================================================================
# 7) Schedules: lr_schedule + k_min_schedule
# =============================================================================
def lr_schedule(step, *, warmup, total, base_lr):
    """
    Linear warmup then cosine decay to zero.

    Returns `base_lr * step / warmup` for `step < warmup`, then
    `base_lr * 0.5 * (1 + cos(pi * (step - warmup) / (total - warmup)))`
    until `step >= total` (where it stays at 0).

    Parameters
    ----------
    step : int
        Current training step.

    warmup : int
        Number of linear-warmup steps before cosine decay begins.
    total : int
        Total training steps. Decay reaches 0 at `step == total`.
    base_lr : float
        Peak learning rate (the value reached at the end of warmup).

    Returns
    -------
    lr : float
        Learning rate at `step`.
    """
    if step < warmup:
        return base_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def k_min_schedule(step, *, total, N_simps_per_ft, gap_start):
    """
    Linear anneal of `k_min` from `N_simps_per_ft - gap_start` at step 0 down
    to 0 at step `total`. With `gap_start=2`: starts at 'all but 2' simps
    pre-placed (curriculum makes the first call easy; later steps see fewer
    pre-placed simps).

    Parameters
    ----------
    step : int
        Current training step.

    total : int
        Total training steps. `k_min` reaches 0 at `step == total`.
    N_simps_per_ft : int
        Number of simps in a full FRT for the current polygon.
    gap_start : int
        Gap from N_simps_per_ft to the initial `k_min`. Setting it `>=
        N_simps_per_ft` disables the curriculum (`k_min == 0` throughout).

    Returns
    -------
    k_min : int
        Minimum number of pre-placed simps for batches drawn at `step`.
    """
    start = max(0, N_simps_per_ft - gap_start)
    return max(0, int(round(start * (1.0 - step / max(1, total)))))


# =============================================================================
# 8) Loss: cross_entropy + target_entropy
# =============================================================================
def cross_entropy(logits, target):
    """
    Soft-target cross-entropy: `-sum(target * log_softmax(logits))`, mean-
    reduced over the batch. `nan_to_num` zeros out the `0 * (-inf)` entries
    that arise when `target=0` on an illegal (masked-to -inf) simp.

    Parameters
    ----------
    logits : Tensor
        `(batch_size, Nsimps)` float. Per-simp logits from the model;
        illegal simps are expected to be `-inf`.
    target : Tensor
        `(batch_size, Nsimps)` float. Empirical next-simp probability
        distribution; rows sum to 1 (mass on illegal simps is 0).

    Returns
    -------
    loss : Tensor
        Scalar tensor (with grad) holding the mean cross-entropy.
    """
    log_p = F.log_softmax(logits, dim=-1)
    return -(target * log_p).nan_to_num(0).sum(dim=-1).mean()


def target_entropy(target):
    """
    Shannon entropy of the empirical target distribution (per-row), mean-
    reduced over the batch. Used to convert `cross_entropy` into KL via
    `train/kl = loss - target_entropy`.

    Parameters
    ----------
    target : Tensor
        `(batch_size, Nsimps)` float. Empirical next-simp distribution.

    Returns
    -------
    H : Tensor
        Scalar tensor (no grad) holding the mean entropy.
    """
    return -torch.xlogy(target, target).sum(dim=-1).mean()
