#!/usr/bin/env python3
"""Launch the Fig.4 Essential-1K reproduction suite on this machine.

Queue-based scheduler over 4 GPUs. A run is considered done when its
per-round metrics CSV appears in the expected results directory; crashed or
hung runs are retried up to 2 times. Per-run logs: results/fig4/_scripts/logs/.

Protocol (identical to paper + teacher's dev reproduction):
    seed=1 (data split), run=1..N (torch seed), 20 epochs,
    n_init=100, n_query=100, n_round=5, batch 256, simple_loss, fix_evaluation.
IterPert fig4a:  Core-Set + diff_effect + 8 priors, mean_new/max fusion.
Baselines fig4b: kernel_based_active_learning + cross_gene_out.
TypiClust/KMeans: linear_fix_ctrl (run.py maps to diff_effect + add_ctrl,
    matching the authors' original bmdal_reg convention).
Fig4c: single prior per run (8 priors).

Usage:
    python scripts/run_fig4_local.py --only main|fig4c|all  [--dry-run]
"""
import argparse
import glob
import os
import subprocess
import time
from collections import deque
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUN_PY = REPO / "reproduce_repo" / "run.py"
PY = "/home/lihaoran/miniconda3/envs/iterpert_env/bin/python"
RESULTS = REPO / "results"
LOG_DIR = RESULTS / "fig4" / "_scripts" / "logs"
N_GPUS = 4
WATCHDOG_S = 5 * 3600   # a full run is ~2h; 5h is generous
MAX_ATTEMPTS = 3


def base_args(strategy_name, kernel_strategy=None, base_kernel="cross_gene_out",
              use_prior=False, integrate_mode="mean_new", normalize_mode="max",
              single_prior=None):
    a = [
        "--seed", "1", "--device", "cuda",
        "--epoch_per_cycle", "20",
        "--n_init_labeled", "100", "--n_query", "100", "--n_round", "5",
        "--batch_size", "256",
        "--dataset_name", "replogle_k562_essential_1000hvg",
        "--retrain", "--model_name", "GEARS",
        "--strategy_name", strategy_name,
        "--base_kernel", base_kernel,
        "--simple_loss", "--fix_evaluation",
    ]
    if kernel_strategy:
        a += ["--kernel_strategy", kernel_strategy]
    if use_prior:
        a += ["--use_prior", "--integrate_mode", integrate_mode,
              "--normalize_mode", normalize_mode]
        if single_prior:
            a += ["--use_single_prior", "--single_prior", single_prior]
    return a


def make_runs(only):
    runs = []
    if only in ("main", "all"):
        for run in [1, 2, 3]:
            runs.append(("iterpert_r%d" % run, run,
                         base_args("kernel_based_active_learning",
                                   kernel_strategy="Core-Set",
                                   base_kernel="diff_effect", use_prior=True),
                         RESULTS / "fig4" / "essential_1k" / "iterpert" / "runs",
                         "priormean_new_max"))
        for slug, k in [("random", "Random"), ("bald", "BALD"),
                        ("batchbald", "BatchBALD"), ("core_set", "Core-Set"),
                        ("badge", "BADGE"), ("acs_fw", "ACS-FW"),
                        ("lcmd", "LCMD")]:
            for run in ([1, 2] if slug == "random" else [1]):
                runs.append((f"{slug}_r{run}", run,
                             base_args("kernel_based_active_learning",
                                       kernel_strategy=k,
                                       base_kernel="cross_gene_out"),
                             RESULTS / "fig4" / "essential_1k" / "baselines" / slug / "runs",
                             f"_run{{run}}_{k}_cross_gene_out".format(run=run)))
        for slug, strat in [("typiclust", "TypiClust"), ("kmeans", "KMeansSampling")]:
            runs.append((f"{slug}_r1", 1,
                         base_args(strat, base_kernel="linear_fix_ctrl"),
                         RESULTS / "fig4" / "essential_1k" / "baselines" / slug / "runs",
                         f"_run1_diff_effect_{strat}"))
    if only in ("ablation", "all"):
        # fig4a/b ablation: IterPert without the model kernel (prior-only fusion).
        # Same naming as the authors' `prior_onlymean_new_max` runs (notebook cell 6).
        for run in [1, 2, 3]:
            a = base_args("kernel_based_active_learning",
                          kernel_strategy="Core-Set",
                          base_kernel="diff_effect", use_prior=True)
            a += ["--use_prior_only"]
            runs.append((f"iterpert_prior_only_r{run}", run, a,
                         RESULTS / "fig4" / "essential_1k" / "iterpert" / "runs",
                         "_prior_only"))
    if only in ("verif", "all"):
        # verification re-runs for the uncertainty-based baselines, whose
        # selection depends on the trained model's gradient features and is
        # sensitive to torch/hardware differences (see results/LOCAL_VERIFICATION.md).
        for slug, k in [("bald", "BALD"), ("batchbald", "BatchBALD")]:
            for run in [2]:
                runs.append((f"{slug}_r{run}", run,
                             base_args("kernel_based_active_learning",
                                       kernel_strategy=k,
                                       base_kernel="cross_gene_out"),
                             RESULTS / "fig4" / "essential_1k" / "baselines" / slug / "runs",
                             f"_run{{run}}_{k}_cross_gene_out".format(run=run)))
    if only in ("fig4c", "all"):
        for prior in ["pops_kernel", "rpe1_kernel", "esm_kernel", "biogpt_kernel",
                      "node2vec_kernel", "ops_A549_kernel", "ops_HeLa_HPLM_kernel",
                      "ops_HeLa_DMEM_kernel"]:
            runs.append((f"single_{prior}_r1", 1,
                         base_args("kernel_based_active_learning",
                                   kernel_strategy="Core-Set",
                                   base_kernel="diff_effect", use_prior=True,
                                   single_prior=prior),
                         RESULTS / "fig4c" / "single_prior" / prior / "runs",
                         f"_single_{prior}"))
    return runs


def is_done(run_no, out_dir, token=None):
    if token:
        pat = str(out_dir / f"*{token}*_run{run_no}_*_metrics.csv")
    else:
        pat = str(out_dir / f"*_run{run_no}_*_metrics.csv")
    return len(glob.glob(pat)) > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["main", "fig4c", "ablation", "verif", "all"], default="all")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    runs = make_runs(args.only)
    if args.dry_run:
        for i, (name, run, a, out, tok) in enumerate(runs):
            print(f"[slot {i%N_GPUS}] {name} run={run} token={tok} -> {out}")
        print(f"total {len(runs)} runs")
        return

    # (name, run_no, args, out_dir, token, attempts_left)
    queue = deque((name, run, a, out, tok, MAX_ATTEMPTS) for name, run, a, out, tok in runs)
    slots = [None] * N_GPUS          # None or (proc, name, run_no, out_dir, t0)
    logf = open(LOG_DIR / "scheduler.log", "a")
    logf.write(f"\n=== suite start {time.strftime('%F %T')} {len(queue)} queued ===\n")

    def kick(slot):
        if queue:
            name, run, a, out, tok, att = queue.popleft()
            if is_done(run, out, tok):
                logf.write(f"[skip] {name} already on disk\n"); logf.flush()
                kick(slot)
                return
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(slot),
                       PYTHONUNBUFFERED="1")
            with open(LOG_DIR / f"{name}.log", "ab") as fh:
                fh.write(f"\n=== {name} @ {time.strftime('%F %T')} "
                         f"gpu={slot} (attempts left {att}) ===\n".encode())
            proc = subprocess.Popen(
                [PY, "-u", str(RUN_PY)] + a + ["--run", str(run)],
                cwd=RUN_PY.parent, env=env,
                stdout=open(LOG_DIR / f"{name}.log", "ab"),
                stderr=subprocess.STDOUT)
            slots[slot] = (proc, name, run, a, out, tok, att, time.time())
            logf.write(f"[launch] {name} gpu={slot} "
                       f"(remaining {len(queue)})\n"); logf.flush()

    for slot in range(N_GPUS):
        kick(slot)

    while any(s is not None for s in slots):
        for slot in range(N_GPUS):
            s = slots[slot]
            if s is None:
                kick(slot)
                continue
            proc, name, run, a, out, tok, att, t0 = s
            rc = proc.poll()
            if rc is not None:
                ok = is_done(run, out, tok)
                logf.write(f"[exit] {name} rc={rc} metrics={ok} "
                           f"({(time.time()-t0)/60:.0f} min)\n"); logf.flush()
                if not ok and att > 1:
                    queue.appendleft((name, run, a, out, tok, att - 1))
                    logf.write(f"[requeue] {name} attempts left {att-1}\n")
                slots[slot] = None
            elif time.time() - t0 > WATCHDOG_S:
                proc.kill()
                logf.write(f"[watchdog] {name} killed at 5h\n"); logf.flush()
                if att > 1:
                    queue.appendleft((name, run, a, out, tok, att - 1))
                slots[slot] = None
        time.sleep(60)

    logf.write(f"=== suite end {time.strftime('%F %T')} ===\n")
    logf.close()


if __name__ == "__main__":
    main()
