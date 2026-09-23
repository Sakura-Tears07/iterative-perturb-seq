#!/usr/bin/env python3
"""Launch the cell-budget feedback ablation: 4 arms x 3 repeats on 4 GPUs.

Arms: random | static_prior | iterpert_full | iterpert_frozen
Shared: data split seed=1, initial set (initialize_labels seed 42), 100 cells/pert,
        100 initial perts + 3 queries x 100 perts, 20 epochs, hidden 64.
Each arm's Random query seed is independent (1000+run).
"""
import json, os, subprocess, sys, time
from collections import deque
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = '/home/lihaoran/miniconda3/envs/iterpert_env/bin/python'
RES = Path('/data/lhr/ai4s/iterpert/scratch/perturb_seq_data/gears_data/res')
LOGD = REPO / 'results' / 'cellbudget' / 'logs'
ARMS = ['random', 'static_prior', 'iterpert_full', 'iterpert_frozen']
RUNS = [1, 2, 3]
N_GPUS, WATCHDOG = 4, 4 * 3600


def done(tag, n_round):
    f = RES / f'{tag}_metrics.json'
    if not f.exists():
        return False
    try:
        d = json.load(open(f))
        return any(r.get('round') == n_round for r in d['curve'])
    except Exception:
        return False


def main():
    LOGD.mkdir(parents=True, exist_ok=True)
    jobs = deque((a, r) for r in RUNS for a in ARMS)
    slots = [None] * N_GPUS
    logf = open(LOGD / 'scheduler.log', 'a')
    logf.write(f'\n=== cellbudget suite start {time.strftime("%F %T")} {len(jobs)} jobs ===\n')

    def kick(slot):
        while jobs:
            arm, run = jobs.popleft()
            tag = f'cb_{arm}_r{run}'
            if done(tag, 3):
                logf.write(f'[skip] {tag} already complete\n'); continue
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(slot), PYTHONUNBUFFERED='1')
            fh = open(LOGD / f'{tag}.log', 'ab')
            fh.write(f'\n=== {tag} @ {time.strftime("%F %T")} gpu={slot} ===\n'.encode())
            p = subprocess.Popen([PY, '-u', str(REPO / 'scripts' / 'cellbudget_loop.py'),
                                  '--arm', arm, '--run', str(run), '--cells', '100',
                                  '--n_init', '100', '--n_round', '3', '--n_query', '100',
                                  '--epochs', '20', '--tag', tag],
                                 cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT)
            slots[slot] = (p, tag, time.time())
            logf.write(f'[launch] {tag} gpu={slot} (remaining {len(jobs)})\n'); logf.flush()
            return
        slots[slot] = None

    for s in range(N_GPUS):
        kick(s)
    while any(s is not None for s in slots):
        for s in range(N_GPUS):
            if slots[s] is None:
                kick(s); continue
            p, tag, t0 = slots[s]
            rc = p.poll()
            if rc is not None:
                ok = done(tag, 3)
                logf.write(f'[exit] {tag} rc={rc} complete={ok} ({(time.time()-t0)/60:.0f} min)\n')
                logf.flush(); slots[s] = None
            elif time.time() - t0 > WATCHDOG:
                p.kill(); logf.write(f'[watchdog] {tag} killed at 4h\n'); logf.flush()
                slots[s] = None
        time.sleep(60)
    logf.write(f'=== cellbudget suite end {time.strftime("%F %T")} ===\n'); logf.close()


if __name__ == '__main__':
    main()
