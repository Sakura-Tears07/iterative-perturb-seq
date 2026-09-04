# Idea 3b — Detector audit (CPU, same-state).
# Frozen protocol: configs/experiments/idea3_detection/PROTOCOL.md
#
#   python scripts/analyze_prior_reliability_detection.py
from __future__ import annotations

import pickle
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_utils import (
    DATA_ROOT,
    ESSENTIAL_PRIORS,
    REPO_ROOT,
    ensure_analysis_dir,
    load_prior_kernel,
)

PROTOCOL_SRC = REPO_ROOT / "configs" / "experiments" / "idea3_detection" / "PROTOCOL.md"
STATE_PKL_ROOT = DATA_ROOT / "round_states"
DUMP_ROOT = DATA_ROOT / "common_states"
PERM_DIR = DATA_ROOT / "idea3"
PAPER_STEM = (
    "GEARS_100_4_100_256_1_rt_simple_loss_v2_ess_1k_fix_eval_"
    "priormean_new_max_run{run}_Core-Set_diff_effect"
)
RUNS = (1, 2, 3)
ROUND_N = {1: 100, 2: 200, 3: 300, 4: 400}
N500 = 500
LAMBDAS = (0.0, 0.5, 1.0)
PERM_SEEDS = (20260826, 20260827, 20260828, 20260829, 20260830)
PRIMARY_SEED = 20260826
K_NN = 10
OTHER_PRIORS = [p for p in ESSENTIAL_PRIORS if p != "rpe1_kernel"]
SIGNALS = ("obs", "pm", "cons", "loc")
SIGNAL_LABEL = {
    "obs": "Observed KA",
    "pm": "Prior–model KA",
    "cons": "Cross-prior consensus",
    "loc": "Local 10-NN overlap",
    "cons_full": "Consensus (full kernel)",
}


def canon(g: str) -> str:
    g = str(g).strip()
    return g[:-5] if g.endswith("+ctrl") else g


def ka(a: np.ndarray, b: np.ndarray) -> float:
    """Uncentered KA. Uses Frobenius inner product (equals tr(AB) for symmetric A,B)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.linalg.norm(a, "fro") * np.linalg.norm(b, "fro")
    if denom == 0:
        return float("nan")
    return float(np.sum(a * b) / denom)


def mix_rpe1(K: np.ndarray, perm: np.ndarray, lam: float) -> np.ndarray:
    if lam <= 0:
        return K
    k_perm = K[np.ix_(perm, perm)]
    if lam >= 1:
        return k_perm
    return (1.0 - lam) * K + lam * k_perm


def load_or_make_perm(n: int, seed: int) -> np.ndarray:
    PERM_DIR.mkdir(parents=True, exist_ok=True)
    path = PERM_DIR / f"perm_seed{int(seed)}_n{int(n)}.npy"
    if path.is_file():
        perm = np.load(path)
        if int(perm.shape[0]) != int(n):
            raise ValueError(f"{path} has len {perm.shape[0]}, expected {n}")
        return perm
    if int(seed) == PRIMARY_SEED:
        raise FileNotFoundError(f"Refusing to regenerate campaign perm {path}")
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(int(n))
    np.save(path, perm)
    print(f"Saved permutation {path} head={perm[:8].tolist()}")
    return perm


def local_nn_overlap(Ki: np.ndarray, Ky: np.ndarray, k: int = K_NN) -> float:
    n = int(Ki.shape[0])
    k = min(int(k), n - 1)
    if k < 1:
        return float("nan")
    hits = []
    for i in range(n):
        pi = np.array(Ki[i], dtype=float, copy=True)
        yi = np.array(Ky[i], dtype=float, copy=True)
        pi[i] = -np.inf
        yi[i] = -np.inf
        nn_p = set(np.argpartition(pi, -k)[-k:])
        nn_y = set(np.argpartition(yi, -k)[-k:])
        hits.append(len(nn_p & nn_y) / k)
    return float(np.mean(hits))


def load_round_state(run: int, round_id: int) -> dict:
    path = STATE_PKL_ROOT / f"{PAPER_STEM.format(run=run)}_round{round_id}_state.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def dump_labeled_n500(run: int) -> list[str]:
    path = DUMP_ROOT / f"paper_run{run}" / "n500" / "labeled_genes.txt"
    genes = [canon(x) for x in path.read_text().splitlines() if x.strip()]
    if len(genes) != N500:
        raise ValueError(f"{path} has {len(genes)} genes, expected {N500}")
    return genes


def collect_states() -> list[dict]:
    states = []
    for run in RUNS:
        for rd, n in ROUND_N.items():
            obj = load_round_state(run, rd)
            train = [canon(g) for g in obj["train_genes"]]
            if len(train) != n:
                raise ValueError(f"run{run} round{rd}: |S|={len(train)} expected {n}")
            k_model = np.asarray(obj["base_kernel"], dtype=float)
            n_train = len(train)
            if k_model.shape[0] < n_train:
                raise ValueError(f"base_kernel {k_model.shape} smaller than |S|={n_train}")
            states.append({
                "run": run,
                "round": rd,
                "n_labeled": n,
                "genes": train,
                "K_model_SS": k_model[-n_train:, -n_train:],
                "logged_rpe1_obs": float(obj.get("alignments", {}).get("rpe1", np.nan)),
                "logged_model_obs": float(obj.get("alignments", {}).get("model", np.nan)),
            })
        states.append({
            "run": run,
            "round": 5,
            "n_labeled": N500,
            "genes": dump_labeled_n500(run),
            "K_model_SS": None,
            "logged_rpe1_obs": np.nan,
            "logged_model_obs": np.nan,
        })
    return states


def index_of(genes: list[str], idxmap: dict[str, int]) -> np.ndarray:
    missing = [g for g in genes if g not in idxmap]
    if missing:
        raise KeyError(f"{len(missing)} genes not in kernel, e.g. {missing[:5]}")
    return np.array([idxmap[g] for g in genes], dtype=int)


def frac_pos(series: pd.Series) -> tuple[int, int, float]:
    s = series.dropna()
    k = int((s > 0).sum())
    n = int(len(s))
    return k, n, (k / n if n else float("nan"))


def dgate_for_signal(paired: pd.DataFrame, signal: str) -> str:
    sub = paired[paired["signal"] == signal]
    if sub.empty:
        return "incomplete"
    primary = sub[sub["perm_seed"] == PRIMARY_SEED]
    k, n, p = frac_pos(primary["d_z_0_1"])
    need = 0.80
    if n == 0 or np.isnan(p):
        return "incomplete"
    run_ok = True
    for run in RUNS:
        kr, nr, pr = frac_pos(primary.loc[primary["run"] == run, "d_z_0_1"])
        if nr == 0 or pr < need:
            run_ok = False
            break
    seed_ok = True
    for seed in PERM_SEEDS:
        ks, ns, ps = frac_pos(sub.loc[sub["perm_seed"] == seed, "d_z_0_1"])
        if ns == 0 or ps < need:
            seed_ok = False
            break
    k05, n05, p05 = frac_pos(primary["d_z_0_05"])
    extreme_ok = (p >= need) and run_ok and seed_ok
    graded_ok = (not np.isnan(p05)) and (p05 >= need)
    if extreme_ok and graded_ok:
        return "A"
    if extreme_ok and not graded_ok:
        return "B"
    return "C"


def write_plots(long_df: pd.DataFrame, paired: pd.DataFrame, out: Path) -> None:
    primary = long_df[long_df["perm_seed"] == PRIMARY_SEED]
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.6))
    for ax, sig in zip(axes.ravel(), SIGNALS):
        sub = primary[primary["signal"] == sig]
        if sub.empty:
            ax.set_title(f"{SIGNAL_LABEL[sig]} (empty)")
            continue
        data = [sub.loc[sub["lambda"] == lam, "z"].dropna().to_numpy() for lam in LAMBDAS]
        ax.boxplot(data, showfliers=False)
        ax.set_xticklabels(["0", "0.5", "1"])
        rng = np.random.default_rng(0)
        for i, ys in enumerate(data, start=1):
            x = i + rng.uniform(-0.12, 0.12, size=len(ys))
            ax.scatter(x, ys, s=12, alpha=0.55, c="#1f77b4")
        ax.set_title(SIGNAL_LABEL[sig])
        ax.set_xlabel("λ")
        ax.set_ylabel("z")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Same-state detector scores (perm 20260826, 15 states)")
    fig.tight_layout()
    fig.savefig(out / "z_by_lambda.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.6), sharey=True)
    prim_p = paired[paired["perm_seed"] == PRIMARY_SEED]
    for ax, sig in zip(axes.ravel(), SIGNALS):
        sub = prim_p[prim_p["signal"] == sig]
        if sub.empty:
            continue
        for run, g in sub.groupby("run"):
            ax.plot(g["n_labeled"], g["d_z_0_1"], "o-", label=f"run{int(run)}", lw=1.6)
        ax.axhline(0.0, color="0.6", lw=0.8)
        ax.set_title(SIGNAL_LABEL[sig])
        ax.set_xlabel("n labeled")
        ax.set_xticks([100, 200, 300, 400, 500])
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, fontsize=7)
    axes[0, 0].set_ylabel(r"$\Delta z = z_0 - z_1$")
    axes[1, 0].set_ylabel(r"$\Delta z = z_0 - z_1$")
    fig.suptitle("Paired clean vs λ=1 (perm 20260826)")
    fig.tight_layout()
    fig.savefig(out / "paired_delta_by_n.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    rows = []
    for sig in SIGNALS + ("cons_full",):
        for seed in PERM_SEEDS:
            hit = paired[(paired["signal"] == sig) & (paired["perm_seed"] == seed)]
            if hit.empty:
                continue
            _, _, p = frac_pos(hit["d_z_0_1"])
            rows.append({"signal": sig, "perm_seed": seed, "frac": p})
    if rows:
        wide = pd.DataFrame(rows)
        signals = [s for s in SIGNALS + ("cons_full",) if s in set(wide["signal"])]
        x = np.arange(len(signals))
        width = 0.15
        for i, seed in enumerate(PERM_SEEDS):
            ys = [
                float(wide[(wide["signal"] == s) & (wide["perm_seed"] == seed)]["frac"].iloc[0])
                if len(wide[(wide["signal"] == s) & (wide["perm_seed"] == seed)])
                else np.nan
                for s in signals
            ]
            ax.bar(x + (i - 2) * width, ys, width, label=str(seed))
        ax.axhline(0.80, color="#d62728", ls="--", lw=1, label="80% bar")
        ax.set_xticks(x)
        ax.set_xticklabels([SIGNAL_LABEL.get(s, s) for s in signals], rotation=20, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(r"fraction of states with $\Delta z_{0,1}>0$")
        ax.legend(frameon=False, fontsize=7, ncol=2)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "frac_pos_by_perm_seed.png", dpi=160)
    plt.close(fig)


def gate_text(paired: pd.DataFrame, sanity: list[str]) -> str:
    lines = [
        PROTOCOL_SRC.read_text().strip(),
        "",
        "--- live D-Gate ---",
        "",
    ]
    overall = []
    for sig in SIGNALS + ("cons_full",):
        g = dgate_for_signal(paired, sig)
        overall.append(g)
        primary = paired[(paired["signal"] == sig) & (paired["perm_seed"] == PRIMARY_SEED)]
        k1, n1, p1 = frac_pos(primary["d_z_0_1"])
        k5, n5, p5 = frac_pos(primary["d_z_0_05"])
        med1 = float(primary["d_z_0_1"].median()) if len(primary) else float("nan")
        lines.append(
            f"{SIGNAL_LABEL.get(sig, sig):<28} D-Gate {g}   "
            f"Δz(0,1)>0 {k1}/{n1} ({p1:.2f})  median={med1:+.4f}   "
            f"Δz(0,.5)>0 {k5}/{n5} ({p5:.2f})"
        )
        for run in RUNS:
            kr, nr, pr = frac_pos(primary.loc[primary["run"] == run, "d_z_0_1"])
            lines.append(f"    run{run}: {kr}/{nr} ({pr:.2f})")
        seed_bits = []
        for seed in PERM_SEEDS:
            ks, ns, ps = frac_pos(
                paired.loc[(paired["signal"] == sig) & (paired["perm_seed"] == seed), "d_z_0_1"]
            )
            seed_bits.append(f"{seed}:{ks}/{ns}")
        lines.append("    perm seeds Δz(0,1): " + "  ".join(seed_bits))
        lines.append("")

    if "A" in overall:
        lines.append(
            "OVERALL: D-Gate A — at least one pre-registered signal has observable reliability."
        )
        lines.append("Next (later): hard-reject with held-out corruption seed. No softmax yet.")
    elif "B" in overall:
        lines.append(
            "OVERALL: D-Gate B — only severe λ=1 is separable. Future method = hard reject, not continuous weight."
        )
    else:
        lines.append(
            "OVERALL: D-Gate C — no stable observable. Stop adaptive reliability. Do not train a classifier."
        )
    if sanity:
        lines.append("")
        lines.append("Sanity (reproduce logged campaign KA on λ=0, same S):")
        lines.extend(f"  {s}" for s in sanity)
    return "\n".join(lines)


def main() -> None:
    out = ensure_analysis_dir("idea3_detection")
    shutil.copyfile(PROTOCOL_SRC, out / "PROTOCOL.md")

    pert, K_rpe1 = load_prior_kernel("rpe1_kernel")
    pert = [canon(p) for p in pert]
    _, K_y_full = load_prior_kernel("ground_truth_delta")
    idxmap = {g: i for i, g in enumerate(pert)}
    n_all = len(pert)
    others = {name: load_prior_kernel(name)[1] for name in OTHER_PRIORS}

    perms = {seed: load_or_make_perm(n_all, seed) for seed in PERM_SEEDS}
    states = collect_states()
    print(f"{len(states)} common states, {len(perms)} permutation seeds")

    sanity = []
    long_rows = []
    for st in states:
        ix = index_of(st["genes"], idxmap)
        Ky = np.asarray(K_y_full[np.ix_(ix, ix)], dtype=float)
        K_model = st["K_model_SS"]
        if st["n_labeled"] == 100 and st["run"] == 1:
            z0 = ka(K_rpe1[np.ix_(ix, ix)], Ky)
            sanity.append(
                f"run1 n=100 obs λ=0 KA={z0:.6f}  logged={st['logged_rpe1_obs']:.6f}"
            )
            if K_model is not None:
                zm = ka(K_model, Ky)
                sanity.append(
                    f"run1 n=100 model-vs-gold KA={zm:.6f}  logged={st['logged_model_obs']:.6f}"
                )

        for seed, perm in perms.items():
            for lam in LAMBDAS:
                Klam = mix_rpe1(K_rpe1, perm, lam)
                Ki = np.asarray(Klam[np.ix_(ix, ix)], dtype=float)
                z_obs = ka(Ki, Ky)
                z_pm = ka(Ki, K_model) if K_model is not None else np.nan
                z_cons = float(np.mean([
                    ka(Ki, np.asarray(others[name][np.ix_(ix, ix)], dtype=float))
                    for name in OTHER_PRIORS
                ]))
                z_loc = local_nn_overlap(Ki, Ky, K_NN)
                z_cons_full = float(np.mean([
                    ka(Klam, np.asarray(others[name], dtype=float))
                    for name in OTHER_PRIORS
                ]))
                base = {
                    "run": st["run"],
                    "round": st["round"],
                    "n_labeled": st["n_labeled"],
                    "perm_seed": seed,
                    "lambda": lam,
                }
                for sig, val in (
                    ("obs", z_obs),
                    ("pm", z_pm),
                    ("cons", z_cons),
                    ("loc", z_loc),
                    ("cons_full", z_cons_full),
                ):
                    long_rows.append({**base, "signal": sig, "z": val})

    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(out / "detector_long.csv", index=False)

    pair_rows = []
    keys = ["run", "round", "n_labeled", "perm_seed", "signal"]
    for key, g in long_df.groupby(keys):
        z = {float(r["lambda"]): float(r["z"]) for _, r in g.iterrows()}
        z0, z05, z1 = z.get(0.0, np.nan), z.get(0.5, np.nan), z.get(1.0, np.nan)
        pair_rows.append({
            "run": key[0],
            "round": key[1],
            "n_labeled": key[2],
            "perm_seed": key[3],
            "signal": key[4],
            "z0": z0,
            "z05": z05,
            "z1": z1,
            "d_z_0_05": z0 - z05 if np.isfinite(z0) and np.isfinite(z05) else np.nan,
            "d_z_0_1": z0 - z1 if np.isfinite(z0) and np.isfinite(z1) else np.nan,
            "d_z_05_1": z05 - z1 if np.isfinite(z05) and np.isfinite(z1) else np.nan,
            "ordered_0_gt_05_gt_1": (
                np.isfinite(z0) and np.isfinite(z05) and np.isfinite(z1) and (z0 > z05 > z1)
            ),
        })
    paired = pd.DataFrame(pair_rows)
    paired.to_csv(out / "paired_deltas.csv", index=False)

    # compact primary-seed table for the 15 states
    prim = paired[paired["perm_seed"] == PRIMARY_SEED]
    wide = prim.pivot_table(
        index=["run", "n_labeled"],
        columns="signal",
        values="d_z_0_1",
        aggfunc="first",
    ).reset_index()
    wide.to_csv(out / "delta01_primary_seed.csv", index=False)

    write_plots(long_df, paired, out)
    verdict = gate_text(paired, sanity)
    (out / "DGATE.md").write_text(verdict + "\n")

    print(wide.to_string(index=False))
    print()
    print(verdict)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
