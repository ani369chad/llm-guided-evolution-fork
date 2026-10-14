"""Analyze an LLM-GE Titanic run and compare it with ML and MOGP Pareto fronts.

Put this file in the root of your llm-guided-evolution clone and run, for example:

    uv run python llmge_titanic_analysis.py \
        --global_dir checkpoints/global_data \
        --ml ml_results.csv \
        --mogp mogp_pareto.csv \
        --n_neg 110 --n_pos 69 \
        --out_dir analysis_out

Inputs
  --global_dir  folder holding the global_gen_<g>.pkl files LLM-GE writes each generation
                (search with: find . -name "global_gen_*.pkl")
  --ml          optional CSV with columns label,fpr,fnr (or label,fp,fn), one row per ML model
  --mogp        optional CSV with columns fpr,fnr (or fp,fn), the MOGP Pareto individuals
  --n_neg/--n_pos  number of actual negatives / positives in the holdout (106 and 73 for
                the team's random_state=2600 split). LLM-GE reports FP/FN counts, so these
                turn everything into FPR/FNR rates. The front is closed at (0, 1) and (1, 0),
                the same AUC the team notebook uses (lower is better).

Outputs (in --out_dir)
  generation_stats.csv   per generation counts of new individuals, errors, Pareto changes
  pareto_comparison.png  ML vs MOGP vs LLM-GE fronts on one plot with AUC in the legend
  pareto_by_generation.png  how the LLM-GE front moved each generation
  llmge_pareto.csv       the final LLM-GE Pareto individuals (gene id, fp, fn)
"""

import argparse
import glob
import math
import os
import pickle
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_generations(global_dir):
    files = glob.glob(os.path.join(global_dir, "global_gen_*.pkl"))
    gens = {}
    for f in files:
        m = re.search(r"global_gen_(\d+)\.pkl$", f)
        if not m:
            continue
        with open(f, "rb") as fh:
            data = pickle.load(fh)
        merged = {}
        merged.update(data.get("GLOBAL_DATA_HIST", {}) or {})
        merged.update(data.get("GLOBAL_DATA", {}) or {})
        gens[int(m.group(1))] = (merged, data.get("GLOBAL_DATA_ANCESTRY", {}) or {})
    if not gens:
        raise SystemExit(f"No global_gen_*.pkl files found in {global_dir}")
    return dict(sorted(gens.items()))


def is_valid(fit):
    if fit is None:
        return False
    try:
        vals = [float(x) for x in fit]
    except (TypeError, ValueError):
        return False
    if len(vals) < 2:
        return False
    return all(math.isfinite(v) and abs(v) < 1e9 for v in vals[:2])


def pareto(points):
    """Nondominated (fp, fn) points, both minimized, sorted by fp."""
    pts = sorted(set(points))
    front, best_fn = [], math.inf
    for fp, fn in pts:
        if fn < best_fn:
            front.append((fp, fn))
            best_fn = fn
    return front


def auc(front, n_neg=None, n_pos=None):
    pts = list(front)
    if n_neg is not None and n_pos is not None:
        pts = [(0.0, 1.0)] + pts + [(1.0, 0.0)]
    pts = pareto(pts)
    if len(pts) < 2:
        return float("nan")
    xs = np.array([p[0] for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts], dtype=float)
    # Step-wise area (a point covers everything to its right until the next point)
    return float(np.sum(np.diff(xs) * ys[:-1]))


def step_xy(front, n_neg=None, n_pos=None):
    pts = list(front)
    if n_neg is not None and n_pos is not None:
        pts = [(0.0, 1.0)] + pts + [(1.0, 0.0)]
    pts = pareto(pts)
    xs, ys = [], []
    for i, (x, y) in enumerate(pts):
        if i > 0:
            xs.append(x)
            ys.append(pts[i - 1][1])
        xs.append(x)
        ys.append(y)
    return xs, ys


def read_points(path, n_neg, n_pos):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    if "fpr" in cols and "fnr" in cols:
        pts = list(zip(df[cols["fpr"]].astype(float), df[cols["fnr"]].astype(float)))
    else:
        pts = list(zip(df[cols["fp"]].astype(float), df[cols["fn"]].astype(float)))
        if n_neg and n_pos:
            pts = [(a / n_neg, b / n_pos) for a, b in pts]
    labels = df[cols["label"]].astype(str).tolist() if "label" in cols else [""] * len(pts)
    return pts, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--global_dir", required=True)
    ap.add_argument("--ml", default=None)
    ap.add_argument("--mogp", default=None)
    ap.add_argument("--n_neg", type=int, default=None)
    ap.add_argument("--n_pos", type=int, default=None)
    ap.add_argument("--out_dir", default="analysis_out")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    gens = load_generations(args.global_dir)

    seen = set()
    prev_front = set()
    all_valid = {}
    rows = []
    fronts_by_gen = {}
    for g, (data, ancestry) in gens.items():
        new_ids = [k for k in data if k not in seen]
        seen.update(data.keys())
        n_err = sum(1 for k in new_ids if not is_valid(data[k].get("fitness")))
        n_pending = sum(1 for k in new_ids if data[k].get("fitness") is None)
        mut_types = {}
        for k in new_ids:
            t = (ancestry.get(k, {}).get("MUTATE_TYPE") or ["?"])[-1]
            mut_types[t] = mut_types.get(t, 0) + 1
        for k, v in data.items():
            if is_valid(v.get("fitness")):
                fp, fn = float(v["fitness"][0]), float(v["fitness"][1])
                if args.n_neg and args.n_pos:
                    fp, fn = fp / args.n_neg, fn / args.n_pos
                all_valid[k] = (fp, fn)
        pts_to_id = {}
        for k, p in all_valid.items():
            pts_to_id.setdefault(p, k)
        front = pareto(all_valid.values())
        front_ids = {pts_to_id[p] for p in front}
        fronts_by_gen[g] = front
        new_pareto = len(front_ids - prev_front)
        prev_front = front_ids
        rows.append({
            "generation": g,
            "new_individuals": len(new_ids),
            "errored_or_invalid": n_err,
            "of_which_never_finished": n_pending,
            "error_rate": round(n_err / len(new_ids), 3) if new_ids else float("nan"),
            "pareto_size": len(front),
            "new_pareto_individuals": new_pareto,
            "auc": auc(front, args.n_neg, args.n_pos),
            "mutation_types": ";".join(f"{a}:{b}" for a, b in sorted(mut_types.items())),
        })

    stats = pd.DataFrame(rows)
    stats.to_csv(os.path.join(args.out_dir, "generation_stats.csv"), index=False)
    print(stats.to_string(index=False))

    last = fronts_by_gen[max(fronts_by_gen)]
    pts_to_id = {}
    for k, p in all_valid.items():
        pts_to_id.setdefault(p, k)
    pd.DataFrame([{"gene_id": pts_to_id[p], "fp": p[0], "fn": p[1]} for p in last]).to_csv(
        os.path.join(args.out_dir, "llmge_pareto.csv"), index=False)

    # Comparison plot
    fig, ax = plt.subplots(figsize=(7, 6))
    series = []
    if args.ml:
        ml_pts, ml_labels = read_points(args.ml, args.n_neg, args.n_pos)
        series.append(("ML", pareto(ml_pts), "#1f77b4"))
        for (x, y), lab in zip(ml_pts, ml_labels):
            ax.annotate(lab, (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.scatter(*zip(*ml_pts), color="#1f77b4", alpha=0.35, s=18)
    if args.mogp:
        mg_pts, _ = read_points(args.mogp, args.n_neg, args.n_pos)
        series.append(("MOGP", pareto(mg_pts), "#2ca02c"))
    series.append(("LLM-GE", last, "#d62728"))
    ax.scatter(*zip(*all_valid.values()), color="#d62728", alpha=0.15, s=10, label="LLM-GE all valid")

    for name, front, color in series:
        xs, ys = step_xy(front, args.n_neg, args.n_pos)
        a = auc(front, args.n_neg, args.n_pos)
        ax.plot(xs, ys, color=color, lw=2, label=f"{name} front (AUC={a:.3f})")
        ax.scatter([p[0] for p in front], [p[1] for p in front], color=color, s=30, zorder=3)

    rates = bool(args.n_neg and args.n_pos)
    ax.set_xlabel("False positive rate" if rates else "False positives")
    ax.set_ylabel("False negative rate" if rates else "False negatives")
    ax.set_title("Titanic holdout Pareto fronts: ML vs MOGP vs LLM-GE")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "pareto_comparison.png"), dpi=200)

    # Front per generation
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.get_cmap("viridis", max(len(fronts_by_gen), 2))
    for i, (g, front) in enumerate(fronts_by_gen.items()):
        xs, ys = step_xy(front, args.n_neg, args.n_pos)
        ax.plot(xs, ys, color=cmap(i), lw=1.5, label=f"gen {g}")
    ax.set_xlabel("False positives")
    ax.set_ylabel("False negatives")
    ax.set_title("LLM-GE Pareto front by generation")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "pareto_by_generation.png"), dpi=200)
    print(f"Wrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
