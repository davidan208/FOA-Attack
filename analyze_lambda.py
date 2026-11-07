#!/usr/bin/env python3
"""Summarise the lambda / gradient logs written by SOTAttack.py (log.lambda_log, log.diag_grad).

    python analyze_lambda.py <log_dir> [--steps 300] [--early 50] [--out <dir>]

<log_dir> holds lambda.jsonl (and grad_diag.jsonl if the diagnostic ran). Writes summary.md,
summary.json and hist_one_minus_lambda.png to --out (default: <log_dir>).

Rows are deduplicated on (img, t, enc), keeping the last one, so a run resumed after an
interruption is counted once. Statistics are over image x step rows; with batch_size > 1 all
images of a batch share r, lambda and w (they are computed from the batch-mean score).
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np

LAMBDA_ONE_TOL = 1e-6  # r = s(t)/(s(t-1)+1e-8) puts lambda a hair below 1 when r ~ 1


def read_jsonl(path, key_fields):
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                rows[tuple(row[k] for k in key_fields)] = row
    return list(rows.values())


def pct(mask):
    return 100.0 * float(np.mean(mask)) if len(mask) else float("nan")


def lambda_stats(rows):
    lam = np.array([r["lam"] for r in rows])
    rr = np.array([r["r"] for r in rows])
    one_minus = 1.0 - lam
    q = np.percentile(rr, [5, 25, 50, 75, 95]) if len(rr) else [float("nan")] * 5
    return {
        "n": len(rows),
        "mean_1m_lam": float(np.mean(one_minus)),
        "median_1m_lam": float(np.median(one_minus)),
        "pct_lam_eq_1": pct(lam >= 1.0 - LAMBDA_ONE_TOL),
        "pct_lam_exact_1": pct(lam == 1.0),
        "pct_lam_lt_095": pct(lam < 0.95),
        "pct_lam_lt_09": pct(lam < 0.9),
        "pct_lam_lt_0": pct(lam < 0.0),
        "r_q05": float(q[0]), "r_q25": float(q[1]), "r_q50": float(q[2]),
        "r_q75": float(q[3]), "r_q95": float(q[4]),
    }


def weight_stats(rows):
    # One entry per (batch, step): the encoders' weights at that step.
    per_step = defaultdict(dict)
    for r in rows:
        per_step[(r["batch"], r["t"])][r["enc"]] = r["w"]
    spreads, mins, maxs = [], [], []
    for ws in per_step.values():
        v = list(ws.values())
        mins.append(min(v))
        maxs.append(max(v))
        spreads.append(max(v) - min(v))
    return {
        "steps": len(per_step),
        "mean_step_min": float(np.mean(mins)), "mean_step_max": float(np.mean(maxs)),
        "lowest_w": float(np.min(mins)), "highest_w": float(np.max(maxs)),
        "mean_max_minus_min": float(np.mean(spreads)),
        "median_max_minus_min": float(np.median(spreads)),
    }


def diag_stats(rows):
    out = {"n": len(rows)}
    for k in ("flip", "share", "zero_base", "flip_nz"):
        v = np.array([r[k] for r in rows if k in r])
        if len(v):
            out[f"mean_{k}"] = float(np.mean(v))
            out[f"median_{k}"] = float(np.median(v))
    return out


def fmt(x, nd=4):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def table(headers, body):
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in body]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log_dir")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--early", type=int, default=50, help="first N steps vs the rest")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_dir = args.out or args.log_dir
    os.makedirs(out_dir, exist_ok=True)

    rows = read_jsonl(os.path.join(args.log_dir, "lambda.jsonl"), ("img", "t", "enc"))
    diag_path = os.path.join(args.log_dir, "grad_diag.jsonl")
    diag = read_jsonl(diag_path, ("img", "t")) if os.path.exists(diag_path) else []
    encs = sorted({r["enc"] for r in rows}, key=lambda e: [r["enc"] for r in rows].index(e))
    images = sorted({r["img"] for r in rows})

    # Completeness: steps x encoders rows per image.
    expected = args.steps * len(encs)
    per_img = defaultdict(int)
    for r in rows:
        per_img[r["img"]] += 1
    incomplete = {i: c for i, c in per_img.items() if c != expected}

    groups = {e: [r for r in rows if r["enc"] == e] for e in encs}
    groups["pooled"] = rows
    lam = {g: lambda_stats(v) for g, v in groups.items()}
    early = {g: lambda_stats([r for r in v if r["t"] < args.early]) for g, v in groups.items()}
    late = {g: lambda_stats([r for r in v if r["t"] >= args.early]) for g, v in groups.items()}
    w = weight_stats(rows)
    mean_w = {e: float(np.mean([r["w"] for r in groups[e]])) for e in encs}
    d_all = diag_stats(diag) if diag else None
    d_early = diag_stats([r for r in diag if r["t"] < args.early]) if diag else None
    d_late = diag_stats([r for r in diag if r["t"] >= args.early]) if diag else None

    md = [f"# Lambda log summary: `{args.log_dir}`", ""]
    md.append(f"- images: {len(images)} ({images[0]}-{images[-1]})" if images else "- images: 0")
    md.append(f"- rows: {len(rows)} (expected {expected} per image = {args.steps} steps x {len(encs)} encoders)")
    md.append(f"- complete images: {len(images) - len(incomplete)}/{len(images)}"
              + (f"; incomplete: {dict(sorted(incomplete.items())[:10])}" if incomplete else ""))
    md.append(f"- 'lambda = 1' counts lambda >= 1 - {LAMBDA_ONE_TOL:g} (kappa = 1e-8 keeps r ~ 1 a hair below 1)")
    md += ["", "## 1 - lambda and r", ""]
    body = []
    for g in list(encs) + ["pooled"]:
        s = lam[g]
        body.append([g, s["n"], fmt(s["mean_1m_lam"]), fmt(s["median_1m_lam"]),
                     fmt(s["pct_lam_eq_1"], 1), fmt(s["pct_lam_lt_095"], 1), fmt(s["pct_lam_lt_09"], 1),
                     fmt(s["pct_lam_lt_0"], 1),
                     fmt(s["r_q05"]), fmt(s["r_q25"]), fmt(s["r_q50"]), fmt(s["r_q75"]), fmt(s["r_q95"])])
    md.append(table(["encoder", "rows", "mean 1-λ", "median 1-λ", "% λ=1", "% λ<0.95", "% λ<0.9",
                     "% λ<0", "r p5", "r p25", "r p50", "r p75", "r p95"], body))
    md += ["", f"## First {args.early} steps vs steps {args.early}-{args.steps - 1}", ""]
    body = []
    for g in list(encs) + ["pooled"]:
        a, b = early[g], late[g]
        body.append([g, fmt(a["mean_1m_lam"]), fmt(b["mean_1m_lam"]), fmt(a["pct_lam_lt_095"], 1),
                     fmt(b["pct_lam_lt_095"], 1), fmt(a["r_q50"]), fmt(b["r_q50"])])
    md.append(table(["encoder", "mean 1-λ early", "mean 1-λ late", "% λ<0.95 early", "% λ<0.95 late",
                     "median r early", "median r late"], body))
    md += ["", "## Encoder weights w", ""]
    md.append(table(["steps (batch x t)", "mean min w", "mean max w", "lowest w", "highest w",
                     "mean (max-min)", "median (max-min)"],
                    [[w["steps"], fmt(w["mean_step_min"]), fmt(w["mean_step_max"]), fmt(w["lowest_w"]),
                      fmt(w["highest_w"]), fmt(w["mean_max_minus_min"]), fmt(w["median_max_minus_min"])]]))
    md.append("")
    md.append("Mean w per encoder: " + ", ".join(f"{e} {fmt(mean_w[e])}" for e in encs))
    if d_all:
        md += ["", "## Gradient diagnostic (sign of full vs base-only gradient)", ""]
        md.append("flip = share of pixels whose update sign changes because of the multi-view term; "
                  "share = ||g_full - g_base|| / ||g_full||; zero_base = pixels with g_base exactly 0 "
                  "(outside the base crop, so sign(g_base) = 0); flip_nz = flip on pixels with g_base != 0.")
        md.append("")
        body = []
        for name, d in (("all", d_all), (f"t < {args.early}", d_early), (f"t >= {args.early}", d_late)):
            body.append([name, d["n"], fmt(d.get("mean_flip")), fmt(d.get("mean_share")),
                         fmt(d.get("mean_zero_base")), fmt(d.get("mean_flip_nz")),
                         fmt(d.get("median_flip")), fmt(d.get("median_share"))])
        md.append(table(["steps", "rows", "mean flip", "mean share", "mean zero_base", "mean flip_nz",
                         "median flip", "median share"], body))

    text = "\n".join(md) + "\n"
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"lambda": lam, "early": early, "late": late, "weights": w, "mean_w": mean_w,
                   "diag": {"all": d_all, "early": d_early, "late": d_late},
                   "incomplete_images": incomplete}, f, indent=2)
    print(text)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    vals = {e: 1.0 - np.array([r["lam"] for r in groups[e]]) for e in encs}
    hi = max(1e-3, max(float(np.max(v)) for v in vals.values()))
    bins = np.linspace(0.0, hi, 41)
    for e in encs:
        axes[0].hist(vals[e], bins=bins, histtype="step", linewidth=1.5, label=e)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("1 - λ  (weight of the multi-view term)")
    axes[0].set_ylabel("image x step rows (log)")
    axes[0].set_title("All steps")
    axes[0].legend()
    nz = {e: v[v > LAMBDA_ONE_TOL] for e, v in vals.items()}
    for e in encs:
        if len(nz[e]):
            axes[1].hist(nz[e], bins=bins, histtype="step", linewidth=1.5, label=f"{e} ({100 * len(nz[e]) / len(vals[e]):.1f}%)")
    axes[1].set_xlabel("1 - λ, steps with λ < 1 only")
    axes[1].set_title("Steps where the multi-view term is active")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "hist_one_minus_lambda.png"), dpi=200)
    print(f"Wrote summary.md, summary.json, hist_one_minus_lambda.png to {out_dir}")


if __name__ == "__main__":
    main()
