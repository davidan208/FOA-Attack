#!/usr/bin/env python3
"""Re-score stored descriptions of one victim with the DeepSeek judge (text only, no GPU).

Answers: do the semantic results on the Gemma 4 31B victim change when Gemma is not its own judge?
Reads the files of the main experiments (adversarial description, target description, Gemma score per pair),
asks DeepSeek for the similarity with the same prompt as evaluate.py, and compares the two judges.

    python rejudge_gemma.py --inputs /path/to/results/semantic_similarity/outputs --out /path/to/outputs_deepseek

Needs DEEPSEEK_API_KEY in .env here (or in the environment). Resumes: scored pairs are kept.
"""
import argparse
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from evaluate import SIMILARITY_PROMPT, DeepSeekJudge, load_env, parse_score, with_retries

HERE = os.path.dirname(os.path.abspath(__file__))
METHODS = ["MAttack", "FOA_Attack", "SOTAttack_fgsm", "SOTAttack_fgsm_seed2026"]


def metrics(scores):
    n = len(scores)
    return {f"asr{t}": 100 * sum(s >= t / 10 for s in scores) / n for t in (3, 5, 7)} | {"avgsim": sum(scores) / n}


def sign_test(a, b):
    """Two-sided exact sign test on paired scores (ties dropped)."""
    plus, minus = sum(x > y for x, y in zip(a, b)), sum(x < y for x, y in zip(a, b))
    n, k = plus + minus, min(plus, minus)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return plus, minus, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, help="folder with <victim>_similarity_<method>.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--victim", default="FRT-gemma-4-31B-it")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    load_env(os.path.join(HERE, ".env"))
    judge = DeepSeekJudge()
    os.makedirs(args.out, exist_ok=True)

    gemma, deepseek = {}, {}
    for method in METHODS:
        with open(os.path.join(args.inputs, f"{args.victim}_similarity_{method}.json"), encoding="utf-8") as f:
            pairs = {k: v for k, v in json.load(f).items() if isinstance(v, dict) and "adversarial_description" in v}
        path = os.path.join(args.out, f"{args.victim}_deepseek_{method}.json")
        done = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
        todo = [k for k in pairs if k not in done]

        def score(k):
            p = pairs[k]
            return k, with_retries(lambda: parse_score(judge.ask(SIMILARITY_PROMPT.format(
                text1=p["target_description"].strip(), text2=p["adversarial_description"].strip()))), "deepseek similarity")

        with tqdm(total=len(todo), desc=method, unit="pair") as bar, ThreadPoolExecutor(args.workers) as pool:
            for i, fut in enumerate(as_completed([pool.submit(score, k) for k in todo]), 1):
                k, s = fut.result()
                done[k] = s
                bar.update()
                if i % 50 == 0 or i == len(todo):
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(done, f)
        keys = sorted(pairs, key=int)
        gemma[method] = [float(pairs[k]["similarity_score"]) for k in keys]
        deepseek[method] = [float(done[k]) for k in keys]

    lines = [f"Victim {args.victim}: Gemma 4 31B as judge vs DeepSeek as judge ({len(deepseek[METHODS[0]])} pairs)", "",
             "| Method | Judge | ASR@0.3 | ASR@0.5 | ASR@0.7 | AvgSim |", "|---|---|---|---|---|---|"]
    for method in METHODS:
        for name, table in (("Gemma 4 31B", gemma), ("DeepSeek", deepseek)):
            m = metrics(table[method])
            lines.append(f"| {method} | {name} | {m['asr3']:.1f} | {m['asr5']:.1f} | {m['asr7']:.1f} | {m['avgsim']:.3f} |")
    lines.append("")
    for name, table in (("Gemma 4 31B", gemma), ("DeepSeek", deepseek)):
        for sot in ("SOTAttack_fgsm", "SOTAttack_fgsm_seed2026"):
            for ref in ("FOA_Attack", "MAttack"):
                plus, minus, p = sign_test(table[sot], table[ref])
                gap = metrics(table[sot])["asr5"] - metrics(table[ref])["asr5"]
                lines.append(f"{name} judge, {sot} vs {ref}: ASR@0.5 gap {gap:+.1f} points; sign test +{plus}/-{minus}, p = {p:.2g}")
    text = "\n".join(lines) + "\n"
    print(text)
    with open(os.path.join(args.out, f"{args.victim}_judge_comparison.md"), "w", encoding="utf-8") as f:
        f.write(text)


if __name__ == "__main__":
    main()
