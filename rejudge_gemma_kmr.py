#!/usr/bin/env python3
"""Re-match the target keywords of one victim's stored descriptions with the DeepSeek judge (text only, no GPU).

Companion of rejudge_gemma.py: that script re-scores the semantic similarity, this one the keyword matching.
For the Gemma 4 31B victim the keyword matching was done by Gemma itself; here DeepSeek matches the same
keywords against the same adversarial descriptions with the same prompt as evaluate.py.

    python rejudge_gemma_kmr.py --inputs /path/to/results/keyword_matching/outputs --out /path/to/outputs_deepseek

Needs DEEPSEEK_API_KEY in .env here (or in the environment). Resumes: matched pairs are kept.
"""
import argparse
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from evaluate import KEYWORD_PROMPT, DeepSeekJudge, load_env, parse_keywords, with_retries

HERE = os.path.dirname(os.path.abspath(__file__))
METHODS = ["MAttack", "FOA_Attack", "SOTAttack_fgsm"]


def metrics(kmr):
    n = len(kmr)
    return {"a": 100 * sum(x >= 0.25 for x in kmr) / n, "b": 100 * sum(x >= 0.5 for x in kmr) / n,
            "c": 100 * sum(x >= 1.0 - 1e-9 for x in kmr) / n, "avg": 100 * sum(kmr) / n}


def sign_test(a, b):
    plus, minus = sum(x > y for x, y in zip(a, b)), sum(x < y for x, y in zip(a, b))
    n, k = plus + minus, min(plus, minus)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return plus, minus, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, help="folder with <victim>_benchmark_<method>.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--victim", default="FRT-gemma-4-31B-it")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    load_env(os.path.join(HERE, ".env"))
    judge = DeepSeekJudge()
    os.makedirs(args.out, exist_ok=True)

    own, deep = {}, {}
    for method in METHODS:
        with open(os.path.join(args.inputs, f"{args.victim}_benchmark_{method}.json"), encoding="utf-8") as f:
            pairs = {k: v for k, v in json.load(f).items() if k.isdigit() and "keywords" in v}
        path = os.path.join(args.out, f"{args.victim}_deepseek_kmr_{method}.json")
        done = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
        todo = [k for k in pairs if k not in done]

        def match(k):
            p = pairs[k]
            cleaned = [w.strip().replace("\n", " ") for w in p["keywords"] if w.strip()]
            listing = '["' + '", "'.join(cleaned) + '"]'
            got = with_retries(lambda: parse_keywords(judge.ask(
                KEYWORD_PROMPT.format(description=p["description"].strip(), keywords=listing))), "deepseek keywords")
            wanted = {w.lower(): w for w in cleaned}
            hit = sorted({wanted[w.strip().lower()] for w in got if w.strip().lower() in wanted})
            return k, {"matched": hit, "kmr": len(hit) / len(cleaned)}

        with tqdm(total=len(todo), desc=method, unit="pair") as bar, ThreadPoolExecutor(args.workers) as pool:
            for i, fut in enumerate(as_completed([pool.submit(match, k) for k in todo]), 1):
                k, r = fut.result()
                done[k] = r
                bar.update()
                if i % 25 == 0 or i == len(todo):
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(done, f)
        keys = sorted(pairs, key=int)
        own[method] = [float(pairs[k]["matching_rate"]) for k in keys]
        deep[method] = [float(done[k]["kmr"]) for k in keys]

    n = len(own[METHODS[0]])
    lines = [f"Victim {args.victim}: keyword matching by Gemma 4 31B (self) vs DeepSeek ({n} pairs)", "",
             "| Method | Matcher | KMR_a | KMR_b | KMR_c | Avg KMR |", "|---|---|---|---|---|---|"]
    for method in METHODS:
        for name, table in (("Gemma 4 31B (self)", own), ("DeepSeek", deep)):
            m = metrics(table[method])
            lines.append(f"| {method} | {name} | {m['a']:.1f} | {m['b']:.1f} | {m['c']:.1f} | {m['avg']:.1f} |")
    lines.append("")
    for method in METHODS:
        g, d = own[method], deep[method]
        mg, md = sum(g) / n, sum(d) / n
        cov = sum((x - mg) * (y - md) for x, y in zip(g, d))
        den = math.sqrt(sum((x - mg) ** 2 for x in g) * sum((y - md) ** 2 for y in d))
        agree = sum(abs(x - y) < 1e-9 for x, y in zip(g, d))
        lines.append(f"{method}: Pearson r between matchers over pairs = {cov / den:.3f}; identical KMR on {agree}/{n} pairs")
    lines.append("")
    for name, table in (("Gemma 4 31B (self)", own), ("DeepSeek", deep)):
        for ref in ("FOA_Attack", "MAttack"):
            plus, minus, p = sign_test(table["SOTAttack_fgsm"], table[ref])
            gap = metrics(table["SOTAttack_fgsm"])["avg"] - metrics(table[ref])["avg"]
            lines.append(f"{name}, SOTAttack_fgsm vs {ref}: Avg KMR gap {gap:+.1f} points; sign test +{plus}/-{minus}, p = {p:.2g}")
    text = "\n".join(lines) + "\n"
    print(text)
    with open(os.path.join(args.out, f"{args.victim}_kmr_comparison.md"), "w", encoding="utf-8") as f:
        f.write(text)


if __name__ == "__main__":
    main()
