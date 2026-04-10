#!/usr/bin/env python3
"""
merge_meltingpot_results.py
===========================

Merge shard-specific Melting Pot result JSON files into one SSOT file and
recompute the final statistics.
"""

import argparse
import json
import os
import numpy as np


def compute_statistics(results):
    """Compute the same Welch-style statistics used in meltingpot_final.py."""
    baseline = [r for r in results if r["floor_prob"] == 0.0]
    floored = [r for r in results if r["floor_prob"] == 0.2]

    if len(baseline) < 2 or len(floored) < 2:
        return {"error": "Not enough seeds to compute statistics"}

    stats = {}
    for metric_key, label in [("late_train_mean", "late_train"),
                              ("eval_mean", "eval")]:
        x = np.array([r[metric_key] for r in floored])
        y = np.array([r[metric_key] for r in baseline])

        nx, ny = len(x), len(y)
        mx, my = np.mean(x), np.mean(y)
        sx, sy = np.std(x, ddof=1), np.std(y, ddof=1)

        se = np.sqrt(sx**2 / nx + sy**2 / ny)
        if se < 1e-12:
            t_stat = 0.0
            p_value = 0.5
        else:
            t_stat = (mx - my) / se
            try:
                from scipy.stats import t as t_dist
                num = (sx**2 / nx + sy**2 / ny) ** 2
                den = (sx**2 / nx)**2 / (nx - 1) + (sy**2 / ny)**2 / (ny - 1)
                df = num / den if den > 0 else nx + ny - 2
                p_value = 1.0 - t_dist.cdf(t_stat, df)
            except ImportError:
                from math import erfc, sqrt
                p_value = 0.5 * erfc(t_stat / sqrt(2))

        pooled_std = np.sqrt(((nx - 1) * sx**2 + (ny - 1) * sy**2)
                             / (nx + ny - 2))
        cohens_d = (mx - my) / pooled_std if pooled_std > 1e-12 else 0.0

        stats[label] = {
            "floor_mean": round(float(mx), 3),
            "floor_std": round(float(sx), 3),
            "floor_n": nx,
            "baseline_mean": round(float(my), 3),
            "baseline_std": round(float(sy), 3),
            "baseline_n": ny,
            "t_stat": round(float(t_stat), 4),
            "p_value": round(float(p_value), 6),
            "cohens_d": round(float(cohens_d), 3),
            "significant_p05": bool(p_value < 0.05),
        }

    return stats


def load_payload(path):
    with open(path, "r") as f:
        return json.load(f)


def save_payload(path, payload):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge shard-specific Melting Pot result JSON files."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input result JSON files. Earlier files win on duplicates.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Merged output JSON path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    merged = []
    seen = {}
    duplicates = []
    source_summary = []
    base_config = None

    for path in args.inputs:
        payload = load_payload(path)
        if base_config is None:
            base_config = payload.get("config", {})

        results = payload.get("results", [])
        source_summary.append({
            "path": path,
            "n_results": len(results),
        })

        for result in results:
            key = (result["seed"], result["floor_prob"])
            if key in seen:
                duplicates.append({
                    "seed": result["seed"],
                    "floor_prob": result["floor_prob"],
                    "kept_from": seen[key],
                    "skipped_from": path,
                })
                continue
            seen[key] = path
            merged.append(result)

    merged.sort(key=lambda r: (r["seed"], r["floor_prob"]))
    stats = compute_statistics(merged)

    output_payload = {
        "experiment": "meltingpot_final_merged",
        "config": base_config or {},
        "source_files": args.inputs,
        "source_summary": source_summary,
        "duplicates_skipped": duplicates,
        "results": merged,
        "statistics": stats,
    }
    save_payload(args.output, output_payload)

    print(f"Merged files: {len(args.inputs)}")
    print(f"Unique results: {len(merged)}")
    print(f"Duplicates skipped: {len(duplicates)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
