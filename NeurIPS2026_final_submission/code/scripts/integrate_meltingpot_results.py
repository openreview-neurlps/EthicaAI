"""Generate branch-ready Melting Pot manuscript snippets from result JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any


RESULTS_DIR = Path(__file__).resolve().parents[1] / "outputs" / "meltingpot"
DEFAULT_MERGED = RESULTS_DIR / "meltingpot_final_results_merged.json"
DEFAULT_MAC = RESULTS_DIR / "meltingpot_final_results.json"
DEFAULT_LEGACY = RESULTS_DIR / "meltingpot_cleanup_cnn.json"
DEFAULT_OUTPUT_JSON = RESULTS_DIR / "meltingpot_paper_update.json"
DEFAULT_OUTPUT_MD = RESULTS_DIR / "meltingpot_paper_update.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Integrate Melting Pot results into branch-ready paper snippets."
    )
    parser.add_argument(
        "--cleanup",
        help="Explicit cleanup result JSON. Defaults to merged, then Mac partial, then legacy cleanup JSON.",
    )
    parser.add_argument(
        "--floor",
        type=float,
        default=0.2,
        help="Floor probability to summarize for clean_up.",
    )
    parser.add_argument(
        "--output-json",
        default=str(DEFAULT_OUTPUT_JSON),
        help="Where to write the structured paper-update JSON.",
    )
    parser.add_argument(
        "--output-md",
        default=str(DEFAULT_OUTPUT_MD),
        help="Where to write the markdown summary.",
    )
    return parser.parse_args()


def choose_cleanup_path(explicit_path: str | None) -> Path:
    if explicit_path:
        return Path(explicit_path)
    for candidate in (DEFAULT_MERGED, DEFAULT_MAC, DEFAULT_LEGACY):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No cleanup result JSON found.")


def compute_basic_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": float(mean(values)),
        "std": float(stdev(values)) if len(values) >= 2 else 0.0,
        "n": len(values),
    }


def load_cleanup_summary(path: Path, target_floor: float) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(payload.get("results"), list):
        results = payload["results"]
        baseline = [item for item in results if float(item["floor_prob"]) == 0.0]
        floored = [item for item in results if float(item["floor_prob"]) == float(target_floor)]
        if not baseline or not floored:
            raise ValueError(f"Expected floor=0.0 and floor={target_floor} results in {path}")

        stats = payload.get("statistics", {}) or {}
        return {
            "source": str(path),
            "format": "merged",
            "floor_prob": target_floor,
            "late_train": {
                "baseline_mean": float(stats.get("late_train", {}).get("baseline_mean", mean([item["late_train_mean"] for item in baseline]))),
                "baseline_std": float(stats.get("late_train", {}).get("baseline_std", stdev([item["late_train_mean"] for item in baseline]) if len(baseline) >= 2 else 0.0)),
                "baseline_n": int(stats.get("late_train", {}).get("baseline_n", len(baseline))),
                "floor_mean": float(stats.get("late_train", {}).get("floor_mean", mean([item["late_train_mean"] for item in floored]))),
                "floor_std": float(stats.get("late_train", {}).get("floor_std", stdev([item["late_train_mean"] for item in floored]) if len(floored) >= 2 else 0.0)),
                "floor_n": int(stats.get("late_train", {}).get("floor_n", len(floored))),
                "p_value": stats.get("late_train", {}).get("p_value"),
                "cohens_d": stats.get("late_train", {}).get("cohens_d"),
                "significant_p05": bool(stats.get("late_train", {}).get("significant_p05", False)),
            },
            "eval": {
                "baseline_mean": float(stats.get("eval", {}).get("baseline_mean", mean([item["eval_mean"] for item in baseline]))),
                "baseline_std": float(stats.get("eval", {}).get("baseline_std", stdev([item["eval_mean"] for item in baseline]) if len(baseline) >= 2 else 0.0)),
                "baseline_n": int(stats.get("eval", {}).get("baseline_n", len(baseline))),
                "floor_mean": float(stats.get("eval", {}).get("floor_mean", mean([item["eval_mean"] for item in floored]))),
                "floor_std": float(stats.get("eval", {}).get("floor_std", stdev([item["eval_mean"] for item in floored]) if len(floored) >= 2 else 0.0)),
                "floor_n": int(stats.get("eval", {}).get("floor_n", len(floored))),
                "p_value": stats.get("eval", {}).get("p_value"),
                "cohens_d": stats.get("eval", {}).get("cohens_d"),
                "significant_p05": bool(stats.get("eval", {}).get("significant_p05", False)),
            },
        }

    legacy_results = payload.get("results", {})
    baseline_key = next((key for key in legacy_results if key.endswith("0.0") or key.endswith("_0.0")), None)
    floor_key = next((key for key in legacy_results if f"{target_floor}" in key), None)
    if baseline_key is None or floor_key is None:
        raise ValueError(f"Expected legacy baseline/floor keys in {path}")

    baseline = legacy_results[baseline_key]
    floored = legacy_results[floor_key]
    return {
        "source": str(path),
        "format": "legacy",
        "floor_prob": target_floor,
        "late_train": None,
        "eval": {
            "baseline_mean": float(baseline["mean"]),
            "baseline_std": float(baseline["std"]),
            "baseline_n": len(baseline.get("seeds", baseline.get("per_seed", []))),
            "floor_mean": float(floored["mean"]),
            "floor_std": float(floored["std"]),
            "floor_n": len(floored.get("seeds", floored.get("per_seed", []))),
            "p_value": None,
            "cohens_d": None,
            "significant_p05": False,
        },
    }


def classify_branch(summary: dict[str, Any]) -> tuple[str, str]:
    eval_stats = summary["eval"]
    train_stats = summary["late_train"]

    eval_delta = eval_stats["floor_mean"] - eval_stats["baseline_mean"]
    if train_stats is None:
        if eval_delta > 0:
            return "mixed", "Legacy cleanup result shows a positive evaluation delta, but the richer 25-seed train/eval split is not available yet."
        return "negative", "Legacy cleanup result does not show a reliable positive evaluation delta."

    train_delta = train_stats["floor_mean"] - train_stats["baseline_mean"]
    if train_delta > 0 and eval_delta > 0 and (train_stats["significant_p05"] or eval_stats["significant_p05"]):
        return "positive", "Both late-train and evaluation move upward, with at least one statistically reliable gain."
    if train_delta > 0 or eval_delta > 0:
        return "mixed", "At least one metric improves, but the evidence is not yet clean enough for a strong visual-MARL generalization claim."
    return "negative", "The rerun does not preserve the earlier clean_up gain; the paper should treat this as a boundary-condition result."


def fmt_num(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def fmt_p(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def emphasize(text: str, enabled: bool) -> str:
    return f"\\textbf{{{text}}}" if enabled else text


def build_body_paragraph(summary: dict[str, Any], branch: str) -> str:
    eval_stats = summary["eval"]
    train_stats = summary["late_train"]
    run_label = "the 25-seed rerun" if summary["format"] == "merged" else "the current pilot run"
    eval_clause = (
        f"evaluation reward from {fmt_num(eval_stats['baseline_mean'])} $\\pm$ {fmt_num(eval_stats['baseline_std'])} "
        f"to {fmt_num(eval_stats['floor_mean'])} $\\pm$ {fmt_num(eval_stats['floor_std'])} "
        f"(n={eval_stats['floor_n']}, p={fmt_p(eval_stats['p_value'])})"
    )

    if train_stats is None:
        metric_clause = eval_clause
    else:
        metric_clause = (
            f"late-train reward from {fmt_num(train_stats['baseline_mean'])} $\\pm$ {fmt_num(train_stats['baseline_std'])} "
            f"to {fmt_num(train_stats['floor_mean'])} $\\pm$ {fmt_num(train_stats['floor_std'])} "
            f"(n={train_stats['floor_n']}, p={fmt_p(train_stats['p_value'])}), and {eval_clause}"
        )

    if branch == "positive":
        closing = (
            "This supports a cautious visual-MARL extension of the floor hypothesis, while preserving the negative "
            "boundary condition on non-TPSD \\texttt{commons\\_harvest\\_\\_open}."
        )
    elif branch == "mixed":
        closing = (
            "We therefore use \\texttt{clean\\_up} as a boundary-condition check rather than as a standalone flagship "
            "positive result."
        )
    else:
        closing = (
            "We therefore remove the earlier strong claim and present this benchmark as evidence that floors need both "
            "TPSD structure and sufficient trainability to transfer."
        )

    return (
        f"On \\texttt{{clean\\_up}} (TPSD-like), {run_label} with $\\phi_1{{=}}0.2$ changes "
        f"{metric_clause}. {closing}"
    )


def build_appendix_rows(summary: dict[str, Any]) -> str:
    eval_stats = summary["eval"]
    train_stats = summary["late_train"]
    baseline_train = "n/a"
    floor_train = "n/a"
    if train_stats is not None:
        baseline_train = f"{fmt_num(train_stats['baseline_mean'])} $\\pm$ {fmt_num(train_stats['baseline_std'])}"
        floor_train = f"{fmt_num(train_stats['floor_mean'])} $\\pm$ {fmt_num(train_stats['floor_std'])}"

    baseline_eval = f"{fmt_num(eval_stats['baseline_mean'])} $\\pm$ {fmt_num(eval_stats['baseline_std'])}"
    floor_eval = f"{fmt_num(eval_stats['floor_mean'])} $\\pm$ {fmt_num(eval_stats['floor_std'])}"

    baseline_bold = False
    floor_bold = False
    if train_stats is not None:
        floor_bold = train_stats["floor_mean"] > train_stats["baseline_mean"] or eval_stats["floor_mean"] > eval_stats["baseline_mean"]
        baseline_bold = not floor_bold
    else:
        floor_bold = eval_stats["floor_mean"] > eval_stats["baseline_mean"]
        baseline_bold = not floor_bold

    return "\n".join(
        [
            rf"\multirow{{2}}{{*}}{{clean\_up ({eval_stats['floor_n']} seeds)}} & IPPO (no floor) & {emphasize(baseline_train, baseline_bold)} & {emphasize(baseline_eval, baseline_bold)} & \multirow{{2}}{{*}}{{Yes}} \\",
            rf"  & IPPO + $\phi_1{{=}}{summary['floor_prob']}$ & {emphasize(floor_train, floor_bold)} & {emphasize(floor_eval, floor_bold)} & \\",
        ]
    )


def build_markdown(summary: dict[str, Any], branch: str, reason: str, body_paragraph: str, appendix_rows: str, ready_for_paper: bool) -> str:
    lines = [
        "# Melting Pot Paper Update",
        "",
        f"- Source: `{summary['source']}`",
        f"- Format: `{summary['format']}`",
        f"- Ready for direct paper replacement: `{ready_for_paper}`",
        f"- Decision branch: `{branch}`",
        f"- Reason: {reason}",
        "",
        "## Suggested Body Paragraph",
        "",
        body_paragraph,
        "",
        "## Suggested Appendix Rows",
        "",
        "```tex",
        appendix_rows,
        "```",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    cleanup_path = choose_cleanup_path(args.cleanup)
    summary = load_cleanup_summary(cleanup_path, args.floor)
    branch, reason = classify_branch(summary)
    body_paragraph = build_body_paragraph(summary, branch)
    appendix_rows = build_appendix_rows(summary)
    ready_for_paper = summary["format"] == "merged" and summary["late_train"] is not None

    output_payload = {
        "cleanup_summary": summary,
        "ready_for_paper": ready_for_paper,
        "branch": branch,
        "reason": reason,
        "body_paragraph": body_paragraph,
        "appendix_rows": appendix_rows,
    }

    output_json_path = Path(args.output_json)
    output_json_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")

    output_md_path = Path(args.output_md)
    output_md_path.write_text(
        build_markdown(summary, branch, reason, body_paragraph, appendix_rows, ready_for_paper),
        encoding="utf-8",
    )

    print(f"Cleanup source: {cleanup_path}")
    print(f"Decision branch: {branch}")
    print(f"JSON: {output_json_path}")
    print(f"Markdown: {output_md_path}")


if __name__ == "__main__":
    main()
