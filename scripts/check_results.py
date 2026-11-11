"""Verify the result tables and rebuild their task-group aggregates."""

import argparse
from collections import defaultdict
import csv
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vector_search_compression.compression import choose_layout

DATA = ROOT / "data"
METRICS = ("ndcg_at_10", "recall_at_10", "mrr_at_10")
MODES = ("both_quantized", "original_docs_quantized_queries")
GROUP_SIZE = {"NanoBEIR": 13, "CQA": 3}
METHODS = ("rtn", "gptq", "gptaq", "awq")


def read(name):
    with (DATA / f"{name}.csv").open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def by_key(rows, fields):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in fields)].append(row)
    return grouped


def macro(rows, fields, scores):
    result = []
    for key, group in sorted(by_key(rows, fields).items()):
        tasks = {row["task"] for row in group}
        require(len(tasks) == len(group), f"Repeated task for {key}")
        group_name = group[0]["group"]
        if len(tasks) != GROUP_SIZE[group_name]:
            continue
        result.append(dict(zip(fields, key, strict=True), tasks=len(tasks), **{
            field: math.fsum(float(row[field]) for row in group) / len(tasks)
            for field in scores
        }))
    return result


def near(a, b):
    return math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-12)


def verify():
    direct = read("direct_per_task")
    direct_fields = ("model", "scheme", "method", "bits", "mode", "payload_bytes", "group")
    direct_group = macro(direct, direct_fields, METRICS)
    require(len(direct) == 1120 and len(direct_group) == 140, "Direct result coverage differs")

    task_rows = read("matched_budget_per_task")
    group_rows = read("matched_budget_group")
    match_fields = ("model", "mode", "method", "budget_bits", "factor_precision", "group")
    sides = tuple(prefix + metric for prefix in ("direct_", "decomposition_", "delta_") for metric in METRICS)
    matched_group = macro(task_rows, match_fields, sides)
    matched = {tuple(row[key] for key in match_fields): row for row in matched_group}
    infeasible = 0
    for row in group_rows:
        key = tuple(row[field] for field in match_fields)
        if row["status"] != "completed":
            require(key not in matched and not row["decomposition_payload_bytes"]
                    and not row["decomposition_ndcg_at_10"]
                    and int(row["original_minimum_bytes"]) > int(row["budget_bytes"]),
                    f"Invalid infeasibility record: {key}")
            infeasible += 1
            continue
        observed = matched.pop(key)
        require(int(row["tasks"]) == observed["tasks"] and all(near(row[m], observed[m]) for m in sides),
                f"Task-group aggregate differs: {key}")
    require(not matched and len(task_rows) == 1152 and infeasible == 48, "Matched budget coverage differs")
    for row in task_rows:
        budget, direct_bytes, factor_bytes = map(int, (
            row["budget_bytes"], row["direct_payload_bytes"], row["decomposition_payload_bytes"]))
        require(factor_bytes <= budget == direct_bytes and budget - factor_bytes == int(row["slack_bytes"]),
                f"Invalid storage comparison: {row['task']}")
        for metric in METRICS:
            direct_value = float(row["direct_" + metric])
            factor_value = float(row["decomposition_" + metric])
            require(0 <= direct_value <= 1 and 0 <= factor_value <= 1
                    and near(row["delta_" + metric], factor_value - direct_value),
                    f"Invalid difference: {row['task']}, {metric}")

    decisions = read("layout_selection")
    for row in decisions:
        selected = choose_layout(*(float(row[k]) for k in
                                  ("float_CE", "block_CE", "float_margin_MSE", "block_margin_MSE")))
        require(selected == row["selected"], "Held-out decision differs")
    scores = read("scoring_comparison_per_task")
    score_fields = ("model", "method", "mode", "direct_budget_bits", "factor_bits", "group", "selected")
    score_columns = tuple(prefix + metric for prefix in
                          ("floating_", "block_", "selected_minus_other_") for metric in METRICS)
    scoring_group = macro(scores, score_fields, score_columns)

    winners = read("method_selection")
    for row in winners:
        best = max(float(row[method]) for method in METHODS)
        methods = [method for method in METHODS if float(row[method]) == best]
        require(set(row["tied_winners"].split("; ")) == set(methods)
                and row["execution_representative"] == methods[0], "Method choice differs")
        allowed_models = set(row["included_models"].split("; "))
        mode = row["mode"]
        bits = row["direct_bits_or_budget"]
        if row["representation"] == "direct":
            candidates = [r for r in direct if r["mode"] == mode and r["bits"] == bits
                          and r["model"] in allowed_models and r["method"] in METHODS]
            score = "ndcg_at_10"
        else:
            candidates = [r for r in task_rows if r["mode"] == mode and r["budget_bits"] == bits
                          and r["model"] in allowed_models
                          and r["factor_precision"] == row["factor_precision"]]
            score = "decomposition_ndcg_at_10"
        for method in METHODS:
            values = [float(r[score]) for r in candidates if r["method"] == method]
            require(len(values) == int(row["cells_per_method"]) and near(math.fsum(values), row[method]),
                    f"Method score differs: {method}, {mode}, {bits}, {row['factor_precision']}")

    supplement = read("supplement_per_task")
    supplement_fields = ("case", "model", "scheme", "mode", "payload_bytes", "group")
    supplement_group = macro(supplement, supplement_fields, METRICS) + read("supplement_reference_group")
    fixed = read("fixed_layout")
    e179 = [r for r in supplement_group if r["case"] == "E179"]
    require(len(e179) == 1 and all(near(e179[0][m], fixed[-1][m]) for m in METRICS), "Fixed-layout mean differs")
    full = read("full_corpus")
    require(len(full) == 33 and {r["dataset"] for r in full} == {"NQ", "FiQA"}, "Full-corpus coverage differs")

    return {"direct_group": direct_group, "matched_budget_group": matched_group,
            "scoring_comparison_group": scoring_group, "supplement_group": supplement_group,
            "full_corpus": full, "fixed_layout": fixed,
            "factor_weighting_ablation": read("factor_weighting_ablation")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write checked tables to this directory")
    args = parser.parse_args()
    tables = verify()
    for name, rows in tables.items():
        print(f"{name}: {len(rows)} rows")
        if args.output:
            args.output.mkdir(parents=True, exist_ok=True)
            with (args.output / f"{name}.csv").open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, rows[0])
                writer.writeheader()
                writer.writerows(rows)


if __name__ == "__main__":
    main()

