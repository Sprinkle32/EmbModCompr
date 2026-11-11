"""Plot all four quantizers in the three Figure 4 storage comparisons."""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

METHODS = (
    ("rtn", "RTN", "#008000", "o"),
    ("gptq", "GPTQ", "#F00000", "s"),
    ("gptaq", "GPTAQ", "#BE00BE", "^"),
    ("awq", "AWQ", "#2411F5", "p"),
)
TASK_GROUPS = (("NanoBEIR", .23, True), ("CQA", -.23, False))
NEUTRAL = "#333333"
ZERO_STYLE = (0, (2.3, 2.1))


PANELS = (
    (2, "4", "Direct 2-bit vs. factorized 4-bit", (-5, 72), (0, 20, 40, 60)),
    (4, "8", "Direct 4-bit vs. factorized 8-bit", (-7, 1), (-6, -4, -2, 0)),
    (8, "FP16", "Direct 8-bit vs. factorized FP16", (-7, 1), (-6, -4, -2, 0)),
)
MODELS = ("BGE-base", "NV-Embed-v2")
GROUPS = (("NanoBEIR", 13, "#235B9A"), ("CQA", 3, "#9A3B70"))
QUANTIZERS = {"rtn", "gptq", "gptaq", "awq"}


def read_comparisons(path):
    requested = {(str(bits), precision) for bits, precision, *_ in PANELS}
    by_group = defaultdict(dict)
    selected = []
    with path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            design = row["budget_bits"], row["factor_precision"]
            if design not in requested or row["mode"] != "both_quantized":
                continue
            if row["status"] != "completed":
                raise ValueError(f"Incomplete comparison: {design}, {row['model']}, {row['method']}")
            model, group, method = row["model"], row["group"], row["method"]
            key = (*design, model, group)
            if method in by_group[key]:
                raise ValueError(f"Duplicate comparison: {key}, {method}")
            assert int(row["direct_payload_bytes"]) == int(row["budget_bytes"])
            assert int(row["decomposition_payload_bytes"]) <= int(row["budget_bytes"])
            assert row["retained_dense_precision"] == design[1]
            assert int(row["tasks"]) == {group: count for group, count, _ in GROUPS}[group]
            direct = float(row["direct_ndcg_at_10"])
            factorized = float(row["decomposition_ndcg_at_10"])
            difference = float(row["delta_ndcg_at_10"])
            assert 0 <= direct <= 1 and 0 <= factorized <= 1
            assert math.isclose(difference, factorized - direct, rel_tol=0, abs_tol=1e-12)
            by_group[key][method] = 100 * difference
            selected.append(row)

    expected = {
        (*design, model, group)
        for design in requested for model in MODELS for group, _, _ in GROUPS
    }
    assert set(by_group) == expected
    assert len(selected) == 48
    for key, values in by_group.items():
        assert values.keys() == QUANTIZERS
    return selected


def plot(rows, output):
    output.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "font.size": 8.5,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": .8,
        "axes.grid": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "xtick.direction": "in",
        "ytick.direction": "in",
    })
    scores = {
        (r["budget_bits"], r["factor_precision"], r["model"], r["method"], r["group"]):
        100 * float(r["delta_ndcg_at_10"])
        for r in rows
    }

    fig, axes = plt.subplots(
        1, 3, sharey=True, figsize=(6.95, 2.65),
        gridspec_kw={"left": .117, "right": .992, "bottom": .238, "top": .836, "wspace": .125},
    )
    model_centers = (6.9, 2.3)
    method_offsets = (1.5, .5, -.5, -1.5)
    for panel, (ax, settings) in enumerate(zip(axes, PANELS, strict=True)):
        bits, precision, label, limits, ticks = settings
        ax.axhline(4.6, color="#D6D6D6", lw=.55, zorder=0)
        ax.axvline(0, color=NEUTRAL, lw=.85, ls=ZERO_STYLE, zorder=1)
        for model, center in zip(MODELS, model_centers, strict=True):
            for (method, _, color, marker), offset in zip(METHODS, method_offsets, strict=True):
                for group, dodge, filled in TASK_GROUPS:
                    delta = scores.pop((str(bits), precision, model, method, group))
                    assert limits[0] < delta < limits[1]
                    ax.plot(
                        delta, center + offset + dodge,
                        marker=marker, linestyle="none", markeredgecolor=color,
                        markerfacecolor=color if filled else "none",
                        markersize=4.65, markeredgewidth=.85, zorder=2,
                    )

        ax.set(xlim=limits, ylim=(.02, 9.18), xticks=ticks)
        ax.set_yticks(model_centers, MODELS)
        ax.tick_params(axis="x", top=True, length=2.65, width=.75, pad=2.0, labelsize=8.8)
        ax.tick_params(axis="y", left=False, labelleft=panel == 0, length=0, pad=4, labelsize=8.4)
        for tick in ax.get_xticklabels():
            tick.set_fontweight("bold")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("black")
        bounds = ax.get_position()
        fig.text((bounds.x0 + bounds.x1) / 2, .124, f"({chr(97 + panel)}) {label}",
                 ha="center", va="center", fontsize=8.6)
    assert not scores

    method_handles = [
        Line2D([], [], marker=marker, color=color, markerfacecolor="none",
               markersize=5.3, markeredgewidth=.95, linestyle="none", label=label)
        for _, label, color, marker in METHODS
    ]
    group_handles = [
        Line2D([], [], marker="o", color=NEUTRAL, markerfacecolor=NEUTRAL if filled else "none",
               markersize=4.7, markeredgewidth=.9, linestyle="none", label=group)
        for group, _, filled in TASK_GROUPS
    ]
    zero_handle = Line2D([], [], color=NEUTRAL, lw=.85, ls=ZERO_STYLE, label="Equal nDCG@10")
    fig.legend(
        handles=[*method_handles, *group_handles, zero_handle],
        ncol=7, loc="upper center", bbox_to_anchor=(.546, .98),
        prop={"family": "serif", "size": 9.2, "style": "italic", "weight": "bold"},
        frameon=False, borderaxespad=0, handlelength=1.35,
        handletextpad=.35, columnspacing=1.15,
    )
    fig.text(.555, .038, "Factorized minus direct nDCG@10 (%)",
             ha="center", va="center", fontsize=9.6, weight="bold", style="italic")

    fig.savefig(output / "fig4_storage_budget_quantizers_percent.pdf")
    fig.savefig(output / "fig4_storage_budget_quantizers_percent.svg")
    fig.savefig(output / "fig4_storage_budget_quantizers_percent.png", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--data", type=Path, default=root / "data" / "matched_budget_group.csv")
    parser.add_argument("--output", type=Path, default=root / "figures")
    args = parser.parse_args()
    plot(read_comparisons(args.data), args.output)
