"""Draw the per-dataset nDCG@10 differences for Figure 5."""
import argparse
import csv
import math
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D

PANELS = (
    ("2", "4", "Direct 2-bit vs. factorized 4-bit"),
    ("4", "8", "Direct 4-bit vs. factorized 8-bit"),
    ("8", "FP16", "Direct 8-bit vs. factorized FP16"),
)
MODELS = ("BGE-base", "NV-Embed-v2")
METHODS = (
    ("rtn", "RTN", "#008000", "o"),
    ("gptq", "GPTQ", "#F00000", "s"),
    ("gptaq", "GPTAQ", "#BE00BE", "^"),
    ("awq", "AWQ", "#2411F5", "p"),
)
DATASETS = (
    ("NanoClimateFEVER", "ClimateFEVER"),
    ("NanoDBPedia", "DBpedia"),
    ("NanoFEVER", "FEVER"),
    ("NanoFiQA2018", "NanoFiQA"),
    ("NanoHotpotQA", "HotpotQA"),
    ("NanoMSMARCO", "MS MARCO"),
    ("NanoNFCorpus", "NFCorpus"),
    ("NanoNQ", "NanoNQ"),
    ("NanoQuoraRetrieval", "Quora"),
    ("NanoSCIDOCS", "SCIDOCS"),
    ("NanoArguAna", "ArguAna"),
    ("NanoSciFact", "SciFact"),
    ("NanoTouche2020", "Touché"),
    ("CQA_android", "Android"),
    ("CQA_programmers", "Programmers"),
    ("CQA_tex", "TeX"),
)
BOUNDARIES = (-100, -50, -25, -10, -4, -1, 0, 1, 4, 10, 25, 50, 100)
COLORS = (
    "#802B16", "#A94320", "#C96135", "#DF875A", "#EDAC85", "#F7D2BB",
    "#D1E7F2", "#A9D2E6", "#7CB7D4", "#4F99C3", "#2576AA", "#164D7B",
)
def load_data(path):
    designs = {(bits, precision) for bits, precision, _ in PANELS}
    scores = {}
    with path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            design = row["budget_bits"], row["factor_precision"]
            if design not in designs or row["mode"] != "both_quantized":
                continue
            key = (*design, row["task"], row["model"], row["method"])
            assert row["status"] == "completed" and key not in scores, key
            assert row["retained_dense_precision"] == design[1], key
            assert int(row["direct_payload_bytes"]) == int(row["budget_bytes"]), key
            assert int(row["decomposition_payload_bytes"]) <= int(row["budget_bytes"]), key
            direct = float(row["direct_ndcg_at_10"])
            factorized = float(row["decomposition_ndcg_at_10"])
            delta = float(row["delta_ndcg_at_10"])
            assert 0 <= direct <= 1 and 0 <= factorized <= 1, key
            assert math.isclose(delta, factorized - direct, rel_tol=0, abs_tol=1e-12), key
            scores[key] = 100 * delta

    columns = [(model, method) for model in MODELS for method, *_ in METHODS]
    expected = {
        (*design, task, model, method)
        for design in designs for task, _ in DATASETS for model, method in columns
    }
    assert scores.keys() == expected and len(scores) == 384
    return scores, columns


def draw(scores, columns, output):
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

    cmap = ListedColormap(COLORS)
    cmap.set_bad("#FFFFFF")
    norm = BoundaryNorm(BOUNDARIES, cmap.N)
    fig, axes = plt.subplots(
        1, 3, sharey=True, figsize=(6.95, 3.44),
        gridspec_kw={"left": .16, "right": .994, "bottom": .232, "top": .803, "wspace": .083},
    )
    for index, (ax, (bits, precision, title)) in enumerate(zip(axes, PANELS, strict=True)):
        values = np.asarray([
            [scores[bits, precision, task, model, method] for model, method in columns]
            for task, _ in DATASETS
        ])
        assert values.shape == (16, 8) and np.all(np.isfinite(values))
        assert np.min(values) >= BOUNDARIES[0] and np.max(values) <= BOUNDARIES[-1]
        mesh = ax.pcolormesh(
            np.arange(9), np.arange(17), np.ma.masked_equal(values, 0),
            cmap=cmap, norm=norm, edgecolors="white", linewidth=.35, shading="flat",
        )
        ax.set(xlim=(0, 8), ylim=(16, 0), xticks=[])
        ax.set_yticks(np.arange(16) + .5, [name for _, name in DATASETS])
        ax.tick_params(axis="y", length=0, labelsize=7.65, pad=3, labelleft=index == 0)
        ax.axvline(4, color="#333333", lw=.65, zorder=3)
        ax.axhline(13, color="#333333", lw=.65, zorder=3)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("black")
        for model_index, model in enumerate(MODELS):
            ax.text((model_index + .5) / 2, 1.125, model, transform=ax.transAxes,
                    ha="center", va="center", fontsize=8.5)
            for method_index, (_, _, color, marker) in enumerate(METHODS):
                column = 4 * model_index + method_index
                ax.plot(
                    (column + .5) / 8, 1.044, marker=marker, color=color,
                    markerfacecolor="none", markersize=4.3, markeredgewidth=.82,
                    linestyle="none", transform=ax.transAxes, clip_on=False,
                )
        bounds = ax.get_position()
        fig.text((bounds.x0 + bounds.x1) / 2, .184, f"({chr(97 + index)}) {title}",
                 ha="center", va="center", fontsize=8.35)

    first = axes[0].get_position()
    for label, row_center in (("NanoBEIR", 6.5), ("CQA", 14.5)):
        y = first.y1 - first.height * row_center / 16
        fig.text(.021, y, label, rotation=90, rotation_mode="anchor", ha="center",
                 va="center", fontsize=7.6, weight="bold", style="italic")

    handles = [
        Line2D([], [], marker=marker, color=color, markerfacecolor="none",
               markersize=5.3, markeredgewidth=.95, linestyle="none", label=label)
        for _, label, color, marker in METHODS
    ]
    fig.legend(
        handles=handles, ncol=4, loc="upper center", bbox_to_anchor=(.58, .983),
        prop={"family": "serif", "size": 9.2, "style": "italic", "weight": "bold"},
        frameon=False, borderaxespad=0, handlelength=1.4, handletextpad=.45,
        columnspacing=1.65,
    )

    cax = fig.add_axes([.211, .091, .728, .026])
    bar = fig.colorbar(mesh, cax=cax, orientation="horizontal", spacing="uniform",
                       ticks=BOUNDARIES, drawedges=True)
    bar.ax.set_xticklabels([str(value).replace("-", "−") for value in BOUNDARIES])
    bar.ax.tick_params(axis="x", length=2.35, width=.65, pad=1.7, labelsize=7.45)
    bar.ax.minorticks_off()
    bar.dividers.set_color("white")
    bar.dividers.set_linewidth(.37)
    bar.outline.set_linewidth(.65)
    fig.text(.393, .132, "Direct higher", color="#802B16", fontsize=7.7,
             ha="center", va="center")
    fig.text(.757, .132, "Factorized higher", color="#164D7B", fontsize=7.7,
             ha="center", va="center")
    fig.text(.575, .022, "Factorized minus direct nDCG@10 (%)",
             ha="center", va="center", fontsize=9.6, weight="bold", style="italic")

    fig.savefig(output / "fig5_dataset_deltas.pdf")
    fig.savefig(output / "fig5_dataset_deltas.svg")
    fig.savefig(output / "fig5_dataset_deltas.png", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--data", type=Path, default=root / "data" / "matched_budget_per_task.csv")
    parser.add_argument("--output", type=Path, default=root / "figures")
    args = parser.parse_args()
    scores, columns = load_data(args.data)
    draw(scores, columns, args.output)
