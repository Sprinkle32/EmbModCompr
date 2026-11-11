"""Plot direct quantization and the FP32 and FP16 retrieval references."""
import argparse
import csv
from collections import defaultdict
import math
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

FIGURE_WIDTH, FIGURE_HEIGHT = 500.4, 157.68
BITS = (2, 3, 4, 8)
MODE = "both_quantized"
METHODS = (
    ("RTN", "rtn", "#008000", "o"),
    ("GPTQ", "gptq", "#F00000", "s"),
    ("GPTAQ", "gptaq", "#BE00BE", "^"),
    ("AWQ", "awq", "#2411F5", "p"),
)
PANELS = (
    ("BGE-base", "nano13", "NanoBEIR"),
    ("BGE-base", "cqa3", "CQA"),
    ("NV-Embed-v2", "nano13", "NanoBEIR"),
    ("NV-Embed-v2", "cqa3", "CQA"),
)


def load_scores(source):
    grouped = defaultdict(dict)
    groups = {"NanoBEIR": ("nano13", 13), "CQA": ("cqa3", 3)}
    with source.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            scheme = row["scheme"]
            mode = "shared_original" if scheme == "fp32" else MODE
            if row["mode"] != mode:
                continue
            group, _ = groups[row["group"]]
            key = row["model"], scheme, group
            task, value = row["task"], float(row["ndcg_at_10"])
            if task in grouped[key] or not 0 <= value <= 1:
                raise ValueError(f"Invalid direct result: {key}, {task}")
            grouped[key][task] = value
    expected = {
        (model, scheme, group)
        for model, group, _ in PANELS
        for scheme in ("fp32", "fp16", *(f"{method}_direct{bit}"
                                       for _, method, _, _ in METHODS for bit in BITS))
    }
    if set(grouped) != expected:
        raise ValueError("Incomplete direct quantization results")
    scores = {}
    for key, values in grouped.items():
        size = 13 if key[-1] == "nano13" else 3
        if len(values) != size:
            raise ValueError(f"Incomplete task group: {key}")
        scores[key] = 100 * math.fsum(values.values()) / size
    return scores


def draw(scores, output):
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

    fig, axes = plt.subplots(
        1, len(PANELS), sharey=True, figsize=(FIGURE_WIDTH/72, FIGURE_HEIGHT/72),
        gridspec_kw={"left": .084, "right": .991, "bottom": .278, "top": .835, "wspace": .11},
    )
    x = list(range(len(BITS)))
    for panel, (ax, (model, metric, group)) in enumerate(zip(axes, PANELS, strict=True)):
        fp32 = scores[model, "fp32", metric]
        fp16 = scores[model, "fp16", metric]
        full = ax.axhline(fp32, color="#8a8a8a", lw=1.05, zorder=1, label="FP32")
        half = ax.axhline(fp16, color="#242424", lw=.8, ls=(0, (2.5, 2.0)),
                         zorder=1.1, label="FP16")
        series = []
        for method, key, color, marker in METHODS:
            values = [scores[model, f"{key}_direct{bit}", metric] for bit in BITS]
            line, = ax.plot(x, values, marker=marker, color=color, markerfacecolor="none",
                            markeredgecolor=color, markersize=5.4, markeredgewidth=.9,
                            lw=1.05, zorder=2, label=method)
            series.append(line)
        ax.set(xlim=(-.22, 3.22), ylim=(-3.3, 80),
               xticks=x, xticklabels=[str(bit) for bit in BITS], yticks=[0, 20, 40, 60, 80])
        ax.tick_params(which="major", top=True, right=True, length=2.65,
                       width=.75, pad=2.0, labelsize=8.8)
        for tick in ax.get_xticklabels()+ax.get_yticklabels():
            tick.set_fontweight("bold")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("black")
        position = ax.get_position()
        fig.text((position.x0+position.x1)/2, .153, f"({chr(97+panel)}) {model} ({group})",
                 ha="center", va="center", fontsize=8.6)
        if panel == 0:
            legend = [*series, full, half]

    fig.text(.020, .554, "nDCG@10 (%)", rotation=90, rotation_mode="anchor", ha="center", va="center",
             fontsize=9.6, weight="bold", style="italic")
    fig.text(.54, .027, "Weight precision (bits)", ha="center", va="center",
             fontsize=9.6, weight="bold", style="italic")
    fig.legend(handles=legend, ncol=len(legend), loc="upper center", bbox_to_anchor=(.54, .985),
               prop={"family": "serif", "size": 9.2, "style": "italic", "weight": "bold"},
               frameon=False, borderaxespad=0, handletextpad=.54, handlelength=2.2,
               columnspacing=1.8, markerscale=1.12)

    fig.savefig(output/"fig3_direct_quantization_clean.pdf")
    fig.savefig(output/"fig3_direct_quantization_clean.png", dpi=320)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--data", type=Path, default=root / "data" / "direct_per_task.csv")
    parser.add_argument("--output", type=Path, default=root / "figures")
    args = parser.parse_args()
    draw(load_scores(args.data), args.output)
