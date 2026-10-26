"""由已保存的 T4 BGE 指标生成课程提交用 PNG 图。"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


ROOT = Path(__file__).resolve().parent
FONT = FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc")
LABELS = {"bm25": "BM25", "dense": "BGE + FAISS", "hybrid": "BM25 + BGE (RRF)"}
COLORS = {"bm25": "#2563eb", "dense": "#e76f51", "hybrid": "#2a9d8f"}


def main():
    with (ROOT / "t4_metrics.csv").open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if {row["method"] for row in rows} != set(LABELS) or any(row["questions"] != "20" for row in rows):
        raise ValueError("T4 指标不是预期的三种方法、20 道题")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [1.7, 1]})
    ks = [1, 3, 5, 10]
    for row in rows:
        name = row["method"]
        axes[0].plot(ks, [float(row[f"recall@{k}"]) for k in ks], marker="o", linewidth=2.2,
                     color=COLORS[name], label=LABELS[name])
    axes[0].set_xticks(ks)
    axes[0].set_ylim(0.35, 1.05)
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Recall@k")
    axes[0].set_title("三种检索方法的 Recall@k", fontproperties=FONT, fontsize=13)
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    names = [row["method"] for row in rows]
    values = [float(row["mrr@10"]) for row in rows]
    bars = axes[1].bar([LABELS[name] for name in names], values,
                       color=[COLORS[name] for name in names], width=0.6)
    axes[1].set_ylim(0, 1.0)
    axes[1].set_ylabel("MRR@10")
    axes[1].set_title("MRR@10", fontsize=13)
    axes[1].tick_params(axis="x", rotation=20, labelsize=8)
    axes[1].bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("T4：正式评测集 20 题", fontproperties=FONT, fontsize=15)
    fig.tight_layout()
    target = ROOT / "t4_bge_metrics.png"
    fig.savefig(target, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(target)


if __name__ == "__main__":
    main()
