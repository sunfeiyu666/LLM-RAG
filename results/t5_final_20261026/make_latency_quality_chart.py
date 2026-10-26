"""由正式 T5 汇总 CSV 绘制真实的延迟—答案质量权衡图。"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

HERE = Path(__file__).resolve().parent
FONT = FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc")
COLORS = {"direct": "#2563eb", "rule": "#2a9d8f", "model": "#e76f51"}
LABELS = {"direct": "直接 top-3", "rule": "规则重排", "model": "Cross-Encoder 重排"}


def read_csv(name):
    with (HERE / name).open(encoding="utf-8-sig", newline="") as file:
        return {(row["strategy"], row["compressed"]): row for row in csv.DictReader(file)}


def main():
    latency = read_csv("t5_full_summary.csv")
    quality = read_csv("t5_quality_summary.csv")
    expected = {(strategy, compressed) for strategy in COLORS for compressed in ("False", "True")}
    if set(latency) != expected or set(quality) != expected:
        raise ValueError("六组延迟或质量评分不完整")
    fig, ax = plt.subplots(figsize=(9, 5.7))
    all_x, all_y = [], []
    for strategy in COLORS:
        points = []
        for compressed in ("True", "False"):
            key = strategy, compressed
            if latency[key]["questions"] != "20" or quality[key]["questions"] != "20":
                raise ValueError(f"不是完整20题设置：{key}")
            point = (float(latency[key]["mean_total_seconds"]),
                     float(quality[key]["mean_accuracy_0_2"]))
            points.append(point)
            all_x.append(point[0]); all_y.append(point[1])
        xs, ys = zip(*points)
        ax.plot(xs, ys, color=COLORS[strategy], linewidth=2.2, label=LABELS[strategy])
        ax.scatter([xs[0]], [ys[0]], color=COLORS[strategy], marker="o", s=90, zorder=3)
        ax.scatter([xs[1]], [ys[1]], color=COLORS[strategy], marker="s", s=72, zorder=3)
        compressed_offset = (8, -18) if strategy == "direct" else (8, 8)
        ax.annotate("压缩后", points[0], textcoords="offset points", xytext=compressed_offset,
                    fontproperties=FONT, fontsize=9)
        ax.annotate("压缩前", points[1], textcoords="offset points", xytext=(8, -18),
                    fontproperties=FONT, fontsize=9)
    x_pad = max(0.5, (max(all_x) - min(all_x)) * 0.15)
    y_pad = max(0.1, (max(all_y) - min(all_y)) * 0.2)
    ax.set_xlim(min(all_x) - x_pad, max(all_x) + x_pad)
    ax.set_ylim(max(0, min(all_y) - y_pad), min(2, max(all_y) + y_pad))
    ax.set_xlabel("平均端到端延迟（秒/题）", fontproperties=FONT, fontsize=11)
    ax.set_ylabel("平均答案准确性评分（0—2）", fontproperties=FONT, fontsize=11)
    ax.set_title("T5：延迟—质量权衡（正式20题）", fontproperties=FONT, fontsize=14)
    ax.grid(alpha=0.25)
    ax.legend(prop=FONT, frameon=False)
    fig.tight_layout()
    target = HERE / "t5_latency_quality.png"
    fig.savefig(target, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(target)


if __name__ == "__main__":
    main()
