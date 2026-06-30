"""Figure 2: Ablation deltas summary. Output: fig2_ablation_deltas.pdf"""
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib

matplotlib.rcParams.update({
    "font.size": 9,
    "font.family": "serif",
    "axes.linewidth": 0.8,
})

fig, ax = plt.subplots(figsize=(3.3, 3.0))

conditions = [
    ("A8 top-k=10 (post-fix)", -19.1, 2.12, "headline"),
    ("A9 chunk 512 (post-fix)", -17.3, 0.91, "headline"),
    ("A4 filter top-5 sent.", -14.3, 0.47, "headline"),
    ("A2 Llama3.1-8B", -9.5, 1.15, "background"),
    ("A9 chunk 128", -5.2, 1.01, "background"),
    ("A6 no reranker", -5.2, 1.70, "background"),
    ("A7 HyDE", -1.8, 0.98, "background"),
    ("A1 dense retriever", -1.2, 0.77, "background"),
    ("A1 hybrid retriever", -1.0, 2.39, "background"),
    ("A5 bge-small", -0.2, 0.87, "background"),
    ("A3 JSON format", +0.4, 0.61, "background"),
    ("A9 chunk 256", +0.7, 0.81, "background"),
    ("A3 XML format", +0.9, 1.50, "background"),
    ("A5 bge-large", +1.0, 0.57, "background"),
]

labels = [c[0] for c in conditions]
deltas = [c[1] for c in conditions]
errs = [c[2] for c in conditions]
colors = ["#d62728" if c[3] == "headline" else "#7f7f7f" for c in conditions]

y_pos = range(len(conditions))

ax.barh(
    y_pos,
    deltas,
    xerr=errs,
    color=colors,
    height=0.6,
    error_kw={"elinewidth": 0.8, "capsize": 2},
)
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=7)
ax.invert_yaxis()
ax.axvline(x=0, color="black", linewidth=0.8)
ax.axvspan(-1.52, 1.52, color="gray", alpha=0.12, zorder=0)

ax.set_xlabel("$\Delta$ vs. locked baseline (83.22)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)

red_patch = mlines.Line2D([], [], color="#d62728", lw=4, label="Headline findings")
gray_patch = mlines.Line2D([], [], color="#7f7f7f", lw=4, label="Background (expected)")
ax.legend(handles=[red_patch, gray_patch], fontsize=6.5, loc="lower right", frameon=False)

plt.tight_layout()
plt.savefig("fig2_ablation_deltas.pdf", bbox_inches="tight")
print("Wrote fig2_ablation_deltas.pdf")
