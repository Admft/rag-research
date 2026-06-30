"""Figure 1: Judge overflow cliff (Section 3.1). Output: fig1_judge_overflow.pdf"""
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib

matplotlib.rcParams.update({
    "font.size": 9,
    "font.family": "serif",
    "axes.linewidth": 0.8,
})

fig, ax = plt.subplots(figsize=(3.3, 2.85))

# Judge input tokens for A8 top-k ∈ {5, 7, 10} at 256-word chunks
context_tokens = [1280, 1792, 2560]
pre_fix_scores = [83.73, 82.64, 9.94]
post_fix_scores = [83.73, 83.72, 64.14]

ax.plot(
    context_tokens,
    pre_fix_scores,
    "o--",
    color="#d62728",
    label="Pre-fix (uncapped judge)",
    markersize=6,
    linewidth=1.3,
    zorder=3,
)
ax.plot(
    context_tokens,
    post_fix_scores,
    "s-",
    color="#1f77b4",
    label="Post-fix (capped judge)",
    markersize=6,
    linewidth=1.3,
    zorder=3,
)

# Threshold between top-k=7 and top-k=10
ax.axvline(x=2500, color="#888888", linestyle=":", linewidth=1, zorder=1)
ax.annotate(
    "~2,500 token\nthreshold",
    xy=(2500, 88),
    xytext=(2500, 88),
    fontsize=6.5,
    color="#555555",
    ha="center",
    va="top",
)

# Condition names on x-axis (no floating point labels)
ax.set_xticks(context_tokens)
ax.set_xticklabels(["top-k=5", "top-k=7", "top-k=10"], fontsize=7)

ax.set_xlabel("Judge input context length (tokens)", fontsize=8)
ax.set_ylabel("Mean final score", fontsize=8)
ax.set_xlim(1050, 2850)
ax.set_ylim(0, 92)
ax.set_yticks([0, 20, 40, 60, 80])
ax.grid(axis="y", linestyle=":", alpha=0.35, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Small value labels only at the cliff (top-k=10) where two series diverge
ax.annotate(
    "9.9",
    xy=(2560, 9.94),
    xytext=(8, 0),
    textcoords="offset points",
    fontsize=6,
    color="#d62728",
    ha="left",
    va="center",
)
ax.annotate(
    "64.1",
    xy=(2560, 64.14),
    xytext=(8, 0),
    textcoords="offset points",
    fontsize=6,
    color="#1f77b4",
    ha="left",
    va="center",
)

# Legend below chart (matches Figure 2 style)
red_line = mlines.Line2D([], [], color="#d62728", marker="o", linestyle="--", linewidth=1.3,
                         markersize=5, label="Pre-fix (uncapped judge)")
blue_line = mlines.Line2D([], [], color="#1f77b4", marker="s", linestyle="-", linewidth=1.3,
                          markersize=5, label="Post-fix (capped judge)")
ax.legend(
    handles=[red_line, blue_line],
    fontsize=6.5,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.22),
    ncol=1,
    frameon=False,
    handlelength=2.5,
)

plt.tight_layout()
plt.savefig("fig1_judge_overflow.pdf", bbox_inches="tight")
print("Wrote fig1_judge_overflow.pdf")
