"""Figure 1: Judge overflow cliff (Section 3.1). Output: fig1_judge_overflow.pdf"""
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({
    "font.size": 9,
    "font.family": "serif",
    "axes.linewidth": 0.8,
})

fig, ax = plt.subplots(figsize=(3.3, 2.4))

# Approximate judge input tokens: top5, top7, top10 (256-word chunks)
context_tokens = [1280, 1792, 2560]
labels = ["top-k=5", "top-k=7", "top-k=10"]

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
)
ax.plot(
    context_tokens,
    post_fix_scores,
    "s-",
    color="#1f77b4",
    label="Post-fix (capped judge)",
    markersize=6,
    linewidth=1.3,
)

ax.axvline(x=2500, color="gray", linestyle=":", linewidth=1)
ax.text(2520, 45, "~2,500 token\nthreshold", fontsize=7, color="gray")

for x, y, label in zip(context_tokens, pre_fix_scores, labels):
    ax.annotate(
        label,
        (x, y),
        textcoords="offset points",
        xytext=(0, -14),
        fontsize=7,
        ha="center",
    )

ax.set_xlabel("Judge input context length (tokens)")
ax.set_ylabel("Mean final score")
ax.set_ylim(-5, 95)
ax.legend(fontsize=7, loc="center left", frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle=":", alpha=0.3)

plt.tight_layout()
plt.savefig("fig1_judge_overflow.pdf", bbox_inches="tight")
print("Wrote fig1_judge_overflow.pdf")
