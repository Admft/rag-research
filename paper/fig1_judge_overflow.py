# fig1_judge_overflow.py
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({
    'font.size': 9,
    'font.family': 'serif',
    'axes.linewidth': 0.8,
})

fig, ax = plt.subplots(figsize=(3.4, 2.8))  # slightly taller for breathing room

context_tokens = [1280, 1792, 2560]
labels = ['top-k=5', 'top-k=7', 'top-k=10']

pre_fix_scores = [83.73, 82.64, 9.94]
post_fix_scores = [83.73, 83.72, 64.14]

# Plot lines/markers first
ax.plot(context_tokens, pre_fix_scores, 'o--', color='#d62728',
        label='Pre-fix (uncapped judge)', markersize=6, linewidth=1.3,
        zorder=3)
ax.plot(context_tokens, post_fix_scores, 's-', color='#1f77b4',
        label='Post-fix (capped judge)', markersize=6, linewidth=1.3,
        zorder=3)

# Threshold line — drawn behind everything
ax.axvline(x=2500, color='gray', linestyle=':', linewidth=1, zorder=1)

# Threshold label placed in clear space, upper-left of the line,
# away from both the legend and the data points
ax.text(2380, 88, '~2,500 token\nthreshold', fontsize=6.5,
        color='gray', ha='right', va='top')

# Point labels — each placed in open space, away from lines/markers
# top-k=5: label above-left of point
ax.annotate('top-k=5', (context_tokens[0], post_fix_scores[0]),
            textcoords="offset points", xytext=(-2, 12),
            fontsize=7, ha='center')

# top-k=7: label above the point (clear of both lines, which sit below)
ax.annotate('top-k=7', (context_tokens[1], post_fix_scores[1]),
            textcoords="offset points", xytext=(0, 12),
            fontsize=7, ha='center')

# top-k=10: label to the right of the lower (red) point,
# clear of the x-axis and the threshold line
ax.annotate('top-k=10', (context_tokens[2], pre_fix_scores[2]),
            textcoords="offset points", xytext=(-32, -16),
            fontsize=7, ha='center')

ax.set_xlabel('Judge input context length (tokens)')
ax.set_ylabel('Mean final score')
ax.set_xlim(1150, 2750)
ax.set_ylim(-8, 95)

ax.legend(fontsize=7, loc='lower left', frameon=False,
          bbox_to_anchor=(0.02, 0.30))

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', linestyle=':', alpha=0.3, zorder=0)

plt.tight_layout()
plt.savefig('fig1_judge_overflow.pdf', bbox_inches='tight')
print('Wrote fig1_judge_overflow.pdf')
