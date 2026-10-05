"""Plot 500-rollout success rates for the KL-DSRL β sweep (seed-1 runs)."""
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SRC = "eval/kl_dsrl_sweep_500ep/beta*/kl_dsrl_checkpoint_eval.csv"
OUT = "eval/plots/kl_dsrl_500ep_success"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MARKERS = ["o", "s", "^", "D", "v"]
INK, INK2, GRID, SURF = "#1f1f1e", "#6b6a63", "#e6e5df", "#fcfcfb"

df = pd.concat(pd.read_csv(f) for f in glob.glob(SRC))
df["se"] = np.sqrt(df.success_rate * (1 - df.success_rate) / df.num_episodes)
betas = sorted(df.beta.unique())

plt.rcParams.update({"font.size": 11, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [2.2, 1]},
                              facecolor=SURF)
for a in (ax, ax2):
    a.set_facecolor(SURF)
    a.grid(axis="y", color=GRID, lw=1)
    a.set_axisbelow(True)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)

# Left: success rate vs. checkpoint, ±1.96·SE band
for i, b in enumerate(betas):
    d = df[df.beta == b].sort_values("checkpoint_step")
    x, y, e = d.checkpoint_step / 1e3, d.success_rate, 1.96 * d.se
    ax.fill_between(x, y - e, y + e, color=COLORS[i], alpha=0.10, lw=0)
    ax.plot(x, y, color=COLORS[i], lw=2, marker=MARKERS[i], ms=6,
            mec=SURF, mew=1.5, label=f"β = {b:g}")

# Direct end labels, nudged apart so they don't collide
ends = sorted((df[(df.beta == b) & (df.checkpoint_step == df.checkpoint_step.max())].success_rate.item(), b)
              for b in betas)
last = -1.0
x_end = df.checkpoint_step.max() / 1e3
for y_end, b in ends:
    y_lab = max(y_end, last + 0.024)
    last = y_lab
    ax.annotate(f"β={b:g}  {y_end:.3f}", (x_end, y_end), xytext=(x_end + 7, y_lab), textcoords="data",
                va="center", fontsize=9.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6) if y_lab - y_end > 0.005 else None)
ax.set_xlim(35, 290)
# 240k and 250k are too close to label both; mark 240k with a bare tick
steps_k = sorted(df.checkpoint_step.unique() / 1e3)
ax.set_xticks(steps_k)
ax.set_xticklabels(["" if v == 240 else f"{int(v)}k" for v in steps_k])
ax.set_ylim(0.38, 0.95)
ax.set_xlabel("Training step (checkpoint)")
ax.set_ylabel("Success rate")
ax.set_title("Success rate over training", loc="left", fontsize=12, fontweight="bold")
ax.legend(frameon=False, loc="lower right", ncol=1, fontsize=9.5)

# Right: final checkpoint, 95% CI
fin = df[df.checkpoint_step == df.checkpoint_step.max()].set_index("beta").loc[betas]
for i, b in enumerate(betas):
    r = fin.loc[b]
    ax2.errorbar(i, r.success_rate, yerr=1.96 * r.se, fmt=MARKERS[i], color=COLORS[i],
                 ms=9, mec=SURF, mew=1.5, elinewidth=2, capsize=4)
    ax2.annotate(f"{r.success_rate:.3f}", (i, r.success_rate + 1.96 * r.se), xytext=(0, 5),
                 textcoords="offset points", ha="center", fontsize=9.5, color=INK)
ax2.set_xticks(range(len(betas)))
ax2.set_xticklabels([f"{b:g}" for b in betas])
ax2.set_xlim(-0.6, len(betas) - 0.4)
ax2.set_ylim(0.72, 0.94)
ax2.set_xlabel("β (KL weight)")
ax2.set_title(f"Final checkpoint ({int(df.checkpoint_step.max()/1e3)}k)", loc="left",
              fontsize=12, fontweight="bold")

fig.text(0.01, 0.005, "robomimic Square · training seed 1 · 500 rollouts per point (eval seed 101) · "
         "bands / bars = 95% CI (binomial)", fontsize=8.5, color=INK2)
fig.tight_layout(rect=(0, 0.03, 1, 1))
for ext in ("png", "svg"):
    fig.savefig(f"{OUT}.{ext}", dpi=200, facecolor=SURF)
print(f"saved {OUT}.png/.svg")
