import csv
from collections import defaultdict
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

Z = 1.96   # 95 % confidence

# baseline can't be loaded any more (old target features), value from its training log
TRAINING_ONLY = {"1 baseline": 3.0}

def wilson(successes, n):
    """95 % Wilson interval for a success rate, returns (rate, lower, upper)"""
    p = successes / n
    centre = (p + Z**2 / (2 * n)) / (1 + Z**2 / n)
    half = Z * sqrt(p * (1 - p) / n + Z**2 / (4 * n**2)) / (1 + Z**2 / n)
    return p, centre - half, centre + half





rows = defaultdict(list) # model -> [(episodes, successes), ...]
for results_file in sorted(Path("evaluation_history").glob("*.csv")):
    with open(results_file) as f:
        for row in csv.DictReader(f):
            if "scenario" in row["config"]:
                continue
            rows[row["model"]].append((int(row["episodes"]), int(row["successes"])))

fig, ax = plt.subplots(figsize=(7, 4.5))
rng = np.random.default_rng(0)

labels = list(TRAINING_ONLY) + list(rows)

for i, label in enumerate(labels):
    if label in TRAINING_ONLY:
        # hollow marker: measured during training (with random actions), no confidence interval
        ax.scatter(i, TRAINING_ONLY[label], s=70, facecolors="none", edgecolors="C0", linewidths=2, zorder=2)
        ax.annotate(f"{TRAINING_ONLY[label]:.1f} %*", (i + 0.2, TRAINING_ONLY[label]), va="center")
        continue

    results = rows[label]
    # single scenarios: spread between scenarios
    rates = [100 * s / n for n, s in results]
    ax.scatter(i + rng.uniform(-0.15, 0.15, len(rates)), rates, s=12, color="0.75", zorder=1)

    # all episodes pooled: rate with 95 % interval
    n_total = sum(n for n, _ in results)
    s_total = sum(s for _, s in results)
    p, low, high = wilson(s_total, n_total)
    ax.errorbar(i, 100 * p, yerr=[[100 * (p - low)], [100 * (high - p)]],
                fmt="o", color="C0", markersize=8, capsize=6, linewidth=2, zorder=2)
    ax.annotate(f"{100 * p:.1f} %", (i + 0.2, 100 * p), va="center")

ax.set_xticks(range(len(labels)), labels, rotation=15)
#ax.text(0.01, 0.01, "* aus dem Training, nicht vergleichbar gemessen", transform=ax.transAxes, fontsize=8, color="0.4")

ax.set_ylabel("Erfolgsquote [%]")
ax.set_ylim(0, 100)
ax.set_title("Erfolgsquote je Entwicklungsstufe (95-%-Konfidenzintervall)")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("history.png", dpi=200)
plt.show()