"""Research figures built from results/tables/*.csv (Phase 9).

Colour follows the method (fixed mapping, same in every figure). Palette:
the dataviz reference categorical order, light mode; random is a neutral grey
baseline. Every figure states n, the metric definition and the CI method.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {
    "semantic": "#2a78d6", "tnr": "#eb6834", "uniform": "#1baf7a", "semantic_generic": "#eda100",
    "tnr_generic": "#e87ba4", "semantic_nms": "#6250d6", "random": "#8a8a85",
}
LABELS = {
    "random": "Random (10 seeds)", "uniform": "Uniform", "semantic": "Semantic top-k",
    "tnr": "TNR (r=2, 50% anchors)", "semantic_generic": "Semantic, generic prompt (control)",
    "tnr_generic": "TNR, generic prompt (control)", "semantic_nms": "Semantic + NMS gap 4 (post-hoc)",
}
GROUPS = ["Overall", "Spatial", "Temporal", "Multi-goal"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
COV_DEF = "goal coverage = fraction of goals with ≥1 selected frame in that goal's valid set"
CI_DEF = "95% CI: episode cluster bootstrap (5000 reps)"


def _style() -> None:
    plt.rcParams.update({
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
        "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "font.size": 9,
        "axes.titlesize": 10, "legend.frameon": False, "figure.facecolor": "white",
    })


def _save(fig, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def method_comparison(e1: pd.DataFrame, methods: list[str], out: Path, k: int = 8) -> Path:
    """Fig 1: grouped bars of goal coverage per method and task group."""
    _style()
    d = e1[e1.metric == "goal_coverage"]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    w = 0.8 / len(methods)
    for i, m in enumerate(methods):
        x = d[d.method == m].set_index("group").loc[GROUPS]
        pos = np.arange(len(GROUPS)) - 0.4 + w * (i + 0.5)
        ax.bar(pos, x["mean"] * 100, w - 0.02, color=COLORS[m], label=LABELS[m])
        ax.errorbar(pos, x["mean"] * 100, yerr=[(x["mean"] - x.ci_lo) * 100, (x.ci_hi - x["mean"]) * 100],
                    fmt="none", ecolor=INK, elinewidth=0.9, capsize=2)
    n = d[d.method == methods[0]].set_index("group").loc[GROUPS].n_questions
    ax.set_xticks(range(len(GROUPS)), [f"{g}\n(n={int(v)} q)" for g, v in zip(GROUPS, n)])
    ax.set_ylabel("goal coverage (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"Retrieval methods at k={k} frames (of 96), FindingDory val, 100 episodes")
    ax.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.text(0.01, -0.2, f"{COV_DEF}. {CI_DEF}.", fontsize=7, color=MUTED)
    return _save(fig, out)


def budget_curves(e2: pd.DataFrame, methods: list[str], out: Path) -> Path:
    """Fig 2: goal coverage vs memory budget k, one panel per task group."""
    _style()
    d = e2[e2.metric == "goal_coverage"]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4), sharey=True)
    for ax, g in zip(axes, GROUPS):
        for m in methods:
            x = d[(d.group == g) & (d.method == m)].sort_values("k")
            ax.plot(x.k, x["mean"] * 100, color=COLORS[m], lw=2, marker="o", ms=4, label=LABELS[m])
            ax.fill_between(x.k, x.ci_lo * 100, x.ci_hi * 100, color=COLORS[m], alpha=0.12, lw=0)
        ax.set_xscale("log", base=2)
        ax.set_xticks([4, 8, 16, 32], ["4", "8", "16", "32"])
        n = int(d[(d.group == g)].n_questions.iloc[0])
        ax.set_title(f"{g} (n={n})")
        ax.set_xlabel("budget k (frames)")
    axes[0].set_ylabel("goal coverage (%)")
    axes[0].set_ylim(20, 100)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=len(l), fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Coverage vs memory budget (bands: 95% CI, episode bootstrap)", y=1.02)
    return _save(fig, out)


def radius_ablation(e3: pd.DataFrame, out: Path) -> Path:
    """Fig 3: TNR − semantic coverage difference vs radius, by anchor fraction and k."""
    _style()
    d = e3[(e3.group == "Overall") & (e3.metric == "goal_coverage")]
    shades = {0.25: "#a7c7ef", 0.5: "#2a78d6", 0.75: "#0d3c78"}  # one hue, light -> dark
    ks = sorted(d.k.unique())
    fig, axes = plt.subplots(1, len(ks), figsize=(12, 3.2), sharey=True)
    for ax, k in zip(axes, ks):
        ax.axhline(0, color=MUTED, lw=1)
        for af, c in shades.items():
            x = d[(d.k == k) & ((d.anchor_frac == af) | (d.radius == 0))].sort_values("radius")
            ax.errorbar(x.radius, x.diff_vs_semantic * 100,
                        yerr=[(x.diff_vs_semantic - x.diff_lo) * 100, (x.diff_hi - x.diff_vs_semantic) * 100],
                        color=c, lw=2, marker="o", ms=4, capsize=2, label=f"anchor fraction {af}")
        ax.set_xticks([0, 1, 2, 4])
        ax.set_xlabel("neighbourhood radius r (frames)")
        ax.set_title(f"k = {k}")
    axes[0].set_ylabel("TNR − semantic coverage (pp)")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Radius ablation, all 5,595 questions (r=0 ≡ semantic top-k; bars: 95% CI)", y=1.02)
    return _save(fig, out)


def category_forest(e4: pd.DataFrame, out: Path) -> Path:
    """Fig 4: per-category paired differences with CIs (forest plot)."""
    _style()
    d = e4[e4.kind == "diff"]
    order = (e4[(e4.kind == "level") & (e4.name == "semantic")]
             .sort_values(["high", "low"])[["low", "high", "n_questions"]].reset_index(drop=True))
    series = [("tnr - semantic", "TNR − semantic", COLORS["tnr"], -0.17),
              ("semantic - semantic_generic", "Semantic − generic-prompt control", COLORS["semantic_generic"], 0.17)]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.axvline(0, color=MUTED, lw=1)
    for name, lab, c, off in series:
        x = d[d.name == name].set_index("low").loc[order.low]
        y = np.arange(len(order)) + off
        ax.errorbar(x["mean"] * 100, y, xerr=[(x["mean"] - x.ci_lo) * 100, (x.ci_hi - x["mean"]) * 100],
                    fmt="o", color=c, ms=5, capsize=2, lw=1.5, label=lab)
    short = {"Single-Goal Spatial Tasks": "S", "Single-Goal Temporal Tasks": "T", "Multi-Goal Tasks": "M"}
    ax.set_yticks(range(len(order)), [f"[{short[h]}] {l} (n={int(n)})" for l, h, n in order.values])
    ax.invert_yaxis()
    ax.set_xlabel("difference in goal coverage at k=8 (percentage points)")
    ax.set_title("Per-category paired differences (S spatial, T temporal, M multi-goal)")
    ax.legend(fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.4, -0.1))
    fig.text(0.01, -0.06, CI_DEF, fontsize=7, color=MUTED)
    return _save(fig, out)


def mechanism(seg: pd.DataFrame, marg: pd.DataFrame, out: Path) -> Path:
    """Fig 5: why TNR loses: segment diversity and the marginal value of slots 5–8."""
    _style()
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.4))
    seg = seg.sort_values("n_segments")
    a.barh([LABELS[m] for m in seg.method], seg.n_segments, color=[COLORS[m] for m in seg.method], height=0.6)
    a.set_xlabel("distinct temporal segments among k=8 selected frames")
    a.set_title("Temporal spread of the selection")
    a.set_xlim(0, 8.5)
    b.bar([LABELS[m] for m in marg.method], marg.gain * 100, color=[COLORS[m] for m in marg.method], width=0.5)
    b.set_ylabel("coverage gain from slots 5–8 (pp)")
    b.set_title("Same 4 anchors, different use of the next 4 slots")
    for i, v in enumerate(marg.gain * 100):
        b.text(i, v + 0.3, f"+{v:.1f}", ha="center", color=INK, fontsize=9)
    b.tick_params(axis="x", labelsize=8)
    return _save(fig, out)


def posthoc_diversity(ph: pd.DataFrame, e2: pd.DataFrame, out: Path) -> Path:
    """Fig 6 (post-hoc): semantic + temporal NMS vs semantic and TNR across budgets."""
    _style()
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4), sharey=True)
    d2 = e2[e2.metric == "goal_coverage"]
    for ax, g in zip(axes, GROUPS):
        for m in ["semantic", "tnr", "uniform"]:
            x = d2[(d2.group == g) & (d2.method == m)].sort_values("k")
            ax.plot(x.k, x["mean"] * 100, color=COLORS[m], lw=2, marker="o", ms=4, label=LABELS[m])
        x = ph[(ph.group == g) & (ph.method == "nms_g4")].sort_values("k")
        ax.plot(x.k, x.coverage * 100, color=COLORS["semantic_nms"], lw=2, marker="s", ms=4,
                ls="--", label=LABELS["semantic_nms"])
        ax.fill_between(x.k, x.lo * 100, x.hi * 100, color=COLORS["semantic_nms"], alpha=0.12, lw=0)
        ax.set_xscale("log", base=2)
        ax.set_xticks([4, 8, 16, 32], ["4", "8", "16", "32"])
        ax.set_title(g)
        ax.set_xlabel("budget k (frames)")
    axes[0].set_ylabel("goal coverage (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=len(l), fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("POST-HOC: temporal diversity (NMS) instead of temporal context (gap chosen post-hoc "
                 "from {1,2,4,8}; optimistic)", y=1.02)
    return _save(fig, out)
