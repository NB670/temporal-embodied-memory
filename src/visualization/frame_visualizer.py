"""Visualise retrieved frames against the similarity curve and ground truth.

Ground truth is drawn for *inspection only*. It is never passed to a retriever.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GOAL_COLORS = ["#2a9d8f", "#e76f51", "#8e7dbe", "#e9c46a", "#457b9d", "#f4a261"]


def _runs(idx: list[int]) -> list[tuple[int, int]]:
    s = sorted(set(i for i in idx if i >= 0))
    out: list[tuple[int, int]] = []
    for i in s:
        if out and i == out[-1][1] + 1:
            out[-1] = (out[-1][0], i)
        else:
            out.append((i, i))
    return out


def plot_retrieval(
    frames: list[np.ndarray],
    similarity: np.ndarray,
    selected: dict[str, list[int]],
    goals: list[list[int]],
    title: str,
    out: Path,
    scores: dict[str, list[float]] | None = None,
    max_thumbs: int = 8,
) -> Path:
    """One figure: similarity curve with GT spans and selections, then one
    thumbnail row per method (frames in chronological order)."""
    methods = list(selected)
    n_rows = 1 + len(methods)
    fig = plt.figure(figsize=(max_thumbs * 1.55, 2.6 + 2.55 * len(methods)))
    gs = fig.add_gridspec(n_rows, max_thumbs, height_ratios=[1.3] + [1.6] * len(methods))

    ax = fig.add_subplot(gs[0, :])
    t = np.arange(len(similarity))
    ax.plot(t, similarity, color="#333", lw=1.2, label="cos(question, frame)")
    gt_union = {i for g in goals for i in g if i >= 0}
    for gi, g in enumerate(goals):
        for a, b in _runs(g):
            ax.axvspan(a - 0.5, b + 0.5, color=GOAL_COLORS[gi % len(GOAL_COLORS)], alpha=0.25,
                       label=f"GT goal {gi}" if (a, b) == _runs(g)[0] and gi < 6 else None)
    markers = ["v", "^", "s", "D"]
    lo, hi = np.nanmin(similarity[np.isfinite(similarity)]), np.nanmax(similarity[np.isfinite(similarity)])
    for mi, m in enumerate(methods):
        y = hi + (mi + 1) * 0.08 * (hi - lo)
        ax.scatter(selected[m], [y] * len(selected[m]), marker=markers[mi % 4], s=28, label=f"{m} picks")
    ax.set_xlim(-1, len(similarity))
    ax.set_xlabel("frame index (position in 96-frame video)")
    ax.set_ylabel("cosine sim.")
    ax.legend(fontsize=7, ncol=4, loc="lower left", framealpha=0.8)
    ax.set_title(title, fontsize=9)

    for mi, m in enumerate(methods):
        picks = selected[m][:max_thumbs]
        for j in range(max_thumbs):
            a = fig.add_subplot(gs[1 + mi, j])
            a.axis("off")
            if j >= len(picks):
                continue
            i = picks[j]
            a.imshow(frames[i])
            hit = i in gt_union
            sc = f" s={scores[m][j]:.3f}" if scores and m in scores else ""
            a.set_title(f"{i}{sc}\n{'GT hit' if hit else 'miss'}", fontsize=7,
                        color="#1b7f3b" if hit else "#b00020")
            if j == 0:
                a.text(-0.08, 0.5, m, transform=a.transAxes, rotation=90, va="center", ha="right", fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=85, bbox_inches="tight")
    plt.close(fig)
    return out
