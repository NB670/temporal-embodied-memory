"""Phase 2: inspect the FindingDory 96-frame dataset and build the episode subset.

Usage:
    python -m src.data.inspect_dataset --n-episodes 30 --seed 0

Outputs (under data/metadata/):
    subset_episodes.json         chosen (split, ep_id) list + selection config
    subset_questions.csv         all questions for those episodes (+ parsed fields)
    label_stats_validation.csv   per-question label structure stats (whole split)
    task_templates.csv           one example question per task_id
    video_check.csv              decoded frame count / resolution per episode
and a figure results/figures/phase2_examples.png.
"""

from __future__ import annotations

import argparse
import json
from math import comb

import numpy as np
import pandas as pd

from src.data.load_dataset import (
    ANSWER_FRAMES,
    NUM_FRAMES,
    PROJECT_ROOT,
    REVISION,
    RemoteZipVideoFetcher,
    decode_video,
    load_questions,
    select_episodes,
)

META = PROJECT_ROOT / "data" / "metadata"
FIG_DIR = PROJECT_ROOT / "results" / "figures"


def contiguous_runs(idx: list[int]) -> int:
    """Number of maximal runs of consecutive integers in ``idx``."""
    s = sorted(set(i for i in idx if i >= 0))
    return 0 if not s else 1 + sum(b != a + 1 for a, b in zip(s, s[1:]))


def chance_any_hit(n_valid: int, k: int, n: int = NUM_FRAMES) -> float:
    """P(at least one of k uniformly random distinct frames is valid)."""
    if n_valid <= 0:
        return 0.0
    return 1.0 - comb(n - n_valid, k) / comb(n, k)


def label_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-question statistics of the ground-truth structure."""
    rows = []
    for r in df.itertuples():
        valid_goals = [g for g in r.goals if g and g != [-1]]
        union = sorted({i for g in valid_goals for i in g})
        rows.append(
            dict(
                ep_id=r.ep_id,
                task_id=r.task_id,
                high=r.high_level_category,
                low=r.low_level_category,
                answer_kind=r.answer_kind,
                num_goals=r.num_goals,
                n_valid_frames=len(union),
                frac_valid=len(union) / NUM_FRAMES,
                runs_per_goal=np.mean([contiguous_runs(g) for g in valid_goals]) if valid_goals else 0,
                goals_overlap=len(union) < sum(len(set(g)) for g in valid_goals),
                chance_hit_k8=chance_any_hit(len(union), 8),
            )
        )
    return pd.DataFrame(rows)


def print_header(t: str) -> None:
    print(f"\n{'=' * 8} {t} {'=' * (70 - len(t))}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-episodes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="validation", choices=["train", "validation"])
    args = ap.parse_args()
    META.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_questions(args.split)

    print_header(f"SPLIT: {args.split}  (revision {REVISION[:8]})")
    print(f"questions={len(df)}  episodes={df.ep_id.nunique()}  task_ids={df.task_id.nunique()}")
    print("columns:", list(df.columns[:8]))
    print("video field example:", df.video.iloc[0], "(relative path, not bytes)")
    print("\nanswer_kind counts:\n", df.answer_kind.value_counts().to_string())
    print("\nhigh x answer_kind:\n", pd.crosstab(df.high_level_category, df.answer_kind).to_string())

    stats = label_stats(df)
    stats.to_csv(META / f"label_stats_{args.split}.csv", index=False)
    ok = stats[stats.answer_kind == ANSWER_FRAMES]
    print_header("GROUND-TRUTH STRUCTURE (answer_kind == frames)")
    agg = ok.groupby("low").agg(
        n=("task_id", "size"),
        goals=("num_goals", "mean"),
        valid_frames=("n_valid_frames", "mean"),
        frac_valid=("frac_valid", "mean"),
        runs_per_goal=("runs_per_goal", "mean"),
        chance_hit_k8=("chance_hit_k8", "mean"),
    )
    print(agg.round(3).to_string())
    print(f"\nmulti-goal questions whose goal sets overlap: {int(ok[ok.num_goals > 1].goals_overlap.sum())}"
          f" / {int((ok.num_goals > 1).sum())}")

    templates = df.drop_duplicates("task_id")[["task_id", "high_level_category", "low_level_category", "question"]]
    templates = templates.assign(k=templates.task_id.str[5:].astype(int)).sort_values("k").drop(columns="k")
    templates.to_csv(META / "task_templates.csv", index=False)

    # ---------------- subset selection (metadata only, no model outputs) -------------
    eps = select_episodes(df, args.n_episodes, seed=args.seed)
    sub = df[df.ep_id.isin(eps)].copy()
    sub["goals"] = sub["goals"].map(json.dumps)
    sub.to_csv(META / "subset_questions.csv", index=False)
    (META / "subset_episodes.json").write_text(json.dumps(dict(
        split=args.split, revision=REVISION, seed=args.seed, n_episodes=len(eps),
        strategy="stratified by num_interactions; metadata only", ep_ids=eps), indent=2))
    print_header(f"SUBSET: {len(eps)} episodes, {len(sub)} questions")
    print("num_interactions:", sub.drop_duplicates("ep_id").num_interactions.value_counts().sort_index().to_dict())
    print(pd.crosstab(sub.high_level_category, sub.answer_kind).to_string())

    print_header("10 EXAMPLE QUESTIONS (one per low-level category, first subset episode)")
    ex = sub[sub.ep_id == eps[0]].drop_duplicates("low_level_category").head(10)
    for r in ex.itertuples():
        print(f"[{r.ep_id} {r.task_id} | {r.low_level_category} | kind={r.answer_kind}]")
        print(f"  Q: {r.question[21:]}")
        print(f"  A: {r.answer}")

    # ---------------- videos -----------------------------------------------------------
    print_header("VIDEO FETCH + DECODE CHECK")
    fetcher = RemoteZipVideoFetcher()
    checks = []
    decoded = {}
    for ep in eps:
        rel = sub.loc[sub.ep_id == ep, "video"].iloc[0]
        path = fetcher.fetch(rel)
        v = decode_video(path)
        h, w = v.frames[0].shape[:2]
        checks.append(dict(ep_id=ep, video=rel, n_frames=len(v.frames), height=h, width=w,
                           fps=v.fps, mb=round(path.stat().st_size / 1e6, 2)))
        if ep in ex.ep_id.values:
            decoded[ep] = v
    vc = pd.DataFrame(checks)
    vc.to_csv(META / "video_check.csv", index=False)
    print(vc.describe().loc[["min", "max"]].to_string())
    print(f"all videos have {NUM_FRAMES} frames: {bool((vc.n_frames == NUM_FRAMES).all())}")
    print(f"bytes downloaded this run: {fetcher.bytes_fetched / 1e6:.1f} MB; cache total {vc.mb.sum():.1f} MB")

    # ---------------- figure -----------------------------------------------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    v = decoded[eps[0]]
    show = ex[ex.answer_kind == ANSWER_FRAMES].head(6)
    fig = plt.figure(figsize=(10, 3.9 * len(show)))
    for sf, r in zip(fig.subfigures(len(show), 1), show.itertuples()):
        goals = json.loads(r.goals)
        picks = []  # first / middle / last valid frame of each goal, max 4 panels
        for gi, g in enumerate(goals):
            for idx in (g[0], g[len(g) // 2], g[-1]):
                picks.append((gi, idx))
        picks = list(dict.fromkeys(picks))[:4]
        axes = sf.subplots(1, 4)
        for ax in axes:
            ax.axis("off")
        for ax, (gi, idx) in zip(axes, picks):
            ax.imshow(v.frames[idx])
            ax.set_title(f"goal {gi} | frame {idx}", fontsize=9)
        gt = r.answer if len(r.answer) < 110 else r.answer[:107] + "..."
        sf.suptitle(f"{r.task_id} [{r.low_level_category}] {r.question[21:]}\nGT: {gt}", fontsize=9)
    fig.suptitle(f"{args.split} {eps[0]}: sample ground-truth frames (index = position in 96-frame video)",
                 fontsize=11, y=1.0)
    out = FIG_DIR / "phase2_examples.png"
    fig.savefig(out, dpi=90, bbox_inches="tight")
    print(f"\nsaved figure: {out}")


if __name__ == "__main__":
    main()
