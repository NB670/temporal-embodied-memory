"""Phase 6/7: run all methods on all evaluable questions and write analysis tables.

Usage: python -m scripts.run_experiments [--config configs/experiment.yaml] [--rerun]

Pre-specified primary comparison: TNR (r=main_radius, anchor_frac=main) vs
semantic top-k at k=main_budget, on goal coverage, overall and per
high-level category. Every other comparison is exploratory.

Outputs: results/tables/*.csv, results/tables/summary.md,
results/tables/raw_results.parquet (one row per question x configuration).
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from src.data.load_dataset import PROJECT_ROOT
from src.evaluation.evaluate import build_encoder, load_config, load_eval_questions, run_grid
from src.evaluation.statistics import cluster_bootstrap, paired_difference
from src.memory.frame_store import FrameStore

TAB = PROJECT_ROOT / "results" / "tables"
GROUPS = {"Overall": None, "Spatial": "Single-Goal Spatial Tasks",
          "Temporal": "Single-Goal Temporal Tasks", "Multi-goal": "Multi-Goal Tasks"}
MAIN_METHODS = ["random", "uniform", "semantic_generic", "semantic", "tnr_generic", "tnr"]
PAIRS = [("tnr", "semantic"), ("semantic", "uniform"), ("semantic", "random"),
         ("semantic", "semantic_generic"), ("tnr", "tnr_generic"), ("tnr", "uniform"), ("tnr", "random")]
METRICS = ["goal_coverage", "precision", "all_goals"]


def collapse(raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """One row per (qid, method, k) for the main configuration of each method.
    Random is averaged over seeds (expected performance of random selection)."""
    keys = ["qid", "ep_id", "task_id", "high", "low", "num_interactions", "num_goals", "dev_subset", "k"]
    num = ["any_hit", "goal_coverage", "all_goals", "precision", "recall", "latency_ms"]
    rnd = raw[raw.method == "random"].groupby(keys, as_index=False)[num].mean().assign(method="random")
    r, af = cfg["main_radius"], cfg["main_anchor_frac"]
    main = raw[raw.method.isin(["uniform", "semantic", "semantic_generic"])
               | ((raw.method == "tnr") & (raw.radius == r) & (raw.anchor_frac == af))
               | ((raw.method == "tnr_generic") & (raw.radius == r))]
    return pd.concat([rnd, main[keys + num + ["method"]]], ignore_index=True)


def subset(df: pd.DataFrame, group: str) -> pd.DataFrame:
    return df if GROUPS[group] is None else df[df.high == GROUPS[group]]


def wide(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    return df.pivot_table(index=["qid", "ep_id"], columns="method", values=metric).reset_index()


def est_row(e, **kw) -> dict:
    return dict(**kw, mean=e.mean, ci_lo=e.lo, ci_hi=e.hi, p_value=e.p_value,
                n_questions=e.n_questions, n_episodes=e.n_episodes)


def method_table(df: pd.DataFrame, methods, groups, metrics, bs) -> pd.DataFrame:
    rows = []
    for g in groups:
        d = subset(df, g)
        for metric in metrics:
            if metric == "all_goals" and g not in ("Multi-goal",):
                continue
            for m in methods:
                x = d[d.method == m]
                rows.append(est_row(cluster_bootstrap(x[metric].to_numpy(), x.ep_id.to_numpy(), **bs),
                                    group=g, metric=metric, method=m))
    return pd.DataFrame(rows)


def paired_table(df: pd.DataFrame, pairs, groups, metrics, bs, by_col: str = "high") -> pd.DataFrame:
    rows = []
    for g in groups:
        d = subset(df, g) if by_col == "high" else df[df[by_col] == g]
        for metric in metrics:
            if metric == "all_goals" and g != "Multi-goal":
                continue
            w = wide(d, metric)
            for a, b in pairs:
                if a not in w or b not in w:
                    continue
                e = paired_difference(w[a].to_numpy(), w[b].to_numpy(), w.ep_id.to_numpy(), **bs)
                rows.append(est_row(e, group=g, metric=metric, comparison=f"{a} - {b}"))
    return pd.DataFrame(rows)


def fmt(m, lo, hi, p=None, pct=True) -> str:
    s = 100 if pct else 1
    out = f"{m * s:+.1f} [{lo * s:+.1f}, {hi * s:+.1f}]" if p is not None else f"{m * s:.1f} [{lo * s:.1f}, {hi * s:.1f}]"
    return out + (f" (p={p:.3f})" if p is not None else "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "experiment.yaml"))
    ap.add_argument("--rerun", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    bs = dict(reps=cfg["statistics"]["bootstrap_reps"], seed=cfg["statistics"]["bootstrap_seed"],
              ci=cfg["statistics"]["ci"])
    TAB.mkdir(parents=True, exist_ok=True)
    K = cfg["main_budget"]

    store = FrameStore(build_encoder(cfg))
    questions = load_eval_questions(cfg)
    raw_path = TAB / "raw_results.parquet"
    if raw_path.exists() and not args.rerun:
        raw = pd.read_parquet(raw_path)
    else:
        t0 = time.perf_counter()
        raw = run_grid(cfg, store, questions)
        print(f"grid: {len(raw)} rows in {time.perf_counter() - t0:.1f}s")
        raw.to_parquet(raw_path, index=False)
    df = collapse(raw, cfg)
    df.to_csv(TAB / "results_main_configs.csv", index=False)
    print(f"evaluable questions: {df.qid.nunique()}  episodes: {df.ep_id.nunique()}")

    # ---------------- E1: baseline comparison at k = main ----------------------------
    d8 = df[df.k == K]
    e1 = method_table(d8, MAIN_METHODS, GROUPS, METRICS, bs)
    e1.to_csv(TAB / "e1_methods_k8.csv", index=False)
    e1p = paired_table(d8, PAIRS, GROUPS, METRICS, bs)
    e1p.to_csv(TAB / "e1_paired_k8.csv", index=False)

    # held-out check: the 70 episodes never inspected in Phase 2/3
    ho = paired_table(d8[~d8.dev_subset], PAIRS[:4], GROUPS, ["goal_coverage", "precision"], bs)
    ho.to_csv(TAB / "e1_paired_k8_heldout70.csv", index=False)

    # ---------------- E2: memory budget ----------------------------------------------
    e2 = pd.concat([method_table(df[df.k == k], MAIN_METHODS, GROUPS, ["goal_coverage", "precision"], bs)
                    .assign(k=k) for k in cfg["budgets"]])
    e2.to_csv(TAB / "e2_budget_methods.csv", index=False)
    e2p = pd.concat([paired_table(df[df.k == k], PAIRS[:4], GROUPS, ["goal_coverage", "precision"], bs)
                     .assign(k=k) for k in cfg["budgets"]])
    e2p.to_csv(TAB / "e2_budget_paired.csv", index=False)

    # ---------------- E3: radius x anchor fraction -----------------------------------
    sem = raw[raw.method == "semantic"][["qid", "ep_id", "high", "k", "goal_coverage", "precision"]]
    rows = []
    for (r, af), t in raw[raw.method == "tnr"].groupby(["radius", "anchor_frac"]):
        m = t.merge(sem, on=["qid", "ep_id", "high", "k"], suffixes=("", "_sem"))
        for k, mk in m.groupby("k"):
            for g in GROUPS:
                x = subset(mk, g)
                for metric in ["goal_coverage", "precision"]:
                    lvl = cluster_bootstrap(x[metric].to_numpy(), x.ep_id.to_numpy(), **bs)
                    dif = paired_difference(x[metric].to_numpy(), x[f"{metric}_sem"].to_numpy(),
                                            x.ep_id.to_numpy(), **bs)
                    rows.append(dict(radius=int(r), anchor_frac=af, k=k, group=g, metric=metric,
                                     tnr=lvl.mean, tnr_lo=lvl.lo, tnr_hi=lvl.hi,
                                     diff_vs_semantic=dif.mean, diff_lo=dif.lo, diff_hi=dif.hi,
                                     p_value=dif.p_value, n_questions=dif.n_questions))
    e3 = pd.DataFrame(rows)
    e3.to_csv(TAB / "e3_radius_anchor.csv", index=False)
    # how often does TNR actually differ from semantic (selection overlap)?
    sel = raw[(raw.method.isin(["semantic", "tnr"]))].copy()
    sel["sel"] = sel.selected.str.split().map(lambda x: frozenset(map(int, x)))
    s_sem = sel[sel.method == "semantic"].set_index(["qid", "k"]).sel
    ov = []
    for (r, af), t in sel[sel.method == "tnr"].groupby(["radius", "anchor_frac"]):
        t = t.set_index(["qid", "k"])
        o = pd.Series([len(a & s_sem[i]) / len(a) for i, a in t.sel.items()], index=t.index)
        for k, v in o.groupby(level="k"):
            ov.append(dict(radius=int(r), anchor_frac=af, k=k, mean_overlap_with_semantic=v.mean(),
                           frac_identical=(v == 1).mean()))
    pd.DataFrame(ov).to_csv(TAB / "e3_selection_overlap.csv", index=False)

    # ---------------- E4: fine-grained categories ------------------------------------
    lows = sorted(d8.low.unique())
    rows = []
    for low in lows:
        x = d8[d8.low == low]
        w = wide(x, "goal_coverage")
        for m in ["random", "uniform", "semantic_generic", "semantic", "tnr"]:
            e = cluster_bootstrap(w[m].to_numpy(), w.ep_id.to_numpy(), **bs)
            rows.append(est_row(e, low=low, high=x.high.iloc[0], kind="level", name=m))
        for a, b in [("tnr", "semantic"), ("semantic", "uniform"), ("semantic", "semantic_generic")]:
            e = paired_difference(w[a].to_numpy(), w[b].to_numpy(), w.ep_id.to_numpy(), **bs)
            rows.append(est_row(e, low=low, high=x.high.iloc[0], kind="diff", name=f"{a} - {b}"))
    e4 = pd.DataFrame(rows)
    e4.to_csv(TAB / "e4_lowlevel_k8.csv", index=False)

    # ---------------- Q5: episode complexity -----------------------------------------
    w = wide(d8, "goal_coverage").merge(d8[["qid", "num_interactions", "high"]].drop_duplicates(), on="qid")
    w["diff"] = w.tnr - w.semantic
    w["sem_minus_uni"] = w.semantic - w.uniform
    rows = []
    for n, x in w.groupby("num_interactions"):
        e = cluster_bootstrap(x["diff"].to_numpy(), x.ep_id.to_numpy(), test_zero=True, **bs)
        rows.append(est_row(e, num_interactions=n, comparison="tnr - semantic",
                            semantic=x.semantic.mean(), tnr=x.tnr.mean(), uniform=x.uniform.mean()))
    e5 = pd.DataFrame(rows)
    ep = w.groupby(["ep_id", "num_interactions"])[["diff", "sem_minus_uni"]].mean().reset_index()
    rho, p = spearmanr(ep.num_interactions, ep["diff"])
    rho2, p2 = spearmanr(ep.num_interactions, ep["sem_minus_uni"])
    e5.to_csv(TAB / "e5_complexity_k8.csv", index=False)

    # ---------------- random seed variance, latency ----------------------------------
    rs = (raw[raw.method == "random"].groupby(["k", "seed"]).goal_coverage.mean()
          .groupby("k").agg(["mean", "std", "min", "max"]))
    rs.to_csv(TAB / "random_seed_variance.csv")
    lat = raw.groupby(["method", "k"]).latency_ms.agg(["median", "mean", lambda x: x.quantile(0.99)])
    lat.columns = ["median_ms", "mean_ms", "p99_ms"]
    lat.to_csv(TAB / "latency.csv")
    env = json.loads((store.cache_dir / "environment.json").read_text())

    # ---------------- special answers ([[-1]], [[]]) -----------------------------------
    special = questions[questions.answer_kind != "frames"]
    mixed = questions[questions.task_id.isin(special.task_id.unique())]
    emb = store.text_embeddings(mixed.question.tolist())
    mx = [store.get_episode(r.split, r.ep_id, r.video).similarity(emb[i]).max()
          for i, r in enumerate(mixed.itertuples())]
    mixed = mixed.assign(max_sim=mx, unanswerable=(mixed.answer_kind != "frames").astype(int))
    sp_rows = []
    for t, x in mixed.groupby("task_id"):
        if x.unanswerable.nunique() == 2:
            sp_rows.append(dict(task_id=t, low=x.low_level_category.iloc[0], n=len(x),
                                n_unanswerable=int(x.unanswerable.sum()),
                                auroc_max_sim_predicts_unanswerable=roc_auc_score(x.unanswerable, -x.max_sim)))
        else:
            sp_rows.append(dict(task_id=t, low=x.low_level_category.iloc[0], n=len(x),
                                n_unanswerable=int(x.unanswerable.sum()), auroc_max_sim_predicts_unanswerable=np.nan))
    sp = pd.DataFrame(sp_rows)
    sp.to_csv(TAB / "special_answers.csv", index=False)

    # ---------------- summary markdown -----------------------------------------------
    L = ["# Results summary (generated by scripts/run_experiments.py)\n",
         f"Split: validation, {df.ep_id.nunique()} episodes, {df.qid.nunique()} evaluable questions "
         f"(answer_kind = frames). Encoder {env['encoder']}. 95% CIs from a {bs['reps']}-rep episode "
         f"cluster bootstrap. Percentages. TNR = r={cfg['main_radius']}, anchor_frac={cfg['main_anchor_frac']}. "
         f"Random = mean over {len(cfg['random_seeds'])} seeds.\n",
         f"## E1: goal coverage at k={K}\n", "| method | " + " | ".join(GROUPS) + " |", "|---" * (len(GROUPS) + 1) + "|"]
    for m in MAIN_METHODS:
        cells = []
        for g in GROUPS:
            r = e1[(e1.group == g) & (e1.metric == "goal_coverage") & (e1.method == m)].iloc[0]
            cells.append(fmt(r["mean"], r.ci_lo, r.ci_hi))
        L.append(f"| {m} | " + " | ".join(cells) + " |")
    n_q = {g: int(e1[(e1.group == g) & (e1.metric == "goal_coverage")].n_questions.iloc[0]) for g in GROUPS}
    L.append("| n questions | " + " | ".join(str(n_q[g]) for g in GROUPS) + " |")
    L += [f"\n## E1: frame precision at k={K}\n", "| method | " + " | ".join(GROUPS) + " |", "|---" * (len(GROUPS) + 1) + "|"]
    for m in MAIN_METHODS:
        cells = []
        for g in GROUPS:
            r = e1[(e1.group == g) & (e1.metric == "precision") & (e1.method == m)].iloc[0]
            cells.append(fmt(r["mean"], r.ci_lo, r.ci_hi))
        L.append(f"| {m} | " + " | ".join(cells) + " |")
    L += [f"\n## E1: paired differences at k={K} (percentage points)\n",
          "| comparison | metric | " + " | ".join(GROUPS) + " |", "|---" * (len(GROUPS) + 2) + "|"]
    for metric in ["goal_coverage", "precision"]:
        for a, b in PAIRS:
            cells = []
            for g in GROUPS:
                r = e1p[(e1p.group == g) & (e1p.metric == metric) & (e1p.comparison == f"{a} - {b}")].iloc[0]
                cells.append(fmt(r["mean"], r.ci_lo, r.ci_hi, r.p_value))
            L.append(f"| {a} − {b} | {metric} | " + " | ".join(cells) + " |")
    mg = e1[(e1.metric == "all_goals")]
    L.append(f"\nMulti-goal strict success (all goals covered) at k={K}: " +
             ", ".join(f"{r.method} {r['mean'] * 100:.1f}%" for _, r in mg.iterrows()))
    L += ["\n## Held-out check (70 episodes not inspected in Phases 2–3), goal coverage, k=8\n",
          "| comparison | " + " | ".join(GROUPS) + " |", "|---" * (len(GROUPS) + 1) + "|"]
    for a, b in PAIRS[:4]:
        cells = [fmt(*ho[(ho.group == g) & (ho.metric == "goal_coverage") & (ho.comparison == f"{a} - {b}")]
                     [["mean", "ci_lo", "ci_hi", "p_value"]].iloc[0]) for g in GROUPS]
        L.append(f"| {a} − {b} | " + " | ".join(cells) + " |")
    L += ["\n## E2: TNR − semantic goal coverage by budget\n", "| k | " + " | ".join(GROUPS) + " |",
          "|---" * (len(GROUPS) + 1) + "|"]
    for k in cfg["budgets"]:
        cells = [fmt(*e2p[(e2p.k == k) & (e2p.group == g) & (e2p.metric == "goal_coverage")
                          & (e2p.comparison == "tnr - semantic")][["mean", "ci_lo", "ci_hi", "p_value"]].iloc[0])
                 for g in GROUPS]
        L.append(f"| {k} | " + " | ".join(cells) + " |")
    L += ["\n## E3: TNR − semantic goal coverage (Overall) by radius and anchor fraction\n",
          "| radius | anchor_frac | " + " | ".join(f"k={k}" for k in cfg["budgets"]) + " |",
          "|---" * (len(cfg["budgets"]) + 2) + "|"]
    for (r, af), x in e3[(e3.group == "Overall") & (e3.metric == "goal_coverage")].groupby(["radius", "anchor_frac"]):
        cells = [fmt(*x[x.k == k][["diff_vs_semantic", "diff_lo", "diff_hi", "p_value"]].iloc[0]) for k in cfg["budgets"]]
        L.append(f"| {r} | {af} | " + " | ".join(cells) + " |")
    L += ["\n## E4: goal coverage by low-level category, k=8\n",
          "| category | n | uniform | semantic | TNR | TNR − semantic | semantic − generic |", "|---" * 7 + "|"]
    for low in lows:
        x = e4[e4.low == low]
        lv = lambda n: x[(x.kind == "level") & (x.name == n)].iloc[0]
        df_ = lambda n: x[(x.kind == "diff") & (x.name == n)].iloc[0]
        a, b = df_("tnr - semantic"), df_("semantic - semantic_generic")
        L.append(f"| {low} | {int(lv('tnr').n_questions)} | {lv('uniform')['mean'] * 100:.1f} | "
                 f"{lv('semantic')['mean'] * 100:.1f} | {lv('tnr')['mean'] * 100:.1f} | "
                 f"{fmt(a['mean'], a.ci_lo, a.ci_hi, a.p_value)} | {fmt(b['mean'], b.ci_lo, b.ci_hi, b.p_value)} |")
    L += ["\n## Q5: complexity\n", f"Spearman (episode-level, n={len(ep)}) between num_interactions and "
          f"TNR − semantic coverage: rho={rho:.3f}, p={p:.3f}; semantic − uniform: rho={rho2:.3f}, p={p2:.3f}.\n",
          "| num_interactions | n_q | uniform | semantic | TNR | TNR − semantic |", "|---" * 6 + "|"]
    for _, r in e5.iterrows():
        L.append(f"| {int(r.num_interactions)} | {int(r.n_questions)} | {r.uniform * 100:.1f} | {r.semantic * 100:.1f} | "
                 f"{r.tnr * 100:.1f} | {fmt(r['mean'], r.ci_lo, r.ci_hi, r.p_value)} |")
    L += ["\n## Random baseline: across-seed spread of overall goal coverage\n", rs.mul(100).round(2).to_markdown()]
    L += ["\n## Latency (selection only; embeddings precomputed)\n",
          lat[lat.index.get_level_values("k") == K].round(4).to_markdown(),
          f"\nOne-off embedding cost: {env['n_episodes']} episodes x 96 frames encoded in "
          f"{env['image_encode_s_uncached_part']} s ({env['device']}); questions {env['text_encode_s']} s."]
    L += ["\n## Special answers (excluded from retrieval metrics)\n",
          f"{len(special)} questions: {(special.answer_kind == 'unseen').sum()} `[[-1]]` (target never observed), "
          f"{(special.answer_kind == 'empty').sum()} `[[]]`. Can max cos(question, frame) flag them within the "
          "same template? (AUROC, 0.5 = no)\n", sp.round(3).to_markdown(index=False)]
    (TAB / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
