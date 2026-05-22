#!/usr/bin/env python3
"""
kyber_rank_merge.py
════════════════════════════════════════════════════════════════════
Reads multiple kyber_scores CSV files (each produced by opt.py),
aggregates rankings across all runs, writes a consensus CSV, and
prints a rich insight report.

Usage:
    python kyber_rank_merge.py                          # default filenames
    python kyber_rank_merge.py -f s0.csv s1.csv s2.csv  # custom files
    python kyber_rank_merge.py --top 50                 # show top-50 in report
    python kyber_rank_merge.py --out consensus.csv      # custom output name

Output:
    kyber_consensus.csv     consensus-ranked (x,h) pairs with full stats
    (insight report printed to stdout)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Kyber CSV rank merger & analyser")
    p.add_argument(
        "-f", "--files", nargs="+",
        default=["kyber_scores.csv",
                 "kyber_scores_1.csv",
                 "kyber_scores_2.csv",
                 "kyber_scores_3.csv"],
        help="Input CSV files (default: kyber_scores*.csv)",
    )
    p.add_argument("--out",  default="kyber_consensus.csv",
                   help="Output CSV path  [default: kyber_consensus.csv]")
    p.add_argument("--top",  type=int, default=30,
                   help="How many top pairs to highlight in the report  [30]")
    p.add_argument("--top-n-threshold", type=int, dest="top_n", default=100,
                   help="N used for 'appears in top-N of all runs' metric  [100]")
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

RANK_COL  = "rank"
MEAN_COL  = "mean_remaining"
STD_COL   = "std_remaining"
WIN_COL   = "win_rate"
KEY_COLS  = ["x", "h"]


def load_csvs(paths: list[str]) -> list[pd.DataFrame]:
    dfs = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            print(f"  [!] File not found: {p} — skipping", file=sys.stderr)
            continue
        df = pd.read_csv(path)
        required = set(KEY_COLS + [RANK_COL, MEAN_COL, STD_COL, WIN_COL])
        missing  = required - set(df.columns)
        if missing:
            print(f"  [!] {p} missing columns {missing} — skipping",
                  file=sys.stderr)
            continue
        dfs.append(df)
        print(f"  Loaded  {p}  ({len(df):,} rows)")
    return dfs


def borda_score(rank: float, n_total: int) -> float:
    """Lower rank → higher Borda score (like points in a race)."""
    return n_total - rank + 1


# ── Main aggregation ──────────────────────────────────────────────────────────

def build_consensus(dfs: list[pd.DataFrame], top_n: int) -> pd.DataFrame:
    n_runs    = len(dfs)
    n_total   = len(dfs[0])   # all runs have same (x,h) pairs

    # Suffix each run
    tagged = []
    for i, df in enumerate(dfs):
        sub = df[KEY_COLS + [RANK_COL, MEAN_COL, STD_COL, WIN_COL]].copy()
        sub = sub.rename(columns={
            RANK_COL : f"rank_{i}",
            MEAN_COL : f"mean_{i}",
            STD_COL  : f"std_{i}",
            WIN_COL  : f"wr_{i}",
        })
        tagged.append(sub)

    # Merge all runs on (x, h)
    merged = tagged[0]
    for sub in tagged[1:]:
        merged = pd.merge(merged, sub, on=KEY_COLS, how="inner")

    rank_cols = [f"rank_{i}" for i in range(n_runs)]
    mean_cols = [f"mean_{i}" for i in range(n_runs)]
    std_cols  = [f"std_{i}"  for i in range(n_runs)]
    wr_cols   = [f"wr_{i}"   for i in range(n_runs)]

    # ── Borda count (sum of (n_total - rank + 1) across runs, higher = better)
    for i in range(n_runs):
        merged[f"borda_{i}"] = borda_score(merged[f"rank_{i}"], n_total)
    borda_cols = [f"borda_{i}" for i in range(n_runs)]
    merged["borda_total"]  = merged[borda_cols].sum(axis=1)

    # ── Aggregate statistics
    merged["avg_rank"]          = merged[rank_cols].mean(axis=1)
    merged["rank_std"]          = merged[rank_cols].std(axis=1)   # low = consistent
    merged["best_rank"]         = merged[rank_cols].min(axis=1)
    merged["worst_rank"]        = merged[rank_cols].max(axis=1)
    merged["rank_range"]        = merged["worst_rank"] - merged["best_rank"]

    merged["avg_mean_remaining"] = merged[mean_cols].mean(axis=1)
    merged["std_of_means"]       = merged[mean_cols].std(axis=1)  # cross-run variance
    merged["avg_std_remaining"]  = merged[std_cols].mean(axis=1)  # within-run variance
    merged["avg_win_rate"]       = merged[wr_cols].mean(axis=1)

    # ── How many runs had this pair in their top-N?
    top_n_flags = [(merged[f"rank_{i}"] <= top_n).astype(int)
                   for i in range(n_runs)]
    merged["top_n_appearances"] = sum(top_n_flags)   # 0..n_runs

    # ── Composite score (normalised): weight avg_mean_remaining + stability bonus
    mean_min  = merged["avg_mean_remaining"].min()
    mean_max  = merged["avg_mean_remaining"].max()
    norm_mean = (merged["avg_mean_remaining"] - mean_min) / (mean_max - mean_min + 1e-9)

    rank_std_max = merged["rank_std"].max()
    stability    = 1.0 - merged["rank_std"] / (rank_std_max + 1e-9)

    # composite = 80% mean score + 20% stability
    merged["composite_score"] = 0.80 * norm_mean + 0.20 * stability

    # ── Consensus rank (sort: composite desc, avg_mean desc, avg_rank asc)
    merged = merged.sort_values(
        ["composite_score", "avg_mean_remaining", "avg_rank"],
        ascending=[False, False, True]
    ).reset_index(drop=True)
    merged["consensus_rank"] = merged.index + 1

    return merged, rank_cols, mean_cols


# ── Output CSV ────────────────────────────────────────────────────────────────

EXPORT_COLS = [
    "consensus_rank", "x", "h",
    "avg_mean_remaining", "avg_win_rate",
    "avg_rank", "rank_std", "best_rank", "worst_rank", "rank_range",
    "borda_total", "top_n_appearances",
    "std_of_means", "avg_std_remaining",
    "composite_score",
]


def save_consensus(df: pd.DataFrame, path: str, n_runs: int):
    # Add per-run columns at the end for full transparency
    per_run = []
    for i in range(n_runs):
        per_run += [f"rank_{i}", f"mean_{i}", f"std_{i}", f"wr_{i}"]
    cols = EXPORT_COLS + [c for c in per_run if c in df.columns]
    df[cols].to_csv(path, index=False)
    print(f"\n  Consensus CSV  →  {path}  ({len(df):,} rows)")


# ── Insight report ────────────────────────────────────────────────────────────

def sep(char="─", width=70):
    print(char * width)


def insight_report(df: pd.DataFrame, dfs_orig: list[pd.DataFrame],
                   rank_cols: list, mean_cols: list, top_k: int, top_n: int):

    n_runs  = len(dfs_orig)
    n_total = len(df)

    sep("═")
    print("  KYBER CONSENSUS  —  Insight Report")
    sep("═")
    print(f"  Runs merged   : {n_runs}")
    print(f"  (x,h) pairs   : {n_total:,}")
    print(f"  Top-N threshold: {top_n}")

    # ── 1. Per-run top-1 summary ──────────────────────────────────────────────
    sep()
    print("  PER-RUN WINNERS")
    sep()
    for i, orig in enumerate(dfs_orig):
        best = orig.sort_values("rank").iloc[0]
        print(f"  Run {i}:  x={int(best['x']):>3}  h={int(best['h']):>2}"
              f"  mean_remaining={best['mean_remaining']:>10.2f}"
              f"  win_rate={best['win_rate']:.4f}")

    # ── 2. Consensus top-K ────────────────────────────────────────────────────
    sep()
    print(f"  CONSENSUS TOP-{top_k}")
    sep()
    hdr = (f"  {'#':>4}  {'x':>4}  {'h':>3}  "
           f"{'avg_mean':>10}  {'avg_rank':>9}  {'rank_std':>9}  "
           f"{'top_n_apps':>10}  {'borda':>8}  {'composite':>10}")
    print(hdr)
    print("  " + "─" * 68)
    for _, row in df.head(top_k).iterrows():
        print(f"  {int(row['consensus_rank']):>4}  "
              f"{int(row['x']):>4}  {int(row['h']):>3}  "
              f"{row['avg_mean_remaining']:>10.2f}  "
              f"{row['avg_rank']:>9.1f}  "
              f"{row['rank_std']:>9.1f}  "
              f"{int(row['top_n_appearances']):>10}  "
              f"{row['borda_total']:>8.0f}  "
              f"{row['composite_score']:>10.4f}")

    # ── 3. Stability analysis ─────────────────────────────────────────────────
    sep()
    print("  RANK CONSISTENCY  (pairs in top-N of ALL runs)")
    sep()
    for threshold in [10, 50, 100, 500]:
        # Re-check per original dfs
        mask = np.ones(len(df), dtype=bool)
        for i, orig in enumerate(dfs_orig):
            top_set = set(zip(orig[orig["rank"] <= threshold]["x"],
                              orig[orig["rank"] <= threshold]["h"]))
            in_top = df.apply(lambda r: (r["x"], r["h"]) in top_set, axis=1)
            mask &= in_top.values
        count = mask.sum()
        print(f"  In top-{threshold:>5} of ALL {n_runs} runs: "
              f"{count:>5} pairs  ({count/n_total*100:.2f}%)")

    # ── 4. Spearman / Kendall rank correlations ───────────────────────────────
    sep()
    print("  RANK CORRELATIONS BETWEEN RUNS")
    sep()
    print(f"  {'':8}", end="")
    for i in range(n_runs):
        print(f"  Run {i}  ", end="")
    print()
    for i in range(n_runs):
        print(f"  Run {i}  ", end="")
        for j in range(n_runs):
            if i == j:
                print(f"{'  1.0000':>8}", end="")
            elif j > i:
                rho, _ = spearmanr(df[f"rank_{i}"], df[f"rank_{j}"])
                print(f"  {rho:6.4f}", end="")
            else:
                tau, _ = kendalltau(df[f"rank_{i}"].values[:5000],
                                    df[f"rank_{j}"].values[:5000])
                print(f"  {tau:6.4f}", end="")
        print()
    print("  (upper triangle = Spearman ρ, lower triangle = Kendall τ)")

    # ── 5. Cross-run agreement on mean_remaining ──────────────────────────────
    sep()
    print("  MEAN_REMAINING CROSS-RUN AGREEMENT")
    sep()
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            corr = df[mean_cols[i]].corr(df[mean_cols[j]])
            diff = (df[mean_cols[i]] - df[mean_cols[j]]).abs()
            print(f"  Run {i} vs Run {j}:  Pearson r = {corr:.6f}  "
                  f"  mean |Δ| = {diff.mean():.2f}  "
                  f"  max |Δ| = {diff.max():.2f}")

    # ── 6. Optimal h distribution for top pairs ───────────────────────────────
    sep()
    print(f"  h-VALUE DISTRIBUTION  (top-{top_k} consensus pairs)")
    sep()
    top_h = df.head(top_k)["h"].values
    unique_h, counts = np.unique(top_h, return_counts=True)
    for h_val, cnt in sorted(zip(unique_h, counts), key=lambda t: -t[1])[:15]:
        bar = "█" * cnt
        print(f"  h={int(h_val):>2}  {bar:<40}  {cnt}")

    # ── 7. Optimal x distribution for top pairs ───────────────────────────────
    sep()
    print(f"  x-VALUE DISTRIBUTION  (top-{top_k} consensus pairs)")
    sep()
    top_x = df.head(top_k)["x"].values
    print(f"  min={top_x.min():.0f}  max={top_x.max():.0f}  "
          f"mean={top_x.mean():.1f}  median={np.median(top_x):.1f}  "
          f"std={top_x.std():.1f}")

    # Bucket into ranges
    buckets = [(0,50),(51,100),(101,150),(151,200),(201,250),(251,300)]
    for lo, hi in buckets:
        cnt = ((top_x >= lo) & (top_x <= hi)).sum()
        bar = "█" * cnt
        print(f"  x=[{lo:>3},{hi:>3}]  {bar:<40}  {cnt}")

    # ── 8. Best fully-consistent pair (top-N in all runs, highest avg_mean) ───
    sep()
    print(f"  RECOMMENDED PAIRS  (in top-{top_n} of all runs, sorted by avg_mean)")
    sep()
    consistent = df[df["top_n_appearances"] == n_runs].head(20)
    if len(consistent) == 0:
        print(f"  No pairs found in top-{top_n} of ALL {n_runs} runs.")
        print("  (Try --top-n-threshold with a larger N)")
    else:
        for _, row in consistent.iterrows():
            print(f"  consensus #{int(row['consensus_rank']):<5}  "
                  f"x={int(row['x']):>3}  h={int(row['h']):>2}  "
                  f"avg_mean={row['avg_mean_remaining']:>10.2f}  "
                  f"rank_std={row['rank_std']:>6.1f}  "
                  f"borda={row['borda_total']:>8.0f}")

    # ── 9. Diversity of top pairs: unique x and h values ──────────────────────
    sep()
    print(f"  DIVERSITY STATS  (top-{top_k})")
    sep()
    top_df = df.head(top_k)
    print(f"  Unique x values : {top_df['x'].nunique()}")
    print(f"  Unique h values : {top_df['h'].nunique()}")
    print(f"  x range         : [{top_df['x'].min():.0f}, {top_df['x'].max():.0f}]")
    print(f"  h range         : [{top_df['h'].min():.0f}, {top_df['h'].max():.0f}]")
    print(f"  Mean composite  : {top_df['composite_score'].mean():.4f}")
    print(f"  Pairs with rank_std < 100 : "
          f"{(top_df['rank_std'] < 100).sum()} / {top_k}")

    # ── 10. Diminishing returns: x vs avg_mean for best-h-per-x ──────────────
    sep()
    print("  BEST h PER x  (best consensus avg_mean for each x)")
    sep()
    best_per_x = (df.groupby("x")["avg_mean_remaining"]
                    .max()
                    .reset_index()
                    .sort_values("avg_mean_remaining", ascending=False)
                    .head(20))
    print(f"  {'x':>5}  {'best_avg_mean':>14}")
    for _, row in best_per_x.iterrows():
        print(f"  {int(row['x']):>5}  {row['avg_mean_remaining']:>14.2f}")

    sep("═")
    print("  Done.")
    sep("═")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("\n" + "═" * 70)
    print("  Kyber Rank Merger")
    print("═" * 70)

    dfs = load_csvs(args.files)
    if len(dfs) < 2:
        print("\n[ERROR] Need at least 2 valid CSV files.", file=sys.stderr)
        sys.exit(1)

    print(f"\n  Merging {len(dfs)} CSVs …")
    consensus_df, rank_cols, mean_cols = build_consensus(dfs, top_n=args.top_n)

    save_consensus(consensus_df, args.out, n_runs=len(dfs))

    insight_report(
        df        = consensus_df,
        dfs_orig  = dfs,
        rank_cols = rank_cols,
        mean_cols = mean_cols,
        top_k     = args.top,
        top_n     = args.top_n,
    )


if __name__ == "__main__":
    main()
