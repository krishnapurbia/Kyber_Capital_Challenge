#!/usr/bin/env python3
"""
kyber_sweep_cache.py
═══════════════════════════════════════════════════════════════════════════════
Sweeps every h in [H_START, H_END] (default 5 → 60) over the full x range,
finds the top-N best x values per h, and writes everything to a CSV so future
runs can skip already-computed h values.

Resumable:  If the CSV already exists, any h that already has a row is skipped
            automatically.  Delete the CSV (or specific rows) to force a rerun.

Output CSV columns:
    h, rank, x, avg_remaining, std_remaining, win_rate,
    avg_thorn_cross, avg_fence_cross, avg_diag_dist, sharpe

Usage:
    python kyber_sweep_cache.py                    # defaults
    python kyber_sweep_cache.py --h-start 5 --h-end 30
    python kyber_sweep_cache.py --top-n 3 --x-step 5 --n-sims 100 --seeds 10
    python kyber_sweep_cache.py --force            # rerun everything
    python kyber_sweep_cache.py --force-h 20 25    # rerun only h=20,25

Options:
    --h-start   First h value to sweep                   [5]
    --h-end     Last  h value to sweep (inclusive)       [60]
    --top-n     How many best x values to save per h     [2]
    --x-min     Minimum x                                [0]
    --x-max     Maximum x                                [300]
    --x-step    Step between x values                    [10]
    --n-sims    Simulation iterations per seed           [50]
    --seeds     Independent seeds to average over        [5]
    --seed      Master RNG seed (None = time-based)      [None]
    --csv       Output CSV path                          [kyber_best.csv]
    --metric    Ranking metric: avg_remaining | sharpe   [avg_remaining]
    --force     Ignore cache, recompute all h values
    --force-h   Recompute specific h values only (space-separated)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from kyber_simulate import simulate_competition, X_MIN, X_MAX, H_MIN, H_MAX
except ImportError:
    print("[ERROR] kyber_simulate.py not found in the current directory.", file=sys.stderr)
    sys.exit(1)


# ── CSV schema ────────────────────────────────────────────────────────────────
CSV_COLS = [
    "h", "rank", "x",
    "avg_remaining", "std_remaining", "win_rate",
    "avg_thorn_cross", "avg_fence_cross", "avg_diag_dist",
    "sharpe",
]


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sweep h=[5,60], cache top-N best x per h to CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--h-start",  type=int, default=5,    dest="h_start")
    p.add_argument("--h-end",    type=int, default=60,   dest="h_end")
    p.add_argument("--top-n",    type=int, default=2,    dest="top_n")
    p.add_argument("--x-min",    type=int, default=X_MIN, dest="x_min")
    p.add_argument("--x-max",    type=int, default=X_MAX, dest="x_max")
    p.add_argument("--x-step",   type=int, default=10,   dest="x_step")
    p.add_argument("--n-sims",   type=int, default=50,   dest="n_sims")
    p.add_argument("--seeds",    type=int, default=5)
    p.add_argument("--seed",     type=int, default=None)
    p.add_argument("--csv",      type=str, default="kyber_best.csv")
    p.add_argument("--metric",   type=str, default="avg_remaining",
                   choices=["avg_remaining", "sharpe"],
                   help="Metric used to rank x values per h  [avg_remaining]")
    p.add_argument("--force",    action="store_true",
                   help="Ignore cache entirely; recompute all h values")
    p.add_argument("--force-h",  type=int, nargs="+", default=[],
                   dest="force_h",
                   help="Force recompute for specific h values only")
    return p.parse_args()


# ── Simulation driver ─────────────────────────────────────────────────────────

def sweep_h(h_val: int, x_vals: np.ndarray,
            n_sims: int, seeds: int, h_seed: int) -> pd.DataFrame:
    """Run the full x-sweep for one h value and return the raw results."""
    submissions = [
        {"x": int(x), "h": h_val, "label": f"x{int(x)}_h{h_val}"}
        for x in x_vals
    ]
    df = simulate_competition(
        submissions=submissions,
        n_sims=n_sims,
        n_seeds=seeds,
        master_seed=h_seed,
    )
    df["h_fixed"] = h_val
    df = df.sort_values("x").reset_index(drop=True)
    return df


def pick_top_n(df: pd.DataFrame, h_val: int, top_n: int,
               metric: str) -> pd.DataFrame:
    """
    From a full-sweep DataFrame, extract the top-N rows by the chosen metric
    and return a tidy DataFrame matching CSV_COLS.
    """
    if metric == "sharpe":
        df = df.copy()
        df["sharpe"] = df["avg_remaining"] / (df["std_remaining"] + 1.0)
        sort_col = "sharpe"
    else:
        df = df.copy()
        df["sharpe"] = df["avg_remaining"] / (df["std_remaining"] + 1.0)
        sort_col = "avg_remaining"

    top = (df.sort_values(sort_col, ascending=False)
             .head(top_n)
             .reset_index(drop=True))

    rows = []
    for rank_idx, row in top.iterrows():
        rows.append({
            "h":               h_val,
            "rank":            int(rank_idx) + 1,
            "x":               int(row["x"]),
            "avg_remaining":   round(float(row["avg_remaining"]), 4),
            "std_remaining":   round(float(row["std_remaining"]), 4),
            "win_rate":        round(float(row["win_rate"]), 6),
            "avg_thorn_cross": round(float(row["avg_thorn_cross"]), 4),
            "avg_fence_cross": round(float(row["avg_fence_cross"]), 4),
            "avg_diag_dist":   round(float(row["avg_diag_dist"]), 4),
            "sharpe":          round(float(row["sharpe"]), 6),
        })
    return pd.DataFrame(rows, columns=CSV_COLS)


# ── Cache helpers ─────────────────────────────────────────────────────────────

def load_cache(csv_path: Path) -> pd.DataFrame:
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        # Ensure all expected columns are present
        for col in CSV_COLS:
            if col not in df.columns:
                df[col] = np.nan
        return df[CSV_COLS]
    return pd.DataFrame(columns=CSV_COLS)


def save_cache(df: pd.DataFrame, csv_path: Path):
    df.to_csv(csv_path, index=False)


def already_computed(cache: pd.DataFrame, h_val: int) -> bool:
    return int(h_val) in cache["h"].values


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.seed is None:
        args.seed = int(time.time() * 1000) % (2**32 - 1)

    h_range = list(range(args.h_start, args.h_end + 1))
    x_vals  = np.arange(
        max(args.x_min, X_MIN),
        min(args.x_max, X_MAX) + 1,
        args.x_step,
        dtype=int,
    )

    csv_path = Path(args.csv)
    cache    = pd.DataFrame(columns=CSV_COLS) if args.force else load_cache(csv_path)

    # Remove forced-h rows from cache so they get recomputed
    if args.force_h:
        cache = cache[~cache["h"].isin(args.force_h)].reset_index(drop=True)

    print()
    print("═" * 65)
    print("  Kyber Sweep Cache  —  h ∈ [{}, {}]".format(args.h_start, args.h_end))
    print("═" * 65)
    print(f"  h values    : {len(h_range)}  ({args.h_start} → {args.h_end})")
    print(f"  x range     : {x_vals[0]} → {x_vals[-1]}  (step {args.x_step})")
    print(f"  x points    : {len(x_vals)}")
    print(f"  top-N saved : {args.top_n}  (ranked by {args.metric})")
    print(f"  n_sims      : {args.n_sims}")
    print(f"  seeds       : {args.seeds}")
    print(f"  iterations  : {args.n_sims * args.seeds:,} per h")
    print(f"  master seed : {args.seed}")
    print(f"  csv output  : {csv_path.resolve()}")
    print()

    skipped = 0
    computed = 0
    all_new_rows = []

    for h_val in h_range:
        if already_computed(cache, h_val):
            skipped += 1
            print(f"  h={h_val:>3}  [CACHED — skipping]")
            continue

        # Deterministic but distinct seed per h
        h_seed = (args.seed + h_val * 7919) % (2**32 - 1)

        print(f"  h={h_val:>3}  sweeping {len(x_vals)} x-values … ", end="", flush=True)
        t0 = time.time()

        raw_df  = sweep_h(h_val, x_vals, args.n_sims, args.seeds, h_seed)
        top_df  = pick_top_n(raw_df, h_val, args.top_n, args.metric)
        all_new_rows.append(top_df)
        computed += 1

        elapsed = time.time() - t0
        best_row = top_df.iloc[0]
        print(f"done ({elapsed:.1f}s)  best x={int(best_row['x'])}  "
              f"avg_rem={best_row['avg_remaining']:.2f}  "
              f"sharpe={best_row['sharpe']:.3f}")

        # Append and save after every h so progress is never lost
        if all_new_rows:
            new_data = pd.concat(all_new_rows, ignore_index=True)
            updated  = pd.concat([cache, new_data], ignore_index=True)
            save_cache(updated, csv_path)
            # Update the in-memory cache too
            cache = updated

    print()
    print(f"  Computed : {computed} h values")
    print(f"  Skipped  : {skipped} h values (from cache)")
    print(f"  CSV rows : {len(cache)}")
    print(f"  Saved  → {csv_path.resolve()}")
    print()

    # Pretty-print summary table
    print("═" * 65)
    print("  TOP-1 SUMMARY  (best x per h)")
    print("═" * 65)
    top1 = cache[cache["rank"] == 1].sort_values("h")
    print(f"  {'h':>4}  {'best_x':>6}  {'avg_rem':>9}  "
          f"{'win%':>7}  {'sharpe':>8}")
    print("  " + "─" * 45)
    for _, row in top1.iterrows():
        print(f"  {int(row['h']):>4}  {int(row['x']):>6}  "
              f"{row['avg_remaining']:>9.2f}  "
              f"{row['win_rate']*100:>6.1f}%  "
              f"{row['sharpe']:>8.4f}")
    print("═" * 65)
    print()


if __name__ == "__main__":
    main()