#!/usr/bin/env python3
"""
kyber_simulate.py
════════════════════════════════════════════════════════════════════
Simulates the Kyber challenge competition with full crowd penalty.

Two modes:
  single        Your (x, h) + top-C from consensus CSV.
                See how you score alongside the best known strategies.

  distributed   Top-C from consensus CSV + Y pairs sampled from a
                Normal distribution fitted to the top-X consensus pairs.
                Models a crowded field where many players pick near-
                optimal but noisy values.

Usage — single mode:
    python kyber_simulate.py single \\
        --x 200 --h 20 \\
        --csv kyber_consensus.csv \\
        --top-c 20 \\
        --seeds 10

Usage — distributed mode:
    python kyber_simulate.py distributed \\
        --csv kyber_consensus.csv \\
        --top-c 20 \\
        --top-x 10 \\          # fit Normal to the top-10 pairs
        --y-samples 100 \\     # sample 100 extra (x,h) pairs
        --seeds 10

Common options:
    --n-sims    50      competition simulation iterations  [50]
    --seeds     5       number of independent seeds to average over  [5]
    --seed      [None]  master RNG seed (uses system time if omitted)
    --top-c     20      top-C pairs to pull from the consensus CSV  [20]
    --verbose           print per-submission detail table

New per-submission metrics (reported in results table):
    avg_thorn_cross   – avg number of thorn-enhanced fences crossed per walk
                        (fences where toll > MAX_T = 30)
    avg_fence_cross   – avg number of plain fences crossed per walk
                        (fences where toll ≤ MAX_T = 30)
    avg_diag_dist     – avg perpendicular distance from the diagonal (y=x line)
                        computed as mean of |cx − cy| / √2 over all 2·GRID+1
                        positions visited on the walk
"""

import argparse
import sys
import time

import numpy as np
import pandas as pd


# ── Game constants ────────────────────────────────────────────────────────────

GRID  = 50
MIN_T = 5
MAX_T = 30
THORN = 40
NSIMS = 50

X_MIN, X_MAX = 0,   300
H_MIN, H_MAX = 5,    60

_SQRT2 = np.sqrt(2.0)


# ── Core simulation ───────────────────────────────────────────────────────────

def build_base_fences(rng: np.random.Generator):
    """Base fence tolls (shared by all submissions in one iteration)."""
    ef = rng.integers(MIN_T, MAX_T + 1, (GRID, GRID + 1)).astype(np.float64)
    nf = rng.integers(MIN_T, MAX_T + 1, (GRID + 1, GRID)).astype(np.float64)
    return ef, nf


def add_thorns(ef_base: np.ndarray, nf_base: np.ndarray,
               x_val: int, rng: np.random.Generator):
    """Copy base grid and add 2*x_val thorns at random positions."""
    ef = ef_base.copy()
    nf = nf_base.copy()
    n  = 2 * x_val
    if n > 0:
        idxs = rng.integers(0, ef.size + nf.size, size=n)
        east_mask = idxs < ef.size
        np.add.at(ef.ravel(), idxs[east_mask],             THORN)
        np.add.at(nf.ravel(), idxs[~east_mask] - ef.size,  THORN)
    return ef, nf


def walk_once(ef: np.ndarray, nf: np.ndarray,
              h: int, rng: np.random.Generator):
    """
    Walk from (0,0) to (GRID,GRID) using the h-threshold rule.

    Returns
    -------
    eu             : east-fence use-count array, shape (GRID, GRID+1)
    nu             : north-fence use-count array, shape (GRID+1, GRID)
    raw_toll       : total toll paid (before any crowd penalty)
    thorn_crossings: number of fences crossed whose toll > MAX_T
    fence_crossings: number of fences crossed whose toll <= MAX_T
    avg_diag_dist  : mean perpendicular distance from diagonal y=x,
                     computed as mean( |cx - cy| / √2 ) over all
                     2·GRID + 1 positions visited (including start & end)
    """
    cx, cy   = 0, 0
    eu       = np.zeros((GRID, GRID + 1), dtype=np.int32)
    nu       = np.zeros((GRID + 1, GRID), dtype=np.int32)
    raw_toll = 0.0

    thorn_crossings = 0
    fence_crossings = 0

    # Accumulate |cx - cy| / √2 for every position visited on the path.
    # The path has exactly 2·GRID steps, visiting 2·GRID + 1 points
    # (start (0,0) + one point after each step, ending at (GRID,GRID)).
    diag_dist_sum = abs(cx - cy) / _SQRT2   # record starting position

    while cx < GRID or cy < GRID:
        can_e = cx < GRID
        can_n = cy < GRID

        if can_e and can_n:
            te, tn = ef[cx, cy], nf[cx, cy]
            e_ok   = te <= h
            n_ok   = tn <= h
            if   e_ok and not n_ok: mv = "e"
            elif n_ok and not e_ok: mv = "n"
            else:                   mv = "e" if rng.random() < 0.5 else "n"
        elif can_e:
            mv = "e"
        else:
            mv = "n"

        if mv == "e":
            toll = ef[cx, cy]
            raw_toll += toll
            eu[cx, cy] += 1
            cx += 1
        else:
            toll = nf[cx, cy]
            raw_toll += toll
            nu[cx, cy] += 1
            cy += 1

        # ── classify the fence just crossed ─────────────────────────
        if toll > MAX_T:
            thorn_crossings += 1
        else:
            fence_crossings += 1

        # ── record diagonal distance at the new position ─────────────
        diag_dist_sum += abs(cx - cy) / _SQRT2

    # Total positions visited = 2·GRID + 1
    avg_diag_dist = diag_dist_sum / (2 * GRID + 1)

    return eu, nu, raw_toll, thorn_crossings, fence_crossings, avg_diag_dist


def penalised_toll_from_parts(ef: np.ndarray, nf: np.ndarray,
                               eu: np.ndarray, nu: np.ndarray,
                               frac_e: np.ndarray, frac_n: np.ndarray) -> float:
    """
    Crowd-penalised toll for ONE submission given global usage fractions.
    final_toll = Σ_fence  toll_fence × (1 + frac_fence)³  ×  (1 if crossed else 0)
    """
    pen_e = ef * (1.0 + frac_e) ** 3
    pen_n = nf * (1.0 + frac_n) ** 3
    return float((pen_e * eu).sum() + (pen_n * nu).sum())


# ── One competition iteration (all submissions on same base grid) ─────────────

def run_one_iteration(submissions: list[dict],
                      base_rng: np.random.Generator,
                      sub_rngs: list[np.random.Generator]) -> list[dict]:
    """
    One competition iteration.

    submissions : list of dicts  {'x': int, 'h': int, 'label': str}
    base_rng    : RNG for the shared base fence grid
    sub_rngs    : per-submission RNG (thorns + walk)

    Returns list of dicts per submission (same order):
        pen_toll, thorn_crossings, fence_crossings, avg_diag_dist
    """
    n = len(submissions)

    # Shared base fences
    ef_base, nf_base = build_base_fences(base_rng)

    # (ef_i, nf_i, eu_i, nu_i, raw_toll_i, thorn_i, fence_i, diag_i)
    sub_data = []

    for idx, sub in enumerate(submissions):
        rng = sub_rngs[idx]
        ef_i, nf_i = add_thorns(ef_base, nf_base, sub["x"], rng)
        eu_i, nu_i, toll_i, thorn_i, fence_i, diag_i = walk_once(
            ef_i, nf_i, sub["h"], rng)
        sub_data.append((ef_i, nf_i, eu_i, nu_i, toll_i, thorn_i, fence_i, diag_i))

    # Global usage fractions (sum over all submissions / n)
    total_eu = sum(d[2] for d in sub_data).astype(np.float64)
    total_nu = sum(d[3] for d in sub_data).astype(np.float64)
    frac_e   = total_eu / n
    frac_n   = total_nu / n

    results = []
    for ef_i, nf_i, eu_i, nu_i, _, thorn_i, fence_i, diag_i in sub_data:
        results.append({
            "pen_toll"       : penalised_toll_from_parts(
                                   ef_i, nf_i, eu_i, nu_i, frac_e, frac_n),
            "thorn_crossings": thorn_i,
            "fence_crossings": fence_i,
            "avg_diag_dist"  : diag_i,
        })

    return results


# ── Full competition simulation ───────────────────────────────────────────────

def simulate_competition(
        submissions : list[dict],
        n_sims      : int = NSIMS,
        n_seeds     : int = 5,
        master_seed : int = 42,
        verbose     : bool = False,
) -> pd.DataFrame:
    """
    Run the full competition n_sims × n_seeds times and aggregate.

    Returns a DataFrame with one row per submission including the new columns:
        avg_thorn_cross  – avg thorn-fence crossings per walk
        avg_fence_cross  – avg plain-fence crossings per walk
        avg_diag_dist    – avg perpendicular distance from diagonal per walk
    """
    n = len(submissions)
    master_rng = np.random.default_rng(master_seed)

    # Accumulators per submission label
    all_remaining    = {sub["label"]: [] for sub in submissions}
    all_thorn_cross  = {sub["label"]: [] for sub in submissions}
    all_fence_cross  = {sub["label"]: [] for sub in submissions}
    all_diag_dist    = {sub["label"]: [] for sub in submissions}

    # Generate seeds for each (n_seeds × n_sims) iteration
    seed_matrix = master_rng.integers(0, 10_000_000, size=(n_seeds, n_sims))

    for s_idx in range(n_seeds):
        for sim_idx in range(n_sims):
            iter_seed = int(seed_matrix[s_idx, sim_idx])

            iter_rng = np.random.default_rng(iter_seed)
            base_rng = np.random.default_rng(iter_rng.integers(0, 10_000_000))
            sub_rngs = [np.random.default_rng(iter_rng.integers(0, 10_000_000))
                        for _ in range(n)]

            iter_results = run_one_iteration(submissions, base_rng, sub_rngs)

            for i, sub in enumerate(submissions):
                lbl   = sub["label"]
                start = 2500 + sub["x"]
                r     = iter_results[i]
                all_remaining[lbl].append(start - r["pen_toll"])
                all_thorn_cross[lbl].append(r["thorn_crossings"])
                all_fence_cross[lbl].append(r["fence_crossings"])
                all_diag_dist[lbl].append(r["avg_diag_dist"])

    # Aggregate
    records = []
    for sub in submissions:
        lbl   = sub["label"]
        start = 2500 + sub["x"]
        rems  = np.array(all_remaining[lbl])
        thorn = np.array(all_thorn_cross[lbl])
        fence = np.array(all_fence_cross[lbl])
        diag  = np.array(all_diag_dist[lbl])
        records.append({
            "label"           : lbl,
            "x"               : sub["x"],
            "h"               : sub["h"],
            "start_apples"    : start,
            "avg_remaining"   : float(rems.mean()),
            "std_remaining"   : float(rems.std()),
            "win_rate"        : float((rems >= 0).mean()),
            "min_remaining"   : float(rems.min()),
            "max_remaining"   : float(rems.max()),
            # ── new metrics ──────────────────────────────────────────
            "avg_thorn_cross" : float(thorn.mean()),
            "avg_fence_cross" : float(fence.mean()),
            "avg_diag_dist"   : float(diag.mean()),
        })

    result_df = pd.DataFrame(records).sort_values("avg_remaining",
                                                   ascending=False)
    result_df["competition_rank"] = range(1, len(result_df) + 1)
    return result_df


# ── Submission builders ───────────────────────────────────────────────────────

def load_top_c(csv_path: str, top_c: int) -> list[dict]:
    """Load the top-C rows from the consensus CSV (sorted by consensus_rank)."""
    df = pd.read_csv(csv_path)
    rank_col = "consensus_rank" if "consensus_rank" in df.columns else "rank"
    df = df.sort_values(rank_col).head(top_c)
    subs = []
    for i, row in df.iterrows():
        x = int(row["x"])
        h = int(row["h"])
        subs.append({"x": x, "h": h,
                     "label": f"csv_top{int(row[rank_col])}_x{x}_h{h}"})
    return subs


def sample_normal_submissions(
        top_x_subs : list[dict],
        y_samples  : int,
        rng        : np.random.Generator,
) -> list[dict]:
    """
    Fit a Normal distribution to the (x, h) of top_x_subs and sample
    y_samples new submissions, clipping to valid ranges.
    """
    xs = np.array([s["x"] for s in top_x_subs], dtype=float)
    hs = np.array([s["h"] for s in top_x_subs], dtype=float)

    mu_x, sig_x = xs.mean(), xs.std() + 1e-3
    mu_h, sig_h = hs.mean(), hs.std() + 1e-3

    sampled_x = rng.normal(mu_x, sig_x, size=y_samples).round().astype(int)
    sampled_h = rng.normal(mu_h, sig_h, size=y_samples).round().astype(int)

    sampled_x = np.clip(sampled_x, X_MIN, X_MAX)
    sampled_h = np.clip(sampled_h, H_MIN, H_MAX)

    subs = []
    for i, (x, h) in enumerate(zip(sampled_x, sampled_h)):
        subs.append({"x": int(x), "h": int(h),
                     "label": f"norm_sample_{i}_x{x}_h{h}"})
    return subs


# ── Result printer ────────────────────────────────────────────────────────────

def print_results(result_df: pd.DataFrame, target_label: str = None,
                  top_k: int = 20):
    print()
    print("═" * 106)
    print("  COMPETITION RESULTS  (with crowd penalty)")
    print("═" * 106)
    print(f"  {'Rank':>4}  {'Label':<35}  {'x':>4}  {'h':>3}  "
          f"{'avg_rem':>10}  {'std':>8}  {'win%':>6}  "
          f"{'thorn_x':>8}  {'fence_x':>8}  {'diag_d':>8}")
    print("  " + "─" * 102)

    for _, row in result_df.head(top_k).iterrows():
        marker = " ◄ YOU" if row["label"] == target_label else ""
        print(f"  {int(row['competition_rank']):>4}  "
              f"{row['label']:<35}  "
              f"{int(row['x']):>4}  {int(row['h']):>3}  "
              f"{row['avg_remaining']:>10.2f}  "
              f"{row['std_remaining']:>8.2f}  "
              f"{row['win_rate']*100:>5.1f}%  "
              f"{row['avg_thorn_cross']:>8.2f}  "
              f"{row['avg_fence_cross']:>8.2f}  "
              f"{row['avg_diag_dist']:>8.4f}"
              f"{marker}")

    if target_label and target_label not in result_df.head(top_k)["label"].values:
        tgt = result_df[result_df["label"] == target_label]
        if len(tgt):
            row = tgt.iloc[0]
            print("  ...")
            print(f"  {int(row['competition_rank']):>4}  "
                  f"{row['label']:<35}  "
                  f"{int(row['x']):>4}  {int(row['h']):>3}  "
                  f"{row['avg_remaining']:>10.2f}  "
                  f"{row['std_remaining']:>8.2f}  "
                  f"{row['win_rate']*100:>5.1f}%  "
                  f"{row['avg_thorn_cross']:>8.2f}  "
                  f"{row['avg_fence_cross']:>8.2f}  "
                  f"{row['avg_diag_dist']:>8.4f}  ◄ YOU")

    print()
    print("  Column guide:")
    print("    thorn_x  – avg fences crossed where toll > MAX_T (thorn-enhanced)")
    print("    fence_x  – avg fences crossed where toll ≤ MAX_T (plain only)")
    print("    diag_d   – avg perpendicular distance from diagonal y=x  "
          "(mean |cx−cy|/√2 over all 2·GRID+1 path positions)")
    print()

    best = result_df.iloc[0]
    print(f"  Winner       : {best['label']}  (x={best['x']}, h={best['h']})")
    print(f"  Best avg     : {best['avg_remaining']:.2f}")

    if target_label:
        tgt = result_df[result_df["label"] == target_label]
        if len(tgt):
            row = tgt.iloc[0]
            print(f"  Your avg     : {row['avg_remaining']:.2f}  "
                  f"(rank {int(row['competition_rank'])} / {len(result_df)})")
            gap = best["avg_remaining"] - row["avg_remaining"]
            print(f"  Gap to first : {gap:.2f} apples")
            print(f"  Your thorn_x : {row['avg_thorn_cross']:.2f}")
            print(f"  Your fence_x : {row['avg_fence_cross']:.2f}")
            print(f"  Your diag_d  : {row['avg_diag_dist']:.4f}")

    print()
    print(f"  Total submissions in field : {len(result_df)}")
    print(f"  Positive-remaining rate    : "
          f"{(result_df['avg_remaining'] >= 0).mean()*100:.1f}% of submissions")
    print("═" * 106)


# ── Sampling insight ──────────────────────────────────────────────────────────

def print_sampling_info(top_x_subs: list[dict], sampled_subs: list[dict]):
    xs = np.array([s["x"] for s in top_x_subs])
    hs = np.array([s["h"] for s in top_x_subs])
    print()
    print("─" * 60)
    print("  Normal-distribution sampling summary")
    print("─" * 60)
    print(f"  Fitted on top-{len(top_x_subs)} pairs:")
    print(f"    x: μ={xs.mean():.1f}  σ={xs.std():.1f}  "
          f"range=[{xs.min()}, {xs.max()}]")
    print(f"    h: μ={hs.mean():.1f}  σ={hs.std():.1f}  "
          f"range=[{hs.min()}, {hs.max()}]")
    sxs = np.array([s["x"] for s in sampled_subs])
    shs = np.array([s["h"] for s in sampled_subs])
    print(f"  Sampled {len(sampled_subs)} new pairs:")
    print(f"    x: μ={sxs.mean():.1f}  σ={sxs.std():.1f}  "
          f"range=[{sxs.min()}, {sxs.max()}]")
    print(f"    h: μ={shs.mean():.1f}  σ={shs.std():.1f}  "
          f"range=[{shs.min()}, {shs.max()}]")
    print("─" * 60)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Kyber competition simulator with crowd penalty",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="mode", required=True)

    def add_common(sp):
        sp.add_argument("--csv",    default="kyber_consensus.csv",
                        help="Consensus / scores CSV  [kyber_consensus.csv]")
        sp.add_argument("--top-c",  type=int, dest="top_c", default=20,
                        help="Top-C pairs to pull from CSV  [20]")
        sp.add_argument("--n-sims", type=int, dest="n_sims", default=NSIMS,
                        help="Competition iterations per seed  [50]")
        sp.add_argument("--seeds",  type=int, default=5,
                        help="Independent seeds to average over  [5]")
        sp.add_argument("--seed",   type=int, default=None,
                        help="Master RNG seed (defaults to current system time)")
        sp.add_argument("--top-show", type=int, dest="top_show", default=20,
                        help="How many rows to print in result table  [20]")
        sp.add_argument("--out-csv", dest="out_csv", default=None,
                        help="Save result table to CSV  [None]")
        sp.add_argument("--verbose", action="store_true")

    sp1 = sub.add_parser(
        "single",
        help="Your (x,h) vs top-C CSV pairs.  "
             "E.g.:  python kyber_simulate.py single --x 200 --h 20")
    add_common(sp1)
    sp1.add_argument("--x", type=int, required=True,
                     help="Your x submission (0–300)")
    sp1.add_argument("--h", type=int, required=True,
                     help="Your h submission (5–60)")

    sp2 = sub.add_parser(
        "distributed",
        help="Top-C pairs + Y normal-sampled pairs around top-X.  "
             "E.g.:  python kyber_simulate.py distributed "
             "--top-c 20 --top-x 10 --y-samples 100")
    add_common(sp2)
    sp2.add_argument("--top-x",     type=int, dest="top_x", default=10,
                     help="Top-X pairs to fit Normal distribution on  [10]")
    sp2.add_argument("--y-samples", type=int, dest="y_samples", default=50,
                     help="Number of Normal-sampled pairs to add  [50]")

    return p.parse_args()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.seed is None:
        args.seed = int(time.time() * 1000) % (2**32 - 1)

    print("\n" + "═" * 70)
    print(f"  Kyber Competition Simulator  —  mode: {args.mode}")
    print("═" * 70)

    if args.mode == "single":
        x, h = args.x, args.h
        if not (X_MIN <= x <= X_MAX):
            print(f"[ERROR] x={x} out of range [{X_MIN}, {X_MAX}]", file=sys.stderr)
            sys.exit(1)
        if not (H_MIN <= h <= H_MAX):
            print(f"[ERROR] h={h} out of range [{H_MIN}, {H_MAX}]", file=sys.stderr)
            sys.exit(1)

    print(f"\n  Loading top-{args.top_c} pairs from  {args.csv} …")
    try:
        csv_subs = load_top_c(args.csv, args.top_c)
    except FileNotFoundError:
        print(f"[ERROR] CSV not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    print(f"  Loaded {len(csv_subs)} CSV submissions")

    if args.mode == "single":
        your_label  = f"YOU_x{x}_h{h}"
        your_sub    = {"x": x, "h": h, "label": your_label}
        submissions = [your_sub] + csv_subs
        print(f"  Your submission: x={x}, h={h}  →  label: {your_label}")
        target_label = your_label

    else:  # distributed
        rng_sample  = np.random.default_rng(args.seed + 777)
        top_x_subs  = csv_subs[:args.top_x]
        sampled     = sample_normal_submissions(top_x_subs, args.y_samples, rng_sample)
        print_sampling_info(top_x_subs, sampled)
        submissions  = csv_subs + sampled
        target_label = None

    print(f"\n  Total submissions in field : {len(submissions)}")
    print(f"  Simulations per seed       : {args.n_sims}")
    print(f"  Seeds                      : {args.seeds}")
    print(f"  Master RNG Seed (Time)     : {args.seed}")
    print(f"  Total iterations           : {args.n_sims * args.seeds:,}")
    print()

    result_df = simulate_competition(
        submissions = submissions,
        n_sims      = args.n_sims,
        n_seeds     = args.seeds,
        master_seed = args.seed,
        verbose     = args.verbose,
    )

    print_results(result_df, target_label=target_label, top_k=args.top_show)

    if args.out_csv:
        result_df.to_csv(args.out_csv, index=False)
        print(f"\n  Results saved to  {args.out_csv}")

    if args.mode == "distributed":
        print()
        print("─" * 70)
        print("  TOP-C vs SAMPLED performance comparison")
        print("─" * 70)
        csv_labels    = {s["label"] for s in csv_subs}
        sample_labels = {s["label"] for s in sampled}

        csv_rows    = result_df[result_df["label"].isin(csv_labels)]
        sample_rows = result_df[result_df["label"].isin(sample_labels)]

        print(f"  CSV top-{args.top_c} avg remaining  : "
              f"{csv_rows['avg_remaining'].mean():.2f}  "
              f"(std {csv_rows['avg_remaining'].std():.2f})")
        print(f"  Sampled-{args.y_samples} avg remaining : "
              f"{sample_rows['avg_remaining'].mean():.2f}  "
              f"(std {sample_rows['avg_remaining'].std():.2f})")

        # New metric comparisons
        print(f"\n  CSV top-{args.top_c} avg thorn crossings : "
              f"{csv_rows['avg_thorn_cross'].mean():.2f}")
        print(f"  Sampled-{args.y_samples} avg thorn crossings : "
              f"{sample_rows['avg_thorn_cross'].mean():.2f}")
        print(f"\n  CSV top-{args.top_c} avg fence crossings : "
              f"{csv_rows['avg_fence_cross'].mean():.2f}")
        print(f"  Sampled-{args.y_samples} avg fence crossings : "
              f"{sample_rows['avg_fence_cross'].mean():.2f}")
        print(f"\n  CSV top-{args.top_c} avg diagonal dist   : "
              f"{csv_rows['avg_diag_dist'].mean():.4f}")
        print(f"  Sampled-{args.y_samples} avg diagonal dist   : "
              f"{sample_rows['avg_diag_dist'].mean():.4f}")

        best_sample = sample_rows.iloc[0] if len(sample_rows) else None
        if best_sample is not None:
            print(f"\n  Best sampled pair  : "
                  f"x={best_sample['x']}, h={best_sample['h']}  "
                  f"→  avg_rem={best_sample['avg_remaining']:.2f}  "
                  f"thorn_x={best_sample['avg_thorn_cross']:.2f}  "
                  f"fence_x={best_sample['avg_fence_cross']:.2f}  "
                  f"diag_d={best_sample['avg_diag_dist']:.4f}")
        print("─" * 70)


if __name__ == "__main__":
    main()