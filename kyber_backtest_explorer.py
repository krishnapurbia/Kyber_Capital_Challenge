#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════╗
║         KYBER CHALLENGE — 7-Submission Portfolio Backtest               ║
║   Simulates the full crowd-penalty competition for your 7 submissions   ║
╚══════════════════════════════════════════════════════════════════════════╝

Usage:
    python kyber_backtest.py

Outputs:
    kyber_backtest_results.txt    — Full report
    kyber_portfolio_scores.csv    — Per-submission detail
"""

import numpy as np
import csv
import time
from itertools import combinations

# ─── Constants ────────────────────────────────────────────────────────────────
GRID   = 50
MIN_T  = 5
MAX_T  = 30
THORN  = 40
NSIMS  = 50       # competition iterations per seed
N_SEEDS = 20      # seeds to average over (higher = more stable estimate)
TOTAL_FENCES = GRID * (GRID + 1) + (GRID + 1) * GRID  # = 5100

# ─── Core Game Functions ──────────────────────────────────────────────────────

def build_base_fences(rng: np.random.Generator):
    """Shared base fence grid for one iteration (all submissions share this)."""
    ef = rng.integers(MIN_T, MAX_T + 1, (GRID, GRID + 1)).astype(np.float64)
    nf = rng.integers(MIN_T, MAX_T + 1, (GRID + 1, GRID)).astype(np.float64)
    return ef, nf


def add_thorns(ef_base, nf_base, x_val, rng):
    """Copy base grid and sprinkle 2*x_val thorns at random fences."""
    ef = ef_base.copy()
    nf = nf_base.copy()
    n  = 2 * x_val
    if n > 0:
        idxs      = rng.integers(0, ef.size + nf.size, size=n)
        east_mask = idxs < ef.size
        np.add.at(ef.ravel(), idxs[east_mask],             THORN)
        np.add.at(nf.ravel(), idxs[~east_mask] - ef.size,  THORN)
    return ef, nf


def walk_once(ef, nf, h, rng):
    """
    Walk from (0,0) → (GRID,GRID).
    Returns (eu, nu, raw_toll):
      eu[cx,cy]  = 1 if east fence at (cx,cy) was crossed, else 0
      nu[cx,cy]  = 1 if north fence at (cx,cy) was crossed, else 0
    """
    cx, cy   = 0, 0
    eu       = np.zeros((GRID, GRID + 1), dtype=np.int32)
    nu       = np.zeros((GRID + 1, GRID), dtype=np.int32)
    raw_toll = 0.0

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
            raw_toll += ef[cx, cy]
            eu[cx, cy] += 1
            cx += 1
        else:
            raw_toll += nf[cx, cy]
            nu[cx, cy] += 1
            cy += 1

    return eu, nu, raw_toll


def penalised_toll(ef, nf, eu, nu, frac_e, frac_n):
    """
    Crowd-penalised toll for ONE submission.
    final_toll_fence = base_toll × (1 + fraction_crossed)³
    """
    pen_e = ef * (1.0 + frac_e) ** 3
    pen_n = nf * (1.0 + frac_n) ** 3
    return float((pen_e * eu).sum() + (pen_n * nu).sum())


# ─── One Competition Iteration ────────────────────────────────────────────────

def run_iteration(submissions, base_rng, sub_rngs):
    """
    One contest iteration: all submissions play on the same base fence grid.
    Returns list of (raw_toll, penalised_toll) per submission.
    """
    n = len(submissions)
    ef_base, nf_base = build_base_fences(base_rng)

    sub_data = []
    for idx, sub in enumerate(submissions):
        ef_i, nf_i = add_thorns(ef_base, nf_base, sub["x"], sub_rngs[idx])
        eu_i, nu_i, raw_i = walk_once(ef_i, nf_i, sub["h"], sub_rngs[idx])
        sub_data.append((ef_i, nf_i, eu_i, nu_i, raw_i))

    # Global usage fractions across ALL submissions
    total_eu = sum(d[2] for d in sub_data).astype(np.float64)
    total_nu = sum(d[3] for d in sub_data).astype(np.float64)
    frac_e   = total_eu / n
    frac_n   = total_nu / n

    results = []
    for ef_i, nf_i, eu_i, nu_i, raw_i in sub_data:
        pen = penalised_toll(ef_i, nf_i, eu_i, nu_i, frac_e, frac_n)
        results.append((raw_i, pen))
    return results


# ─── Full Simulation ──────────────────────────────────────────────────────────

def simulate(submissions, n_sims=NSIMS, n_seeds=N_SEEDS, master_seed=42, verbose=False):
    """
    Run full competition simulation.
    Returns dict: label → list of remaining-apple values (len = n_seeds × n_sims)
    """
    n = len(submissions)
    master_rng = np.random.default_rng(master_seed)
    seeds = master_rng.integers(0, 10_000_000, size=(n_seeds, n_sims))

    remaining = {sub["label"]: [] for sub in submissions}

    for s_idx in range(n_seeds):
        for sim_idx in range(n_sims):
            iter_rng  = np.random.default_rng(int(seeds[s_idx, sim_idx]))
            base_rng  = np.random.default_rng(int(iter_rng.integers(0, 10_000_000)))
            sub_rngs  = [np.random.default_rng(int(iter_rng.integers(0, 10_000_000)))
                         for _ in range(n)]

            results = run_iteration(submissions, base_rng, sub_rngs)
            for i, sub in enumerate(submissions):
                raw, pen = results[i]
                start = 2500 + sub["x"]
                remaining[sub["label"]].append(start - pen)

    return {k: np.array(v) for k, v in remaining.items()}


# ─── Build Simulated Competitor Field ─────────────────────────────────────────

def make_competitor_field(n_competitors=43, seed=999):
    """
    Simulate what a typical competition field looks like.
    Competitors cluster near individually-optimal params (x=0, h~17)
    with some spread.
    """
    rng = np.random.default_rng(seed)
    comps = []

    # 60% cluster near individual optimum (x~0, h~17)
    n_cluster = int(n_competitors * 0.60)
    xs = np.clip(rng.normal(10, 20, n_cluster).round().astype(int), 0, 300)
    hs = np.clip(rng.normal(17, 5,  n_cluster).round().astype(int), 5, 60)
    for i, (x, h) in enumerate(zip(xs, hs)):
        comps.append({"x": int(x), "h": int(h), "label": f"comp_cluster_{i}_x{x}_h{h}"})

    # 25% spread across h values, low x
    n_spread = int(n_competitors * 0.25)
    xs = np.clip(rng.integers(0, 50, n_spread), 0, 300)
    hs = np.clip(rng.integers(5, 60, n_spread), 5, 60)
    for i, (x, h) in enumerate(zip(xs, hs)):
        comps.append({"x": int(x), "h": int(h), "label": f"comp_spread_{i}_x{x}_h{h}"})

    # 15% random
    n_rand = n_competitors - n_cluster - n_spread
    xs = rng.integers(0, 300, n_rand)
    hs = rng.integers(5, 60, n_rand)
    for i, (x, h) in enumerate(zip(xs, hs)):
        comps.append({"x": int(x), "h": int(h), "label": f"comp_rand_{i}_x{x}_h{h}"})

    return comps


# ─── Portfolio Definitions ─────────────────────────────────────────────────────

# Each portfolio is a list of 7 (x, h) pairs with a strategic rationale.

PORTFOLIOS = {

    # ── STRATEGY 1: Maximum Diversity across parameter space ──────────────────
    # Use very different x and h values to force geometrically distinct paths.
    # High x creates unique thorn-warped fence grids → unique paths.
    # Varied h creates different routing decisions on the same grid.
    # Self-crowd-penalty is minimised because the 7 paths rarely share fences.
    "A_MaxDiversity": [
        {"x":   0, "h": 17, "label": "A1_x0_h17"},     # Raw individual optimum
        {"x":   0, "h": 60, "label": "A2_x0_h60"},     # Random walk (h > all tolls)
        {"x":  50, "h": 44, "label": "A3_x50_h44"},    # Light thorns, avoid thorn fences
        {"x": 100, "h": 44, "label": "A4_x100_h44"},   # Moderate thorns, avoidance
        {"x": 200, "h": 44, "label": "A5_x200_h44"},   # Heavy thorns, very different path
        {"x": 300, "h": 44, "label": "A6_x300_h44"},   # Max thorns, most unique grid
        {"x":   0, "h":  8, "label": "A7_x0_h8"},      # Very selective (ultra-picky)
    ],

    # ── STRATEGY 2: h-Spread on x=0 ──────────────────────────────────────────
    # No thorns (preserving 2500 start), but fan out h values so routing
    # decisions differ.  Different h → different "preference zones" → 
    # somewhat different paths on the same base fence grid.
    "B_hSpread": [
        {"x": 0, "h":  5, "label": "B1_x0_h5"},
        {"x": 0, "h": 10, "label": "B2_x0_h10"},
        {"x": 0, "h": 17, "label": "B3_x0_h17"},
        {"x": 0, "h": 25, "label": "B4_x0_h25"},
        {"x": 0, "h": 35, "label": "B5_x0_h35"},
        {"x": 0, "h": 45, "label": "B6_x0_h45"},
        {"x": 0, "h": 60, "label": "B7_x0_h60"},
    ],

    # ── STRATEGY 3: Thorn Specialists (h=44 = thorn-avoidance sweet-spot) ───
    # h=44 is just below the minimum thorn-fence toll (5+40=45), so the farmer
    # ALWAYS avoids a thorn fence when the other direction is thorn-free.
    # Different x values → different random thorn positions → different paths.
    # Higher x also gives more starting apples (offset by thorn encounters).
    "C_ThornSpecialist": [
        {"x":   0, "h": 17, "label": "C1_x0_h17"},     # Baseline (no thorns)
        {"x":  25, "h": 44, "label": "C2_x25_h44"},
        {"x":  75, "h": 44, "label": "C3_x75_h44"},
        {"x": 125, "h": 44, "label": "C4_x125_h44"},
        {"x": 175, "h": 44, "label": "C5_x175_h44"},
        {"x": 250, "h": 44, "label": "C6_x250_h44"},
        {"x": 300, "h": 44, "label": "C7_x300_h44"},
    ],

    # ── STRATEGY 4: Crowd-Dodge (anti-herd) ──────────────────────────────────
    # Assume the crowd clusters at x=0, h≈17-30.  Those diagonal-band fences
    # get heavy crowd penalty.  We deliberately use x>0 with h=44 to route
    # around thorns AND away from the crowd's paths.  One "scout" at baseline.
    "D_CrowdDodge": [
        {"x":   0, "h": 17, "label": "D1_x0_h17"},
        {"x": 100, "h": 44, "label": "D2_x100_h44"},
        {"x": 150, "h": 44, "label": "D3_x150_h44"},
        {"x": 200, "h": 44, "label": "D4_x200_h44"},
        {"x": 250, "h": 44, "label": "D5_x250_h44"},
        {"x": 300, "h": 44, "label": "D6_x300_h44"},
        {"x":   0, "h": 60, "label": "D7_x0_h60"},
    ],

    # ── STRATEGY 5: Conservative (all near individual optimum) ───────────────
    # Pure raw-score maximisation ignoring crowd effects.
    # 7 near-identical submissions → high self-crowd-penalty.
    # Included as a *baseline / what NOT to do* benchmark.
    "E_Conservative_BASELINE": [
        {"x": 0, "h": 15, "label": "E1_x0_h15"},
        {"x": 0, "h": 16, "label": "E2_x0_h16"},
        {"x": 0, "h": 17, "label": "E3_x0_h17"},
        {"x": 0, "h": 18, "label": "E4_x0_h18"},
        {"x": 0, "h": 19, "label": "E5_x0_h19"},
        {"x": 0, "h": 20, "label": "E6_x0_h20"},
        {"x": 0, "h": 21, "label": "E7_x0_h21"},
    ],
}


# ─── Analysis Helpers ─────────────────────────────────────────────────────────

def analyse_path_overlap(submissions, n_samples=200, seed=42):
    """
    Estimate average pairwise path overlap (fraction of shared fences)
    between all pairs of submissions in a portfolio.
    Lower is better (less self-crowd-pressure).
    """
    rng = np.random.default_rng(seed)
    ef_base, nf_base = build_base_fences(rng)
    pair_overlaps = []

    for i, j in combinations(range(len(submissions)), 2):
        si, sj = submissions[i], submissions[j]
        shared_frac = []
        for _ in range(n_samples):
            rng_i = np.random.default_rng(rng.integers(0, 10_000_000))
            rng_j = np.random.default_rng(rng.integers(0, 10_000_000))
            ef_i, nf_i = add_thorns(ef_base, nf_base, si["x"], rng_i)
            ef_j, nf_j = add_thorns(ef_base, nf_base, sj["x"], rng_j)
            eu_i, nu_i, _ = walk_once(ef_i, nf_i, si["h"], rng_i)
            eu_j, nu_j, _ = walk_once(ef_j, nf_j, sj["h"], rng_j)
            # Shared = fences crossed by BOTH
            shared = (eu_i & eu_j).sum() + (nu_i & nu_j).sum()
            total  = eu_i.sum() + nu_i.sum()  # always 100
            shared_frac.append(shared / total)
        pair_overlaps.append(np.mean(shared_frac))

    return np.mean(pair_overlaps), pair_overlaps


def estimate_expected_raw_toll(x_val, h_val, n_samples=2000, seed=7):
    """Quick estimate of raw (no-crowd) expected toll for a single (x,h) pair."""
    rng = np.random.default_rng(seed)
    tolls = []
    for _ in range(n_samples):
        ef, nf = build_base_fences(rng)
        ef, nf = add_thorns(ef, nf, x_val, rng)
        _, _, toll = walk_once(ef, nf, h_val, rng)
        tolls.append(toll)
    return np.mean(tolls), np.std(tolls)


# ─── Main Backtest Runner ─────────────────────────────────────────────────────

def run_full_backtest():
    print("\n" + "═"*72)
    print("  KYBER CHALLENGE — 7-Submission Portfolio Backtest")
    print("═"*72)
    print(f"  Simulations per seed : {NSIMS}")
    print(f"  Seeds averaged       : {N_SEEDS}")
    print(f"  Total iterations     : {NSIMS * N_SEEDS:,}")
    print()

    # ── Quick individual (x,h) raw performance sweep ───────────────────────
    print("─"*72)
    print("  STEP 1: Individual (x,h) raw performance (no crowd penalty)")
    print("─"*72)
    probe_pairs = [
        (0,  5), (0, 10), (0, 14), (0, 17), (0, 20), (0, 25), (0, 30), (0, 44), (0, 60),
        (50, 17), (50, 44),
        (100, 17), (100, 44),
        (200, 17), (200, 44),
        (300, 17), (300, 44),
    ]
    print(f"  {'(x,h)':<12} {'start':>6} {'mean_rem':>10} {'std':>8} {'win%':>7}")
    print("  " + "─"*46)
    raw_results = {}
    for x, h in probe_pairs:
        mu, sd = estimate_expected_raw_toll(x, h, n_samples=3000)
        start  = 2500 + x
        rem    = start - mu
        win    = (start - np.random.default_rng(42).normal(mu, sd, 10000)) >= 0
        print(f"  ({x:>3},{h:>3})     {start:>6}  {rem:>10.1f}  {sd:>8.1f}  {100*win.mean():>6.1f}%")
        raw_results[(x, h)] = (rem, sd)

    print()

    # ── Portfolio path-overlap analysis ───────────────────────────────────
    print("─"*72)
    print("  STEP 2: Portfolio path-overlap analysis (lower = more diverse)")
    print("─"*72)
    print("  (Fraction of fences shared between pairs of submissions)")
    print()
    overlap_results = {}
    for name, portfolio in PORTFOLIOS.items():
        mean_ov, _ = analyse_path_overlap(portfolio, n_samples=300)
        overlap_results[name] = mean_ov
        print(f"  {name:<30}  mean overlap: {mean_ov:.3f}  ({mean_ov*100:.1f}%)")
    print()

    # ── Full crowd-penalty simulation ─────────────────────────────────────
    print("─"*72)
    print("  STEP 3: Full crowd-penalty simulation")
    print()
    print("  Scenario A: Only your 7 submissions (small contest, N=7)")
    print("  Scenario B: Your 7 + 43 simulated competitors (N=50)")
    print("─"*72)

    competitors = make_competitor_field(n_competitors=43, seed=999)

    portfolio_summary = {}

    for name, portfolio in PORTFOLIOS.items():
        print(f"\n  ── Portfolio {name} ──────────────────────────────────")

        # Scenario A: self only
        rem_a = simulate(portfolio, n_sims=NSIMS, n_seeds=N_SEEDS, master_seed=42)

        # Scenario B: with 43 competitors
        full_field = portfolio + competitors
        rem_b_all  = simulate(full_field, n_sims=NSIMS, n_seeds=N_SEEDS, master_seed=42)
        # Extract only your 7
        rem_b = {sub["label"]: rem_b_all[sub["label"]] for sub in portfolio}

        print(f"  {'Submission':<18} {'x':>4} {'h':>4} "
              f"{'Scen-A avg':>12} {'Scen-B avg':>12} {'B win%':>7}")
        print("  " + "─"*62)

        avg_a_list, avg_b_list = [], []
        sub_details = []

        for sub in portfolio:
            lbl  = sub["label"]
            a_mu = rem_a[lbl].mean()
            b_mu = rem_b[lbl].mean()
            b_win = (rem_b[lbl] >= 0).mean()
            short = lbl.split("_")[0] + "_" + lbl.split("_")[1]
            print(f"  {short:<18} {sub['x']:>4} {sub['h']:>4} "
                  f"  {a_mu:>10.1f}   {b_mu:>10.1f}  {100*b_win:>5.1f}%")
            avg_a_list.append(a_mu)
            avg_b_list.append(b_mu)
            sub_details.append({
                "portfolio": name,
                "label": lbl,
                "x": sub["x"], "h": sub["h"],
                "scen_a_avg": round(a_mu, 2),
                "scen_b_avg": round(b_mu, 2),
                "scen_b_win": round(b_win, 4),
            })

        pf_a = np.mean(avg_a_list)
        pf_b = np.mean(avg_b_list)
        ov   = overlap_results[name]
        print(f"  {'PORTFOLIO AVG':<18} {'':>4} {'':>4} "
              f"  {pf_a:>10.1f}   {pf_b:>10.1f}  (overlap={ov:.3f})")

        portfolio_summary[name] = {
            "scen_a_avg": pf_a,
            "scen_b_avg": pf_b,
            "overlap":    ov,
            "details":    sub_details,
        }

    # ── Final ranking ──────────────────────────────────────────────────────
    print()
    print("═"*72)
    print("  FINAL PORTFOLIO RANKING (by Scenario B — realistic field)")
    print("═"*72)
    print(f"  {'Rank':<6} {'Portfolio':<30} {'Scen-A':>10} {'Scen-B':>10} {'Overlap':>9}")
    print("  " + "─"*66)
    ranked = sorted(portfolio_summary.items(),
                    key=lambda kv: kv[1]["scen_b_avg"], reverse=True)
    for rank, (name, info) in enumerate(ranked, 1):
        marker = " ← BEST" if rank == 1 else ""
        print(f"  {rank:<6} {name:<30} {info['scen_a_avg']:>10.1f} "
              f"{info['scen_b_avg']:>10.1f} {info['overlap']:>9.3f}{marker}")

    # ── Print winning portfolio's submissions ──────────────────────────────
    best_name, best_info = ranked[0]
    print()
    print("═"*72)
    print(f"  RECOMMENDED PORTFOLIO: {best_name}")
    print("═"*72)
    print()
    for d in best_info["details"]:
        print(f"    x={d['x']:>3},  h={d['h']:>2}   →  "
              f"Scen-B avg remaining: {d['scen_b_avg']:>8.1f}  "
              f"win: {100*d['scen_b_win']:.1f}%")

    # ── Save CSV ──────────────────────────────────────────────────────────
    all_details = []
    for name, info in portfolio_summary.items():
        all_details.extend(info["details"])
    csv_path = "/mnt/user-data/outputs/kyber_portfolio_scores.csv"
    fields   = ["portfolio","label","x","h","scen_a_avg","scen_b_avg","scen_b_win"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_details)
    print(f"\n  CSV saved → {csv_path}")
    return ranked, portfolio_summary


# ─── Deep Analysis of Best Portfolio + Crowd-sensitivity ─────────────────────

def deep_analysis(ranked, portfolio_summary):
    """Test best portfolio across different competitor field sizes."""
    best_name = ranked[0][0]
    best_portfolio = PORTFOLIOS[best_name]

    print()
    print("─"*72)
    print(f"  DEEP ANALYSIS: {best_name} vs different field sizes")
    print("─"*72)
    print(f"  {'N total':>8}  {'Your avg':>12}  {'Your best':>12}  {'Your win%':>10}")
    print("  " + "─"*48)

    for n_comp in [0, 10, 20, 43, 93, 193]:
        field = best_portfolio + make_competitor_field(n_comp, seed=999)
        rem_all = simulate(field, n_sims=NSIMS, n_seeds=N_SEEDS, master_seed=42)
        your_avgs = [rem_all[sub["label"]].mean() for sub in best_portfolio]
        your_wins = [(rem_all[sub["label"]] >= 0).mean() for sub in best_portfolio]
        portfolio_avg  = np.mean(your_avgs)
        portfolio_best = np.max(your_avgs)
        portfolio_win  = np.mean(your_wins)
        n_total = 7 + n_comp
        print(f"  {n_total:>8}  {portfolio_avg:>12.1f}  {portfolio_best:>12.1f}  "
              f"{100*portfolio_win:>9.1f}%")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.time()
    ranked, summary = run_full_backtest()
    deep_analysis(ranked, summary)
    elapsed = time.time() - t0
    print(f"\n  Total runtime: {elapsed:.1f}s")
    print("═"*72)
