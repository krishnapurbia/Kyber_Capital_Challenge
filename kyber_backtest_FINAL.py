#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║       KYBER CHALLENGE — Full Strategy Backtest  (7 submissions)      ║
╚══════════════════════════════════════════════════════════════════════╝

Run:   python kyber_backtest_FINAL.py

Outputs a ranked comparison of every portfolio + deep dive on winner.
N_SEEDS=20, NSIMS=50 → 1 000 iterations each; takes ~2-3 min.
"""

import numpy as np
import csv, time

# ─── Constants ────────────────────────────────────────────────────────────────
GRID   = 50
MIN_T  = 5
MAX_T  = 30
THORN  = 40
NSIMS  = 50          # competition iterations per seed (matches the contest)
N_SEEDS = 20         # independent seeds to average over

# ─── Core Simulation ─────────────────────────────────────────────────────────

def build_base_fences(rng):
    ef = rng.integers(MIN_T, MAX_T + 1, (GRID, GRID + 1)).astype(np.float64)
    nf = rng.integers(MIN_T, MAX_T + 1, (GRID + 1, GRID)).astype(np.float64)
    return ef, nf

def add_thorns(ef_base, nf_base, x_val, rng):
    ef, nf = ef_base.copy(), nf_base.copy()
    n = 2 * x_val
    if n > 0:
        idxs = rng.integers(0, ef.size + nf.size, size=n)
        em   = idxs < ef.size
        np.add.at(ef.ravel(), idxs[em],             THORN)
        np.add.at(nf.ravel(), idxs[~em] - ef.size,  THORN)
    return ef, nf

def walk_once(ef, nf, h, rng):
    """
    Farmer walks (0,0)→(50,50).  Routing rule:
      · exactly one direction ≤ h  → take it
      · both ≤ h  or  both > h    → 50/50 random
      · only one direction left    → forced
    Returns (eu, nu, raw_toll).
    """
    cx = cy = 0
    eu = np.zeros((GRID, GRID + 1), dtype=np.int32)
    nu = np.zeros((GRID + 1, GRID), dtype=np.int32)
    toll = 0.0
    while cx < GRID or cy < GRID:
        can_e, can_n = cx < GRID, cy < GRID
        if can_e and can_n:
            te, tn = ef[cx, cy], nf[cx, cy]
            e_ok, n_ok = te <= h, tn <= h
            if   e_ok and not n_ok: mv = "e"
            elif n_ok and not e_ok: mv = "n"
            else:                   mv = "e" if rng.random() < 0.5 else "n"
        elif can_e: mv = "e"
        else:       mv = "n"
        if mv == "e":
            toll += ef[cx, cy]; eu[cx, cy] += 1; cx += 1
        else:
            toll += nf[cx, cy]; nu[cx, cy] += 1; cy += 1
    return eu, nu, toll

def run_one_iteration(submissions, base_rng, sub_rngs):
    """
    One competition iteration (all subs share the same base fence grid).
    Returns list of crowd-penalised tolls, one per submission.

    Crowd penalty formula: final_toll = base_toll × (1 + fraction_crossed)³
    where fraction_crossed = (# submissions crossing this fence) / (total subs).
    """
    n = len(submissions)
    ef_base, nf_base = build_base_fences(base_rng)
    sub_data = []
    for i, sub in enumerate(submissions):
        ef_i, nf_i = add_thorns(ef_base, nf_base, sub["x"], sub_rngs[i])
        eu_i, nu_i, _ = walk_once(ef_i, nf_i, sub["h"], sub_rngs[i])
        sub_data.append((ef_i, nf_i, eu_i, nu_i))

    total_eu = sum(d[2] for d in sub_data).astype(np.float64)
    total_nu = sum(d[3] for d in sub_data).astype(np.float64)
    frac_e, frac_n = total_eu / n, total_nu / n

    pen_tolls = []
    for ef_i, nf_i, eu_i, nu_i in sub_data:
        pen = float(
            (ef_i * (1 + frac_e) ** 3 * eu_i).sum() +
            (nf_i * (1 + frac_n) ** 3 * nu_i).sum()
        )
        pen_tolls.append(pen)
    return pen_tolls

def simulate(submissions, n_sims=NSIMS, n_seeds=N_SEEDS, master_seed=42):
    """Run full competition simulation. Returns {label: array_of_remaining_apples}."""
    n = len(submissions)
    master_rng = np.random.default_rng(master_seed)
    seeds = master_rng.integers(0, 10_000_000, size=(n_seeds, n_sims))
    remaining = {sub["label"]: [] for sub in submissions}

    for s_idx in range(n_seeds):
        for sim_idx in range(n_sims):
            iter_rng = np.random.default_rng(int(seeds[s_idx, sim_idx]))
            base_rng = np.random.default_rng(int(iter_rng.integers(0, 10_000_000)))
            sub_rngs = [
                np.random.default_rng(int(iter_rng.integers(0, 10_000_000)))
                for _ in range(n)
            ]
            pen_tolls = run_one_iteration(submissions, base_rng, sub_rngs)
            for i, sub in enumerate(submissions):
                remaining[sub["label"]].append(2500 + sub["x"] - pen_tolls[i])

    return {k: np.array(v) for k, v in remaining.items()}

# ─── Simulated Competitor Field ───────────────────────────────────────────────

def make_competitor_field(n_competitors=43, seed=999):
    """
    Realistic competitor field:
      60% cluster near (x≈0-20, h≈15-22)  — optimisers who read the math
      25% spread across (low-x, random h)  — experimenters
      15% fully random                     — noise
    """
    rng = np.random.default_rng(seed)
    comps = []
    n1 = int(n_competitors * 0.60)
    xs = np.clip(rng.normal(10, 20, n1).round().astype(int), 0, 300)
    hs = np.clip(rng.normal(17, 5,  n1).round().astype(int), 5, 60)
    for i, (x, h) in enumerate(zip(xs, hs)):
        comps.append({"x": int(x), "h": int(h), "label": f"comp_cluster_{i}"})
    n2 = int(n_competitors * 0.25)
    xs = np.clip(rng.integers(0, 50, n2), 0, 300)
    hs = np.clip(rng.integers(5, 60, n2), 5, 60)
    for i, (x, h) in enumerate(zip(xs, hs)):
        comps.append({"x": int(x), "h": int(h), "label": f"comp_spread_{i}"})
    n3 = n_competitors - n1 - n2
    xs = rng.integers(0, 300, n3)
    hs = rng.integers(5,  60, n3)
    for i, (x, h) in enumerate(zip(xs, hs)):
        comps.append({"x": int(x), "h": int(h), "label": f"comp_rand_{i}"})
    return comps

# ─── Portfolios ───────────────────────────────────────────────────────────────

PORTFOLIOS = {
    # ── RECOMMENDED: Crowd-Aware Hybrid ────────────────────────────────────
    # Subs 1-3: x=0 keeps starting apples at 2500 (no thorns).
    #           h=14/17/21 spans the analytically-optimal zone where
    #           E[toll/step] = (5+h)/2 + 13×((30-h)/26)² is minimised (~h=17).
    #           Different h values route differently on the same base grid,
    #           keeping pairwise path overlap low within the group.
    # Subs 4-6: x=200/250/300 adds many thorns (400/500/600) which randomise
    #           the fence layout uniquely per submission.  h=44 is just below
    #           the minimum thorn-fence toll (5+40=45), so the farmer ALWAYS
    #           avoids thorn fences when the other direction is clear.
    #           These take fundamentally different paths from the h≈17 crowd
    #           → their fences carry low crowd-fraction → low penalty multiplier.
    # Sub 7:    x=0, h=8 — extremely selective; only 15% of base fences are
    #           "cheap", so routing is nearly random and diverges from the pack.
    "RECOMMENDED_Hybrid": [
        {"x":   0, "h": 14, "label": "S1_x0_h14"},
        {"x":   0, "h": 17, "label": "S2_x0_h17"},
        {"x":   0, "h": 21, "label": "S3_x0_h21"},
        {"x": 300, "h": 44, "label": "S4_x300_h44"},
        {"x": 250, "h": 44, "label": "S5_x250_h44"},
        {"x": 200, "h": 44, "label": "S6_x200_h44"},
        {"x":   0, "h":  8, "label": "S7_x0_h8"},
    ],

    # ── Conservative: all near individual optimum ───────────────────────────
    # Highest raw performance per submission but path overlap = 16% → heavy
    # self-crowd-penalty in small contests (N<20).  Still competitive at N≥40.
    "Conservative_x0_h15-21": [
        {"x": 0, "h": 15, "label": "C1_x0_h15"},
        {"x": 0, "h": 16, "label": "C2_x0_h16"},
        {"x": 0, "h": 17, "label": "C3_x0_h17"},
        {"x": 0, "h": 18, "label": "C4_x0_h18"},
        {"x": 0, "h": 19, "label": "C5_x0_h19"},
        {"x": 0, "h": 20, "label": "C6_x0_h20"},
        {"x": 0, "h": 21, "label": "C7_x0_h21"},
    ],

    # ── Thorn Specialists: diverse x, all h=44 ─────────────────────────────
    "ThornSpecialists_h44": [
        {"x":   0, "h": 17, "label": "T1_x0_h17"},
        {"x":  25, "h": 44, "label": "T2_x25_h44"},
        {"x":  75, "h": 44, "label": "T3_x75_h44"},
        {"x": 125, "h": 44, "label": "T4_x125_h44"},
        {"x": 175, "h": 44, "label": "T5_x175_h44"},
        {"x": 250, "h": 44, "label": "T6_x250_h44"},
        {"x": 300, "h": 44, "label": "T7_x300_h44"},
    ],

    # ── h-Spread on x=0 ────────────────────────────────────────────────────
    "hSpread_x0": [
        {"x": 0, "h":  5, "label": "H1_x0_h5"},
        {"x": 0, "h": 10, "label": "H2_x0_h10"},
        {"x": 0, "h": 14, "label": "H3_x0_h14"},
        {"x": 0, "h": 17, "label": "H4_x0_h17"},
        {"x": 0, "h": 25, "label": "H5_x0_h25"},
        {"x": 0, "h": 44, "label": "H6_x0_h44"},
        {"x": 0, "h": 60, "label": "H7_x0_h60"},
    ],
}

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("\n" + "═"*70)
    print("  KYBER CHALLENGE — 7-Submission Portfolio Backtest")
    print("═"*70)
    print(f"  Sims/seed: {NSIMS}   Seeds: {N_SEEDS}   "
          f"Total iterations: {NSIMS * N_SEEDS:,}")
    print()

    # ── Step 1: Individual performance table ─────────────────────────────
    print("─"*70)
    print("  INDIVIDUAL (x,h) PERFORMANCE  (no crowd penalty, 3 000 samples)")
    print("─"*70)
    print(f"  {'(x, h)':<12}  {'start':>6}  {'E[remain]':>10}  {'std':>8}  "
          f"{'win%':>7}  {'E[toll/step]':>14}")
    print("  " + "─"*58)
    check = [(0,5),(0,10),(0,14),(0,17),(0,20),(0,25),(0,30),(0,44),(0,60),
             (50,44),(100,44),(200,44),(250,44),(300,44)]
    rng_probe = np.random.default_rng(7)
    for x, h in check:
        tolls = []
        for _ in range(3000):
            ef, nf = build_base_fences(rng_probe)
            ef, nf = add_thorns(ef, nf, x, rng_probe)
            _, _, t = walk_once(ef, nf, h, rng_probe)
            tolls.append(t)
        tolls = np.array(tolls)
        start = 2500 + x
        rem   = start - tolls
        print(f"  ({x:>3}, {h:>2})      {start:>6}  {rem.mean():>10.1f}  "
              f"{rem.std():>8.1f}  {100*(rem>=0).mean():>6.1f}%  "
              f"{tolls.mean()/100:>14.2f}")

    print()
    print("  Insight: h≈17 minimises E[toll/step] for base fences (see formula).")
    print("  h=44 behaves like h=30 when x=0 (all base fences ≤ 30 < 44).")

    # ── Step 2: Crowd-penalty simulation ─────────────────────────────────
    print()
    print("─"*70)
    print("  CROWD-PENALTY SIMULATION  (N=50: your 7 + 43 competitors)")
    print("─"*70)

    competitors = make_competitor_field(n_competitors=43, seed=999)
    all_rows = []
    pf_summary = []

    for name, portfolio in PORTFOLIOS.items():
        full_field = portfolio + competitors
        rem_all    = simulate(full_field)
        rem_yours  = {s["label"]: rem_all[s["label"]] for s in portfolio}

        sub_avgs = [rem_yours[s["label"]].mean() for s in portfolio]
        sub_wins = [(rem_yours[s["label"]] >= 0).mean() for s in portfolio]
        pf_avg   = np.mean(sub_avgs)
        pf_win   = np.mean(sub_wins)
        pf_summary.append((name, pf_avg, pf_win, portfolio, sub_avgs, sub_wins))

    pf_summary.sort(key=lambda r: r[1], reverse=True)

    for rank, (name, pf_avg, pf_win, portfolio, sub_avgs, sub_wins) in enumerate(pf_summary, 1):
        marker = "  ← WINNER" if rank == 1 else ""
        print(f"\n  [{rank}] {name}{marker}")
        print(f"  {'Submission':<18} {'x':>4} {'h':>4}  "
              f"{'avg remain':>11}  {'win%':>7}")
        print("  " + "─"*50)
        for i, sub in enumerate(portfolio):
            print(f"  {sub['label']:<18} {sub['x']:>4} {sub['h']:>4}  "
                  f"{sub_avgs[i]:>11.1f}  {100*sub_wins[i]:>6.1f}%")
            all_rows.append({
                "rank": rank, "portfolio": name,
                "label": sub["label"], "x": sub["x"], "h": sub["h"],
                "avg_remaining": round(sub_avgs[i], 2),
                "win_rate": round(sub_wins[i], 4),
            })
        print(f"  {'PORTFOLIO AVG':<18} {'':>4} {'':>4}  "
              f"{pf_avg:>11.1f}  {100*pf_win:>6.1f}%")

    # ── Step 3: Field-size sensitivity ───────────────────────────────────
    print()
    print("─"*70)
    print(f"  SENSITIVITY: {pf_summary[0][0]} vs field size N")
    print("─"*70)
    print(f"  {'N':>6}  {'Pf avg':>10}  {'Best sub':>10}  {'Win%':>8}")
    print("  " + "─"*40)
    best_portfolio = pf_summary[0][3]
    for n_comp in [0, 6, 13, 43, 93, 193]:
        cfield = make_competitor_field(n_comp, seed=999)
        field  = best_portfolio + cfield
        rall   = simulate(field)
        avgs   = [rall[s["label"]].mean() for s in best_portfolio]
        wins   = [(rall[s["label"]] >= 0).mean() for s in best_portfolio]
        print(f"  {7+n_comp:>6}  {np.mean(avgs):>10.1f}  "
              f"{np.max(avgs):>10.1f}  {100*np.mean(wins):>7.1f}%")

    # ── Save CSV ──────────────────────────────────────────────────────────
    csv_path = "kyber_portfolio_scores.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        w.writeheader(); w.writerows(all_rows)
    print(f"\n  Results saved → {csv_path}")
    print(f"  Runtime: {time.time()-t0:.0f}s")
    print("═"*70)

if __name__ == "__main__":
    main()
