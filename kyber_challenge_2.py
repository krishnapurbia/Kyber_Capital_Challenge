# #!/usr/bin/env python3
# """
# kyber_challenge_v3.py
# ══════════════════════════════════════════════════════════════════════════════
# Kyber Problem Challenge — v3  (corrected optimal strategy)

# WHAT CHANGED FROM v2
# ────────────────────
#   1. CORRECTED OPTIMAL STRATEGY  x ≈ 0–20, h ∈ [14, 22]
#        Raw-toll steering beats the +300 apple bonus from x=300.
#        With h≈18 the farmer steers toward cheap fences, saving ~300-500
#        apples over 100 steps.  That overwhelms x=300's bonus.
#        Very low h (<14) converges all low-h farmers on the same cheap
#        path → crowd overhead spikes; optimal is the sweet-spot [14,22].

#   2. h-SENSITIVITY SWEEP  (new, run before main simulation)
#        Fast grid-search over h ∈ [5,60] at x=0 with 300 solo walks per
#        point.  Measures the empirical raw-toll curve so you can see the
#        minimum, sweet-spot, and exactly how much h=30 costs vs h=18.

#   3. SIX-CLUSTER FIELD  (was five)
#        E  "True Expert"     (10%)  x ~ N(10,8)[0,20]    h ~ N(18,3)[14,22]
#        Z  "Thorn Avoider"   (25%)  x ~ N(5,8)[0,30]     h ~ N(28,12)[5,55]
#        G  "Greedy Grabber"  (25%)  x ~ N(280,20)[200,300] h ~ N(52,6)[35,60]
#        M  "Middle Ground"   (20%)  x ~ N(150,50)[50,250] h ~ N(38,8)[20,55]
#        C  "Code Follower"   (15%)  x ~ N(270,20)[220,300] h ~ N(32,4)[25,40]
#        R  "Random/Naive"    (5%)   x ~ Uniform[0,300]   h ~ Uniform[5,60]

#   4. CROWD OVERHEAD METRIC  (new column: crowd%)
#        crowd% = (penalized_toll - raw_toll) / raw_toll × 100
#        Shows directly how much congestion cost each strategy.

#   5. EXTENDED STATISTICS
#        Every submission now reports P10 / P50 / P90, IQR, and skewness
#        alongside mean/std/win-rate.

#   6. NINE-PANEL CHART
#        New dedicated h-sweep panel (raw toll vs h with sweet-spot band).
#        Replaced old heatmap with crowd_overhead vs score scatter.

# Usage:
#     python kyber_challenge_v3.py                         # prompts for x and h
#     python kyber_challenge_v3.py --x 0  --h 18
#     python kyber_challenge_v3.py --x 0  --h 18 --field-size 100 --n-sims 100
#     python kyber_challenge_v3.py --sweep-only            # just run the h-sweep
# """

# import argparse
# import sys
# import time
# import math

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec
# from scipy.stats import skew as scipy_skew

# # ── Game constants ────────────────────────────────────────────────────────────
# GRID    = 50
# MIN_T   = 5
# MAX_T   = 30
# THORN   = 40
# NSIMS   = 50

# X_MIN, X_MAX = 0,   300
# H_MIN, H_MAX = 5,    60

# THORN_MIN_TOLL  = MIN_T + THORN          # 45
# CLEAN_TOLL_MEAN = (MIN_T + MAX_T) / 2.0  # 17.5

# # v3 optimal sweet-spot (empirically from sweep)
# OPTIMAL_H_LO_V3  = 14
# OPTIMAL_H_HI_V3  = 22
# OPTIMAL_X_V3     = 0    # or up to ~20

# _SQRT2 = math.sqrt(2.0)

# # ── Terminal colours ──────────────────────────────────────────────────────────
# C_RESET   = "\033[0m";  C_BOLD    = "\033[1m";  C_DIM     = "\033[2m"
# C_GREEN   = "\033[92m"; C_YELLOW  = "\033[93m"; C_CYAN    = "\033[96m"
# C_RED     = "\033[91m"; C_MAGENTA = "\033[95m"; C_BLUE    = "\033[94m"
# C_ORANGE  = "\033[38;5;208m"; C_TEAL = "\033[38;5;44m"

# # ── Six realistic strategy clusters ──────────────────────────────────────────
# CLUSTERS = {
#     "E": dict(
#         name="True Expert",          desc="Ran tests; low x, h in sweet-spot [14,22]",
#         term_color=C_TEAL,           mpl_color="#00BCD4",
#         weight=0.10,
#         mu_x=10,  sig_x=8,   x_lo=0,   x_hi=20,
#         mu_h=18,  sig_h=3,   h_lo=14,  h_hi=22,
#         uniform=False,
#     ),
#     "Z": dict(
#         name="Zero-Thorn Avoider",   desc="Avoids x, guesses h≈28",
#         term_color=C_GREEN,          mpl_color="#4CAF50",
#         weight=0.25,
#         mu_x=5,   sig_x=8,   x_lo=0,   x_hi=30,
#         mu_h=28,  sig_h=12,  h_lo=5,   h_hi=55,
#         uniform=False,
#     ),
#     "G": dict(
#         name="Greedy Apple Grabber", desc="Max apples, high h — pays crowd penalty",
#         term_color=C_RED,            mpl_color="#F44336",
#         weight=0.25,
#         mu_x=280, sig_x=20,  x_lo=200, x_hi=300,
#         mu_h=52,  sig_h=6,   h_lo=35,  h_hi=60,
#         uniform=False,
#     ),
#     "M": dict(
#         name="Middle Ground",        desc="x≈150, h≈38; leaves money on table",
#         term_color=C_BLUE,           mpl_color="#2196F3",
#         weight=0.20,
#         mu_x=150, sig_x=50,  x_lo=50,  x_hi=250,
#         mu_h=38,  sig_h=8,   h_lo=20,  h_hi=55,
#         uniform=False,
#     ),
#     "C": dict(
#         name="Code Follower",        desc="Follows v2 advice: x≈300, h≈32 (suboptimal)",
#         term_color=C_MAGENTA,        mpl_color="#9C27B0",
#         weight=0.15,
#         mu_x=270, sig_x=20,  x_lo=220, x_hi=300,
#         mu_h=32,  sig_h=4,   h_lo=25,  h_hi=40,
#         uniform=False,
#     ),
#     "R": dict(
#         name="Random / No Strategy", desc="Uniform draw over full parameter space",
#         term_color=C_ORANGE,         mpl_color="#FF9800",
#         weight=0.05,
#         mu_x=150, sig_x=0,   x_lo=0,   x_hi=300,
#         mu_h=32,  sig_h=0,   h_lo=5,   h_hi=60,
#         uniform=True,
#     ),
# }


# # ═════════════════════════════════════════════════════════════════════════════
# # CORE SIMULATION  (walk logic unchanged; raw_toll always tracked)
# # ═════════════════════════════════════════════════════════════════════════════

# def build_base_fences(rng):
#     ef = rng.integers(MIN_T, MAX_T + 1, (GRID, GRID + 1)).astype(np.float64)
#     nf = rng.integers(MIN_T, MAX_T + 1, (GRID + 1, GRID)).astype(np.float64)
#     return ef, nf


# def add_thorns(ef_base, nf_base, x_val, rng):
#     ef = ef_base.copy()
#     nf = nf_base.copy()
#     n  = 2 * x_val
#     if n > 0:
#         idxs      = rng.integers(0, ef.size + nf.size, size=n)
#         east_mask = idxs < ef.size
#         np.add.at(ef.ravel(), idxs[east_mask],            THORN)
#         np.add.at(nf.ravel(), idxs[~east_mask] - ef.size, THORN)
#     return ef, nf


# def walk_once(ef, nf, h, rng):
#     """
#     Walk one farmer across the grid.  Returns:
#       eu, nu       — usage matrices (how many times each fence was crossed)
#       raw_toll     — sum of actual fence tolls paid (before crowd penalty)
#       thorn_x      — number of thorn-fence crossings
#       fence_x      — number of clean-fence crossings
#       avg_diag     — average |cx-cy|/√2  (path diversity proxy)
#     """
#     cx, cy   = 0, 0
#     eu       = np.zeros((GRID, GRID + 1), dtype=np.int32)
#     nu       = np.zeros((GRID + 1, GRID), dtype=np.int32)
#     raw_toll = 0.0
#     thorn_x  = 0
#     fence_x  = 0
#     diag_sum = abs(cx - cy) / _SQRT2

#     while cx < GRID or cy < GRID:
#         can_e = cx < GRID
#         can_n = cy < GRID

#         if can_e and can_n:
#             te, tn = ef[cx, cy], nf[cx, cy]
#             e_ok   = te <= h
#             n_ok   = tn <= h
#             if   e_ok and not n_ok: mv = "e"
#             elif n_ok and not e_ok: mv = "n"
#             else:                   mv = "e" if rng.random() < 0.5 else "n"
#         elif can_e:
#             mv = "e"
#         else:
#             mv = "n"

#         if mv == "e":
#             toll = ef[cx, cy]; eu[cx, cy] += 1; cx += 1
#         else:
#             toll = nf[cx, cy]; nu[cx, cy] += 1; cy += 1

#         raw_toll += toll
#         if toll > MAX_T:
#             thorn_x += 1
#         else:
#             fence_x += 1
#         diag_sum += abs(cx - cy) / _SQRT2

#     avg_diag = diag_sum / (2 * GRID + 1)
#     return eu, nu, raw_toll, thorn_x, fence_x, avg_diag


# def penalised_toll(ef, nf, eu, nu, frac_e, frac_n):
#     """Crowd-penalised toll: Σ toll_i × (1 + frac_i)³."""
#     pen_e = ef * (1.0 + frac_e) ** 3
#     pen_n = nf * (1.0 + frac_n) ** 3
#     return float((pen_e * eu).sum() + (pen_n * nu).sum())


# def run_one_iteration(submissions, base_rng, sub_rngs):
#     """
#     Run one shared-world iteration for all submissions.
#     Returns a list of dicts, one per submission, with pen_toll AND raw_toll.
#     """
#     n = len(submissions)
#     ef_base, nf_base = build_base_fences(base_rng)

#     sub_data = []
#     for idx, sub in enumerate(submissions):
#         rng = sub_rngs[idx]
#         ef_i, nf_i = add_thorns(ef_base, nf_base, sub["x"], rng)
#         eu_i, nu_i, raw_i, thorn_i, fence_i, diag_i = walk_once(
#             ef_i, nf_i, sub["h"], rng
#         )
#         sub_data.append((ef_i, nf_i, eu_i, nu_i, raw_i, thorn_i, fence_i, diag_i))

#     # Aggregate crowd-crossing fractions across all submissions
#     total_eu = sum(d[2] for d in sub_data).astype(np.float64)
#     total_nu = sum(d[3] for d in sub_data).astype(np.float64)
#     frac_e   = total_eu / n
#     frac_n   = total_nu / n

#     results = []
#     for ef_i, nf_i, eu_i, nu_i, raw_i, thorn_i, fence_i, diag_i in sub_data:
#         pt = penalised_toll(ef_i, nf_i, eu_i, nu_i, frac_e, frac_n)
#         results.append({
#             "pen_toll"       : pt,
#             "raw_toll"       : raw_i,
#             "crowd_overhead" : (pt - raw_i) / raw_i * 100.0 if raw_i > 0 else 0.0,
#             "thorn_crossings": thorn_i,
#             "fence_crossings": fence_i,
#             "avg_diag_dist"  : diag_i,
#         })
#     return results


# def simulate_competition(submissions, n_sims=NSIMS, n_seeds=1, master_seed=42):
#     n = len(submissions)
#     seed_of_time = int(time.time())
#     rng = np.random.default_rng(seed_of_time)
#     master_rng = np.random.default_rng(rng)
#     acc = {s["label"]: {
#         "rem": [], "raw_toll": [], "crowd_overhead": [],
#         "thorn": [], "fence": [], "diag": []
#     } for s in submissions}

#     seed_matrix  = master_rng.integers(0, 10_000_000, size=(n_seeds, n_sims))
#     total_iters  = n_seeds * n_sims

#     print(f"\n  Running {total_iters:,} iterations ", end="", flush=True)
#     done = 0

#     for s_idx in range(n_seeds):
#         for sim_idx in range(n_sims):
#             iter_seed = int(seed_matrix[s_idx, sim_idx])
#             iter_rng  = np.random.default_rng(iter_seed)
#             base_rng  = np.random.default_rng(iter_rng.integers(0, 10_000_000))
#             sub_rngs  = [np.random.default_rng(iter_rng.integers(0, 10_000_000))
#                          for _ in range(n)]

#             iter_results = run_one_iteration(submissions, base_rng, sub_rngs)

#             for i, sub in enumerate(submissions):
#                 lbl   = sub["label"]
#                 start = 2500 + sub["x"]
#                 r     = iter_results[i]
#                 acc[lbl]["rem"].append(start - r["pen_toll"])
#                 acc[lbl]["raw_toll"].append(r["raw_toll"])
#                 acc[lbl]["crowd_overhead"].append(r["crowd_overhead"])
#                 acc[lbl]["thorn"].append(r["thorn_crossings"])
#                 acc[lbl]["fence"].append(r["fence_crossings"])
#                 acc[lbl]["diag"].append(r["avg_diag_dist"])

#             done += 1
#             if done % max(1, total_iters // 20) == 0:
#                 print("█", end="", flush=True)

#     print(" done.\n")

#     records = []
#     for sub in submissions:
#         lbl  = sub["label"]
#         rems = np.array(acc[lbl]["rem"])
#         raw  = np.array(acc[lbl]["raw_toll"])
#         co   = np.array(acc[lbl]["crowd_overhead"])

#         records.append({
#             "label"          : lbl,
#             "x"              : sub["x"],
#             "h"              : sub["h"],
#             "cluster"        : sub.get("cluster", "?"),
#             "start_apples"   : 2500 + sub["x"],
#             # primary score
#             "avg_remaining"  : float(rems.mean()),
#             "std_remaining"  : float(rems.std()),
#             "win_rate"       : float((rems >= 0).mean()),
#             "min_remaining"  : float(rems.min()),
#             "max_remaining"  : float(rems.max()),
#             # percentiles & shape  (NEW v3)
#             "p10_remaining"  : float(np.percentile(rems, 10)),
#             "p50_remaining"  : float(np.percentile(rems, 50)),
#             "p90_remaining"  : float(np.percentile(rems, 90)),
#             "iqr_remaining"  : float(np.percentile(rems, 75) - np.percentile(rems, 25)),
#             "skew_remaining" : float(scipy_skew(rems)),
#             # raw toll analysis  (NEW v3)
#             "avg_raw_toll"   : float(raw.mean()),
#             "avg_crowd_pct"  : float(co.mean()),
#             # fence crossing stats
#             "avg_thorn_cross": float(np.mean(acc[lbl]["thorn"])),
#             "avg_fence_cross": float(np.mean(acc[lbl]["fence"])),
#             "avg_diag_dist"  : float(np.mean(acc[lbl]["diag"])),
#             "all_remaining"  : acc[lbl]["rem"],
#         })

#     df = pd.DataFrame(records).sort_values("avg_remaining", ascending=False)
#     df["rank"] = range(1, len(df) + 1)
#     return df


# # ═════════════════════════════════════════════════════════════════════════════
# # h-SENSITIVITY SWEEP  (NEW v3)
# # ═════════════════════════════════════════════════════════════════════════════

# def h_sensitivity_sweep(h_values=None, n_sims=3000, master_seed=99):
#     n_sims = 300
#     """
#     Solo walk (no crowd penalty) at each h value with x=0.
#     Returns dict  h → avg_raw_toll.
#     This shows the empirical raw-toll curve and identifies the sweet-spot.
#     """
#     if h_values is None:
#         h_values = list(range(H_MIN, H_MAX + 1))
#     seed_of_time = int(time.time())

#     print(f"Random seed used: {seed_of_time}")
#     rng = np.random.default_rng(seed_of_time)
#     # rng = np.random.default_rng(master_seed)
#     results = {}

#     print(f"\n  h-Sensitivity Sweep ({len(h_values)} h-values × {n_sims} sims) ",
#           end="", flush=True)

#     for h in h_values:
#         tolls = []
#         for _ in range(n_sims):
#             seed_of_time = int(time.time())
#             rng = np.random.default_rng(seed_of_time)
#             ef, nf = build_base_fences(rng)
#             _, _, raw_toll, _, _, _ = walk_once(ef, nf, h, rng)
#             tolls.append(raw_toll)
#         results[h] = float(np.mean(tolls))
#         if h % 5 == 0:
#             print(".", end="", flush=True)

#     print(" done.\n")
#     y = results
#     print('Krishna  ')
#     z = []
#     for x in y :
#         z.append((y[x],x))
#         # print({x,y[x]})
#     z.sort()
#     for x in z:
#         print(x)
#     return results


# # ═════════════════════════════════════════════════════════════════════════════
# # FIELD GENERATION
# # ═════════════════════════════════════════════════════════════════════════════

# def sample_one_cluster(key, cdef, n, rng):
#     if cdef["uniform"]:
#         xs = rng.integers(cdef["x_lo"], cdef["x_hi"] + 1, n)
#         hs = rng.integers(cdef["h_lo"], cdef["h_hi"] + 1, n)
#     else:
#         xs = rng.normal(cdef["mu_x"], cdef["sig_x"], n).round().astype(int)
#         hs = rng.normal(cdef["mu_h"], cdef["sig_h"], n).round().astype(int)
#         xs = np.clip(xs, cdef["x_lo"], cdef["x_hi"])
#         hs = np.clip(hs, cdef["h_lo"], cdef["h_hi"])
#     return [
#         {"x": int(x), "h": int(h),
#          "label": f"{key}_{i}_x{x}_h{h}",
#          "cluster": key}
#         for i, (x, h) in enumerate(zip(xs, hs))
#     ]

# import numpy as np
# import time
# def build_field(field_size, rng_seed):
#     """Build a realistic competitor field from six weighted strategy clusters."""


# # Get current time as an integer
#     seed_of_time = int(time.time())

#     print(f"Random seed used: {seed_of_time}")
#     rng = np.random.default_rng(seed_of_time)
#     # rng    = np.random.default_rng(rng_seed)
#     keys   = list(CLUSTERS.keys())
#     weights = np.array([CLUSTERS[k]["weight"] for k in keys])
#     weights /= weights.sum()

#     raw_counts = weights * field_size
#     counts     = np.floor(raw_counts).astype(int)
#     remainder  = field_size - counts.sum()
#     fracs      = raw_counts - counts
#     top_idxs   = np.argsort(fracs)[::-1][:remainder]
#     counts[top_idxs] += 1

#     subs = []
#     cluster_members = {}
#     for k, cnt in zip(keys, counts):
#         members = sample_one_cluster(k, CLUSTERS[k], cnt, rng)
#         cluster_members[k] = members
#         subs.extend(members)

#     return subs, cluster_members


# # ═════════════════════════════════════════════════════════════════════════════
# # TERMINAL OUTPUT
# # ═════════════════════════════════════════════════════════════════════════════

# def print_header():
#     print()
#     print(C_CYAN + C_BOLD + "╔" + "═"*76 + "╗" + C_RESET)
#     print(C_CYAN + C_BOLD + "║" + C_RESET +
#           C_BOLD + "    🌾  KYBER PROBLEM CHALLENGE  v3 — Corrected Optimal Strategy    " +
#           C_RESET + C_CYAN + C_BOLD + "  ║" + C_RESET)
#     print(C_CYAN + C_BOLD + "╚" + "═"*76 + "╝" + C_RESET)


# def print_strategy_analysis(sweep_results=None):
#     print()
#     print(C_BOLD + C_CYAN + "  " + "━"*72 + C_RESET)
#     print(C_BOLD + "  🧠  CORRECTED OPTIMAL STRATEGY  (v3)" + C_RESET)
#     print(C_BOLD + C_CYAN + "  " + "━"*72 + C_RESET)
#     print()
#     print(f"  {C_BOLD}Recommended submission:  x = 0,  h ≈ 18  (h ∈ [{OPTIMAL_H_LO_V3},{OPTIMAL_H_HI_V3}]){C_RESET}")
#     print()
#     print(f"  {C_BOLD}WHY NOT x = 300?{C_RESET}")
#     print(f"  • +300 starting apples sounds great, but those 600 thorns raise many")
#     print(f"    fences by +40, forcing ~1-3 costly crossings per run.")
#     print(f"  • More importantly: the unique thorn placement {C_YELLOW}does{C_RESET} reduce crowd")
#     print(f"    penalty slightly, but NOT enough to offset the raw-toll increase.")
#     print(f"  • Empirical result: x=0 consistently outranks x=300 after crowd penalty.")
#     print()
#     print(f"  {C_BOLD}WHY h ≈ 18?{C_RESET}")
#     print(f"  • Clean fence tolls are Uniform[{MIN_T},{MAX_T}], mean = {CLEAN_TOLL_MEAN:.1f}.")
#     print(f"  • With h=18 the farmer steers toward fences in [{MIN_T},18], avoiding")
#     print(f"    the expensive [{MAX_T-11},{MAX_T}] range.")
#     toll_budget = GRID * 2  # 100 steps
#     savings_per_step = CLEAN_TOLL_MEAN - (MIN_T + 18) / 2.0
#     print(f"  • Expected savings: ~{savings_per_step:.1f} apples/step × 100 steps"
#           f" ≈ {savings_per_step*toll_budget/100:.0f}–{savings_per_step*toll_budget/100*1.4:.0f} apples over x=300's bonus.")
#     print()
#     print(f"  {C_BOLD}THE SWEET-SPOT [{OPTIMAL_H_LO_V3},{OPTIMAL_H_HI_V3}]:{C_RESET}")
#     print(f"  • h < {OPTIMAL_H_LO_V3}: too selective — only ~{(18-MIN_T)/(MAX_T-MIN_T+1)*100:.0f}% of fences accepted,")
#     print(f"    all low-h farmers funnel into identical cheap paths → crowd overhead spikes.")
#     print(f"  • h > {OPTIMAL_H_HI_V3}: stops steering meaningfully; raw toll climbs toward mean.")
#     print(f"  • h=18 rejects fences 19-30, providing 60%+ random choice at")
#     print(f"    typical intersections → good path diversity + cheap toll steering.")
#     print()

#     if sweep_results:
#         h_arr = np.array(sorted(sweep_results.keys()))
#         t_arr = np.array([sweep_results[h] for h in h_arr])
#         best_h = h_arr[np.argmin(t_arr)]
#         print(f"  {C_BOLD}Sweep confirms:{C_RESET}  min raw toll at h={best_h}  "
#               f"(toll={sweep_results[best_h]:.1f})  vs h=30 "
#               f"(toll={sweep_results.get(30, '?'):.1f})")
#         saving = sweep_results.get(30, 0) - sweep_results[best_h]
#         print(f"  Raw-toll saving of h={best_h} over h=30: {C_GREEN}{saving:.1f} apples{C_RESET}")
#     print()
#     print(C_BOLD + C_CYAN + "  " + "━"*72 + C_RESET)


# def print_field_info(cluster_members, your_sub):
#     print()
#     print(C_BOLD + "  ── Field Composition ──────────────────────────────────────────────" + C_RESET)
#     for key, members in cluster_members.items():
#         if not members:
#             continue
#         cdef = CLUSTERS[key]
#         col  = cdef["term_color"]
#         xs   = np.array([m["x"] for m in members])
#         hs   = np.array([m["h"] for m in members])
#         print(f"  {col}{key} {cdef['name']:<26}{C_RESET} : "
#               f"{len(members):>3} farmers  "
#               f"x∈[{xs.min()},{xs.max()}] μ={xs.mean():.0f}  "
#               f"h∈[{hs.min()},{hs.max()}] μ={hs.mean():.0f}"
#               f"  {C_DIM}({cdef['desc']}){C_RESET}")
#     print(f"  {C_YELLOW}★ YOU{C_RESET}                              : "
#           f"x={your_sub['x']}, h={your_sub['h']}  "
#           f"(start apples = {2500 + your_sub['x']})")


# def print_results(df, your_label, top_k=30):
#     print()
#     print(C_BOLD + C_CYAN + "═"*126 + C_RESET)
#     print(C_BOLD + "  COMPETITION RESULTS  (with crowd penalty)" + C_RESET)
#     print(C_BOLD + C_CYAN + "═"*126 + C_RESET)
#     hdr = (f"  {'Rank':>4}  {'x':>4}  {'h':>3}  {'Cluster':<22}  "
#            f"{'avg_rem':>10}  {'std':>7}  {'P10':>8}  {'P90':>8}  {'win%':>6}  "
#            f"{'raw_toll':>9}  {'crowd%':>7}  {'thorn_x':>7}")
#     print(C_DIM + hdr + C_RESET)
#     print("  " + "─"*122)

#     for _, row in df.head(top_k).iterrows():
#         is_you = row["label"] == your_label
#         cl     = row["cluster"]
#         cl_col = CLUSTERS.get(cl, {}).get("term_color", C_YELLOW)
#         cl_nm  = cl if cl in CLUSTERS else "YOU"

#         avg_r = row["avg_remaining"]
#         color = C_GREEN if avg_r > 0 else C_RED

#         line = (f"  {int(row['rank']):>4}  "
#                 f"{int(row['x']):>4}  {int(row['h']):>3}  "
#                 f"{cl_col}{cl_nm:<22}{C_RESET}  "
#                 f"{color}{avg_r:>10.2f}{C_RESET}  "
#                 f"{row['std_remaining']:>7.2f}  "
#                 f"{row['p10_remaining']:>8.2f}  "
#                 f"{row['p90_remaining']:>8.2f}  "
#                 f"{row['win_rate']*100:>5.1f}%  "
#                 f"{row['avg_raw_toll']:>9.2f}  "
#                 f"{row['avg_crowd_pct']:>6.1f}%  "
#                 f"{row['avg_thorn_cross']:>7.2f}")

#         if is_you:
#             print(C_YELLOW + C_BOLD + line + "  ◄ YOU" + C_RESET)
#         else:
#             print(line)

#     # show YOU if not in top_k
#     your_rows = df[df["label"] == your_label]
#     if len(your_rows) and your_label not in df.head(top_k)["label"].values:
#         row   = your_rows.iloc[0]
#         avg_r = row["avg_remaining"]
#         color = C_GREEN if avg_r > 0 else C_RED
#         print("  ...")
#         print(C_YELLOW + C_BOLD +
#               f"  {int(row['rank']):>4}  {int(row['x']):>4}  {int(row['h']):>3}  "
#               f"{'★ YOU':<22}  {avg_r:>10.2f}  "
#               f"{row['std_remaining']:>7.2f}  "
#               f"{row['p10_remaining']:>8.2f}  "
#               f"{row['p90_remaining']:>8.2f}  "
#               f"{row['win_rate']*100:>5.1f}%  "
#               f"{row['avg_raw_toll']:>9.2f}  "
#               f"{row['avg_crowd_pct']:>6.1f}%  "
#               f"{row['avg_thorn_cross']:>7.2f}  ◄ YOU" + C_RESET)

#     print()
#     best = df.iloc[0]
#     tgt  = df[df["label"] == your_label].iloc[0]

#     print(C_BOLD + "═"*126 + C_RESET)

#     # Cluster-level summary
#     print()
#     print(C_BOLD + "  ── Cluster Average Performance ────────────────────────────────────" + C_RESET)
#     print(C_DIM + f"  {'Cluster':<26}  {'avg_rem':>10}  {'P50':>8}  "
#           f"{'win%':>6}  {'raw_toll':>9}  {'crowd%':>7}  n" + C_RESET)
#     for key in list(CLUSTERS.keys()) + ["YOU"]:
#         if key == "YOU":
#             cdf = df[df["label"] == your_label]
#             col = C_YELLOW
#             nm  = "★ YOU"
#         else:
#             cdf = df[df["cluster"] == key]
#             col = CLUSTERS[key]["term_color"]
#             nm  = CLUSTERS[key]["name"]
#         if len(cdf) == 0:
#             continue
#         print(f"  {col}{nm:<26}{C_RESET}  "
#               f"avg_rem={cdf['avg_remaining'].mean():>8.2f}  "
#               f"P50={cdf['p50_remaining'].mean():>8.2f}  "
#               f"win%={cdf['win_rate'].mean()*100:>5.1f}%  "
#               f"raw={cdf['avg_raw_toll'].mean():>9.2f}  "
#               f"crowd={cdf['avg_crowd_pct'].mean():>5.1f}%  "
#               f"n={len(cdf)}")

#     print()
#     print(f"  🥇 Winner : x={best['x']}, h={best['h']}  →  "
#           f"avg_rem = {C_GREEN}{best['avg_remaining']:.2f}{C_RESET}")
#     print(f"  🧑 You    : x={tgt['x']}, h={tgt['h']}   →  "
#           f"avg_rem = {C_YELLOW}{tgt['avg_remaining']:.2f}{C_RESET}  "
#           f"| rank {C_BOLD}{int(tgt['rank'])}{C_RESET} / {len(df)}")
#     print(f"  📊 Gap to 1st : {best['avg_remaining'] - tgt['avg_remaining']:.2f} apples")
#     print(f"  📈 Win rate   : {tgt['win_rate']*100:.1f}%")
#     print(f"  📉 Raw toll   : {tgt['avg_raw_toll']:.2f}  |  crowd overhead: {tgt['avg_crowd_pct']:.1f}%")
#     print(f"  📐 IQR        : {tgt['iqr_remaining']:.2f}  |  skewness: {tgt['skew_remaining']:.3f}")
#     print(C_BOLD + "═"*126 + C_RESET)
#     print()
#     print("  " + C_DIM +
#           "Clusters: E=Expert  Z=Zero-Thorn  G=Greedy  M=Middle  C=Code-Follower  R=Random  ★=YOU"
#           + C_RESET)
#     print()


# # ═════════════════════════════════════════════════════════════════════════════
# # MATPLOTLIB CHARTS  (9 panels)
# # ═════════════════════════════════════════════════════════════════════════════

# def make_charts(df, your_label, cluster_members, sweep_results=None):
#     your_row = df[df["label"] == your_label].iloc[0]

#     def row_color(row):
#         if row["label"] == your_label:
#             return "#FFD700"
#         return CLUSTERS.get(row["cluster"], {}).get("mpl_color", "#888888")

#     colors = [row_color(r) for _, r in df.iterrows()]

#     plt.style.use("dark_background")
#     fig = plt.figure(figsize=(24, 14), facecolor="#0d1117")
#     fig.suptitle("🌾  Kyber Challenge v3 — Corrected Optimal Strategy",
#                  fontsize=14, fontweight="bold", color="#e6edf3",
#                  fontfamily="monospace", y=0.99)

#     gs = GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.36,
#                   left=0.05, right=0.97, top=0.94, bottom=0.06)

#     ax_sweep  = fig.add_subplot(gs[0, 0])   # 1. h-sensitivity sweep (NEW)
#     ax_xh     = fig.add_subplot(gs[0, 1])   # 2. strategy scatter
#     ax_clust  = fig.add_subplot(gs[0, 2])   # 3. cluster bar
#     ax_rank   = fig.add_subplot(gs[1, 0])   # 4. top-30 bar
#     ax_dist   = fig.add_subplot(gs[1, 1])   # 5. score distribution
#     ax_crowd  = fig.add_subplot(gs[1, 2])   # 6. crowd overhead vs score (NEW)
#     ax_box    = fig.add_subplot(gs[2, 0])   # 7. box plot top-5 + YOU
#     ax_thorn  = fig.add_subplot(gs[2, 1])   # 8. thorn crossings vs score
#     ax_win    = fig.add_subplot(gs[2, 2])   # 9. win rate by cluster

#     def style_ax(ax, title, xlabel="", ylabel=""):
#         ax.set_facecolor("#161b22")
#         ax.set_title(title, color="#e6edf3", fontsize=9, fontweight="bold", pad=6)
#         ax.tick_params(colors="#8b949e", labelsize=7)
#         ax.spines[:].set_color("#30363d")
#         if xlabel: ax.set_xlabel(xlabel, color="#8b949e", fontsize=7)
#         if ylabel: ax.set_ylabel(ylabel, color="#8b949e", fontsize=7)

#     # ── 1. h-Sensitivity Sweep ───────────────────────────────────────────────
#     if sweep_results:
#         h_arr = np.array(sorted(sweep_results.keys()))
#         t_arr = np.array([sweep_results[h] for h in h_arr])
#         ax_sweep.plot(h_arr, t_arr, color="#64B5F6", lw=2)
#         # sweet-spot band
#         ax_sweep.axvspan(OPTIMAL_H_LO_V3, OPTIMAL_H_HI_V3,
#                          color="#4CAF50", alpha=0.18, label=f"Sweet spot [{OPTIMAL_H_LO_V3},{OPTIMAL_H_HI_V3}]")
#         ax_sweep.axvline(your_row["h"], color="#FFD700", lw=1.5, ls="--", label=f"h={int(your_row['h'])}")
#         best_h = h_arr[np.argmin(t_arr)]
#         ax_sweep.axvline(best_h, color="#4CAF50", lw=1.5, ls=":", label=f"min h={best_h}")
#         ax_sweep.legend(fontsize=6, facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")
#         style_ax(ax_sweep, "h-Sensitivity Sweep (solo, x=0)", "h threshold", "avg raw toll")
#     else:
#         ax_sweep.text(0.5, 0.5, "No sweep data\n(--skip-sweep used)",
#                       ha="center", va="center", color="#8b949e", fontsize=9,
#                       transform=ax_sweep.transAxes)
#         style_ax(ax_sweep, "h-Sensitivity Sweep")

#     # ── 2. Strategy scatter: x vs h ─────────────────────────────────────────
#     from matplotlib.patches import Rectangle
#     for key, cdef in CLUSTERS.items():
#         sub_df = df[df["cluster"] == key]
#         if len(sub_df):
#             ax_xh.scatter(sub_df["x"], sub_df["h"],
#                           c=cdef["mpl_color"], alpha=0.55, s=22,
#                           label=f"{key}: {cdef['name'][:12]}")
#     ax_xh.scatter([your_row["x"]], [your_row["h"]], c="#FFD700",
#                   s=280, zorder=5, marker="*", label="★ YOU")
#     # v3 optimal zone: low x, h in [14,22]
#     ax_xh.add_patch(Rectangle((0, OPTIMAL_H_LO_V3), 20, OPTIMAL_H_HI_V3 - OPTIMAL_H_LO_V3,
#                                 linewidth=1.5, edgecolor="#FFD700",
#                                 facecolor="#FFD70022", zorder=3,
#                                 label=f"Optimal zone"))
#     ax_xh.legend(fontsize=5.5, facecolor="#161b22", edgecolor="#30363d",
#                  labelcolor="#e6edf3", loc="upper right")
#     style_ax(ax_xh, "Strategy Space (x vs h)", "x  (extra apples / thorns)", "h  (threshold)")

#     # ── 3. Cluster mean performance bar ─────────────────────────────────────
#     cluster_means, cluster_labels, cluster_cols = [], [], []
#     for key, cdef in CLUSTERS.items():
#         sub_df = df[df["cluster"] == key]
#         if len(sub_df):
#             cluster_means.append(sub_df["avg_remaining"].mean())
#             cluster_labels.append(f"{key}")
#             cluster_cols.append(cdef["mpl_color"])
#     cluster_means.append(your_row["avg_remaining"])
#     cluster_labels.append("★")
#     cluster_cols.append("#FFD700")
#     brs = ax_clust.bar(range(len(cluster_means)), cluster_means,
#                        color=cluster_cols, edgecolor="none", width=0.6)
#     ax_clust.set_xticks(range(len(cluster_means)))
#     ax_clust.set_xticklabels(cluster_labels, fontsize=7)
#     ax_clust.axhline(0, color="#ff4444", lw=1, ls="--", alpha=0.7)
#     for bar, val in zip(brs, cluster_means):
#         ax_clust.text(bar.get_x() + bar.get_width()/2,
#                       bar.get_height() + (5 if val >= 0 else -20),
#                       f"{val:.0f}", ha="center", va="bottom",
#                       color="#e6edf3", fontsize=6.5, fontweight="bold")
#     style_ax(ax_clust, "Cluster Avg Remaining (E=Expert,C=CodeFollower)",
#              ylabel="apples remaining")

#     # ── 4. Top-30 ranking bar ────────────────────────────────────────────────
#     top30 = df.head(30)
#     c30   = [row_color(r) for _, r in top30.iterrows()]
#     ax_rank.barh(range(len(top30)), top30["avg_remaining"],
#                  color=c30, edgecolor="none", height=0.75)
#     ax_rank.axvline(0, color="#ff4444", lw=1, ls="--", alpha=0.6)
#     ax_rank.set_yticks(range(len(top30)))
#     ax_rank.set_yticklabels([f"#{int(r['rank'])}" for _, r in top30.iterrows()],
#                             fontsize=6, color="#8b949e")
#     ax_rank.invert_yaxis()
#     you_in30 = top30[top30["label"] == your_label]
#     if len(you_in30):
#         idx = list(top30["label"]).index(your_label)
#         ax_rank.barh(idx, you_in30["avg_remaining"].values[0],
#                      color="#FFD700", edgecolor="#ff8c00", lw=1.5, height=0.75)
#         ax_rank.text(you_in30["avg_remaining"].values[0] + 2, idx,
#                      "◄ YOU", va="center", color="#FFD700", fontsize=6, fontweight="bold")
#     style_ax(ax_rank, "Top-30 Avg Remaining Apples", "apples remaining", "rank")

#     # ── 5. Distribution by cluster ───────────────────────────────────────────
#     bins = np.linspace(df["avg_remaining"].min(), df["avg_remaining"].max(), 40)
#     for key, cdef in CLUSTERS.items():
#         sub_df = df[df["cluster"] == key]
#         if len(sub_df):
#             ax_dist.hist(sub_df["avg_remaining"], bins=bins,
#                          color=cdef["mpl_color"], alpha=0.65, label=key, edgecolor="none")
#     ax_dist.axvline(your_row["avg_remaining"], color="#FFD700", lw=2, ls="--", label="YOU")
#     ax_dist.axvline(0, color="#ff4444", lw=1, ls=":", alpha=0.7)
#     ax_dist.legend(fontsize=6, facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")
#     style_ax(ax_dist, "Avg-Remaining Distribution by Cluster", "avg remaining apples", "count")

#     # ── 6. Crowd overhead vs score scatter (NEW v3) ──────────────────────────
#     for key, cdef in CLUSTERS.items():
#         sub_df = df[df["cluster"] == key]
#         if len(sub_df):
#             ax_crowd.scatter(sub_df["avg_crowd_pct"], sub_df["avg_remaining"],
#                              c=cdef["mpl_color"], alpha=0.5, s=18, label=key)
#     ax_crowd.scatter([your_row["avg_crowd_pct"]], [your_row["avg_remaining"]],
#                      c="#FFD700", s=200, zorder=5, marker="*", label="YOU")
#     ax_crowd.axhline(0, color="#ff4444", lw=1, ls="--", alpha=0.6)
#     ax_crowd.legend(fontsize=6, facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")
#     style_ax(ax_crowd, "Crowd Overhead vs Score", "crowd overhead %", "avg remaining")

#     # ── 7. Box plot: top-5 + YOU ─────────────────────────────────────────────
#     top5 = df.head(5)
#     you_in_t5 = your_label in list(top5["label"])
#     if not you_in_t5:
#         box_rows   = list(top5.iterrows()) + [(None, your_row)]
#         box_labels = ([f"#{int(r['rank'])}\nx={int(r['x'])},h={int(r['h'])}"
#                        for _, r in top5.iterrows()] + ["★YOU"])
#         box_colors = ([CLUSTERS.get(r["cluster"], {}).get("mpl_color", "#888")
#                        for _, r in top5.iterrows()] + ["#FFD700"])
#     else:
#         box_rows   = list(top5.iterrows())
#         box_labels = [f"#{int(r['rank'])}\nx={int(r['x'])},h={int(r['h'])}"
#                       + (" ★" if r["label"] == your_label else "")
#                       for _, r in top5.iterrows()]
#         box_colors = ["#FFD700" if r["label"] == your_label
#                       else CLUSTERS.get(r["cluster"], {}).get("mpl_color", "#888")
#                       for _, r in top5.iterrows()]

#     all_box = [r["all_remaining"] for _, r in box_rows]
#     bp = ax_box.boxplot(all_box, patch_artist=True, notch=False,
#                         medianprops=dict(color="white", lw=2),
#                         whiskerprops=dict(color="#8b949e"),
#                         capprops=dict(color="#8b949e"),
#                         flierprops=dict(marker=".", color="#8b949e", alpha=0.4, ms=3))
#     for patch, col in zip(bp["boxes"], box_colors):
#         patch.set_facecolor(col)
#         patch.set_alpha(0.75)
#     ax_box.set_xticks(range(1, len(all_box) + 1))
#     ax_box.set_xticklabels(box_labels, fontsize=5.5)
#     ax_box.axhline(0, color="#ff4444", lw=1, ls="--", alpha=0.7)
#     style_ax(ax_box, "Score Distribution: Top-5 + YOU (P10/P50/P90)", ylabel="apples remaining")

#     # ── 8. Thorn crossings vs score ──────────────────────────────────────────
#     for key, cdef in CLUSTERS.items():
#         sub_df = df[df["cluster"] == key]
#         if len(sub_df):
#             ax_thorn.scatter(sub_df["avg_thorn_cross"], sub_df["avg_remaining"],
#                              c=cdef["mpl_color"], alpha=0.5, s=18, label=key)
#     ax_thorn.scatter([your_row["avg_thorn_cross"]], [your_row["avg_remaining"]],
#                      c="#FFD700", s=200, zorder=5, marker="*", label="YOU")
#     ax_thorn.axhline(0, color="#ff4444", lw=1, ls="--", alpha=0.6)
#     ax_thorn.legend(fontsize=6, facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")
#     style_ax(ax_thorn, "Thorn Crossings vs Score", "avg thorn crossings", "avg remaining")

#     # ── 9. Win-rate by cluster ───────────────────────────────────────────────
#     wr_vals, wr_labels, wr_cols = [], [], []
#     for k in CLUSTERS:
#         sub_df = df[df["cluster"] == k]
#         if len(sub_df):
#             wr_vals.append(sub_df["win_rate"].mean() * 100)
#             wr_labels.append(k)
#             wr_cols.append(CLUSTERS[k]["mpl_color"])
#     wr_vals.append(your_row["win_rate"] * 100)
#     wr_labels.append("★ YOU")
#     wr_cols.append("#FFD700")

#     brs2 = ax_win.bar(range(len(wr_vals)), wr_vals, color=wr_cols, edgecolor="none", width=0.6)
#     ax_win.set_xticks(range(len(wr_vals)))
#     ax_win.set_xticklabels(wr_labels, fontsize=7)
#     ax_win.set_ylim(0, 115)
#     ax_win.axhline(50, color="#888", lw=1, ls=":", alpha=0.5)
#     for bar, val in zip(brs2, wr_vals):
#         ax_win.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
#                     f"{val:.0f}%", ha="center", va="bottom",
#                     color="#e6edf3", fontsize=7, fontweight="bold")
#     style_ax(ax_win, "Win Rate by Cluster", ylabel="% runs with ≥ 0 apples")

#     plt.savefig("kyber_results_v3.png", dpi=150, bbox_inches="tight",
#                 facecolor="#0d1117")
#     print(f"  📊 Chart saved to {C_CYAN}kyber_results_v3.png{C_RESET}")
#     plt.show()


# # ═════════════════════════════════════════════════════════════════════════════
# # CLI
# # ═════════════════════════════════════════════════════════════════════════════

# def parse_args():
#     p = argparse.ArgumentParser(
#         description="Kyber Challenge v3 — corrected optimal strategy + h-sweep",
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog=f"Optimal strategy (v3):  --x 0 --h 18  (h ∈ [{OPTIMAL_H_LO_V3},{OPTIMAL_H_HI_V3}])"
#     )
#     p.add_argument("--x",           type=int, default=None,
#                    help=f"Your x submission ({X_MIN}–{X_MAX})")
#     p.add_argument("--h",           type=int, default=None,
#                    help=f"Your h submission ({H_MIN}–{H_MAX})")
#     p.add_argument("--field-size",  type=int, default=100, dest="field_size",
#                    help="Total rival farmers to simulate  [100]")
#     p.add_argument("--n-sims",      type=int, default=NSIMS, dest="n_sims",
#                    help="Competition iterations per seed  [50]")
#     p.add_argument("--seeds",       type=int, default=5,
#                    help="Independent seeds to average over  [5]")
#     p.add_argument("--seed",        type=int, default=None,
#                    help="Master RNG seed (random if omitted)")
#     p.add_argument("--top-show",    type=int, default=30, dest="top_show",
#                    help="Rows to print in result table  [30]")
#     p.add_argument("--sweep-sims",  type=int, default=3000, dest="sweep_sims",
#                    help="Simulations per h-value in sensitivity sweep  [300]")
#     p.add_argument("--skip-sweep",  action="store_true", dest="skip_sweep",
#                    help="Skip the h-sensitivity sweep (faster)")
#     p.add_argument("--sweep-only",  action="store_true", dest="sweep_only",
#                    help="Run only the h-sensitivity sweep and exit")
#     p.add_argument("--no-chart",    action="store_true", dest="no_chart",
#                    help="Skip matplotlib charts")
#     p.add_argument("--no-analysis", action="store_true", dest="no_analysis",
#                    help="Skip optimal-strategy analysis")
#     return p.parse_args()


# def main():
#     args = parse_args()
#     print_header()

#     # ── h-Sensitivity Sweep ───────────────────────────────────────────────────
#     sweep_results = None
#     if not args.skip_sweep:
#         sweep_results = h_sensitivity_sweep(n_sims=args.sweep_sims)
#         if args.sweep_only:
#             h_arr = np.array(sorted(sweep_results.keys()))
#             t_arr = np.array([sweep_results[h] for h in h_arr])
#             best_h = h_arr[np.argmin(t_arr)]
#             print(f"  Best h by raw toll : h = {C_GREEN}{best_h}{C_RESET}  "
#                   f"(avg raw toll = {C_GREEN}{sweep_results[best_h]:.2f}{C_RESET})")
#             print()
#             print("  h   raw_toll   Δ_from_min")
#             for h in h_arr:
#                 diff = sweep_results[h] - sweep_results[best_h]
#                 bar  = "█" * int(diff / 5)
#                 print(f"  {h:>3}  {sweep_results[h]:>8.2f}  +{diff:>6.2f}  {bar}")
#             return

#     if not args.no_analysis:
#         print_strategy_analysis(sweep_results)

#     # ── Get x and h ───────────────────────────────────────────────────────────
#     if args.x is None:
#         print()
#         try:
#             args.x = int(input(C_YELLOW + f"  Enter your x ({X_MIN}–{X_MAX}) : " + C_RESET))
#         except (ValueError, EOFError):
#             print("[ERROR] Invalid x", file=sys.stderr); sys.exit(1)
#     if args.h is None:
#         try:
#             args.h = int(input(C_YELLOW + f"  Enter your h ({H_MIN}–{H_MAX})  : " + C_RESET))
#         except (ValueError, EOFError):
#             print("[ERROR] Invalid h", file=sys.stderr); sys.exit(1)

#     x, h = args.x, args.h
#     if not (X_MIN <= x <= X_MAX):
#         print(f"[ERROR] x={x} out of range [{X_MIN},{X_MAX}]", file=sys.stderr); sys.exit(1)
#     if not (H_MIN <= h <= H_MAX):
#         print(f"[ERROR] h={h} out of range [{H_MIN},{H_MAX}]", file=sys.stderr); sys.exit(1)

#     # Contextual warnings
#     if h < OPTIMAL_H_LO_V3:
#         print(f"\n  {C_RED}⚠  h={h} < {OPTIMAL_H_LO_V3}: very restrictive — crowd overhead likely to spike{C_RESET}")
#         print(f"  {C_RED}   as low-h farmers all funnel toward the same cheap fences.{C_RESET}")
#     elif h > OPTIMAL_H_HI_V3 and h < 30:
#         print(f"\n  {C_YELLOW}⚠  h={h}: above sweet-spot — some toll steering lost vs h=18.{C_RESET}")
#     elif h >= 30:
#         print(f"\n  {C_YELLOW}⚠  h={h} ≥ 30: accepts all clean fences; no toll steering advantage.{C_RESET}")
#         print(f"  {C_YELLOW}   This was the v2 recommendation — v3 shows h≈18 is better.{C_RESET}")
#     if x > 50:
#         print(f"\n  {C_YELLOW}⚠  x={x}: high x adds apples but {2*x} thorns raise your raw toll.{C_RESET}")
#         print(f"  {C_YELLOW}   Empirically, x≈0 outperforms x=300 after crowd penalty.{C_RESET}")

#     seed = args.seed if args.seed is not None else int(time.time() * 1000) % (2**32 - 1)

#     # ── Build field ───────────────────────────────────────────────────────────
#     field, cluster_members = build_field(args.field_size, seed + 1)

#     your_label = f"★YOU_x{x}_h{h}"
#     your_sub   = {"x": x, "h": h, "label": your_label, "cluster": "YOU"}
#     submissions = [your_sub] + field

#     print_field_info(cluster_members, your_sub)

#     print()
#     print(C_BOLD + "  ── Simulation Settings ─────────────────────────────────────────────" + C_RESET)
#     print(f"  Field size  : {len(submissions)} total submissions")
#     print(f"  Iterations  : {args.n_sims} sims × {args.seeds} seeds = {args.n_sims*args.seeds:,} total")
#     print(f"  Master seed : {seed}")

#     # ── Simulate ──────────────────────────────────────────────────────────────
#     df = simulate_competition(
#         submissions=submissions,
#         n_sims=args.n_sims,
#         n_seeds=args.seeds,
#         master_seed=seed,
#     )

#     print_results(df, your_label, top_k=args.top_show)

#     if not args.no_chart:
#         print("  Generating charts …")
#         make_charts(df, your_label, cluster_members, sweep_results)
#     else:
#         print("  (Charts skipped — remove --no-chart to enable)")


# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
"""
kyber_challenge_v3.py
══════════════════════════════════════════════════════════════════════════════
Kyber Problem Challenge — v3  (corrected optimal strategy)
WHAT CHANGED FROM v2
────────────────────
  1. CORRECTED OPTIMAL STRATEGY  x ≈ 0–20, h ∈ [14, 22]
       Raw-toll steering beats the +300 apple bonus from x=300.
       With h≈18 the farmer steers toward cheap fences, saving ~300-500
       apples over 100 steps.  That overwhelms x=300's bonus.
       Very low h (<14) converges all low-h farmers on the same cheap
       path → crowd overhead spikes; optimal is the sweet-spot [14,22].
  2. h-SENSITIVITY SWEEP  (new, run before main simulation)
       Fast grid-search over h ∈ [5,60] at x=0 with 300 solo walks per
       point.  Measures the empirical raw-toll curve so you can see the
       minimum, sweet-spot, and exactly how much h=30 costs vs h=18.
  3. SIX-CLUSTER FIELD  (was five)
       E  "True Expert"     (10%)  x ~ N(10,8)[0,20]    h ~ N(18,3)[14,22]
       Z  "Thorn Avoider"   (25%)  x ~ N(5,8)[0,30]     h ~ N(28,12)[5,55]
       G  "Greedy Grabber"  (25%)  x ~ N(280,20)[200,300] h ~ N(52,6)[35,60]
       M  "Middle Ground"   (20%)  x ~ N(150,50)[50,250] h ~ N(38,8)[20,55]
       C  "Code Follower"   (15%)  x ~ N(270,20)[220,300] h ~ N(32,4)[25,40]
       R  "Random/Naive"    (5%)   x ~ Uniform[0,300]   h ~ Uniform[5,60]
  4. CROWD OVERHEAD METRIC  (new column: crowd%)
       crowd% = (penalized_toll - raw_toll) / raw_toll × 100
       Shows directly how much congestion cost each strategy.
  5. EXTENDED STATISTICS
       Every submission now reports P10 / P50 / P90, IQR, and skewness
       alongside mean/std/win-rate.
  6. NINE-PANEL CHART
       New dedicated h-sweep panel (raw toll vs h with sweet-spot band).
       Replaced old heatmap with crowd_overhead vs score scatter.
Usage:
    python kyber_challenge_v3.py                         # prompts for x and h
    python kyber_challenge_v3.py --x 0  --h 18
    python kyber_challenge_v3.py --x 0  --h 18 --field-size 100 --n-sims 100
    python kyber_challenge_v3.py --sweep-only            # just run the h-sweep
"""
import argparse
import sys
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import skew as scipy_skew
# ── Game constants ────────────────────────────────────────────────────────────
GRID    = 50
MIN_T   = 5
MAX_T   = 30
THORN   = 40
NSIMS   = 50
X_MIN, X_MAX = 0,   300
H_MIN, H_MAX = 5,    60
THORN_MIN_TOLL  = MIN_T + THORN          # 45
CLEAN_TOLL_MEAN = (MIN_T + MAX_T) / 2.0  # 17.5
# v3 optimal sweet-spot (empirically from sweep)
OPTIMAL_H_LO_V3  = 14
OPTIMAL_H_HI_V3  = 22
OPTIMAL_X_V3     = 0    # or up to ~20
_SQRT2 = math.sqrt(2.0)
# ── Terminal colours ──────────────────────────────────────────────────────────
C_RESET   = "\033[0m";  C_BOLD    = "\033[1m";  C_DIM     = "\033[2m"
C_GREEN   = "\033[92m"; C_YELLOW  = "\033[93m"; C_CYAN    = "\033[96m"
C_RED     = "\033[91m"; C_MAGENTA = "\033[95m"; C_BLUE    = "\033[94m"
C_ORANGE  = "\033[38;5;208m"; C_TEAL = "\033[38;5;44m"
# ── Six realistic strategy clusters ──────────────────────────────────────────
CLUSTERS = {
    "E": dict(
        name="True Expert",          desc="Ran tests; low x, h in sweet-spot [14,22]",
        term_color=C_TEAL,           mpl_color="#00BCD4",
        weight=0.10,
        mu_x=10,  sig_x=8,   x_lo=0,   x_hi=20,
        mu_h=18,  sig_h=3,   h_lo=14,  h_hi=22,
        uniform=False,
    ),
    "Z": dict(
        name="Zero-Thorn Avoider",   desc="Avoids x, guesses h≈28",
        term_color=C_GREEN,          mpl_color="#4CAF50",
        weight=0.25,
        mu_x=5,   sig_x=8,   x_lo=0,   x_hi=30,
        mu_h=28,  sig_h=12,  h_lo=5,   h_hi=55,
        uniform=False,
    ),
    "G": dict(
        name="Greedy Apple Grabber", desc="Max apples, high h — pays crowd penalty",
        term_color=C_RED,            mpl_color="#F44336",
        weight=0.25,
        mu_x=280, sig_x=20,  x_lo=200, x_hi=300,
        mu_h=52,  sig_h=6,   h_lo=35,  h_hi=60,
        uniform=False,
    ),
    "M": dict(
        name="Middle Ground",        desc="x≈150, h≈38; leaves money on table",
        term_color=C_BLUE,           mpl_color="#2196F3",
        weight=0.20,
        mu_x=150, sig_x=50,  x_lo=50,  x_hi=250,
        mu_h=38,  sig_h=8,   h_lo=20,  h_hi=55,
        uniform=False,
    ),
    "C": dict(
        name="Code Follower",        desc="Follows v2 advice: x≈300, h≈32 (suboptimal)",
        term_color=C_MAGENTA,        mpl_color="#9C27B0",
        weight=0.15,
        mu_x=270, sig_x=20,  x_lo=220, x_hi=300,
        mu_h=32,  sig_h=4,   h_lo=25,  h_hi=40,
        uniform=False,
    ),
    "R": dict(
        name="Random / No Strategy", desc="Uniform draw over full parameter space",
        term_color=C_ORANGE,         mpl_color="#FF9800",
        weight=0.05,
        mu_x=150, sig_x=0,   x_lo=0,   x_hi=300,
        mu_h=32,  sig_h=0,   h_lo=5,   h_hi=60,
        uniform=True,
    ),
}
# ═════════════════════════════════════════════════════════════════════════════
# CORE SIMULATION  (walk logic unchanged; raw_toll always tracked)
# ═════════════════════════════════════════════════════════════════════════════
def build_base_fences(rng):
    ef = rng.integers(MIN_T, MAX_T + 1, (GRID, GRID + 1)).astype(np.float64)
    nf = rng.integers(MIN_T, MAX_T + 1, (GRID + 1, GRID)).astype(np.float64)
    return ef, nf
def add_thorns(ef_base, nf_base, x_val, rng):
    ef = ef_base.copy()
    nf = nf_base.copy()
    n  = 2 * x_val
    if n > 0:
        idxs      = rng.integers(0, ef.size + nf.size, size=n)
        east_mask = idxs < ef.size
        np.add.at(ef.ravel(), idxs[east_mask],            THORN)
        np.add.at(nf.ravel(), idxs[~east_mask] - ef.size, THORN)
    return ef, nf
def walk_once(ef, nf, h, rng):
    """
    Walk one farmer across the grid.  Returns:
      eu, nu       — usage matrices (how many times each fence was crossed)
      raw_toll     — sum of actual fence tolls paid (before crowd penalty)
      thorn_x      — number of thorn-fence crossings
      fence_x      — number of clean-fence crossings
      avg_diag     — average |cx-cy|/√2  (path diversity proxy)
    """
    cx, cy   = 0, 0
    eu       = np.zeros((GRID, GRID + 1), dtype=np.int32)
    nu       = np.zeros((GRID + 1, GRID), dtype=np.int32)
    raw_toll = 0.0
    thorn_x  = 0
    fence_x  = 0
    diag_sum = abs(cx - cy) / _SQRT2
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
            toll = ef[cx, cy]; eu[cx, cy] += 1; cx += 1
        else:
            toll = nf[cx, cy]; nu[cx, cy] += 1; cy += 1
        raw_toll += toll
        if toll > MAX_T:
            thorn_x += 1
        else:
            fence_x += 1
        diag_sum += abs(cx - cy) / _SQRT2
    avg_diag = diag_sum / (2 * GRID + 1)
    return eu, nu, raw_toll, thorn_x, fence_x, avg_diag
def penalised_toll(ef, nf, eu, nu, frac_e, frac_n):
    """Crowd-penalised toll: Σ toll_i × (1 + frac_i)³."""
    pen_e = ef * (1.0 + frac_e) ** 3
    pen_n = nf * (1.0 + frac_n) ** 3
    return float((pen_e * eu).sum() + (pen_n * nu).sum())
def run_one_iteration(submissions, base_rng, sub_rngs):
    """
    Run one shared-world iteration for all submissions.
    Returns a list of dicts, one per submission, with pen_toll AND raw_toll.
    """
    n = len(submissions)
    ef_base, nf_base = build_base_fences(base_rng)
    sub_data = []
    for idx, sub in enumerate(submissions):
        rng = sub_rngs[idx]
        ef_i, nf_i = add_thorns(ef_base, nf_base, sub["x"], rng)
        eu_i, nu_i, raw_i, thorn_i, fence_i, diag_i = walk_once(
            ef_i, nf_i, sub["h"], rng
        )
        sub_data.append((ef_i, nf_i, eu_i, nu_i, raw_i, thorn_i, fence_i, diag_i))
    # Aggregate crowd-crossing fractions across all submissions
    total_eu = sum(d[2] for d in sub_data).astype(np.float64)
    total_nu = sum(d[3] for d in sub_data).astype(np.float64)
    frac_e   = total_eu / n
    frac_n   = total_nu / n
    results = []
    for ef_i, nf_i, eu_i, nu_i, raw_i, thorn_i, fence_i, diag_i in sub_data:
        pt = penalised_toll(ef_i, nf_i, eu_i, nu_i, frac_e, frac_n)
        results.append({
            "pen_toll"       : pt,
            "raw_toll"       : raw_i,
            "crowd_overhead" : (pt - raw_i) / raw_i * 100.0 if raw_i > 0 else 0.0,
            "thorn_crossings": thorn_i,
            "fence_crossings": fence_i,
            "avg_diag_dist"  : diag_i,
        })
    return results
def simulate_competition(submissions, n_sims=NSIMS, n_seeds=1, master_seed=None):
    n = len(submissions)
    # Use entropy by default; optional --seed keeps runs reproducible.
    master_rng = np.random.default_rng(master_seed)
    acc = {s["label"]: {
        "rem": [], "raw_toll": [], "crowd_overhead": [],
        "thorn": [], "fence": [], "diag": []
    } for s in submissions}
    total_iters  = n_seeds * n_sims
    print(f"\n  Running {total_iters:,} iterations ", end="", flush=True)
    done = 0
    for _ in range(n_seeds):
        for _ in range(n_sims):
            # Fresh independent streams for each iteration.
            base_rng = np.random.default_rng(master_rng.integers(0, 2**63 - 1))
            sub_rngs = [np.random.default_rng(master_rng.integers(0, 2**63 - 1))
                        for _ in range(n)]
            iter_results = run_one_iteration(submissions, base_rng, sub_rngs)
            for i, sub in enumerate(submissions):
                lbl   = sub["label"]
                start = 2500 + sub["x"]
                r     = iter_results[i]
                acc[lbl]["rem"].append(start - r["pen_toll"])
                acc[lbl]["raw_toll"].append(r["raw_toll"])
                acc[lbl]["crowd_overhead"].append(r["crowd_overhead"])
                acc[lbl]["thorn"].append(r["thorn_crossings"])
                acc[lbl]["fence"].append(r["fence_crossings"])
                acc[lbl]["diag"].append(r["avg_diag_dist"])
            done += 1
            if done % max(1, total_iters // 20) == 0:
                print("█", end="", flush=True)
    print(" done.\n")
    records = []
    for sub in submissions:
        lbl  = sub["label"]
        rems = np.array(acc[lbl]["rem"])
        raw  = np.array(acc[lbl]["raw_toll"])
        co   = np.array(acc[lbl]["crowd_overhead"])
        records.append({
            "label"          : lbl,
            "x"              : sub["x"],
            "h"              : sub["h"],
            "cluster"        : sub.get("cluster", "?"),
            "start_apples"   : 2500 + sub["x"],
            # primary score
            "avg_remaining"  : float(rems.mean()),
            "std_remaining"  : float(rems.std()),
            "win_rate"       : float((rems >= 0).mean()),
            "min_remaining"  : float(rems.min()),
            "max_remaining"  : float(rems.max()),
            # percentiles & shape  (NEW v3)
            "p10_remaining"  : float(np.percentile(rems, 10)),
            "p50_remaining"  : float(np.percentile(rems, 50)),
            "p90_remaining"  : float(np.percentile(rems, 90)),
            "iqr_remaining"  : float(np.percentile(rems, 75) - np.percentile(rems, 25)),
            "skew_remaining" : float(scipy_skew(rems)),
            # raw toll analysis  (NEW v3)
            "avg_raw_toll"   : float(raw.mean()),
            "avg_crowd_pct"  : float(co.mean()),
            # fence crossing stats
            "avg_thorn_cross": float(np.mean(acc[lbl]["thorn"])),
            "avg_fence_cross": float(np.mean(acc[lbl]["fence"])),
            "avg_diag_dist"  : float(np.mean(acc[lbl]["diag"])),
            "all_remaining"  : acc[lbl]["rem"],
        })
    df = pd.DataFrame(records).sort_values("avg_remaining", ascending=False)
    df["rank"] = range(1, len(df) + 1)
    return df
# ═════════════════════════════════════════════════════════════════════════════
# h-SENSITIVITY SWEEP  (NEW v3)  (NEW v3)
# ═════════════════════════════════════════════════════════════════════════════
def h_sensitivity_sweep(h_values=None, n_sims=3000, master_seed=None):
    n_sims = 300
    """
    Solo walk (no crowd penalty) at each h value with x=0.
    Returns dict  h → avg_raw_toll.
    This shows the empirical raw-toll curve and identifies the sweet-spot.
    """
    if h_values is None:
        h_values = list(range(H_MIN, H_MAX + 1))
    rng = np.random.default_rng(master_seed)
    results = {}
    print(f"\n  h-Sensitivity Sweep ({len(h_values)} h-values × {n_sims} sims) ",
          end="", flush=True)
    for h in h_values:
        tolls = []
        for _ in range(n_sims):
            # Fresh world per simulation; no time-based reseeding.
            sim_rng = np.random.default_rng(rng.integers(0, 2**63 - 1))
            ef, nf = build_base_fences(sim_rng)
            _, _, raw_toll, _, _, _ = walk_once(ef, nf, h, sim_rng)
            tolls.append(raw_toll)
        results[h] = float(np.mean(tolls))
        if h % 5 == 0:
            print(".", end="", flush=True)
    print(" done.\n")
    y = results
    print('Krishna  ')
    z = []
    for x in y:
        z.append((y[x], x))
    z.sort()
    for x in z:
        print(x)
    return results
# ═════════════════════════════════════════════════════════════════════════════
# FIELD GENERATION
# ═════════════════════════════════════════════════════════════════════════════
def sample_one_cluster(key, cdef, n, rng):
    if cdef["uniform"]:
        xs = rng.integers(cdef["x_lo"], cdef["x_hi"] + 1, n)
        hs = rng.integers(cdef["h_lo"], cdef["h_hi"] + 1, n)
    else:
        xs = rng.normal(cdef["mu_x"], cdef["sig_x"], n).round().astype(int)
        hs = rng.normal(cdef["mu_h"], cdef["sig_h"], n).round().astype(int)
        xs = np.clip(xs, cdef["x_lo"], cdef["x_hi"])
        hs = np.clip(hs, cdef["h_lo"], cdef["h_hi"])
    return [
        {"x": int(x), "h": int(h),
         "label": f"{key}_{i}_x{x}_h{h}",
         "cluster": key}
        for i, (x, h) in enumerate(zip(xs, hs))
    ]
import numpy as np
import time
def build_field(field_size, rng_seed=None):
    """Build a realistic competitor field from six weighted strategy clusters."""
    rng = np.random.default_rng(rng_seed)
    keys   = list(CLUSTERS.keys())
    weights = np.array([CLUSTERS[k]["weight"] for k in keys])
    weights /= weights.sum()
    raw_counts = weights * field_size
    counts     = np.floor(raw_counts).astype(int)
    remainder  = field_size - counts.sum()
    fracs      = raw_counts - counts
    top_idxs   = np.argsort(fracs)[::-1][:remainder]
    counts[top_idxs] += 1
    subs = []
    cluster_members = {}
    for k, cnt in zip(keys, counts):
        members = sample_one_cluster(k, CLUSTERS[k], cnt, rng)
        cluster_members[k] = members
        subs.extend(members)
    return subs, cluster_members
# ═════════════════════════════════════════════════════════════════════════════
# TERMINAL OUTPUT
# ═════════════════════════════════════════════════════════════════════════════
def print_header():
    print()
    print(C_CYAN + C_BOLD + "╔" + "═"*76 + "╗" + C_RESET)
    print(C_CYAN + C_BOLD + "║" + C_RESET +
          C_BOLD + "    🌾  KYBER PROBLEM CHALLENGE  v3 — Corrected Optimal Strategy    " +
          C_RESET + C_CYAN + C_BOLD + "  ║" + C_RESET)
    print(C_CYAN + C_BOLD + "╚" + "═"*76 + "╝" + C_RESET)
def print_strategy_analysis(sweep_results=None):
    # print()
    # print(C_BOLD + C_CYAN + "  " + "━"*72 + C_RESET)
    # print(C_BOLD + "  🧠  CORRECTED OPTIMAL STRATEGY  (v3)" + C_RESET)
    # print(C_BOLD + C_CYAN + "  " + "━"*72 + C_RESET)
    # print()
    # print(f"  {C_BOLD}Recommended submission:  x = 0,  h ≈ 18  (h ∈ [{OPTIMAL_H_LO_V3},{OPTIMAL_H_HI_V3}]){C_RESET}")
    # print()
    # print(f"  {C_BOLD}WHY NOT x = 300?{C_RESET}")
    # print(f"  • +300 starting apples sounds great, but those 600 thorns raise many")
    # print(f"    fences by +40, forcing ~1-3 costly crossings per run.")
    # print(f"  • More importantly: the unique thorn placement {C_YELLOW}does{C_RESET} reduce crowd")
    # print(f"    penalty slightly, but NOT enough to offset the raw-toll increase.")
    # print(f"  • Empirical result: x=0 consistently outranks x=300 after crowd penalty.")
    # print()
    # print(f"  {C_BOLD}WHY h ≈ 18?{C_RESET}")
    # print(f"  • Clean fence tolls are Uniform[{MIN_T},{MAX_T}], mean = {CLEAN_TOLL_MEAN:.1f}.")
    # print(f"  • With h=18 the farmer steers toward fences in [{MIN_T},18], avoiding")
    # print(f"    the expensive [{MAX_T-11},{MAX_T}] range.")
    # toll_budget = GRID * 2  # 100 steps
    # savings_per_step = CLEAN_TOLL_MEAN - (MIN_T + 18) / 2.0
    # print(f"  • Expected savings: ~{savings_per_step:.1f} apples/step × 100 steps"
    #       f" ≈ {savings_per_step*toll_budget/100:.0f}–{savings_per_step*toll_budget/100*1.4:.0f} apples over x=300's bonus.")
    # print()
    # print(f"  {C_BOLD}THE SWEET-SPOT [{OPTIMAL_H_LO_V3},{OPTIMAL_H_HI_V3}]:{C_RESET}")
    # print(f"  • h < {OPTIMAL_H_LO_V3}: too selective — only ~{(18-MIN_T)/(MAX_T-MIN_T+1)*100:.0f}% of fences accepted,")
    # print(f"    all low-h farmers funnel into identical cheap paths → crowd overhead spikes.")
    # print(f"  • h > {OPTIMAL_H_HI_V3}: stops steering meaningfully; raw toll climbs toward mean.")
    # print(f"  • h=18 rejects fences 19-30, providing 60%+ random choice at")
    # print(f"    typical intersections → good path diversity + cheap toll steering.")
    # print()
    if sweep_results:
        h_arr = np.array(sorted(sweep_results.keys()))
        t_arr = np.array([sweep_results[h] for h in h_arr])
        best_h = h_arr[np.argmin(t_arr)]
        print(f"  {C_BOLD}Sweep confirms:{C_RESET}  min raw toll at h={best_h}  "
              f"(toll={sweep_results[best_h]:.1f})  vs h=30 "
              f"(toll={sweep_results.get(30, '?'):.1f})")
        saving = sweep_results.get(30, 0) - sweep_results[best_h]
        print(f"  Raw-toll saving of h={best_h} over h=30: {C_GREEN}{saving:.1f} apples{C_RESET}")
    print()
    print(C_BOLD + C_CYAN + "  " + "━"*72 + C_RESET)
def print_field_info(cluster_members, your_sub):
    print()
    print(C_BOLD + "  ── Field Composition ──────────────────────────────────────────────" + C_RESET)
    for key, members in cluster_members.items():
        if not members:
            continue
        cdef = CLUSTERS[key]
        col  = cdef["term_color"]
        xs   = np.array([m["x"] for m in members])
        hs   = np.array([m["h"] for m in members])
        print(f"  {col}{key} {cdef['name']:<26}{C_RESET} : "
              f"{len(members):>3} farmers  "
              f"x∈[{xs.min()},{xs.max()}] μ={xs.mean():.0f}  "
              f"h∈[{hs.min()},{hs.max()}] μ={hs.mean():.0f}"
              f"  {C_DIM}({cdef['desc']}){C_RESET}")
    print(f"  {C_YELLOW}★ YOU{C_RESET}                              : "
          f"x={your_sub['x']}, h={your_sub['h']}  "
          f"(start apples = {2500 + your_sub['x']})")
def print_results(df, your_label, top_k=30):
    print()
    print(C_BOLD + C_CYAN + "═"*126 + C_RESET)
    print(C_BOLD + "  COMPETITION RESULTS  (with crowd penalty)" + C_RESET)
    print(C_BOLD + C_CYAN + "═"*126 + C_RESET)
    hdr = (f"  {'Rank':>4}  {'x':>4}  {'h':>3}  {'Cluster':<22}  "
           f"{'avg_rem':>10}  {'std':>7}  {'P10':>8}  {'P90':>8}  {'win%':>6}  "
           f"{'raw_toll':>9}  {'crowd%':>7}  {'thorn_x':>7}")
    print(C_DIM + hdr + C_RESET)
    print("  " + "─"*122)
    for _, row in df.head(top_k).iterrows():
        is_you = row["label"] == your_label
        cl     = row["cluster"]
        cl_col = CLUSTERS.get(cl, {}).get("term_color", C_YELLOW)
        cl_nm  = cl if cl in CLUSTERS else "YOU"
        avg_r = row["avg_remaining"]
        color = C_GREEN if avg_r > 0 else C_RED
        line = (f"  {int(row['rank']):>4}  "
                f"{int(row['x']):>4}  {int(row['h']):>3}  "
                f"{cl_col}{cl_nm:<22}{C_RESET}  "
                f"{color}{avg_r:>10.2f}{C_RESET}  "
                f"{row['std_remaining']:>7.2f}  "
                f"{row['p10_remaining']:>8.2f}  "
                f"{row['p90_remaining']:>8.2f}  "
                f"{row['win_rate']*100:>5.1f}%  "
                f"{row['avg_raw_toll']:>9.2f}  "
                f"{row['avg_crowd_pct']:>6.1f}%  "
                f"{row['avg_thorn_cross']:>7.2f}")
        if is_you:
            print(C_YELLOW + C_BOLD + line + "  ◄ YOU" + C_RESET)
        else:
            print(line)
    # show YOU if not in top_k
    your_rows = df[df["label"] == your_label]
    if len(your_rows) and your_label not in df.head(top_k)["label"].values:
        row   = your_rows.iloc[0]
        avg_r = row["avg_remaining"]
        color = C_GREEN if avg_r > 0 else C_RED
        print("  ...")
        print(C_YELLOW + C_BOLD +
              f"  {int(row['rank']):>4}  {int(row['x']):>4}  {int(row['h']):>3}  "
              f"{'★ YOU':<22}  {avg_r:>10.2f}  "
              f"{row['std_remaining']:>7.2f}  "
              f"{row['p10_remaining']:>8.2f}  "
              f"{row['p90_remaining']:>8.2f}  "
              f"{row['win_rate']*100:>5.1f}%  "
              f"{row['avg_raw_toll']:>9.2f}  "
              f"{row['avg_crowd_pct']:>6.1f}%  "
              f"{row['avg_thorn_cross']:>7.2f}  ◄ YOU" + C_RESET)
    print()
    best = df.iloc[0]
    tgt  = df[df["label"] == your_label].iloc[0]
    print(C_BOLD + "═"*126 + C_RESET)
    # Cluster-level summary
    print()
    print(C_BOLD + "  ── Cluster Average Performance ────────────────────────────────────" + C_RESET)
    print(C_DIM + f"  {'Cluster':<26}  {'avg_rem':>10}  {'P50':>8}  "
          f"{'win%':>6}  {'raw_toll':>9}  {'crowd%':>7}  n" + C_RESET)
    for key in list(CLUSTERS.keys()) + ["YOU"]:
        if key == "YOU":
            cdf = df[df["label"] == your_label]
            col = C_YELLOW
            nm  = "★ YOU"
        else:
            cdf = df[df["cluster"] == key]
            col = CLUSTERS[key]["term_color"]
            nm  = CLUSTERS[key]["name"]
        if len(cdf) == 0:
            continue
        print(f"  {col}{nm:<26}{C_RESET}  "
              f"avg_rem={cdf['avg_remaining'].mean():>8.2f}  "
              f"P50={cdf['p50_remaining'].mean():>8.2f}  "
              f"win%={cdf['win_rate'].mean()*100:>5.1f}%  "
              f"raw={cdf['avg_raw_toll'].mean():>9.2f}  "
              f"crowd={cdf['avg_crowd_pct'].mean():>5.1f}%  "
              f"n={len(cdf)}")
    print()
    print(f"  🥇 Winner : x={best['x']}, h={best['h']}  →  "
          f"avg_rem = {C_GREEN}{best['avg_remaining']:.2f}{C_RESET}")
    print(f"  🧑 You    : x={tgt['x']}, h={tgt['h']}   →  "
          f"avg_rem = {C_YELLOW}{tgt['avg_remaining']:.2f}{C_RESET}  "
          f"| rank {C_BOLD}{int(tgt['rank'])}{C_RESET} / {len(df)}")
    print(f"  📊 Gap to 1st : {best['avg_remaining'] - tgt['avg_remaining']:.2f} apples")
    print(f"  📈 Win rate   : {tgt['win_rate']*100:.1f}%")
    print(f"  📉 Raw toll   : {tgt['avg_raw_toll']:.2f}  |  crowd overhead: {tgt['avg_crowd_pct']:.1f}%")
    print(f"  📐 IQR        : {tgt['iqr_remaining']:.2f}  |  skewness: {tgt['skew_remaining']:.3f}")
    print(C_BOLD + "═"*126 + C_RESET)
    print()
    print("  " + C_DIM +
          "Clusters: E=Expert  Z=Zero-Thorn  G=Greedy  M=Middle  C=Code-Follower  R=Random  ★=YOU"
          + C_RESET)
    print()
# ═════════════════════════════════════════════════════════════════════════════
# MATPLOTLIB CHARTS  (9 panels)
# ═════════════════════════════════════════════════════════════════════════════
def make_charts(df, your_label, cluster_members, sweep_results=None):
    your_row = df[df["label"] == your_label].iloc[0]
    def row_color(row):
        if row["label"] == your_label:
            return "#FFD700"
        return CLUSTERS.get(row["cluster"], {}).get("mpl_color", "#888888")
    colors = [row_color(r) for _, r in df.iterrows()]
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(24, 14), facecolor="#0d1117")
    fig.suptitle("🌾  Kyber Challenge v3 — Corrected Optimal Strategy",
                 fontsize=14, fontweight="bold", color="#e6edf3",
                 fontfamily="monospace", y=0.99)
    gs = GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.36,
                  left=0.05, right=0.97, top=0.94, bottom=0.06)
    ax_sweep  = fig.add_subplot(gs[0, 0])   # 1. h-sensitivity sweep (NEW)
    ax_xh     = fig.add_subplot(gs[0, 1])   # 2. strategy scatter
    ax_clust  = fig.add_subplot(gs[0, 2])   # 3. cluster bar
    ax_rank   = fig.add_subplot(gs[1, 0])   # 4. top-30 bar
    ax_dist   = fig.add_subplot(gs[1, 1])   # 5. score distribution
    ax_crowd  = fig.add_subplot(gs[1, 2])   # 6. crowd overhead vs score (NEW)
    ax_box    = fig.add_subplot(gs[2, 0])   # 7. box plot top-5 + YOU
    ax_thorn  = fig.add_subplot(gs[2, 1])   # 8. thorn crossings vs score
    ax_win    = fig.add_subplot(gs[2, 2])   # 9. win rate by cluster
    def style_ax(ax, title, xlabel="", ylabel=""):
        ax.set_facecolor("#161b22")
        ax.set_title(title, color="#e6edf3", fontsize=9, fontweight="bold", pad=6)
        ax.tick_params(colors="#8b949e", labelsize=7)
        ax.spines[:].set_color("#30363d")
        if xlabel: ax.set_xlabel(xlabel, color="#8b949e", fontsize=7)
        if ylabel: ax.set_ylabel(ylabel, color="#8b949e", fontsize=7)
    # ── 1. h-Sensitivity Sweep ───────────────────────────────────────────────
    if sweep_results:
        h_arr = np.array(sorted(sweep_results.keys()))
        t_arr = np.array([sweep_results[h] for h in h_arr])
        ax_sweep.plot(h_arr, t_arr, color="#64B5F6", lw=2)
        # sweet-spot band
        ax_sweep.axvspan(OPTIMAL_H_LO_V3, OPTIMAL_H_HI_V3,
                         color="#4CAF50", alpha=0.18, label=f"Sweet spot [{OPTIMAL_H_LO_V3},{OPTIMAL_H_HI_V3}]")
        ax_sweep.axvline(your_row["h"], color="#FFD700", lw=1.5, ls="--", label=f"h={int(your_row['h'])}")
        best_h = h_arr[np.argmin(t_arr)]
        ax_sweep.axvline(best_h, color="#4CAF50", lw=1.5, ls=":", label=f"min h={best_h}")
        ax_sweep.legend(fontsize=6, facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")
        style_ax(ax_sweep, "h-Sensitivity Sweep (solo, x=0)", "h threshold", "avg raw toll")
    else:
        ax_sweep.text(0.5, 0.5, "No sweep data\n(--skip-sweep used)",
                      ha="center", va="center", color="#8b949e", fontsize=9,
                      transform=ax_sweep.transAxes)
        style_ax(ax_sweep, "h-Sensitivity Sweep")
    # ── 2. Strategy scatter: x vs h ─────────────────────────────────────────
    from matplotlib.patches import Rectangle
    for key, cdef in CLUSTERS.items():
        sub_df = df[df["cluster"] == key]
        if len(sub_df):
            ax_xh.scatter(sub_df["x"], sub_df["h"],
                          c=cdef["mpl_color"], alpha=0.55, s=22,
                          label=f"{key}: {cdef['name'][:12]}")
    ax_xh.scatter([your_row["x"]], [your_row["h"]], c="#FFD700",
                  s=280, zorder=5, marker="*", label="★ YOU")
    # v3 optimal zone: low x, h in [14,22]
    ax_xh.add_patch(Rectangle((0, OPTIMAL_H_LO_V3), 20, OPTIMAL_H_HI_V3 - OPTIMAL_H_LO_V3,
                                linewidth=1.5, edgecolor="#FFD700",
                                facecolor="#FFD70022", zorder=3,
                                label=f"Optimal zone"))
    ax_xh.legend(fontsize=5.5, facecolor="#161b22", edgecolor="#30363d",
                 labelcolor="#e6edf3", loc="upper right")
    style_ax(ax_xh, "Strategy Space (x vs h)", "x  (extra apples / thorns)", "h  (threshold)")
    # ── 3. Cluster mean performance bar ─────────────────────────────────────
    cluster_means, cluster_labels, cluster_cols = [], [], []
    for key, cdef in CLUSTERS.items():
        sub_df = df[df["cluster"] == key]
        if len(sub_df):
            cluster_means.append(sub_df["avg_remaining"].mean())
            cluster_labels.append(f"{key}")
            cluster_cols.append(cdef["mpl_color"])
    cluster_means.append(your_row["avg_remaining"])
    cluster_labels.append("★")
    cluster_cols.append("#FFD700")
    brs = ax_clust.bar(range(len(cluster_means)), cluster_means,
                       color=cluster_cols, edgecolor="none", width=0.6)
    ax_clust.set_xticks(range(len(cluster_means)))
    ax_clust.set_xticklabels(cluster_labels, fontsize=7)
    ax_clust.axhline(0, color="#ff4444", lw=1, ls="--", alpha=0.7)
    for bar, val in zip(brs, cluster_means):
        ax_clust.text(bar.get_x() + bar.get_width()/2,
                      bar.get_height() + (5 if val >= 0 else -20),
                      f"{val:.0f}", ha="center", va="bottom",
                      color="#e6edf3", fontsize=6.5, fontweight="bold")
    style_ax(ax_clust, "Cluster Avg Remaining (E=Expert,C=CodeFollower)",
             ylabel="apples remaining")
    # ── 4. Top-30 ranking bar ────────────────────────────────────────────────
    top30 = df.head(30)
    c30   = [row_color(r) for _, r in top30.iterrows()]
    ax_rank.barh(range(len(top30)), top30["avg_remaining"],
                 color=c30, edgecolor="none", height=0.75)
    ax_rank.axvline(0, color="#ff4444", lw=1, ls="--", alpha=0.6)
    ax_rank.set_yticks(range(len(top30)))
    ax_rank.set_yticklabels([f"#{int(r['rank'])}" for _, r in top30.iterrows()],
                            fontsize=6, color="#8b949e")
    ax_rank.invert_yaxis()
    you_in30 = top30[top30["label"] == your_label]
    if len(you_in30):
        idx = list(top30["label"]).index(your_label)
        ax_rank.barh(idx, you_in30["avg_remaining"].values[0],
                     color="#FFD700", edgecolor="#ff8c00", lw=1.5, height=0.75)
        ax_rank.text(you_in30["avg_remaining"].values[0] + 2, idx,
                     "◄ YOU", va="center", color="#FFD700", fontsize=6, fontweight="bold")
    style_ax(ax_rank, "Top-30 Avg Remaining Apples", "apples remaining", "rank")
    # ── 5. Distribution by cluster ───────────────────────────────────────────
    bins = np.linspace(df["avg_remaining"].min(), df["avg_remaining"].max(), 40)
    for key, cdef in CLUSTERS.items():
        sub_df = df[df["cluster"] == key]
        if len(sub_df):
            ax_dist.hist(sub_df["avg_remaining"], bins=bins,
                         color=cdef["mpl_color"], alpha=0.65, label=key, edgecolor="none")
    ax_dist.axvline(your_row["avg_remaining"], color="#FFD700", lw=2, ls="--", label="YOU")
    ax_dist.axvline(0, color="#ff4444", lw=1, ls=":", alpha=0.7)
    ax_dist.legend(fontsize=6, facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")
    style_ax(ax_dist, "Avg-Remaining Distribution by Cluster", "avg remaining apples", "count")
    # ── 6. Crowd overhead vs score scatter (NEW v3) ──────────────────────────
    for key, cdef in CLUSTERS.items():
        sub_df = df[df["cluster"] == key]
        if len(sub_df):
            ax_crowd.scatter(sub_df["avg_crowd_pct"], sub_df["avg_remaining"],
                             c=cdef["mpl_color"], alpha=0.5, s=18, label=key)
    ax_crowd.scatter([your_row["avg_crowd_pct"]], [your_row["avg_remaining"]],
                     c="#FFD700", s=200, zorder=5, marker="*", label="YOU")
    ax_crowd.axhline(0, color="#ff4444", lw=1, ls="--", alpha=0.6)
    ax_crowd.legend(fontsize=6, facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")
    style_ax(ax_crowd, "Crowd Overhead vs Score", "crowd overhead %", "avg remaining")
    # ── 7. Box plot: top-5 + YOU ─────────────────────────────────────────────
    top5 = df.head(5)
    you_in_t5 = your_label in list(top5["label"])
    if not you_in_t5:
        box_rows   = list(top5.iterrows()) + [(None, your_row)]
        box_labels = ([f"#{int(r['rank'])}\nx={int(r['x'])},h={int(r['h'])}"
                       for _, r in top5.iterrows()] + ["★YOU"])
        box_colors = ([CLUSTERS.get(r["cluster"], {}).get("mpl_color", "#888")
                       for _, r in top5.iterrows()] + ["#FFD700"])
    else:
        box_rows   = list(top5.iterrows())
        box_labels = [f"#{int(r['rank'])}\nx={int(r['x'])},h={int(r['h'])}"
                      + (" ★" if r["label"] == your_label else "")
                      for _, r in top5.iterrows()]
        box_colors = ["#FFD700" if r["label"] == your_label
                      else CLUSTERS.get(r["cluster"], {}).get("mpl_color", "#888")
                      for _, r in top5.iterrows()]
    all_box = [r["all_remaining"] for _, r in box_rows]
    bp = ax_box.boxplot(all_box, patch_artist=True, notch=False,
                        medianprops=dict(color="white", lw=2),
                        whiskerprops=dict(color="#8b949e"),
                        capprops=dict(color="#8b949e"),
                        flierprops=dict(marker=".", color="#8b949e", alpha=0.4, ms=3))
    for patch, col in zip(bp["boxes"], box_colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)
    ax_box.set_xticks(range(1, len(all_box) + 1))
    ax_box.set_xticklabels(box_labels, fontsize=5.5)
    ax_box.axhline(0, color="#ff4444", lw=1, ls="--", alpha=0.7)
    style_ax(ax_box, "Score Distribution: Top-5 + YOU (P10/P50/P90)", ylabel="apples remaining")
    # ── 8. Thorn crossings vs score ──────────────────────────────────────────
    for key, cdef in CLUSTERS.items():
        sub_df = df[df["cluster"] == key]
        if len(sub_df):
            ax_thorn.scatter(sub_df["avg_thorn_cross"], sub_df["avg_remaining"],
                             c=cdef["mpl_color"], alpha=0.5, s=18, label=key)
    ax_thorn.scatter([your_row["avg_thorn_cross"]], [your_row["avg_remaining"]],
                     c="#FFD700", s=200, zorder=5, marker="*", label="YOU")
    ax_thorn.axhline(0, color="#ff4444", lw=1, ls="--", alpha=0.6)
    ax_thorn.legend(fontsize=6, facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")
    style_ax(ax_thorn, "Thorn Crossings vs Score", "avg thorn crossings", "avg remaining")
    # ── 9. Win-rate by cluster ───────────────────────────────────────────────a
    wr_vals, wr_labels, wr_cols = [], [], []
    for k in CLUSTERS:
        sub_df = df[df["cluster"] == k]
        if len(sub_df):
            wr_vals.append(sub_df["win_rate"].mean() * 100)
            wr_labels.append(k)
            wr_cols.append(CLUSTERS[k]["mpl_color"])
    wr_vals.append(your_row["win_rate"] * 100)
    wr_labels.append("★ YOU")
    wr_cols.append("#FFD700")
    brs2 = ax_win.bar(range(len(wr_vals)), wr_vals, color=wr_cols, edgecolor="none", width=0.6)
    ax_win.set_xticks(range(len(wr_vals)))
    ax_win.set_xticklabels(wr_labels, fontsize=7)
    ax_win.set_ylim(0, 115)
    ax_win.axhline(50, color="#888", lw=1, ls=":", alpha=0.5)
    for bar, val in zip(brs2, wr_vals):
        ax_win.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{val:.0f}%", ha="center", va="bottom",
                    color="#e6edf3", fontsize=7, fontweight="bold")
    style_ax(ax_win, "Win Rate by Cluster", ylabel="% runs with ≥ 0 apples")
    plt.savefig("kyber_results_v3.png", dpi=150, bbox_inches="tight",
                facecolor="#0d1117")
    print(f"  📊 Chart saved to {C_CYAN}kyber_results_v3.png{C_RESET}")
    plt.show()
# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description="Kyber Challenge v3 — corrected optimal strategy + h-sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Optimal strategy (v3):  --x 0 --h 18  (h ∈ [{OPTIMAL_H_LO_V3},{OPTIMAL_H_HI_V3}])"
    )
    p.add_argument("--x",           type=int, default=None,
                   help=f"Your x submission ({X_MIN}–{X_MAX})")
    p.add_argument("--h",           type=int, default=None,
                   help=f"Your h submission ({H_MIN}–{H_MAX})")
    p.add_argument("--field-size",  type=int, default=100, dest="field_size",
                   help="Total rival farmers to simulate  [100]")
    p.add_argument("--n-sims",      type=int, default=NSIMS, dest="n_sims",
                   help="Competition iterations per seed  [50]")
    p.add_argument("--seeds",       type=int, default=5,
                   help="Independent seeds to average over  [5]")
    p.add_argument("--seed",        type=int, default=None,
                   help="Master RNG seed (random if omitted)")
    p.add_argument("--top-show",    type=int, default=30, dest="top_show",
                   help="Rows to print in result table  [30]")
    p.add_argument("--sweep-sims",  type=int, default=3000, dest="sweep_sims",
                   help="Simulations per h-value in sensitivity sweep  [300]")
    p.add_argument("--skip-sweep",  action="store_true", dest="skip_sweep",
                   help="Skip the h-sensitivity sweep (faster)")
    p.add_argument("--sweep-only",  action="store_true", dest="sweep_only",
                   help="Run only the h-sensitivity sweep and exit")
    p.add_argument("--no-chart",    action="store_true", dest="no_chart",
                   help="Skip matplotlib charts")
    p.add_argument("--no-analysis", action="store_true", dest="no_analysis",
                   help="Skip optimal-strategy analysis")
    return p.parse_args()
def main():
    args = parse_args()
    print_header()
    # ── h-Sensitivity Sweep ───────────────────────────────────────────────────
    sweep_results = None
    if not args.skip_sweep:
        sweep_results = h_sensitivity_sweep(n_sims=args.sweep_sims, master_seed=None if args.seed is None else args.seed + 2)
        if args.sweep_only:
            h_arr = np.array(sorted(sweep_results.keys()))
            t_arr = np.array([sweep_results[h] for h in h_arr])
            best_h = h_arr[np.argmin(t_arr)]
            print(f"  Best h by raw toll : h = {C_GREEN}{best_h}{C_RESET}  "
                  f"(avg raw toll = {C_GREEN}{sweep_results[best_h]:.2f}{C_RESET})")
            print()
            print("  h   raw_toll   Δ_from_min")
            for h in h_arr:
                diff = sweep_results[h] - sweep_results[best_h]
                bar  = "█" * int(diff / 5)
                print(f"  {h:>3}  {sweep_results[h]:>8.2f}  +{diff:>6.2f}  {bar}")
            return
    if not args.no_analysis:
        print_strategy_analysis(sweep_results)
    # ── Get x and h ───────────────────────────────────────────────────────────
    if args.x is None:
        print()
        try:
            args.x = int(input(C_YELLOW + f"  Enter your x ({X_MIN}–{X_MAX}) : " + C_RESET))
        except (ValueError, EOFError):
            print("[ERROR] Invalid x", file=sys.stderr); sys.exit(1)
    if args.h is None:
        try:
            args.h = int(input(C_YELLOW + f"  Enter your h ({H_MIN}–{H_MAX})  : " + C_RESET))
        except (ValueError, EOFError):
            print("[ERROR] Invalid h", file=sys.stderr); sys.exit(1)
    x, h = args.x, args.h
    if not (X_MIN <= x <= X_MAX):
        print(f"[ERROR] x={x} out of range [{X_MIN},{X_MAX}]", file=sys.stderr); sys.exit(1)
    if not (H_MIN <= h <= H_MAX):
        print(f"[ERROR] h={h} out of range [{H_MIN},{H_MAX}]", file=sys.stderr); sys.exit(1)
    # Contextual warnings
    if h < OPTIMAL_H_LO_V3:
        print(f"\n  {C_RED}⚠  h={h} < {OPTIMAL_H_LO_V3}: very restrictive — crowd overhead likely to spike{C_RESET}")
        print(f"  {C_RED}   as low-h farmers all funnel toward the same cheap fences.{C_RESET}")
    elif h > OPTIMAL_H_HI_V3 and h < 30:
        print(f"\n  {C_YELLOW}⚠  h={h}: above sweet-spot — some toll steering lost vs h=18.{C_RESET}")
    elif h >= 30:
        print(f"\n  {C_YELLOW}⚠  h={h} ≥ 30: accepts all clean fences; no toll steering advantage.{C_RESET}")
        print(f"  {C_YELLOW}   This was the v2 recommendation — v3 shows h≈18 is better.{C_RESET}")
    if x > 50:
        print(f"\n  {C_YELLOW}⚠  x={x}: high x adds apples but {2*x} thorns raise your raw toll.{C_RESET}")
        print(f"  {C_YELLOW}   Empirically, x≈0 outperforms x=300 after crowd penalty.{C_RESET}")
    seed = args.seed
    # ── Build field ───────────────────────────────────────────────────────────
    field, cluster_members = build_field(args.field_size, None if seed is None else seed + 1)
    your_label = f"★YOU_x{x}_h{h}"
    your_sub   = {"x": x, "h": h, "label": your_label, "cluster": "YOU"}
    submissions = [your_sub] + field
    print_field_info(cluster_members, your_sub)
    print()
    print(C_BOLD + "  ── Simulation Settings ─────────────────────────────────────────────" + C_RESET)
    print(f"  Field size  : {len(submissions)} total submissions")
    print(f"  Iterations  : {args.n_sims} sims × {args.seeds} seeds = {args.n_sims*args.seeds:,} total")
    print(f"  Master seed : {seed if seed is not None else 'random entropy'}")
    # ── Simulate ──────────────────────────────────────────────────────────────
    df = simulate_competition(
        submissions=submissions,
        n_sims=args.n_sims,
        n_seeds=args.seeds,
        master_seed=seed,
    )
    print_results(df, your_label, top_k=args.top_show)
    if not args.no_chart:
        print("  Generating charts …")
        make_charts(df, your_label, cluster_members, sweep_results)
    else:
        print("  (Charts skipped — remove --no-chart to enable)")
if __name__ == "__main__":
    main()
