#!/usr/bin/env python3
"""
kyber_optimizer.py
══════════════════════════════════════════════════════════════════════════════
Finds the optimal (x, h) for the Kyber Problem Challenge using a three-phase
adaptive search strategy:

  Phase 1 — Coarse grid scan      (fast, broad)
  Phase 2 — Fine-grained zoom     (around top-K candidates)
  Phase 3 — Deep estimation       (high-N confirmation on finalists)

Scoring metric: Risk-Adjusted Score = mean − 0.5 × std
  (penalises high-variance strategies; you want consistent wins, not lucky spikes)

Also reports: mean, P10, P50, P90, win-rate, IQR for every candidate.

WHAT THE REAL DATA TOLD US (from your contest screenshots)
──────────────────────────────────────────────────────────
  • h=17 gave both 1380 and 1597 in different runs — HUGE variance.
  • h=7,8 gave 1547,1585 — surprisingly competitive.
  • h=18 gave 1362 one run, 1472 another — noise dominates with few sims.
  → A single simulation is meaningless. Only averaging 200+ runs is reliable.
  → The optimizer therefore runs many sims and tracks the full distribution.

Usage:
    python kyber_optimizer.py                         # full 3-phase search
    python kyber_optimizer.py --quick                 # faster (fewer sims)
    python kyber_optimizer.py --x-vals 0,5,10 --h-range 14,22   # custom
    python kyber_optimizer.py --field-size 200        # larger competition field
    python kyber_optimizer.py --phase1-only           # coarse scan only
    python kyber_optimizer.py --workers 8             # override CPU count
"""

import argparse
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ═════════════════════════════════════════════════════════════════════════════
# GAME CONSTANTS  (must match the contest spec exactly)
# ═════════════════════════════════════════════════════════════════════════════
GRID   = 50
MIN_T  = 5
MAX_T  = 30
THORN  = 40
_SQRT2 = math.sqrt(2.0)

# ═════════════════════════════════════════════════════════════════════════════
# TERMINAL COLOURS
# ═════════════════════════════════════════════════════════════════════════════
R = "\033[0m"; B = "\033[1m"; D = "\033[2m"
G = "\033[92m"; Y = "\033[93m"; C = "\033[96m"
RE = "\033[91m"; M = "\033[95m"; BL = "\033[94m"
OR = "\033[38;5;208m"; TE = "\033[38;5;44m"

# ═════════════════════════════════════════════════════════════════════════════
# SIMULATION CORE  (identical physics to v3)
# ═════════════════════════════════════════════════════════════════════════════

def build_fences(rng):
    ef = rng.integers(MIN_T, MAX_T + 1, (GRID, GRID + 1)).astype(np.float64)
    nf = rng.integers(MIN_T, MAX_T + 1, (GRID + 1, GRID)).astype(np.float64)
    return ef, nf


def add_thorns(ef, nf, x_val, rng):
    ef, nf = ef.copy(), nf.copy()
    n = 2 * x_val
    if n > 0:
        idx  = rng.integers(0, ef.size + nf.size, size=n)
        em   = idx < ef.size
        np.add.at(ef.ravel(), idx[em],          THORN)
        np.add.at(nf.ravel(), idx[~em]-ef.size, THORN)
    return ef, nf


def walk(ef, nf, h, rng):
    cx, cy = 0, 0
    eu = np.zeros((GRID, GRID + 1), dtype=np.int32)
    nu = np.zeros((GRID + 1, GRID), dtype=np.int32)
    raw = 0.0
    while cx < GRID or cy < GRID:
        ce, cn = cx < GRID, cy < GRID
        if ce and cn:
            eo = ef[cx, cy] <= h
            no = nf[cx, cy] <= h
            if eo and not no:   mv = "e"
            elif no and not eo: mv = "n"
            else:               mv = "e" if rng.random() < 0.5 else "n"
        elif ce:  mv = "e"
        else:     mv = "n"
        if mv == "e": raw += ef[cx, cy]; eu[cx, cy] += 1; cx += 1
        else:          raw += nf[cx, cy]; nu[cx, cy] += 1; cy += 1
    return eu, nu, raw


def pen_toll(ef, nf, eu, nu, fe, fn):
    return float(((ef * (1 + fe)**3) * eu).sum() +
                 ((nf * (1 + fn)**3) * nu).sum())


def run_competition_once(submissions, base_rng, sub_rngs):
    """One shared-world iteration.  Returns list of (pen_toll, raw_toll)."""
    n  = len(submissions)
    ef0, nf0 = build_fences(base_rng)

    walks = []
    for i, sub in enumerate(submissions):
        ef_i, nf_i = add_thorns(ef0, nf0, sub["x"], sub_rngs[i])
        eu, nu, rt = walk(ef_i, nf_i, sub["h"], sub_rngs[i])
        walks.append((ef_i, nf_i, eu, nu, rt))

    te = sum(w[2] for w in walks).astype(np.float64)
    tn = sum(w[3] for w in walks).astype(np.float64)
    fe, fn = te / n, tn / n

    out = []
    for ef_i, nf_i, eu, nu, rt in walks:
        pt = pen_toll(ef_i, nf_i, eu, nu, fe, fn)
        out.append((pt, rt))
    return out


def estimate_candidate(x, h, field_subs, n_sims, master_seed):
    """
    Run `n_sims` competitions with the candidate (x,h) inserted into
    the given field.  Returns full array of remaining-apple samples.
    """
    rng     = np.random.default_rng(master_seed)
    subs    = [{"x": x, "h": h}] + field_subs
    n       = len(subs)
    remainders = []

    for _ in range(n_sims):
        base_rng = np.random.default_rng(rng.integers(0, 10_000_000))
        sub_rngs = [np.random.default_rng(rng.integers(0, 10_000_000))
                    for _ in range(n)]
        res = run_competition_once(subs, base_rng, sub_rngs)
        pt, _  = res[0]
        remainders.append(2500 + x - pt)

    return np.array(remainders)


# ─────────────────────────────────────────────────────────────────────────────
# TOP-LEVEL WORKER — must be importable by child processes (module-level def)
# ─────────────────────────────────────────────────────────────────────────────

def _worker(args_tuple):
    """
    Standalone function called in a child process.
    Returns (x, h, remainders_array).
    """
    x, h, field_subs, n_sims, seed = args_tuple
    arr = estimate_candidate(x, h, field_subs, n_sims, seed)
    return x, h, arr


# ═════════════════════════════════════════════════════════════════════════════
# FIELD BUILDER  (six clusters, same as v3)
# ═════════════════════════════════════════════════════════════════════════════
CLUSTERS = {
    "E": dict(w=0.10, mu_x=10,  sx=8,  xl=0,   xh=20,  mu_h=18, sh=3,  hl=14, hh=22, uni=False),
    "Z": dict(w=0.25, mu_x=5,   sx=8,  xl=0,   xh=30,  mu_h=28, sh=12, hl=5,  hh=55, uni=False),
    "G": dict(w=0.25, mu_x=280, sx=20, xl=200, xh=300, mu_h=52, sh=6,  hl=35, hh=60, uni=False),
    "M": dict(w=0.20, mu_x=150, sx=50, xl=50,  xh=250, mu_h=38, sh=8,  hl=20, hh=55, uni=False),
    "C": dict(w=0.15, mu_x=270, sx=20, xl=220, xh=300, mu_h=32, sh=4,  hl=25, hh=40, uni=False),
    "R": dict(w=0.05, mu_x=150, sx=0,  xl=0,   xh=300, mu_h=32, sh=0,  hl=5,  hh=60, uni=True),
}


def build_field(field_size, seed):
    rng  = np.random.default_rng(seed)
    keys = list(CLUSTERS.keys())
    ws   = np.array([CLUSTERS[k]["w"] for k in keys]); ws /= ws.sum()
    raw  = ws * field_size
    cnts = np.floor(raw).astype(int)
    for i in np.argsort(raw - cnts)[::-1][:field_size - cnts.sum()]:
        cnts[i] += 1

    subs = []
    for k, cnt in zip(keys, cnts):
        c = CLUSTERS[k]
        if c["uni"]:
            xs = rng.integers(c["xl"], c["xh"]+1, cnt)
            hs = rng.integers(c["hl"], c["hh"]+1, cnt)
        else:
            xs = np.clip(rng.normal(c["mu_x"], c["sx"], cnt).round().astype(int), c["xl"], c["xh"])
            hs = np.clip(rng.normal(c["mu_h"], c["sh"], cnt).round().astype(int), c["hl"], c["hh"])
        subs.extend([{"x": int(x), "h": int(h)} for x, h in zip(xs, hs)])
    return subs


# ═════════════════════════════════════════════════════════════════════════════
# STATS HELPER
# ═════════════════════════════════════════════════════════════════════════════

def stats(arr):
    """Compute all key statistics for a sample array."""
    return {
        "mean"    : float(arr.mean()),
        "std"     : float(arr.std()),
        "p10"     : float(np.percentile(arr, 10)),
        "p50"     : float(np.percentile(arr, 50)),
        "p90"     : float(np.percentile(arr, 90)),
        "iqr"     : float(np.percentile(arr, 75) - np.percentile(arr, 25)),
        "win_rate": float((arr >= 0).mean()),
        "ras"     : 0.0,   # filled in later with chosen alpha
    }


def risk_adjusted(mean, std, alpha):
    return mean - alpha * std


# ═════════════════════════════════════════════════════════════════════════════
# PARALLEL SCAN HELPER
# ═════════════════════════════════════════════════════════════════════════════

def _run_parallel(points, field, sims, alpha, seed, n_workers, label):
    """
    Evaluate every (x, h) in `points` in parallel.
    Returns list of (x, h, stats_dict).
    Prints a live progress bar to stdout.
    """
    rng = np.random.default_rng(seed)
    # Build work items — each gets a unique seed so workers are independent
    work = [(x, h, field, sims, int(rng.integers(0, 10_000_000)))
            for x, h in points]

    total   = len(work)
    results = {}   # key: (x,h) → arr
    done    = 0

    print(f"    Workers: {n_workers}  |  Points: {total}  |  "
          f"Sims/pt: {sims}  |  Total sims: {total * sims:,}")
    print(f"    Running ", end="", flush=True)
    tick = max(1, total // 20)

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_worker, item): item for item in work}
        for fut in as_completed(futures):
            x, h, arr = fut.result()
            results[(x, h)] = arr
            done += 1
            if done % tick == 0:
                print("█", end="", flush=True)

    print(" done.")

    out = []
    for x, h in points:
        arr = results[(x, h)]
        s   = stats(arr)
        s["ras"] = risk_adjusted(s["mean"], s["std"], alpha)
        out.append((x, h, s))
    return out


# ═════════════════════════════════════════════════════════════════════════════
# THREE-PHASE OPTIMIZER  (now all parallel)
# ═════════════════════════════════════════════════════════════════════════════

def phase1_coarse(field, sims, alpha, seed, n_workers):
    x_grid = [0, 5, 10, 20, 50, 100, 150, 200, 250, 300]
    h_grid = [5, 7, 9, 11, 13, 15, 16, 17, 18, 19, 20, 21, 22, 25, 28, 30, 35, 40, 50, 60]
    points = list(product(x_grid, h_grid))
    return _run_parallel(points, field, sims, alpha, seed, n_workers, "Phase 1")


def phase2_zoom(top_candidates, field, sims, alpha, seed, n_workers):
    seen = set(); points = []
    for x, h, _ in top_candidates:
        for dx in range(-5, 6):
            for dh in range(-4, 5):
                nx, nh = x + dx, h + dh
                if 0 <= nx <= 300 and 5 <= nh <= 60 and (nx, nh) not in seen:
                    seen.add((nx, nh)); points.append((nx, nh))
    return _run_parallel(points, field, sims, alpha, seed, n_workers, "Phase 2")


def phase3_deep(top_candidates, field, sims, alpha, seed, n_workers):
    points = [(x, h) for x, h, _ in top_candidates]
    results = _run_parallel(points, field, sims, alpha, seed, n_workers, "Phase 3")

    # Attach 95% CI
    out = []
    for (x, h, s) in results:
        s["ci95"] = 1.96 * s["std"] / math.sqrt(sims)
        out.append((x, h, s))
    return out


# ═════════════════════════════════════════════════════════════════════════════
# PRETTY PRINTING
# ═════════════════════════════════════════════════════════════════════════════

def print_header():
    print()
    print(C+B + "╔" + "═"*72 + "╗" + R)
    print(C+B + "║" + R + B + "   🔬  KYBER OPTIMIZER  — Parallel Adaptive 3-Phase Grid Search   " + R + C+B + " ║" + R)
    print(C+B + "╚" + "═"*72 + "╝" + R)


def print_phase(n, title):
    print()
    print(B+C + f"  ── Phase {n}: {title} " + "─"*(60-len(title)) + R)


def print_table(rows, alpha, top_n=20, label="Results"):
    rows_s = sorted(rows, key=lambda r: -r[2]["ras"])
    print()
    print(B + f"  {label}  (sorted by Risk-Adjusted Score, α={alpha})" + R)
    hdr = (f"  {'Rank':>4}  {'x':>4}  {'h':>3}  "
           f"{'mean':>9}  {'std':>7}  {'P10':>8}  {'P50':>8}  {'P90':>8}  "
           f"{'IQR':>7}  {'win%':>6}  {'RAS':>9}")
    print(D + hdr + R)
    print("  " + "─"*106)
    for rank, (x, h, s) in enumerate(rows_s[:top_n], 1):
        mean_c = G if s["mean"] > 0 else RE
        ras_c  = G if s["ras"]  > 0 else RE
        optimal = (0 <= x <= 20 and 14 <= h <= 22)
        flag = f"  {TE}◄ sweet-spot{R}" if optimal else ""
        print(f"  {rank:>4}  {x:>4}  {h:>3}  "
              f"{mean_c}{s['mean']:>9.1f}{R}  "
              f"{s['std']:>7.1f}  "
              f"{s['p10']:>8.1f}  "
              f"{s['p50']:>8.1f}  "
              f"{s['p90']:>8.1f}  "
              f"{s['iqr']:>7.1f}  "
              f"{s['win_rate']*100:>5.1f}%  "
              f"{ras_c}{s['ras']:>9.1f}{R}"
              f"{flag}")
    return rows_s


# ═════════════════════════════════════════════════════════════════════════════
# CHARTS
# ═════════════════════════════════════════════════════════════════════════════

def make_charts(all_results, top_final, your_x=None, your_h=None):
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(22, 14), facecolor="#0d1117")
    fig.suptitle("🔬  Kyber Optimizer — (x, h) Search Landscape",
                 fontsize=13, fontweight="bold", color="#e6edf3",
                 fontfamily="monospace", y=0.99)
    gs = GridSpec(2, 3, figure=fig, hspace=0.44, wspace=0.34,
                  left=0.05, right=0.97, top=0.94, bottom=0.07)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    def sax(ax, title, xl="", yl=""):
        ax.set_facecolor("#161b22"); ax.spines[:].set_color("#30363d")
        ax.tick_params(colors="#8b949e", labelsize=7)
        ax.set_title(title, color="#e6edf3", fontsize=9, fontweight="bold", pad=5)
        if xl: ax.set_xlabel(xl, color="#8b949e", fontsize=7)
        if yl: ax.set_ylabel(yl, color="#8b949e", fontsize=7)

    df = pd.DataFrame([
        {"x": x, "h": h, "mean": s["mean"], "std": s["std"],
         "ras": s["ras"], "p10": s["p10"], "p50": s["p50"], "p90": s["p90"],
         "win": s["win_rate"]}
        for x, h, s in all_results
    ])

    best_per_h = df.loc[df.groupby("h")["mean"].idxmax()].sort_values("h")
    ax1.plot(best_per_h["h"], best_per_h["mean"], color="#64B5F6", lw=2, marker="o", ms=3)
    ax1.fill_between(best_per_h["h"],
                     best_per_h["mean"] - best_per_h["std"],
                     best_per_h["mean"] + best_per_h["std"],
                     color="#64B5F6", alpha=0.15)
    ax1.axvspan(14, 22, color="#4CAF50", alpha=0.12, label="sweet-spot [14,22]")
    ax1.axhline(0, color="#ff4444", lw=1, ls="--", alpha=0.5)
    if your_h is not None:
        ax1.axvline(your_h, color="#FFD700", lw=1.5, ls=":", label=f"your h={your_h}")
    ax1.legend(fontsize=6, facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")
    sax(ax1, "Mean Score by h (best x per h)", "h threshold", "avg remaining apples")

    sc = ax2.scatter(df["x"], df["h"], c=df["ras"], cmap="RdYlGn",
                     s=22, alpha=0.8, vmin=df["ras"].quantile(0.05),
                     vmax=df["ras"].quantile(0.95))
    plt.colorbar(sc, ax=ax2, label="Risk-Adj Score", fraction=0.04)
    from matplotlib.patches import Rectangle
    ax2.add_patch(Rectangle((0, 14), 20, 8, lw=1.5, edgecolor="#FFD700",
                             facecolor="#FFD70020", zorder=3, label="Optimal zone"))
    if your_x is not None and your_h is not None:
        ax2.scatter([your_x], [your_h], c="#FFD700", s=200, zorder=5, marker="*", label="★ YOU")
    top3 = sorted(all_results, key=lambda r: -r[2]["ras"])[:3]
    for i, (tx, th, ts) in enumerate(top3):
        ax2.scatter([tx], [th], c="#fff", s=60, zorder=6, marker="D",
                    edgecolors="#FFD700", lw=1)
        ax2.text(tx+3, th+0.5, f"#{i+1}", color="#FFD700", fontsize=6)
    ax2.legend(fontsize=5.5, facecolor="#161b22", edgecolor="#30363d",
               labelcolor="#e6edf3", loc="upper right")
    sax(ax2, "Search Landscape (x vs h, colour=RAS)", "x", "h")

    top20 = sorted(all_results, key=lambda r: -r[2]["ras"])[:20]
    labels = [f"x={x},h={h}" for x, h, _ in top20]
    means  = [s["mean"] for _, _, s in top20]
    stds   = [s["std"]  for _, _, s in top20]
    bar_c  = ["#FFD700" if (0<=x<=20 and 14<=h<=22) else "#64B5F6"
               for x, h, _ in top20]
    ax3.barh(range(len(top20)), means, xerr=stds, color=bar_c,
             edgecolor="none", height=0.7, ecolor="#555", capsize=2)
    ax3.set_yticks(range(len(top20)))
    ax3.set_yticklabels(labels, fontsize=5.5, color="#8b949e")
    ax3.invert_yaxis()
    ax3.axvline(0, color="#ff4444", lw=1, ls="--", alpha=0.5)
    if your_x is not None and your_h is not None:
        your_match = [i for i, (x,h,_) in enumerate(top20) if x==your_x and h==your_h]
        if your_match:
            ax3.barh(your_match[0], means[your_match[0]], color="#FFD700",
                     edgecolor="#ff8c00", lw=1.5, height=0.7)
    sax(ax3, "Top-20 by RAS  (mean ± std; gold=sweet-spot)", "mean remaining", "")

    best_ras_per_h = df.loc[df.groupby("h")["ras"].idxmax()].sort_values("h")
    ax4.plot(best_ras_per_h["h"], best_ras_per_h["ras"], color="#CE93D8", lw=2, marker="o", ms=3)
    ax4.axvspan(14, 22, color="#4CAF50", alpha=0.12, label="sweet-spot [14,22]")
    ax4.axhline(0, color="#ff4444", lw=1, ls="--", alpha=0.5)
    if your_h is not None:
        ax4.axvline(your_h, color="#FFD700", lw=1.5, ls=":", label=f"your h={your_h}")
    ax4.legend(fontsize=6, facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")
    sax(ax4, "Risk-Adjusted Score by h (best x per h)", "h threshold", "RAS")

    best_win_per_h = df.loc[df.groupby("h")["win"].idxmax()].sort_values("h")
    ax5.bar(best_win_per_h["h"], best_win_per_h["win"] * 100,
            color="#81C784", edgecolor="none", width=0.8)
    ax5.axvspan(14, 22, color="#FFD700", alpha=0.10, label="sweet-spot [14,22]")
    ax5.axhline(50, color="#888", lw=1, ls=":", alpha=0.5)
    if your_h is not None:
        ax5.axvline(your_h, color="#FFD700", lw=1.5, ls=":", label=f"your h={your_h}")
    ax5.legend(fontsize=6, facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")
    sax(ax5, "Win-Rate % by h (best x per h)", "h threshold", "win rate %")

    if top_final:
        tf_labels = [f"x={x}\nh={h}" for x, h, _ in top_final]
        p10s = [s["p10"] for _, _, s in top_final]
        p50s = [s["p50"] for _, _, s in top_final]
        p90s = [s["p90"] for _, _, s in top_final]
        xs   = range(len(top_final))
        ax6.fill_between(xs, p10s, p90s, alpha=0.25, color="#64B5F6", label="P10–P90")
        ax6.plot(xs, p50s, color="#64B5F6", lw=2, marker="o", ms=4, label="P50")
        ax6.plot(xs, p10s, color="#FF8A65", lw=1, ls="--", label="P10")
        ax6.plot(xs, p90s, color="#81C784", lw=1, ls="--", label="P90")
        ax6.set_xticks(range(len(top_final)))
        ax6.set_xticklabels(tf_labels, fontsize=6)
        ax6.axhline(0, color="#ff4444", lw=1, ls="--", alpha=0.5)
        ax6.legend(fontsize=6, facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")
        sax(ax6, "P10 / P50 / P90  (Top Finalists — Deep Phase)", "candidate", "apples remaining")
    else:
        ax6.text(0.5, 0.5, "Run all 3 phases\nfor P10/P50/P90", ha="center", va="center",
                 color="#8b949e", fontsize=9, transform=ax6.transAxes)
        sax(ax6, "P10 / P50 / P90 (Phase 3)")

    plt.savefig("kyber_optimizer.png", dpi=150, bbox_inches="tight", facecolor="#0d1117")
    print(f"\n  📊 Chart saved to {C}kyber_optimizer.png{R}")
    plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Kyber Optimizer — parallel 3-phase adaptive grid search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--field-size",    type=int,   default=100, dest="field_size")
    p.add_argument("--field-seed",    type=int,   default=42,  dest="field_seed")
    p.add_argument("--risk-aversion", type=float, default=0.5, dest="alpha",
                   help="α in RAS = mean − α×std  [0.5]")
    p.add_argument("--p1-sims",  type=int, default=80,  dest="p1_sims")
    p.add_argument("--p2-sims",  type=int, default=200, dest="p2_sims")
    p.add_argument("--p3-sims",  type=int, default=500, dest="p3_sims")
    p.add_argument("--top-k",    type=int, default=15,  dest="top_k")
    p.add_argument("--workers",  type=int, default=None, dest="workers",
                   help="Number of parallel worker processes [auto = CPU count]")
    p.add_argument("--quick",        action="store_true",
                   help="Fast mode: p1=30, p2=80, p3=200 sims")
    p.add_argument("--phase1-only",  action="store_true", dest="p1only")
    p.add_argument("--no-chart",     action="store_true", dest="no_chart")
    p.add_argument("--your-x",       type=int, default=None, dest="your_x")
    p.add_argument("--your-h",       type=int, default=None, dest="your_h")
    return p.parse_args()


def main():
    args = parse_args()

    if args.quick:
        args.p1_sims = 30
        args.p2_sims = 80
        args.p3_sims = 200

    # Determine worker count
    n_workers = args.workers or os.cpu_count() or 4

    print_header()
    print()
    print(B + "  STRATEGY" + R)
    print(f"  • Phase 1: coarse grid ({args.p1_sims} sims/pt) — identify promising regions")
    print(f"  • Phase 2: fine zoom   ({args.p2_sims} sims/pt) — refine around top-{args.top_k}")
    print(f"  • Phase 3: deep confirm({args.p3_sims} sims/pt) — tight CI on finalists")
    print(f"  • Risk-aversion α = {args.alpha}  (RAS = mean − {args.alpha}×std)")
    print(f"  • Parallel workers: {G}{n_workers}{R}  (of {os.cpu_count()} logical CPUs)")
    print()
    print(B + "  KEY INSIGHT FROM YOUR CONTEST DATA" + R)
    print("  h=17 gave 1380 AND 1597 in different runs — single-sim results are noise.")
    print("  This optimizer runs 500+ sims per finalist to estimate the true distribution.")

    t0 = time.time()
    seed = int(time.time() * 1000) % (2**32 - 1)

    print(f"\n  Building rival field (n={args.field_size}, seed={args.field_seed}) …")
    field = build_field(args.field_size, args.field_seed)

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    print_phase(1, "Coarse Grid Scan  [parallel]")
    p1_results = phase1_coarse(field, args.p1_sims, args.alpha, seed, n_workers)
    p1_top = print_table(p1_results, args.alpha, top_n=20, label="Phase 1 Top-20")

    if args.p1only:
        print(f"\n  ⏱  Elapsed: {time.time()-t0:.1f}s")
        if not args.no_chart:
            make_charts(p1_results, [], args.your_x, args.your_h)
        return

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    print_phase(2, "Fine Zoom  [parallel]")
    candidates_2 = p1_top[:args.top_k]
    p2_results = phase2_zoom(candidates_2, field, args.p2_sims, args.alpha, seed+1, n_workers)
    all_results = p1_results + p2_results
    p2_top = print_table(p2_results, args.alpha, top_n=20, label="Phase 2 Top-20")

    # ── Phase 3 ───────────────────────────────────────────────────────────────
    print_phase(3, "Deep Confirmation  [parallel]")
    combined = sorted(all_results, key=lambda r: -r[2]["ras"])
    seen_pts = set(); deduped = []
    for row in combined:
        key_pt = (row[0], row[1])
        if key_pt not in seen_pts:
            seen_pts.add(key_pt); deduped.append(row)
    candidates_3 = deduped[:args.top_k]
    p3_results = phase3_deep(candidates_3, field, args.p3_sims, args.alpha, seed+2, n_workers)
    p3_top = print_table(p3_results, args.alpha, top_n=20, label="Phase 3 Final Rankings")

    # ── Final recommendation ──────────────────────────────────────────────────
    best = p3_top[0]
    bx, bh, bs = best

    print()
    print(B+C + "  " + "═"*72 + R)
    print(B + "  🏆  FINAL RECOMMENDATION" + R)
    print(B+C + "  " + "═"*72 + R)
    print()
    print(f"  Submit:  {B+G}x = {bx},  h = {bh}{R}")
    print()
    print(f"  {'Metric':<20}  {'Value':>12}")
    print("  " + "─"*36)
    print(f"  {'Mean remaining':<20}  {bs['mean']:>12.2f}")
    print(f"  {'Std deviation':<20}  {bs['std']:>12.2f}")
    print(f"  {'P10 (worst 10%)':<20}  {bs['p10']:>12.2f}")
    print(f"  {'P50 (median)':<20}  {bs['p50']:>12.2f}")
    print(f"  {'P90 (best 10%)':<20}  {bs['p90']:>12.2f}")
    print(f"  {'IQR':<20}  {bs['iqr']:>12.2f}")
    print(f"  {'Win rate':<20}  {bs['win_rate']*100:>11.1f}%")
    print(f"  {'Risk-Adj Score':<20}  {bs['ras']:>12.2f}")
    ci = bs.get("ci95", 1.96 * bs["std"] / math.sqrt(args.p3_sims))
    print(f"  {'95% CI on mean':<20}  ±{ci:>11.2f}")
    print()
    print(f"  {D}Runner-up:{R}")
    if len(p3_top) > 1:
        rx, rh, rs = p3_top[1]
        print(f"    x={rx}, h={rh}  |  mean={rs['mean']:.1f}  "
              f"RAS={rs['ras']:.1f}  P10={rs['p10']:.1f}")
    print()
    print(f"  ⏱  Total elapsed: {time.time()-t0:.1f}s")
    print(B+C + "  " + "═"*72 + R)
    print()

    if not args.no_chart:
        print("  Generating charts …")
        make_charts(all_results, p3_results, args.your_x, args.your_h)


# ─────────────────────────────────────────────────────────────────────────────
# IMPORTANT: guard required for ProcessPoolExecutor on Windows / frozen apps
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()