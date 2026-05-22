#!/usr/bin/env python3
"""
+======================================================================+
|   Kyber Problem -- Multi-CPU Optimizer + Full Plot Suite              |
|   No Crowd Penalty                                                   |
+======================================================================+

Usage:
    python kyber_multicore.py                  # auto-detect CPUs
    python kyber_multicore.py --cpus 8         # force 8 workers
    python kyber_multicore.py --seeds 5        # more seeds = smoother results
    python kyber_multicore.py --walkers 100    # walkers per grid
    python kyber_multicore.py --no-plots       # skip plotting, just CSV

Outputs:
    kyber_scores.csv        all (x,h) pairs ranked by mean_remaining
    kyber_optimal.txt       human-readable top-20 summary
    kyber_plots.png         full 8-panel analytics figure
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np

# --- Game constants -----------------------------------------------------------
GRID   = 50
MIN_T  = 5
MAX_T  = 30
THORN  = 40

X_MIN, X_MAX = 0,  320
H_MIN, H_MAX = 5,   65

# --- Theme --------------------------------------------------------------------
BG      = "#0d0d1a"
PANEL   = "#13132b"
PANEL2  = "#1a1a35"
ACCENT  = "#e94560"
BLUE    = "#4a9eff"
TEAL    = "#00b4d8"
GOLD    = "#ffd166"
GREEN   = "#06d6a0"
PURPLE  = "#c77dff"
ORANGE  = "#ff9a00"
TEXT    = "#dde1f0"
SUBTEXT = "#7b7fa8"
BORDER  = "#2a2a50"


# --- Core simulation (module-level so it's picklable) -------------------------

def _build_fences(x_val: int, rng: np.random.Generator):
    ef = rng.integers(MIN_T, MAX_T + 1, (GRID, GRID + 1)).astype(np.float64)
    nf = rng.integers(MIN_T, MAX_T + 1, (GRID + 1, GRID)).astype(np.float64)
    n  = 2 * x_val
    if n > 0:
        idxs = rng.integers(0, ef.size + nf.size, size=n)
        np.add.at(ef.ravel(), idxs[idxs < ef.size],            THORN)
        np.add.at(nf.ravel(), idxs[idxs >= ef.size] - ef.size, THORN)
    return ef, nf


def _walk_batch(ef, nf, h: int, K: int, rng: np.random.Generator):
    """Simulate K independent walkers on the same fence grid."""
    cx    = np.zeros(K, np.int32)
    cy    = np.zeros(K, np.int32)
    tolls = np.zeros(K, np.float64)
    for _ in range(2 * GRID):
        alive = (cx < GRID) | (cy < GRID)
        if not alive.any():
            break
        can_e = cx < GRID
        can_n = cy < GRID
        te    = ef[np.minimum(cx, GRID - 1), np.minimum(cy, GRID)]
        tn    = nf[np.minimum(cx, GRID),     np.minimum(cy, GRID - 1)]
        e_ok  = (te <= h) & can_e
        n_ok  = (tn <= h) & can_n
        go_e  = (can_e & ~can_n) | (e_ok & ~n_ok)
        rand  = (e_ok == n_ok) & can_e & can_n
        go_e  = (go_e | (rand & (rng.random(K) < 0.5))) & alive
        go_n  = alive & ~go_e
        tolls += np.where(go_e, ef[np.minimum(cx, GRID-1), np.minimum(cy, GRID)],     0)
        tolls += np.where(go_n, nf[np.minimum(cx, GRID),   np.minimum(cy, GRID - 1)], 0)
        cx += go_e.astype(np.int32)
        cy += go_n.astype(np.int32)
    return tolls


def _evaluate_x_chunk(args):
    """
    Worker function: evaluate every h-value for a single x-value, across all seeds.
    Returns list of dicts, one per (x, h).
    """
    x_val, h_list, seeds, K = args
    start = 2500 + x_val
    # accum[h] = {means, stds, wins}
    accum = {h: {"means": [], "stds": [], "wins": []} for h in h_list}

    for seed in seeds:
        rng = np.random.default_rng(seed * 999_983 + x_val * 61)
        ef, nf = _build_fences(x_val, rng)          # one grid per (seed, x)
        for h in h_list:
            tolls = _walk_batch(ef, nf, h, K, rng)
            rems  = start - tolls
            accum[h]["means"].append(float(rems.mean()))
            accum[h]["stds"].append(float(rems.std()))
            accum[h]["wins"].append(float((rems >= 0).mean()))

    rows = []
    for h in h_list:
        d = accum[h]
        rows.append({
            "x"              : x_val,
            "h"              : h,
            "start_apples"   : start,
            "thorns"         : 2 * x_val,
            "mean_remaining" : round(float(np.mean(d["means"])), 2),
            "std_remaining"  : round(float(np.mean(d["stds"])),  2),
            "win_rate"       : round(float(np.mean(d["wins"])),  4),
        })
    return rows


# --- Sweep --------------------------------------------------------------------

def run_sweep(n_cpus: int, seeds: int, K: int) -> list:
    x_list = list(range(X_MIN, X_MAX + 1))
    h_list = list(range(H_MIN, H_MAX + 1))
    # seed_list = list(range(seeds))
    rng_master = np.random.default_rng(int(time.time()))
    seed_list = rng_master.integers(0, 10_000_000, size=seeds).tolist()

    total_grids = len(x_list) * seeds

    print(f"\n{'='*64}")
    print(f"  Kyber Optimizer  --  No Crowd Penalty")
    print(f"  x: {X_MIN}-{X_MAX} ({len(x_list)} values)   "
          f"h: {H_MIN}-{H_MAX} ({len(h_list)} values)")
    print(f"  Seeds: {seeds}   Walkers/grid: {K}   CPUs: {n_cpus}")
    print(f"  Total fence grids: {total_grids:,}   "
          f"Total walker-steps: ~{total_grids * K * 2 * GRID:,}")
    print(f"{'='*64}\n")

    tasks = [(x, h_list, seed_list, K) for x in x_list]
    all_rows = []
    t0 = time.time()

    with Pool(processes=n_cpus) as pool:
        for i, chunk in enumerate(pool.imap_unordered(_evaluate_x_chunk, tasks,
                                                       chunksize=max(1, len(x_list) // (n_cpus * 4)))):
            all_rows.extend(chunk)
            if (i + 1) % 30 == 0 or i == len(x_list) - 1:
                done = i + 1
                pct  = done / len(x_list) * 100
                ela  = time.time() - t0
                eta  = ela / done * (len(x_list) - done) if done else 0
                print(f"  {pct:5.1f}%  x-chunks done: {done}/{len(x_list)}"
                      f"   elapsed: {ela:.0f}s   ETA: {eta:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"\n  [OK] Sweep complete in {elapsed:.1f}s  "
          f"({len(all_rows):,} (x,h) pairs)\n")

    all_rows.sort(key=lambda r: (-r["mean_remaining"], -r["win_rate"]))
    for rank, row in enumerate(all_rows, 1):
        row["rank"] = rank
    return all_rows


# --- Save CSV -----------------------------------------------------------------

def save_csv(rows: list, path: str):
    fields = ["rank", "x", "h", "start_apples", "thorns",
              "mean_remaining", "std_remaining", "win_rate"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"  CSV  ->  {path}  ({len(rows):,} rows)")


# --- Save summary text --------------------------------------------------------

def save_summary(rows: list, path: str):
    best  = rows[0]
    top20 = rows[:20]
    bot5  = rows[-5:]
    L = []
    L += ["=" * 64,
          "  KYBER PROBLEM -- OPTIMAL (x, h)  [No Crowd Penalty]",
          "=" * 64, ""]
    L += [f"  x in [0, 300]   h in [5, 60]",
          f"  Evaluated {len(rows):,} (x,h) pairs", ""]
    L += ["-" * 64, "  *  BEST CONFIGURATION", "-" * 64,
          f"    x              = {best['x']}",
          f"    h              = {best['h']}",
          f"    Starting apples= {best['start_apples']:,}",
          f"    Thorns placed  = {best['thorns']:,}",
          f"    Mean remaining = {best['mean_remaining']:,.2f} apples",
          f"    Std deviation  = +/-{best['std_remaining']:,.2f}",
          f"    Win rate (>=0)  = {best['win_rate']:.1%}", ""]
    L += ["-" * 64, "  TOP 20 CONFIGURATIONS", "-" * 64,
          f"  {'Rank':>4}  {'x':>5}  {'h':>5}  {'Start':>7}  "
          f"{'Thorns':>7}  {'Mean Rem':>10}  {'Std':>8}  {'WinRate':>8}"]
    for r in top20:
        L.append(f"  {r['rank']:>4}  {r['x']:>5}  {r['h']:>5}  "
                 f"{r['start_apples']:>7}  {r['thorns']:>7}  "
                 f"{r['mean_remaining']:>10.2f}  {r['std_remaining']:>8.2f}  "
                 f"{r['win_rate']:>8.1%}")
    L += ["", "-" * 64, "  WORST 5", "-" * 64,
          f"  {'Rank':>6}  {'x':>5}  {'h':>5}  {'Mean Rem':>10}  {'WinRate':>8}"]
    for r in bot5:
        L.append(f"  {r['rank']:>6}  {r['x']:>5}  {r['h']:>5}  "
                 f"{r['mean_remaining']:>10.2f}  {r['win_rate']:>8.1%}")
    L += ["", "=" * 64]
    txt = "\n".join(L)
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(f"  TXT  ->  {path}")
    print("\n" + txt)


# --- Plotting -----------------------------------------------------------------

def plot_results(rows: list, out_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    import matplotlib.ticker as mticker

    print("\n  Building plots ...")

    # -- Reconstruct 2-D grids ----------------------------------------------
    x_vals = sorted({r["x"] for r in rows})
    h_vals = sorted({r["h"] for r in rows})
    xi     = {x: i for i, x in enumerate(x_vals)}
    hi     = {h: i for i, h in enumerate(h_vals)}
    NX, NH = len(x_vals), len(h_vals)

    mean_grid = np.full((NX, NH), np.nan)
    std_grid  = np.full((NX, NH), np.nan)
    win_grid  = np.full((NX, NH), np.nan)

    for r in rows:
        i, j = xi[r["x"]], hi[r["h"]]
        mean_grid[i, j] = r["mean_remaining"]
        std_grid[i, j]  = r["std_remaining"]
        win_grid[i, j]  = r["win_rate"]

    best     = rows[0]
    top20    = rows[:20]
    bottom50 = rows[-50:]

    # -- Custom colormaps ---------------------------------------------------
    cmap_mean = LinearSegmentedColormap.from_list(
        "mean", ["#0d0d1a", "#1e3a5f", "#4a9eff", "#ffd166", "#06d6a0"], N=512)
    cmap_std  = LinearSegmentedColormap.from_list(
        "std",  ["#06d6a0", "#ffd166", "#e94560", "#9d0036"], N=512)
    cmap_win  = LinearSegmentedColormap.from_list(
        "win",  ["#e94560", "#ffd166", "#06d6a0"], N=512)

    # -- Figure layout ------------------------------------------------------
    fig = plt.figure(figsize=(24, 18))
    fig.patch.set_facecolor(BG)

    gs = gridspec.GridSpec(
        3, 3,
        figure=fig,
        left=0.06, right=0.97,
        top=0.91,  bottom=0.06,
        hspace=0.48, wspace=0.38,
    )

    axes = [fig.add_subplot(gs[r, c]) for r in range(3) for c in range(3)]
    for ax in axes:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=SUBTEXT, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)

    def _cbar(fig, ax, im, label):
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="3.5%", pad=0.06)
        cax.set_facecolor(PANEL)
        cb  = fig.colorbar(im, cax=cax)
        cb.set_label(label, color=SUBTEXT, fontsize=8)
        cb.ax.tick_params(colors=SUBTEXT, labelsize=7)

    def _star(ax, x, h, color=ACCENT, size=260):
        ax.scatter([x], [h], marker="*", s=size, color=color,
                   edgecolors="white", linewidths=0.8, zorder=10)

    # ---------------------------------------------------------------------
    # Panel 0: Mean remaining heatmap (x vs h)
    # ---------------------------------------------------------------------
    ax = axes[0]
    im = ax.imshow(
        mean_grid.T, origin="lower", aspect="auto",
        extent=[X_MIN, X_MAX, H_MIN, H_MAX],
        cmap=cmap_mean, interpolation="nearest",
    )
    _cbar(fig, ax, im, "Mean Remaining Apples")
    _star(ax, best["x"], best["h"])
    ax.set_xlabel("x  (extra apples / 1/2 thorns)", color=SUBTEXT, fontsize=9)
    ax.set_ylabel("h  (threshold)",               color=SUBTEXT, fontsize=9)
    ax.set_title("Mean Remaining Apples  (x vs h)",
                 color=TEXT, fontsize=10.5, fontweight="bold", pad=6)
    ax.text(best["x"] + 5, best["h"] + 1,
            f"* x={best['x']}, h={best['h']}\n  mu={best['mean_remaining']:.0f}",
            color=ACCENT, fontsize=7.5, fontweight="bold",
            bbox=dict(fc=PANEL2, ec=BORDER, boxstyle="round,pad=0.3"))

    # ---------------------------------------------------------------------
    # Panel 1: Std deviation heatmap
    # ---------------------------------------------------------------------
    ax = axes[1]
    im = ax.imshow(
        std_grid.T, origin="lower", aspect="auto",
        extent=[X_MIN, X_MAX, H_MIN, H_MAX],
        cmap=cmap_std, interpolation="nearest",
    )
    _cbar(fig, ax, im, "Std Dev of Remaining")
    _star(ax, best["x"], best["h"])
    ax.set_xlabel("x", color=SUBTEXT, fontsize=9)
    ax.set_ylabel("h", color=SUBTEXT, fontsize=9)
    ax.set_title("Risk (Std Dev)  --  lower = safer",
                 color=TEXT, fontsize=10.5, fontweight="bold", pad=6)

    # ---------------------------------------------------------------------
    # Panel 2: Win-rate heatmap
    # ---------------------------------------------------------------------
    ax = axes[2]
    im = ax.imshow(
        win_grid.T, origin="lower", aspect="auto",
        extent=[X_MIN, X_MAX, H_MIN, H_MAX],
        cmap=cmap_win, vmin=0, vmax=1, interpolation="nearest",
    )
    _cbar(fig, ax, im, "Win Rate (remaining >= 0)")
    _star(ax, best["x"], best["h"])
    ax.set_xlabel("x", color=SUBTEXT, fontsize=9)
    ax.set_ylabel("h", color=SUBTEXT, fontsize=9)
    ax.set_title("Win Rate  (fraction of sims >= 0 apples)",
                 color=TEXT, fontsize=10.5, fontweight="bold", pad=6)

    # ---------------------------------------------------------------------
    # Panel 3: Mean remaining vs x  (aggregated over h)
    # ---------------------------------------------------------------------
    ax = axes[3]
    best_mean_per_x  = [mean_grid[xi[x], :].max()  for x in x_vals]
    worst_mean_per_x = [mean_grid[xi[x], :].min()  for x in x_vals]
    avg_mean_per_x   = [np.nanmean(mean_grid[xi[x], :]) for x in x_vals]

    ax.fill_between(x_vals, worst_mean_per_x, best_mean_per_x,
                    color=BLUE, alpha=0.18, label="Range (min-max over h)", zorder=2)
    ax.plot(x_vals, avg_mean_per_x,  color=TEAL,   lw=1.6, label="Mean (avg over h)", zorder=3)
    ax.plot(x_vals, best_mean_per_x, color=GREEN,  lw=2.0, label="Best h per x",       zorder=4)
    ax.axvline(best["x"], color=ACCENT, lw=1.4, ls="--", label=f"Optimal x={best['x']}", zorder=5)
    ax.set_xlabel("x", color=SUBTEXT, fontsize=9)
    ax.set_ylabel("Mean Remaining Apples", color=SUBTEXT, fontsize=9)
    ax.set_title("Effect of x  (best/avg/worst h)",
                 color=TEXT, fontsize=10.5, fontweight="bold", pad=6)
    ax.legend(fontsize=7.5, facecolor=PANEL2, edgecolor=BORDER,
              labelcolor=TEXT, framealpha=0.9)
    ax.grid(True, color="#22224a", alpha=0.5)

    # ---------------------------------------------------------------------
    # Panel 4: Mean remaining vs h  (aggregated over x)
    # ---------------------------------------------------------------------
    ax = axes[4]
    best_mean_per_h  = [mean_grid[:, hi[h]].max()  for h in h_vals]
    worst_mean_per_h = [mean_grid[:, hi[h]].min()  for h in h_vals]
    avg_mean_per_h   = [np.nanmean(mean_grid[:, hi[h]]) for h in h_vals]

    ax.fill_between(h_vals, worst_mean_per_h, best_mean_per_h,
                    color=PURPLE, alpha=0.18, label="Range (min-max over x)", zorder=2)
    ax.plot(h_vals, avg_mean_per_h,  color=TEAL,   lw=1.6, label="Mean (avg over x)", zorder=3)
    ax.plot(h_vals, best_mean_per_h, color=GREEN,  lw=2.0, label="Best x per h",       zorder=4)
    ax.axvline(best["h"], color=ACCENT, lw=1.4, ls="--", label=f"Optimal h={best['h']}", zorder=5)
    ax.set_xlabel("h", color=SUBTEXT, fontsize=9)
    ax.set_ylabel("Mean Remaining Apples", color=SUBTEXT, fontsize=9)
    ax.set_title("Effect of h  (best/avg/worst x)",
                 color=TEXT, fontsize=10.5, fontweight="bold", pad=6)
    ax.legend(fontsize=7.5, facecolor=PANEL2, edgecolor=BORDER,
              labelcolor=TEXT, framealpha=0.9)
    ax.grid(True, color="#22224a", alpha=0.5)

    # ---------------------------------------------------------------------
    # Panel 5: Top-20 bar chart  (colored by h)
    # ---------------------------------------------------------------------
    ax = axes[5]
    t20_x     = [r["x"] for r in top20]
    t20_h     = [r["h"] for r in top20]
    t20_mean  = [r["mean_remaining"] for r in top20]
    t20_std   = [r["std_remaining"]  for r in top20]
    h_norm    = Normalize(H_MIN, H_MAX)
    h_cmap    = LinearSegmentedColormap.from_list(
        "hc", ["#4a9eff", "#ffd166", "#e94560"], N=256)
    bar_cols  = [h_cmap(h_norm(h)) for h in t20_h]

    bars = ax.barh(range(20), t20_mean, xerr=t20_std, color=bar_cols,
                   edgecolor=PANEL2, linewidth=0.5,
                   error_kw=dict(elinewidth=0.8, ecolor="#ffffff44"),
                   height=0.7, zorder=3)
    ax.set_yticks(range(20))
    ax.set_yticklabels([f"x={r['x']} h={r['h']}" for r in top20],
                       fontsize=7.5, color=TEXT)
    ax.invert_yaxis()
    ax.set_xlabel("Mean Remaining Apples  (+/-1sigma)", color=SUBTEXT, fontsize=9)
    ax.set_title("Top 20 Configurations",
                 color=TEXT, fontsize=10.5, fontweight="bold", pad=6)
    ax.grid(True, axis="x", color="#22224a", alpha=0.5)

    sm = plt.cm.ScalarMappable(cmap=h_cmap, norm=h_norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, orientation="vertical", pad=0.01, shrink=0.85)
    cb.set_label("h value", color=SUBTEXT, fontsize=7)
    cb.ax.tick_params(colors=SUBTEXT, labelsize=6)

    # ---------------------------------------------------------------------
    # Panel 6: Risk-reward scatter  (all (x,h) points)
    # ---------------------------------------------------------------------
    ax = axes[6]
    all_mean = [r["mean_remaining"] for r in rows]
    all_std  = [r["std_remaining"]  for r in rows]
    all_wr   = [r["win_rate"]       for r in rows]

    sc = ax.scatter(all_std, all_mean,
                    c=all_wr, cmap=cmap_win,
                    s=2.5, alpha=0.55, linewidths=0, zorder=3,
                    vmin=0, vmax=1)
    # Highlight top-20
    ax.scatter([r["std_remaining"]  for r in top20],
               [r["mean_remaining"] for r in top20],
               color=GOLD, s=35, zorder=6,
               edgecolors="white", linewidths=0.5, label="Top 20")
    # Star the best
    ax.scatter([best["std_remaining"]], [best["mean_remaining"]],
               marker="*", s=280, color=ACCENT, zorder=7,
               edgecolors="white", linewidths=0.7,
               label=f"Best x={best['x']} h={best['h']}")
    ax.set_xlabel("Std Dev (risk)",             color=SUBTEXT, fontsize=9)
    ax.set_ylabel("Mean Remaining (reward)",    color=SUBTEXT, fontsize=9)
    ax.set_title("Risk-Reward Scatter  (color = win rate)",
                 color=TEXT, fontsize=10.5, fontweight="bold", pad=6)
    ax.legend(fontsize=7.5, facecolor=PANEL2, edgecolor=BORDER,
              labelcolor=TEXT, framealpha=0.9)
    ax.grid(True, color="#22224a", alpha=0.5)
    cb = fig.colorbar(sc, ax=ax, pad=0.02, shrink=0.85)
    cb.set_label("Win Rate", color=SUBTEXT, fontsize=7)
    cb.ax.tick_params(colors=SUBTEXT, labelsize=6)

    # ---------------------------------------------------------------------
    # Panel 7: Distribution of mean_remaining  (histogram)
    # ---------------------------------------------------------------------
    ax = axes[7]
    arr = np.array(all_mean)
    bins = np.linspace(arr.min(), arr.max(), 60)
    ax.hist(arr, bins=bins, color=BLUE, alpha=0.75,
            edgecolor=PANEL, linewidth=0.4, zorder=3, label="All (x,h)")
    # Shade top-5%
    p95 = np.percentile(arr, 95)
    mask = arr >= p95
    ax.hist(arr[mask], bins=bins, color=GOLD, alpha=0.85,
            edgecolor=PANEL, linewidth=0.4, zorder=4, label=f"Top 5% (>={p95:.0f})")
    ax.axvline(best["mean_remaining"], color=ACCENT, lw=1.6, ls="--",
               label=f"Best = {best['mean_remaining']:.0f}")
    ax.axvline(arr.mean(), color=TEAL, lw=1.2, ls=":",
               label=f"Overall mu = {arr.mean():.0f}")
    ax.set_xlabel("Mean Remaining Apples", color=SUBTEXT, fontsize=9)
    ax.set_ylabel("# Configurations",     color=SUBTEXT, fontsize=9)
    ax.set_title("Score Distribution  (all 16,856 configs)",
                 color=TEXT, fontsize=10.5, fontweight="bold", pad=6)
    ax.legend(fontsize=7.5, facecolor=PANEL2, edgecolor=BORDER,
              labelcolor=TEXT, framealpha=0.9)
    ax.grid(True, axis="x", color="#22224a", alpha=0.5)

    # ---------------------------------------------------------------------
    # Panel 8: Optimal h for each x  (line plot)
    # ---------------------------------------------------------------------
    ax = axes[8]
    opt_h_per_x = [h_vals[int(np.argmax(mean_grid[xi[x], :]))] for x in x_vals]
    opt_v_per_x = [mean_grid[xi[x], :].max() for x in x_vals]

    # Color segments by optimal h value
    for i in range(len(x_vals) - 1):
        ax.plot(x_vals[i:i+2], opt_h_per_x[i:i+2],
                color=h_cmap(h_norm(opt_h_per_x[i])), lw=2.0, zorder=3)

    ax2 = ax.twinx()
    ax2.set_facecolor(PANEL)
    ax2.plot(x_vals, opt_v_per_x, color=GREEN, lw=1.2,
             alpha=0.6, ls="--", label="Best score at that x", zorder=2)
    ax2.tick_params(colors=SUBTEXT, labelsize=7)
    ax2.set_ylabel("Max mean remaining", color=GREEN, fontsize=8)
    ax2.yaxis.label.set_color(GREEN)

    ax.axvline(best["x"], color=ACCENT, lw=1.4, ls="--",
               label=f"Best x={best['x']}")
    ax.axhline(best["h"], color=GOLD,   lw=1.0, ls=":",
               label=f"Best h={best['h']}")
    ax.set_xlabel("x",                   color=SUBTEXT, fontsize=9)
    ax.set_ylabel("Optimal h  (per x)",  color=SUBTEXT, fontsize=9)
    ax.set_title("Optimal h for Each x  (colored by h, dashed = score)",
                 color=TEXT, fontsize=10.5, fontweight="bold", pad=6)
    ax.legend(fontsize=7.5, facecolor=PANEL2, edgecolor=BORDER,
              labelcolor=TEXT, framealpha=0.9, loc="upper left")
    ax.grid(True, color="#22224a", alpha=0.5)

    # -- Add colorbar for h on panel 8 --
    sm2 = plt.cm.ScalarMappable(cmap=h_cmap, norm=h_norm)
    sm2.set_array([])
    cb2 = fig.colorbar(sm2, ax=ax, orientation="vertical", pad=0.08, shrink=0.85)
    cb2.set_label("h value", color=SUBTEXT, fontsize=7)
    cb2.ax.tick_params(colors=SUBTEXT, labelsize=6)

    # -- Master title ------------------------------------------------------
    fig.text(0.5, 0.965,
             "[A]  Kyber Problem -- Full Analytics  (No Crowd Penalty)",
             ha="center", fontsize=19, fontweight="bold",
             color=ACCENT, fontfamily="monospace")
    fig.text(0.5, 0.943,
             f"301 x 56 parameter grid  .  "
             f"* Optimal: x={best['x']}, h={best['h']}  "
             f"->  mu={best['mean_remaining']:.1f} apples remaining",
             ha="center", fontsize=11, color=SUBTEXT)

    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close()
    print(f"  PNG  ->  {out_path}")


# --- CLI ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Kyber multi-CPU optimizer (no crowd penalty)")
    p.add_argument("--cpus",     type=int, default=None,
                   help="Number of worker processes (default: all CPUs)")
    p.add_argument("--seeds",    type=int, default=5,
                   help="Seeds averaged per (x,h) pair  [default: 5]")
    p.add_argument("--walkers",  type=int, default=50,
                   help="Walkers per fence grid  [default: 50]")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip matplotlib plotting")
    p.add_argument("--outdir",   type=str, default=".",
                   help="Output directory  [default: current dir]")
    return p.parse_args()


# --- Entry point --------------------------------------------------------------

if __name__ == "__main__":
    args    = parse_args()
    n_cpus  = args.cpus or cpu_count()
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = run_sweep(n_cpus=n_cpus, seeds=args.seeds, K=args.walkers)

    save_csv(rows,     str(out_dir / "kyber_scores.csv"))
    save_summary(rows, str(out_dir / "kyber_optimal.txt"))

    if not args.no_plots:
        try:
            plot_results(rows, str(out_dir / "kyber_plots.png"))
        except ImportError:
            print("\n  [!]  matplotlib not found -- skipping plots.")
            print("     Install with:  pip install matplotlib\n")

    print(f"\n  All outputs in:  {out_dir.resolve()}\n")