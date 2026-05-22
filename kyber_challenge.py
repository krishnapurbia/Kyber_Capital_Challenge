#!/usr/bin/env python3
"""
kyber_challenge.py
══════════════════════════════════════════════════════════════════════════════
Kyber Problem Challenge — Self-contained simulator
No CSV needed. Generates a competitive field from two known strategy clusters,
then benchmarks YOUR submission against them.

TWO-CLUSTER FIELD MODEL
───────────────────────
  Cluster A  (low-x / low-h)  : x ~ N(25, 10) clipped [0, 50]
                                  h ~ N(18,  3) clipped [5, 23]
  Cluster B  (high-x / high-h): x ~ N(295,  3) clipped [290, 300]
                                  h ~ N(42,  8) clipped [25, 60]

Usage:
    python kyber_challenge.py          # prompts for x and h
    python kyber_challenge.py --x 200 --h 20
    python kyber_challenge.py --x 200 --h 20 --field-size 80 --n-sims 50
"""

import argparse
import sys
import time
import math

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── Game constants ────────────────────────────────────────────────────────────
GRID  = 50
MIN_T = 5
MAX_T = 30
THORN = 40
NSIMS = 50

X_MIN, X_MAX = 0,   300
H_MIN, H_MAX = 5,    60

_SQRT2 = math.sqrt(2.0)

# Cluster definitions
CLUSTER_A = dict(mu_x=12,  sig_x=10, mu_h=15, sig_h=3,  x_lo=0,   x_hi=50,  h_lo=5,  h_hi=23)
CLUSTER_B = dict(mu_x=297, sig_x=3,  mu_h=40, sig_h=8,  x_lo=290, x_hi=300, h_lo=25, h_hi=60)

# Terminal colour codes
C_RESET  = "\033[0m"
C_BOLD   = "\033[1m"
C_GREEN  = "\033[92m"
C_YELLOW = "\033[93m"
C_CYAN   = "\033[96m"
C_RED    = "\033[91m"
C_MAGENTA= "\033[95m"
C_DIM    = "\033[2m"


# ── Core simulation ───────────────────────────────────────────────────────────

def build_base_fences(rng):
    ef = rng.integers(MIN_T, MAX_T + 1, (GRID, GRID + 1)).astype(np.float64)
    nf = rng.integers(MIN_T, MAX_T + 1, (GRID + 1, GRID)).astype(np.float64)
    return ef, nf


def add_thorns(ef_base, nf_base, x_val, rng):
    ef = ef_base.copy()
    nf = nf_base.copy()
    n  = 2 * x_val
    if n > 0:
        idxs = rng.integers(0, ef.size + nf.size, size=n)
        east_mask = idxs < ef.size
        np.add.at(ef.ravel(), idxs[east_mask],            THORN)
        np.add.at(nf.ravel(), idxs[~east_mask] - ef.size, THORN)
    return ef, nf


def walk_once(ef, nf, h, rng):
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
        (thorn_x if toll > MAX_T else fence_x)  # count below
        if toll > MAX_T: thorn_x += 1
        else:            fence_x  += 1
        diag_sum += abs(cx - cy) / _SQRT2

    avg_diag = diag_sum / (2 * GRID + 1)
    return eu, nu, raw_toll, thorn_x, fence_x, avg_diag


def penalised_toll(ef, nf, eu, nu, frac_e, frac_n):
    pen_e = ef * (1.0 + frac_e) ** 3
    pen_n = nf * (1.0 + frac_n) ** 3
    return float((pen_e * eu).sum() + (pen_n * nu).sum())


def run_one_iteration(submissions, base_rng, sub_rngs):
    n = len(submissions)
    ef_base, nf_base = build_base_fences(base_rng)

    sub_data = []
    for idx, sub in enumerate(submissions):
        rng = sub_rngs[idx]
        ef_i, nf_i = add_thorns(ef_base, nf_base, sub["x"], rng)
        eu_i, nu_i, _, thorn_i, fence_i, diag_i = walk_once(ef_i, nf_i, sub["h"], rng)
        sub_data.append((ef_i, nf_i, eu_i, nu_i, thorn_i, fence_i, diag_i))

    total_eu = sum(d[2] for d in sub_data).astype(np.float64)
    total_nu = sum(d[3] for d in sub_data).astype(np.float64)
    frac_e   = total_eu / n
    frac_n   = total_nu / n

    results = []
    for ef_i, nf_i, eu_i, nu_i, thorn_i, fence_i, diag_i in sub_data:
        results.append({
            "pen_toll"       : penalised_toll(ef_i, nf_i, eu_i, nu_i, frac_e, frac_n),
            "thorn_crossings": thorn_i,
            "fence_crossings": fence_i,
            "avg_diag_dist"  : diag_i,
        })
    return results


def simulate_competition(submissions, n_sims=NSIMS, n_seeds=1, master_seed=42):
    n = len(submissions)
    master_rng = np.random.default_rng(master_seed)

    acc_rem   = {s["label"]: [] for s in submissions}
    acc_thorn = {s["label"]: [] for s in submissions}
    acc_fence = {s["label"]: [] for s in submissions}
    acc_diag  = {s["label"]: [] for s in submissions}

    seed_matrix = master_rng.integers(0, 10_000_000, size=(n_seeds, n_sims))
    total_iters = n_seeds * n_sims

    print(f"\n  Running {total_iters:,} iterations ", end="", flush=True)
    done = 0

    for s_idx in range(n_seeds):
        for sim_idx in range(n_sims):
            iter_seed = int(seed_matrix[s_idx, sim_idx])
            iter_rng  = np.random.default_rng(iter_seed)
            base_rng  = np.random.default_rng(iter_rng.integers(0, 10_000_000))
            sub_rngs  = [np.random.default_rng(iter_rng.integers(0, 10_000_000))
                         for _ in range(n)]

            iter_results = run_one_iteration(submissions, base_rng, sub_rngs)

            for i, sub in enumerate(submissions):
                lbl   = sub["label"]
                start = 2500 + sub["x"]
                r     = iter_results[i]
                acc_rem[lbl].append(start - r["pen_toll"])
                acc_thorn[lbl].append(r["thorn_crossings"])
                acc_fence[lbl].append(r["fence_crossings"])
                acc_diag[lbl].append(r["avg_diag_dist"])

            done += 1
            if done % max(1, total_iters // 20) == 0:
                print("█", end="", flush=True)

    print(" done.\n")

    records = []
    for sub in submissions:
        lbl  = sub["label"]
        rems = np.array(acc_rem[lbl])
        records.append({
            "label"          : lbl,
            "x"              : sub["x"],
            "h"              : sub["h"],
            "cluster"        : sub.get("cluster", "?"),
            "start_apples"   : 2500 + sub["x"],
            "avg_remaining"  : float(rems.mean()),
            "std_remaining"  : float(rems.std()),
            "win_rate"       : float((rems >= 0).mean()),
            "min_remaining"  : float(rems.min()),
            "max_remaining"  : float(rems.max()),
            "avg_thorn_cross": float(np.mean(acc_thorn[lbl])),
            "avg_fence_cross": float(np.mean(acc_fence[lbl])),
            "avg_diag_dist"  : float(np.mean(acc_diag[lbl])),
            "all_remaining"  : acc_rem[lbl],      # keep for plotting
        })

    df = pd.DataFrame(records).sort_values("avg_remaining", ascending=False)
    df["rank"] = range(1, len(df) + 1)
    return df


# ── Field generation ──────────────────────────────────────────────────────────

def sample_cluster(cluster, n, rng, label_prefix):
    xs = rng.normal(cluster["mu_x"], cluster["sig_x"], n).round().astype(int)
    hs = rng.normal(cluster["mu_h"], cluster["sig_h"], n).round().astype(int)
    xs = np.clip(xs, cluster["x_lo"], cluster["x_hi"])
    hs = np.clip(hs, cluster["h_lo"], cluster["h_hi"])
    return [{"x": int(x), "h": int(h),
             "label": f"{label_prefix}_{i}_x{x}_h{h}",
             "cluster": label_prefix}
            for i, (x, h) in enumerate(zip(xs, hs))]


def build_field(field_size, rng_seed):
    rng = np.random.default_rng(rng_seed)
    half = field_size // 2
    subs_a = sample_cluster(CLUSTER_A, half,            rng, "A")
    subs_b = sample_cluster(CLUSTER_B, field_size - half, rng, "B")
    return subs_a + subs_b


# ── Pretty terminal printing ──────────────────────────────────────────────────

def print_header():
    print()
    print(C_CYAN + C_BOLD + "╔" + "═" * 76 + "╗" + C_RESET)
    print(C_CYAN + C_BOLD + "║" + C_RESET +
          C_BOLD + "      🌾  KYBER PROBLEM CHALLENGE  —  Competition Simulator       " + C_RESET +
          C_CYAN + C_BOLD + "      ║" + C_RESET)
    print(C_CYAN + C_BOLD + "╚" + "═" * 76 + "╝" + C_RESET)


def print_field_info(subs_a, subs_b, your_sub):
    xs_a = np.array([s["x"] for s in subs_a])
    hs_a = np.array([s["h"] for s in subs_a])
    xs_b = np.array([s["x"] for s in subs_b])
    hs_b = np.array([s["h"] for s in subs_b])
    print()
    print(C_BOLD + "  ── Field Composition ──────────────────────────────────────────────" + C_RESET)
    print(f"  {C_GREEN}Cluster A{C_RESET} (low-x / cautious)  : {len(subs_a):>3} farmers  "
          f"x∈[{xs_a.min()},{xs_a.max()}] μ={xs_a.mean():.0f}  "
          f"h∈[{hs_a.min()},{hs_a.max()}] μ={hs_a.mean():.0f}")
    print(f"  {C_MAGENTA}Cluster B{C_RESET} (high-x / bold)     : {len(subs_b):>3} farmers  "
          f"x∈[{xs_b.min()},{xs_b.max()}] μ={xs_b.mean():.0f}  "
          f"h∈[{hs_b.min()},{hs_b.max()}] μ={hs_b.mean():.0f}")
    print(f"  {C_YELLOW}YOU{C_RESET}                          : x={your_sub['x']}, h={your_sub['h']}  "
          f"(start apples = {2500+your_sub['x']})")


def print_results(df, your_label, top_k=25):
    print()
    print(C_BOLD + C_CYAN + "═" * 108 + C_RESET)
    print(C_BOLD + "  COMPETITION RESULTS  (with crowd penalty)" + C_RESET)
    print(C_BOLD + C_CYAN + "═" * 108 + C_RESET)
    hdr = (f"  {'Rank':>4}  {'x':>4}  {'h':>3}  {'Cluster':>7}  "
           f"{'avg_rem':>10}  {'std':>8}  {'win%':>6}  "
           f"{'thorn_x':>8}  {'fence_x':>8}  {'diag_d':>7}")
    print(C_DIM + hdr + C_RESET)
    print("  " + "─" * 104)

    for _, row in df.head(top_k).iterrows():
        is_you = row["label"] == your_label
        cl     = row["cluster"]
        cl_col = C_GREEN if cl == "A" else (C_MAGENTA if cl == "B" else C_YELLOW)
        cl_sym = "A" if cl == "A" else ("B" if cl == "B" else "★")

        avg_r  = row["avg_remaining"]
        color  = C_GREEN if avg_r > 0 else C_RED

        line = (f"  {int(row['rank']):>4}  "
                f"{int(row['x']):>4}  {int(row['h']):>3}  "
                f"{cl_col}{cl_sym:>7}{C_RESET}  "
                f"{color}{avg_r:>10.2f}{C_RESET}  "
                f"{row['std_remaining']:>8.2f}  "
                f"{row['win_rate']*100:>5.1f}%  "
                f"{row['avg_thorn_cross']:>8.2f}  "
                f"{row['avg_fence_cross']:>8.2f}  "
                f"{row['avg_diag_dist']:>7.4f}")

        if is_you:
            print(C_YELLOW + C_BOLD + line + "  ◄ YOU" + C_RESET)
        else:
            print(line)

    # show YOU if not in top_k
    if your_label not in df.head(top_k)["label"].values:
        tgt = df[df["label"] == your_label]
        if len(tgt):
            row = tgt.iloc[0]
            print("  ...")
            avg_r = row["avg_remaining"]
            color = C_GREEN if avg_r > 0 else C_RED
            print(C_YELLOW + C_BOLD +
                  f"  {int(row['rank']):>4}  {int(row['x']):>4}  {int(row['h']):>3}  "
                  f"{'★':>7}  {avg_r:>10.2f}  "
                  f"{row['std_remaining']:>8.2f}  "
                  f"{row['win_rate']*100:>5.1f}%  "
                  f"{row['avg_thorn_cross']:>8.2f}  "
                  f"{row['avg_fence_cross']:>8.2f}  "
                  f"{row['avg_diag_dist']:>7.4f}  ◄ YOU" + C_RESET)

    print()
    best = df.iloc[0]
    tgt  = df[df["label"] == your_label].iloc[0]

    print(C_BOLD + "═" * 108 + C_RESET)
    print(f"  🥇 Winner   : x={best['x']}, h={best['h']}  →  avg_rem = "
          f"{C_GREEN}{best['avg_remaining']:.2f}{C_RESET} apples")
    print(f"  🧑 You      : x={tgt['x']}, h={tgt['h']}   →  avg_rem = "
          f"{C_YELLOW}{tgt['avg_remaining']:.2f}{C_RESET} apples  "
          f"| rank {C_BOLD}{int(tgt['rank'])}{C_RESET} / {len(df)}")
    print(f"  📊 Gap to 1st: {best['avg_remaining'] - tgt['avg_remaining']:.2f} apples")
    print(f"  📈 Win rate  : {tgt['win_rate']*100:.1f}%  "
          f"(finished with ≥ 0 apples in that fraction of runs)")
    print(C_BOLD + "═" * 108 + C_RESET)
    print()
    print("  " + C_DIM + "Cluster key: A = low-x/cautious  B = high-x/bold  ★ = YOU" + C_RESET)
    print()


# ── Matplotlib charts ─────────────────────────────────────────────────────────

def make_charts(df, your_label, subs_a, subs_b):
    your_row = df[df["label"] == your_label].iloc[0]

    # Colour map: A=green, B=purple, YOU=gold
    def row_color(row):
        if row["label"] == your_label: return "#FFD700"
        return "#4CAF50" if row["cluster"] == "A" else "#9C27B0"

    colors = [row_color(r) for _, r in df.iterrows()]

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(18, 11), facecolor="#0d1117")
    fig.suptitle("🌾  Kyber Challenge — Simulation Results",
                 fontsize=16, fontweight="bold", color="#e6edf3",
                 fontfamily="monospace", y=0.98)
    gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35,
                  left=0.06, right=0.97, top=0.93, bottom=0.07)

    ax_rank   = fig.add_subplot(gs[0, 0])
    ax_xh     = fig.add_subplot(gs[0, 1])
    ax_dist   = fig.add_subplot(gs[0, 2])
    ax_thorn  = fig.add_subplot(gs[1, 0])
    ax_win    = fig.add_subplot(gs[1, 1])
    ax_box    = fig.add_subplot(gs[1, 2])

    # common style helper
    def style_ax(ax, title, xlabel="", ylabel=""):
        ax.set_facecolor("#161b22")
        ax.set_title(title, color="#e6edf3", fontsize=10, fontweight="bold", pad=7)
        ax.tick_params(colors="#8b949e", labelsize=8)
        ax.spines[:].set_color("#30363d")
        if xlabel: ax.set_xlabel(xlabel, color="#8b949e", fontsize=8)
        if ylabel: ax.set_ylabel(ylabel, color="#8b949e", fontsize=8)

    # ── 1. Ranking bar: top 30 avg_remaining ────────────────────────────────
    top30  = df.head(30)
    c30    = [row_color(r) for _, r in top30.iterrows()]
    bars   = ax_rank.barh(range(len(top30)), top30["avg_remaining"],
                          color=c30, edgecolor="none", height=0.75)
    ax_rank.axvline(0, color="#ff4444", lw=1, ls="--", alpha=0.6)
    ax_rank.set_yticks(range(len(top30)))
    ax_rank.set_yticklabels([f"#{int(r['rank'])}" for _, r in top30.iterrows()],
                            fontsize=7, color="#8b949e")
    ax_rank.invert_yaxis()
    # mark YOU
    you_in30 = top30[top30["label"] == your_label]
    if len(you_in30):
        idx = list(top30["label"]).index(your_label)
        ax_rank.barh(idx, you_in30["avg_remaining"].values[0],
                     color="#FFD700", edgecolor="#ff8c00", lw=1.5, height=0.75)
        ax_rank.text(you_in30["avg_remaining"].values[0] + 2, idx,
                     "◄ YOU", va="center", color="#FFD700",
                     fontsize=7, fontweight="bold")
    style_ax(ax_rank, "Top-30 Avg Remaining Apples", "apples", "rank")

    # ── 2. Scatter: x vs h, coloured by cluster ─────────────────────────────
    ca = df[df["cluster"] == "A"]
    cb = df[df["cluster"] == "B"]
    ax_xh.scatter(ca["x"], ca["h"], c="#4CAF50", alpha=0.65, s=28, label="Cluster A")
    ax_xh.scatter(cb["x"], cb["h"], c="#9C27B0", alpha=0.65, s=28, label="Cluster B")
    ax_xh.scatter([your_row["x"]], [your_row["h"]], c="#FFD700",
                  s=220, zorder=5, marker="*", label="YOU")
    ax_xh.legend(fontsize=7, facecolor="#161b22", edgecolor="#30363d",
                 labelcolor="#e6edf3")
    style_ax(ax_xh, "Strategy Space  (x vs h)", "x  (extra apples / thorns)", "h  (threshold)")

    # ── 3. Distribution: avg_remaining by cluster ────────────────────────────
    bins = np.linspace(df["avg_remaining"].min(), df["avg_remaining"].max(), 35)
    ax_dist.hist(ca["avg_remaining"], bins=bins, color="#4CAF50",
                 alpha=0.7, label="Cluster A", edgecolor="none")
    ax_dist.hist(cb["avg_remaining"], bins=bins, color="#9C27B0",
                 alpha=0.7, label="Cluster B", edgecolor="none")
    ax_dist.axvline(your_row["avg_remaining"], color="#FFD700",
                    lw=2, ls="--", label="YOU")
    ax_dist.axvline(0, color="#ff4444", lw=1, ls=":", alpha=0.7)
    ax_dist.legend(fontsize=7, facecolor="#161b22", edgecolor="#30363d",
                   labelcolor="#e6edf3")
    style_ax(ax_dist, "Distribution of Avg Remaining", "avg remaining apples", "count")

    # ── 4. Thorn crossings vs avg_remaining ─────────────────────────────────
    ax_thorn.scatter(ca["avg_thorn_cross"], ca["avg_remaining"],
                     c="#4CAF50", alpha=0.55, s=22, label="A")
    ax_thorn.scatter(cb["avg_thorn_cross"], cb["avg_remaining"],
                     c="#9C27B0", alpha=0.55, s=22, label="B")
    ax_thorn.scatter([your_row["avg_thorn_cross"]], [your_row["avg_remaining"]],
                     c="#FFD700", s=200, zorder=5, marker="*", label="YOU")
    ax_thorn.axhline(0, color="#ff4444", lw=1, ls="--", alpha=0.6)
    ax_thorn.legend(fontsize=7, facecolor="#161b22", edgecolor="#30363d",
                    labelcolor="#e6edf3")
    style_ax(ax_thorn, "Thorn Crossings vs Score", "avg thorn crossings", "avg remaining")

    # ── 5. Win rate by cluster (bar chart) ───────────────────────────────────
    wr_a   = ca["win_rate"].mean() * 100
    wr_b   = cb["win_rate"].mean() * 100
    wr_you = your_row["win_rate"] * 100
    cats   = ["Cluster A\n(low-x)", "Cluster B\n(high-x)", "YOU"]
    vals   = [wr_a, wr_b, wr_you]
    cols   = ["#4CAF50", "#9C27B0", "#FFD700"]
    brs    = ax_win.bar(cats, vals, color=cols, edgecolor="none", width=0.5)
    for bar, val in zip(brs, vals):
        ax_win.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", va="bottom",
                    color="#e6edf3", fontsize=9, fontweight="bold")
    ax_win.set_ylim(0, 110)
    ax_win.axhline(50, color="#888", lw=1, ls=":", alpha=0.5)
    style_ax(ax_win, "Average Win Rate by Group", ylabel="% of runs ≥ 0 apples")

    # ── 6. Box plot of per-run remaining for YOU + top-5 ────────────────────
    top5   = df.head(5)
    box_data  = [r["all_remaining"] for _, r in top5.iterrows()]
    you_data  = [your_row["all_remaining"]]
    you_in_t5 = your_label in list(top5["label"])
    if not you_in_t5:
        all_box   = box_data + you_data
        labels_bx = [f"#{int(r['rank'])}\nx={int(r['x'])}" for _, r in top5.iterrows()] + ["★YOU"]
        col_bx    = ["#4CAF50" if r["cluster"]=="A" else "#9C27B0"
                     for _, r in top5.iterrows()] + ["#FFD700"]
    else:
        all_box   = box_data
        labels_bx = [f"#{int(r['rank'])}\nx={int(r['x'])}" +
                     (" ★" if r["label"]==your_label else "")
                     for _, r in top5.iterrows()]
        col_bx    = ["#FFD700" if r["label"]==your_label
                     else ("#4CAF50" if r["cluster"]=="A" else "#9C27B0")
                     for _, r in top5.iterrows()]

    bp = ax_box.boxplot(all_box, patch_artist=True, notch=False,
                        medianprops=dict(color="white", lw=2),
                        whiskerprops=dict(color="#8b949e"),
                        capprops=dict(color="#8b949e"),
                        flierprops=dict(marker=".", color="#8b949e", alpha=0.4, ms=4))
    for patch, col in zip(bp["boxes"], col_bx):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)
    ax_box.set_xticks(range(1, len(all_box) + 1))
    ax_box.set_xticklabels(labels_bx, fontsize=7)
    ax_box.axhline(0, color="#ff4444", lw=1, ls="--", alpha=0.7)
    style_ax(ax_box, "Score Distribution: Top-5 + YOU", ylabel="apples remaining")

    plt.savefig("kyber_results.png", dpi=150, bbox_inches="tight",
                facecolor="#0d1117")
    print(f"  📊 Chart saved to {C_CYAN}kyber_results.png{C_RESET}")
    plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Kyber Challenge — two-cluster field simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--x",          type=int, default=None,
                   help="Your x submission (0–300)")
    p.add_argument("--h",          type=int, default=None,
                   help="Your h submission (5–60)")
    p.add_argument("--field-size", type=int, default=100, dest="field_size",
                   help="Total rival farmers to simulate  [100]")
    p.add_argument("--n-sims",     type=int, default=NSIMS,  dest="n_sims",
                   help="Competition iterations per seed  [50]")
    p.add_argument("--seeds",      type=int, default=5,
                   help="Independent seeds to average over  [5]")
    p.add_argument("--seed",       type=int, default=None,
                   help="Master RNG seed (random if omitted)")
    p.add_argument("--top-show",   type=int, default=25, dest="top_show",
                   help="Rows to print in result table  [25]")
    p.add_argument("--no-chart",   action="store_true", dest="no_chart",
                   help="Skip matplotlib charts")
    return p.parse_args()


def main():
    args = parse_args()

    print_header()

    # ── Get x and h ──────────────────────────────────────────────────────────
    if args.x is None:
        print()
        try:
            args.x = int(input(C_YELLOW + "  Enter your x (0–300) : " + C_RESET))
        except (ValueError, EOFError):
            print("[ERROR] Invalid x", file=sys.stderr); sys.exit(1)
    if args.h is None:
        try:
            args.h = int(input(C_YELLOW + "  Enter your h (5–60)  : " + C_RESET))
        except (ValueError, EOFError):
            print("[ERROR] Invalid h", file=sys.stderr); sys.exit(1)

    x, h = args.x, args.h
    if not (X_MIN <= x <= X_MAX):
        print(f"[ERROR] x={x} out of range [{X_MIN},{X_MAX}]", file=sys.stderr)
        sys.exit(1)
    if not (H_MIN <= h <= H_MAX):
        print(f"[ERROR] h={h} out of range [{H_MIN},{H_MAX}]", file=sys.stderr)
        sys.exit(1)

    seed = args.seed if args.seed is not None else int(time.time() * 1000) % (2**32 - 1)

    # ── Build field ───────────────────────────────────────────────────────────
    field = build_field(args.field_size, seed + 1)
    subs_a = [s for s in field if s["cluster"] == "A"]
    subs_b = [s for s in field if s["cluster"] == "B"]

    your_label = f"★YOU_x{x}_h{h}"
    your_sub   = {"x": x, "h": h, "label": your_label, "cluster": "YOU"}
    submissions = [your_sub] + field

    print_field_info(subs_a, subs_b, your_sub)

    print()
    print(C_BOLD + "  ── Simulation Settings ─────────────────────────────────────────────" + C_RESET)
    print(f"  Field size    : {len(submissions)} total submissions")
    print(f"  Iterations    : {args.n_sims} sims × {args.seeds} seeds = "
          f"{args.n_sims * args.seeds:,} total")
    print(f"  Master seed   : {seed}")

    # ── Simulate ──────────────────────────────────────────────────────────────
    df = simulate_competition(
        submissions  = submissions,
        n_sims       = args.n_sims,
        n_seeds      = args.seeds,
        master_seed  = seed,
    )

    # ── Print results ─────────────────────────────────────────────────────────
    print_results(df, your_label, top_k=args.top_show)

    # ── Charts ────────────────────────────────────────────────────────────────
    if not args.no_chart:
        print("  Generating charts …")
        make_charts(df, your_label, subs_a, subs_b)
    else:
        print("  (Charts skipped — remove --no-chart to enable)")


if __name__ == "__main__":
    main()