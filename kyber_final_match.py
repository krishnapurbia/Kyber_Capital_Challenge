#!/usr/bin/env python3
"""
kyber_tournament.py
═══════════════════════════════════════════════════════════════════════════════
Reads the kyber_best.csv produced by kyber_sweep_cache.py, takes the top-2
x values from every h, and runs a multi-round crowd-penalised tournament to
find the overall champion configuration.

Tournament structure
────────────────────
  Grand Pool Round  – all top-2 entries (up to 2 × 56 = 112 submissions) run
                      as ONE competition so crowd-penalty is applied globally
                      (many similar x values are penalised together).

  Semi-Final        – Top-16 from the pool by avg_remaining re-compete.

  Final             – Top-4 from the semi-final re-compete.

  Champion          – The single winner from the final.

A full diagnostic plot is produced for each round, plus a summary CSV of
the final standings.

Usage:
    python kyber_tournament.py                         # defaults
    python kyber_tournament.py --csv kyber_best.csv
    python kyber_tournament.py --semi-n 16 --final-n 4
    python kyber_tournament.py --n-sims 200 --seeds 20

Options:
    --csv       Path to the cached best-x CSV       [kyber_best.csv]
    --semi-n    Pool survivors to semi-final         [16]
    --final-n   Semi survivors to final              [4]
    --n-sims    Simulation iterations per seed       [100]
    --seeds     Independent seeds                    [10]
    --seed      Master RNG seed (None = time-based)  [None]
    --out-dir   Directory for output files           [./tournament_out]
    --dpi       Plot resolution                      [150]
    --no-show   Don't open plots interactively
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

try:
    from kyber_simulate import simulate_competition
except ImportError:
    print("[ERROR] kyber_simulate.py not found in the current directory.", file=sys.stderr)
    sys.exit(1)


# ── Visuals ───────────────────────────────────────────────────────────────────
_BG     = "#0F1117"
_PANEL  = "#181C27"
_GRID   = "#2A2F3D"
_TEXT   = "#E0E6F0"
_MUTED  = "#6B7A99"
_GOLD   = "#FFD54F"
_SILVER = "#90CAF9"
_BRONZE = "#FFAB91"
_RED    = "#FF6B6B"

_TIER_COLORS = {
    "grand_pool": "#4FC3F7",
    "semi_final": "#CE93D8",
    "final":      "#FFD54F",
    "champion":   "#FF8A65",
}


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Kyber tournament — top-2-per-h compete across rounds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--csv",      type=str, default="kyber_best.csv")
    p.add_argument("--semi-n",   type=int, default=16,  dest="semi_n")
    p.add_argument("--final-n",  type=int, default=4,   dest="final_n")
    p.add_argument("--n-sims",   type=int, default=100, dest="n_sims")
    p.add_argument("--seeds",    type=int, default=10)
    p.add_argument("--seed",     type=int, default=None)
    p.add_argument("--out-dir",  type=str, default="tournament_out", dest="out_dir")
    p.add_argument("--dpi",      type=int, default=150)
    p.add_argument("--no-show",  action="store_true", dest="no_show")
    return p.parse_args()


# ── Data loading ──────────────────────────────────────────────────────────────

def load_entries(csv_path: Path, top_n: int = 2) -> list[dict]:
    """
    Load the cache CSV and return a list of submission dicts.
    Only keeps ranks 1 and 2 (or up to `top_n`).
    """
    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}", file=sys.stderr)
        print("  Run kyber_sweep_cache.py first to generate it.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    df = df[df["rank"] <= top_n].copy()

    if df.empty:
        print("[ERROR] No entries found in CSV.", file=sys.stderr)
        sys.exit(1)

    entries = []
    for _, row in df.iterrows():
        entries.append({
            "x":     int(row["x"]),
            "h":     int(row["h"]),
            "rank":  int(row["rank"]),
            "label": f"h{int(row['h'])}_x{int(row['x'])}_r{int(row['rank'])}",
        })
    return entries


# ── Simulation helpers ────────────────────────────────────────────────────────

def run_round(submissions: list[dict],
              n_sims: int, seeds: int, seed: int,
              round_name: str) -> pd.DataFrame:
    """Run one tournament round and return sorted results."""
    print(f"  [{round_name}]  {len(submissions)} contestants  "
          f"({n_sims * seeds:,} iters each) … ", end="", flush=True)
    t0 = time.time()

    df = simulate_competition(
        submissions=submissions,
        n_sims=n_sims,
        n_seeds=seeds,
        master_seed=seed,
    )
    df["sharpe"] = df["avg_remaining"] / (df["std_remaining"] + 1.0)

    # Re-attach metadata from submissions dict
    label_map = {s["label"]: s for s in submissions}
    df["h_fixed"] = df["label"].map(lambda l: label_map.get(l, {}).get("h", np.nan))
    df["rank_in"] = df["label"].map(lambda l: label_map.get(l, {}).get("rank", np.nan))

    df = df.sort_values("avg_remaining", ascending=False).reset_index(drop=True)
    df["tournament_rank"] = df.index + 1

    elapsed = time.time() - t0
    print(f"done ({elapsed:.1f}s)  winner: {df.iloc[0]['label']}  "
          f"avg_rem={df.iloc[0]['avg_remaining']:.3f}")
    return df


# ── Plot builders ─────────────────────────────────────────────────────────────

def _style_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor(_PANEL)
    ax.set_title(title, color=_TEXT, fontsize=10, fontweight="bold", pad=6)
    ax.set_xlabel(xlabel, color=_MUTED, fontsize=8)
    ax.set_ylabel(ylabel, color=_MUTED, fontsize=8)
    ax.tick_params(colors=_MUTED, labelsize=7.5)
    for spine in ax.spines.values():
        spine.set_color(_GRID)
    ax.grid(True, color=_GRID, linewidth=0.6, linestyle="--", alpha=0.7)


def plot_round(df: pd.DataFrame, round_name: str,
               color: str, out_path: Path, dpi: int, no_show: bool,
               top_highlight: int = 16):
    matplotlib.rcParams["font.family"] = "monospace"

    n   = len(df)
    idx = np.arange(n)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=_BG)
    fig.suptitle(f"Kyber Tournament  ·  {round_name}  ({n} contestants)",
                 color=_TEXT, fontsize=14, fontweight="bold", y=1.01)

    # Bar colours: top survivors gold, rest muted
    bar_colors = [_GOLD if i < top_highlight else color for i in range(n)]

    # ── Panel 1: avg_remaining ────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor(_PANEL)
    bars = ax.barh(idx, df["avg_remaining"].values,
                   color=bar_colors, alpha=0.85, height=0.7)
    ax.axvline(0, color=_RED, linewidth=0.9, linestyle="--", alpha=0.8)
    ax.set_yticks(idx)
    ax.set_yticklabels(df["label"].values, fontsize=6, color=_MUTED)
    ax.invert_yaxis()
    _style_ax(ax, "① Avg Remaining Apples", "avg remaining", "")

    # ── Panel 2: win rate ─────────────────────────────────────────────────
    ax = axes[1]
    ax.set_facecolor(_PANEL)
    ax.barh(idx, df["win_rate"].values * 100,
            color=bar_colors, alpha=0.85, height=0.7)
    ax.set_yticks(idx)
    ax.set_yticklabels(df["label"].values, fontsize=6, color=_MUTED)
    ax.invert_yaxis()
    _style_ax(ax, "② Win Rate  (%)", "win rate %", "")

    # ── Panel 3: sharpe ───────────────────────────────────────────────────
    ax = axes[2]
    ax.set_facecolor(_PANEL)
    ax.barh(idx, df["sharpe"].values,
            color=bar_colors, alpha=0.85, height=0.7)
    ax.set_yticks(idx)
    ax.set_yticklabels(df["label"].values, fontsize=6, color=_MUTED)
    ax.invert_yaxis()
    _style_ax(ax, "③ Sharpe Score  (avg / (std+1))", "sharpe", "")

    plt.tight_layout(pad=1.5)
    fig.patch.set_facecolor(_BG)

    fig.savefig(out_path, dpi=dpi, facecolor=_BG, bbox_inches="tight")
    print(f"    Plot → {out_path}")
    if not no_show:
        plt.show()
    plt.close(fig)


def plot_final_podium(grand: pd.DataFrame, semi: pd.DataFrame,
                      final: pd.DataFrame, out_path: Path,
                      dpi: int, no_show: bool):
    """Single-page visual comparing all 3 rounds' top-3 side by side."""
    matplotlib.rcParams["font.family"] = "monospace"

    fig, axes = plt.subplots(1, 3, figsize=(18, 7), facecolor=_BG)
    fig.suptitle("Kyber Tournament  ·  Podium Summary",
                 color=_TEXT, fontsize=15, fontweight="bold", y=1.01)

    rounds = [
        (grand.head(10), "Grand Pool  (top 10)", _TIER_COLORS["grand_pool"]),
        (semi.head(8),   "Semi-Final  (top 8)",  _TIER_COLORS["semi_final"]),
        (final,          "Final",                _TIER_COLORS["final"]),
    ]

    medal_colors = [_GOLD, _SILVER, _BRONZE]

    for ax, (df_r, title, color) in zip(axes, rounds):
        ax.set_facecolor(_PANEL)
        n   = len(df_r)
        idx = np.arange(n)

        colors = [medal_colors[i] if i < 3 else color for i in range(n)]
        ax.barh(idx, df_r["avg_remaining"].values,
                color=colors, alpha=0.9, height=0.7)

        for i, (val, lbl) in enumerate(zip(df_r["avg_remaining"].values,
                                           df_r["label"].values)):
            prefix = ["🥇", "🥈", "🥉"][i] if i < 3 else "  "
            ax.text(val + 0.02, i, f" {prefix} {lbl}", va="center",
                    color=_TEXT, fontsize=7.5)

        ax.axvline(0, color=_RED, linewidth=0.9, linestyle="--", alpha=0.7)
        ax.set_yticks(idx)
        ax.set_yticklabels(df_r["label"].values, fontsize=7, color=_MUTED)
        ax.invert_yaxis()
        _style_ax(ax, title, "avg remaining apples", "")

    plt.tight_layout(pad=1.5)
    fig.patch.set_facecolor(_BG)
    fig.savefig(out_path, dpi=dpi, facecolor=_BG, bbox_inches="tight")
    print(f"  Podium plot → {out_path}")
    if not no_show:
        plt.show()
    plt.close(fig)


# ── CSV saver ─────────────────────────────────────────────────────────────────

def save_results(df: pd.DataFrame, path: Path, round_name: str):
    df.to_csv(path, index=False)
    print(f"    Results CSV → {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.seed is None:
        args.seed = int(time.time() * 1000) % (2**32 - 1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(args.csv)
    entries  = load_entries(csv_path, top_n=2)

    print()
    print("═" * 65)
    print("  Kyber Tournament")
    print("═" * 65)
    print(f"  Contestants  : {len(entries)}  (top-2 from each h in CSV)")
    print(f"  Semi-final N : top {args.semi_n} advance")
    print(f"  Final N      : top {args.final_n} advance")
    print(f"  n_sims       : {args.n_sims}")
    print(f"  seeds        : {args.seeds}")
    print(f"  master seed  : {args.seed}")
    print(f"  output dir   : {out_dir.resolve()}")
    print()

    # ── GRAND POOL ────────────────────────────────────────────────────────
    print("  ROUND 1 — Grand Pool")
    pool_subs = [{"x": e["x"], "h": e["h"], "label": e["label"]} for e in entries]
    seed_pool = (args.seed + 1) % (2**32 - 1)
    grand_df  = run_round(pool_subs, args.n_sims, args.seeds,
                          seed_pool, "Grand Pool")
    save_results(grand_df, out_dir / "round1_grand_pool.csv", "Grand Pool")
    plot_round(grand_df, "Round 1 — Grand Pool",
               _TIER_COLORS["grand_pool"],
               out_dir / "round1_grand_pool.png",
               args.dpi, args.no_show, top_highlight=args.semi_n)

    # ── SEMI-FINAL ────────────────────────────────────────────────────────
    print()
    print(f"  ROUND 2 — Semi-Final  (top {args.semi_n} from grand pool)")
    semi_entries_df = grand_df.head(args.semi_n)
    semi_subs = [
        {"x": int(row["x"]), "h": int(row["h_fixed"]), "label": row["label"]}
        for _, row in semi_entries_df.iterrows()
    ]
    seed_semi = (args.seed + 2) % (2**32 - 1)
    semi_df   = run_round(semi_subs, args.n_sims, args.seeds,
                          seed_semi, "Semi-Final")
    save_results(semi_df, out_dir / "round2_semi_final.csv", "Semi-Final")
    plot_round(semi_df, "Round 2 — Semi-Final",
               _TIER_COLORS["semi_final"],
               out_dir / "round2_semi_final.png",
               args.dpi, args.no_show, top_highlight=args.final_n)

    # ── FINAL ─────────────────────────────────────────────────────────────
    print()
    print(f"  ROUND 3 — Final  (top {args.final_n} from semi-final)")
    final_entries_df = semi_df.head(args.final_n)
    final_subs = [
        {"x": int(row["x"]), "h": int(row["h_fixed"]), "label": row["label"]}
        for _, row in final_entries_df.iterrows()
    ]
    seed_final = (args.seed + 3) % (2**32 - 1)
    final_df   = run_round(final_subs, args.n_sims, args.seeds,
                           seed_final, "Final")
    save_results(final_df, out_dir / "round3_final.csv", "Final")
    plot_round(final_df, "Round 3 — Final",
               _TIER_COLORS["final"],
               out_dir / "round3_final.png",
               args.dpi, args.no_show, top_highlight=1)

    # ── PODIUM SUMMARY PLOT ───────────────────────────────────────────────
    print()
    print("  Building podium summary …")
    plot_final_podium(grand_df, semi_df, final_df,
                      out_dir / "podium_summary.png",
                      args.dpi, args.no_show)

    # ── CHAMPION ─────────────────────────────────────────────────────────
    champ = final_df.iloc[0]
    print()
    print("╔" + "═" * 55 + "╗")
    print("║  🏆  CHAMPION                                         ║")
    print("║" + "─" * 55 + "║")
    print(f"║  Label        : {champ['label']:<37} ║")
    print(f"║  h            : {int(champ['h_fixed']):<37} ║")
    print(f"║  x            : {int(champ['x']):<37} ║")
    print(f"║  avg_remaining: {champ['avg_remaining']:<37.4f} ║")
    print(f"║  win_rate     : {champ['win_rate']*100:<36.2f}% ║")
    print(f"║  sharpe       : {champ['sharpe']:<37.4f} ║")
    print("╚" + "═" * 55 + "╝")
    print()

    # ── FULL STANDINGS ────────────────────────────────────────────────────
    final_df.to_csv(out_dir / "final_standings.csv", index=False)
    print(f"  Full standings → {(out_dir / 'final_standings.csv').resolve()}")
    print()


if __name__ == "__main__":
    main()