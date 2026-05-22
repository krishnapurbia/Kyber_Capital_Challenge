# #!/usr/bin/env python3
# """
# kyber_analysis.py
# ═══════════════════════════════════════════════════════════════════════════════
# Sweeps x over [x_min, x_max] for one or more fixed h values, runs the full
# crowd-penalised Kyber simulation, and produces a 6-panel diagnostic plot.

# Each h value is run as its own independent competition (all x submissions for
# that h compete against each other with crowd penalty).

# Usage — single h:
#     python kyber_analysis.py --h 20

# Usage — compare multiple h values:
#     python kyber_analysis.py --h 10 20 30 40 --x-step 10 --seeds 5

# Options:
#     --h          H value(s) to analyse  (space-separated)  [20]
#     --x-min      Minimum x in sweep                        [0]
#     --x-max      Maximum x in sweep                        [300]
#     --x-step     Step size for x sweep                     [10]
#     --n-sims     Simulation iterations per seed            [50]
#     --seeds      Independent seeds to average over         [5]
#     --seed       Master RNG seed  (None = time-based)      [None]
#     --out        Output file path (.png / .pdf / .svg)     [kyber_analysis.png]
#     --no-show    Don't open the plot interactively
#     --dpi        Output resolution                         [150]

# Panels produced
# ───────────────
#   1. Avg Remaining Apples ± 1σ  vs  X   ← primary metric
#   2. Win Rate (%)               vs  X
#   3. Fence-crossing breakdown   vs  X   (thorn vs plain, stacked area)
#   4. Avg Diagonal Distance      vs  X
#   5. Volatility (std_remaining) vs  X
#   6. Risk-adjusted score        vs  X   (avg_rem / (std_rem + 1), Sharpe-like)
# """

# import argparse
# import sys
# import time
# from pathlib import Path

# import numpy as np
# import pandas as pd
# import matplotlib
# import matplotlib.pyplot as plt
# import matplotlib.ticker as mticker
# from matplotlib.gridspec import GridSpec
# from matplotlib.lines import Line2D
# from matplotlib.patches import Patch

# # ── import simulator ──────────────────────────────────────────────────────────
# try:
#     from kyber_simulate import simulate_competition, X_MIN, X_MAX, H_MIN, H_MAX
# except ImportError:
#     print("[ERROR] kyber_simulate.py not found in the current directory.", file=sys.stderr)
#     sys.exit(1)


# # ── Colour palette ────────────────────────────────────────────────────────────
# # A hand-picked sequence that is readable on the dark background and stays
# # distinguishable for up to 6 different h values.
# _PALETTE = [
#     "#4FC3F7",   # sky blue
#     "#FF8A65",   # soft orange
#     "#81C784",   # sage green
#     "#CE93D8",   # lavender
#     "#FFD54F",   # amber
#     "#F06292",   # rose
# ]

# _BG      = "#0F1117"
# _PANEL   = "#181C27"
# _GRID_C  = "#2A2F3D"
# _TEXT    = "#E0E6F0"
# _MUTED   = "#6B7A99"
# _ZERO    = "#FF6B6B"     # colour for the break-even line


# # ── CLI ───────────────────────────────────────────────────────────────────────

# def parse_args() -> argparse.Namespace:
#     p = argparse.ArgumentParser(
#         description="Kyber x-sweep analysis with diagnostic plots",
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#     )
#     p.add_argument("--h",      type=int, nargs="+", default=[20],
#                    help="H value(s) to analyse  (space-separated)  [20]")
#     p.add_argument("--x-min",  type=int, default=X_MIN,   dest="x_min",
#                    help=f"Min x  [{X_MIN}]")
#     p.add_argument("--x-max",  type=int, default=X_MAX,   dest="x_max",
#                    help=f"Max x  [{X_MAX}]")
#     p.add_argument("--x-step", type=int, default=10,      dest="x_step",
#                    help="Step size for x sweep  [10]")
#     p.add_argument("--n-sims", type=int, default=50,      dest="n_sims",
#                    help="Iterations per seed  [50]")
#     p.add_argument("--seeds",  type=int, default=5,
#                    help="Independent seeds  [5]")
#     p.add_argument("--seed",   type=int, default=None,
#                    help="Master RNG seed  [time-based]")
#     p.add_argument("--out",    type=str, default="kyber_analysis.png",
#                    help="Output file path  [kyber_analysis.png]")
#     p.add_argument("--no-show", action="store_true", dest="no_show",
#                    help="Don't open the plot interactively")
#     p.add_argument("--dpi",    type=int, default=150,
#                    help="Output resolution  [150]")
#     return p.parse_args()


# # ── Simulation driver ─────────────────────────────────────────────────────────

# def run_sweep(h_val: int, x_vals: np.ndarray,
#               n_sims: int, seeds: int, master_seed: int) -> pd.DataFrame:
#     """
#     Build one submission per x (all with the same h_val), run simulate_competition
#     once so all submissions share the same crowd-penalty field, and return the
#     result DataFrame filtered & sorted by x.
#     """
#     submissions = [
#         {"x": int(x), "h": h_val, "label": f"x{int(x)}_h{h_val}"}
#         for x in x_vals
#     ]

#     print(f"  h={h_val:>3}  →  {len(submissions)} submissions "
#           f"({n_sims * seeds:,} total iterations) …", end="", flush=True)

#     df = simulate_competition(
#         submissions = submissions,
#         n_sims      = n_sims,
#         n_seeds     = seeds,
#         master_seed = master_seed,
#     )

#     # Attach h column and sort by x for clean plotting
#     df["h_fixed"] = h_val
#     df = df.sort_values("x").reset_index(drop=True)
#     print("  done.")
#     return df


# # ── Plot helpers ──────────────────────────────────────────────────────────────

# def _style_ax(ax, title: str, xlabel: str, ylabel: str, xlim=None):
#     """Apply the dark-panel style to a single Axes."""
#     ax.set_facecolor(_PANEL)
#     ax.set_title(title, color=_TEXT, fontsize=11, fontweight="bold", pad=8)
#     ax.set_xlabel(xlabel, color=_MUTED, fontsize=9)
#     ax.set_ylabel(ylabel, color=_MUTED, fontsize=9)
#     ax.tick_params(colors=_MUTED, labelsize=8)
#     for spine in ax.spines.values():
#         spine.set_color(_GRID_C)
#     ax.grid(True, color=_GRID_C, linewidth=0.6, linestyle="--", alpha=0.7)
#     if xlim:
#         ax.set_xlim(xlim)


# def _annotate_peak(ax, xs, ys, color):
#     """Drop a dashed vertical line and label at the x with maximum y."""
#     best_idx = np.argmax(ys)
#     bx, by   = xs[best_idx], ys[best_idx]
#     ax.axvline(bx, color=color, linewidth=0.8, linestyle=":", alpha=0.6)
#     ax.annotate(f"x={bx}", xy=(bx, by),
#                 xytext=(4, -14), textcoords="offset points",
#                 color=color, fontsize=7.5, alpha=0.9)


# # ── Main plot builder ─────────────────────────────────────────────────────────

# def build_plot(sweep_results: dict[int, pd.DataFrame],
#                x_vals: np.ndarray, args: argparse.Namespace):
#     """
#     sweep_results : {h_val: DataFrame}  one entry per h value

#     Panel layout (2 rows × 3 cols)
#     ┌────────────────────┬────────────────┬──────────────────────┐
#     │ 1. Avg rem ± 1σ   │  2. Win rate   │  3. Crossing breakdn │
#     ├────────────────────┼────────────────┼──────────────────────┤
#     │ 4. Diag distance  │  5. Volatility │  6. Risk-adj score   │
#     └────────────────────┴────────────────┴──────────────────────┘
#     """
#     matplotlib.rcParams["font.family"] = "monospace"

#     fig = plt.figure(figsize=(18, 10), facecolor=_BG)
#     fig.suptitle(
#         "Kyber Challenge  ·  X-sweep Analysis"
#         + (f"   (h = {args.h[0]})" if len(args.h) == 1 else "   (multi-h comparison)"),
#         color=_TEXT, fontsize=15, fontweight="bold", y=0.97
#     )

#     gs = GridSpec(2, 3, figure=fig,
#                   hspace=0.42, wspace=0.32,
#                   left=0.07, right=0.97, top=0.91, bottom=0.09)

#     ax1 = fig.add_subplot(gs[0, 0])   # avg remaining ± σ
#     ax2 = fig.add_subplot(gs[0, 1])   # win rate
#     ax3 = fig.add_subplot(gs[0, 2])   # crossing breakdown
#     ax4 = fig.add_subplot(gs[1, 0])   # diagonal distance
#     ax5 = fig.add_subplot(gs[1, 1])   # volatility (std)
#     ax6 = fig.add_subplot(gs[1, 2])   # risk-adjusted score

#     xlim = (x_vals.min() - 5, x_vals.max() + 5)

#     legend_handles = []

#     for idx, (h_val, df) in enumerate(sweep_results.items()):
#         c   = _PALETTE[idx % len(_PALETTE)]
#         xs  = df["x"].values
#         lbl = f"h = {h_val}"

#         # ── derived series ────────────────────────────────────────────────
#         avg_rem    = df["avg_remaining"].values
#         std_rem    = df["std_remaining"].values
#         win_rate   = df["win_rate"].values * 100
#         thorn_x    = df["avg_thorn_cross"].values
#         fence_x    = df["avg_fence_cross"].values
#         diag_d     = df["avg_diag_dist"].values
#         sharpe     = avg_rem / (std_rem + 1.0)     # Sharpe-like

#         # ── Panel 1 : avg_remaining ± 1σ ─────────────────────────────────
#         ax1.fill_between(xs, avg_rem - std_rem, avg_rem + std_rem,
#                          color=c, alpha=0.15)
#         ax1.plot(xs, avg_rem, color=c, linewidth=1.8, label=lbl)
#         _annotate_peak(ax1, xs, avg_rem, c)

#         # ── Panel 2 : win rate ────────────────────────────────────────────
#         ax2.plot(xs, win_rate, color=c, linewidth=1.8)
#         _annotate_peak(ax2, xs, win_rate, c)

#         # ── Panel 3 : crossing breakdown (stacked area, single h only) ───
#         # For multi-h we use a line plot with dashes for thorn
#         if len(sweep_results) == 1:
#             # Stacked area: thorn on top of fence
#             ax3.fill_between(xs, 0, fence_x, color=c, alpha=0.35, label="Plain fences")
#             ax3.fill_between(xs, fence_x, fence_x + thorn_x,
#                              color=_ZERO, alpha=0.40, label="Thorn fences")
#             ax3.plot(xs, fence_x,             color=c,    linewidth=1.4)
#             ax3.plot(xs, fence_x + thorn_x,   color=_ZERO,linewidth=1.4)
#         else:
#             ax3.plot(xs, thorn_x, color=c, linewidth=1.6, linestyle="--",
#                      label=f"{lbl} · thorn")
#             ax3.plot(xs, fence_x, color=c, linewidth=1.6, linestyle="-",
#                      label=f"{lbl} · plain")

#         # ── Panel 4 : diagonal distance ───────────────────────────────────
#         ax4.plot(xs, diag_d, color=c, linewidth=1.8)

#         # ── Panel 5 : volatility (std) ────────────────────────────────────
#         ax5.plot(xs, std_rem, color=c, linewidth=1.8)

#         # ── Panel 6 : risk-adjusted score ─────────────────────────────────
#         ax6.plot(xs, sharpe, color=c, linewidth=1.8)
#         _annotate_peak(ax6, xs, sharpe, c)

#         legend_handles.append(Line2D([0], [0], color=c, linewidth=2, label=lbl))

#     # ── break-even line on panel 1 ────────────────────────────────────────
#     ax1.axhline(0, color=_ZERO, linewidth=0.9, linestyle="--", alpha=0.8,
#                 label="break-even")

#     # ── style every panel ─────────────────────────────────────────────────
#     _style_ax(ax1, "① Avg Remaining Apples  (± 1 σ band)",
#               "x  (thorns budget)", "Apples remaining", xlim)
#     _style_ax(ax2, "② Win Rate  — P(remaining ≥ 0)",
#               "x  (thorns budget)", "Win rate (%)", xlim)
#     _style_ax(ax3,
#               "③ Fence Crossings Breakdown"
#               + ("  [plain = solid | thorn = dashed]"
#                  if len(sweep_results) > 1 else "  [stacked]"),
#               "x  (thorns budget)", "Avg fences crossed per walk", xlim)
#     _style_ax(ax4, "④ Avg Diagonal Distance  (|cx − cy| / √2)",
#               "x  (thorns budget)", "Avg dist from diagonal", xlim)
#     _style_ax(ax5, "⑤ Volatility  (std of remaining apples)",
#               "x  (thorns budget)", "Std (apples)", xlim)
#     _style_ax(ax6, "⑥ Risk-Adjusted Score  (avg / (std + 1))",
#               "x  (thorns budget)", "Score (higher = better)", xlim)

#     # ── panel-specific legends ────────────────────────────────────────────
#     ax1.legend(handles=legend_handles + [
#                    Line2D([0], [0], color=_ZERO, linewidth=1,
#                           linestyle="--", label="break-even")],
#                facecolor=_PANEL, edgecolor=_GRID_C,
#                labelcolor=_TEXT, fontsize=8, loc="best")

#     if len(sweep_results) == 1:
#         # stacked area legend for panel 3
#         ax3.legend(handles=[
#                        Patch(facecolor=_PALETTE[0], alpha=0.5, label="Plain fences"),
#                        Patch(facecolor=_ZERO,       alpha=0.5, label="Thorn fences")],
#                    facecolor=_PANEL, edgecolor=_GRID_C,
#                    labelcolor=_TEXT, fontsize=8, loc="best")
#     else:
#         ax3.legend(facecolor=_PANEL, edgecolor=_GRID_C,
#                    labelcolor=_TEXT, fontsize=7, loc="best")

#     for ax in (ax2, ax4, ax5, ax6):
#         ax.legend(handles=legend_handles, facecolor=_PANEL, edgecolor=_GRID_C,
#                   labelcolor=_TEXT, fontsize=8, loc="best")

#     # ── footnote ──────────────────────────────────────────────────────────
#     foot = (f"n_sims={args.n_sims} × seeds={args.seeds}  "
#             f"({args.n_sims * args.seeds:,} iterations per h) · "
#             f"x step={args.x_step} · seed={args.seed}")
#     fig.text(0.5, 0.012, foot, ha="center", color=_MUTED, fontsize=7.5)

#     return fig


# # ── Insight summary ───────────────────────────────────────────────────────────

# def print_summary(sweep_results: dict[int, pd.DataFrame]):
#     print()
#     print("═" * 72)
#     print("  SWEEP SUMMARY")
#     print("═" * 72)
#     print(f"  {'h':>4}  {'best_x':>6}  {'best_avg':>10}  "
#           f"{'best_win%':>9}  {'best_sharpe_x':>14}  {'sharpe':>8}")
#     print("  " + "─" * 68)

#     for h_val, df in sweep_results.items():
#         best_avg_row   = df.loc[df["avg_remaining"].idxmax()]
#         sharpe         = df["avg_remaining"] / (df["std_remaining"] + 1.0)
#         best_sharp_row = df.loc[sharpe.idxmax()]

#         print(f"  {h_val:>4}  "
#               f"{int(best_avg_row['x']):>6}  "
#               f"{best_avg_row['avg_remaining']:>10.2f}  "
#               f"{best_avg_row['win_rate']*100:>8.1f}%  "
#               f"{int(best_sharp_row['x']):>14}  "
#               f"{sharpe.max():>8.3f}")

#     print("═" * 72)
#     print()
#     print("  Columns:")
#     print("    best_x        – x with highest avg remaining apples")
#     print("    best_avg      – avg remaining apples at that x")
#     print("    best_win%     – win rate at that x")
#     print("    best_sharpe_x – x with highest risk-adjusted score")
#     print("    sharpe        – risk-adjusted score  = avg / (std + 1)")
#     print("═" * 72)
#     print()


# # ── Entry point ───────────────────────────────────────────────────────────────

# def main():
#     args = parse_args()

#     # Time-based seed if not supplied
#     if args.seed is None:
#         args.seed = int(time.time() * 1000) % (2**32 - 1)

#     # Validate
#     bad_h = [h for h in args.h if not (H_MIN <= h <= H_MAX)]
#     if bad_h:
#         print(f"[ERROR] h values {bad_h} out of range [{H_MIN}, {H_MAX}]",
#               file=sys.stderr)
#         sys.exit(1)

#     x_vals = np.arange(
#         max(args.x_min, X_MIN),
#         min(args.x_max, X_MAX) + 1,
#         args.x_step,
#         dtype=int,
#     )
#     if len(x_vals) < 2:
#         print("[ERROR] x sweep produces fewer than 2 points — reduce --x-step "
#               "or widen --x-min / --x-max", file=sys.stderr)
#         sys.exit(1)

#     print()
#     print("═" * 60)
#     print("  Kyber Analysis  —  X sweep")
#     print("═" * 60)
#     print(f"  h values    : {args.h}")
#     print(f"  x range     : {x_vals[0]} → {x_vals[-1]}  (step {args.x_step})")
#     print(f"  x points    : {len(x_vals)}")
#     print(f"  n_sims      : {args.n_sims}")
#     print(f"  seeds       : {args.seeds}")
#     print(f"  master seed : {args.seed}")
#     print(f"  iterations  : {args.n_sims * args.seeds:,} per h")
#     print()

#     # ── Run sweeps ────────────────────────────────────────────────────────
#     sweep_results: dict[int, pd.DataFrame] = {}
#     for h_val in args.h:
#         # Use a deterministic but distinct seed per h so runs are reproducible
#         # yet independent
#         h_seed = (args.seed + h_val * 7919) % (2**32 - 1)
#         sweep_results[h_val] = run_sweep(
#             h_val, x_vals, args.n_sims, args.seeds, h_seed)

#     print()
#     print_summary(sweep_results)

#     # ── Build & save plot ─────────────────────────────────────────────────
#     fig = build_plot(sweep_results, x_vals, args)

#     out_path = Path(args.out)
#     fig.savefig(out_path, dpi=args.dpi, facecolor=_BG, bbox_inches="tight")
#     print(f"  Plot saved →  {out_path.resolve()}")

#     if not args.no_show:
#         plt.show()


# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
"""
kyber_analysis.py
═══════════════════════════════════════════════════════════════════════════════
Sweeps x over [x_min, x_max] for one or more fixed h values, runs the full
crowd-penalised Kyber simulation, and produces a 6-panel diagnostic plot.

When multiple h values are supplied ALL (x, h) submissions are placed into
a SINGLE competition so they share the same crowd-penalty field.  The results
are split by h afterwards for plotting.

Usage — single h:
    python kyber_analysis.py --h 20

Usage — compare multiple h values (all compete together):
    python kyber_analysis.py --h 10 20 30 40 --x-step 10 --seeds 5

Options:
    --h          H value(s) to analyse  (space-separated)  [20]
    --x-min      Minimum x in sweep                        [0]
    --x-max      Maximum x in sweep                        [300]
    --x-step     Step size for x sweep                     [10]
    --n-sims     Simulation iterations per seed            [50]
    --seeds      Independent seeds to average over         [5]
    --seed       Master RNG seed  (None = time-based)      [None]
    --out        Output file path (.png / .pdf / .svg)     [kyber_analysis.png]
    --no-show    Don't open the plot interactively
    --dpi        Output resolution                         [150]

Panels produced
───────────────
  1. Avg Remaining Apples ± 1σ  vs  X   ← primary metric
  2. Win Rate (%)               vs  X
  3. Fence-crossing breakdown   vs  X   (thorn vs plain, stacked area)
  4. Avg Diagonal Distance      vs  X
  5. Volatility (std_remaining) vs  X
  6. Risk-adjusted score        vs  X   (avg_rem / (std_rem + 1), Sharpe-like)
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
from matplotlib.patches import Patch

# ── import simulator ──────────────────────────────────────────────────────────
try:
    from kyber_simulate import simulate_competition, X_MIN, X_MAX, H_MIN, H_MAX
except ImportError:
    print("[ERROR] kyber_simulate.py not found in the current directory.", file=sys.stderr)
    sys.exit(1)


# ── Colour palette ────────────────────────────────────────────────────────────
_PALETTE = [
    "#4FC3F7",   # sky blue
    "#FF8A65",   # soft orange
    "#81C784",   # sage green
    "#CE93D8",   # lavender
    "#FFD54F",   # amber
    "#F06292",   # rose
]

_BG      = "#0F1117"
_PANEL   = "#181C27"
_GRID_C  = "#2A2F3D"
_TEXT    = "#E0E6F0"
_MUTED   = "#6B7A99"
_ZERO    = "#FF6B6B"


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Kyber x-sweep analysis with diagnostic plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--h",      type=int, nargs="+", default=[20],
                   help="H value(s) to analyse  (space-separated)  [20]")
    p.add_argument("--x-min",  type=int, default=X_MIN,   dest="x_min",
                   help=f"Min x  [{X_MIN}]")
    p.add_argument("--x-max",  type=int, default=X_MAX,   dest="x_max",
                   help=f"Max x  [{X_MAX}]")
    p.add_argument("--x-step", type=int, default=10,      dest="x_step",
                   help="Step size for x sweep  [10]")
    p.add_argument("--n-sims", type=int, default=50,      dest="n_sims",
                   help="Iterations per seed  [50]")
    p.add_argument("--seeds",  type=int, default=5,
                   help="Independent seeds  [5]")
    p.add_argument("--seed",   type=int, default=None,
                   help="Master RNG seed  [time-based]")
    p.add_argument("--out",    type=str, default="kyber_analysis.png",
                   help="Output file path  [kyber_analysis.png]")
    p.add_argument("--no-show", action="store_true", dest="no_show",
                   help="Don't open the plot interactively")
    p.add_argument("--dpi",    type=int, default=150,
                   help="Output resolution  [150]")
    return p.parse_args()


# ── Simulation driver ─────────────────────────────────────────────────────────

def run_combined_sweep(
    h_vals: list[int],
    x_vals: np.ndarray,
    n_sims: int,
    seeds: int,
    master_seed: int,
) -> dict[int, pd.DataFrame]:
    """
    Build ONE submission per (x, h) pair, run a SINGLE simulate_competition
    call so every submission shares the same crowd-penalty field, then split
    the returned DataFrame by h value and return {h: df} sorted by x.
    """
    submissions = [
        {"x": int(x), "h": int(h), "label": f"x{int(x)}_h{int(h)}"}
        for h in h_vals
        for x in x_vals
    ]

    total = len(submissions)
    print(f"  Running ONE combined competition:  "
          f"{total} submissions  ({n_sims * seeds:,} iterations each) …",
          end="", flush=True)

    df = simulate_competition(
        submissions = submissions,
        n_sims      = n_sims,
        n_seeds     = seeds,
        master_seed = master_seed,
    )
    print("  done.")

    # Split by h and sort each slice by x
    results: dict[int, pd.DataFrame] = {}
    for h_val in h_vals:
        slice_df = (
            df[df["h"] == h_val]
            .copy()
            .sort_values("x")
            .reset_index(drop=True)
        )
        slice_df["h_fixed"] = h_val
        results[h_val] = slice_df

    return results


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _style_ax(ax, title: str, xlabel: str, ylabel: str, xlim=None):
    ax.set_facecolor(_PANEL)
    ax.set_title(title, color=_TEXT, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, color=_MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=_MUTED, fontsize=9)
    ax.tick_params(colors=_MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(_GRID_C)
    ax.grid(True, color=_GRID_C, linewidth=0.6, linestyle="--", alpha=0.7)
    if xlim:
        ax.set_xlim(xlim)


def _annotate_peak(ax, xs, ys, color):
    best_idx = np.argmax(ys)
    bx, by   = xs[best_idx], ys[best_idx]
    ax.axvline(bx, color=color, linewidth=0.8, linestyle=":", alpha=0.6)
    ax.annotate(f"x={bx}", xy=(bx, by),
                xytext=(4, -14), textcoords="offset points",
                color=color, fontsize=7.5, alpha=0.9)


# ── Main plot builder ─────────────────────────────────────────────────────────

def build_plot(sweep_results: dict[int, pd.DataFrame],
               x_vals: np.ndarray, args: argparse.Namespace):
    matplotlib.rcParams["font.family"] = "monospace"

    fig = plt.figure(figsize=(18, 10), facecolor=_BG)
    fig.suptitle(
        "Kyber Challenge  ·  X-sweep Analysis"
        + (f"   (h = {args.h[0]})" if len(args.h) == 1
           else "   (multi-h · single combined competition)"),
        color=_TEXT, fontsize=15, fontweight="bold", y=0.97
    )

    gs = GridSpec(2, 3, figure=fig,
                  hspace=0.42, wspace=0.32,
                  left=0.07, right=0.97, top=0.91, bottom=0.09)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    xlim = (x_vals.min() - 5, x_vals.max() + 5)
    legend_handles = []

    for idx, (h_val, df) in enumerate(sweep_results.items()):
        c   = _PALETTE[idx % len(_PALETTE)]
        xs  = df["x"].values
        lbl = f"h = {h_val}"

        avg_rem  = df["avg_remaining"].values
        std_rem  = df["std_remaining"].values
        win_rate = df["win_rate"].values * 100
        thorn_x  = df["avg_thorn_cross"].values
        fence_x  = df["avg_fence_cross"].values
        diag_d   = df["avg_diag_dist"].values
        sharpe   = avg_rem / (std_rem + 1.0)

        # Panel 1 — avg remaining ± 1σ
        ax1.fill_between(xs, avg_rem - std_rem, avg_rem + std_rem,
                         color=c, alpha=0.15)
        ax1.plot(xs, avg_rem, color=c, linewidth=1.8, label=lbl)
        _annotate_peak(ax1, xs, avg_rem, c)

        # Panel 2 — win rate
        ax2.plot(xs, win_rate, color=c, linewidth=1.8)
        _annotate_peak(ax2, xs, win_rate, c)

        # Panel 3 — crossing breakdown
        if len(sweep_results) == 1:
            ax3.fill_between(xs, 0, fence_x, color=c, alpha=0.35, label="Plain fences")
            ax3.fill_between(xs, fence_x, fence_x + thorn_x,
                             color=_ZERO, alpha=0.40, label="Thorn fences")
            ax3.plot(xs, fence_x,           color=c,    linewidth=1.4)
            ax3.plot(xs, fence_x + thorn_x, color=_ZERO,linewidth=1.4)
        else:
            ax3.plot(xs, thorn_x, color=c, linewidth=1.6, linestyle="--",
                     label=f"{lbl} · thorn")
            ax3.plot(xs, fence_x, color=c, linewidth=1.6, linestyle="-",
                     label=f"{lbl} · plain")

        # Panel 4 — diagonal distance
        ax4.plot(xs, diag_d, color=c, linewidth=1.8)

        # Panel 5 — volatility
        ax5.plot(xs, std_rem, color=c, linewidth=1.8)

        # Panel 6 — risk-adjusted score
        ax6.plot(xs, sharpe, color=c, linewidth=1.8)
        _annotate_peak(ax6, xs, sharpe, c)

        legend_handles.append(Line2D([0], [0], color=c, linewidth=2, label=lbl))

    ax1.axhline(0, color=_ZERO, linewidth=0.9, linestyle="--", alpha=0.8,
                label="break-even")

    _style_ax(ax1, "① Avg Remaining Apples  (± 1 σ band)",
              "x  (thorns budget)", "Apples remaining", xlim)
    _style_ax(ax2, "② Win Rate  — P(remaining ≥ 0)",
              "x  (thorns budget)", "Win rate (%)", xlim)
    _style_ax(ax3,
              "③ Fence Crossings Breakdown"
              + ("  [plain = solid | thorn = dashed]"
                 if len(sweep_results) > 1 else "  [stacked]"),
              "x  (thorns budget)", "Avg fences crossed per walk", xlim)
    _style_ax(ax4, "④ Avg Diagonal Distance  (|cx − cy| / √2)",
              "x  (thorns budget)", "Avg dist from diagonal", xlim)
    _style_ax(ax5, "⑤ Volatility  (std of remaining apples)",
              "x  (thorns budget)", "Std (apples)", xlim)
    _style_ax(ax6, "⑥ Risk-Adjusted Score  (avg / (std + 1))",
              "x  (thorns budget)", "Score (higher = better)", xlim)

    ax1.legend(handles=legend_handles + [
                   Line2D([0], [0], color=_ZERO, linewidth=1,
                          linestyle="--", label="break-even")],
               facecolor=_PANEL, edgecolor=_GRID_C,
               labelcolor=_TEXT, fontsize=8, loc="best")

    if len(sweep_results) == 1:
        ax3.legend(handles=[
                       Patch(facecolor=_PALETTE[0], alpha=0.5, label="Plain fences"),
                       Patch(facecolor=_ZERO,       alpha=0.5, label="Thorn fences")],
                   facecolor=_PANEL, edgecolor=_GRID_C,
                   labelcolor=_TEXT, fontsize=8, loc="best")
    else:
        ax3.legend(facecolor=_PANEL, edgecolor=_GRID_C,
                   labelcolor=_TEXT, fontsize=7, loc="best")

    for ax in (ax2, ax4, ax5, ax6):
        ax.legend(handles=legend_handles, facecolor=_PANEL, edgecolor=_GRID_C,
                  labelcolor=_TEXT, fontsize=8, loc="best")

    foot = (f"n_sims={args.n_sims} × seeds={args.seeds}  "
            f"({args.n_sims * args.seeds:,} iterations per submission) · "
            f"x step={args.x_step} · seed={args.seed} · "
            f"{'1 combined competition' if len(args.h) > 1 else '1 competition'}")
    fig.text(0.5, 0.012, foot, ha="center", color=_MUTED, fontsize=7.5)

    return fig


# ── Insight summary ───────────────────────────────────────────────────────────

def print_summary(sweep_results: dict[int, pd.DataFrame]):
    print()
    print("═" * 72)
    print("  SWEEP SUMMARY  (all h values competed together)")
    print("═" * 72)
    print(f"  {'h':>4}  {'best_x':>6}  {'best_avg':>10}  "
          f"{'best_win%':>9}  {'best_sharpe_x':>14}  {'sharpe':>8}")
    print("  " + "─" * 68)

    for h_val, df in sweep_results.items():
        best_avg_row   = df.loc[df["avg_remaining"].idxmax()]
        sharpe         = df["avg_remaining"] / (df["std_remaining"] + 1.0)
        best_sharp_row = df.loc[sharpe.idxmax()]

        print(f"  {h_val:>4}  "
              f"{int(best_avg_row['x']):>6}  "
              f"{best_avg_row['avg_remaining']:>10.2f}  "
              f"{best_avg_row['win_rate']*100:>8.1f}%  "
              f"{int(best_sharp_row['x']):>14}  "
              f"{sharpe.max():>8.3f}")

    print("═" * 72)
    print()
    print("  Columns:")
    print("    best_x        – x with highest avg remaining apples")
    print("    best_avg      – avg remaining apples at that x")
    print("    best_win%     – win rate at that x")
    print("    best_sharpe_x – x with highest risk-adjusted score")
    print("    sharpe        – risk-adjusted score  = avg / (std + 1)")
    print("═" * 72)
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.seed is None:
        args.seed = int(time.time() * 1000) % (2**32 - 1)

    bad_h = [h for h in args.h if not (H_MIN <= h <= H_MAX)]
    if bad_h:
        print(f"[ERROR] h values {bad_h} out of range [{H_MIN}, {H_MAX}]",
              file=sys.stderr)
        sys.exit(1)

    x_vals = np.arange(
        max(args.x_min, X_MIN),
        min(args.x_max, X_MAX) + 1,
        args.x_step,
        dtype=int,
    )
    if len(x_vals) < 2:
        print("[ERROR] x sweep produces fewer than 2 points — reduce --x-step "
              "or widen --x-min / --x-max", file=sys.stderr)
        sys.exit(1)

    total_subs = len(args.h) * len(x_vals)

    print()
    print("═" * 60)
    print("  Kyber Analysis  —  X sweep  (single combined competition)")
    print("═" * 60)
    print(f"  h values      : {args.h}")
    print(f"  x range       : {x_vals[0]} → {x_vals[-1]}  (step {args.x_step})")
    print(f"  x points      : {len(x_vals)}")
    print(f"  total subs    : {total_subs}  ({len(args.h)} h × {len(x_vals)} x)")
    print(f"  n_sims        : {args.n_sims}")
    print(f"  seeds         : {args.seeds}")
    print(f"  master seed   : {args.seed}")
    print(f"  iterations    : {args.n_sims * args.seeds:,} per submission")
    print()

    # ── Single combined run ───────────────────────────────────────────────
    sweep_results = run_combined_sweep(
        h_vals      = args.h,
        x_vals      = x_vals,
        n_sims      = args.n_sims,
        seeds       = args.seeds,
        master_seed = args.seed,
    )

    print()
    print_summary(sweep_results)

    # ── Build & save plot ─────────────────────────────────────────────────
    fig = build_plot(sweep_results, x_vals, args)

    out_path = Path(args.out)
    fig.savefig(out_path, dpi=args.dpi, facecolor=_BG, bbox_inches="tight")
    print(f"  Plot saved →  {out_path.resolve()}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()