#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║    Kyber Problem Challenge — Enhanced Interactive Visualizer  ║
╚══════════════════════════════════════════════════════════════╝

Run:   python kyber_viz.py
Deps:  numpy  matplotlib

━━━ Controls ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  x-slider      →  extra apples deposited  (thorns = 2x)
  h-slider      →  farmer's cheap-fence threshold
  ▶ Run         →  new seed, re-run 50 sims, full redraw
  ⏵ Step        →  walk the last run one intersection at a time
  ↺ Reset       →  rewind step animation
  [Crowd Pen.]  →  toggle crowd-penalty on/off
  [Thorns]      →  toggle thorn-fence highlighting
  [All Paths]   →  overlay all 50 simulated paths
  [Heatmap]     →  switch grid to path-frequency heatmap
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize, LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe

# ─── Theme ────────────────────────────────────────────────────────────────────
BG       = "#0d0d1a"
PANEL    = "#13132b"
PANEL2   = "#1a1a35"
ACCENT   = "#e94560"
BLUE     = "#4a9eff"
TEAL     = "#00b4d8"
GOLD     = "#ffd166"
GREEN    = "#06d6a0"
PURPLE   = "#c77dff"
ORANGE   = "#ff9a00"
TEXT     = "#dde1f0"
SUBTEXT  = "#7b7fa8"
BORDER   = "#2a2a50"

FENCE_CMAP  = LinearSegmentedColormap.from_list(
    "fence", ["#06d6a0", "#ffd166", "#e94560", "#9d4edd"], N=512)
HEAT_CMAP   = LinearSegmentedColormap.from_list(
    "heat", ["#0d0d1a", "#1e3a5f", "#0096c7", "#ade8f4", "#ffffff"], N=512)
CUMCOST_CLR = "#ffd166"

# ─── Constants ────────────────────────────────────────────────────────────────
GRID  = 50
MIN_T = 5
MAX_T = 30
THORN = 40
NSIMS = 50

# ─── Game Logic ───────────────────────────────────────────────────────────────

def build_fences(x_val: int, rng: np.random.Generator):
    """
    ef[i,j] = east  fence toll  (i,j)→(i+1,j), shape (GRID, GRID+1)
    nf[i,j] = north fence toll  (i,j)→(i,j+1), shape (GRID+1, GRID)
    thorn_mask_e, thorn_mask_n = bool arrays marking which fences got thorns
    """
    ef = rng.integers(MIN_T, MAX_T + 1, (GRID, GRID + 1)).astype(float)
    nf = rng.integers(MIN_T, MAX_T + 1, (GRID + 1, GRID)).astype(float)
    tme = np.zeros_like(ef, dtype=bool)
    tmn = np.zeros_like(nf, dtype=bool)

    n_thorns = 2 * x_val
    if n_thorns > 0:
        idxs = rng.integers(0, ef.size + nf.size, size=n_thorns)
        for idx in idxs:
            if idx < ef.size:
                ef.flat[idx] += THORN
                tme.flat[idx] = True
            else:
                nf.flat[idx - ef.size] += THORN
                tmn.flat[idx - ef.size] = True
    return ef, nf, tme, tmn


def walk_once(ef, nf, h: int, rng: np.random.Generator):
    """Return (path, toll, step_costs, east_use_counts, north_use_counts)."""
    cx, cy   = 0, 0
    path     = [(0, 0)]
    toll     = 0.0
    step_costs = []
    eu = np.zeros_like(ef, dtype=np.int32)
    nu = np.zeros_like(nf, dtype=np.int32)

    while cx < GRID or cy < GRID:
        can_e = cx < GRID
        can_n = cy < GRID

        if can_e and can_n:
            te, tn = ef[cx, cy], nf[cx, cy]
            e_ok, n_ok = te <= h, tn <= h
            if   e_ok and not n_ok: mv = "e"
            elif n_ok and not e_ok: mv = "n"
            else:                   mv = "e" if rng.random() < 0.5 else "n"
        elif can_e: mv = "e"
        else:       mv = "n"

        if mv == "e":
            c = ef[cx, cy]; toll += c; step_costs.append(c); eu[cx, cy] += 1; cx += 1
        else:
            c = nf[cx, cy]; toll += c; step_costs.append(c); nu[cx, cy] += 1; cy += 1
        path.append((cx, cy))

    return path, toll, np.array(step_costs), eu, nu


def apply_crowd_penalty(ef, nf, eu, nu, n_submissions: int = 1):
    """Return (ef_mod, nf_mod) with crowd penalty applied."""
    p_e = eu / max(n_submissions, 1)
    p_n = nu / max(n_submissions, 1)
    return ef * (1 + p_e) ** 3, nf * (1 + p_n) ** 3


def penalised_toll(ef, nf, eu, nu, n_submissions=1):
    ef_m, nf_m = apply_crowd_penalty(ef, nf, eu, nu, n_submissions)
    return float((ef_m * eu).sum() + (nf_m * nu).sum())


def run_batch(x_val, h_val, seed=42):
    """Run NSIMS simulations. Returns dict of arrays + last run field."""
    rng   = np.random.default_rng(seed)
    start = 2500 + x_val

    raw_rem, pen_rem, toll_arr = [], [], []
    all_paths, all_step_costs  = [], []
    # Accumulate path frequency grids
    freq_e = np.zeros((GRID, GRID + 1), dtype=np.int32)
    freq_n = np.zeros((GRID + 1, GRID), dtype=np.int32)

    last = {}
    for i in range(NSIMS):
        ef, nf, tme, tmn = build_fences(x_val, rng)
        path, raw_toll, step_costs, eu, nu = walk_once(ef, nf, h_val, rng)

        pen = penalised_toll(ef, nf, eu, nu, n_submissions=1)
        raw_rem.append(start - raw_toll)
        pen_rem.append(start - pen)
        toll_arr.append(raw_toll)
        all_paths.append(path)
        all_step_costs.append(step_costs)
        freq_e += eu
        freq_n += nu

        if i == NSIMS - 1:
            last = dict(ef=ef, nf=nf, tme=tme, tmn=tmn,
                        path=path, step_costs=step_costs, eu=eu, nu=nu)

    return dict(
        raw_rem    = np.array(raw_rem),
        pen_rem    = np.array(pen_rem),
        tolls      = np.array(toll_arr),
        all_paths  = all_paths,
        all_step_costs = all_step_costs,
        freq_e     = freq_e,
        freq_n     = freq_n,
        start      = start,
        **last,
    )


# ─── Fence segment builders ───────────────────────────────────────────────────

def fence_segments(ef, nf):
    segs, vals = [], []
    for cx in range(GRID):
        for cy in range(GRID + 1):
            segs.append([(cx, cy), (cx + 1, cy)])
            vals.append(ef[cx, cy])
    for cx in range(GRID + 1):
        for cy in range(GRID):
            segs.append([(cx, cy), (cx, cy + 1)])
            vals.append(nf[cx, cy])
    return segs, np.array(vals)


def thorn_segments(ef, nf, tme, tmn):
    """Return only thorn-hit fence segments."""
    segs = []
    for cx in range(GRID):
        for cy in range(GRID + 1):
            if tme[cx, cy]:
                segs.append([(cx, cy), (cx + 1, cy)])
    for cx in range(GRID + 1):
        for cy in range(GRID):
            if tmn[cx, cy]:
                segs.append([(cx, cy), (cx, cy + 1)])
    return segs


def heat_image(freq_e, freq_n):
    """
    Build a (GRID, GRID) image where each cell value = total crossings
    of its four surrounding fences.
    """
    img = np.zeros((GRID, GRID))
    for cy in range(GRID):
        for cx in range(GRID):
            v  = freq_e[cx, cy] + freq_e[cx, cy + 1]   # south & north edge (east fences)
            v += freq_n[cx, cy] + freq_n[cx + 1, cy]   # west  & east edge  (north fences)
            img[cy, cx] = v
    return img


# ─── Visualizer ───────────────────────────────────────────────────────────────

class KyberViz:

    # toggle states
    _show_penalty  = True
    _show_thorns   = True
    _show_all_paths= False
    _show_heatmap  = False

    def __init__(self):
        self.x_val = 100
        self.h_val = 20
        self.seed  = 42
        self._anim_step = 0
        self._data  = None

        self._apply_theme()
        self._build_layout()
        self._run_and_draw()
        plt.show()

    # ── Theme ─────────────────────────────────────────────────────────────────
    def _apply_theme(self):
        matplotlib.rcParams.update({
            "figure.facecolor" : BG,
            "axes.facecolor"   : PANEL,
            "axes.edgecolor"   : BORDER,
            "axes.labelcolor"  : TEXT,
            "xtick.color"      : SUBTEXT,
            "ytick.color"      : SUBTEXT,
            "text.color"       : TEXT,
            "grid.color"       : "#22224a",
            "grid.linestyle"   : "--",
            "grid.alpha"       : 0.45,
        })

    # ── Layout ────────────────────────────────────────────────────────────────
    def _build_layout(self):
        self.fig = plt.figure(figsize=(22, 12))
        self.fig.patch.set_facecolor(BG)

        # Main grid (left) + 4 right panels stacked
        gs_outer = gridspec.GridSpec(
            1, 2,
            figure=self.fig,
            left=0.03, right=0.98,
            top=0.89, bottom=0.16,
            wspace=0.28,
            width_ratios=[2.2, 1],
        )
        # Right column: 4 stacked panels
        gs_right = gridspec.GridSpecFromSubplotSpec(
            4, 1, subplot_spec=gs_outer[1], hspace=0.55)

        self.ax_grid  = self.fig.add_subplot(gs_outer[0])
        self.ax_hist  = self.fig.add_subplot(gs_right[0])   # remaining apples hist
        self.ax_bar   = self.fig.add_subplot(gs_right[1])   # toll per sim
        self.ax_cum   = self.fig.add_subplot(gs_right[2])   # cumulative cost curve
        self.ax_stat  = self.fig.add_subplot(gs_right[3])   # stats text

        self.ax_grid.set_facecolor(PANEL)
        self.ax_grid.set_aspect("equal")
        self.ax_grid.set_xlim(-1, GRID + 1)
        self.ax_grid.set_ylim(-1, GRID + 1)

        # Dedicated colorbar axis (reused on redraw)
        div = make_axes_locatable(self.ax_grid)
        self.ax_cbar = div.append_axes("right", size="3%", pad=0.06)
        self.ax_cbar.set_facecolor(PANEL)

        # ── Title ─────────────────────────────────────────────────────────────
        self.fig.text(0.5, 0.965,
            "🍎  Kyber Problem Challenge  —  Enhanced Visualizer",
            ha="center", fontsize=18, fontweight="bold",
            color=ACCENT, fontfamily="monospace")
        self.fig.text(0.5, 0.944,
            "Adjust x & h then ▶ Run  |  Use toggles to explore different views",
            ha="center", fontsize=10, color=SUBTEXT)

        # ── Sliders ───────────────────────────────────────────────────────────
        sl_kw = dict(facecolor="#0d0d22")
        ax_sx = self.fig.add_axes([0.05, 0.105, 0.38, 0.025], **sl_kw)
        ax_sh = self.fig.add_axes([0.05, 0.060, 0.38, 0.025], **sl_kw)
        self.sl_x = Slider(ax_sx, "x  [0–300]", 0, 310,
                           valinit=self.x_val, valstep=1, color=ACCENT)
        self.sl_h = Slider(ax_sh, "h  [5–60]",  5,  65,
                           valinit=self.h_val, valstep=1, color=GREEN)
        for sl in (self.sl_x, self.sl_h):
            sl.label.set_color(TEXT);   sl.label.set_fontsize(10)
            sl.valtext.set_color(GOLD); sl.valtext.set_fontsize(10)

        # ── Action buttons ────────────────────────────────────────────────────
        btn_specs = [
            # (left, label, base_color, hover)
            (0.50, "▶  Run",    ACCENT,   "#ff6b6b"),
            (0.61, "⏵  Step",   "#1a3a6e", "#2255aa"),
            (0.72, "↺  Reset",  "#222244", "#3a3a66"),
        ]
        self._btns = {}
        for x0, label, col, hov in btn_specs:
            ax = self.fig.add_axes([x0, 0.062, 0.09, 0.050], facecolor=col)
            b  = Button(ax, label, color=col, hovercolor=hov)
            b.label.set_color("white"); b.label.set_fontweight("bold")
            b.label.set_fontsize(11)
            self._btns[label.strip()] = b

        # ── Toggle buttons ────────────────────────────────────────────────────
        tog_specs = [
            ("Crowd Pen.",  0.50, self._show_penalty),
            ("Thorns",      0.61, self._show_thorns),
            ("All Paths",   0.72, self._show_all_paths),
            ("Heatmap",     0.83, self._show_heatmap),
        ]
        self._tog_axes  = {}
        self._tog_btns  = {}
        self._tog_state = {}
        for name, x0, init in tog_specs:
            col = GREEN if init else "#222244"
            ax  = self.fig.add_axes([x0, 0.010, 0.09, 0.042], facecolor=col)
            b   = Button(ax, f"{'✔' if init else '○'}  {name}",
                         color=col, hovercolor="#2a5a2a" if init else "#333366")
            b.label.set_color("white"); b.label.set_fontsize(9)
            self._tog_axes[name]  = ax
            self._tog_btns[name]  = b
            self._tog_state[name] = init

        # Legend hint
        self.fig.text(0.05, 0.030,
            "Green fence = cheap  ·  Red/Purple = thorn-hit  ·  Blue line = farmer path  "
            "·  Cyan dot = start  ·  Pink star = goal",
            fontsize=8.5, color=SUBTEXT)

        # ── Wire up callbacks ─────────────────────────────────────────────────
        self._btns["▶  Run"].on_clicked(self._on_run)
        self._btns["⏵  Step"].on_clicked(self._on_step)
        self._btns["↺  Reset"].on_clicked(self._on_reset_step)
        for name in self._tog_state:
            # Python closure trick
            self._tog_btns[name].on_clicked(
                lambda _, n=name: self._on_toggle(n))

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_run(self, _):
        self.x_val = int(self.sl_x.val)
        self.h_val = int(self.sl_h.val)
        self.seed  += 1
        self._run_and_draw()

    def _on_step(self, _):
        if self._data is None: return
        max_s = len(self._data["path"])
        if self._anim_step < max_s:
            self._anim_step += 1
            self._draw_grid(self._data, self._anim_step)
            self._draw_cum(self._data, self._anim_step)
            self.fig.canvas.draw_idle()

    def _on_reset_step(self, _):
        self._anim_step = 0
        if self._data:
            self._draw_grid(self._data, 0)
            self._draw_cum(self._data, 0)
            self.fig.canvas.draw_idle()

    def _on_toggle(self, name):
        self._tog_state[name] = not self._tog_state[name]
        on   = self._tog_state[name]
        col  = GREEN if on else "#222244"
        hov  = "#2a5a2a" if on else "#333366"
        icon = "✔" if on else "○"
        ax   = self._tog_axes[name]
        btn  = self._tog_btns[name]
        ax.set_facecolor(col)
        btn.color      = col
        btn.hovercolor = hov
        btn.label.set_text(f"{icon}  {name}")
        if self._data:
            self._draw_grid(self._data, self._anim_step)
            self._draw_hist(self._data)
            self.fig.canvas.draw_idle()

    # ── Core run + draw ───────────────────────────────────────────────────────

    def _run_and_draw(self):
        self._data      = run_batch(self.x_val, self.h_val, self.seed)
        self._anim_step = len(self._data["path"])  # show full path after Run
        self._draw_all(self._data)

    def _draw_all(self, d):
        self._draw_grid(d, self._anim_step)
        self._draw_hist(d)
        self._draw_bars(d)
        self._draw_cum(d, self._anim_step)
        self._draw_stats(d)
        self.fig.canvas.draw_idle()

    # ──────────────────────────────────────────────────────────────────────────
    # PANEL 1: Main Grid
    # ──────────────────────────────────────────────────────────────────────────

    def _draw_grid(self, d, n_steps):
        ax = self.ax_grid
        ax.cla()
        ax.set_facecolor(PANEL)
        ax.set_aspect("equal")
        ax.set_xlim(-1, GRID + 1)
        ax.set_ylim(-1, GRID + 1)
        ax.set_xlabel("→  East",  color=SUBTEXT, fontsize=10)
        ax.set_ylabel("↑  North", color=SUBTEXT, fontsize=10)

        pen_on  = self._tog_state["Crowd Pen."]
        thorn_on= self._tog_state["Thorns"]
        allp_on = self._tog_state["All Paths"]
        heat_on = self._tog_state["Heatmap"]

        mode_str = ("  [HEATMAP]"   if heat_on  else
                    "  [ALL PATHS]" if allp_on  else "")
        pen_str  = "  Crowd Penalty ON" if pen_on else "  Crowd Penalty OFF"
        ax.set_title(
            f"50×50 Apple Field   x={self.x_val}  h={self.h_val}"
            f"   start={d['start']} 🍎   thorns={2*self.x_val}"
            f"{mode_str}{pen_str}",
            color=TEXT, fontsize=10.5, pad=6)

        ef, nf = d["ef"], d["nf"]

        # ── Apply crowd penalty to displayed fence costs ───────────────────
        if pen_on:
            ef_disp, nf_disp = apply_crowd_penalty(ef, nf, d["eu"], d["nu"])
        else:
            ef_disp, nf_disp = ef.copy(), nf.copy()

        thorn_max = MAX_T + 6 * THORN
        norm = Normalize(vmin=MIN_T, vmax=thorn_max)

        if heat_on:
            # ── Path frequency heatmap ─────────────────────────────────────
            img = heat_image(d["freq_e"], d["freq_n"])
            im  = ax.imshow(
                img, origin="lower", aspect="equal",
                extent=[0, GRID, 0, GRID],
                cmap=HEAT_CMAP,
                norm=LogNorm(vmin=max(img.min(), 0.1), vmax=max(img.max(), 1)),
                zorder=2,
            )
            self.ax_cbar.cla()
            self.fig.colorbar(im, cax=self.ax_cbar)
            self.ax_cbar.set_ylabel("Path Visits (log)", color=TEXT, fontsize=8)
            self.ax_cbar.tick_params(colors=SUBTEXT, labelsize=7)
        else:
            # ── Fence color segments ───────────────────────────────────────
            segs, vals = fence_segments(ef_disp, nf_disp)
            lc = LineCollection(
                segs, array=vals, cmap=FENCE_CMAP, norm=norm,
                linewidths=0.75, alpha=0.82, zorder=2)
            ax.add_collection(lc)

            self.ax_cbar.cla()
            sm = plt.cm.ScalarMappable(cmap=FENCE_CMAP, norm=norm)
            sm.set_array([])
            self.fig.colorbar(sm, cax=self.ax_cbar)
            lbl = "Penalised Toll" if pen_on else "Raw Toll"
            self.ax_cbar.set_ylabel(lbl, color=TEXT, fontsize=8)
            self.ax_cbar.tick_params(colors=SUBTEXT, labelsize=7)

            # h-threshold line on colorbar
            if MIN_T <= self.h_val <= thorn_max:
                h_frac = (self.h_val - MIN_T) / (thorn_max - MIN_T)
                self.ax_cbar.axhline(
                    self.h_val, color=GOLD, lw=1.5, ls="--", alpha=0.8)
                self.ax_cbar.text(
                    1.05, self.h_val, f" h={self.h_val}",
                    transform=self.ax_cbar.get_yaxis_transform(),
                    color=GOLD, fontsize=7, va="center")

        # ── Thorn overlay ──────────────────────────────────────────────────
        if thorn_on and not heat_on:
            tsegs = thorn_segments(ef, nf, d["tme"], d["tmn"])
            if tsegs:
                tlc = LineCollection(
                    tsegs, colors=PURPLE, linewidths=1.8,
                    alpha=0.70, zorder=4, linestyle=(0, (2, 2)))
                ax.add_collection(tlc)

        # ── All paths overlay ──────────────────────────────────────────────
        if allp_on:
            for path in d["all_paths"]:
                px = [p[0] for p in path]
                py = [p[1] for p in path]
                ax.plot(px, py, color=BLUE, lw=0.55, alpha=0.18, zorder=5)

        # ── Single animated path ───────────────────────────────────────────
        path = d["path"]
        if n_steps > 0:
            sub = path[:n_steps]
            px  = [p[0] for p in sub]
            py  = [p[1] for p in sub]
            ax.plot(px, py, color="#74b9ff", lw=2.4, alpha=0.95,
                    solid_capstyle="round", solid_joinstyle="round",
                    zorder=6)
            # Animate dot at current position
            ax.scatter([px[-1]], [py[-1]], s=110, color=ACCENT,
                       zorder=10, edgecolors="white", linewidths=0.9)
            # Step counter
            ax.text(1, GRID - 1,
                    f"Step {n_steps-1}/{len(path)-1}",
                    color=GOLD, fontsize=9, fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=0.3", fc=PANEL2, ec=BORDER))

        # Start / end
        ax.scatter([0],    [0],    s=130, color=TEAL,  zorder=9, marker="o",
                   edgecolors="white", linewidths=0.9, label="Start (0,0)")
        ax.scatter([GRID], [GRID], s=200, color=ACCENT, zorder=9, marker="*",
                   edgecolors="white", linewidths=0.9, label="End (50,50)")

        # Compass rose
        for dx, dy, lbl in [(2,0,"E"),(0,2,"N")]:
            ax.annotate("", xy=(-0.5+dx, -0.5+dy), xytext=(-0.5,-0.5),
                        arrowprops=dict(arrowstyle="->",color=SUBTEXT,lw=1.2))
            ax.text(-0.5+dx*1.1, -0.5+dy*1.1, lbl,
                    color=SUBTEXT, fontsize=8, ha="center", va="center")

        legend_els = [
            mpatches.Patch(color=TEAL,   label="Start (0,0)"),
            mpatches.Patch(color=ACCENT, label="End (50,50)"),
            mpatches.Patch(color="#74b9ff", label="Farmer path"),
        ]
        if thorn_on:
            legend_els.append(
                mpatches.Patch(color=PURPLE, label="Thorn fence"))
        if allp_on:
            legend_els.append(
                mpatches.Patch(color=BLUE, alpha=0.5, label="All sim paths"))

        ax.legend(handles=legend_els, fontsize=8, loc="lower right",
                  facecolor=PANEL2, edgecolor=BORDER,
                  labelcolor=TEXT, framealpha=0.88)

    # ──────────────────────────────────────────────────────────────────────────
    # PANEL 2: Remaining Apples Histogram
    # ──────────────────────────────────────────────────────────────────────────

    def _draw_hist(self, d):
        ax = self.ax_hist
        ax.cla(); ax.set_facecolor(PANEL)
        pen_on = self._tog_state["Crowd Pen."]

        ax.set_title("Remaining Apples  (50 sims)", color=TEXT, fontsize=9.5)
        bins = min(22, NSIMS)
        ax.hist(d["raw_rem"], bins=bins, color=BLUE, alpha=0.75,
                edgecolor=BG, label="Raw toll", zorder=3)
        if pen_on:
            ax.hist(d["pen_rem"], bins=bins, color=ACCENT, alpha=0.60,
                    edgecolor=BG, label="Crowd penalty", zorder=4)
        ax.axvline(d["raw_rem"].mean(), color=GOLD, lw=1.8, ls="--",
                   label=f"Raw μ = {d['raw_rem'].mean():.0f}", zorder=5)
        if pen_on:
            ax.axvline(d["pen_rem"].mean(), color="#ff99aa", lw=1.8, ls=":",
                       label=f"Pen μ = {d['pen_rem'].mean():.0f}", zorder=5)
        ax.axvline(0, color="white", lw=0.8, alpha=0.35)
        ax.set_xlabel("Remaining apples", color=SUBTEXT, fontsize=8)
        ax.legend(fontsize=7, facecolor=PANEL2, edgecolor=BORDER,
                  labelcolor=TEXT, framealpha=0.85)
        ax.grid(True, axis="x")

    # ──────────────────────────────────────────────────────────────────────────
    # PANEL 3: Toll per Simulation Bar Chart
    # ──────────────────────────────────────────────────────────────────────────

    def _draw_bars(self, d):
        ax = self.ax_bar
        ax.cla(); ax.set_facecolor(PANEL)
        ax.set_title("Toll Paid per Simulation Run", color=TEXT, fontsize=9.5)
        tolls = d["tolls"]
        mu    = tolls.mean()
        colors = [GREEN if t <= mu else ACCENT for t in tolls]
        ax.bar(range(1, NSIMS + 1), tolls, color=colors, alpha=0.82, width=0.8, zorder=3)
        ax.axhline(mu, color=GOLD, lw=1.5, ls="--",
                   label=f"Mean = {mu:.0f}", zorder=4)
        ax.axhline(d["start"], color=TEAL, lw=1.0, ls=":",
                   label=f"Start = {d['start']}", zorder=4)
        ax.set_xlabel("Sim #",      color=SUBTEXT, fontsize=8)
        ax.set_ylabel("Total toll", color=SUBTEXT, fontsize=8)
        ax.legend(fontsize=7, facecolor=PANEL2, edgecolor=BORDER,
                  labelcolor=TEXT, framealpha=0.85)
        ax.grid(True, axis="y")

    # ──────────────────────────────────────────────────────────────────────────
    # PANEL 4: Cumulative Cost Along Last Path
    # ──────────────────────────────────────────────────────────────────────────

    def _draw_cum(self, d, n_steps):
        ax = self.ax_cum
        ax.cla(); ax.set_facecolor(PANEL)
        ax.set_title("Cumulative Cost Along Path", color=TEXT, fontsize=9.5)

        sc = d["step_costs"]            # cost of each fence crossing
        cumcost = np.cumsum(sc)         # shape (num_steps,)
        steps_x = np.arange(1, len(cumcost) + 1)

        # Light fan: all 50 sim cumulative costs
        for sc2 in d["all_step_costs"]:
            cc2 = np.cumsum(sc2)
            ax.plot(np.arange(1, len(cc2)+1), cc2,
                    color=BLUE, lw=0.4, alpha=0.12, zorder=2)

        # Mean cumulative cost band
        max_len = max(len(s) for s in d["all_step_costs"])
        padded  = [np.pad(np.cumsum(s), (0, max_len - len(s)),
                          mode="edge") for s in d["all_step_costs"]]
        padded  = np.array(padded)
        mu_cc   = padded.mean(axis=0)
        sd_cc   = padded.std(axis=0)
        ax.fill_between(range(1, max_len+1),
                        mu_cc - sd_cc, mu_cc + sd_cc,
                        color=BLUE, alpha=0.12, zorder=3)
        ax.plot(range(1, max_len+1), mu_cc,
                color=BLUE, lw=1.2, ls="--", alpha=0.6,
                label="Mean ±1σ", zorder=4)

        # Last run full curve
        ax.plot(steps_x, cumcost, color=CUMCOST_CLR, lw=1.6,
                alpha=0.85, zorder=5, label="Last run")

        # Animated portion
        if n_steps > 1:
            sub_steps = min(n_steps - 1, len(cumcost))
            ax.plot(steps_x[:sub_steps], cumcost[:sub_steps],
                    color=ACCENT, lw=2.4, zorder=6,
                    label=f"Step {sub_steps}")
            if sub_steps > 0:
                ax.scatter([steps_x[sub_steps-1]], [cumcost[sub_steps-1]],
                           color=ACCENT, s=55, zorder=7, edgecolors="white", lw=0.8)

        # Starting apples line
        ax.axhline(d["start"], color=GREEN, lw=1.0, ls=":",
                   alpha=0.7, label=f"Start={d['start']}")
        ax.set_xlabel("Fence crossings (steps)", color=SUBTEXT, fontsize=8)
        ax.set_ylabel("Cumulative toll",          color=SUBTEXT, fontsize=8)
        ax.legend(fontsize=7, facecolor=PANEL2, edgecolor=BORDER,
                  labelcolor=TEXT, framealpha=0.85)
        ax.grid(True)

    # ──────────────────────────────────────────────────────────────────────────
    # PANEL 5: Statistics Text
    # ──────────────────────────────────────────────────────────────────────────

    def _draw_stats(self, d):
        ax = self.ax_stat
        ax.cla(); ax.axis("off"); ax.set_facecolor(PANEL)
        ax.set_title("Summary Statistics", color=TEXT, fontsize=9.5)

        pen_on = self._tog_state["Crowd Pen."]
        rem    = d["raw_rem"]
        prem   = d["pen_rem"]
        tolls  = d["tolls"]
        win    = (rem >= 0).mean()
        pwin   = (prem >= 0).mean()

        def row(lbl, val, col=TEXT):
            return (lbl, val, col)

        rows = [
            row("━━ Setup ━━",           "",              ACCENT),
            row("  x  /  h",             f"{self.x_val} / {self.h_val}", GOLD),
            row("  Starting apples",     f"{d['start']:,}",              TEXT),
            row("  Thorns placed",       f"{2*self.x_val:,}",            ORANGE),
            row("  Path length",         f"{len(d['path'])-1} steps",    TEXT),
            ("", "", ""),
            row("━━ Raw Toll ━━",        "",              BLUE),
            row("  Mean remaining",      f"{rem.mean():,.1f}",           GREEN if rem.mean()>=0 else ACCENT),
            row("  Std dev",             f"±{rem.std():,.1f}",           SUBTEXT),
            row("  Min / Max",           f"{rem.min():,.0f} / {rem.max():,.0f}", TEXT),
            row("  Win rate (≥0)",       f"{win:.0%}",                   GREEN if win>0.5 else ACCENT),
        ]
        if pen_on:
            rows += [
                ("", "", ""),
                row("━━ After Penalty ━━",    "",       ACCENT),
                row("  Mean remaining",  f"{prem.mean():,.1f}", GREEN if prem.mean()>=0 else ACCENT),
                row("  Win rate (≥0)",   f"{pwin:.0%}",         GREEN if pwin>0.5 else ACCENT),
                row("  Penalty factor",  f"×{prem.mean()/rem.mean():.3f}" if rem.mean()!=0 else "N/A", ORANGE),
            ]
        rows += [
            ("", "", ""),
            row("━━ Toll Stats ━━",          "",   TEAL),
            row("  Mean toll",       f"{tolls.mean():,.1f}", TEXT),
            row("  Std dev",         f"±{tolls.std():,.1f}", SUBTEXT),
        ]

        y = 0.99
        for label, val, color in rows:
            if not label and not val:
                y -= 0.032; continue
            if val == "":
                ax.text(0.02, y, label, transform=ax.transAxes,
                        color=color, fontsize=8.5, fontweight="bold",
                        fontfamily="monospace")
            else:
                ax.text(0.02, y, label, transform=ax.transAxes,
                        color=SUBTEXT, fontsize=8.2)
                ax.text(0.98, y, val,  transform=ax.transAxes,
                        color=color, fontsize=8.2, fontweight="bold", ha="right")
            y -= 0.062


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        KyberViz()
    except KeyboardInterrupt:
        sys.exit(0)