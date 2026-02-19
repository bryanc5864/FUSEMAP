#!/usr/bin/env python3
"""
Pres Fig 8: Integrated FUSEMAP Pipeline
Single large flowchart showing the full pipeline with all modules.
S2A is a prediction model (activity prediction via physics features), not calibration.
"""
import sys
sys.path.insert(0, '/home/bcheng/sequence_optimization/FUSEMAP/presentation_figures/pres_figures')
from pres_style import *
from matplotlib.patches import FancyBboxPatch

apply_pres_style()

# ── FIGURE ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xlim(-0.5, 14)
ax.set_ylim(0.3, 10.5)
ax.axis('off')

# Title
ax.text(6.75, 10.2, 'FUSEMAP Integrated Pipeline',
        fontsize=FONTS['title'] + 4, fontweight='bold',
        ha='center', va='top', color=COLORS['primary'])

# ── LAYOUT CONSTANTS ─────────────────────────────────────────────────────────
Y_INPUT      = 8.5
Y_PREDICTION = 6.7
Y_UNCERT     = 4.9
Y_VALIDATE   = 3.1
Y_OUTPUT     = 1.5

# Horizontal — 4 prediction boxes
X1 = 1.8    # CADENCE
X2 = 4.8    # S2A
X3 = 7.8    # PhysInformer
X4 = 10.8   # TileFormer
X_MID = 6.3  # center for input/output/oracle/place

# Box dimensions
BOX_W  = 2.6
BOX_H  = 1.15
IO_W   = 3.8
IO_H   = 0.95

# ── Module definitions ───────────────────────────────────────────────────────
# (x, y, w, h, label, sublabel, color, metric)
modules = {
    'input':        (X_MID,  Y_INPUT,      IO_W,  IO_H,  'DNA Sequence Input',
                     '4 \u00d7 L one-hot encoded', COLORS['blue3'], ''),
    'cadence':      (X1,     Y_PREDICTION, BOX_W, BOX_H, 'CADENCE',
                     'Activity Prediction', COLORS['blue1'], 'r = 0.96'),
    's2a':          (X2,     Y_PREDICTION, BOX_W, BOX_H, 'S2A',
                     'Physics \u2192 Activity', COLORS['accent3'], r'$\rho$ = 0.70'),
    'physinformer': (X3,     Y_PREDICTION, BOX_W, BOX_H, 'PhysInformer',
                     'Biophysical Features', COLORS['accent1'], '268 features'),
    'tileformer':   (X4,     Y_PREDICTION, BOX_W, BOX_H, 'TileFormer',
                     'Neural Approximator', COLORS['accent2'], r'$R^2$ > 0.95'),
    'place':        (X_MID,  Y_UNCERT,     3.6,   BOX_H, 'PLACE',
                     'Uncertainty Quantification', COLORS['periwinkle'], '90% coverage'),
    'oracle':       (X_MID,  Y_VALIDATE,   3.6,   BOX_H, 'OracleCheck',
                     'Naturality Validation', '#37474F', '83\u201399% pass rate'),
    'output':       (X_MID,  Y_OUTPUT,     5.5,   IO_H,  'Validated Therapeutic Designs',
                     '', COLORS['blue1'], '99% HepG2 specificity'),
}

# ── HELPER: draw a module box ────────────────────────────────────────────────
def draw_module(ax, key, x, y, w, h, label, sublabel, color, metric):
    bbox = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle='round,pad=0.15',
        facecolor=color, edgecolor='white',
        linewidth=1.8, alpha=0.92, zorder=3,
    )
    ax.add_patch(bbox)

    is_io = key in ('input', 'output')
    label_fs = FONTS['subtitle'] + 3 if is_io else FONTS['subtitle'] + 4
    text_y_offset = 0.14 if sublabel else 0.0
    ax.text(x, y + text_y_offset, label, ha='center', va='center',
            fontsize=label_fs, fontweight='bold', color='white', zorder=4)

    if sublabel:
        ax.text(x, y - 0.20, sublabel, ha='center', va='center',
                fontsize=FONTS['annotation'] + 2, color=(1, 1, 1, 0.85),
                fontstyle='italic', zorder=4)

    if metric:
        mx = x
        my = y - h / 2 - 0.15
        ax.text(mx, my, metric, fontsize=FONTS['bar_label'] + 2,
                fontweight='bold', ha='center', va='top', color=color,
                bbox=dict(boxstyle='round,pad=0.12', facecolor='white',
                          edgecolor=color, linewidth=1.0, alpha=0.95),
                zorder=5)


for key, vals in modules.items():
    draw_module(ax, key, *vals)

# ── ARROW HELPERS ─────────────────────────────────────────────────────────────
ARROW_COLOR = COLORS['primary']

def _edge(key, side):
    x, y, w, h = modules[key][:4]
    if side == 'top':    return (x, y + h / 2)
    if side == 'bottom': return (x, y - h / 2)
    if side == 'left':   return (x - w / 2, y)
    if side == 'right':  return (x + w / 2, y)

def solid_arrow(src_key, src_side, dst_key, dst_side, rad=0.0, lw=1.8, shrink=2):
    src = _edge(src_key, src_side)
    dst = _edge(dst_key, dst_side)
    ax.annotate('', xy=dst, xytext=src,
                arrowprops=dict(arrowstyle='-|>', color=ARROW_COLOR, lw=lw,
                                connectionstyle=f'arc3,rad={rad}',
                                shrinkA=shrink, shrinkB=shrink, mutation_scale=14))

def dashed_arrow(src_key, src_side, dst_key, dst_side, rad=0.0, lw=1.2, shrink=2):
    src = _edge(src_key, src_side)
    dst = _edge(dst_key, dst_side)
    ax.annotate('', xy=dst, xytext=src,
                arrowprops=dict(arrowstyle='<|-|>', color=ARROW_COLOR, lw=lw,
                                linestyle='--', connectionstyle=f'arc3,rad={rad}',
                                shrinkA=shrink, shrinkB=shrink, mutation_scale=12,
                                alpha=0.7))

# ── ARROWS ───────────────────────────────────────────────────────────────────

# Input fan-out → sequence-based modules (CADENCE, PhysInformer, TileFormer)
solid_arrow('input', 'bottom', 'cadence',      'top', rad=0.0)
solid_arrow('input', 'bottom', 'physinformer',  'top', rad=0.0)
solid_arrow('input', 'bottom', 'tileformer',    'top', rad=0.0)

# PhysInformer → S2A (S2A uses predicted biophysical features)
solid_arrow('physinformer', 'left', 's2a', 'right', rad=0.0)

# TileFormer → S2A (S2A uses neural-approximated features)
solid_arrow('tileformer', 'left', 's2a', 'right', rad=-0.3)

# PhysInformer ↔ TileFormer (dashed, acceleration)
dashed_arrow('physinformer', 'right', 'tileformer', 'left')

# CADENCE → PLACE
solid_arrow('cadence', 'bottom', 'place', 'top', rad=0.15)

# S2A → PLACE
solid_arrow('s2a', 'bottom', 'place', 'top', rad=0.0)

# PLACE → OracleCheck
solid_arrow('place', 'bottom', 'oracle', 'top')

# OracleCheck → Output
solid_arrow('oracle', 'bottom', 'output', 'top', lw=2.0)

# ── LAYER LABELS (right margin) ──────────────────────────────────────────────
layer_info = [
    (Y_INPUT,      'Input'),
    (Y_PREDICTION, 'Prediction'),
    (Y_UNCERT,     'Uncertainty'),
    (Y_VALIDATE,   'Validation'),
    (Y_OUTPUT,     'Output'),
]
for ly, ltxt in layer_info:
    ax.text(13.5, ly, ltxt, fontsize=FONTS['subtitle'], color=COLORS['text_light'],
            ha='right', va='center', fontstyle='italic')

# ── Shared bracket annotation: >10,000× Speedup spanning PhysInformer + TileFormer
from matplotlib.lines import Line2D
bracket_y_top = Y_PREDICTION + BOX_H / 2 + 0.22
bracket_y_label = bracket_y_top + 0.08
bracket_left = X3 - BOX_W / 2 + 0.05
bracket_right = X4 + BOX_W / 2 - 0.05
bracket_mid = (bracket_left + bracket_right) / 2
bracket_tick = 0.10  # height of the end ticks

# Draw bracket: left tick, horizontal bar, right tick
bracket_color = COLORS['accent1']
for seg in [
    ([bracket_left, bracket_left], [bracket_y_top - bracket_tick, bracket_y_top]),
    ([bracket_left, bracket_right], [bracket_y_top, bracket_y_top]),
    ([bracket_right, bracket_right], [bracket_y_top - bracket_tick, bracket_y_top]),
]:
    ax.add_line(Line2D(seg[0], seg[1], color=bracket_color, lw=2.0, zorder=10))

ax.text(bracket_mid, bracket_y_label, '>10,000\u00d7 Speedup',
        fontsize=FONTS['annotation'] + 2, fontweight='bold',
        ha='center', va='bottom', color=bracket_color, zorder=10,
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.95, pad=2))

# ── SAVE ─────────────────────────────────────────────────────────────────────
save_pres_fig(fig, 'pres_fig8_pipeline')
print('Done.')
