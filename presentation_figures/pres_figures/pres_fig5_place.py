#!/usr/bin/env python3
"""
Pres Fig 5: PLACE — Uncertainty Quantification
3 panels:
  (A) Methodology flowchart showing the PLACE pipeline
  (B) Uncertainty scatter: noise_var vs residual_std per dataset
  (C) Grouped bar chart of uncertainty metrics by dataset
"""
import sys, json, os
sys.path.insert(0, '/home/bcheng/sequence_optimization/FUSEMAP/presentation_figures/pres_figures')
from pres_style import *
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.patches import Patch

apply_pres_style()

BASE = '/home/bcheng/sequence_optimization/FUSEMAP'

# ── LOAD PLACE METADATA ─────────────────────────────────────────────────────
place_dir = f'{BASE}/cadence_place'
place_data = {}
for d in sorted(os.listdir(place_dir)):
    mp = os.path.join(place_dir, d, 'place_metadata.json')
    if os.path.exists(mp):
        with open(mp) as f:
            m = json.load(f)
        ps = m.get('place_stats', {})
        if '_all_' not in d and 'config' not in d:
            place_data[d] = ps

print(f"Loaded PLACE data for {len(place_data)} models")
for k, v in place_data.items():
    print(f"  {k}: noise_var={v.get('noise_var',0):.4f}, resid_std={v.get('residual_std',0):.4f}")

# ── SHARED DATASET CONFIGURATION ────────────────────────────────────────────
dataset_display = {
    'cadence_k562_v2':        ('K562',      COLORS['human']),
    'cadence_hepg2_v2':       ('HepG2',     COLORS['human']),
    'cadence_wtc11_v2':       ('WTC11',     COLORS['human']),
    'cadence_deepstarr_v2':   ('DeepSTARR', COLORS['drosophila']),
    'cadence_arabidopsis_v1': ('Arabid.',   COLORS['plant']),
    'cadence_maize_v1':       ('Maize',     COLORS['plant']),
    'cadence_sorghum_v1':     ('Sorghum',   COLORS['plant']),
    'cadence_yeast_v1':       ('Yeast',     COLORS['yeast']),
}

display_order = [
    'cadence_k562_v2', 'cadence_hepg2_v2', 'cadence_wtc11_v2',
    'cadence_deepstarr_v2', 'cadence_arabidopsis_v1', 'cadence_maize_v1',
    'cadence_sorghum_v1', 'cadence_yeast_v1',
]

# ── FIGURE ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 6.5))
gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 0.9, 0.9], wspace=0.30)


# ══════════════════════════════════════════════════════════════════════════════
# Panel A: PLACE Methodology Flowchart
# ══════════════════════════════════════════════════════════════════════════════
ax_a = fig.add_subplot(gs[0])
ax_a.axis('off')
ax_a.set_xlim(-0.5, 11)
ax_a.set_ylim(-0.2, 11.2)

# ---- Helper: draw a rounded box with centered text ----
def draw_box(ax, cx, cy, w, h, text, facecolor, edgecolor,
             fontsize=None, fontweight='bold', textcolor=None,
             alpha=1.0, zorder=3):
    if fontsize is None:
        fontsize = FONTS['flowchart']
    if textcolor is None:
        textcolor = COLORS['text']
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle='round,pad=0.15', facecolor=facecolor,
        edgecolor=edgecolor, linewidth=1.4, alpha=alpha, zorder=zorder,
    )
    ax.add_patch(box)
    ax.text(cx, cy, text, fontsize=fontsize, fontweight=fontweight,
            ha='center', va='center', color=textcolor, zorder=zorder + 1)

def draw_arrow(ax, x_start, y_start, x_end, y_end, color=COLORS['text_light'],
               lw=1.2):
    ax.annotate(
        '', xy=(x_end, y_end), xytext=(x_start, y_start),
        arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                        connectionstyle='arc3,rad=0'), zorder=2,
    )

# ---- Layout constants ----
LX = 2.8       # left branch center x
RX = 7.5       # right branch center x
MX = 5.2       # merge center x
BW = 3.8       # box width
BH = 0.85      # box height

col_laplace   = '#E8EAF6'
col_conformal = '#E3F2FD'
col_merge     = '#F3E5F5'
col_final     = '#EDE7F6'

edge_laplace   = COLORS['accent1']
edge_conformal = COLORS['blue1']
edge_merge     = COLORS['accent2']
edge_final     = COLORS['primary']

# ---- Branch labels ----
y_label = 10.8
ax_a.text(LX, y_label, 'Laplace Branch', fontsize=FONTS['flowchart'],
          ha='center', va='top', color=edge_laplace, fontstyle='italic')
ax_a.text(RX, y_label, 'Conformal Branch', fontsize=FONTS['flowchart'],
          ha='center', va='top', color=edge_conformal, fontstyle='italic')

# ---- Row 1: Two starting boxes ----
y1 = 9.7
draw_box(ax_a, LX, y1, BW, BH,
         'Trained CADENCE\n' + r'weights $\theta^*$',
         col_laplace, edge_laplace, fontsize=FONTS['flowchart_detail'])
draw_box(ax_a, RX, y1, BW, BH,
         'Calibration Set\n(held-out data)',
         col_conformal, edge_conformal, fontsize=FONTS['flowchart_detail'])

# ---- Row 2 ----
y2 = 8.2
draw_box(ax_a, LX, y2, BW, BH,
         'Last-Layer Hessian\n' + r'$H = \nabla^2 L$',
         col_laplace, edge_laplace, fontsize=FONTS['flowchart_detail'])
draw_box(ax_a, RX, y2, BW, BH,
         'Feature Extraction\n' + r'$\varphi(x)$ penultimate',
         col_conformal, edge_conformal, fontsize=FONTS['flowchart_detail'])

draw_arrow(ax_a, LX, y1 - BH/2, LX, y2 + BH/2, color=edge_laplace)
draw_arrow(ax_a, RX, y1 - BH/2, RX, y2 + BH/2, color=edge_conformal)

# ---- Row 3 ----
y3 = 6.7
draw_box(ax_a, LX, y3, BW, BH,
         r'Epistemic $\sigma(x)$' + '\nLaplace posterior',
         col_laplace, edge_laplace, fontsize=FONTS['flowchart_detail'])
draw_box(ax_a, RX, y3, BW, BH,
         'KNN (k=200)\nneighbor residuals',
         col_conformal, edge_conformal, fontsize=FONTS['flowchart_detail'])

draw_arrow(ax_a, LX, y2 - BH/2, LX, y3 + BH/2, color=edge_laplace)
draw_arrow(ax_a, RX, y2 - BH/2, RX, y3 + BH/2, color=edge_conformal)

# ---- Row 4 (right branch only) ----
y4 = 5.2
draw_box(ax_a, RX, y4, BW, BH,
         r'Conformal $\hat{q}$' + '\nweighted by distance',
         col_conformal, edge_conformal, fontsize=FONTS['flowchart_detail'])

draw_arrow(ax_a, RX, y3 - BH/2, RX, y4 + BH/2, color=edge_conformal)

# ---- Merge arrows ----
y_merge = 3.4
merge_w = 6.5

# Left branch down to merge
draw_arrow(ax_a, LX, y3 - BH/2, MX - 1.0, y_merge + BH/2 + 0.05, color=edge_laplace)
# Right branch down to merge
draw_arrow(ax_a, RX, y4 - BH/2, MX + 1.0, y_merge + BH/2 + 0.05, color=edge_conformal)

# ---- Merge box ----
draw_box(ax_a, MX, y_merge, merge_w, BH + 0.1,
         r'Interval: $[\hat{y} - \hat{q}\sigma,\;\hat{y} + \hat{q}\sigma]$',
         col_merge, edge_merge, fontsize=FONTS['flowchart'], fontweight='bold',
         textcolor=COLORS['primary'])

# ---- Final box ----
y_final = 1.8
final_w = 5.5
draw_box(ax_a, MX, y_final, final_w, BH + 0.1,
         r'Coverage Guarantee $\geq$ 90%',
         col_final, edge_final, fontsize=FONTS['flowchart'], fontweight='bold',
         textcolor=COLORS['primary'])

draw_arrow(ax_a, MX, y_merge - (BH + 0.1)/2, MX, y_final + (BH + 0.1)/2,
           color=edge_final, lw=1.5)

ax_a.set_title('PLACE Pipeline', fontsize=FONTS['subtitle'], fontweight='bold',
               pad=12, color=COLORS['text'])
add_panel_label(ax_a, 'A', x=-0.04)


# ══════════════════════════════════════════════════════════════════════════════
# Panel B: Uncertainty scatter — noise_var vs residual_std
# ══════════════════════════════════════════════════════════════════════════════
ax_b = fig.add_subplot(gs[1])

for key in display_order:
    if key not in place_data:
        continue
    name, color = dataset_display[key]
    nv = place_data[key].get('noise_var', 0)
    rs = place_data[key].get('residual_std', 0)
    ns = place_data[key].get('n_samples', 0)

    size = np.clip(ns / 400, 50, 180)

    ax_b.scatter(nv, rs, color=color, s=size * 0.7, alpha=0.9,
                 edgecolor='white', linewidth=0.8, zorder=3)

    # Label placement — tuned to avoid overlap between nearby points
    offset_x, offset_y = 7, 5
    if name == 'K562':
        offset_x, offset_y = -10, -14
    elif name == 'HepG2':
        offset_x, offset_y = 8, 6
    elif name == 'WTC11':
        offset_x, offset_y = 8, -10
    elif name == 'DeepSTARR':
        offset_x, offset_y = -72, -4
    elif name == 'Yeast':
        offset_x, offset_y = 8, -8
    elif name == 'Arabid.':
        offset_x, offset_y = -44, -10
    elif name == 'Sorghum':
        offset_x, offset_y = 8, 6
    elif name == 'Maize':
        offset_x, offset_y = -35, 6

    ax_b.annotate(name, (nv, rs), textcoords='offset points',
                  xytext=(offset_x, offset_y), fontsize=FONTS['bar_label'],
                  fontweight='bold', color=color)

# Diagonal reference line
nv_range = np.linspace(0.25, 1.15, 100)
ax_b.plot(nv_range, np.sqrt(nv_range), color=COLORS['text_light'], ls='--',
          lw=1.0, alpha=0.45, zorder=1, label=r'$\sigma_r = \sqrt{\sigma^2_n}$')

ax_b.set_xlim(0.25, 1.15)
ax_b.set_ylim(0.50, 1.10)

# Coverage annotation
ax_b.text(0.97, 0.04, r'$\alpha = 0.1$ (90% coverage)',
          fontsize=FONTS['annotation'] - 1, ha='right', va='bottom',
          transform=ax_b.transAxes, color=COLORS['accent1'], fontweight='bold',
          bbox=dict(boxstyle='round,pad=0.25', facecolor=COLORS['bg_light'],
                    edgecolor=COLORS['accent1'], linewidth=0.8, alpha=0.9))

style_axis(ax_b, title='Noise Var vs Residual Std (8 Datasets)',
           xlabel='Noise Variance', ylabel='Residual Std')
ax_b.legend(fontsize=FONTS['legend'] - 2, loc='upper left', framealpha=0.9)
add_panel_label(ax_b, 'B', x=-0.12)


# ══════════════════════════════════════════════════════════════════════════════
# Panel C: Grouped bar chart — noise_var and residual_std by dataset
# ══════════════════════════════════════════════════════════════════════════════
ax_c = fig.add_subplot(gs[2])

names_c, noise_vars, resid_stds, colors_c = [], [], [], []
for key in display_order:
    if key in place_data:
        name, color = dataset_display[key]
        names_c.append(name)
        noise_vars.append(place_data[key].get('noise_var', 0))
        resid_stds.append(place_data[key].get('residual_std', 0))
        colors_c.append(color)

x = np.arange(len(names_c))
w = 0.32

bars_nv = ax_c.bar(x - w / 2, noise_vars, w,
                   color=colors_c, alpha=0.35,
                   edgecolor=colors_c, linewidth=0.7,
                   label='Noise Var', zorder=3)
bars_rs = ax_c.bar(x + w / 2, resid_stds, w,
                   color=colors_c, edgecolor='white', linewidth=0.5,
                   label='Residual Std', zorder=3)

# Value labels on residual std bars
for i in range(len(names_c)):
    ax_c.text(x[i] + w / 2, resid_stds[i] + 0.02, f'{resid_stds[i]:.2f}',
              ha='center', va='bottom', fontsize=FONTS['bar_label'] - 2,
              fontweight='bold', color=colors_c[i])

ax_c.set_xticks(x)
ax_c.set_xticklabels(names_c, fontsize=FONTS['tick'] - 2, rotation=40, ha='right')
ax_c.set_ylim(0, 1.25)

leg_patches = [
    Patch(facecolor=COLORS['border'], alpha=0.5, edgecolor=COLORS['accent1'],
          linewidth=0.8, label='Noise Var'),
    Patch(facecolor=COLORS['accent1'], label='Residual Std'),
]
ax_c.legend(handles=leg_patches, fontsize=FONTS['legend'] - 2, loc='upper left',
            framealpha=0.9)

style_axis(ax_c, title='Uncertainty Metrics by Dataset', ylabel='Value')
add_panel_label(ax_c, 'C', x=-0.12)


# ── SAVE ─────────────────────────────────────────────────────────────────────
fig.subplots_adjust(left=0.03, right=0.98, bottom=0.15, top=0.88, wspace=0.30)
save_pres_fig(fig, 'pres_fig5_place')
print('Done.')
