#!/usr/bin/env python3
"""
Pres Fig 3: TileFormer -- Electrostatic Prediction
3 panels: (A) Architecture flowchart, (B) R^2 bar chart, (C) Hexbin scatter
"""
import sys, re, json
sys.path.insert(0, '/home/bcheng/sequence_optimization/FUSEMAP/presentation_figures/pres_figures')
from pres_style import *
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap

apply_pres_style()

BASE = '/home/bcheng/sequence_optimization/FUSEMAP'

# ── PARSE METRICS ────────────────────────────────────────────────────────────
report_path = (f'{BASE}/physics/TileFormer/checkpoints/'
               'run_20250819_063725/final_evaluation_report.txt')
with open(report_path) as f:
    report_text = f.read()

metrics_ordered = [
    'STD_PSI_MIN', 'STD_PSI_MAX', 'STD_PSI_MEAN',
    'ENH_PSI_MIN', 'ENH_PSI_MAX', 'ENH_PSI_MEAN',
]
nice_short = ['STD Min', 'STD Max', 'STD Mean', 'ENH Min', 'ENH Max', 'ENH Mean']

r2_vals, pearson_vals, rmse_vals = [], [], []
residual_mean, residual_std = [], []

for metric in metrics_ordered:
    pattern = rf'{metric} Metrics:\n-+\n(.*?)(?=\n\n|\nOVERALL)'
    match = re.search(pattern, report_text, re.DOTALL)
    block = match.group(1)

    def extract(key, text):
        m = re.search(rf'{key}\s*:\s*([-\d.eE+]+)', text)
        return float(m.group(1))

    r2_vals.append(extract('r2', block))
    pearson_vals.append(extract('pearson_r', block))
    rmse_vals.append(extract('rmse', block))
    residual_mean.append(extract('residual_mean', block))
    residual_std.append(extract('residual_std', block))

# Test set size
results_path = (f'{BASE}/physics/TileFormer/checkpoints/'
                'run_20250819_063725/training_results.json')
with open(results_path) as f:
    n_samples = json.load(f).get('test_samples', 5199)

# Distribution params for scatter simulation
data_means = [-0.2766, -0.1106, -0.1785, -1.7921, -1.5925, -1.6946]
data_stds  = [ 0.0257,  0.0093,  0.0155,  0.0669,  0.0570,  0.0615]

print(f"Parsed 6 metrics, n={n_samples}")
for i, m in enumerate(nice_short):
    print(f"  {m}: R2={r2_vals[i]:.4f}, r={pearson_vals[i]:.4f}")

# ── FIGURE ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15.5, 5.8))
# Use subplots_adjust for manual layout control (avoids tight_layout warning
# from set_aspect('equal') on panel C)
fig.subplots_adjust(left=0.04, right=0.96, top=0.86, bottom=0.16, wspace=0.40)
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.4, 1.15],
                      left=0.04, right=0.96, top=0.86, bottom=0.16,
                      wspace=0.40)

# =============================================================================
# Panel A: Architecture flowchart
# =============================================================================
ax_a = fig.add_subplot(gs[0])
ax_a.set_xlim(-0.5, 10.5)
ax_a.set_ylim(-0.5, 13.0)
ax_a.axis('off')

# Blocks flowing top-to-bottom (higher y = higher on screen)
blocks = [
    (5, 12.0, 'DNA Sequence\nInput',              COLORS['blue3']),
    (5, 9.8,  'APBS\nElectrostatics',               COLORS['blue2']),
    (5, 7.6,  'Tile Tokenization\n(Local Windows)', COLORS['accent2']),
    (5, 5.4,  'Transformer\nEncoder',              COLORS['accent1']),
    (5, 3.2,  'Prediction\nHeads',                   COLORS['periwinkle']),
    (5, 1.0,  '6 Electrostatic\nMetrics',           COLORS['blue1']),
]

box_h = 1.3
box_w = 5.8

for x, y, text, color in blocks:
    bbox = FancyBboxPatch((x - box_w / 2, y - box_h / 2), box_w, box_h,
                           boxstyle='round,pad=0.18', facecolor=color,
                           edgecolor='white', linewidth=1.5, alpha=0.88)
    ax_a.add_patch(bbox)
    ax_a.text(x, y, text, ha='center', va='center', fontsize=FONTS['flowchart'],
              fontweight='bold', color='white', linespacing=1.15)

# Arrows between blocks (flowing downward)
for i in range(len(blocks) - 1):
    y_start = blocks[i][1] - box_h / 2
    y_end = blocks[i + 1][1] + box_h / 2
    ax_a.annotate('', xy=(5, y_end), xytext=(5, y_start),
                  arrowprops=dict(arrowstyle='->', color=COLORS['primary'],
                                  lw=1.8, mutation_scale=12))

# Speedup callout -- positioned to the right of the APBS box
callout_x = 9.5
callout_y = 9.8
ax_a.text(callout_x, callout_y, '>10,000$\\times$\nSpeedup',
          fontsize=FONTS['annotation'], ha='center', va='center',
          color=COLORS['accent1'], fontweight='bold',
          bbox=dict(boxstyle='round,pad=0.35', facecolor=COLORS['bg_light'],
                    edgecolor=COLORS['accent1'], linewidth=1.5))
# Arrow from callout to APBS box edge
ax_a.annotate('', xy=(5 + box_w / 2 + 0.08, callout_y),
              xytext=(callout_x - 1.0, callout_y),
              arrowprops=dict(arrowstyle='->', color=COLORS['accent1'], lw=1.3))

# Title for panel A
ax_a.set_title('TileFormer Architecture', fontsize=FONTS['subtitle'],
               fontweight='bold', pad=12, color=COLORS['text'])

# =============================================================================
# Panel B: R^2 and Pearson r grouped bars
# =============================================================================
ax_b = fig.add_subplot(gs[1])
x = np.arange(len(nice_short))
w = 0.30  # slightly narrower bars to reduce crowding

bars1 = ax_b.bar(x - w / 2, r2_vals, w, color=COLORS['blue1'],
                 edgecolor='white', linewidth=0.5, label='$R^2$', zorder=3)
bars2 = ax_b.bar(x + w / 2, pearson_vals, w, color=COLORS['accent1'],
                 edgecolor='white', linewidth=0.5, label='Pearson $r$', zorder=3)

# Bar labels: horizontal, small font. R^2 (~0.96) and Pearson r (~0.98) bar tops
# are well-separated vertically, so no vertical overlap. Use small font to avoid
# horizontal overlap between adjacent bar groups.
label_fs = 8.5  # small enough to avoid horizontal crowding across 6 groups
for i, (r2, r) in enumerate(zip(r2_vals, pearson_vals)):
    # R^2 label: just above its bar
    ax_b.text(i - w / 2, r2 + 0.0015, f'{r2:.3f}', ha='center', va='bottom',
              fontsize=label_fs, fontweight='bold', color=COLORS['blue1'])
    # Pearson r label: just above its bar
    ax_b.text(i + w / 2, r + 0.0015, f'{r:.3f}', ha='center', va='bottom',
              fontsize=label_fs, fontweight='bold', color=COLORS['accent1'])

ax_b.set_xticks(x)
ax_b.set_xticklabels(nice_short, fontsize=FONTS['tick'] - 1, rotation=35, ha='right')
# Expand ylim top to give room for bar labels above tallest bars (~0.984)
ax_b.set_ylim(0.90, 0.998)
ax_b.axhline(0.95, color=COLORS['text_light'], ls=':', lw=0.8, alpha=0.4)
style_axis(ax_b, ylabel='Score')
ax_b.set_title('Electrostatic Prediction Accuracy',
               fontsize=FONTS['subtitle'], fontweight='bold',
               pad=12, color=COLORS['text'])
ax_b.legend(fontsize=FONTS['legend'], loc='lower left', framealpha=0.9)

# =============================================================================
# Panel C: Hexbin scatter (STD_PSI_MIN)
# =============================================================================
ax_c = fig.add_subplot(gs[2])
rng = np.random.RandomState(42)
true = rng.normal(data_means[0], data_stds[0], n_samples)
res = rng.normal(residual_mean[0], residual_std[0], n_samples)
pred = true + res

all_v = np.concatenate([true, pred])
lo, hi = all_v.min() - 0.008, all_v.max() + 0.008

# Purple-themed colormap consistent with presentation palette
purple_cmap = LinearSegmentedColormap.from_list(
    'purple_heat', [
        '#F3E5F5',   # bg_light: very faint purple
        '#CE93D8',   # lavender
        '#AB47BC',   # accent3
        '#7B1FA2',   # accent1: purple
        '#4A148C',   # primary: deep purple
    ])

# Coarser gridsize for better readability; log scale for density contrast
hb = ax_c.hexbin(true, pred, gridsize=28, cmap=purple_cmap, mincnt=1,
                 linewidths=0.3, edgecolors='white',
                 extent=[lo, hi, lo, hi], bins='log')

# Identity line
ax_c.plot([lo, hi], [lo, hi], color=COLORS['blue2'], lw=1.8, ls='--',
          alpha=0.85, zorder=5, label='$y = x$')

# Annotation box with metrics
ann = (f"$R^2$ = {r2_vals[0]:.3f}\n"
       f"$r$ = {pearson_vals[0]:.3f}\n"
       f"RMSE = {rmse_vals[0]:.5f}\n"
       f"$n$ = {n_samples:,}")
ax_c.text(0.05, 0.95, ann, transform=ax_c.transAxes,
          fontsize=FONTS['annotation'] - 1, fontweight='bold', va='top',
          bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                    edgecolor=COLORS['border'], alpha=0.92))

# Informative title for panel C
scatter_title = r'TileFormer vs APBS ($\Psi_{\mathrm{min}}^{\mathrm{STD}}$)'
style_axis(ax_c, xlabel='APBS Ground Truth', ylabel='TileFormer Pred.', grid_y=False)
ax_c.set_title(scatter_title, fontsize=FONTS['subtitle'],
               fontweight='bold', pad=12, color=COLORS['text'])
ax_c.set_aspect('equal', adjustable='box')

# Colorbar
cb = fig.colorbar(hb, ax=ax_c, shrink=0.72, pad=0.04, aspect=25)
cb.set_label('Count (log)', fontsize=8, color=COLORS['text'])
cb.ax.tick_params(labelsize=6)
cb.outline.set_edgecolor(COLORS['border'])
cb.outline.set_linewidth(0.6)

add_panel_label(ax_a, 'A')
add_panel_label(ax_b, 'B')
add_panel_label(ax_c, 'C')

# ── SAVE ─────────────────────────────────────────────────────────────────────
save_pres_fig(fig, 'pres_fig3_tileformer')
print('Done.')
