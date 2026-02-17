#!/usr/bin/env python3
"""
Figure 15. Therapeutic enhancer design process and outcomes.

6-panel composite figure:
  (A) Cell-type specificity distributions for ISM-designed sequences by target
  (B) OracleCheck verdict distribution by target and method (stacked bar)
  (C) Method comparison: mean specificity across all methods and cell types
  (D) Target activity vs specificity scatter (Pareto-like) for ISM_target
  (E) Specificity distributions by method (ISM vs PINCSD vs EMOO vs ISM_target)
  (F) Pass rate comparison across methods and cell types
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
})

FUSEMAP_ROOT = Path('/home/bcheng/sequence_optimization/FUSEMAP')
RESULTS_DIR = FUSEMAP_ROOT / 'results/therapeutic_method_comparison'

# Colors
CELL_COLORS = {
    'K562': '#4A90D9',
    'HepG2': '#E67E22',
    'WTC11': '#27AE60',
}

METHOD_COLORS = {
    'ism': '#1565C0',
    'ism_target': '#7B1FA2',
    'pincsd': '#C62828',
    'emoo': '#2E7D32',
}

METHOD_LABELS = {
    'ism': 'ISM\n(specificity)',
    'ism_target': 'ISM-Target\n(activity only)',
    'pincsd': 'PINCSD',
    'emoo': 'EMOO',
}

VERDICT_COLORS = {
    'GREEN': '#4CAF50',
    'YELLOW': '#FFC107',
    'RED': '#F44336',
}

# ---------------------------------------------------------------------------
# Load all per-method result CSVs
# ---------------------------------------------------------------------------
methods = ['pincsd', 'emoo', 'ism', 'ism_target']
cell_types = ['K562', 'HepG2', 'WTC11']

print("Loading result CSVs...")
data = {}
for method in methods:
    for cell in cell_types:
        fpath = RESULTS_DIR / f'{method}_{cell}_results.csv'
        if fpath.exists():
            df = pd.read_csv(fpath)
            data[(method, cell)] = df
            print(f"  {method} / {cell}: {len(df)} sequences")
        else:
            print(f"  {method} / {cell}: NOT FOUND")

# ---------------------------------------------------------------------------
# Compute summary statistics from loaded data
# ---------------------------------------------------------------------------
summary_rows = []
for (method, cell), df in data.items():
    n = len(df)
    n_green = (df['oracle_verdict'] == 'GREEN').sum()
    n_yellow = (df['oracle_verdict'] == 'YELLOW').sum()
    n_red = (df['oracle_verdict'] == 'RED').sum()
    summary_rows.append({
        'method': method,
        'cell': cell,
        'n': n,
        'mean_spec': df['specificity'].mean(),
        'max_spec': df['specificity'].max(),
        'std_spec': df['specificity'].std(),
        'mean_target': df['target_activity'].mean(),
        'pass_rate': (n_green + n_yellow) / n if n > 0 else 0,
        'n_green': n_green,
        'n_yellow': n_yellow,
        'n_red': n_red,
    })

summary = pd.DataFrame(summary_rows)
print("\nSummary:")
print(summary.to_string(index=False))

# ---------------------------------------------------------------------------
# Create figure
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(22, 16))
gs = gridspec.GridSpec(2, 3, hspace=0.38, wspace=0.30,
                       left=0.06, right=0.97, top=0.92, bottom=0.06)

# ===== Panel A: Specificity distributions by target (ISM_target) ===========
ax_a = fig.add_subplot(gs[0, 0])

positions = []
violins_data = []
colors = []
labels = []

for i, cell in enumerate(cell_types):
    key = ('ism_target', cell)
    if key in data:
        df = data[key]
        positions.append(i)
        violins_data.append(df['specificity'].values)
        colors.append(CELL_COLORS[cell])
        labels.append(cell)

parts = ax_a.violinplot(violins_data, positions=positions, showmeans=True,
                        showmedians=True, showextrema=False)

for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors[i])
    pc.set_alpha(0.6)
    pc.set_edgecolor(colors[i])

parts['cmeans'].set_color('black')
parts['cmeans'].set_linewidth(2)
parts['cmedians'].set_color('white')
parts['cmedians'].set_linewidth(2)

# Overlay individual points (jittered)
for i, (pos, vals) in enumerate(zip(positions, violins_data)):
    jitter = np.random.normal(0, 0.04, size=len(vals))
    ax_a.scatter(pos + jitter, vals, c=colors[i], alpha=0.15, s=8,
                 edgecolors='none', zorder=2)

# Annotate means
for i, (pos, vals) in enumerate(zip(positions, violins_data)):
    mean_val = np.mean(vals)
    ax_a.text(pos + 0.25, mean_val, f'  {mean_val:.2f}', va='center',
              fontsize=9, fontweight='bold', color=colors[i])

ax_a.set_xticks(positions)
ax_a.set_xticklabels(labels, fontsize=11, fontweight='bold')
ax_a.set_ylabel('Specificity Score (log-units)', fontsize=10)
ax_a.set_title('Cell-Type Specificity\n(ISM Target-Only Optimization)', fontsize=11)
ax_a.axhline(0, color='black', linewidth=0.5, linestyle='-', alpha=0.3)

ax_a.text(-0.08, 1.06, '(A)', transform=ax_a.transAxes,
          fontsize=16, fontweight='bold', va='top')

# ===== Panel B: OracleCheck verdict stacked bar ============================
ax_b = fig.add_subplot(gs[0, 1])

# Group by method + cell
bar_labels = []
green_vals = []
yellow_vals = []
red_vals = []
bar_edge_colors = []

for method in methods:
    for cell in cell_types:
        row = summary[(summary['method'] == method) & (summary['cell'] == cell)]
        if len(row) == 0:
            continue
        row = row.iloc[0]
        n = row['n']
        bar_labels.append(f'{method}\n{cell}')
        green_vals.append(row['n_green'] / n * 100)
        yellow_vals.append(row['n_yellow'] / n * 100)
        red_vals.append(row['n_red'] / n * 100)
        bar_edge_colors.append(CELL_COLORS[cell])

x_b = np.arange(len(bar_labels))
width = 0.7

ax_b.bar(x_b, green_vals, width, label='GREEN', color=VERDICT_COLORS['GREEN'],
         edgecolor='white', linewidth=0.5)
ax_b.bar(x_b, yellow_vals, width, bottom=green_vals, label='YELLOW',
         color=VERDICT_COLORS['YELLOW'], edgecolor='white', linewidth=0.5)
bottom_ry = np.array(green_vals) + np.array(yellow_vals)
ax_b.bar(x_b, red_vals, width, bottom=bottom_ry, label='RED',
         color=VERDICT_COLORS['RED'], edgecolor='white', linewidth=0.5)

ax_b.set_xticks(x_b)
ax_b.set_xticklabels(bar_labels, fontsize=6, fontweight='bold', rotation=45, ha='right')
ax_b.set_ylabel('Percentage (%)', fontsize=10)
ax_b.set_ylim(0, 105)
ax_b.set_title('OracleCheck Verdict Distribution', fontsize=11)
ax_b.legend(fontsize=9, loc='lower right', framealpha=0.9)

# Method divider lines
for i in range(1, len(methods)):
    ax_b.axvline(i * len(cell_types) - 0.5, color='black', linewidth=0.8,
                 linestyle='--', alpha=0.3)

ax_b.text(-0.08, 1.06, '(B)', transform=ax_b.transAxes,
          fontsize=16, fontweight='bold', va='top')

# ===== Panel C: Mean specificity comparison (grouped bar) ==================
ax_c = fig.add_subplot(gs[0, 2])

n_methods = len(methods)
n_cells = len(cell_types)
width_c = 0.18
x_c = np.arange(n_cells)

for i, method in enumerate(methods):
    vals = []
    for cell in cell_types:
        row = summary[(summary['method'] == method) & (summary['cell'] == cell)]
        vals.append(row.iloc[0]['mean_spec'] if len(row) > 0 else 0)
    offset = (i - n_methods/2 + 0.5) * width_c
    bars = ax_c.bar(x_c + offset, vals, width_c,
                    color=METHOD_COLORS[method],
                    label=METHOD_LABELS[method].replace('\n', ' '),
                    edgecolor='white', linewidth=0.5, zorder=3)
    # Value labels
    for xi, val in zip(x_c + offset, vals):
        ax_c.text(xi, val + 0.08, f'{val:.1f}', ha='center', va='bottom',
                  fontsize=7, fontweight='bold', rotation=45)

ax_c.set_xticks(x_c)
ax_c.set_xticklabels(cell_types, fontsize=11, fontweight='bold')
ax_c.set_ylabel('Mean Specificity (log-units)', fontsize=10)
ax_c.set_title('Method Comparison: Mean Specificity', fontsize=11)
ax_c.legend(fontsize=8, loc='upper right', framealpha=0.9)
ax_c.axhline(0, color='black', linewidth=0.5, linestyle='-', alpha=0.3)

ax_c.text(-0.08, 1.06, '(C)', transform=ax_c.transAxes,
          fontsize=16, fontweight='bold', va='top')

# ===== Panel D: Target activity vs specificity scatter =====================
ax_d = fig.add_subplot(gs[1, 0])

for cell in cell_types:
    key = ('ism_target', cell)
    if key not in data:
        continue
    df = data[key]
    # Color by verdict
    for verdict, vcolor in VERDICT_COLORS.items():
        mask = df['oracle_verdict'] == verdict
        if mask.sum() > 0:
            ax_d.scatter(df.loc[mask, 'target_activity'],
                        df.loc[mask, 'specificity'],
                        c=vcolor, s=20, alpha=0.5, edgecolors='none',
                        label=f'{cell} {verdict}' if cell == 'HepG2' else None,
                        marker='o' if cell == 'K562' else ('s' if cell == 'HepG2' else '^'))

# Legend for cell types
for cell, marker in [('K562', 'o'), ('HepG2', 's'), ('WTC11', '^')]:
    ax_d.scatter([], [], c='gray', s=40, marker=marker, label=cell)

ax_d.set_xlabel('Target Activity (log₂)', fontsize=10)
ax_d.set_ylabel('Specificity Score', fontsize=10)
ax_d.set_title('Activity vs Specificity\n(ISM Target-Only)', fontsize=11)
ax_d.axhline(0, color='black', linewidth=0.5, linestyle='-', alpha=0.3)
ax_d.legend(fontsize=8, loc='upper left', framealpha=0.9, ncol=2)

ax_d.text(-0.08, 1.06, '(D)', transform=ax_d.transAxes,
          fontsize=16, fontweight='bold', va='top')

# ===== Panel E: Specificity distributions by method (HepG2 focus) =========
ax_e = fig.add_subplot(gs[1, 1])

cell_focus = 'HepG2'
violins_e = []
positions_e = []
colors_e = []
labels_e = []

for i, method in enumerate(methods):
    key = (method, cell_focus)
    if key in data:
        df = data[key]
        violins_e.append(df['specificity'].values)
        positions_e.append(i)
        colors_e.append(METHOD_COLORS[method])
        labels_e.append(METHOD_LABELS[method])

if violins_e:
    parts_e = ax_e.violinplot(violins_e, positions=positions_e, showmeans=True,
                              showmedians=True, showextrema=False)
    for i, pc in enumerate(parts_e['bodies']):
        pc.set_facecolor(colors_e[i])
        pc.set_alpha(0.6)
        pc.set_edgecolor(colors_e[i])
    parts_e['cmeans'].set_color('black')
    parts_e['cmeans'].set_linewidth(2)
    parts_e['cmedians'].set_color('white')
    parts_e['cmedians'].set_linewidth(2)

    # Overlay jittered points
    for i, (pos, vals) in enumerate(zip(positions_e, violins_e)):
        jitter = np.random.normal(0, 0.04, size=len(vals))
        ax_e.scatter(pos + jitter, vals, c=colors_e[i], alpha=0.12, s=8,
                     edgecolors='none', zorder=2)

    # Annotate means
    for i, (pos, vals) in enumerate(zip(positions_e, violins_e)):
        mean_val = np.mean(vals)
        ax_e.text(pos, np.max(vals) + 0.3, f'μ={mean_val:.2f}', ha='center',
                  fontsize=8, fontweight='bold', color=colors_e[i])

ax_e.set_xticks(positions_e)
ax_e.set_xticklabels(labels_e, fontsize=8, fontweight='bold')
ax_e.set_ylabel('Specificity Score (log-units)', fontsize=10)
ax_e.set_title(f'Method Comparison: {cell_focus} Target', fontsize=11)
ax_e.axhline(0, color='black', linewidth=0.5, linestyle='-', alpha=0.3)

ax_e.text(-0.08, 1.06, '(E)', transform=ax_e.transAxes,
          fontsize=16, fontweight='bold', va='top')

# ===== Panel F: Pass rate comparison =======================================
ax_f = fig.add_subplot(gs[1, 2])

width_f = 0.18
x_f = np.arange(n_cells)

for i, method in enumerate(methods):
    vals = []
    for cell in cell_types:
        row = summary[(summary['method'] == method) & (summary['cell'] == cell)]
        vals.append(row.iloc[0]['pass_rate'] * 100 if len(row) > 0 else 0)
    offset = (i - n_methods/2 + 0.5) * width_f
    bars = ax_f.bar(x_f + offset, vals, width_f,
                    color=METHOD_COLORS[method],
                    label=METHOD_LABELS[method].replace('\n', ' '),
                    edgecolor='white', linewidth=0.5, zorder=3)
    # Value labels
    for xi, val in zip(x_f + offset, vals):
        ax_f.text(xi, val + 0.5, f'{val:.0f}%', ha='center', va='bottom',
                  fontsize=7, fontweight='bold', rotation=45)

ax_f.set_xticks(x_f)
ax_f.set_xticklabels(cell_types, fontsize=11, fontweight='bold')
ax_f.set_ylabel('OracleCheck Pass Rate (%)', fontsize=10)
ax_f.set_ylim(0, 110)
ax_f.set_title('OracleCheck Pass Rate by Method', fontsize=11)
ax_f.legend(fontsize=8, loc='lower right', framealpha=0.9)

ax_f.text(-0.08, 1.06, '(F)', transform=ax_f.transAxes,
          fontsize=16, fontweight='bold', va='top')

# ===== Suptitle =============================================================
fig.suptitle(
    'Figure 15. Therapeutic Enhancer Design Process and Outcomes',
    fontsize=16, fontweight='bold', y=0.97)

# ===== Save =================================================================
out_base = str(FUSEMAP_ROOT / 'presentation_figures/figure15_therapeutic_design')
fig.savefig(out_base + '.png', dpi=200, bbox_inches='tight', facecolor='white')
fig.savefig(out_base + '.pdf', bbox_inches='tight', facecolor='white')
plt.close(fig)

print(f'\nSaved: {out_base}.png')
print(f'Saved: {out_base}.pdf')
