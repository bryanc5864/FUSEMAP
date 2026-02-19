#!/usr/bin/env python3
"""
Pres Fig 1: Dataset Overview
4 panels: (A) Species/kingdom tree, (B) Dataset summary table,
          (C) Activity distribution violins, (D) Sequence length bar chart

Improvements over v1:
  - Panel A: Wider vertical spacing between leaf nodes, FancyBboxPatch nodes,
    cleaner connection lines with no overlap
  - Panel B: Explicit column widths, tighter cell padding, cleaner alternating
    row shading
  - Panel C: Wider violin positions to prevent sample-size label overlap,
    labels placed above violins instead of near axis
  - Panel D: Horizontal bar chart for cleaner layout, avoids multiline tick
    labels
"""
import sys, os, csv
sys.path.insert(0, '/home/bcheng/sequence_optimization/FUSEMAP/presentation_figures/pres_figures')
from pres_style import *
from matplotlib.patches import FancyBboxPatch

apply_pres_style()

BASE = '/home/bcheng/sequence_optimization/FUSEMAP'

# ── HELPER ──────────────────────────────────────────────────────────────────
def load_tsv_column(filepath, col_name):
    """Load a single column from a TSV file without pandas."""
    vals = []
    with open(filepath) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            v = row.get(col_name, '')
            if v and v != 'NA' and v != 'nan':
                try:
                    vals.append(float(v))
                except ValueError:
                    pass
    return np.array(vals)

# ── FIGURE SETUP ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.28,
                      left=0.06, right=0.97, top=0.93, bottom=0.06,
                      height_ratios=[1, 1.15])

# ════════════════════════════════════════════════════════════════════════════
# Panel A: Species/kingdom tree diagram
# ════════════════════════════════════════════════════════════════════════════
ax_a = fig.add_subplot(gs[0, 0])
ax_a.set_xlim(-0.5, 11)
ax_a.set_ylim(-0.5, 11.5)
ax_a.axis('off')

# ---- tree data -----------------------------------------------------------
#  (kingdom_name, color, [leaf_names], y_center)
#  Leaf spacing = 1.1; kingdoms positioned so no leaf-to-leaf gap < 0.8
kingdoms = [
    ('Human',      COLORS['human'],      ['K562', 'HepG2', 'WTC11'],    9.5),
    ('Drosophila', COLORS['drosophila'], ['S2 (Dev)', 'S2 (Hk)'],       6.3),
    ('Plant',      COLORS['plant'],      ['Maize', 'Sorghum', 'Arabid.'], 3.3),
    ('Yeast',      COLORS['yeast'],      ['DREAM'],                      0.8),
]

# ---- draw root node ------------------------------------------------------
root_x, root_y = 1.3, 5.15
root_box = FancyBboxPatch((root_x - 1.1, root_y - 0.7), 2.2, 1.4,
                           boxstyle='round,pad=0.15',
                           facecolor=COLORS['bg_light'],
                           edgecolor=COLORS['primary'], linewidth=1.8,
                           zorder=4)
ax_a.add_patch(root_box)
ax_a.text(root_x, root_y, 'FUSEMAP\nDatasets',
          fontsize=FONTS['subtitle'], fontweight='bold',
          ha='center', va='center', color=COLORS['primary'], zorder=5)

# ---- draw kingdoms and leaves --------------------------------------------
kingdom_x = 4.8
leaf_x = 8.5

for kname, kcolor, leaves, ky in kingdoms:
    # --- branch from root to kingdom node ----------------------------------
    ax_a.plot([root_x + 1.15, kingdom_x - 0.9], [root_y, ky],
              color=COLORS['border'], lw=1.3, solid_capstyle='round', zorder=1)

    # --- kingdom box -------------------------------------------------------
    kw, kh = 1.8, 0.55
    k_box = FancyBboxPatch((kingdom_x - kw / 2, ky - kh / 2), kw, kh,
                            boxstyle='round,pad=0.12',
                            facecolor=kcolor, edgecolor=kcolor,
                            linewidth=1.0, zorder=3)
    ax_a.add_patch(k_box)
    ax_a.text(kingdom_x, ky, kname,
              fontsize=FONTS['flowchart'], fontweight='bold',
              ha='center', va='center', color='white', zorder=4)

    # --- leaf nodes --------------------------------------------------------
    n_leaves = len(leaves)
    # spacing: 1.1 units between leaves -- compact but no overlap
    leaf_spacing = 1.1 if n_leaves > 1 else 0
    leaf_start_y = ky + (n_leaves - 1) * leaf_spacing / 2

    for i, leaf in enumerate(leaves):
        ly = leaf_start_y - i * leaf_spacing

        # line from kingdom to leaf
        ax_a.plot([kingdom_x + kw / 2 + 0.05, leaf_x - 0.75],
                  [ky, ly],
                  color=kcolor, lw=0.9, alpha=0.55, solid_capstyle='round',
                  zorder=1)

        # leaf box
        lw_box, lh_box = 1.5, 0.48
        l_box = FancyBboxPatch((leaf_x - lw_box / 2, ly - lh_box / 2),
                                lw_box, lh_box,
                                boxstyle='round,pad=0.1',
                                facecolor='white', edgecolor=kcolor,
                                linewidth=0.9, alpha=0.95, zorder=3)
        ax_a.add_patch(l_box)
        ax_a.text(leaf_x, ly, leaf,
                  fontsize=FONTS['flowchart_detail'], fontweight='semibold',
                  ha='center', va='center', color=kcolor, zorder=4)

ax_a.set_title('Cross-Kingdom Dataset Coverage', fontsize=FONTS['subtitle'],
               fontweight='bold', pad=12, color=COLORS['text'])
add_panel_label(ax_a, 'A')

# ════════════════════════════════════════════════════════════════════════════
# Panel B: Dataset summary table
# ════════════════════════════════════════════════════════════════════════════
ax_b = fig.add_subplot(gs[0, 1])
ax_b.axis('off')

table_data = [
    ['K562',      'Human',      'lentiMPRA', '158,377', '22,613', '22,631', '230'],
    ['HepG2',     'Human',      'lentiMPRA', '97,925',  '13,997', '13,953', '230'],
    ['WTC11',     'Human',      'lentiMPRA', '39,201',  '5,596',  '5,597',  '230'],
    ['DeepSTARR', 'Drosophila', 'STARR-seq', '352,009', '40,570', '41,186', '249'],
    ['Maize',     'Plant',      'STARR-seq', '24,209',  '3,458',  '3,001',  '170'],
    ['Sorghum',   'Plant',      'STARR-seq', '19,502',  '2,785',  '2,467',  '170'],
    ['Arabid.',   'Plant',      'STARR-seq', '13,169',  '1,881',  '1,686',  '170'],
    ['Yeast',     'Yeast',      'FACS-seq',  '~6.7M',   '33,696', '71,103', '110'],
]
col_labels = ['Dataset', 'Species', 'Assay', 'Train', 'Val', 'Test', 'Len']

# Column width ratios (sum to 1.0) -- give more room to Dataset, Species, Train
col_widths = [0.15, 0.15, 0.14, 0.16, 0.14, 0.14, 0.10]
# Adjust to [0,1] x [0,1] coordinate space: cumulative left edges
col_edges = [sum(col_widths[:j]) for j in range(len(col_widths))]

org_colors_map = {
    'Human': COLORS['human'],
    'Drosophila': COLORS['drosophila'],
    'Plant': COLORS['plant'],
    'Yeast': COLORS['yeast'],
}

tbl = ax_b.table(cellText=table_data, colLabels=col_labels,
                 cellLoc='center', loc='center',
                 colWidths=col_widths)
tbl.auto_set_font_size(False)
tbl.set_fontsize(FONTS['flowchart_detail'])
tbl.scale(1.25, 1.6)

# Style header row
for j in range(len(col_labels)):
    cell = tbl[0, j]
    cell.set_facecolor(COLORS['primary'])
    cell.set_text_props(color='white', fontweight='bold',
                        fontsize=FONTS['flowchart'])
    cell.set_edgecolor(COLORS['border'])
    cell.set_linewidth(0.6)

# Style data rows
for i in range(len(table_data)):
    org = table_data[i][1]
    c = org_colors_map.get(org, COLORS['text_light'])
    stripe = '#F7F2FA' if i % 2 == 0 else 'white'
    for j in range(len(col_labels)):
        cell = tbl[i + 1, j]
        cell.set_edgecolor('#D1C4E9')
        cell.set_linewidth(0.4)
        cell.set_facecolor(stripe)
        if j == 0:
            cell.set_text_props(fontweight='bold', color=c,
                                fontsize=FONTS['flowchart_detail'])
        elif j == 1:
            cell.set_text_props(color=c, fontsize=FONTS['flowchart_detail'])
        else:
            cell.set_text_props(color=COLORS['text'],
                                fontsize=FONTS['flowchart_detail'])

ax_b.set_title('Dataset Summary (9 Datasets, 4 Kingdoms)', fontsize=FONTS['subtitle'],
               fontweight='bold', pad=12, color=COLORS['text'])
add_panel_label(ax_b, 'B')

# ════════════════════════════════════════════════════════════════════════════
# Panel C: Activity distribution violins
# ════════════════════════════════════════════════════════════════════════════
ax_c = fig.add_subplot(gs[1, 0])

# Load test data
distributions = {}
dist_colors = {}

datasets_to_load = [
    ('K562',     f'{BASE}/physics/data/lentiMPRA_data/K562/K562_test_with_features.tsv',
     'activity', COLORS['human']),
    ('HepG2',    f'{BASE}/physics/data/lentiMPRA_data/HepG2/HepG2_test_with_features.tsv',
     'activity', COLORS['human']),
    ('WTC11',    f'{BASE}/physics/data/lentiMPRA_data/WTC11/WTC11_test_with_features.tsv',
     'activity', COLORS['human']),
    ('DeepSTARR', '/home/shared/genomic_data/deepstarr/Sequences_activity_Test.txt',
     'Dev_log2_enrichment', COLORS['drosophila']),
    ('Arabid.',  '/home/shared/genomic_data/jores_plants/tobacco_leaf/arabidopsis/test.tsv',
     'activity', COLORS['plant']),
    ('Yeast',    '/home/shared/genomic_data/dream_yeast/yeast_test.txt',
     'maude_expression', COLORS['yeast']),
]

for name, path, col, color in datasets_to_load:
    try:
        vals = load_tsv_column(path, col)
        if len(vals) > 0:
            distributions[name] = vals
            dist_colors[name] = color
            print(f'  Loaded {name}: n={len(vals)}')
    except Exception as e:
        print(f'  {name} load failed: {e}')

names = list(distributions.keys())
data_list = [distributions[n] for n in names]
colors_list = [dist_colors[n] for n in names]

if data_list:
    n_v = len(names)
    spacing = 1.0
    positions = [i * spacing for i in range(n_v)]

    parts = ax_c.violinplot(data_list, positions=positions,
                            showmeans=False, showmedians=False,
                            showextrema=False, widths=0.7)

    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors_list[i])
        pc.set_edgecolor(colors_list[i])
        pc.set_alpha(0.45)

    # Overlay box plots for quartile visibility
    bp = ax_c.boxplot(data_list, positions=positions, widths=0.18,
                      patch_artist=True, showfliers=False, zorder=4)
    for i, (box, median, whisker_lo, whisker_hi, cap_lo, cap_hi) in enumerate(
        zip(bp['boxes'], bp['medians'],
            bp['whiskers'][::2], bp['whiskers'][1::2],
            bp['caps'][::2], bp['caps'][1::2])):
        box.set_facecolor(colors_list[i])
        box.set_edgecolor('white')
        box.set_linewidth(1.0)
        box.set_alpha(0.85)
        median.set_color('white')
        median.set_linewidth(1.5)
        for w in [whisker_lo, whisker_hi]:
            w.set_color(colors_list[i])
            w.set_linewidth(0.8)
        for c in [cap_lo, cap_hi]:
            c.set_color(colors_list[i])
            c.set_linewidth(0.8)

    ax_c.set_xticks(positions)
    ax_c.set_xticklabels(names, fontsize=FONTS['tick'], rotation=0, ha='center')

    # Sample-size labels below x-axis
    for i, n in enumerate(names):
        ax_c.annotate(f'n={len(distributions[n]):,}',
                      xy=(positions[i], 0), xycoords=('data', 'axes fraction'),
                      xytext=(0, -28), textcoords='offset points',
                      ha='center', va='top',
                      fontsize=FONTS['caption'] - 1, color=COLORS['text_light'],
                      fontstyle='italic')

style_axis(ax_c, title='Activity Distributions (Test Sets)',
           ylabel='Activity / Expression')
add_panel_label(ax_c, 'C')

# ════════════════════════════════════════════════════════════════════════════
# Panel D: Sequence length bar chart (horizontal for cleaner labels)
# ════════════════════════════════════════════════════════════════════════════
ax_d = fig.add_subplot(gs[1, 1])

seq_data = [
    ('Yeast (DREAM)',          110, COLORS['yeast']),
    ('Plants (Jores)',         170, COLORS['plant']),
    ('Human (lentiMPRA)',      230, COLORS['human']),
    ('Drosophila (DeepSTARR)', 249, COLORS['drosophila']),
]

labels_d = [d[0] for d in seq_data]
lengths = [d[1] for d in seq_data]
colors_d = [d[2] for d in seq_data]

bars = ax_d.barh(range(len(seq_data)), lengths,
                 color=colors_d, edgecolor='white',
                 linewidth=0.5, height=0.55, zorder=3)

# Place "XXX bp" label at the end of each bar
for i, (label, length, color) in enumerate(seq_data):
    ax_d.text(length + 4, i, f'{length} bp',
              ha='left', va='center',
              fontsize=FONTS['bar_label'], fontweight='bold', color=color)

ax_d.set_yticks(range(len(seq_data)))
ax_d.set_yticklabels(labels_d, fontsize=FONTS['tick'])
ax_d.set_xlim(0, 305)
ax_d.set_xticks([0, 50, 100, 150, 200, 250])
ax_d.invert_yaxis()

style_axis(ax_d, title='Input Sequence Lengths', xlabel='Length (bp)',
           ylabel=None, grid_y=False)
# Add vertical grid instead for horizontal bars
ax_d.xaxis.grid(True, alpha=0.3, linewidth=0.6, color=COLORS['grid'], zorder=0)
ax_d.set_axisbelow(True)

add_panel_label(ax_d, 'D')

# ── SAVE ────────────────────────────────────────────────────────────────────
# Use subplots_adjust instead of tight_layout to avoid warnings with tables
save_pres_fig(fig, 'pres_fig1_datasets')
print('Done.')
