#!/usr/bin/env python3
"""
Pres Fig 7: OracleCheck -- Multi-Dimensional In-Silico Validation
=================================================================
3 panels:
  (A) Methodology diagram showing the 4 validation dimensions and verdict logic
  (B) Stacked bar chart of GREEN/YELLOW/RED verdicts (ISM method, n=200 each)
  (C) Top-10 specificity score distribution per cell type
"""
import sys, json, os
sys.path.insert(0, '/home/bcheng/sequence_optimization/FUSEMAP/presentation_figures/pres_figures')
from pres_style import *
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

apply_pres_style()

BASE = '/home/bcheng/sequence_optimization/FUSEMAP'

# ── LOAD DATA ────────────────────────────────────────────────────────────────
# Panel B: ISM method comparison data (n=200 per cell type)
comparison_path = f'{BASE}/FUSEMAP_results/therapeutic_design/therapeutic_method_comparison/comparison_report.json'
ism_data = {}
if os.path.exists(comparison_path):
    with open(comparison_path) as f:
        comp = json.load(f)
    for entry in comp['summary']:
        ct = entry['target_cell'].lower()
        ism_data[ct] = {
            'total': entry['n_sequences'],
            'green': entry['n_green'],
            'yellow': entry['n_yellow'],
            'red': entry['n_red'],
            'pass_rate': entry['pass_rate'],
            'mean_spec': entry['mean_specificity'],
        }
else:
    # Fallback: load from individual reports
    for ct in ['hepg2', 'k562', 'wtc11']:
        path = f'{BASE}/FUSEMAP_results/therapeutic_design/therapeutic_{ct}/therapeutic_enhancers_report.json'
        with open(path) as f:
            d = json.load(f)
        s = d['summary']
        ism_data[ct] = {
            'total': s['total_candidates'],
            'green': s.get('n_green', 0),
            'yellow': s.get('n_yellow', 0),
            'red': s.get('n_red', 0),
            'pass_rate': (s.get('n_green', 0) + s.get('n_yellow', 0)) / s['total_candidates'],
            'mean_spec': s['mean_specificity'],
        }

# Panel C: Top-10 specificity from individual reports
top10_data = {}
for ct in ['hepg2', 'k562', 'wtc11']:
    path = f'{BASE}/FUSEMAP_results/therapeutic_design/therapeutic_{ct}/therapeutic_enhancers_report.json'
    with open(path) as f:
        d = json.load(f)
    top10_data[ct] = [c['specificity_score'] for c in d.get('top_10', [])]

# Print summary
for ct in ['k562', 'hepg2', 'wtc11']:
    d = ism_data[ct]
    print(f"{ct.upper()}: n={d['total']}, GREEN={d['green']}({d['green']/d['total']*100:.1f}%), "
          f"YELLOW={d['yellow']}({d['yellow']/d['total']*100:.1f}%), "
          f"RED={d['red']}({d['red']/d['total']*100:.1f}%), "
          f"pass_rate={d['pass_rate']*100:.1f}%, mean_spec={d['mean_spec']:.2f}")

# ── FIGURE ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 5.5))
gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 0.85, 0.75], wspace=0.38)

# ══════════════════════════════════════════════════════════════════════════════
# Panel A: OracleCheck Methodology Diagram
# ══════════════════════════════════════════════════════════════════════════════
ax_a = fig.add_subplot(gs[0])
ax_a.set_xlim(-0.5, 13)
ax_a.set_ylim(0, 10.0)
ax_a.axis('off')
ax_a.set_clip_on(False)
ax_a.set_title('OracleCheck Protocol', fontsize=FONTS['subtitle'],
               fontweight='bold', pad=12, color=COLORS['text'])

# ── Dimension boxes ──
# Four validation dimensions as rounded boxes on the left side
dim_specs = [
    {
        'label': '1. Physics Conformity',
        'detail': 'Z-scores per physics family vs\nnatural enhancers. Soft: |z|>2.5\nHard: |z|>4.0',
        'color': '#1565C0',
        'y': 7.6,
    },
    {
        'label': '2. Composition Hygiene',
        'detail': 'GC: 0.35\u20130.75, CpG ratio,\nentropy, repeats <0.3,\nhomopolymer <10 bp',
        'color': '#7B1FA2',
        'y': 5.5,
    },
    {
        'label': '3. Confidence / OOD',
        'detail': 'Epistemic unc. <P90,\nconformal width <P95,\nKNN OOD score <P95',
        'color': '#E65100',
        'y': 3.4,
    },
    {
        'label': '4. RC Consistency',
        'detail': '|\u0177(seq) \u2212 \u0177(RC)| < 0.1,\nISM symmetry \u226590%',
        'color': '#00695C',
        'y': 1.5,
    },
]

box_x = 0.3
box_w = 4.6
box_h = 1.7

for spec in dim_specs:
    # Rounded box for each dimension
    rect = FancyBboxPatch((box_x, spec['y']), box_w, box_h,
                           boxstyle='round,pad=0.15',
                           facecolor=spec['color'], edgecolor='white',
                           linewidth=1.2, alpha=0.12, zorder=2)
    ax_a.add_patch(rect)
    # Left border accent
    accent = Rectangle((box_x, spec['y']), 0.22, box_h,
                        facecolor=spec['color'], edgecolor='none',
                        alpha=0.85, zorder=3)
    ax_a.add_patch(accent)
    # Dimension label (bold, single line)
    ax_a.text(box_x + 0.45, spec['y'] + box_h - 0.10, spec['label'],
              fontsize=FONTS['flowchart'], fontweight='bold', va='top', ha='left',
              color=spec['color'], zorder=4)
    # Detail text (below label)
    ax_a.text(box_x + 0.45, spec['y'] + box_h - 0.55, spec['detail'],
              fontsize=FONTS['flowchart_detail'] - 1, va='top', ha='left',
              color=COLORS['text'], zorder=4, linespacing=1.15)

# ── Arrows from dimension boxes to verdict box ──
arrow_start_x = box_x + box_w + 0.1
verdict_x = 6.4
verdict_w = 5.8
verdict_h = 7.8
verdict_y = 1.5

for spec in dim_specs:
    mid_y = spec['y'] + box_h / 2
    ax_a.annotate('', xy=(verdict_x, mid_y), xytext=(arrow_start_x, mid_y),
                  arrowprops=dict(arrowstyle='->', color=COLORS['text_light'],
                                  lw=1.0, connectionstyle='arc3,rad=0'))

# ── Verdict box ──
verdict_box = FancyBboxPatch((verdict_x, verdict_y), verdict_w, verdict_h,
                              boxstyle='round,pad=0.25',
                              facecolor='#F5F5F5', edgecolor=COLORS['primary'],
                              linewidth=1.5, alpha=0.95, zorder=2)
ax_a.add_patch(verdict_box)

ax_a.text(verdict_x + verdict_w / 2, verdict_y + verdict_h - 0.25,
          'Verdict Logic', fontsize=FONTS['annotation'], fontweight='bold',
          ha='center', va='top', color=COLORS['primary'], zorder=5)

# ── Verdict entries inside the box ──
verdict_entries = [
    {
        'color': COLORS['green'],
        'label': 'GREEN',
        'text_color': 'white',
        'desc': 'All 4 checks pass',
        'y_center': verdict_y + verdict_h - 2.0,
    },
    {
        'color': COLORS['yellow'],
        'label': 'YELLOW',
        'text_color': '#424242',
        'desc': '\u22641 soft failure only',
        'y_center': verdict_y + verdict_h - 4.0,
    },
    {
        'color': COLORS['red'],
        'label': 'RED',
        'text_color': 'white',
        'desc': 'Any hard failure OR\n\u22652 soft failures',
        'y_center': verdict_y + verdict_h - 6.2,
    },
]

pill_w = 2.0
pill_h = 0.85

for v in verdict_entries:
    # Colored pill for the verdict label
    pill_x = verdict_x + 0.5
    pill_y = v['y_center'] - pill_h / 2
    pill = FancyBboxPatch((pill_x, pill_y), pill_w, pill_h,
                           boxstyle='round,pad=0.15',
                           facecolor=v['color'], edgecolor='white',
                           linewidth=1.5, alpha=0.9, zorder=5)
    ax_a.add_patch(pill)
    ax_a.text(pill_x + pill_w / 2, v['y_center'], v['label'],
              fontsize=FONTS['bar_label'], fontweight='bold',
              ha='center', va='center', color=v['text_color'], zorder=6)
    # Description text
    ax_a.text(pill_x + pill_w + 0.3, v['y_center'], v['desc'],
              fontsize=FONTS['flowchart_detail'], va='center', ha='left',
              color=COLORS['text'], zorder=5, linespacing=1.3)

add_panel_label(ax_a, 'A', x=-0.03)

# ══════════════════════════════════════════════════════════════════════════════
# Panel B: Stacked bar chart (ISM method comparison, n=200)
# ══════════════════════════════════════════════════════════════════════════════
ax_b = fig.add_subplot(gs[1])

cell_types = ['K562', 'HepG2', 'WTC11']
ct_keys = ['k562', 'hepg2', 'wtc11']

greens = [ism_data[ct]['green'] for ct in ct_keys]
yellows = [ism_data[ct]['yellow'] for ct in ct_keys]
reds = [ism_data[ct]['red'] for ct in ct_keys]
totals = [ism_data[ct]['total'] for ct in ct_keys]
pass_rates = [ism_data[ct]['pass_rate'] for ct in ct_keys]

# Compute percentages
green_pct = [g / t * 100 for g, t in zip(greens, totals)]
yellow_pct = [y / t * 100 for y, t in zip(yellows, totals)]
red_pct = [r / t * 100 for r, t in zip(reds, totals)]

x = np.arange(len(cell_types))
w = 0.5

bars_g = ax_b.bar(x, green_pct, w, color=COLORS['green'], edgecolor='white',
                  linewidth=0.5, label='GREEN', zorder=3)
bars_y = ax_b.bar(x, yellow_pct, w, bottom=green_pct, color=COLORS['yellow'],
                  edgecolor='white', linewidth=0.5, label='YELLOW', zorder=3)
bars_r = ax_b.bar(x, red_pct, w,
                  bottom=[g + y for g, y in zip(green_pct, yellow_pct)],
                  color=COLORS['red'], edgecolor='white',
                  linewidth=0.5, label='RED', zorder=3)

# Labels on bars
for i in range(len(cell_types)):
    # Green count + percentage
    if green_pct[i] > 8:
        ax_b.text(i, green_pct[i] / 2,
                  f'{greens[i]}\n({green_pct[i]:.0f}%)',
                  ha='center', va='center', fontsize=FONTS['bar_label'] - 1,
                  fontweight='bold', color='white')
    # Yellow count + percentage
    if yellow_pct[i] > 8:
        ax_b.text(i, green_pct[i] + yellow_pct[i] / 2,
                  f'{yellows[i]}\n({yellow_pct[i]:.0f}%)',
                  ha='center', va='center', fontsize=FONTS['bar_label'],
                  fontweight='bold', color='#424242')
    # Red count + percentage
    if red_pct[i] > 8:
        ax_b.text(i, green_pct[i] + yellow_pct[i] + red_pct[i] / 2,
                  f'{reds[i]}\n({red_pct[i]:.0f}%)',
                  ha='center', va='center', fontsize=FONTS['bar_label'],
                  fontweight='bold', color='white')
    # Pass rate annotation above bar
    ax_b.text(i, 103, f'pass: {pass_rates[i]*100:.0f}%',
              ha='center', va='bottom', fontsize=FONTS['annotation'],
              fontweight='bold', color=COLORS['primary'])
    # Sample size below x-tick label
    ax_b.text(i, -15, f'n={totals[i]}', ha='center', va='top',
              fontsize=FONTS['bar_label'], color=COLORS['text_light'],
              clip_on=False)

ax_b.set_xticks(x)
ax_b.set_xticklabels(cell_types, fontsize=FONTS['tick'], fontweight='bold')
ax_b.tick_params(axis='x', pad=4)
ax_b.set_ylim(0, 122)
ax_b.set_yticks([0, 25, 50, 75, 100])
ax_b.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
style_axis(ax_b, ylabel='Candidates (%)')
ax_b.set_title('Therapeutic Candidate\nValidation Verdicts', fontsize=FONTS['subtitle'],
               fontweight='bold', pad=12, color=COLORS['text'])
ax_b.legend(fontsize=FONTS['legend'] - 1, loc='upper center',
            framealpha=0.9, ncol=3, columnspacing=0.8,
            bbox_to_anchor=(0.5, 0.98))
add_panel_label(ax_b, 'B', x=-0.12)

# ══════════════════════════════════════════════════════════════════════════════
# Panel C: Top-10 Specificity Score Distribution
# ══════════════════════════════════════════════════════════════════════════════
ax_c = fig.add_subplot(gs[2])

ct_display = {'hepg2': 'HepG2', 'k562': 'K562', 'wtc11': 'WTC11'}
ct_plot_colors = {
    'hepg2': '#1565C0',
    'k562': '#7B1FA2',
    'wtc11': '#00695C',
}

spec_data = []
spec_names = []
spec_colors = []
for ct in ct_keys:
    vals = top10_data.get(ct, [])
    if vals:
        spec_data.append(vals)
        spec_names.append(ct_display[ct])
        spec_colors.append(ct_plot_colors[ct])

for i, (vals, name, color) in enumerate(zip(spec_data, spec_names, spec_colors)):
    rng = np.random.RandomState(42 + i)
    jitter = rng.uniform(-0.18, 0.18, len(vals))
    ax_c.scatter(np.full(len(vals), i) + jitter, vals,
                 c=color, s=45, alpha=0.75, edgecolor='white',
                 linewidth=0.5, zorder=3)
    # Mean line
    mean_val = np.mean(vals)
    ax_c.hlines(mean_val, i - 0.3, i + 0.3, color=color, lw=2.5, zorder=4)
    # Place mu label: left-aligned for last item to avoid right-edge clipping
    if i == len(spec_data) - 1:
        ax_c.text(i - 0.35, mean_val + 0.08, f'\u03bc={mean_val:.2f}', fontsize=9,
                  fontweight='bold', color=color, va='bottom', ha='right')
    else:
        ax_c.text(i + 0.35, mean_val, f'\u03bc={mean_val:.2f}', fontsize=9,
                  fontweight='bold', color=color, va='center')

    # Also annotate the ISM mean_spec from the larger sample
    ct_key = ct_keys[i]
    ism_mean = ism_data[ct_key]['mean_spec']
    ax_c.hlines(ism_mean, i - 0.3, i + 0.3, color=color, lw=1.2,
                linestyle='--', zorder=4, alpha=0.6)

# Threshold reference lines
max_y = max(max(v) for v in spec_data) if spec_data else 4
ax_c.axhline(1.0, color=COLORS['green'], ls=':', lw=1.5, alpha=0.7, zorder=1)
ax_c.text(-0.4, 1.05, 'spec = 1.0',
          fontsize=FONTS['annotation'] - 2, color=COLORS['green'], ha='left',
          fontstyle='italic', fontweight='bold', alpha=0.9)

# Custom legend for mean lines
legend_elements = [
    Line2D([0], [0], color='gray', lw=2.5, label='Top-10 mean'),
    Line2D([0], [0], color='gray', lw=1.2, ls='--', alpha=0.6,
           label='ISM mean (n=200)'),
]
ax_c.legend(handles=legend_elements, fontsize=8, loc='upper right',
            framealpha=0.9)

ax_c.set_xticks(range(len(spec_names)))
ax_c.set_xticklabels(spec_names, fontsize=FONTS['tick'], fontweight='bold')
ax_c.set_xlim(-0.5, len(spec_names) - 0.5)
ax_c.set_ylim(0, max_y + 0.5)
style_axis(ax_c, ylabel='Specificity Score')
ax_c.set_title('Top-10 Candidate\nSpecificity Scores', fontsize=FONTS['subtitle'],
               fontweight='bold', pad=12, color=COLORS['text'])
add_panel_label(ax_c, 'C', x=-0.12)

# ── SAVE ─────────────────────────────────────────────────────────────────────
fig.subplots_adjust(left=0.03, right=0.97, bottom=0.10, top=0.90, wspace=0.38)
save_pres_fig(fig, 'pres_fig7_oraclecheck')
print('Done.')
