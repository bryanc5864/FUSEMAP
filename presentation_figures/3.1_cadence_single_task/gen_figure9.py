#!/usr/bin/env python3
"""
Generate Figure 9: Complete PhysInformer Zero-Shot Transfer Matrix.

Heatmap showing mean Pearson r for physics prediction transfer
between all source-target combinations (from paper Table 11).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# Data from paper Table 11
# =============================================================================
sources = ['K562', 'HepG2', 'WTC11']
targets_label = ['K562', 'HepG2', 'WTC11', 'S2\n(Drosophila)', 'Maize', 'Sorghum', 'Arabidopsis']

# Transfer matrix (source=row, target=col), NaN for self or missing
data = np.array([
    [np.nan, 0.847, 0.839, 0.729, 0.680, 0.679, 0.656],  # K562 ->
    [0.657,  np.nan, np.nan, 0.464, np.nan, np.nan, np.nan],  # HepG2 ->
    [0.832,  0.829, np.nan, 0.649, np.nan, np.nan, np.nan],  # WTC11 ->
])

n_rows, n_cols = data.shape

# =============================================================================
# Figure setup
# =============================================================================
fig, ax = plt.subplots(figsize=(14, 8))

# Colormap
cmap = plt.cm.RdYlGn
vmin, vmax = 0.4, 0.9

# Draw heatmap manually for full control over NaN cells
for i in range(n_rows):
    for j in range(n_cols):
        val = data[i, j]
        if np.isnan(val):
            # Gray cell for missing / self
            rect = plt.Rectangle((j, i), 1, 1, facecolor='#D0D0D0',
                                 edgecolor='white', linewidth=2)
            ax.add_patch(rect)
            # Check if this is a self-transfer diagonal
            src_name = sources[i]
            tgt_name = targets_label[j].split('\n')[0]
            if src_name == tgt_name:
                ax.text(j + 0.5, i + 0.5, 'self', ha='center', va='center',
                        fontsize=11, fontweight='bold', color='#888888',
                        fontstyle='italic')
            else:
                ax.text(j + 0.5, i + 0.5, 'N/A', ha='center', va='center',
                        fontsize=11, fontweight='bold', color='#888888',
                        fontstyle='italic')
        else:
            # Colored cell
            norm_val = (val - vmin) / (vmax - vmin)
            norm_val = np.clip(norm_val, 0, 1)
            color = cmap(norm_val)
            rect = plt.Rectangle((j, i), 1, 1, facecolor=color,
                                 edgecolor='white', linewidth=2)
            ax.add_patch(rect)
            # Text color: dark for light backgrounds, white for dark
            luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
            text_color = 'white' if luminance < 0.55 else 'black'
            ax.text(j + 0.5, i + 0.5, f'{val:.3f}', ha='center', va='center',
                    fontsize=14, fontweight='bold', color=text_color)

# Set limits
ax.set_xlim(0, n_cols)
ax.set_ylim(0, n_rows)
ax.invert_yaxis()

# Axis labels
ax.set_xticks([j + 0.5 for j in range(n_cols)])
ax.set_xticklabels(targets_label, fontsize=12, fontweight='bold')
ax.set_yticks([i + 0.5 for i in range(n_rows)])
ax.set_yticklabels(sources, fontsize=12, fontweight='bold')

ax.set_xlabel('Target Dataset', fontsize=14, fontweight='bold', labelpad=10)
ax.set_ylabel('Source Dataset', fontsize=14, fontweight='bold', labelpad=10)

# Remove top/right spines
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

# Tick params
ax.tick_params(axis='both', which='both', length=0)

# =============================================================================
# Colorbar
# =============================================================================
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, aspect=25)
cbar.set_label('Mean Pearson r', fontsize=12, fontweight='bold')
cbar.ax.tick_params(labelsize=11)

# =============================================================================
# Block structure annotations (brackets below the x-axis)
# =============================================================================
bracket_y = n_rows + 0.45

# Within Human: columns 0-2
ax.annotate('', xy=(0, bracket_y), xytext=(3, bracket_y),
            arrowprops=dict(arrowstyle='-', lw=2.5, color='#2166AC'),
            annotation_clip=False)
for xp in [0, 3]:
    ax.plot([xp, xp], [bracket_y - 0.08, bracket_y + 0.08],
            color='#2166AC', lw=2.5, clip_on=False)
ax.text(1.5, bracket_y + 0.25, 'Within Human', ha='center', va='top',
        fontsize=12, fontweight='bold', color='#2166AC',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#D6E8F7',
                  edgecolor='#2166AC', alpha=0.9),
        clip_on=False)

# Cross-Species: column 3
ax.annotate('', xy=(3, bracket_y), xytext=(4, bracket_y),
            arrowprops=dict(arrowstyle='-', lw=2.5, color='#D4790E'),
            annotation_clip=False)
for xp in [3, 4]:
    ax.plot([xp, xp], [bracket_y - 0.08, bracket_y + 0.08],
            color='#D4790E', lw=2.5, clip_on=False)
ax.text(3.5, bracket_y + 0.25, 'Cross-Species', ha='center', va='top',
        fontsize=12, fontweight='bold', color='#D4790E',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FDE8C8',
                  edgecolor='#D4790E', alpha=0.9),
        clip_on=False)

# Cross-Kingdom: columns 4-6
ax.annotate('', xy=(4, bracket_y), xytext=(7, bracket_y),
            arrowprops=dict(arrowstyle='-', lw=2.5, color='#8B2252'),
            annotation_clip=False)
for xp in [4, 7]:
    ax.plot([xp, xp], [bracket_y - 0.08, bracket_y + 0.08],
            color='#8B2252', lw=2.5, clip_on=False)
ax.text(5.5, bracket_y + 0.25, 'Cross-Kingdom', ha='center', va='top',
        fontsize=12, fontweight='bold', color='#8B2252',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#F0D0E0',
                  edgecolor='#8B2252', alpha=0.9),
        clip_on=False)

# =============================================================================
# Dashed lines to separate blocks on the heatmap
# =============================================================================
ax.axvline(x=3, color='#555555', linestyle='--', linewidth=1.5, alpha=0.6)
ax.axvline(x=4, color='#555555', linestyle='--', linewidth=1.5, alpha=0.6)

# =============================================================================
# Titles
# =============================================================================
fig.suptitle('Figure 9. Complete PhysInformer Zero-Shot Transfer Matrix',
             fontsize=18, fontweight='bold', y=0.97)
ax.set_title(
    'Within-human r>0.83  |  Cross-species r\u22480.65\u20130.73  |  '
    'Cross-kingdom r\u22480.66\u20130.68',
    fontsize=13, fontweight='normal', color='#444444', pad=15,
    fontstyle='italic')

# =============================================================================
# Summary statistics text box
# =============================================================================
within_human = data[:, :3]
wh_mean = np.nanmean(within_human)

cross_species = data[:, 3:4]
cs_mean = np.nanmean(cross_species)

cross_kingdom = data[:, 4:]
ck_mean = np.nanmean(cross_kingdom)

summary_text = (
    f"Block Means:\n"
    f"  Within Human:   {wh_mean:.3f}\n"
    f"  Cross-Species:  {cs_mean:.3f}\n"
    f"  Cross-Kingdom:  {ck_mean:.3f}"
)
ax.text(0.98, 0.02, summary_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#FAFAFA',
                  edgecolor='#CCCCCC', alpha=0.9),
        fontfamily='monospace')

# Adjust layout
plt.subplots_adjust(bottom=0.22, top=0.88, left=0.10, right=0.88)

# =============================================================================
# Save
# =============================================================================
output_base = '/home/bcheng/sequence_optimization/FUSEMAP/presentation_figures/figure9_transfer_heatmap'
fig.savefig(output_base + '.png', dpi=200, bbox_inches='tight', facecolor='white')
fig.savefig(output_base + '.pdf', bbox_inches='tight', facecolor='white')
print(f"Saved to {output_base}.png and {output_base}.pdf")
plt.close()
