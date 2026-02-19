"""
FUSEMAP Presentation Style Configuration
=========================================
Purple/lavender/blue color scheme for presentation figures.
All pres figures import this module for consistent styling.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── COLOR PALETTE ────────────────────────────────────────────────────────────
COLORS = {
    'primary':    '#4A148C',   # Deep purple (text, borders)
    'accent1':    '#7B1FA2',   # Purple
    'accent2':    '#9C27B0',   # Medium purple
    'accent3':    '#AB47BC',   # Light purple
    'lavender':   '#CE93D8',   # Lavender
    'blue1':      '#1565C0',   # Dark blue
    'blue2':      '#1E88E5',   # Medium blue
    'blue3':      '#42A5F5',   # Light blue
    'periwinkle': '#7986CB',   # Blue-lavender
    'bg':         '#FFFFFF',
    'bg_light':   '#F3E5F5',   # Faint purple tint
    'grid':       '#E1BEE7',   # Light purple grid
    'border':     '#B39DDB',   # Lavender border
    'text':       '#212121',
    'text_light': '#757575',

    # Organism colors (purple-blue spectrum)
    'human':      '#1565C0',   # Dark blue
    'drosophila': '#7B1FA2',   # Purple
    'plant':      '#4CAF50',   # Green (contrast)
    'yeast':      '#FF8F00',   # Amber (contrast)

    # Traffic light for OracleCheck
    'green':      '#43A047',
    'yellow':     '#FDD835',
    'red':        '#E53935',

    # Physics feature families
    'bending':    '#4CAF50',
    'thermo':     '#FF8F00',
    'entropy':    '#42A5F5',
    'stiffness':  '#AB47BC',
    'advanced':   '#FDD835',
    'pwm':        '#E53935',

    # Transfer scenarios
    'within':     '#1565C0',
    'cross_sp':   '#7B1FA2',
    'cross_kg':   '#4CAF50',

    # SOTA / baseline
    'sota':       '#B0BEC5',
    'cadence':    '#1565C0',
}

# ── FONT SIZES ───────────────────────────────────────────────────────────────
FONTS = {
    'title':      20,
    'subtitle':   16,
    'axis_label': 15,
    'tick':       13,
    'annotation': 12,
    'legend':     12,
    'bar_label':  11,
    'caption':    11,
    'flowchart':  12,   # text inside flowchart boxes
    'flowchart_detail': 10,  # detail text in flowcharts
}

# ── FIGURE SIZES (inches) ───────────────────────────────────────────────────
SIZES = {
    'single':      (5.5, 4.0),
    'wide':        (8.0, 4.0),
    'double':      (8.0, 6.0),
    'quad':        (10.0, 8.0),
    'triple_wide': (10.0, 4.0),
    'six_panel':   (10.0, 7.0),
    'full':        (12.0, 9.0),
}


def apply_pres_style():
    """Apply global matplotlib rcParams for presentation quality."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Nimbus Sans', 'Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'],
        'pdf.fonttype': 42,
        'font.size': FONTS['tick'],
        'font.weight': 'normal',
        'axes.labelsize': FONTS['axis_label'],
        'axes.titlesize': FONTS['title'],
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'axes.linewidth': 1.0,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.labelsize': FONTS['tick'],
        'ytick.labelsize': FONTS['tick'],
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'legend.fontsize': FONTS['legend'],
        'legend.framealpha': 0.9,
        'legend.edgecolor': COLORS['border'],
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'figure.dpi': 150,
    })


def style_axis(ax, title=None, xlabel=None, ylabel=None, grid_y=True):
    """Apply clean presentation styling to a single axis."""
    if title:
        ax.set_title(title, fontsize=FONTS['subtitle'], fontweight='bold',
                     pad=12, color=COLORS['text'])
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FONTS['axis_label'],
                      fontweight='bold', color=COLORS['text'])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FONTS['axis_label'],
                      fontweight='bold', color=COLORS['text'])
    if grid_y:
        ax.yaxis.grid(True, alpha=0.3, linewidth=0.6, color=COLORS['grid'],
                      zorder=0)
        ax.set_axisbelow(True)
    ax.tick_params(axis='both', which='major', labelsize=FONTS['tick'],
                   colors=COLORS['text'])


def add_panel_label(ax, label, x=-0.08, y=1.10):
    """Add panel label (A, B, C...) to subplot."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=FONTS['title'] + 2, fontweight='bold',
            color=COLORS['primary'], va='top', ha='left')


def save_pres_fig(fig, name, outdir=None):
    """Save figure as PNG (300 dpi) and PDF."""
    if outdir is None:
        outdir = '/home/bcheng/sequence_optimization/FUSEMAP/presentation_figures/pres_figures'
    png_path = f'{outdir}/{name}.png'
    pdf_path = f'{outdir}/{name}.pdf'
    fig.savefig(png_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    fig.savefig(pdf_path, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'  Saved: {png_path}')
    print(f'  Saved: {pdf_path}')
    return png_path
