"""
FUSEMAP Poster Style Configuration
===================================
Unified style inspired by STAIR poster: clean, compact, professional.
All poster figures import this module for consistent styling.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── COLOR PALETTE ────────────────────────────────────────────────────────────
# Limited, professional palette inspired by STAIR poster
COLORS = {
    # Primary organism colors
    'human':      '#3B7DD8',   # Blue
    'drosophila': '#8E44AD',   # Purple
    'plant':      '#27AE60',   # Green
    'yeast':      '#E67E22',   # Orange

    # Functional colors
    'primary':    '#2C3E50',   # Dark slate (text, borders)
    'secondary':  '#7F8C8D',   # Gray
    'accent':     '#E74C3C',   # Red (highlights, warnings)
    'success':    '#27AE60',   # Green (positive results)
    'warning':    '#F39C12',   # Amber

    # SOTA / baseline
    'sota':       '#95A5A6',   # Light gray
    'cadence':    '#3B7DD8',   # Blue (same as human for CADENCE)

    # Physics feature families
    'bending':    '#27AE60',   # Green
    'thermo':     '#E67E22',   # Orange
    'entropy':    '#3498DB',   # Light blue
    'stiffness':  '#9B59B6',   # Purple
    'advanced':   '#F1C40F',   # Gold
    'pwm':        '#E74C3C',   # Red

    # Transfer scenarios
    'within':     '#3B7DD8',   # Blue
    'cross_sp':   '#E67E22',   # Orange
    'cross_kg':   '#27AE60',   # Green

    # Neutral
    'bg':         '#FFFFFF',
    'grid':       '#ECEFF1',
    'border':     '#BDC3C7',
    'text':       '#2C3E50',
    'text_light': '#7F8C8D',
}

# ── FONT SIZES (poster-appropriate) ──────────────────────────────────────────
FONTS = {
    'title':      14,
    'subtitle':   11,
    'axis_label': 11,
    'tick':       9,
    'annotation': 9,
    'legend':     9,
    'bar_label':  8,
    'caption':    8,
}

# ── FIGURE SIZES (poster panel-appropriate, inches) ──────────────────────────
# Poster panels are typically 4-7 inches wide
SIZES = {
    'single':     (5.5, 4.0),    # Single panel
    'wide':       (8.0, 4.0),    # Wide single panel
    'double':     (8.0, 6.0),    # 2x1 or 1x2
    'quad':       (8.0, 6.5),    # 2x2 grid
    'triple_wide':(10.0, 4.0),   # 3x1
    'six_panel':  (10.0, 7.0),   # 2x3 or 3x2
}


def apply_poster_style():
    """Apply global matplotlib rcParams for poster quality."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
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
    """Apply clean poster styling to a single axis."""
    if title:
        ax.set_title(title, fontsize=FONTS['title'], fontweight='bold',
                      pad=8, color=COLORS['text'])
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


def add_fig_label(fig, label, x=0.02, y=0.98):
    """Add 'Fig X.' label in STAIR poster style."""
    fig.text(x, y, label, fontsize=FONTS['title'], fontweight='bold',
             color=COLORS['text'], va='top', ha='left',
             transform=fig.transFigure)


def add_panel_label(ax, label, x=-0.08, y=1.05):
    """Add panel label (A, B, C...) to subplot."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=FONTS['title'] + 1, fontweight='bold',
            color=COLORS['text'], va='top', ha='left')


def save_poster_fig(fig, name, outdir=None):
    """Save figure as PNG (300 dpi) and PDF."""
    if outdir is None:
        outdir = '/home/bcheng/sequence_optimization/FUSEMAP/presentation_figures/poster_figures'
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
