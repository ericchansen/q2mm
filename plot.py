#!/usr/bin/python
import argparse
import calculate
import filetypes
import logging
import logging.config
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from matplotlib import cm
from matplotlib import rcParams
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import os
from setup_logging import log_uncaught_exceptions, remove_logs
import sys
import yaml

widthpt = 340.71031
ptperin = 72.27
# goldenmean = (np.sqrt(5.0)-1.0)/2.0 # An apparently pleasing ratio.
goldenmean = (np.sqrt(5.0)-0.6)/2.0 # Modified ratio.
fontsize = 8
linewidth = 0.5
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300
rcParams['figure.figsize'] = 3.3, 3.3*goldenmean
rcParams['axes.linewidth'] = linewidth
rcParams['lines.linewidth'] = linewidth # Might not be necessary.
rcParams['patch.linewidth'] = linewidth # Might not be necessary.
rcParams['xtick.major.size'] = 2
rcParams['ytick.major.size'] = 2
rcParams['font.size'] = fontsize
rcParams['axes.labelsize'] = fontsize
rcParams['axes.titlesize'] = fontsize
rcParams['xtick.labelsize'] = fontsize # Might only be used for 2D.
rcParams['ytick.labelsize'] = fontsize # Might only be used for 2D.
rcParams['font.family'] = 'sans-serif'
# rcParams['font.family'] = 'serif'
# rcParams['font.serif'] = ['Computer Modern Roman']
rcParams['text.antialiased'] = True
rcParams['text.usetex'] = True

# Limits to plots.
xlim = 355
ylim = 355
# Color levels for contour.
colorlevels = 10

# Setup logging.
logger = logging.getLogger(__name__)

def setup_matrices(data):
    xs = [int(d.index[0]) for d in data]
    ys = [int(d.index[1]) for d in data]
    zs = [d.value for d in data] # Should already be float.
    xset = np.array(sorted(set(xs)))
    yset = np.array(sorted(set(ys)))
    # Doesn't work if xs and ys contain strings.
    xmat, ymat = np.meshgrid(xset, yset, indexing='ij')
    zmat = np.zeros(np.shape(xmat))
    for x, y, z in zip(xs, ys, zs):
        ix = np.where(xset == x)[0][0]
        iy = np.where(yset == y)[0][0]
        zmat[ix, iy] = z
    return xmat, ymat, zmat
    
def plot_3d(xmat, ymat, zmat, custom=False):
    fig = plt.figure()
    axs = fig.gca(projection='3d')
    srf = axs.plot_surface(xmat, ymat, zmat, rstride=1, cstride=1,
                           cmap=cm.coolwarm, linewidth=0, antialiased=True)
    plt.locator_params(axis='z', nbins=8)
    axs.view_init(elev=70, azim=-65)
    if custom:
        axs.w_xaxis.set_pane_color((0, 0, 0, 0))
        axs.w_yaxis.set_pane_color((0, 0, 0, 0))
        axs.w_zaxis.set_pane_color((0, 0, 0, 0))
        # Setup for 3.3 in.
        plt.locator_params(axis='x', nbins=6)
        plt.locator_params(axis='y', nbins=6)
        plt.locator_params(axis='z', nbins=4)
        # Adjust xticks.
        xticks = axs.xaxis.get_major_ticks()
        xticks[0].label1.set_visible(False)
        xticks[0].tick1line.set_visible(False)
        xticks[-1].label1.set_visible(False)
        xticks[-1].tick1line.set_visible(False)
        # Adjust yticks.
        yticks = axs.yaxis.get_major_ticks()
        yticks[0].label1.set_visible(False)
        yticks[0].tick1line.set_visible(False)
        yticks[-1].label1.set_visible(False)
        yticks[-1].tick1line.set_visible(False)
        # Adjust z ticks.
        # zticks = axs.zaxis.get_major_ticks()
        # zticks[0].label1.set_visible(False)
        # zticks[-1].label1.set_visible(False)
        # Set limits.
        axs.set_xlim3d(0, xlim)
        axs.set_ylim3d(0, ylim)
        # For whatever reason, the tight layout doesn't work well here.
        fig.tight_layout()
        plt.subplots_adjust(left=-0.1)
    return fig, axs

def plot_conf(xmat, ymat, zmat):
    fig = plt.figure()
    axs1 = plt.contourf(xmat, ymat, zmat, colorlevels, antialiased=False,
                        cmap=cm.coolwarm)
    axs2 = plt.contour(xmat, ymat, zmat, axs1.levels, colors='k',
                       antialiased=True, linewidths=linewidth)
    cbar = plt.colorbar(axs1)
    return fig, axs1, axs2, cbar

def process_args(args):
    parser = argparse.ArgumentParser(
        description='Script to aid in plotting 2D scan energies.')
    disp_opts = parser.add_argument_group('Display options')
    mae_opts = parser.add_argument_group('Arguments for mae files')
    out_opts = parser.add_argument_group('Output options')
    # parser.add_argument(
    #     'input', metavar='filename.mae', type=str,
    #     help='Input filename.')
    parser.add_argument(
        '--calc', '-c', type=str, metavar='"arguments for calculate.py"',
        help='Arguments for calculate.py.')
    # parser.add_argument(
    #     '--units', type=str, metavar='kJ mol^-1', default='kJ mol^-1',
    #     choices = ['kJ mol^-1', 'kcal mol^-1', 'Hartree'],
    #     help='Units for output.')
    disp_opts.add_argument(
        '-c3d', action='store_true',
        help='Use my custom settings for 3D plot.')
    disp_opts.add_argument(
        '-cdsc', action='store_true',
        help='Use my custom text.')
    disp_opts.add_argument(
        '--display', '-d', default='3d',
        choices = ['3d', 'conf'],
        help='Type of plot to produce.')
    disp_opts.add_argument(
        '--text', '-t', type=str, nargs=2,
        metavar='"text;more text; more" "148,146;315,140"',
        help='Adds text to figure.')
    disp_opts.add_argument(
        '-xl', type=str, metavar='"some string"',
        help='x-axis label.')
    disp_opts.add_argument(
        '-yl', type=str, metavar='"some string"',
        help='y-axis label.')
    disp_opts.add_argument(
        '-zl', type=str, metavar='"some string"',
        help='z-axis label.')
    mae_opts.add_argument(
        '-x', type=str, metavar='r_j_d1',
        help='Scan coordinate used for x-axis.')
    mae_opts.add_argument(
        '-y', type=str, metavar='r_j_d2',
        help='Scan coordinate used for y-axis')
    mae_opts.add_argument(
        '-z', type=str, metavar='r_mmod_Potential_Energy-MM3*',
        help='Label used for energies.')
    out_opts.add_argument(
        '--output', '-o', type=str, metavar='filename',
        help='Saves instead of displays.')
    out_opts.add_argument(
        '--filetype', '-f', type=str, metavar='jpg', default='jpg',
        help='Output filetype.')
    out_opts.add_argument(
        '-xi', type=float,
        help='Length of x dimension in inches.')
    out_opts.add_argument(
        '-yi', type=float,
        help='Length of y dimension in inches.')
    opts = vars(parser.parse_args(args))
    opts['calc'] = opts['calc'].split()
    # Old method.
    # mae = filetypes.MaeFile(opts['input'], directory=os.getcwd())
    # logger.debug('{} structures in {}.'.format(len(mae.raw_data), mae.filename))
    data = calculate.process_args(opts['calc'])
    logger.debug('Units of 1st data point: {}'.format(data[0].units))
    xmat, ymat, zmat = setup_matrices(data)
    if opts['display'] == '3d':
        fig, axs = plot_3d(xmat, ymat, zmat, opts['custom'])
    if opts['display'] == 'conf':
        fig, axs1, axs2, cbar = plot_conf(xmat, ymat, zmat)
    if opts['xl']:
        # fig.set_xlabel(opts['xl'])
        plt.xlabel(opts['xl'])
    if opts['yl']:
        plt.ylabel(opts['yl'])
    if opts['zl'] and opts['display'] == '3d':
        axs.set_zlabel(opts['zl'])
    elif opts['zl'] and opts['display'] == 'conf':
        cbar.ax.set_ylabel(opts['zl'])
    if opts['text']:
        texts = opts['text'][0]
        texts = texts.split(';')
        coords = opts['text'][1]
        coords = coords.split(';')
        for text, coord in zip(texts, coords):
            coord = map(int, coord.split(','))
            logger.debug('Adding "{}" to {}.'.format(text, coord))
            plt.text(coord[0], coord[1], text,
                     path_effects=[patheffects.withStroke(
                         linewidth=2, foreground='w')])
    if opts['cdsc']:
        texts = [r'\textbf{2}_{\textbf{a}}',
                 r'\textbf{2}_{\textbf{a}}',
                 r'\textbf{2}_{\textbf{b}}',
                 r'\textbf{2}_{\textbf{c}}^{\textbf{ts}}',
                 r'\textbf{2}_{\textbf{d}}',
                 r'\textbf{2}_{\textbf{e}}^{\textbf{ts}}',
                 r'\textbf{2}_{\textbf{f}}^{\textbf{ts}}']
        coords = [[35, 35],
                  [310, 310],
                  [180, 5],
                  [5, 5],
                  [171, 171],
                  [5, 100],
                  [96, 163]]
        for text, coord in zip(texts, coords):
            logger.debug('Adding "{}" to {}.'.format(text, coord))
            plt.text(coord[0], coord[1], text,
                     path_effects=[patheffects.withStroke(
                         linewidth=2, foreground='w')])
    if opts['output']:
        if opts['display'] == 'conf':
            # Has to be set after adding labels.
            fig.tight_layout()
        logger.debug('DPI: {}'.format(fig.get_dpi()))
        logger.debug('Size (in): {}'.format(fig.get_size_inches()))
        logger.debug('Size (pixels): {}'.format([x*fig.get_dpi() for x in fig.get_size_inches()]))
        plt.savefig('{}.{}'.format(opts['output'], opts['filetype']), dpi=300)
    else:
        plt.show()

if __name__ == '__main__':
    # Setup logs.
    remove_logs()
    sys.excepthook = log_uncaught_exceptions
    with open('options/logging.yaml', 'r') as f:
        log_config = yaml.load(f)
    logging.config.dictConfig(log_config)
    # Process arguments.
    process_args(sys.argv[1:])
