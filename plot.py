#!/usr/bin/python
from itertools import izip
import argparse
import logging
import logging.config
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from matplotlib import cm
from matplotlib import rcParams
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import os
import sys
import yaml

# q2mm modules
import constants
import filetypes

logger = logging.getLogger(__name__)

COLORLEVELS = 10
LINEWIDTH = 0.5
XLIM = 355
YLIM = 355
X_LABEL = 'r_j_d1'
Y_LABEL = 'r_j_d2'
# Z_LABEL = 'r_mmod_Potential_Energy-MM3*'
# Z_LABEL = 'r_j_Gas_Phase_Energy'
Z_LABEL_MM = 'r_mmod_Potential_Energy-MM3*'
Z_LABEL_QM = 'r_j_Gas_Phase_Energy'
X_AXIS_LABEL = '$\phi_{1}$ (degrees)'
Y_AXIS_LABEL = '$\phi_{2}$ (degrees)'
Z_AXIS_LABEL = 'Energy (kJ/mol)'
WIDTHPT = 340.71031
PTPERIN = 72.27
# GOLDENMEAN = (np.sqrt(5.) - 1.0) / 2. # apparently pleasing ratio
GOLDENMEAN = (np.sqrt(5.) - 0.6) / 2. # modified
FONTSIZE = 8
# FONTSIZE = 15
LINEWIDTH = 0.5
# LINEWIDTH = 3

# rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300
rcParams['figure.figsize'] = 3.3, 3.3*GOLDENMEAN
rcParams['lines.linewidth'] = LINEWIDTH
rcParams['axes.linewidth'] = LINEWIDTH # changes bounding box for 2d
# rcParams['patch.linewidth'] = LINEWIDTH # necessary?
rcParams['xtick.major.size'] = 2
rcParams['ytick.major.size'] = 2
rcParams['font.size'] = FONTSIZE
# rcParams['axes.labelsize'] = FONTSIZE
# rcParams['axes.titlesize'] = FONTSIZE # no title so...
# rcParams['xtick.labelsize'] = FONTSIZE # only for 2d?
# rcParams['ytick.labelsize'] = FONTSIZE # only for 2d?
rcParams['font.family'] = 'sans-serif'
# rcParams['font.family'] = 'serif'
# rcParams['font.serif'] = ['Computer Modern Roman']
# rcParams['text.antialiased'] = True
rcParams['text.usetex'] = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Generates matplotlib plots of energy surfaces from '
        'Maestro files (.mae).')
    parser.add_argument(
        '--display', '-d', type=str, default='3d', choices=['3d', '3dc', 'conf'],
        help='Type of graph to generate.')
    parser.add_argument(
        '--extension', '-e', type=str, nargs='+', default=['jpg'], metavar='png',
        help='Generate a file for each extension. Not sure what all works '
        'because these arguments are passed on to the matplotlib innards, but '
        'I use png, jpg, pdf, and eps regularly.')
    parser.add_argument(
        '--filename', '-f', type=str,
        help='Maestro file name (include .mae).')
    parser.add_argument(
        '--key', '-k', type=str, choices=['msa', 'dsc'],
        help='Hard coded options used for specific plots from past projects.')
    parser.add_argument(
        '--output' ,'-o', type=str, metavar='string',
        help="Output plot filename (don't include the extension).")
    parser.add_argument(
        '--mode', '-m', choices=['mm', 'qm'], default='mm',
        help='Look for energies by "{}" (mm) or "{}" (qm).'.format(
            Z_LABEL_MM, Z_LABEL_QM))
    format_args = parser.add_argument_group('format options')
    format_args.add_argument(
        '-xl', type=str, default=X_AXIS_LABEL, metavar='string',
        help='X-axis label.')
    format_args.add_argument(
        '-yl', type=str, default=Y_AXIS_LABEL, metavar='string',
        help='Y-axis label.')
    format_args.add_argument(
        '-zl', type=str, default=Z_AXIS_LABEL, metavar='string',
        help='Z-axis label.')
    opts = parser.parse_args(sys.argv[1:])
    if opts.mode == 'mm':
        Z_LABEL = Z_LABEL_MM
    elif opts.mode == 'qm':
        Z_LABEL = Z_LABEL_QM

    # data extraction tools
    path, ext = os.path.splitext(opts.filename)
    if ext == '.mae':
        mae = filetypes.Mae(opts.filename)
    
    # data setup
    x_data = []
    y_data = []
    z_data = []
    for structure in mae.structures:
        x_data.append(float(structure.props[X_LABEL]))
        y_data.append(float(structure.props[Y_LABEL]))
        z_data.append(float(structure.props[Z_LABEL]))
    print('x: {} y: {} z: {}'.format(len(x_data), len(y_data), len(z_data)))
    if opts.mode == 'qm':
        z_data = [x * constants.hartree_to_kjmol for x in z_data]
    minimum = min(z_data)
    z_data = [x - minimum for x in z_data]
    x_set = np.array(sorted(set(x_data)))
    y_set = np.array(sorted(set(y_data)))
    x_mat, y_mat = np.meshgrid(x_set, y_set, indexing='ij')
    z_mat = np.zeros(np.shape(x_mat))
    for x, y, z in izip(x_data, y_data, z_data):
        ix = np.where(x_set == x)[0][0]
        iy = np.where(y_set == y)[0][0]
        z_mat[ix, iy] = z

    # display 3d
    if '3d' in opts.display:
        fig = plt.figure()
        axs = fig.gca(projection='3d')
        srf = axs.plot_surface(x_mat, y_mat, z_mat, rstride=1, cstride=1,
                               cmap=cm.coolwarm, linewidth=0, antialiased=True)
        plt.locator_params(axis='z', nbins=8)
        axs.view_init(elev=70,azim=-65)
        if 'c' in opts.display:
            axs.w_xaxis.set_pane_color((0, 0, 0, 0))
            axs.w_yaxis.set_pane_color((0, 0, 0, 0))
            axs.w_zaxis.set_pane_color((0, 0, 0, 0))
            # setup for 3.3" figures
            plt.locator_params(axis='x', nbins=6)
            plt.locator_params(axis='y', nbins=6)
            plt.locator_params(axis='z', nbins=4)
            # xticks
            xticks = axs.xaxis.get_major_ticks()
            xticks[0].label1.set_visible(False)
            xticks[0].tick1line.set_visible(False)
            xticks[-1].label1.set_visible(False)
            xticks[-1].tick1line.set_visible(False)
            # yticks
            yticks = axs.yaxis.get_major_ticks()
            yticks[0].label1.set_visible(False)
            yticks[0].tick1line.set_visible(False)
            yticks[-1].label1.set_visible(False)
            yticks[-1].tick1line.set_visible(False)
            # zticks
            # zticks = axs.zaxis.get_major_ticks()
            # zticks[0].label1.set_visible(False)
            # zticks[-1].label1.set_visible(False)
            # Set limits.
            axs.set_xlim3d(0, XLIM)
            axs.set_ylim3d(0, YLIM)
            # tight layout doesn't work well, so make changes after setting.
            fig.tight_layout()
            plt.subplots_adjust(left=-0.1, right=0.95)
            axs.xaxis._axinfo['label']['space_factor'] = 2.
            axs.yaxis._axinfo['label']['space_factor'] = 2.
            axs.zaxis._axinfo['label']['space_factor'] = 2.

    # display contour surface
    elif opts.display == 'conf':
        fig = plt.figure()
        axs1 = plt.contourf(x_mat, y_mat, z_mat, COLORLEVELS, antialiased=False,
                            cmap=cm.coolwarm)
        axs2 = plt.contour(x_mat, y_mat, z_mat, axs1.levels, colors='k',
                           antialiased=True, linewidths=LINEWIDTH)
        cbar = plt.colorbar(axs1)
        fig.tight_layout()
        plt.subplots_adjust(left=0.14, bottom=0.14)

    # add custom text
    added_text = False
    if opts.key == 'dsc':
        added_text = True
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
    if opts.key == 'msa':
        added_text = True
        texts = [r'\textbf{1}_{\textbf{a}}',
                 r'\textbf{1}_{\textbf{a}}',
                 r'\textbf{1}_{\textbf{a}}',
                 r'\textbf{1}_{\textbf{a}}',
                 r'\textbf{1}_{\textbf{b}}',
                 r'\textbf{1}_{\textbf{c}}^{\textbf{ts}}',
                 r'\textbf{1}_{\textbf{d}}^{\textbf{ts}}',
                 r'\textbf{1}_{\textbf{e}}^{\textbf{ts}}']
        coords = [[5, 5],
                  [329, 5],
                  [5, 333],
                  [5, 112],
                  [172, 5],
                  [5, 50],
                  [168, 48],
                  [95, 110]]
    if added_text is True:
        for text, coord in zip(texts, coords):
            print('Adding "{}" to {}.'.format(text, coord))
            plt.text(coord[0], coord[1], text,
                     path_effects=[patheffects.withStroke(
                         linewidth=2, foreground='w')])
            
    if opts.xl:
        plt.xlabel(opts.xl)
    if opts.yl:
        plt.ylabel(opts.yl)
    if opts.zl and '3d' in opts.display:
        axs.set_zlabel(opts.zl)
    elif opts.zl and opts.display == 'conf':
        cbar.ax.set_ylabel(opts.zl)
    
    if opts.output:
        for extension in opts.extension:
            print('Saving: {}.{}'.format(opts.output, extension))
            plt.savefig('{}.{}'.format(opts.output, extension), dpi=300)
    else:
        plt.show()
