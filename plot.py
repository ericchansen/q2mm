#!/usr/bin/python
from itertools import izip
import argparse
import logging
import logging.config
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import os
import sys
import yaml

import filetypes

logger = logging.getLogger(__name__)

COLORLEVELS = 10
LINEWIDTH = 0.5

XLIM = 355
YLIM = 355

X_LABEL = 'r_j_d1'
Y_LABEL = 'r_j_d2'
Z_LABEL = 'r_mmod_Potential_Energy-MM3*'

if __name__ == '__main__':
    with open('logging.yaml', 'r') as f:
        cfg = yaml.load(f)
    logging.config.dictConfig(cfg)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--display', '-d', type=str, default='3d')
    parser.add_argument('--filename', '-f', type=str)
    parser.add_argument('--extension', '-e', type=str, nargs='+', default=['jpg'])
    parser.add_argument('--key', '-k', type=str, nargs='+', default=[])
    parser.add_argument('--output' ,'-o', type=str)
    opts = parser.parse_args(sys.argv[1:])

    path, ext = os.path.splitext(opts.filename)
    if ext == '.mae':
        mae = filetypes.Mae(opts.filename)
    
    x_data = []
    y_data = []
    z_data = []
    for structure in mae.structures:
        x_data.append(float(structure.props[X_LABEL]))
        y_data.append(float(structure.props[Y_LABEL]))
        z_data.append(float(structure.props[Z_LABEL]))
    logger.info('x: {} y: {} z: {}'.format(len(x_data), len(y_data), len(z_data)))
    
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

    if '3d' in opts.display:
        fig = plt.figure()
        axs = fig.gca(projection='3d')
        srf = axs.plot_surface(x_mat, y_mat, z_mat, rstride=1, cstride=1,
                               cmap=cm.coolwarm, linewidth=0, antialiased=True)
        plt.locator_params(axis='z', nbins=8)
        axs.view_init(elev=70,azim=-64)
        if 'c' in opts.display:
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
            axs.set_xlim3d(0, XLIM)
            axs.set_ylim3d(0, YLIM)
            # For whatever reason, the tight layout doesn't work well here.
            fig.tight_layout()
            plt.subplots_adjust(left=-0.1)

    elif 'conf' in opts.display:
        fig = plt.figure()
        axs1 = plt.contourf(x_mat, y_mat, z_mat, COLORLEVELS, antialiased=False,
                            cmap=cm.coolwarm)
        axs2 = plt.contour(x_mat, y_mat, z_mat, axs1.levels, colors='k',
                           antialiased=True, linewidths=LINEWIDTH)

    # Adds some custom text for graphs I was working on.
    if 'dsc' in opts.key:
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
    if 'msa' in opts.key:
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
                  [95, 110],
                  [168, 48]]
        for text, coord in zip(texts, coords):
            logger.debug('Adding "{}" to {}.'.format(text, coord))
            plt.text(coord[0], coord[1], text,
                     path_effects=[patheffects.withStroke(
                        linewidth=2, foreground='w')])

    for extension in opts.extension:
        logger.info('saving to {}.{}'.format(opts.output, extension))
        plt.savefig('{}.{}'.format(opts.output, extension), dpi=300)
