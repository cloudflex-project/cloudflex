"""
Run ray experiments sending N sightlines through the cloud cluster

Usage: python analyze_multi.py <dir1> <dir2> <dir3> <dir4> <dir5>

Where each directory is a run containing rays.h5 and clouds.h5
"""

from cloudflex import \
    SpectrumGenerator, \
    plot_histogram, \
    plot_clouds, \
    Clouds, \
    Rays, \
    plot_covering_frac, \
    plot_multi_clouds, \
    plot_multi_clouds_buffer, \
    make_quad_plot
import sys
import numpy as np
import argparse
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Analyze multiple datasets')
    parser.add_argument("input", nargs='+')
    args = parser.parse_args()
    inn = args.input
    n_dir = len(inn)
    rays_list = []
    clouds_list = []
    for j in range(n_dir):
        rays_fn = os.path.join(inn[j], 'rays.h5')
        clouds_fn = os.path.join(inn[j], 'clouds.h5')
        if not os.path.exists(rays_fn): continue
        rays = Rays(generate=False)
        rays.load(rays_fn)
        rays_list.append(rays)
        clouds = Clouds()
        clouds.load(clouds_fn)
        clouds_list.append(clouds)
    #plot_covering_frac(rays_list, clouds_list)
    #plot_multi_clouds_buffer(clouds_list)
    make_quad_plot(rays_list, clouds_list)
