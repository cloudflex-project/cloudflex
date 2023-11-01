"""
Run ray experiments sending N sightlines through the cloud cluster
"""

from cloudflex import \
    SpectrumGenerator, \
    plot_histogram, \
    plot_clouds, \
    Clouds, \
    Rays, \
    plot_covering_frac
import sys
import numpy as np

if __name__ == '__main__':

    if len(sys.argv) != 3:
        sys.exit('Usage: analyze_rays.py clouds.h5 rays.h5')
    clouds_fn = sys.argv[1]
    rays_fn = sys.argv[2]

    clouds = Clouds()
    clouds.load(clouds_fn)
    params = clouds.params
    rays = Rays(generate=False)
    rays.load(rays_fn)
    #np.random.seed(seed=params['seed'])
    #plot_clouds(clouds=clouds, filename='clouds.png')
    #distances = np.linalg.norm(np.array([0,0,0]) - clouds.centers, axis=1)
    #velocities = np.linalg.norm(np.array([0,0,0]) - clouds.velocities, axis=1)
    #plot_histogram(distances, "cloud radial distances", clouds, log=True, xrange=[-1, 1])
    #plot_histogram(clouds.masses, "cloud masses", clouds, log=True, xrange=[-3, 5])
    #plot_histogram(velocities, "cloud total velocities", clouds, log=True, xrange=[-1,2.5])

    ## Generate and Plot Spectra for N random rays passing through domain
    ## Also plot distribution of clouds and rays

    # Create histograms of EW, clouds intersected, and column density of all rays
    #plot_histogram(rays.n_clouds, "clouds intersected", clouds)
    plot_histogram(rays.column_densities, "column densities", clouds, log=True, xrange=[11,18])
    plot_histogram(rays.EWs, "equivalent widths", clouds, xrange=[0, 1.0], n_bins=40)
    plot_covering_frac(clouds, rays)
