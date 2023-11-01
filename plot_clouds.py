"""
Visualize clouds
"""

from cloudflex import \
    plot_clouds, \
    Clouds
import sys
import numpy as np

if __name__ == '__main__':

    if len(sys.argv) != 2:
        sys.exit('Usage: plot_clouds.py clouds.h5')
    fn = sys.argv[1]

    clouds = Clouds()
    clouds.load(fn)
    params = clouds.params
    np.random.seed(seed=params['seed'])
    plot_clouds(clouds=clouds, filename='clouds.png')
    #plot_clouds(clouds=clouds, cloud_velocity=True, filename='clouds.png')
    #plot_clouds(clouds=clouds, slice_depth=1, cloud_velocity=True, filename='clouds_slice_1.png')
