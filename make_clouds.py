"""
Make a cloud complex based on specified behavior
"""

from cloudflex import \
    plot_histogram, \
    plot_clouds, \
    Clouds, \
    create_and_plot_clouds
import sys
import numpy as np
from unyt import cm, K, proton_mass
import os

if __name__ == '__main__':

    if len(sys.argv) == 3:
        loop = True
        mclmin_low = sys.argv[1]
        mclmin_high = sys.argv[2]
    elif len(sys.argv) == 1:
        loop = False
    else:
        print('Usage: make_clouds.py *or* make_clouds.py log10(mclmin_low) log10(mclmin_high)')
        print('Example: make_clouds.py')
        print('Example: make_clouds.py -2 4')
        sys.exit()

    ### Top Level Variables

    verbose = True

    #### Turbulence Generation Stuff

    # Store everything in parameters dictionary to pass to Clouds class
    params = {}

    # Turbulence Generation Stuff
    N = 128                             # Default: 128
    seed = int(2) # random seed
    np.random.seed(seed=seed)
    dtype = np.float64 # data precision

    params['seed'] = seed
    params['N'] = N
    params['n'] = [N, N, N]
    params['kmin'] = int(1)             # Default: 1
    params['kmax'] = int(N/2)           # Default: N/2
    params['f_solenoidal'] = 2/3.       # Default: 2/3
    params['beta'] = 1/3.               # Default: 1/3
    params['vmax'] = 30                 # Default: 30 km/s

    #### Cloud Generation Stuff
    params['alpha'] = 2                 # Default: 2
    params['zeta'] = 1                  # Default: 1

    # cloud mass min/max (Msun)
    params['mclmin'] = 1e1              # Default: 10 Msun
    params['mclmax'] = 1e5              # Default: 100000 Msun

    # cloud position from center min/max (kpc)
    params['dclmin'] = 0.1              # Default: 0.1 kpc
    params['dclmax'] = 5                # Default: 5 kpc
    params['center'] = [0.0, 0.0]       # Default: (0, 0)

    # background density and temperature of hot medium
    # density and temperature of cool clouds
    params['n_bg'] = 1e-4 / cm**3       # Default: 1e-4 cm^-3
    params['T_bg'] = 1e6 * K            # Default: 1e6 K
    params['P'] = params['n_bg'] * params['T_bg']
    params['T_cl'] = 1e4 * K            # Default: 1e4 K
    params['n_cl'] = params['P'] / params['T_cl']  # Default: 1e-2 cm^-3
    params['Z_cl'] = 0.33               # Default: 0.33
    params['rho_cl'] = params['n_cl'] * proton_mass
    params['total_mass'] = 1e6          # Default: 1e6 Msun
    params['clobber'] = True            # Default: True

    if not loop:
        create_and_plot_clouds(params)

    else:
        # Generate a series of clouds instances, varying mclmin from mclmin_low to
        # mclmin_high stepping in powers of 10

        # mclmin_low and high are like 1, 3 to represent 1e1 to 1e3 mclmins
        mclmin_low = int(mclmin_low)
        mclmin_high = int(mclmin_high)
        mclmins = np.logspace(mclmin_low, mclmin_high, num=mclmin_high-mclmin_low+1)
        print("Looping over mclmins = %s" % mclmins)

        # Ensuring seed state is same for each iteration
        state = np.random.get_state()

        for i in range(len(mclmins)):
            np.random.set_state(state)
            mclmin = mclmins[i]
            print("Generating clouds for mclmin %.1g" % mclmin)
            if mclmin < 1:
                path = './%.1g' % mclmin
            else:
                path = './%1d' % mclmin
            if not os.path.exists(path):
                os.makedirs(path)
            os.chdir(path)
            params['mclmin'] = mclmin
            create_and_plot_clouds(params)
            os.chdir('..')
