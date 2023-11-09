"""
CloudFlex relies on source code taken from the Trident package to perform
the absorption line generation.  The code below was extracted directly from
the Trident source, so as to avoid having Trident+yt as explicit software
dependencies.

Trident is a python-based package extending yt for generating synthetic
observations of hydrodynamical datasets.

Please visit the Trident and yt websites for more information:
http://trident-project.org
http://yt-project.org

#-----------------------------------------------------------------------------
# Copyright (c) 2016, Trident Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
#-----------------------------------------------------------------------------
"""

import numpy as np
from scipy.special import wofz
from unyt import \
    charge_proton_cgs, \
    mass_electron_cgs, \
    speed_of_light_cgs

tau_factor = None
_cs = None

# Taken from Cloudy documentation.
solar_abundance = {
    'H' : 1.00e+00, 'He': 1.00e-01, 'Li': 2.04e-09,
    'Be': 2.63e-11, 'B' : 6.17e-10, 'C' : 2.45e-04,
    'N' : 8.51e-05, 'O' : 4.90e-04, 'F' : 3.02e-08,
    'Ne': 1.00e-04, 'Na': 2.14e-06, 'Mg': 3.47e-05,
    'Al': 2.95e-06, 'Si': 3.47e-05, 'P' : 3.20e-07,
    'S' : 1.84e-05, 'Cl': 1.91e-07, 'Ar': 2.51e-06,
    'K' : 1.32e-07, 'Ca': 2.29e-06, 'Sc': 1.48e-09,
    'Ti': 1.05e-07, 'V' : 1.00e-08, 'Cr': 4.68e-07,
    'Mn': 2.88e-07, 'Fe': 2.82e-05, 'Co': 8.32e-08,
    'Ni': 1.78e-06, 'Cu': 1.62e-08, 'Zn': 3.98e-08}

def voigt(a, u):
    """
    Calculate the numerical Voigt line profile for absorption features.

    This code uses a SciPy routine to numerically calculate the Voigt Profile
    for deposition as absorption features into spectra.

    This method employs the real part of the Fadeeva function, w(z),
    shown here as special.wofz.  It operates on a and u, unitless variables
    representing the frequency/wavelength range and the damping parameter for
    the line.

    Included are notes from a previous implementation:
    a = Voigt "A" parameter.
    u = Frequency in units of the Doppler frequency.

    The line profile "Phi(v)", the doppler width
    "Delv", the voigt parameter "a", and the frequency "u"
    are given by:

    Phi(v) =  Voigt(a,u)/[ Delv * sqrt(pi) ]
    Delv   =  Vo/c * sqrt[ 2kT/m ]
    u      =  V - Vo / Delv
    a      =  GAMMA / [ Delv * 4pi ]
    Gamma  =  Gu + Gl + 2*Vcol
    "Gu" and "Gl" are the widths of the upper and lower states
    "Vcol" is the collisions per unit time
    "Vo" is the line center frequency

    **Parameters**

    :a: float

        Damping parameter or Voigt "A" parameter describing the width of the
        profile. This is the gamma value divided by the b parameter (thermal
        broadening) in the unitless frame of the u array.

    :u: array

        The region over which the voigt profile will be calculated. This too
        is unitless and represents the normalized frequency space over which
        to calculate the Voigt profile.

    """
    x = np.asarray(u).astype(np.float64)
    y = np.asarray(a).astype(np.float64)
    return wofz(x + 1j * y).real

def tau_profile(lambda_0, f_value, gamma, v_doppler, column_density,
                delta_v=None, delta_lambda=None,
                lambda_bins=None, n_lambda=12000, dlambda=0.01):
    """
    Make optical depth vs. wavelength profile for an absorption line.

    This method uses the numerical Voigt Profile integrator in the voigt()
    function to deposit optical depth values for absorption lines into the
    wavelength space surrounding an absorption feature.

    **Parameters**

    :lambda_0: float

       Central wavelength of absorber in angstroms.

    :f_value: float

       The oscillator strength of the absorption line.

    :gamma: float

       Absorption line gamma value.

    :v_doppler: float

       Doppler b-parameter in cm/s.

    :column_density: float

       Column density in cm^-2.

    :delta_v: float

       Velocity offset from central wavelength (lambda_0) in cm/s.
       Default: None (no shift).

    :delta_lambda: float

        Wavelength offset from central wavelength (lambda_0) in angstroms.
        Default: None (no shift).

    :lambda_bins: array

        Wavelength array for line deposition in anstroms.  If None, one will be
        created using n_lambda and dlambda.
        Default: None.

    :n_lambda: int

        Number of lambda bins to create if lambda_bins is None.
        Default: 12000.

    :dlambda: float in angstroms

        Width of lambda bins in angstroms if lambda_bins is None.
        Default: 0.01.

    """
    global tau_factor
    if tau_factor is None:
        tau_factor = (
            np.sqrt(np.pi) * charge_proton_cgs ** 2 /
            (mass_electron_cgs * speed_of_light_cgs)
        ).in_cgs().d

    global _cs
    if _cs is None:
        _cs = speed_of_light_cgs.d[()]

    # shift lambda_0 by delta_v
    if delta_v is not None:
        lam1 = lambda_0 * (1 + delta_v / _cs)
    elif delta_lambda is not None:
        lam1 = lambda_0 + delta_lambda
    else:
        lam1 = lambda_0

    # conversions
    nudop = 1e8 * v_doppler / lam1   # doppler width in Hz

    # create wavelength
    if lambda_bins is None:
        lambda_bins = lam1 + \
            np.arange(n_lambda, dtype=np.float) * dlambda - \
            n_lambda * dlambda / 2  # wavelength vector (angstroms)

    # tau_0
    tau_X = tau_factor * column_density * f_value / v_doppler
    tau0 = tau_X * lambda_0 * 1e-8

    # dimensionless frequency offset in units of doppler freq
    u = _cs / v_doppler * (lam1 / lambda_bins - 1.0)
    a = gamma / (4.0 * np.pi * nudop)               # damping parameter
    phi = voigt(a, u)                               # line profile
    tauphi = tau0 * phi              # profile scaled with tau0

    return (lambda_bins, tauphi)
