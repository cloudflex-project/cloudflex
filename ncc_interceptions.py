import numpy as np
from IPython import embed


## Global CGM properties
MCGM = 1.0e10    ## Total photoionized CGM mass within RCGM, Msun
Rperp = 25.0    ## kpc -- impact parameter of observation


## Cloudlet collections
dclmax = 10.0    ## kpc
rclmax = dclmax/2.0
mtot = 1.0e6    ## Msun
sigma_cl = np.pi * rclmax**2   ## cross section in kpc^2
Vcl = 4.0 * np.pi * (rclmax**3) / 3.0   ## volume in kpc^3


## How big does the CGM have to be?
Ncc = MCGM/mtot
RCGM = (3.0 * Ncc * Vcl / (4.0 * np.pi))**(1./3.)
VCGM = 4.0 * np.pi * (RCGM**3) / 3.0    ## Total CGM volume, kpc^3

## Interesting quantities
print("Number of cloud collections in CGM = {0:5.0f}".format(Ncc))
Ncc_pervol = Ncc/VCGM
print("Number of cloud collections per kpc^3 = {0:9.4f}".format(Ncc_pervol))
fVcc = Ncc * Vcl/VCGM
print("Ratio of cloud collection volume to total CGM volume = {0:5.2f}".format(fVcc))
print("R_CGM = {0:5.2f}".format(RCGM))
pathCGM = 2.0*np.sqrt(RCGM**2 - Rperp**2)   ## total path length through CGM
Ncc_persightline = Ncc_pervol * sigma_cl * pathCGM
print("Number of cloud collections probed = {0:5.0f}".format(Ncc_persightline))

embed()


