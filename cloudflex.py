"""
Code for generating cloud and ray distributions, along with resulting synthetic
observations
"""

import h5py
import numpy as np
import numpy.fft
from math import *
from optparse import OptionParser
import sys
import matplotlib.pyplot as plt
import cmasher as cmr
import warnings
from tqdm import tqdm
from unyt import kpc, cm, K, amu, km, s, Msun, angstrom, c, g, proton_mass
import matplotlib as mpl
import pandas as pd
from trident.ion_balance import solar_abundance
from trident.absorption_spectrum.absorption_line import tau_profile
from yt.utilities.physical_constants import boltzmann_constant_cgs
from astropy.convolution import Gaussian1DKernel, convolve
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import AutoMinorLocator
from matplotlib.collections import PatchCollection
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.patheffects as pe
import matplotlib.gridspec as gridspec
import palettable
import cmyt
import cmocean
import matplotlib.colors as colors
import os.path

## change the default font to Avenir
plt.rcParams['font.family'] = 'Avenir'
# change the math font to Avenir
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Avenir'
plt.rcParams['mathtext.it'] = 'Avenir:italic'
plt.rcParams['mathtext.bf'] = 'Avenir:bold'
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rc('font', size=12)
plt.rc('axes', labelsize=15)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=15)    # fontsize of the tick labels
plt.rc('ytick', labelsize=15)    # fontsize of the tick labels
plt.rc('legend', fontsize=13)    # legend fontsize

def init_perturbations(n, kmin, kmax, beta):

    dtype = np.float64
    # Calculate alpha value in use
    alpha = 2*beta + 1
    if alpha==None:
        print( "You must choose a power law slope, alpha.  See --help.")
    if alpha < 0.:
        print( "alpha is less than zero. Thats probably not what you want.  See --help.")

    kx = np.zeros(n, dtype=dtype)
    ky = np.zeros(n, dtype=dtype)
    kz = np.zeros(n, dtype=dtype)
    # perform fft k-ordering convention shifts
    for j in range(0,n[1]):
        for k in range(0,n[2]):
            kx[:,j,k] = n[0]*np.fft.fftfreq(n[0])
    for i in range(0,n[0]):
        for k in range(0,n[2]):
            ky[i,:,k] = n[1]*np.fft.fftfreq(n[1])
    for i in range(0,n[0]):
        for j in range(0,n[1]):
            kz[i,j,:] = n[2]*np.fft.fftfreq(n[2])

    kx = np.array(kx, dtype=dtype)
    ky = np.array(ky, dtype=dtype)
    kz = np.array(kz, dtype=dtype)
    k = np.sqrt(np.array(kx**2+ky**2+kz**2, dtype=dtype))

    # only use the positive frequencies
    inds = np.where(np.logical_and(k**2 >= kmin**2, k**2 < (kmax+1)**2))
    nr = len(inds[0])

    phasex = np.zeros(n, dtype=dtype)
    phasex[inds] = 2.*pi*np.random.uniform(size=nr)
    fx = np.zeros(n, dtype=dtype)
    fx[inds] = np.random.normal(size=nr)

    phasey = np.zeros(n, dtype=dtype)
    phasey[inds] = 2.*pi*np.random.uniform(size=nr)
    fy = np.zeros(n, dtype=dtype)
    fy[inds] = np.random.normal(size=nr)

    phasez = np.zeros(n, dtype=dtype)
    phasez[inds] = 2.*pi*np.random.uniform(size=nr)
    fz = np.zeros(n, dtype=dtype)
    fz[inds] = np.random.normal(size=nr)

    # rescale perturbation amplitude so that low number statistics
    # at low k do not throw off the desired power law scaling.
    for i in range(kmin, kmax+1):
        slice_inds = np.where(np.logical_and(k >= i, k < i+1))
        rescale = sqrt(np.sum(np.abs(fx[slice_inds])**2 + np.abs(fy[slice_inds])**2 + np.abs(fz[slice_inds])**2))
        fx[slice_inds] = fx[slice_inds]/rescale
        fy[slice_inds] = fy[slice_inds]/rescale
        fz[slice_inds] = fz[slice_inds]/rescale

    # set the power law behavior
    # wave number bins
    fx[inds] = fx[inds]*k[inds]**-(0.5*alpha)
    fy[inds] = fy[inds]*k[inds]**-(0.5*alpha)
    fz[inds] = fz[inds]*k[inds]**-(0.5*alpha)

    # add in phases
    fx = np.cos(phasex)*fx + 1j*np.sin(phasex)*fx
    fy = np.cos(phasey)*fy + 1j*np.sin(phasey)*fy
    fz = np.cos(phasez)*fz + 1j*np.sin(phasez)*fz

    return fx, fy, fz, kx, ky, kz

def make_perturbations(n, kmin, kmax, beta, f_solenoidal, vmax):

    fx, fy, fz, kx, ky, kz = init_perturbations(n, kmin, kmax, beta)
    if f_solenoidal != None:
        k2 = kx**2+ky**2+kz**2
        # solenoidal part
        fxs = 0.; fys =0.; fzs = 0.
        if f_solenoidal != 0.0:
            fxs = np.real(fx - kx*(kx*fx+ky*fy+kz*fz)/np.maximum(k2,1e-16))
            fys = np.real(fy - ky*(kx*fx+ky*fy+kz*fz)/np.maximum(k2,1e-16))
            fzs = np.real(fz - kz*(kx*fx+ky*fy+kz*fz)/np.maximum(k2,1e-16))
            ind = np.where(k2 == 0)
            fxs[ind] = 0.; fys[ind] = 0.; fzs[ind] = 0.
            # need to normalize this before applying relative weighting of solenoidal / compressive components
            norm = np.sqrt(np.sum(fxs**2+fys**2+fzs**2))
            fxs = fxs/norm
            fys = fys/norm
            fzs = fzs/norm
        # compressive part
        # get a different random cube for the compressive part
        # so that we can target the RMS solenoidal fraction,
        # instead of setting a constant solenoidal fraction everywhere.
        fx, fy, fz, kx, ky, kz = init_perturbations(n, kmin, kmax, beta)
        fxc = 0.; fyc =0.; fzc = 0.
        if f_solenoidal != 1.0:
            fxc = np.real(kx*(kx*fx+ky*fy+kz*fz)/np.maximum(k2,1e-16))
            fyc = np.real(ky*(kx*fx+ky*fy+kz*fz)/np.maximum(k2,1e-16))
            fzc = np.real(kz*(kx*fx+ky*fy+kz*fz)/np.maximum(k2,1e-16))
            ind = np.where(k2 == 0)
            fxc[ind] = 0.; fyc[ind] = 0.; fzc[ind] = 0.
            # need to normalize this before applying relative weighting of solenoidal / compressive components
            norm = np.sqrt(np.sum(fxc**2+fyc**2+fzc**2))
            fxc = fxc/norm
            fyc = fyc/norm
            fzc = fzc/norm
        # back to real space
        pertx = np.real(np.fft.ifftn(f_solenoidal*fxs + (1.-f_solenoidal)*fxc))
        perty = np.real(np.fft.ifftn(f_solenoidal*fys + (1.-f_solenoidal)*fyc))
        pertz = np.real(np.fft.ifftn(f_solenoidal*fzs + (1.-f_solenoidal)*fzc))
    else:
        # just convert to real space
        pertx = np.real(np.fft.ifftn(fx))
        perty = np.real(np.fft.ifftn(fy))
        pertz = np.real(np.fft.ifftn(fz))

    # subtract off COM (assuming uniform density)
    pertx = pertx-np.average(pertx)
    perty = perty-np.average(perty)
    pertz = pertz-np.average(pertz)
    # scale RMS of perturbation cube to unity
    pertx, perty, pertz = normalize(pertx, perty, pertz, n)

    # normalize to vmax
    pertx = vmax / np.cbrt(3) * pertx / np.std(pertx)
    perty = vmax / np.cbrt(3) * perty / np.std(perty)
    pertz = vmax / np.cbrt(3) * pertz / np.std(pertz)

    return pertx, perty, pertz

def init_perturbations_log_normal(n, kmin, kmax):
    dtype = np.float64
    kx = np.zeros(n, dtype=dtype)
    ky = np.zeros(n, dtype=dtype)
    kz = np.zeros(n, dtype=dtype)
    # perform fft k-ordering convention shifts
    for j in range(0,n[1]):
        for k in range(0,n[2]):
            kx[:,j,k] = n[0]*np.fft.fftfreq(n[0])
    for i in range(0,n[0]):
        for k in range(0,n[2]):
            ky[i,:,k] = n[1]*np.fft.fftfreq(n[1])
    for i in range(0,n[0]):
        for j in range(0,n[1]):
            kz[i,j,:] = n[2]*np.fft.fftfreq(n[2])

    kx = np.array(kx, dtype=dtype)
    ky = np.array(ky, dtype=dtype)
    kz = np.array(kz, dtype=dtype)
    k = np.sqrt(np.array(kx**2+ky**2+kz**2, dtype=dtype))

    # only use the positive frequencies
    inds = np.where(np.logical_and(k**2 >= kmin**2, k**2 < (kmax+1)**2))
    nr = len(inds[0])

    phasex = np.zeros(n, dtype=dtype)
    phasex[inds] = 2.*pi*np.random.uniform(size=nr)
    fx = np.zeros(n, dtype=dtype)
    fx[inds] = np.random.normal(size=nr)

    phasey = np.zeros(n, dtype=dtype)
    phasey[inds] = 2.*pi*np.random.uniform(size=nr)
    fy = np.zeros(n, dtype=dtype)
    fy[inds] = np.random.normal(size=nr)

    phasez = np.zeros(n, dtype=dtype)
    phasez[inds] = 2.*pi*np.random.uniform(size=nr)
    fz = np.zeros(n, dtype=dtype)
    fz[inds] = np.random.normal(size=nr)

    # rescale perturbation amplitude so that low number statistics
    # at low k do not throw off the desired power law scaling.
    for i in range(kmin, kmax+1):
        slice_inds = np.where(np.logical_and(k >= i, k < i+1))
        rescale = sqrt(np.sum(np.abs(fx[slice_inds])**2 + np.abs(fy[slice_inds])**2 + np.abs(fz[slice_inds])**2))
        fx[slice_inds] = fx[slice_inds]/rescale
        fy[slice_inds] = fy[slice_inds]/rescale
        fz[slice_inds] = fz[slice_inds]/rescale

    # set the power law behavior
    # wave number bins
    fx[inds] = fx[inds]/k[inds] * np.exp( - np.log(k[inds]/k_peak)**2/(2*sigma_k**2))
    fy[inds] = fy[inds]/k[inds] * np.exp( - np.log(k[inds]/k_peak)**2/(2*sigma_k**2))
    fz[inds] = fz[inds]/k[inds] * np.exp( - np.log(k[inds]/k_peak)**2/(2*sigma_k**2))

    # add in phases
    fx = np.cos(phasex)*fx + 1j*np.sin(phasex)*fx
    fy = np.cos(phasey)*fy + 1j*np.sin(phasey)*fy
    fz = np.cos(phasez)*fz + 1j*np.sin(phasez)*fz

    return fx, fy, fz, kx, ky, kz

def make_perturbations_log_normal(n, kmin, kmax, f_solenoidal):
    fx, fy, fz, kx, ky, kz = init_perturbations_log_normal(n, kmin, kmax)
    if f_solenoidal != None:
        k2 = kx**2+ky**2+kz**2
        # solenoidal part
        fxs = 0.; fys =0.; fzs = 0.
        if f_solenoidal != 0.0:
            fxs = np.real(fx - kx*(kx*fx+ky*fy+kz*fz)/np.maximum(k2,1e-16))
            fys = np.real(fy - ky*(kx*fx+ky*fy+kz*fz)/np.maximum(k2,1e-16))
            fzs = np.real(fz - kz*(kx*fx+ky*fy+kz*fz)/np.maximum(k2,1e-16))
            ind = np.where(k2 == 0)
            fxs[ind] = 0.; fys[ind] = 0.; fzs[ind] = 0.
            # need to normalize this before applying relative weighting of solenoidal / compressive components
            norm = np.sqrt(np.sum(fxs**2+fys**2+fzs**2))
            fxs = fxs/norm
            fys = fys/norm
            fzs = fzs/norm
        # compressive part
        # get a different random cube for the compressive part
        # so that we can target the RMS solenoidal fraction,
        # instead of setting a constant solenoidal fraction everywhere.
        fx, fy, fz, kx, ky, kz = init_perturbations_log_normal(n, kmin, kmax)
        fxc = 0.; fyc =0.; fzc = 0.
        if f_solenoidal != 1.0:
            fxc = np.real(kx*(kx*fx+ky*fy+kz*fz)/np.maximum(k2,1e-16))
            fyc = np.real(ky*(kx*fx+ky*fy+kz*fz)/np.maximum(k2,1e-16))
            fzc = np.real(kz*(kx*fx+ky*fy+kz*fz)/np.maximum(k2,1e-16))
            ind = np.where(k2 == 0)
            fxc[ind] = 0.; fyc[ind] = 0.; fzc[ind] = 0.
            # need to normalize this before applying relative weighting of solenoidal / compressive components
            norm = np.sqrt(np.sum(fxc**2+fyc**2+fzc**2))
            fxc = fxc/norm
            fyc = fyc/norm
            fzc = fzc/norm
        # back to real space
        pertx = np.real(np.fft.ifftn(f_solenoidal*fxs + (1.-f_solenoidal)*fxc))
        perty = np.real(np.fft.ifftn(f_solenoidal*fys + (1.-f_solenoidal)*fyc))
        pertz = np.real(np.fft.ifftn(f_solenoidal*fzs + (1.-f_solenoidal)*fzc))
    else:
        # just convert to real space
        pertx = np.real(np.fft.ifftn(fx))
        perty = np.real(np.fft.ifftn(fy))
        pertz = np.real(np.fft.ifftn(fz))

    # subtract off COM (assuming uniform density)
    pertx = pertx-np.average(pertx)
    perty = perty-np.average(perty)
    pertz = pertz-np.average(pertz)
    # scale RMS of perturbation cube to unity
    pertx, perty, pertz = normalize(pertx, perty, pertz, n)
    return pertx, perty, pertz

def normalize(fx, fy, fz, n):
    norm = np.sqrt(np.sum(fx**2 + fy**2 + fz**2)/np.product(n))
    fx = fx/norm
    fy = fy/norm
    fz = fz/norm
    return fx, fy, fz

def get_erot_ke_ratio(pertx, perty, pertz):
    x, y, z = np.mgrid[0:n[0], 0:n[1], 0:n[2]]
    x = x - (n[0]-1)/2.
    y = y - (n[1]-1)/2.
    z = z - (n[2]-1)/2.
    r2 = x**2+y**2+z**2
    erot_ke_ratio = (np.sum(y*pertz-z*perty)**2 +
                     np.sum(z*pertx-x*pertz)**2 +
                     np.sum(x*perty-y*pertx)**2)/(np.sum(r2)*np.product(n))
    return erot_ke_ratio

def plot_spectrum1D(pertx, perty, pertz, n, kmin, kmax):
    dtype = np.float64
    # plot the 1D power to check the scaling.
    fx = np.abs(np.fft.fftn(pertx))
    fy = np.abs(np.fft.fftn(perty))
    fz = np.abs(np.fft.fftn(pertz))
    fx = np.abs(fx)
    fy = np.abs(fy)
    fz = np.abs(fz)
    kx = np.zeros(n, dtype=dtype)
    ky = np.zeros(n, dtype=dtype)
    kz = np.zeros(n, dtype=dtype)
    # perform fft k-ordering convention shifts
    for j in range(0,n[1]):
        for k in range(0,n[2]):
            kx[:,j,k] = n[0]*np.fft.fftfreq(n[0])
    for i in range(0,n[0]):
        for k in range(0,n[2]):
            ky[i,:,k] = n[1]*np.fft.fftfreq(n[1])
    for i in range(0,n[0]):
        for j in range(0,n[1]):
            kz[i,j,:] = n[2]*np.fft.fftfreq(n[2])
    k = np.sqrt(np.array(kx**2+ky**2+kz**2,dtype=dtype))
    k1d = []
    power = []
    for i in range(kmin,kmax+1):
        slice_inds = np.where(np.logical_and(k >= i, k < i+1))
        k1d.append(i+0.5)
        power.append(np.sum(fx[slice_inds]**2 + fy[slice_inds]**2 + fz[slice_inds]**2))
        #print( i,power[-1])
    power = np.array(power)
    power /= power.min()
    fig, ax = plt.subplots()
    plt.loglog(k1d, power)
    plt.xlabel('k')
    plt.ylabel('Power')
    ax.axis('equal')
    plt.savefig("spectrum1D.png", bbox_inches='tight', dpi=200)
    plt.clf()

def plot_slice(pertx,name):
    plt.pcolormesh(pertx[:,:,0])
    ax = plt.gca()
    ax.set_aspect(1.0)
    plt.savefig("pert_slice_%s.png" %name, bbox_inches='tight', dpi=400)
    plt.clf()

def sample_from_mass_CDF(X, alpha, mclmin, mclmax):
    return (mclmin**(-alpha+1) - (mclmin**(-alpha+1) - mclmax**(-alpha+1))*X )**(1/(-alpha+1))

def sample_from_distance_CDF(X, zeta, dclmin, dclmax):
    return (dclmin**(zeta+1) - (dclmin**(zeta+1) - dclmax**(zeta+1))*X )**(1/(zeta+1))

def sph2cart(theta_el, phi_az, r):
    """
    Convert from spherical coords to cartesian coords
    
    theta = polar angle
    phi = azimuthal angle

    """
    z = r * np.cos(theta_el)
    x = r * np.sin(theta_el) * np.cos(phi_az)
    y = r * np.sin(theta_el) * np.sin(phi_az)
    
    return np.array([x, y, z])

def distance(point1, point2):
    """
    Distance between two points
    """
    return np.linalg.norm(point1 - point2)

def cloud_on_cloud_intersection(clouds, center, radius):
    """
    Iteratively check to see if a cloud intersects any existing clouds
    """
    for cl in clouds:
        if distance(cl.center, center) < (cl.r.d + radius.d): return True
    return False

class Cloud:
    """
    Simple cloud object defined as location, velocity, mass, and radius;
    Deprecated in favor of Clouds class
    """
    def __init__(self, x, y, z, vx, vy, vz, m, r, m_min, m_max):
        self.center = np.array([x,y,z])
        self.velocity = np.array([vx,vy,vz])
        self.x = x * kpc
        self.y = y * kpc
        self.z = z * kpc
        self.vx = vx * km/s
        self.vy = vy * km/s
        self.vz = vz * km/s
        self.m = m * Msun
        self.m.convert_to_units('Msun')
        m_min *= Msun
        m_max *= Msun
        m_min.convert_to_units('Msun')
        m_max.convert_to_units('Msun')
        self.m_scaled = np.log10(self.m/m_min) / np.log10(m_max/m_min)
        self.r = r * kpc
        self.center = np.array([self.x, self.y, self.z])

    def __repr__(self):
        return('Cloud: x: [% .02f, % .02f, % .02f] v: [% .02f, % .02f, % .02f], m: %.02f, r: %.02f' % \
               (self.x, self.y, self.z, self.vx, self.vy, self.vz, self.m, self.r))


class Clouds:
    """
    A class to represent a collection of clouds as a series of arrays representing their
    masses, positions, and velocities; Independent of Cloud class.  useful for saving to
    disk.
    """
    def __init__(self, masses=None, radii=None, centers=None, velocities=None,
                 params=None):

        self.masses = masses
        self.radii = radii
        self.centers = centers
        self.velocities = velocities
        self.params = params
        if self.params is None:
            self.params = {}
        if masses is not None:
            self.calculate_stuff()
        self.turbulent_anchor = None


    def __repr__(self):
        return("""Cloud Array:\n %4.2g Clouds\n masses: %s\n radii: %s\n \
centers: %s\n velocities: %s\n params: %s""" % (len(self.masses), self.masses[0:3],
                                    self.radii[0:3], self.centers[0:3],
                                    self.velocities[0:3], self.params))

    def calculate_stuff(self):
        """
        Calculate additional quantities if clouds have been generated
        """
        p = self.params
        p['cloud_mass'] = np.sum(self.masses)
        p['volume'] = 4./3. * np.pi * (p['dclmax'] * kpc)**3
        p['filling_factor'] = (p['cloud_mass'] * Msun / p['rho_cl']) / p['volume']

    def generate_cloud_position(self, N):
        """
        Generates a random cloud at some distance from the origin
        """
        p = self.params
        distance = sample_from_distance_CDF(np.random.rand(N), p['zeta'], p['dclmin'],
                                            p['dclmax'])
        
        ##THIS IS INCORRECT -- KHRR
        #phi = np.random.rand(N)*2*np.pi
        #theta = np.random.rand(N)*np.pi
        
        # Using Drummond's reference frome, with theta = polar angle, phi = azimuthal angle
        phi = 2.0*np.pi*np.random.rand(N)        # azimuthal angle
        costheta = 2.0*np.random.rand(N) - 1.0
        theta = np.arccos(costheta)              # polar angle
        
        return sph2cart(theta, phi, distance).T

    def calculate_cloud_radii(self, masses):
        """
        Calculate the cloud radii for an array of cloud masses
        """
        p = self.params
        # rho = M / V = M / (4/3*pi*R^3), so (4/3*pi*R^3) = (3*M / 4*pi*rho)^(1/3)
        radii = (3*masses*Msun/ (4*np.pi*p['rho_cl']))**(1/3)
        return radii.to('kpc').d

    def generate_velocity(self, center):
        """
        Generate cloud velocities for a cloud center according to turbulent velocity
        spectrum on a 128**3 grid; DEPRECATED: USE generate_velocities()
        """
        # Set up perturbations for generating cloud velocities
        p = self.params
        pertx, perty, pertz = \
            make_perturbations(p['n'], p['kmin'], p['kmax'],
                               p['beta'],  p['f_solenoidal'], p['vmax'])
        X=Y=Z = np.linspace(-p['dclmax'], p['dclmax'], p['N']) * kpc

        # For single center at a time:
        # Identify index closest to position on discretized grid of 128
        ix = np.argmin(np.abs(center[0] - X.d))
        iy = np.argmin(np.abs(center[1] - Y.d))
        iz = np.argmin(np.abs(center[2] - Z.d))

        # Using nearest index, apply perturbations to get velocity
        vx = pertx[ix,iy,iz] * km/s
        vy = perty[ix,iy,iz] * km/s
        vz = pertz[ix,iy,iz] * km/s

        return np.array([vx, vy, vz])

    def generate_velocities(self, centers):
        """
        Generate cloud velocities for a cloud center according to turbulent velocity
        spectrum on a 128**3 grid;
        """
        # Set up perturbations for generating cloud velocities
        print("Determining velocities for clouds")
        p = self.params
        pertx, perty, pertz = \
            make_perturbations(p['n'], p['kmin'], p['kmax'],
                               p['beta'],  p['f_solenoidal'], p['vmax'])
        X=Y=Z = np.linspace(-p['dclmax'], p['dclmax'], p['N']) * kpc

        # For single center at a time:
        # Identify index closest to position on discretized grid of 128
        #ix = np.argmin(np.abs(centers[0] - X.d))
        #iy = np.argmin(np.abs(centers[1] - Y.d))
        #iz = np.argmin(np.abs(centers[2] - Z.d))

        # Broadcasting and vectorizing all velocities simultaneously requires reshaping
        # the matrices a bit.
        N_res = len(X)
        N_clouds = len(centers)

        ix = np.argmin(np.abs(centers.T[0,:].reshape(1,N_clouds) - X.reshape(N_res,1).d), axis=0)
        iy = np.argmin(np.abs(centers.T[1,:].reshape(1,N_clouds) - Y.reshape(N_res,1).d), axis=0)
        iz = np.argmin(np.abs(centers.T[2,:].reshape(1,N_clouds) - Z.reshape(N_res,1).d), axis=0)

        # Using nearest index, apply perturbations to get velocity
        vx = pertx[ix,iy,iz] * km/s
        vy = perty[ix,iy,iz] * km/s
        vz = pertz[ix,iy,iz] * km/s

        return np.array([vx, vy, vz]).T

    def generate_clouds(self):
        """
        """
        if self.params['clobber']:
            print("Allowing cloud clobbering")
        else:
            print("Ensuring no cloud clobbering")

        p = self.params
        masses = []
        centers = []

        # Total volume of cloud space
        volume = (4./3.) * np.pi * (p['dclmax'] * kpc)**3

        # Maximum radius of cloud under these parameters
        max_radius = ((3*p['mclmax']*Msun/ (4*np.pi*p['rho_cl']))**(1/3)).to('kpc').d

        cloud_mass = 0  # total cloud mass running count
        filling_factor = (cloud_mass * Msun / p['rho_cl']) / volume

        # Create random cloud masses until exceed total_mass or reach 1% fill factor
        while (cloud_mass < p['total_mass']):# and (filling_factor < 0.01):
            mass = sample_from_mass_CDF(np.random.rand(), p['alpha'], p['mclmin'], p['mclmax'])
            masses.append(mass)
            cloud_mass += mass

        # Make masses in descending order; easier to add sand to a full bucket than
        # boulders; calculate corresponding radii for clouds
        masses = np.array(masses)
        masses[::-1].sort()
        radii = self.calculate_cloud_radii(masses)
        #print('m_cloud = %s Msun; r_cloud = %s kpc' % (masses[0], radii[0]))

        #import cProfile
        #cProfile.runctx("self.cloud_loop(radii, p['clobber'])", globals(), locals())
        #import sys; sys.exit()
        #centers = self.cloud_loop_kdtree(radii, p['clobber'])
        centers = self.cloud_loop(radii, p['clobber'])
        # Cloud positions are now generated, so now store to Clouds object and save.

        self.masses = masses
        self.centers = np.array(centers)
        self.radii = radii
        self.velocities = self.generate_velocities(centers)
        #plot_spectrum1D(pertx, perty, pertz, p['n'], p['kmin'], p['kmax'])
        self.calculate_stuff()

        #self.turb_broad = self.calculate_subcloud_turbulent_velocity(2.0*radii)
        
        return

    def save(self, filename):
        """
        Saves to an HDF5 file
        """
        print("Saving clouds to %s" % filename)
        with h5py.File(filename, 'w') as f:
            f.create_dataset("masses", data=self.masses)
            f.create_dataset("radii", data=self.radii)
            f.create_dataset("centers", data=self.centers)
            f.create_dataset("velocities", data=self.velocities)
            #f.create_dataset("turb_broad", data=self.turb_broad)
            for i in self.params.keys():
                f.attrs[i] = self.params[i]
        return

    def load(self, filename):
        """
        Loads from an HDF5 file
        """
        print("Loading clouds from %s" % filename)
        with h5py.File(filename, 'r') as f:
            self.masses = f["masses"][:]
            self.radii = f["radii"][:]
            self.centers = f["centers"][:]
            self.velocities = f["velocities"][:]
            #self.turb_broad = f["turb_broad"][:]
            for i in f.attrs.keys():
                self.params[i] = f.attrs[i]
        return

    def clobber_check(self):
        """
        Manually check in post-processing how many clouds are intersecting/clobbering
        """
        centers = self.centers
        radii = self.radii
        N = len(radii)
        clobber = 0
        for i in tqdm(range(0, N), "Checking Clouds for Clobbers"):
            # mask out the ith element
            mask = np.arange(N)!=i
            ## Check to see if cloud overlaps with previous clouds
            # if two clouds overlap, then their distance is less than the sum of
            # their radii: vectorized
            if not (((np.linalg.norm(centers[mask] - centers[i,:], axis=1)
                    - radii[mask] - radii[i]) > 0.).all()):
                clobber += 1
        print("%d clobbers detected." % clobber)

    def cloud_loop_kdtree(self, radii, clobber):
        """
        Alternate method for doing cloud overlap loop to see if a KDTree speeds things.
        """
        from scipy.spatial import KDTree
        tree = KDTree(np.array(centers).reshape(1,3))

        # Create a first cloud to add to the lists, so comparisons don't fail in loop
        centers = []
        center = self.generate_cloud_position(1)[0]
        centers.append(center)

        # Generate individual cloud position with check to ensure they don't clobber others
        for i in tqdm(range(1,len(radii)), "Generating Clouds"):

            # Every 1000 clouds added, build a kd-tree
            if (i % 1 == 0):
                tree = KDTree(np.array(centers))

            radius = radii[i]
            known_radii = radii[:i]

            while(1):
                center = self.generate_cloud_position(1)[0,:]
                # if we allow clobbers then we don't have to check stuff; move to next
                if clobber: break

                # search existing kdtree for all clouds within radius+max_radius
                indices = tree.query_ball_point(center, radii[0]+radius)
                if len(indices) == 0: break
                close_centers = np.array(centers)[indices]
                close_radii = np.array(known_radii)[indices]

                ## check to see if cloud overlaps with previous clouds
                # if two clouds overlap, then their distance is less than the sum of
                # their radii: vectorized
                if ((np.linalg.norm(close_centers - center, axis=1) - close_radii - radius) > 0.).all():
                    break

                # If clouds are too close, then restart loop and choose a new cloud center

            # If cloud doesn't clobber, then append to lists
            centers.append(center)
        return centers

    def cloud_loop(self, radii, clobber):
        """
        Most of the time in generating clouds is spent in this loop
        """
        # Create a first cloud to add to the lists, so comparisons don't fail in loop
        N = len(radii)
        centers = np.zeros([N, 3])
        center = self.generate_cloud_position(1)[0]
        centers[0,:] = center

        # Generate individual cloud position with check to ensure they don't clobber others
        for i in tqdm(range(1,len(radii)), "Generating Clouds"):

            radius = radii[i]
            known_radii = radii[:i]
            known_centers = centers[:i, :]

            while(1):
                center = self.generate_cloud_position(1)[0,:]
                # if we allow clobbers then we don't have to check stuff; move to next
                if clobber: break

                ## check to see if cloud overlaps with previous clouds
                # if two clouds overlap, then their distance is less than the sum of
                # their radii: vectorized
                if ((np.linalg.norm(known_centers - center, axis=1) - known_radii - radius) > 0.).all():
                    break

                # If clouds are too close, then restart loop and choose a new cloud center

            # If cloud doesn't clobber, then append to lists
            centers[i,:] = center
        return centers

    def calculate_turbulence(self, N_points=1e5, N_pairs=1e7, plot=False):
        """
        Calculate the turbulent velocity RMS as a function of separation distance
        The slope will be beta, and the anchor point is set to self.anchor_point
        """
        # Choose N sample points
        print("Initializing Turbulent Power Spectrum")
        N_points = int(N_points)
        N_pairs = int(N_pairs)
        rad = np.random.rand(N_points)*5
        phi = np.random.rand(N_points)*2*np.pi  # azimuthal angle

        ## KHRR changing how thetas are selected
        #theta = np.random.rand(N_points)*np.pi  -- original line
        costheta = 2.0*np.random.rand(N_points) - 1.0
        theta = np.arccos(costheta) 
        
        points = sph2cart(theta, phi, rad).T
        vels = self.generate_velocities(points)
        pairs = np.random.randint(N_points, size=([2,N_pairs]))
        dist = np.linalg.norm(points[pairs[0,:]] - points[pairs[1,:]], axis=1)
        vel_diff = np.linalg.norm(vels[pairs[0,:]] - vels[pairs[1,:]], axis=1)
        vel_diff2 = vel_diff**2
        dist_bins = np.logspace(-2,1)
        vel_bins = np.zeros_like(dist_bins)
        n_bins = len(dist_bins)
        bin_ids  = np.digitize(dist, dist_bins)
        for i in tqdm(range(n_bins), "Analyzing Turbulent Structure Function"):
            mask = bin_ids == i
            vel_bins[i] = np.mean(vel_diff2[mask])
        vel_diff_rms = np.sqrt(vel_bins)

        # examining the plots, the part of the structure function of vel_diff_rms in
        # middle of power-law behavior is at x=log(0.5) -> dist_bin[25]

        # Use that distance and vel_diff_rms as an anchor point for the power-law
        # relationship with exponent beta relating spatial distance and turbulent
        # velocity difference to be encoded as turbulent broadening in each cloud
        # based on its diameter;

        # later, multiply this velocity diff by sqrt(rho_hot / rho_cool) = 0.1 and
        # cap at sound speed = 10 km/s to ensure realistic behavior
        self.turbulent_anchor = (dist_bins[25], vel_diff_rms[25])

        if plot:
            fig, ax = plt.subplots()
            mask = (dist > 0) & (vel_diff > 0)
            plt.hist2d(np.log10(dist[mask]), np.log10(vel_diff[mask]), bins=50,
                       range=[[-2, 1], [0, 2]], norm='symlog')
            plt.plot(np.log10(dist_bins), np.log10(vel_diff_rms), color='w')
            # Anchor point in red
            plt.scatter(np.log10(dist_bins[25]), np.log10(vel_diff_rms[25]), c='red', marker='o')
            dist_limits = [1e-2, 1e1]
            vel_diff_limits = self.extrapolate_turbulent_velocity(dist_limits)
            plt.plot(np.log10(dist_limits), np.log10(vel_diff_limits), c='red')
            # Plot power law with slow beta in red
            ax.set_xlabel('log(Separation Distance (kpc))')
            ax.set_ylabel('log(Velocity Difference (km/s))')
            text = 'Beta: %.2f' % self.params['beta']
            plt.text(0.01, 0.95, text, weight='bold', color='white',transform=ax.transAxes)
            ## overplot the median
            #df = bin_by(np.log10(dist[mask]), np.log10(vel_diff[mask]), nbins=50, bins = None)
            #plt.plot(df.x, df['median'], color = '1', alpha = 0.7, linewidth = 1)
            plt.savefig('turbulence_log.png')
            plt.close()
        return

    def extrapolate_turbulent_velocity(self, distance):
        """
        Calculates the velocity offset v_diff at a separation of l distance:
        proportional to l**beta
        beta = 0.33 for Kolmogorov

        Notably, this is the 3D velocity offset.

        Anchor point is in self.turbulent_anchor (l, v_offset)

        v_diff = A * l**beta
        log(v_diff) = beta*log(l) + log(A)
        """
        if self.turbulent_anchor is None:
            self.calculate_turbulence()
        v_diff = self.turbulent_anchor[1]
        l = self.turbulent_anchor[0]
        beta = self.params['beta']

        logA = np.log10(v_diff) - beta * np.log10(l)
        return 10**( beta * np.log10(distance) + logA )

    def calculate_subcloud_turbulent_velocity(self, distance):
        """
        Same as extrapolate_turbulent_velocity() but multiply this velocity diff
        by sqrt(rho_hot / rho_cool) = 0.1 and cap at sound speed = 10 km/s to
        ensure realistic behavior in cool gas
        """
        #factor = np.sqrt(self.params['n_bg'] / self.params['n_cl'])
        factor = 0.1
        velocities = self.extrapolate_turbulent_velocity(distance) * factor
        return np.clip(velocities, 0, 10)

    def calculate_thermal_b_with_turbulence(self, thermal_b, distances):
        """
        Determine the thermal_b (1D Voigt Profile broadening) including both
        thermal and turbulent broadening from sub-cloud turbulence

        Distances included are the path lengths of sightlines through each cloud.
        Assuming the turbulent velocity field cascades down to small sub-cloud
        scales, it will have cause turbulent broadening in the voigt profile
        of absroption features.  This properly accounts for this effect by
        adding the thermal_b with the turbulent_b (1D) in quadrature.
        """
        turb_broad = self.calculate_subcloud_turbulent_velocity(distances)
        turb_broad /= np.sqrt(3)      # 1D velocity from 3D velocity assuming isotropic
        turb_broad *= km/s
        return np.sqrt(thermal_b**2 + turb_broad**2)

def generate_random_coordinates(N, center, radius):
    """
    Generates N random 2D cartesian coords across a circle
    """
    # In order to get a truly random sample across the circle, first grab 3N
    # random points across NxN box, then filter out only the ones in the inner
    # circle, and take first N values present
    coords = ((np.random.rand(3*N, 2) * 2*radius) - radius) + center
    dist = np.linalg.norm(center - coords, axis=1)
    inside_circle = dist < radius
    return coords[inside_circle][:N]

class Rays:
    """
    A class to represent a collection of rays as a series of arrays representing their
    positions, EWs, column densities, etc.
    """
    def __init__(self, N=None, center=None, radius=None, coords=None, generate=True):
        """
        Can either accept ray coords or generate N random coords
        """
        self.N = N
        self.center = center
        self.radius = radius
        if generate:
            self.coords = generate_random_coordinates(self.N, self.center, self.radius)
            self.index = np.arange(N) #for indexing spectra

        self.clouds = {}
        self.column_densities = np.zeros(N, dtype=float)
        self.n_clouds = np.zeros(N, dtype=float)
        self.EWs = np.zeros(N, dtype=float)
        self.spectra = {}

    def __repr__(self):
        return('Rays: %d rays centered on %s with radius %f' % (self.N, self.center, self.radius))

    def plot_ray(self, i, clouds, velocity=True, filename=None, ax=None):
        """
        Plot ray's trajectory through clouds
        """
        if ax is None:
            fig, ax = plt.subplots()
            fig.set_size_inches(5.5, 1.5)
            standalone = True
        else:
            standalone = False

        # Method for using arbitrary colormap as color cycle
        n = len(self.clouds[i])
        colors = plt.cm.viridis(np.linspace(0,0.7,n))
        cloud_circs = [plt.Circle([clouds.centers[k,2], clouds.centers[k,1]], radius=clouds.radii[k], linewidth=0, color=colors[j]) for j, k in enumerate(self.clouds[i])]
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%+g"))
        for k in range(len(cloud_circs)):
            cloud_circ = cloud_circs[k]
            ax.add_artist(cloud_circ)
            cloud_circ.set_alpha(0.7)
        ax.plot((-5, 5), (self.coords[i, 1], self.coords[i, 1]), color='k', alpha=0.7)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        if velocity:
            ax.quiver(clouds.centers[self.clouds[i],2], clouds.centers[self.clouds[i],1], clouds.velocities[self.clouds[i],2], clouds.velocities[self.clouds[i],1], width=0.005, color = colors)
        ax.set_xlim(-5,5)
        ax.set_ylim(self.coords[i, 1]-1, self.coords[i, 1]+1)
        ax.set_aspect(1)
        ax.set_xlabel('z (kpc)')
        ax.set_ylabel('y (kpc)')
        if standalone:
            plt.tight_layout()
            if filename is None:
                filename = 'ray_%04d.png' % i
            plt.savefig(filename)
            plt.close()

    def save(self, filename):
        """
        Saves to an HDF5 file
        """
        print("Saving rays to %s" % filename)

        with h5py.File(filename, 'w') as f:
            f.create_dataset("coords", data=self.coords)
            f.create_dataset("column_densities", data=self.column_densities)
            f.create_dataset("EWs", data=self.EWs)
            f.create_dataset("n_clouds", data=self.n_clouds)
            f.attrs['N'] = self.N
            f.attrs['center'] = self.center
            f.attrs['radius'] = self.radius
            # TODO: Save attached spectra dictionary as a group with individual
            # spectra as datasets
        return

    def load(self, filename):
        """
        Loads from an HDF5 file
        """
        print("Loading rays from %s" % filename)
        with h5py.File(filename, 'r') as f:
            self.coords = f["coords"][:]
            self.column_densities = f["column_densities"][:]
            self.EWs = f["EWs"][:]
            self.n_clouds = f["n_clouds"][:]
            self.N = f.attrs['N']
            self.center = f.attrs['center']
            self.radius = f.attrs['radius']
        return

def calculate_intersections(coord, clouds):
    """
    Calculate if a Ray intersects a population of clouds of size r.
    If no intersection, or if tangent to sphere: return 0
    If it intersects, then return the path length of the intersection through the sphere

    Algorithm:
    https://diegoinacio.github.io/computer-vision-notebooks-page/pages/ray-intersection_sphere.html
    """
    # Calculate distance between ray and all cloud centers in x/y space
    d = np.linalg.norm(coord - clouds.centers[:, :2], axis=1)

    # all clouds that are intersected have d<r
    # when d>r, this raises " RuntimeWarning: invalid value encountered in sqrt"
    # so suppressing that temporarily
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pathlengths = 2*(clouds.radii**2 - d**2)**0.5

    # when d>r, pathlengths will be imaginary
    return np.nan_to_num(pathlengths)

def deposit_voigt(dl, rv, column_density, lambda_0, f_value, gamma, thermal_b, lambda_field):
    """
    Deposit a voigt profile into a spectrum.

    Checks intersection of a ray and a cloud to identify pathlength
    Then calculates depth, width, and centroid of voigt profile from
    column density, temperature, and radial velocity of cloud.

    ray: Ray object
    rv: radial velocity
    column_density: column density of ion in cgs
    lambda_0: rest frame wavelength of transition in angstroms
    f_value: oscillator strength of transition
    gamma: multiplicity of transition
    thermal_b: thermal broadening b parameter
    lambda_field: array of wavelengths in angstroms

    """
    # Doppler shift = vz (ignores cosmological redshift!)
    rv *= km/s
    redshift = np.sqrt((1 + (rv / c)) / (1 - (rv / c))) - 1
    delta_lambda = lambda_0 * redshift # angstrom

    # Actual deposition step
    lambda_bins, tau_bins = \
        tau_profile(lambda_0, f_value, gamma, thermal_b, column_density, \
                    delta_lambda=delta_lambda, lambda_bins=lambda_field)

    return tau_bins

def plot_clouds(rays=None, clouds=None, number_labels=False, axis_labels=True,
                slice_depth=None, cloud_velocity=False, filename=None):
    """
    Plot clouds and rays as desired
    """
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 8)
    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)
    ax.set_aspect(1)
    if axis_labels:
        ax.set_xlabel('x (kpc)')
        ax.set_ylabel('y (kpc)')

   # Clouds in red
    if clouds is not None:
        print("Plotting Clouds")
        p = clouds.params

        # If a slice_depth is set, then only plot clouds in central slice in z
        if slice_depth is not None:
            slice_mask = np.abs(clouds.centers[:,2]) - slice_depth/2 < 0
            radii = clouds.radii[slice_mask]
            centers = clouds.centers[slice_mask]
            velocities = clouds.velocities[slice_mask]
            text4 = 'Slice in Z = %2.1f' % slice_depth
            plt.text(0.01, 0.89, text4, size='medium', weight='bold', color='black',transform=ax.transAxes)
        else:
            radii = clouds.radii
            centers = clouds.centers
            velocities = clouds.velocities

        alphas = (20 * radii).clip(0,1)
        n_clouds = len(radii)
        N = 1000
        n_batch = int(np.floor(n_clouds / N))
        for j in tqdm(range(n_batch), "Plotting Clouds"):
            circles = [plt.Circle((centers[j*N+i,0], centers[j*N+i,1]), radius=radii[j*N+i]) for i in range(N)]
            collection = PatchCollection(circles)
            #collection.set_alpha(0.2)
            collection.set_alpha(alphas[j*N:(j+1)*N])
            collection.set_color('tab:blue')
            collection.set_linewidth(0)
            ax.add_collection(collection)
        # Plot remainder
        remainder = n_clouds % N
        circles = [plt.Circle((centers[i,0], centers[i,1]), radius=radii[i]) for i in range(N*n_batch,n_clouds)]
        collection = PatchCollection(circles)
        #collection.set_alpha(0.2)
        collection.set_alpha(alphas[n_batch*N:])
        collection.set_color('tab:blue')
        collection.set_linewidth(0)
        ax.add_collection(collection)

        if cloud_velocity:
            plt.quiver(centers[:,0], centers[:,1], velocities[:,0], velocities[:,1], color='k')
        text1 = 'Num Clouds = %2d' % n_clouds
        plt.text(0.01, 0.97, text1, size='medium', weight='bold', color='black',transform=ax.transAxes)
        text2 = 'Cloud Mass Range = [%4.2g, %4.2g] Msun' % (p['mclmin'], p['mclmax'])
        plt.text(0.01, 0.95, text2, size='medium', weight='bold', color='black',transform=ax.transAxes)
        text3 = 'Total Cloud Mass = %4.2g Msun' % p['cloud_mass']
        plt.text(0.01, 0.93, text3, size='medium', weight='bold', color='black',transform=ax.transAxes)
        text4 = 'Volume Filling Factor = %4.4f' % p['filling_factor']
        plt.text(0.01, 0.91, text4, size='medium', weight='bold', color='black',transform=ax.transAxes)

    # Rays in black
    if rays is not None:
        print("Plotting Rays")
        centers = rays.coords
        n_clouds = rays.N
        ray_hits = [plt.Circle((centers[i,0], centers[i,1]), radius=0.02) for i in range(n_clouds) if rays.n_clouds[i] > 0]
        collection_hits = PatchCollection(ray_hits)
        collection_hits.set_alpha(0.7)
        collection_hits.set_color('k')
        collection_hits.set_linewidth(0)
        ax.add_collection(collection_hits)

        ray_misses = [plt.Circle((centers[i,0], centers[i,1]), radius=0.02) for i in range(n_clouds) if rays.n_clouds[i] == 0]
        collection_misses = PatchCollection(ray_misses)
        collection_misses.set_alpha(0.3)
        collection_misses.set_color('k')
        collection_misses.set_linewidth(0)
        ax.add_collection(collection_misses)

    plt.tight_layout()
    if filename is None:
        filename = 'clouds.png'
    plt.savefig(filename)
    plt.close("all")

class SpectrumGenerator():
    """
    Similar but different from Trident's SpectrumGenerator class since not based in yt

    Defined by cloud distribution, temperature, and number density.

    Sets spectrum as centered around Mg II 2796 line
    """
    def __init__(self, clouds, show=True, subcloud_turb=True):
        self.instrument = 'HIRES'
        self.line = 'Mg II'
        self.clouds = clouds
        p = clouds.params
        temperature_cloud = p['T_cl']
        self.subcloud_turb = subcloud_turb
        self.metallicity_cloud = p['Z_cl']

        # Mg II Line Info
        self.lambda_0 = 2796.35 #* angstrom #AA
        self.f_value = 0.6155
        self.gamma = 2.625e+08 #/ s # 1 / s>
        self.mg_mass = 24.305 * amu
        self.show = True

        # Cloud properties (with units)
        self.temperature_cloud = temperature_cloud * K

        # Thermal broadening b parameter
        self.thermal_b = np.sqrt((2 * boltzmann_constant_cgs * self.temperature_cloud) /
                            self.mg_mass)
        self.thermal_b.convert_to_units('cm/s')

        # Define spectral pixel bin width by velocity
        #self.bin_width = 1e-2 # in angstroms
        # HIRES bin widths = 2.2 km/s ~ 0.02 Angstroms for Mg II
        # Use HIRES bin widths / 2 for smoothness in plotting individual cloudlet spectra,
        # then for EW calc and plotting full spectrum, downgrade resolution to HIRES
        velocity_bin_width = 2.2 * km/s
        z = velocity_bin_width / c
        bin_width = (z * self.lambda_0).d # in angstroms
        self.bin_width = bin_width / 2. # 2x finer than HIRES

        # Spectrum consists of lambda, tau, velocity, and flux fields
        # Flux is derived from tau (optical depth) and
        # velocity is derived from lambda (wavelength)

        ## KHRR -- changed below line from 3 Ang
        self.lambda_range = 6 # in angstroms
        self.lambda_min = self.lambda_0 - self.lambda_range / 2. # in angstroms
        self.lambda_max = self.lambda_0 + self.lambda_range / 2. # in angstroms
        self.lambda_field = np.linspace(self.lambda_min, self.lambda_max, \
                                        int(self.lambda_range / self.bin_width))
        z = (self.lambda_field / self.lambda_0) - 1
        self.velocity_field = z * c
        self.velocity_field.convert_to_units('km/s')
        self.clear_spectrum()
        self.create_LSF_kernel()

        # Initialize these for averaging spectrum over multiple sightlines
        self.tau_sum_field = np.zeros_like(self.tau_field)
        self.flux_sum_field = np.zeros_like(self.tau_field)
        self.n_sightlines = 0
        self.sum_intersections = 0
        self.sum_column_density = 0

    def make_spectrum(self, rays, i, mgII_frac, attach=False):
        """
        For a ray passing through the defined cloud distribution, Actually make the
        spectrum by stepping through and depositing voigt profiles for each cloud
        intersected by ray.

        Optionally attach each spectrum to the Rays class that called it
        """
        Z_cl = self.metallicity_cloud # units of Zsolar
        X_hydrogen = 0.74 # Hydrogen mass fraction
        self.rays = rays
        self.i = i
        dls = calculate_intersections(rays.coords[i], self.clouds) * kpc
        mask = np.nonzero(dls)[0]
        self.mask = mask
        rays.clouds[i] = mask
        rays.n_clouds[i] = len(mask)
        column_densities = dls*self.clouds.params['n_cl'] / cm**3
        mg_column_densities = solar_abundance['Mg']*column_densities * Z_cl * X_hydrogen
        mgII_column_densities = mg_column_densities * mgII_frac
        rays.column_densities[i] = np.sum(mgII_column_densities)
        self.tau_field = np.zeros_like(self.lambda_field)
        self.tau_fields = np.zeros([len(mask), self.lambda_field.shape[0]])

        if self.subcloud_turb:
            # Adding intrinsic turbulence in each cloud to thermal_b so voigt profile
            # broadening includes both thermal and turbulent effects
            broadening = self.clouds.calculate_thermal_b_with_turbulence(
                                    self.thermal_b,
                                    dls[mask])
        else:
            broadening = np.repeat(self.thermal_b, len(mask))

        for index, j in enumerate(mask):
            self.tau_fields[index,:] += deposit_voigt(dls[j], self.clouds.velocities[j,2],
                                        mgII_column_densities[j], \
                                        self.lambda_0, self.f_value, self.gamma, \
                                        broadening[index], self.lambda_field)

        self.tau_field = np.sum(self.tau_fields, axis=0)
        self.flux_fields = np.exp(-self.tau_fields)
        self.flux_field = np.exp(-self.tau_field)

        # keep track of how many spectra are being averaged together
        self.n_sightlines += 1
        self.flux_sum_field += self.flux_field
        self.sum_intersections += rays.n_clouds[i]
        self.sum_column_density += rays.column_densities[i]

        # if the ray intersected clouds, then apply LSF and calculate EW
        if rays.n_clouds[i] > 0:
            self.apply_LSF()
            self.calculate_equivalent_width()
            rays.EWs[i] = self.EW

        # Attach the spectra to the Rays object
        if attach:
            self.rays.spectra[i] = Spectrum(self.lambda_field, self.flux_field, \
                                            self.lambda_0, i, self.EW, \
                                            rays.column_densities[i])

    def make_mean_spectrum(self):
        """
        Create the flux mean field as the arithmetic mean of all flux fields
        across all rays
        """
        self.flux_mean_field = self.flux_sum_field / self.n_sightlines
        self.apply_LSF(mean=True)
        self.calculate_equivalent_width(mean=True)

    def clear_spectrum(self, total=False):
        """
        Clear out flux fields for a given spectrum to re-use SpectrumGenerator
        on another Ray
        """
        self.ray = None
        self.tau_field = np.zeros_like(self.lambda_field)
        self.flux_field = np.zeros_like(self.lambda_field)
        self.EW = 0.
        self.mask = []
        self.i = 0

        if total:
            self.n_sightlines = 0
            self.sum_intersections = 0
            self.sum_column_density = 0
            self.rays = None
            self.i = 0
            self.mask = []
            self.tau_sum_field = 0
            self.flux_sum_field = 0
            self.flux_mean_field = 0
            self.n_sightlines = 0
            self.sum_intersections = 0
            self.sum_column_density = 0

    def calculate_equivalent_width(self, mean=False):
        """
        Calculate equivalent width of an absorption spectrum in angstroms
        """
        # degrading resolution of spectrum to HIRES when calculating EW
        if mean:
            flux_field = self.flux_mean_field[::2]
            self.mean_EW = np.sum(np.ones_like(flux_field) - flux_field) * self.bin_width*2 * angstrom
        else:
            flux_field = self.flux_field[::2]
            self.EW = np.sum(np.ones_like(flux_field) - flux_field) * self.bin_width*2 * angstrom

    def create_LSF_kernel(self, resolution=45000.):
        """
        Create line spread function kernel to smear out spectra to appropriate
        spectral resolution.  Defaults to Keck HIRES R=45000.
        """
        delta = self.lambda_0 / resolution
        # Note: delta is resolution ~ FWHM
        # whereas Gaussian1DKernel gives width in 1 stddev
        # FWHM ~ 2.35sigma
        # HIRES FWHM Res ~ 0.062 Angstroms -> 7 km/s

        gaussian_width = delta / (self.bin_width * 2.35)
        self.LSF_kernel = Gaussian1DKernel(gaussian_width)

    def apply_LSF(self, mean=False):
        """
        Apply the line spread function after all voigt profiles
        have been deposited.
        """
        if mean:
            self.flux_mean_field = convolve(self.flux_mean_field, \
                                            self.LSF_kernel, boundary='extend')
            np.clip(self.flux_mean_field, 0, np.inf, out=self.flux_mean_field)
        else:
            for j, k in enumerate(self.mask):
                self.flux_fields[j,:] = convolve(self.flux_fields[j,:], self.LSF_kernel, boundary='extend')
            np.clip(self.flux_field, 0, np.inf, out=self.flux_field)
            self.flux_field = convolve(self.flux_field, self.LSF_kernel, boundary='extend')
            np.clip(self.flux_field, 0, np.inf, out=self.flux_field)


    def plot_spectrum(self, filename=None, component=False,
                      velocity=True, annotate=True, mean=False, cloud_vels=False,
                      ray_plot=False):
        """
        Plot spectrum in angstroms or km/s to filename
        """
        if ray_plot:
            fig, axs = plt.subplots(2, 1, height_ratios=[3,1])
            ax = axs[0]
            fig.set_size_inches(5.5, 6)
            self.rays.plot_ray(self.i, clouds=self.clouds, ax=axs[1])
            j = k = 0
        else:
            fig, ax = plt.subplots()
            fig.set_size_inches(5.5, 4.5)
        if mean:
            flux_field = self.flux_mean_field
            EW_val = self.mean_EW
            column_density = self.sum_column_density
            n_intersections = self.sum_intersections
        else:
            flux_field = self.flux_field
            EW_val = self.rays.EWs[self.i]
            column_density = self.rays.column_densities[self.i]
            n_intersections = self.rays.n_clouds[self.i]
            flux_fields = self.flux_fields

        if velocity:
            x_field = self.velocity_field
            ax.set_xlim(self.velocity_field.min(), self.velocity_field.max())
            ax.set_xlabel('Velocity [km/s]')
            #if cloud_vels:
            #    for cloud in clouds:
            #        plt.plot(2*[cloud.vz], [1-cloud.m_scaled, 1], color='k', \
            #                 alpha=cloud.m_scaled/100)
        else:
            x_field = self.lambda_field
            ax.set_xlim(self.lambda_field.min(), self.lambda_field.max())
            ax.set_xlabel('$\lambda$ [Angstrom]')

        if component:
            # Method for using arbitrary colormap as color cycle
            n = len(self.mask)
            color = plt.cm.viridis(np.linspace(0,0.7,n))
            ax.set_prop_cycle('color', color)

            for j, k in enumerate(self.mask):
                p = ax.plot(x_field, flux_fields[j,:])
                if annotate:
                    # Deal with cutoff annotations
                    if j==39:
                        text = "and %d more..." % (len(self.mask) - j)
                        xcoord = 0.02 + 0.70*(np.floor(j/20))
                        ycoord = 0.78 - 0.04*j + 0.80*np.floor(j/20)
                        ax.text(xcoord, ycoord, text, size='small', weight='bold',
                            color='k',transform=ax.transAxes)
                        continue
                    if j>39: continue
                    text = "%.1g M$_{\odot}$ %2d km/s" % (self.clouds.masses[k], self.clouds.velocities[k,2])
                    cindex = p[0].get_color()
                    xcoord = 0.02 + 0.70*(np.floor(j/20))
                    ycoord = 0.78 - 0.04*j + 0.80*np.floor(j/20)
                    ax.text(xcoord, ycoord, text, size='medium', weight='bold',
                            color=cindex,transform=ax.transAxes)

        # Finally plot the full spectrum
        # Sample every 2 elements to degrade resolution to HIRES
        #ax.plot(x_field, flux_field, 'k') # full resolution HIRES/2
        ax.step(x_field[::2], flux_field[::2], 'k', where='mid', linewidth=2) #HIRES resolution

        ax.set_ylim(0, 1.2)
        ax.set_ylabel('Relative Flux')
        ax.xaxis.set_major_locator(MultipleLocator(50))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        if velocity == True:
            ax.set_xlim(-110, 110)
        if annotate:
            if mean:
                text1 = 'Mean: Num Clouds = %2d' % n_intersections
            else:
                #text1 = '#%04d: Num Clouds = %2d' % (self.i, n_intersections)
                text1 = 'Number of Cloudlets = %2d' % n_intersections
            ax.text(0.02, 0.95, text1, size='medium', weight='bold', color='black',transform=ax.transAxes)
            text2 = 'Column Density = %3.1g ${\\rm cm^{-2}}$' % column_density
            ax.text(0.02, 0.90, text2, size='medium', weight='bold', color='black',transform=ax.transAxes)
            text3 = 'EW = %4.2f ${\\rm \AA}$' % EW_val
            ax.text(0.02, 0.85, text3, size='medium', weight='bold', color='black',transform=ax.transAxes)
            #if self.subcloud_turb:
            #    text3 = 'Subcloud Turbulence'
            #    ax.text(0.98, 0.95, text3, size='medium', weight='bold', color='black', horizontalalignment='right', transform=ax.transAxes)

        plt.tight_layout()
        if filename is None:
            if mean:
                filename = 'spectrum_mean.png'
            else:
                filename = 'spectrum_%04d.png' % self.i

        plt.savefig(filename)
        plt.close()

    def generate_ray_sample(self, N, center, radius, first_n_to_plot=0,
                            component=True, ray_plot=True, attach_spectra=False):
        """
        Create a sample of N rays spanning an aperture with center and radius specified.
        Calculate their spectra passing through the associated cloud distribution.
        Returns set of rays (with EWs, clouds intersected, and column densities).
        """
        rays = Rays(N, center, radius, generate=True)

        # Determine Mg II ion abundance based on cool gas number density
        # mgII_frac = 0.33 # Accurate for n_cl = 0.01
        mgII_frac = calc_mgII_frac(self.clouds.params['n_cl'])

        print("n_cl = %f" % self.clouds.params['n_cl'])
        print("mgII frac = %f" % mgII_frac)

        for i in tqdm(range(N), "Generating Rays"):
            self.make_spectrum(rays, i, mgII_frac, attach=attach_spectra)
            # Plot spectrum after applying line spread function
            if i < first_n_to_plot and self.EW > 0:
                self.plot_spectrum(component=True, ray_plot=True)
            self.clear_spectrum()
        self.clear_spectrum(total=True)
        return rays

class Spectrum():
    """
    A class for individual spectra
    """
    def __init__(self, lambda_field, flux_field, lambda_0, i, EW=None, column_density=None):
        self.lambda_field = lambda_field
        self.flux_field = flux_field
        self.lambda_0 = lambda_0
        self.i = i
        self.EW = EW
        self.column_density = column_density

    def save(self, filename=None):
        """
        Saves spectrum to an HDF5 file
        """

        if filename is None:
            filename = 'spectrum_%04d.h5' % self.i

        with h5py.File(filename, 'w') as f:
            f.create_dataset("lambda_field", data=self.lambda_field)
            f.create_dataset("flux_field", data=self.flux_field)
            f.attrs['lambda_0'] = self.lambda_0
            f.attrs['i'] = self.i
            f.attrs['EW'] = self.EW
            f.attrs['column_density'] = self.column_density
        return

    def load(self, filename):
        """
        Loads spectrum from an HDF5 file
        """
        with h5py.File(filename, 'r') as f:
            self.lambda_field = f['lambda_field'][:]
            self.flux_field = f['flux_field'][:]
            self.lambda_0 = f.attrs['lambda_0']
            self.i = f.attrs['i']
            self.EW = f.attrs['EW']
            self.column_density = f.attrs['column_density']
        return

def plot_histogram(field, label, clouds, x_max=None, log=False, xrange=None, n_bins=None, annotate=None, ax=None):
    """
    Create a simple histogram for field
    """
    print("Plotting histogram of %s" % label)
    p = clouds.params
    if ax is None:
        fig, ax = plt.subplots()
        standalone = True
    else:
        standalone = False
    bins = None
    if n_bins is None:
        n_bins = 20
    if xrange is not None:
        if log:
            bins = np.logspace(xrange[0], xrange[1], num=n_bins)
        else:
            bins = np.linspace(xrange[0], xrange[1], num=n_bins)
    if log:
        plt.xscale('log')
    else:
        plt.xscale('linear')
    if bins is None:
        bins = 20
    y_max = len(field)
    plt.hist(field, bins=bins, log=True, align='mid', edgecolor='black')#, density=True)
    ax.set_ylim(0.8,y_max)
    ax.set_ylabel('N')
    #ax.set_ylabel('PDF')
    ax.set_xlabel(label)
    label = label.replace(" ", "_")
    filename = 'hist_%s.png' % label
    text1 = 'Num Clouds = %2d' % len(clouds.radii)
    plt.text(0.01, 0.95, text1, size='medium', weight='bold', color='black',transform=ax.transAxes)
    text2 = 'Cloud Mass Range = [%4.2g, %4.2g] Msun' % (p['mclmin'], p['mclmax'])
    plt.text(0.01, 0.91, text2, size='medium', weight='bold', color='black',transform=ax.transAxes)
    if annotate is not None:
        plt.text(0.99, 0.95, annotate, size='medium', weight='bold', color='black', horizontalalignment='right', transform=ax.transAxes)
    if standalone:
        plt.tight_layout()
        plt.savefig(filename)
        plt.close("all")

def plot_2histogram(field1, field2, label, label1, label2, text=None, x_max=None, log=False, xrange=None, n_bins=None, fn=None):
    """
    Create a histogram for two separate fields and do a KS test
    """
    print("Plotting histogram of %s" % label)
    fig, ax = plt.subplots()
    bins = None
    if n_bins is None:
        n_bins = 20
    if xrange is not None:
        if log:
            bins = np.logspace(xrange[0], xrange[1], num=n_bins)
        else:
            bins = np.linspace(xrange[0], xrange[1], num=n_bins)
    if log:
        plt.xscale('log')
    else:
        plt.xscale('linear')
    if bins is None:
        bins = 20
    y_max = len(field1)
    plt.hist(field1, bins=bins, log=True, align='mid', edgecolor='black', alpha=0.5, label=label1)
    plt.hist(field2, bins=bins, log=True, align='mid', edgecolor='black', alpha=0.5, label=label2)
    ax.set_ylim(0.8,y_max)
    ax.set_ylabel('N')
    ax.set_xlabel(label)
    label = label.replace(" ", "_")
    if fn is None:
        filename = 'hist2_%s.png' % label
    else:
        filename = fn
    plt.legend()
    if text is not None:
    #text1 = 'KS statistic: %0.3f ; p-value: %0.2f' % (stat, pval)
        plt.text(0.01, 0.95, text, size='medium', weight='bold', color='black',transform=ax.transAxes)
    plt.savefig(filename)
    plt.close("all")

def plot_multi_histogram(fields, names, label, text=None, x_max=None, log=False, xrange=None, n_bins=None, fn=None):
    """
    Create a histogram with several fields plotted on top of each other
    """
    print("Plotting multi histogram of %s" % label)
    fig, ax = plt.subplots()
    bins = None
    if n_bins is None:
        n_bins = 20
    if xrange is not None:
        if log:
            bins = np.logspace(xrange[0], xrange[1], num=n_bins)
        else:
            bins = np.linspace(xrange[0], xrange[1], num=n_bins)
    if log:
        plt.xscale('log')
    else:
        plt.xscale('linear')
    if bins is None:
        bins = 20
    y_max = len(field1)
    for i in range(len(fields)):
        field = fields[i]
        name = names[i]
        plt.hist(field, bins=bins, log=True, align='mid', edgecolor='black', alpha=0.5, label=name)
    ax.set_ylim(0.8,y_max)
    ax.set_ylabel('N')
    ax.set_xlabel(label)
    label = label.replace(" ", "_")
    if fn is None:
        filename = 'hists_%s.png' % label
    else:
        filename = fn
    plt.legend()
    if text is not None:
    #text1 = 'KS statistic: %0.3f ; p-value: %0.2f' % (stat, pval)
        plt.text(0.01, 0.95, text, size='medium', weight='bold', color='black',transform=ax.transAxes)
    plt.savefig(filename)
    plt.close("all")

def bin_by(x, y, nbins=30, bins = None):
    """
    Divide the x axis into sections and return groups of y based on its x value
    Code from: https://github.com/tommlogan/watercolor/blob/master/watercolor.py
    """
    if bins is None:
        bins = np.linspace(x.min(), x.max(), nbins)

    bin_space = (bins[-1] - bins[0])/(len(bins)-1)/2

    indicies = np.digitize(x, bins + bin_space)

    output = []
    for i in range(0, len(bins)):
        output.append(y[indicies==i])
    #
    # prepare a dataframe with cols: median; mean; 1up, 1dn, 2up, 2dn, 3up, 3dn
    df_names = ['mean', 'median', '5th', '95th', '10th', '90th', '25th', '75th']
    df = pd.DataFrame(columns = df_names)
    to_delete = []
    # for each bin, determine the std ranges
    for y_set in output:
        if y_set.size > 0:
            av = y_set.mean()
            intervals = np.percentile(y_set, q = [50, 5, 95, 10, 90, 25, 75])
            res = [av] + list(intervals)
            df = df.append(pd.DataFrame([res], columns = df_names))
        else:
            # just in case there are no elements in the bin
            to_delete.append(len(df) + 1 + len(to_delete))


    # add x values
    bins = np.delete(bins, to_delete)
    df['x'] = bins

    return df

def create_and_plot_clouds(params, cloud_fil='clouds.h5'):
    """
    Just a wrapper function for creating and plotting resultant clouds
    """
    # Actually generate the clouds using parameters, then store to disk for later use
    clouds = Clouds(params=params)
    clouds.generate_clouds()
    #clouds.save('clouds.h5')
    clouds.save(cloud_fil)
    plot_clouds(clouds=clouds, filename='clouds.png')
    distances = np.linalg.norm(np.array([0,0,0]) - clouds.centers, axis=1)
    velocities = np.linalg.norm(np.array([0,0,0]) - clouds.velocities, axis=1)
    plot_histogram(distances, "cloud radial distances", clouds, log=True, xrange=[-1, 1])
    plot_histogram(clouds.masses, "cloud masses", clouds, log=True, xrange=[-3, 5])
    plot_histogram(velocities, "cloud total velocities", clouds, log=True, xrange=[-1,2.5])
    #clouds.calculate_turbulence(plot=True)
    #clouds.clobber_check()

def plot_covering_frac(rays_list, clouds_list):
    """
    Calculate and plot covering fraction for a collection of rays
    """
    fig, ax = plt.subplots()
    n_bins = 31
    bins = np.linspace(0, 0.5, num=n_bins)
    plt.xscale('linear')
    ax.set_ylim(0.0,1.0)
    ax.set_xlim(0.0,0.5)
    ax.set_ylabel('Covering Fraction')
    ax.set_xlabel('Equivalent Width ($\AA$)')
    for i in range(len(rays_list)):
        rays = rays_list[i]
        clouds = clouds_list[i]
        label = '%.1g ${\\rm M_{\odot}}$' % clouds.params['mclmin']
        ones = np.ones_like(rays.column_densities)
        vals = [len(ones[rays.EWs > thresh]) / rays.N for thresh in bins]
        plt.plot(bins, vals, label=label)
    plt.legend()
    fn = 'covering_fraction.png'
    plt.savefig(fn)
    plt.clf()

def plot_multi_clouds(clouds_list):
    """
    Plot three sets of clouds side by side
    """
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for j in range(len(clouds_list)):
        ax = axs[j]
        clouds = clouds_list[j]
        p = clouds.params
        print("Plotting %.1g mclmin clouds" % p['mclmin'])
        radii = clouds.radii
        alphas = (20 * radii).clip(0,1)
        centers = clouds.centers
        n_clouds = len(radii)
        N = 1000
        n_batch = int(np.floor(n_clouds / N))
        for j in tqdm(range(n_batch), "Plotting Clouds"):
            circles = [plt.Circle((centers[j*N+i,0], centers[j*N+i,1]), radius=radii[j*N+i]) for i in range(N)]
            collection = PatchCollection(circles)
            #collection.set_alpha(0.2)
            collection.set_alpha(alphas[j*N:(j+1)*N])
            collection.set_color('tab:blue')
            collection.set_linewidth(0)
            ax.add_collection(collection)
        # Plot remainder
        remainder = n_clouds % N
        circles = [plt.Circle((centers[i,0], centers[i,1]), radius=radii[i]) for i in range(N*n_batch,n_clouds)]
        collection = PatchCollection(circles)
        #collection.set_alpha(0.2)
        collection.set_alpha(alphas[n_batch*N:])
        collection.set_color('tab:blue')
        collection.set_linewidth(0)
        ax.add_collection(collection)

        # Set plot env
        ax.set_ylim(-5, 5)
        ax.set_xlim(-5, 5)
        text = '%1g ${\\rm M_{\odot}}$' % p['mclmin']
        ax.text(0.01, 0.97, text, size=18, weight='bold', color='black',transform=ax.transAxes)
        ax.axis('off')
        ax.set_aspect(1)
        wrap = plt.Circle([0,0], radius=5, linewidth=1, color='k', fill=False)
        ax.add_artist(wrap)

    scalebar = ScaleBar(1, "px", dimension='pixel-length',length_fraction=0.25, width_fraction=0.02, location='lower right', label='2 kpc', scale_loc="none", font_properties={'weight':'bold', 'size':18})
    axs[2].add_artist(scalebar)

    plt.tight_layout()
    fn = 'clouds_multi.png'
    plt.savefig(fn)
    plt.clf()

def plot_clouds_buffer(clouds, filename=None):
    """
    Add all clouds to a buffer image as column densities, then plot in log space of
    column densities.  Cleanest way to show high dynamical range of clouds.
    """
    p = clouds.params
    radii = clouds.radii
    centers = clouds.centers
    velocities = clouds.velocities
    N = 1000    # N pixels on a side of image
    buffer = np.zeros([N, N], dtype='float')
    X=Y= np.linspace(-p['dclmax'], p['dclmax'], N, endpoint=False)
    indices = np.arange(N)
    pix_width = np.abs(X[1] - X[0])
    coords = np.array([X,Y]).T

    for i in tqdm(range(len(clouds.masses)), "Plotting Clouds"):
        # identify pixels that intersect cloud
        # calculate N through center of each pixel
        # add N to buffer for pixel coord
        ixs = np.abs(clouds.centers[i, 0] - X)
        iys = np.abs(clouds.centers[i, 1] - Y)
        mask_x = ixs < clouds.radii[i]
        mask_y = iys < clouds.radii[i]
        arr = np.array(np.meshgrid(X[mask_x], Y[mask_y])).T
        #if 2*np.sqrt(2)*clouds.radii[i] < pix_width:
        # When are too small and  sit in between pixels, then just dump into closest pixel
        if (ixs.min() > clouds.radii[i] or iys.min() > clouds.radii[i]):
            x1 = np.argmin(ixs)
            y1 = np.argmin(iys)
            coldens = 2*clouds.radii[i]
            buffer[x1, y1] += coldens
        # Otherwise see which pixel centers overlap cloud
        else:
            x1 = indices[mask_x][0]
            x2 = indices[mask_x][-1]+1
            y1 = indices[mask_y][0]
            y2 = indices[mask_y][-1]+1
            d = np.linalg.norm(arr - clouds.centers[i, :2], axis=2)
            coldens = np.nan_to_num(2*(clouds.radii[i]**2 - d**2)**0.5)
            buffer[x1:x2, y1:y2] += coldens

    # Get to H column densities
    buffer *= kpc
    buffer *= p['n_cl'] * cm**-3
    buffer /= cm**-2
    print('min = %g max = %g' % (buffer[buffer > 0].min(), buffer.max()))

    # If no filename specified, then return the buffer (for use with other plotting
    # functions)

    if filename is None:
        return buffer

    # colorbar limits in column density
    minbuf = 5e14 # limits which seem to get extrema for different cloud distributions
    maxbuf = 1e21
    fig, ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(buffer, cmap='Blues', norm=colors.LogNorm(vmin=minbuf, vmax=maxbuf))#, norm='symlog')
    #plt.colorbar(im,fraction=0.046, pad=0.04)
    ax.axis('off')
    text = '$10^{%1g}~{\\rm M_{\odot}}$' % np.log10(p['mclmin'])
    #ax.text(0.01, 0.97, text, size=34, weight='bold', color='black',transform=ax.transAxes)

    # Bounding circle for total cloudlet collection distribution
    wrap = plt.Circle([N/2,N/2], radius=(N/2)-1, linewidth=2, color='k', fill=False)
    ax.add_artist(wrap)

    ax.set_aspect(1)
    plt.tight_layout()
    plt.savefig(filename)

def plot_multi_clouds_buffer(clouds_list):
    """
    Plot three sets of clouds side by side using plot_clouds_buffer functionality
    """
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # colorbar limits in column density
    minbuf = 5e14 # limits which seem to get extrema for different cloud distributions
    maxbuf = 1e21

    for j in range(len(clouds_list)):
        ax = axs[j]
        clouds = clouds_list[j]
        p = clouds.params
        print("Plotting %.1g mclmin clouds" % p['mclmin'])
        buffer = plot_clouds_buffer(clouds)
        #text = '%1g ${\\rm M_{\odot}}$' % p['mclmin']
        #ax.text(0.01, 0.97, text, size=18, weight='bold', color='black',transform=ax.transAxes)
        ax.axis('off')
        ax.set_aspect(1)
        im = ax.imshow(buffer, cmap='Blues', norm=colors.LogNorm(vmin=minbuf, vmax=maxbuf))#, norm='symlog')
        ax.axis('off')
        text = '$10^{%1g}~{\\rm M_{\odot}}$' % np.log10(p['mclmin'])
        ax.text(0.00, 0.97, text, size=26, horizontalalignment='left', weight='bold', color='black',transform=ax.transAxes)
        N = len(buffer)

        # Bounding circle for total cloudlet collection distribution
        wrap = plt.Circle([N/2,N/2], radius=(N/2)-3, linewidth=2, color='k', fill=False)
        ax.add_artist(wrap)
        ax.set_aspect(1)

    scalebar = ScaleBar(1.0, 'px', dimension='pixel-length',length_fraction=0.25, width_fraction=0.02, location='lower right', label='2 kpc', scale_loc="none", label_loc="top", font_properties={'weight':'bold', 'size':18}, pad=0, border_pad=0)
    axs[2].add_artist(scalebar)
    plt.tight_layout()
    fn = 'clouds_multi_buff.png'
    plt.savefig(fn)
    plt.clf()

def calc_mgII_frac(n_cl):
    """
    Estimates the Mg II ion fraction based on the cool cloud number density.
    This just does linear interpolation from the Trident ion balance lookup table
    which is based on many CLOUDY models assuming a Haardt-Madau 2012 UV background.
    For this function, we currently assume z=0.25 and T=10^4K.  This function requires
    the ion balance table to be present in the cloudflex directory: hm2012_hr.h5
    or in the directory where Trident automatically installs it: ~/.trident

    Notably, linear interpolation takes place in log space for the density as well as
    the ion fraction.
    """

    ion_balance_filename = os.path.expanduser(os.path.join('~','.trident','hm2012_hr.h5'))
    f = h5py.File(ion_balance_filename, 'r')
    log_MgII_vals = f['Mg'][1,:,2,120] # log(Mg II / Mg abundance)
    log_dens_bins = f['Mg'].attrs['Parameter1'] # log n(H)
    log_n_cl = np.log10(n_cl)
    log_MgII =  np.interp(log_n_cl, log_dens_bins, log_MgII_vals)
    return 10**log_MgII


def make_quad_plot(rays_list, clouds_list, filename='quad.png', flgs_dict=None):
    """
    Make Figure 3 from method paper, creates 4 plots side-by-side of various 
    observational counterparts based on rays.h5 file. PDFs of number of intersected
    clouds, column density of each sightline, EW of each sightline, and covering frac.
    """

    if(flgs_dict==None):

        flgs_dict = {}
        flgs_dict['mclmin'] = True
        flgs_dict['ncl'] = False
        flgs_dict['mtot'] = False
        flgs_dict['dclmax'] = False
        flgs_dict['Zcl'] = False
        flgs_dict['vmax'] = False
        flgs_dict['beta'] = False
        flgs_dict['alpha'] = False
        flgs_dict['zeta'] = False

    # gridspec inside gridspec
    fig = plt.figure(figsize=(16,4))

    gs0 = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[3,1])
    gs00 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs0[0], hspace=0.0, wspace=0.0)

    ax0 = fig.add_subplot(gs00[0])
    ax1 = fig.add_subplot(gs00[1])
    ax2 = fig.add_subplot(gs00[2])
    ax3 = fig.add_subplot(gs0[1])

    n = len(rays_list)
    fiducial = np.floor(n/2)

    for i in range(len(rays_list)):
        rays = rays_list[i]
        clouds = clouds_list[i]
        mask = rays.n_clouds > 0
        p = clouds.params
        if flgs_dict['mclmin']:
            color = plt.cm.viridis(np.linspace(0,0.90,n))  # mclmin
            text = '${\\rm m_{cl,min}}$'
            label = '${\\rm m_{cl, min}}$ = $10^{%1d}$ ${\\rm M_{\odot}}$' % np.log10(clouds.params['mclmin'])
        #if mclmax:
        #    color = plt.cm.viridis(np.linspace(0,0.90,n))  # mclmin
        #    text = '${\\rm m_{cl,max}}$'
        #    label = '${\\rm m_{cl, max}}$ = $10^{%1d}$ ${\\rm M_{\odot}}$' % np.log10(clouds.params['mclmax'])
        if flgs_dict['ncl']:
            color = plt.cm.plasma(np.linspace(0,0.90,n))   # n_cl
            text = '${\\rm n_{cl}}$'
            label = '${\\rm n_{cl}}$ = %.1g ${\\rm cm^{-3}}$' % clouds.params['n_cl']
        if flgs_dict['mtot']:
            color = cmocean.cm.ice(np.linspace(0,0.80,n))   # m_tot
            text = '${\\rm m_{tot}}$'
            label = '${\\rm m_{tot}}$ = %.1g ${\\rm M_{\odot}}$' % clouds.params['total_mass']
        if flgs_dict['dclmax']:
            color = cmocean.cm.dense_r(np.linspace(0,0.80,n))  # dclmax
            text = '${\\rm d_{cl,max}}$'
            label = '${\\rm d_{cl, max}}$ = %2.f kpc' % clouds.params['dclmax']
        if flgs_dict['Zcl']:
            color = cmocean.cm.solar(np.linspace(0,0.90,n))
            text = '${\\rm Z_{cl}}$'
            #Z_vals = [3.33, 1.0, 0.33, 0.1, 0.033]
            #Z_val = Z_vals[i]
            #label = '${\\rm Z_{cl}}$ = %.1g ${\\rm Z_{\odot}}$' % Z_val
            label = '${\\rm Z_{cl}}$ = %.1g ${\\rm Z_{\odot}}$' % clouds.params['Z_cl']
        if flgs_dict['vmax']:
            color = cmocean.cm.algae_r(np.linspace(0,0.90,n))
            text = '${\\rm v_{max}}$'
            label = '${\\rm v_{max}}$ = %.1g km/s' % clouds.params['vmax']
        if flgs_dict['alpha']:
            color = cmocean.cm.rain_r(np.linspace(0,0.90,n))
            text = '${\\rm \\alpha}$'
            label = '${\\rm \\alpha}$ = %.2g' % clouds.params['alpha']
        if flgs_dict['zeta']:
            color = plt.cm.cividis(np.linspace(0,0.90,n))
            text = '${\\rm \\zeta}$'
            label = '${\\rm \\zeta}$ = %.2g' % clouds.params['zeta']
        if flgs_dict['beta']:
            color = cmyt.dusk(np.linspace(0,0.90,n))
            text = '${\\rm \\beta}$'
            label = '${\\rm \\beta}$ = %.2g' % clouds.params['beta']

        # First Plot is PDF of # of clouds intersected
        ax = ax0
        ax.set_prop_cycle('color', color)
        log_n_clouds = np.log10(rays.n_clouds[mask])
        bins = np.linspace(0,3,30)
        # Remove three bins in the interior to get rid of gaps in histogram around 1,2,3
        bins = np.delete(bins, [1,3,5])
        counts, bins = np.histogram(log_n_clouds, bins=bins)
        norm = counts / len(rays.n_clouds) # PDF
        ax.stairs(norm, bins, alpha=0.5, color=color[i], label=label, fill=True)
        if i == fiducial:
            ax.stairs(norm, bins, alpha=0.9, color='k', fill=False, linewidth=2)
        else:
            ax.stairs(norm, bins, alpha=0.5, color='k', fill=False)
        ax.set_yscale('log')

        # Second Plot is PDF of Log(Column Density)
        ax = ax1
        ax.set_prop_cycle('color', color)
        log_col_dens = np.log10(rays.column_densities[mask])
        bins = np.linspace(10,15,30)
        counts, bins = np.histogram(log_col_dens, bins=bins)
        norm = counts / len(rays.n_clouds) # PDF
        ax.stairs(norm, bins, alpha=0.5, color=color[i], fill=True)
        ax.stairs(norm, bins, alpha=0.5, color='k', fill=False)
        if i == fiducial:
            ax.stairs(norm, bins, alpha=0.9, color='k', fill=False, linewidth=2)
        else:
            ax.stairs(norm, bins, alpha=0.5, color='k', fill=False)
        ax.set_yscale('log')

        # Third Plot is PDF of EW
        ax = ax2
        ax.set_prop_cycle('color', color)
        bins = np.linspace(0,1,30)
        counts, bins = np.histogram(rays.EWs[mask], bins=bins)
        norm = counts / len(rays.n_clouds) # PDF
        ax.stairs(norm, bins, alpha=0.5, color=color[i], fill=True)
        if i == fiducial:
            ax.stairs(norm, bins, alpha=0.9, color='k', fill=False, linewidth=2)
        else:
            ax.stairs(norm, bins, alpha=0.5, color='k', fill=False)
        ax.set_yscale('log')

        # Fourth Plot is Covering Fraction vs EW
        ax = ax3
        ax.set_prop_cycle('color', color)
        n_bins = 30
        bins = np.linspace(0, 0.5, num=n_bins)

        ones = np.ones_like(rays.column_densities)
        vals = [len(ones[rays.EWs > thresh]) / rays.N for thresh in bins]
        if i == fiducial:
            #ax.plot(bins, vals, color='k', label=label, linewidth=2, alpha=0.5,
            #        path_effects=[pe.Stroke(linewidth=5, foreground=color[i]), pe.Normal()])
            ax.plot(bins, vals, color='w', linewidth=5,
                     path_effects=[pe.Stroke(linewidth=9, foreground='k', alpha=0.9), pe.Normal()])
        ax.plot(bins, vals, color=color[i], label=label, linewidth=5, alpha=0.5)
# custom plot settings

    ax = ax0
    ax.set_ylim(6e-5,3e-1)
    ax.set_xlim(0.0,2.2)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xlabel('log(Number of Intersected Cloudlets)')
    ax.set_ylabel('PDF')
    ax.grid(True, axis='y', linestyle='-', color='k', alpha=0.2)
    ax.text(0.95, 0.89, text, horizontalalignment='right', fontfamily='DejaVu Sans', size=28, weight='heavy', color='k', transform=ax.transAxes)

    ax = ax1
    ax.set_ylim(6e-5,3e-1)
    ax.set_xlim(9.8,15.0)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xlabel('log(Mg II Column Density (${\\rm cm^{-2}}$))')
    #ax.yaxis.set_visible(False)
    ax.yaxis.set_tick_params(labelleft=False)
    ax.grid(True, axis='y', linestyle='-', color='k', alpha=0.2)

    ax = ax2
    ax.set_ylim(6e-5,3e-1)
    ax.set_xlim(-0.05,1.05)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xlabel('Mg II Equivalent Width (${\\rm \AA}$)')
    #ax.yaxis.set_visible(False)
    ax.yaxis.set_tick_params(labelleft=False)
    ax.grid(True, axis='y', linestyle='-', color='k', alpha=0.2)

    ax = ax3
    ax.set_ylim(0.0,1.0)
    ax.set_xlim(0.0,0.5)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.set_ylabel('Covering Fraction')
    #ax.yaxis.set_tick_params(labelleft=False)
    #ax.yaxis.set_label_position("right")
    ax.grid(True, axis='y', linestyle='-', color='k', alpha=0.2)
    #ax.yaxis.tick_right()
    ax.set_xlabel('Mg II Equivalent Width (${\\rm \AA}$)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close("all")

def make_cloud_table():

    n_cl = 1e-2 / cm**3
    rho_cl = n_cl * proton_mass
    mass = np.logspace(-4, 5, 10, endpoint=True) * Msun
    radius = ((3*mass / (4*np.pi*rho_cl))**(1/3)).to('pc')
    column = 2*radius * n_cl
    Z_cl = 0.33
    mgII_fraction = 0.33 # from FIRE estimates at r=50 kpc for m12i @ z=0.25
    column_MgII = column * solar_abundance['Mg'] * Z_cl * mgII_fraction
    print("Mass         Radius      Column      MgIIColumn")
    for i in range(len(mass)):
        m = mass[i].to('Msun')
        r = radius[i].to('pc')
        N = column[i].in_cgs()
        N_mgII = column_MgII[i].in_cgs()
        print("%2.1g Msun  %02.1g pc   %02.1g   %02.1g" % (m, r, N, N_mgII))
