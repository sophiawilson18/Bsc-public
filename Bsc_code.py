import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from matplotlib.patches import Arc
from matplotlib import gridspec
from mpl_toolkits import mplot3d

from scipy.optimize import curve_fit
from scipy.ndimage import rotate
from scipy.special import hyp2f1
from scipy import optimize
from scipy.stats import norm, halfnorm
from scipy.special import hyperu, gamma
import scipy.optimize

from astropy.cosmology import Planck18 as cosmo
from astropy.constants import G, c, m_p
from astropy import stats
from astropy.io import fits
from astropy.visualization import astropy_mpl_style
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy import units as u
from astropy.modeling.models import Ellipse2D
from astropy.coordinates import Angle

import aplpy
import mpdaf.obj 

import sys 
from importlib import reload  
sys.path.append('../AppStat/AppStatSophia') 
import SophiasExternalFunctions as sw

sys.path.append('../') 
import BachelorProjectExternalFunctions as bp




# ======================================
# Single FRB, functions
# ======================================

def cutpeak(lowlim, highlim):
    '''Slicing data'''
    
    wl = wl_clean[np.where(np.logical_and(wl_clean > lowlim, wl_clean < highlim))]
    flux = flux_clean[np.where(np.logical_and(wl_clean > lowlim, wl_clean < highlim))]
    sig = sig_clean[np.where(np.logical_and(wl_clean > lowlim, wl_clean < highlim))]
    return wl, flux, sig


def redshift(x, z, sigma, N_Hbeta, N_OIII, N_NII1, N_Halpha, N_NII2, N_SII1, N_SII2, c):
    '''Redshift model with fixed redshift and line width'''
    
    exp_Hbeta, exp_OIII, exp_NII1, exp_Halpha, exp_NII2, exp_SII1, exp_SII2 = 4862.68, 5008.240, 6549.86, 6564.61, 6585.27, 6718.29, 6732.67
    
    y = c + 1/(sigma*np.sqrt(2*np.pi))*(N_Hbeta * np.exp(-(x-(z+1)*exp_Hbeta)**2/(2*sigma**2)) 
    + N_OIII * np.exp(-(x-(z+1)*exp_OIII)**2/(2*sigma**2)) 
    + N_NII1 * np.exp(-(x-(z+1)*exp_NII1)**2/(2*sigma**2)) 
    + N_Halpha * np.exp(-(x-(z+1)*exp_Halpha)**2/(2*sigma**2))
    + N_NII2 * np.exp(-(x-(z+1)*exp_NII2)**2/(2*sigma**2))
    + N_SII1 * np.exp(-(x-(z+1)*exp_SII1)**2/(2*sigma**2))
    + N_SII2 * np.exp(-(x-(z+1)*exp_SII2)**2/(2*sigma**2)))
    
    return y


def arc2kpc(theta, z):
    '''Converting arcseconds to kpc'''
    
    c = 299792458     #m/s 
    H0 = 67.31e-3     #m/s/pc
    q0 = 0.5
    
    theta_radian = theta * np.pi / 180 / 3600
    d = c/H0 * (q0*z+(q0-1)*((1+2*q0*z)**0.5-1))/(q0**2*(1+z)**2)
    print(d)
    
    return d*theta_radian /1000



def sim_dist(RA, dec):
    '''Simulating distance in arcsec between two objects where one has Gaussian uncertainties'''
    
    frb_pos = SkyCoord(f'21h33m{RA}s -54d44m{dec}s', frame='icrs')      
    hg_pos = SkyCoord('21h33m24.4648s -54d44m54.862s', frame='icrs')
    sep = frb_pos.separation(hg_pos).arcsecond
    return sep



def prob(m, R0, Rh):
    '''Probability for host galaxy being unrelated'''
    
    r = np.sqrt(R0**2+4*Rh**2)
    sigma = 1/(3600**2*0.334*np.log(10))*10**(0.334*(m-22.963)+4.320)
    n = np.pi*r**2*sigma
    P = 1-np.exp(-n)
    return P



def radial_profile(data):
    'Radial profile of galaxy'
    
    y, x = np.indices((data.shape))
    center = [len(x)/2, len(y)/2]
    
    x0, y0 = len(x)/2, len(y)/2 
    theta = 57*np.pi/180
    
    ellip = 1- 2.85 / 10  #1-b/a

    r_maj = (x-x0)*np.cos(theta) + (y-y0)*np.sin(theta)
    r_min = -(x-x0)*np.sin(theta) + (y-y0)*np.cos(theta)
    r = np.sqrt(r_maj**2 + (r_min/(1-ellip))**2) #+0.5
    r = r.astype(int)
    
    tbin = np.bincount(r.ravel(), data.ravel())  # return a contiguous flattened array
    nr = np.bincount(r.ravel())
    
    radialprofile = tbin / nr
    radialprofile = radialprofile -min(radialprofile)
    
    cum_light = np.cumsum(radialprofile)
    cum_light_normalized = np.cumsum(radialprofile) /max(cum_light)
    
    return radialprofile, cum_light_normalized, r




# ======================================
# Many FRBs, functions
# ======================================


def tau_DM(DM):
    '''Tau(DM) relation'''
    return 2.98e-7 * DM**(1.4) * (1+3.55e-5*DM**(3.1))




def DM_func(z):
    c_ = c.value            # m/s
    G_ = G.value            # m^3/kg/s^2
    H0 = cosmo.H0.value     # km/Mpc/s
    #H0 = H0_stat
    m_p_ = m_p.value        # kg
    pc_to_m = 3.08567758e16 # m/pc
    Omega_b = cosmo.Ob0
    Omega_m = cosmo.Om0
    Omega_de = cosmo.Ode0
    Y_He = 1/4
    f = 0.8
    
    constant = 3*f*Omega_b*H0*c_*(1-Y_He/2)/(8*np.pi*G_*m_p_)
    DM_cosmic = constant * 1/2 * Omega_de**(-1/2) * (1+z)**2 * hyp2f1(1/2, 2/3, 5/3, -Omega_m*(1+z)**3/Omega_de) 
    
    return DM_cosmic * 1/(10**3 * 1 * pc_to_m) / ( pc_to_m/(1e-2)**3 ) 

def DM_z(z):
    '''DM(z) relation'''
    return DM_func(z)-DM_func(0)




def prob_DM_C0(z, DM, C0):
    '''Function for determining C0'''
    
    alpha = 3
    beta = 3
    F = 0.32  # mass ratio of the ISM to stars
    DM_sigma = F*z**(-1/2)

    delta = DM/DM_z(z)
    prob_DM_delta = delta**-beta * np.exp(-(delta**-alpha-C0)**2/(2*alpha**2*DM_sigma**2))

    return prob_DM_delta, prob_DM, delta


def prob_DM(z, DM):
    '''pdf(delta)'''
    
    alpha = 3
    beta = 3
    F = 0.32
    DM_sigma = F*z**(-1/2)
    DMz = DM_z(z)
    delta = DM/DMz
    
    def determine_C(x):
        max_value = delta[np.where(prob_DM2(z, DM, x)[0]==max(prob_DM_C0(z, DM, x)[0]))]
        return abs(max_value-1)
    C0 = 0
    
    if z<0.15:
        start_guess = -20
    else: 
        start_guess = -5
    
    C0 = optimize.minimize(determine_C, [start_guess], method='Nelder-Mead').x[0]
    
    # Normalizing with respect to DM_delta
    A = 1/(DMz*gamma(2/3)*DM_sigma**(2/3)*np.exp(-C0**2/(18*DM_sigma**2))*hyperu(1/3,1/2,C0**2/(18*DM_sigma**2))*6**(-1/3))
    prob_DM_delta = A*delta**-beta * np.exp(-(delta**-alpha-C0)**2/(2*alpha**2*DM_sigma**2)) 
    
    prob_DM = prob_DM_delta * DMz 
    
    return prob_DM_delta*100, prob_DM, delta


### Fit functions

def DM_func_fit_OmegabH0(z, OmegabH0):
    c_ = c.value            # m/s
    G_ = G.value            # m^3/kg/s^2
    #H0 = cosmo.H0.value     # km/Mpc/s
    m_p_ = m_p.value        # kg
    pc_to_m = 3.08567758e16 # m/pc
    #Omega_b = cosmo.Ob0
    Omega_m = cosmo.Om0
    Omega_de = cosmo.Ode0
    Y_He = 1/4
    f = 0.8
    
    constant = 3*f*OmegabH0*c_*(1-Y_He/2)/(8*np.pi*G_*m_p_)
    DM_cosmic = constant * 1/2 * Omega_de**(-1/2) * (1+z)**2 * hyp2f1(1/2, 2/3, 5/3, -Omega_m*(1+z)**3/Omega_de) 
    
    return DM_cosmic * 1/(10**3 * 1 * pc_to_m) / ( pc_to_m/(1e-2)**3 ) 

def DM_z_fit_OmegabH0(z, OmegabH0):
    '''Fit OmegabH0'''
    return DM_func_fit_OmegabH0(z, OmegabH0)-DM_func_fit_OmegabH0(0, OmegabH0)



def DM_func_fit_Omegab(z, Omegab):
    c_ = c.value            # m/s
    G_ = G.value            # m^3/kg/s^2
    H0 = cosmo.H0.value     # km/Mpc/s
    m_p_ = m_p.value        # kg
    pc_to_m = 3.08567758e16 # m/pc
    #Omega_b = cosmo.Ob0
    Omega_m = cosmo.Om0
    Omega_de = cosmo.Ode0
    Y_He = 1/4
    f = 0.8
    
    constant = 3*f*Omegab*H0*c_*(1-Y_He/2)/(8*np.pi*G_*m_p_)
    DM_cosmic = constant * 1/2 * Omega_de**(-1/2) * (1+z)**2 * hyp2f1(1/2, 2/3, 5/3, -Omega_m*(1+z)**3/Omega_de) 
    
    return DM_cosmic * 1/(10**3 * 1 * pc_to_m) / ( pc_to_m/(1e-2)**3 ) 

def DM_z_fit_Omegab(z, Omegab):
    '''Fit Omegab'''
    return DM_func_fit_Omegab(z, Omegab)-DM_func_fit_Omegab(0, Omegab)



def DM_func_fit_H0(z, H0):
    '''Fit H0'''
    c_ = c.value            # m/s
    G_ = G.value            # m^3/kg/s^2
    #H0 = cosmo.H0.value     # km/Mpc/s
    m_p_ = m_p.value        # kg
    pc_to_m = 3.08567758e16 # m/pc
    Omega_b = cosmo.Ob0
    Omega_m = cosmo.Om0
    Omega_de = cosmo.Ode0
    Y_He = 1/4
    f = 0.8
    
    constant = 3*f*224.2/H0*c_*(1-Y_He/2)/(8*np.pi*G_*m_p_)
    DM_cosmic = constant * 1/2 * Omega_de**(-1/2) * (1+z)**2 * hyp2f1(1/2, 2/3, 5/3, -Omega_m*(1+z)**3/Omega_de) 
    
    return DM_cosmic * 1/(10**3 * 1 * pc_to_m) / ( pc_to_m/(1e-2)**3 ) 

def DM_z_fit_H0(z, H0):
    return DM_func_fit_H0(z, H0)-DM_func_fit_H0(0, H0)



### Likelihood: 

def _pdf_(delta, z_final, C0):
    alpha, beta, F = 3, 3, 0.32
    DM_sigma = F*z_final**(-1/2)
    pdf = delta**-beta * np.exp(-(delta**-alpha-C0)**2/(2*alpha**2*DM_sigma**2)) 
    return -pdf
    
def get_pdf_mode(z_final, C0):
    return scipy.optimize.minimize(lambda delta: _pdf_(delta, z_final, C0), [1]).x
    
def guess_C0(z_final):
    return scipy.optimize.root(lambda C0: get_pdf_mode(z_final, C0)-1, [-5]).x

def likelihood_H0(a):
    '''Likelihood function H0'''
    
    alpha, beta, F = 3, 3, 0.32
    DM_sigma = F*zs**(-1/2)
    exp_DM = DM_z_fit_H0(zs, a)  
    delta_DM = DM_cosmic_final/exp_DM    

    A = 1/(exp_DM*gamma(2/3)*DM_sigma**(2/3)*np.exp(-C0**2/(18*DM_sigma**2))*hyperu(1/3,1/2,C0**2/(18*DM_sigma**2))*6**(-1/3))
    
    log_pdf = np.log(A)-beta*np.log(delta_DM)-(delta_DM**-alpha-C0)**2/(2*alpha**2*DM_sigma**2)
    
    logl = -np.sum(log_pdf)
    
    return logl


def likelihood_Omegab(a):
    '''Likelihood function Omegab'''
    
    alpha, beta, F = 3, 3, 0.32
    DM_sigma = F*zs**(-1/2)
    exp_DM = DM_z_fit_Omegab(zs, a)  
    delta_DM = DM_cosmic_final/exp_DM    

    A = 1/(exp_DM*gamma(2/3)*DM_sigma**(2/3)*np.exp(-C0**2/(18*DM_sigma**2))*hyperu(1/3,1/2,C0**2/(18*DM_sigma**2))*6**(-1/3))
    
    log_pdf = np.log(A)-beta*np.log(delta_DM)-(delta_DM**-alpha-C0)**2/(2*alpha**2*DM_sigma**2)
    
    logl = -np.sum(log_pdf)
    
    return logl