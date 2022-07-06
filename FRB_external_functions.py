import numpy as np
import pandas as pd
from iminuit import Minuit                             

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
from scipy import stats
import scipy.optimize

from astropy.cosmology import Planck18 as cosmo
from astropy.constants import G, c, m_p
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


sys.path.append('AppStat/AppStatSophia/ExternalFunctions.py')
from ExternalFunctions import Chi2Regression
from ExternalFunctions import nice_string_output, add_text_to_ax 




# ======================================
#            Single FRB
# ======================================


def unc(stat, sys):
    '''Error propagating statistical and systemmatical uncertainties'''
    return np.sqrt(stat**2 + sys**2)


def cutpeak(wl, flux, sig, lowlim, highlim):
    '''Slicing data'''
    
    wl_cut = wl[np.where(np.logical_and(wl > lowlim, wl < highlim))]
    flux_cut = flux[np.where(np.logical_and(wl > lowlim, wl < highlim))]
    sig_cut = sig[np.where(np.logical_and(wl > lowlim, wl < highlim))]
    return wl_cut, flux_cut, sig_cut


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
    
    
    return d*theta_radian /1000



def sim_dist(RA, dec):
    '''Simulating distance in arcsec between two objects where one has Gaussian uncertainties'''
    '''Change coordinates when using !!!'''
    
    frb_pos = SkyCoord(f'21h33m{RA}s -54d44m{dec}s', frame='icrs')      
    hg_pos = SkyCoord('21h33m24.4648s -54d44m54.862s', frame='icrs')
    sep = frb_pos.separation(hg_pos).arcsecond
    return sep



def prob(m, R0, Rh):
    '''Probability for host galaxy being unrelated'''
    '''Remember to check effective radius !!!'''
    
    r = np.sqrt(R0**2+4*Rh**2)
    sigma = 1/(3600**2*0.334*np.log(10))*10**(0.334*(m-22.963)+4.320)
    n = np.pi*r**2*sigma
    P = 1-np.exp(-n)
    return P



def radial_profile(data, theta):
    'Radial profile of galaxy'
    
    y, x = np.indices((data.shape))
    center = [len(x)/2, len(y)/2]
    
    x0, y0 = len(x)/2, len(y)/2 
    theta *= np.pi/180
    
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
#             Multiple FRBs
# ======================================


def tau_DM(DM):
    '''Tau(DM) relation'''
    return 2.98e-7 * DM**(1.4) * (1+3.55e-5*DM**(3.1))


def DM_func(z):
    '''Integral, DMcosmic as a function of redshift'''
    
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



def pdf_for_optimization(z, DM, C0):
    '''pdf for DM(z) used to optimize C0'''
    
    alpha = 3
    beta = 3
    F = 0.32  # mass ratio of the ISM to stars
    DM_sigma = F*z**(-1/2)

    delta = DM/DM_z(z)
    prob_DM_delta = delta**-beta * np.exp(-(delta**-alpha-C0)**2/(2*alpha**2*DM_sigma**2))

    return prob_DM_delta


def pdf_DM_z(z, DM):
    '''pdf for DM(z) used for creating colormap'''
    
    alpha = 3
    beta = 3
    F = 0.32
    DM_sigma = F*z**(-1/2)
    DMz = DM_z(z)
    delta = DM/DMz
    
    def optimize_C(x):
        max_value = delta[np.where(pdf_for_optimization(z, DM, x)==max(pdf_for_optimization(z, DM, x)))]
        return abs(max_value-1)
    
    
    if z<0.15:
        start_guess = -20
    else: 
        start_guess = -5
    
    C0 = optimize.minimize(optimize_C, [start_guess], method='Nelder-Mead').x[0]
    
    # Normalizing with respect to DM_delta
    A = 1/(DMz*gamma(2/3)*DM_sigma**(2/3)*np.exp(-C0**2/(18*DM_sigma**2))*hyperu(1/3,1/2,C0**2/(18*DM_sigma**2))*6**(-1/3))
    prob_DM_delta = A*delta**-beta * np.exp(-(delta**-alpha-C0)**2/(2*alpha**2*DM_sigma**2)) 
    
    prob_DM = prob_DM_delta * DMz 
    
    return prob_DM_delta*100, prob_DM, delta



def func_Omegab(z, Omegab):
    '''Function used for fitting Omegab to DM(z)'''
    
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

def fit_Omegab(z, Omegab):
    '''Actual fit function'''
    return func_Omegab(z, Omegab)-func_Omegab(0, Omegab)

def func_H0(z, H0):
    '''Function used for fitting H0 to DM(z)'''
    
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

def fit_H0(z, H0):
    '''Actual fit function'''
    return func_H0(z, H0)-func_H0(0, H0)



def chisquarefit(x, y, ysigma, fitfunction, startparameters, ax, plot=False, xlabel='x', ylabel='y', funclabel='Chi2 fit model', label='Data', d_xy=[0.05, 0.30]):
    '''Chi square fit for (X,Y)-data''' 
    
    'Chi-square fit'
    chi2fit = Chi2Regression(fitfunction, x, y, ysigma)
    minuit_chi2 = Minuit(chi2fit, *startparameters)
    minuit_chi2.errordef = 1.0     
    minuit_chi2.migrad()
    
    if (not minuit_chi2.fmin.is_valid):
        print("  WARNING: The chi-square fit DID NOT converge!!! ")
    
    'Parameters and uncertainties'
    par = minuit_chi2.values[:]   
    par_err = minuit_chi2.errors[:]
    par_name = minuit_chi2.parameters[:]
    
    'Chi-square value, number of degress of freedom and probability'
    chi2_value = minuit_chi2.fval 
    Ndof_value = len(x)-len(par)
    chi2_prob = stats.chi2.sf(chi2_value, Ndof_value)
    
    'Plotting'
    if plot==True:
        x_axis = np.linspace(min(x), max(x), 1000000)
        
        d = {'Ndata':    len(x),
             'Chi2':     chi2_value,
             'Ndof':     Ndof_value,
             'Prob':     chi2_prob,
            }
        
        for i in range(len(par)):
            d[f'{par_name[i]}'] = [par[i], par_err[i]]
            
        ax.plot(x, y, 'o', zorder=1)
        ax.plot(x_axis, fitfunction(x_axis, *minuit_chi2.values[:]), label=funclabel,zorder=3) 
        ax.errorbar(x, y, ysigma, fmt='ro', ecolor='k', label=label, elinewidth=2, capsize=2, capthick=1,zorder=2)
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        text = nice_string_output(d, extra_spacing=2, decimals=3)
        add_text_to_ax(*d_xy, text, ax, fontsize=16)
        ax.legend(fontsize=14)
    
    return chi2_value, Ndof_value, chi2_prob, par, par_err




def _pdf_(delta, z_final, C0):
    '''Function 1 for llh-fit: pdf'''
    alpha, beta, F = 3, 3, 0.32
    DM_sigma = F*z_final**(-1/2)
    pdf = delta**-beta * np.exp(-(delta**-alpha-C0)**2/(2*alpha**2*DM_sigma**2)) 
    return -pdf
    
def get_pdf_mode(z_final, C0):
    '''Function 2 for llh-fit: pdf'''
    return scipy.optimize.minimize(lambda delta: _pdf_(delta, z_final, C0), [1]).x
    
def guess_C0(z_final):
    '''Function 3 for llh-fit: pdf'''
    return scipy.optimize.root(lambda C0: get_pdf_mode(z_final, C0)-1, [-5]).x

def likelihood_H0(a):
    '''Actual likelihood function for H0'''
    
    alpha, beta, F = 3, 3, 0.32
    DM_sigma = F*zs**(-1/2)
    exp_DM = DM_z_fit_H0(zs, a)  
    delta_DM = DM_cosmic_final/exp_DM    

    A = 1/(exp_DM*gamma(2/3)*DM_sigma**(2/3)*np.exp(-C0**2/(18*DM_sigma**2))*hyperu(1/3,1/2,C0**2/(18*DM_sigma**2))*6**(-1/3))
    
    log_pdf = np.log(A)-beta*np.log(delta_DM)-(delta_DM**-alpha-C0)**2/(2*alpha**2*DM_sigma**2)
    
    logl = -np.sum(log_pdf)
    
    return logl


def likelihood_Omegab(a):
    '''Actual likelihood function for Omegab'''
    
    alpha, beta, F = 3, 3, 0.32
    DM_sigma = F*zs**(-1/2)
    exp_DM = DM_z_fit_Omegab(zs, a)  
    delta_DM = DM_cosmic_final/exp_DM    

    A = 1/(exp_DM*gamma(2/3)*DM_sigma**(2/3)*np.exp(-C0**2/(18*DM_sigma**2))*hyperu(1/3,1/2,C0**2/(18*DM_sigma**2))*6**(-1/3))
    
    log_pdf = np.log(A)-beta*np.log(delta_DM)-(delta_DM**-alpha-C0)**2/(2*alpha**2*DM_sigma**2)
    
    logl = -np.sum(log_pdf)
    
    return logl


def lin_func(x,a,b):
    return a*x+b

def prop_func(a,x):
    return a*x

def ztest(mu1,mu2,sigma1,sigma2,side):
    'z/t-test onesided or twosided'
    
    dmu = np.abs(mu1-mu2) 
    dsigma = np.sqrt(sigma1**2+sigma2**2)
    nsigma = dmu/dsigma
    
    p = norm.cdf(0, dmu, dsigma)*side
    return nsigma, p 