#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 11:04:19 2023

This file contains all implementations of analytical results of our manuscript:
  * an implementation of the linear stability analysis used to determine a network's Hopf bifurcation (see also Brunel, Hakim 99, Lindner et al. 2001) (linear_stability_analysis)
  * the Gaussian-drift approximation for constant drive (gaussian_drift_approx_stat)
  * the analysis of IFA for piecewise constant drive (analysis_IFA_constant_drive_analytical)
  * the Gaussian-drift approximation for linear drive (gaussian_drift_approx_transient)
  * the numerical search for the trajectories shown in Figs 5-7 within a large look-up table of the analytical results of the Gaussian-drift approximation (find_trajectories_linear_drive)
  * the numerical integration of the delay-differential equation (DDE) (integrate_dde_numerically)

@author: natalie
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import mpmath as mp
import pandas as pd
import scipy
from tools import despine, f_all_scalar, get_gauss, gridline, pm_pi


my_rcParams = json.load(open('settings/matplotlib_my_rcParams.txt'))
matplotlib.rcParams.update(my_rcParams)

my_colors = json.load(open('settings/my_colors.txt'))

cm=1/2.54
width_a4 = 21*cm
width_a4_wmargin = width_a4 - 2*2*cm
panel_labelsize = 9

pi = np.pi
inf = np.inf
nan = np.nan

# labels for figures
str_Iext = '$I_\mathrm{E}$'
str_Iextmin = '$I_\mathrm{E}^\mathrm{min}$'
str_Iextfull = '$I_\mathrm{E}^\mathrm{full}$'
str_Iextcrit = '$I_\mathrm{E}^\mathrm{crit}$'

#%% Linear stability analysis spiking network

def get_fourier_transform_synkernel(freqs, IPSPshape, tl, td=None, tr=None):
  ''' absolute value and complex argument of Fourier transform of syn kernel epsilon
  INPUT:
    IPSPshape: str. Shape of postsyn potential (delta pulse, single exponential decay, or double-exponential form)
    tl: [ms] syn delay
    td: [ms] syn decay time constant for IPSPshape=1exp 
    rtr: [ms] syn rise time constant for IPSPshape = 2exp
  OUTPUT:
    absolute value and complex argument of fourier transform of synaptic kernel'''
  w = 2*pi*freqs
  if IPSPshape == '1exp':
    return 1/np.sqrt(1+(w/1000)**2*td**2)  ,  np.arctan2(-w/1000*td,1)-w/1000*tl
  elif IPSPshape == '2exp': # brunel wang 2003
    return 1/np.sqrt((1+w**2*(tr/1000)**2)*(1+w**2*(td/1000)**2)),  - (w*tl/1000 + np.arctan(w*tr/1000) + np.arctan(w*td/1000))
  elif IPSPshape == 'delta':
    return 1, -w/1000*tl

def ParabolicCylinderD(v,z):
    '''
    parabolic cylinder fct 
    '''
    if not np.isscalar(z):
      raise ValueError('Array inputs for z have not been implemented!')
    if np.isscalar(v):
      D_mp = mp.pcfd(v,z)
      # convert to numpy
      D = np.float(D_mp.real)+ 1j*np.float(D_mp.imag)
    else:
      D = np.zeros(v.size, dtype=complex)
      for i in range(v.size):
        D[i] = ParabolicCylinderD(v[i],z)    
    return D


def getLIFmfp(mu, sigma, vr, vthr, vrest, tm):
  '''
  Mean first passage time for LIF neuron
  mu: [mV] mean drive
  sigma: [mV] here sigma = sqrt(2D) = sqrt(2)*sig_mV, where sig_mV is the SD of membrane potentials for diffusion without bdry conditions
  tm: [ms] membrane time constant
  '''
  # substract resting potential from reset and threshold
  if vrest:
    vr = vr- vrest
    vthr = vthr - vrest
  if not sigma: # constant drive
      if mu > vthr:
        T = tm*np.log((mu-vr)/(mu-vthr))
      else:
        T = np.infty
  else: # stochastic drive
      # take integral
      bdry_up = (mu-vr)/sigma
      bdry_low = (mu-vthr)/sigma
      # store the values for numerical check of the integration:
      if np.min((bdry_low, bdry_up)) > 2.5 : # use asymptotic approximation of erfc function (phi)
#        print('using asymptotic for rate integral...')
        T = tm*(np.log((mu-vr)/(mu-vthr))+(sigma**2)/4*(1/((mu-vr)**2)-1/((mu-vthr)**2)))  # mean 1st passage time
      else:
#        print('using scipy.integrate.quad...')
        def phi(x):
          ''' Sergi, Brunel Eq (9) bzw. Lindner Neusig Skript (5.11)'''
          return np.exp(x**2)*scipy.special.erfc(x)
        T = tm*np.sqrt(pi)*scipy.integrate.quad(phi,bdry_low,bdry_up)[0] # mean 1st passage time
  return T # ms

def getLIFrate(mu, sigma, vr, vthr, vrest, tm, tref):
  '''
  firing rate of LIF neuron driven by white noise
  mu: [mV] mean drive, scalar or array
  sigma: [mV] noise intensity, here sigma = sqrt(2D) = sqrt(2)*sig_mV, where sig_mV is the SD of membrane potentials for diffusion without bdry conditions
  tm, tref: [ms] Membrane time constant and absolute refractory period
  '''
  if np.isscalar(mu):
    T = getLIFmfp(mu, sigma, vr, vthr, vrest, tm) # mean first passage time
    r = 1000/(T+tref)
  else:
    r = np.zeros(mu.size)
    for i in range(mu.size):
      T = getLIFmfp(mu[i], sigma, vr, vthr, vrest, tm)
      r[i] = 1000/(T+tref)
  return r # Hz

def get_stationary_rate_lif(Iext, Ji, Vthr, Vreset, sigma, tm, tref, plot=True):
  ''' compute stationary population rate in the stable AI state for a fully connected, inhibitory LIF network with white noise input
  INPUT:
    Iext: [mV]
    Ji: [mVms] integral over IPSP, negative for inhibition!!
    Vthr, Vreset: [mV] Threshold and reset with leak potential subtracted!
    sigma: [mV] intensity of input white noise (Ie_sig*tm/C)
  OUTPUT:
    A0, I0: [Hz, mV] stationary population rate and average total input (exc-inh)
    figure
  '''
  print('calculating stationary rate...', end='')
  I_min, I_max = 0, 15
  intersection = False
  while not intersection:
    print(I_min, I_max, end='-') 
    I0 = np.arange(I_min, I_max, .01) # mV
  
    lhs = (I0-Iext)/(Ji/1000) # mV/(mV*sec) = Hz
    rhs = getLIFrate(I0, sigma, Vreset, Vthr, 0, tm, tref)
    
    if rhs[-1] > lhs[-1]: # (rhs > lhs).any():
      if rhs[0] < lhs[0]:
        intersection = True
      else:
        I_min -= 15
    else:
      I_max += 15
    
  idx = np.nanargmin(np.abs(lhs-rhs))
  A0 = rhs[idx]
  
  if plot:  
    fig,ax = plt.subplots(figsize=(width_a4_wmargin/2.5, width_a4_wmargin/3.5))
    ax.plot(I0, lhs, 'k--', label=r'$\frac{I_E-I_0}{K\tau_m}$')
    ax.plot(I0, rhs, 'k', label='f-I($I_0$)')
    ax.plot(I0[idx], A0, 'ro')
    ax.legend(fontsize=9)
    ax.set_xlim([I_min, I_max])
    ax.set_ylabel('population rate $r_0$ [spks/s]')
    ax.set_xlabel('total drive $I_0$ [mV]')
    despine(ax)
  else:
    fig, ax = [], []
  print('[done]')
  return A0, I0[idx], fig, ax # Hz, mV

def get_stationary_fpe_lif(v, r0, I0, Vt, Vr, D, tm):
  ''' compute stationary solu of FPE for white-noise-driven, recurrent LIF network (Brunel, Hakim 1999)
  v: [mV] scalar or array of voltages for which to compute p(V), must be SORTED in ascending order!
  r0: [Hz] population rate / mean firing rate in stationary state
  I0: [mV] total recurrent input in stationary state
  Vt, Vr: [mV] threshold and reset (with leak subtracted!)
  D: [mV^2] white noise intensity
  tm: [ms] membrane time constat
  '''
  if np.isscalar(v):
    integral = np.sum(np.exp((np.arange((np.max([v, Vr])-I0)/np.sqrt(2*D), (Vt-I0)/np.sqrt(2*D), 0.0001))**2))*0.0001
    p0 = np.sqrt(2/D)*tm/1000*r0*np.exp(-(v-I0)**2/(2*D))*integral
    return p0    
  else:
    p0_array = np.empty(v.size)
    for i in range(v.size):
      p0_array[i] = get_stationary_fpe_lif(v[i], r0, I0, Vt, Vr, D, tm)
    return p0_array

def get_susceptibility_lif(w, mu, D, vr, vt, tref, r=0, plot=False):
  ''' Eq (41) Lindner Schimansky-Geier 2001
  susceptibility
  w: [1/tm] input frequency 
  mu: [mV] mean input
  D: [mV**2] Gaussian white noise intensity
  vr, vt: reset and threshold with leak potential subtracted
  r: [1/tm] mean firing rate given constant input mu in units of 1 per time const
  tref: [tm] refrac period in units of membrane time constant!!
  '''
  if not r:
    r = getLIFrate(mu, np.sqrt(2*D), vr, vt, 0, 1, tref)/1000 # units of 1/tm, vrest= 0 (already normalized)
  delta = (vr**2-vt**2+2*mu*(vt-vr))/(4*D)
  numerator = ParabolicCylinderD(1j*w-1, (mu-vt)/np.sqrt(D))-np.exp(delta)*ParabolicCylinderD(1j*w-1, (mu-vr)/np.sqrt(D))
  denominator = ParabolicCylinderD(1j*w, (mu-vt)/np.sqrt(D))-np.exp(delta)*np.exp(1j*w*tref)*ParabolicCylinderD(1j*w, (mu-vr)/np.sqrt(D))
  if not np.sum(np.abs(denominator)):
    A = np.ones(denominator.size)*nan
  else:
    A = r*1j*w/np.sqrt(D)/(1j*w-1) * numerator/denominator # 1/tm/mV
  return A
  
def get_phase_ampl_cond_rhs(IPSPshape, freqs, Iext, Ji, Vthr, Vreset, C, tm, tref, tl, \
                            td = None, tr=None,
                            abs_eps_tilde= None, arg_eps_tilde=None, \
                            lif_sigma = None, plot_substeps=False, plot=True):
  '''
  evaluate right-hand-side of amplitude- and phase-condition for a range of frequencies freqs
  INPUTS:
    IPSPshape: str. Shape of inh postsyn potential 
    freqs: np.array. Range of frequencies that potentially solve ampl and phase condition.
    Iext: [mV] External drive 
    Ji: [-mVms] Negative (!) integral over postsyn potential.
    Vthr, Vreset: [mV] Threshold and reset with resting potential subtracted.
    C: [pF] Capacitance.
    tm, tref: [ms] Membrane time constant, absolute refractory period.
    td, tr: [ms] Synaptic decay and rise time constant.
    abs_eps_tilde, arg_eps_tilde: absolute value and complex argument of Fourier transform of synaptic kernel
    lif_sigma: [mV] intensity of Gaussian white noise, here lif_sigma = sqrt(2D) = sqrt(2)*sig_mV, where sig_mV is the SD of membrane potentials for diffusion without bdry conditions
  '''
  w = 2*pi*freqs
  # Fourier transform of syn kernel:
  if not len(arg_eps_tilde):
    abs_eps_tilde, arg_eps_tilde = get_fourier_transform_synkernel(freqs, IPSPshape, tl, td=td, tr=tr)
    
  # find stationary rate of recurrent network given external input Iext
  A0, I0 = get_stationary_rate_lif(Iext, Ji, Vthr, Vreset, lif_sigma, tm, tref, plot=plot_substeps)[:2]  

  # calculate susceptibility around that stationary state 
  G_tilde = get_susceptibility_lif(2*pi*freqs*tm/1000, I0, lif_sigma**2/2, Vreset, Vthr, tref/tm, r=A0*tm/1000)/(tm/1000) # unit: 1/sec/mV
  G_tilde_arg = -np.angle(G_tilde)
  
  G_tilde_abs = np.abs(G_tilde)
  # calculate phase and amplitude condition (right-hand-sides)  
  phase_cond_rhs = pm_pi(pi + G_tilde_arg + arg_eps_tilde) # translate to range -pi, pi
  ampl_cond_rhs = np.abs(Ji)/1000*G_tilde_abs*abs_eps_tilde # mVsec*1/(mVsec) = dimless
  
  # -- analysis: solve phase condition, compute error in amplitude condition
  if np.isnan(G_tilde).all():
    print('All nan susceptibility, probably large drive? Iext= {}'.format(Iext))
    w0 = nan
    err_ampl_cond = 1e100  
  else:
    idx_w0 = np.nanargmin(np.abs(phase_cond_rhs))
    w0 = w[idx_w0] # freq for which phase condition is fulfilled
    err_ampl_cond = ampl_cond_rhs[idx_w0] - 1 # error in amplitude condition at that frequency
    
    if plot:
      fig, ax = plt.subplots(2, sharex=True)
      ax[0].axhline(0)
      ax[0].axvline(w0/2/pi)
      ax[0].plot(freqs, phase_cond_rhs)
      ax[0].set_ylim([-pi,pi])
      
      ax[1].axhline(1)
      ax[1].axvline(w0/2/pi)
      ax[1].plot(freqs, ampl_cond_rhs)
  return phase_cond_rhs, ampl_cond_rhs, w0, err_ampl_cond, A0, I0

def linear_stability_analysis(IPSPshape, Vthr, Vreset, tm, tref, tl, C, N,\
                              IPSCint = None, IPSPint  = None,
                              td = None, tr=None,
                              lif_sigma=None, \
                              fmax=350, Imin = 0, Imax_nA=2, dIext_min_nA=0.1, ampl_error_max = .02, plot_substeps=False, plot=True): #0.1
  ''' Linear stability analysis
  INPUT:
    IPSPshape: str. Shape of inh postsynaptic potential (delta, 1exp, 2exp)
    Vthr, Vreset: [mV] with leak potential subtracted!
    IPSCint, IPSPint: [nAms, mVms] integral over postsyn CURRENT or VOLTAGE response to ONE inhibitory spike
    tl: [ms] synaptic delay
    td: [ms] decay time constant in case of 1exp synaptic filtering
    lif_sigma: [mV] 
    here lif_sigma = sqrt(2D) = sqrt(2)*sig_mV, where sig_mV is the SD of membrane potentials for diffusion without bdry conditions
    Imax_nA: [nA] estimate of an upper boundary for the external current (bifurcation expected BELOW that level)
    dIext_min_nA: [nA] numerical precision for bifurcation estimate in current space
    ampl_error_max: [dimless] maximal allowed numerical error in the amplitude condition
  OUTPUTS:
    Iext_solu_nA: [nA] Critical external drive (bifurcation)
    f_solu [Hz]: Network frequency in bifurcation 
    I0_solu_nA: [nA]: Total drive (external - feedback inhibition) in bifurcation 
    A0_solu: [Hz] Mean unit firing rate in bifurcation
    fig, ax: Plot of solved amplitude and phase condition if plot=True    
  '''
  print('Linear stability analysis...', end='')
  #--- translate inputs from nA to mV
  dIext_min = dIext_min_nA*tm/(C/1000) # nA*ms/nF = mV
  Imax = Imax_nA*tm/(C/1000) # mV
  
  # extract synaptic strength
  if IPSPint:
    Ji = -IPSPint*N # mVms
  elif IPSCint:
    Ji = -tm*IPSCint*N/(C/1000) # ms*nAms/nF = msmV 

  #-- initialize
  freqs= np.arange(.1, fmax, .1)
  phase_cond_rhs, ampl_cond_rhs, w0, err_ampl_cond, A0, I0 \
  = {}, {}, {}, {}, {}, {}
  abs_eps_tilde , arg_eps_tilde = get_fourier_transform_synkernel(freqs, IPSPshape, tl, td=td, tr=tr) # Fourier transform of syn kernel
  
  [left, right] = [Imin, Imax]
  dIext = (right-left)/2
  successful_initialization = False
  n_extend = 0
  while not successful_initialization:
    phase_cond_rhs[left], ampl_cond_rhs[left], w0[left], err_ampl_cond[left], A0[left], I0[left] \
    = get_phase_ampl_cond_rhs(IPSPshape, freqs, left, Ji, Vthr, Vreset, C, tm, tref, tl, abs_eps_tilde= abs_eps_tilde, arg_eps_tilde=arg_eps_tilde, \
                              td=td, tr=tr, lif_sigma = lif_sigma, plot_substeps=plot_substeps, plot=False)
    phase_cond_rhs[right], ampl_cond_rhs[right], w0[right], err_ampl_cond[right], A0[right], I0[right] \
    = get_phase_ampl_cond_rhs(IPSPshape, freqs, right, Ji, Vthr, Vreset, C, tm, tref, tl, abs_eps_tilde= abs_eps_tilde, arg_eps_tilde=arg_eps_tilde, \
                              td=td, tr=tr, lif_sigma = lif_sigma, plot_substeps=plot_substeps, plot=False)
    
    if np.sign(err_ampl_cond[left])==np.sign(err_ampl_cond[right]):
        n_extend += 1
        if np.sign(err_ampl_cond[left])>0:
          left = right
          right = right + 100
        else:
          right = left
          left = np.max([Imin, left-100]) # do not go below Imin, since A0 is not defined there!
        print('All errors have the same sign ({})! Extending initial range of Iext to {:.2f}-{:.2f}mV!'.format(err_ampl_cond, left, right))
        # cutoff criteria to avoid endless loop:
        if (left==right) or (n_extend>3): # no solution exists ?
          print('Phase-amplitude condition could not be solved, returning nan.')
          #-- plot
          fig, ax = plt.subplots(2, figsize=(width_a4_wmargin/2, width_a4_wmargin/4), sharex=True)
          all_keys = np.sort(np.array(list(w0.keys())))
          for I in all_keys:
            lw=.5
            zorder=1
            col = plt.cm.viridis(np.searchsorted(all_keys, I)/(all_keys.size-1))
            ax[0].plot(freqs, phase_cond_rhs[I], color = col, lw=lw, label=str(I/tm*C/1000), zorder=zorder)
            ax[1].plot(freqs, ampl_cond_rhs[I], color = col, lw=lw, label=str(I/tm*C/1000), zorder=zorder)
     
          ax[0].axhline(0, color='lightgray', lw=1)
          ax[0].set_ylim([-pi,pi])
          ax[0].set_yticks([-pi, 0, pi])
          ax[0].set_yticklabels(['$-\pi$', '0', '$\pi$'])
          ax[0].set_ylabel('phase [rad]')
          ax[0].set_title('no solution found!')
          
          ax[1].legend(loc='upper right', title='$I_\mathrm{ext}$ [nA]', fontsize=10)
          ax[1].axhline(1, color='lightgray', lw=1)
          ax[1].set_xlim([freqs[0], freqs[-1]])
          ax[1].set_ylabel('amplitude')
          ax[1].set_xlabel('$\omega/2\pi$ [Hz]')
#          fig.tight_layout()
          return nan, nan, nan, nan, [], []
    else:
      successful_initialization = True
      print('Initialization successful (Imin={}, Imax={}mV).'.format(left, right))
      
  #-- iterate until 2D intersection is found up to resolution dIext_min
  print('dIext_min: {}'.format(dIext_min))
  while dIext > dIext_min:
    # add points in middle
    mid = (left + right)/2
    dIext = right-mid # resolution in Iext space
    print('dIext: {}'.format(dIext))
    # compute phase and amplitude condition in that point
    phase_cond_rhs[mid], ampl_cond_rhs[mid], w0[mid], err_ampl_cond[mid], A0[mid], I0[mid]\
    = get_phase_ampl_cond_rhs(IPSPshape, freqs, mid, Ji, Vthr, Vreset, C, tm, tref, tl, abs_eps_tilde= abs_eps_tilde, arg_eps_tilde=arg_eps_tilde, \
                            td=td, tr=tr, lif_sigma = lif_sigma, plot_substeps=plot_substeps, plot=plot_substeps)
      
    if np.sign(err_ampl_cond[mid]) == np.sign(err_ampl_cond[left]):
      left = mid # drop the old left point
    elif np.sign(err_ampl_cond[mid]) == np.sign(err_ampl_cond[right]): # THIS CAN BE NAN AND GO INTO ENDLESS LOOP ??????
      right = mid # drop the old right point
    if dIext <= dIext_min: # about to stop
      error = [err_ampl_cond[left], err_ampl_cond[right]]
      if np.min(np.abs(error)) > ampl_error_max: # dimless voltage
        dIext_min /= 10
        print('left: {}, right: {}, error: {}'.format(left, right, error))
        print('Decreasing current-accuracy dIext_min to {}'.format(dIext_min))
  
  Iext_solu = [left, right][np.argmin(np.abs(error))] # mV
  Iext_solu_nA = Iext_solu/tm*(C/1000)
  I0_solu_nA = I0[Iext_solu]/tm*(C/1000)
  A0_solu = A0[Iext_solu]
  f_solu = w0[Iext_solu]/2/pi
  
  #-- plot
  if plot:
    fig, ax = plt.subplots(2, figsize=(width_a4_wmargin*.5, width_a4_wmargin*.4), sharex=True)
    all_keys = np.sort(np.array(list(w0.keys())))
    for I in all_keys:
      if I==Iext_solu:
        lw=1
        linestyle='-'
        zorder=5
      else:
        lw=.5
        linestyle='--'
        zorder=4
        col = plt.cm.viridis(np.searchsorted(all_keys, I)/(all_keys.size-1))
      ax[0].plot(freqs, phase_cond_rhs[I], color = col, lw=lw, linestyle=linestyle, label='{:.2f}'.format(I/Vthr), zorder=zorder)
      ax[1].plot(freqs, ampl_cond_rhs[I], color = col, lw=lw, linestyle=linestyle, label='{:.2f}'.format(I/Vthr), zorder=zorder)
    
    gridline(0, ax[0], zorder=3, linestyle='-', color='k')
    ax[0].axvline(w0[Iext_solu]/2/pi, lw=.5, color='k')
    ax[0].set_ylim([-pi,pi])
    ax[0].set_yticks([-pi, 0, pi])
    ax[0].set_yticklabels(['$-\pi$', '0', '$\pi$'])
    ax[0].set_ylabel('phase')
#    ax[0].set_title('Iext={:.2f}nA, I0={:.2f}nA, f={:.0f}Hz, A0={:.2f}Hz'.format(Iext_solu_nA, I0_solu_nA, f_solu, A0_solu))
    
    gridline(1, ax[1], zorder=3, linestyle='-', color='k')
    ax[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='$I_\mathrm{E}$', ncol=1, borderpad=0, borderaxespad=0)
    ax[1].axvline(w0[Iext_solu]/2/pi, lw=.5, color='k')
    ax[1].set_xlim([freqs[0], freqs[-1]])
    ax[1].set_ylabel('amplitude')
    ax[1].set_xlabel('frequency $\omega/2\pi$ [Hz]')
    
    despine(ax[0])
    despine(ax[1])
  else:
    fig, ax = [], []
  
  print('[done]')
  return Iext_solu_nA, f_solu, I0_solu_nA, A0_solu, fig, ax

#%% Gaussian-drift approx. analytical: const. drive
def gaussian_drift_approx_stat(Iext_val, D, Delta, K, tm, Vr=0, Vt=1, plot=False, reset=False, \
                               show_numerical=False, check_mumin = False, return_traces=False, mu_min_start=np.nan): 
  '''
  Gaussian-drift approximation for constant drive

  Parameters
  ----------
  Iext_val : np.array. Range of external drives for which to compute network dynamics.
  D : float [dimless volt] Gaussian white noise intensity.
  Delta : float [ms]. Synaptic delay.
  K : float [dimless volt]. Synaptic coupling strength (inh).
  tm : float [ms] Membrane time constant.
  Vr : float [dimless volt], optional. Reset potential. The default is 0.
  Vt : float [dimless volt], optional. Spike threshold. The default is 1.
  plot : bool, optional. Whether to plot result. The default is False.
  reset : bool, optional. Whether to include phenomenological reset. The default is False.
  show_numerical : bool, optional. Whether to compare the analytical results to a numerical integration of the DDE. The default is False.
  check_mumin : bool, optional. Whether to set results to nan for drives Iext that lie outside of the range of applicability. (Check whether mumin + 3sqrt(D) < Vt). 
    The default is False.
  return_traces : bool, optional. Whether to add an analytical estimate of the evolution of mu and r over time. The default is False.
  mu_min_start : float, optional. Initial value of cycle (not relevant for asymptotic dynamics, only for response to piecewise constant drive). The default is np.nan.

  Returns
  -------
  mu_min, mu_max: local min and max of mean membrane potential.
  f_net, f_unit, sat: network frequency, mean unit firing rate, saturation.
  zeta: between 0,1. Measure for spike synchrony. Not used in manuscript.
  t_on: Estimate for beginning of population spike as time when mu(t) + 3sqrt(D) = VT. Not used in manuscript.
  t_off: End of population spike.
  mu_reset: Reset potential if reset=True, else: nan
  fig, ax: plot of results if plot=True
  t, mu_t, rate_t, I_t: time-evolution of mu, rate, and total current I over one cycle (only if return_traces=True)

  '''

  mu_max = get_mu_max(Iext_val, D, Delta, K, tm, Vt=Vt)
  mu_min, mu_reset = get_mu_min(Iext_val, mu_max, D, Delta, K, tm, Vt=Vt, Vr=Vr, reset=reset)
  if np.isnan(mu_min_start):
    mu_min_start = mu_min # typically, for constant drive / asymptotic behavior, we assume that mumin at the end of the cycle equals the starting value at the beginning.
  
  t_on = get_popspk_start(Iext_val, mu_min_start, D, tm, Vt=Vt)
  t_off = get_popspk_end(Iext_val, mu_min_start, D, Delta, K, tm, Vt=Vt) #(Iext_val, mu_max, mu_min, tm)
  f_net = get_fnet(t_off, Delta)
  sat = get_saturation(mu_max, D, Vt=Vt)
  f_unit = get_funit(f_net, sat)
  zeta = get_popsynch(t_on, t_off, sat, f_net)
  
  if check_mumin:
    mask = mu_min + 3*np.sqrt(D) >= Vt
    mu_min[mask]= nan
    mu_max[mask]= nan
    f_net[mask]= nan
    f_unit[mask]= nan
    sat[mask]= nan
    zeta[mask]= nan
    t_on[mask]= nan
    t_off[mask]= nan
    mu_reset[mask]= nan
  
  if plot:
    fig, ax = plot_gaussian_drift_approx(Iext_val, f_net, f_unit, mu_min, mu_max, zeta, t_on, t_off, mu_reset, Delta, \
                                      Vt=Vt, linestyle='--', marker='', label='(analytical)', reset=reset)
    if show_numerical:
      mu_min_num, mu_max_num, f_net_num, f_unit_num, zeta_num, t_on_num, t_off_num, mu_reset_num\
      = gaussian_drift_approx_stat_numerical(Iext_val, D, Delta, K, tm, Vt=Vt, Vr=Vr, reset=reset, plot=False)[:-3]
      fig, ax = plot_gaussian_drift_approx(Iext_val, f_net_num, f_unit_num, mu_min_num, mu_max_num, zeta_num, t_on_num, t_off_num, mu_reset_num, \
                                        Delta, Vt=Vt, linestyle=':', marker='', label='(numerical)', reset=reset, fig=fig, ax=ax)
  else:
    fig, ax = [], []
  if np.isscalar(Iext_val) and return_traces:
    t, mu_t, rate_t, I_t = gaussian_drift_approx_stat_traces(Iext_val, D, Delta, K, tm, mu_min_start, mu_max, \
                                                        mu_reset, t_off)
    return mu_min, mu_max, f_net, f_unit, sat, zeta, t_on, t_off, mu_reset, fig, ax, t, mu_t, rate_t, I_t
  else:
    return mu_min, mu_max, f_net, f_unit, sat, zeta, t_on, t_off, mu_reset, fig, ax

def gaussian_drift_approx_stat_traces(Iext, D, Delta, K, tm, mumin_start, mu_max, mu_reset, toff,  n_time = 10000, Vt=1):
  '''
  Evolution of mean membrane potential mu, rate r, and total current I over the course of one cycle, as estimated by the theory
  (including simplifying assumption A1)

  Parameters
  ----------
  Iext : float. External drives for which to compute network dynamics.
  D : float [dimless volt] Gaussian white noise intensity.
  Delta : float [ms]. Synaptic delay.
  K : float [dimless volt]. Synaptic coupling strength (inh).
  tm : float [ms] Membrane time constant.
  mumin_start : float. Initial value for mu at beginning of cycle.
  mu_max, mu_reset, toff : results of gaussian-drift approx for constant drive Iext
  n_time : float, optional. Number of discrete time steps. The default is 10000.
  Vt : float [dimless volt], optional. Spike threshold. The default is 1.

  Returns
  -------
  t : np.array. Time steps from 0 to toff+Delta (end of cycle).
  mu_t :  np.array. Trajectory of mu(t).
  rate_t : np.array. Trajectory of r(t).
  I_t : np.array. Trajectory of I(t).

  '''
  # for plotting: infer traces of mu and rate
  delta_window = np.linspace(0, Delta, n_time, endpoint=False) # make sure t_down has exactly as many time steps as window2 below
  dt = np.mean(np.diff(delta_window))
  
  t_up = np.arange(0, toff, dt) # np.linspace(0, toff, n_time)  # time for upstroke of mu
  t_down = toff + delta_window # np.linspace(toff, toff+Delta, n_time) # time for downstroke of mu
  mu_up =  get_mu_up(t_up - t_up[0], mumin_start, Iext, tm) # trajectory of mu on upstroke 
  mu_down = get_mu_down(t_down-t_down[0], mu_reset, Iext, D, Delta, K, mu_max, tm) # trajectory of mu on downstroke
  
  # for rate calculation, take into account two layers of feedback:
  # first  order fb1 from time window [toff-Delta, toff] and 
  # second order fb2 from time window [toff-2Delta, toff-Delta] :
  window1 = np.where((t_up >= toff - 2*Delta) & (t_up < toff - Delta))[0] # np.where(t_up <= toff - Delta)[0] # second order feedback (2Delta before max, influencing mu trajectory)
  window2 =  np.where((t_up >= toff - Delta) & (t_up < toff))[0] # first order feedback (directly before max)
    
  I_up, rate_up = np.zeros(t_up.size), np.zeros(t_up.size) # total current resulting from upstroke, arriving at times [Delta, toff+Delta]
  I_up[:window1[-1]+1] = Iext # first, no inh current
  # rate during time window1 as we integrate it for the feedback in Step 2
  rate_up[window1] = (I_up[window1] - mu_up[window1])/(tm/1000)*get_gauss(Vt, mu_up[window1], np.sqrt(D), broadcast=False) # Hz
  # due to this (small) rate in window1 the current in window2 is slightly smaller than Iext (first inh feedback arrives):
  I_up[window2] = Iext - tm/1000*K*np.pad(rate_up[window1], (len(window2)-len(window1), 0)) # zero-pad rate, if window1 is shorter than window2 (Delta)
  rate_up[window2] = (I_up[window2]- mu_up[window2])*get_gauss(Vt, mu_up[window2], np.sqrt(D), broadcast=False)/(tm/1000) # Hz
  
  # rate_up = get_gauss(Vt, mu_up, np.sqrt(D), broadcast=False) * (Iext - mu_up)/(tm/1e3) # Hz DELETE
  rate_down = np.zeros_like(t_down)
  I_down = Iext - tm/1000*K*rate_up[window2]
  
  t = np.concatenate((t_up, t_down))
  mu_t = np.concatenate((mu_up, mu_down))
  rate_t = np.concatenate((rate_up, rate_down))
  I_t = np.concatenate((I_up, I_down))
  return t, mu_t, rate_t, I_t
  

def get_mu_max(Iext, D, Delta, K, tm, Vt=1, check_masks=True):
  '''
  find mu_max (Eq. 38)

  Parameters
  ----------
  Iext, D, Delta, K, tm: float or np.array
    network parameters: drive (dimless), noise (dimless), syn. delay (ms), coupling strength (dimless), membrane time const. (ms)
  Vt: Firing threshold. The default is 1.
  check_masks : bool, optional
    Check if drive is weak enough s.th. membrane potential just settles into steadystate (no oscillations). In that case return mu_max=nan. The default is True.

  Returns
  -------
  mu_max (dimless)
  '''
  if not check_masks:
    return (1-np.exp(-Delta/tm))*Iext + np.exp(-Delta/tm)*(Vt - np.sqrt(2*D*np.log(K/np.sqrt(2*pi*D)*np.exp(Delta/tm))))
  else:
    all_scalar = f_all_scalar(Iext, D, Delta, K, tm)
    Iext, D, Delta, K, tm = np.broadcast_arrays(Iext, D, Delta, K, tm)   
    weak_drive = Iext <= Vt - np.sqrt(2*D*np.log(K/np.sqrt(2*pi*D)*np.exp(Delta/tm)))
    mu_max = np.array(np.ones_like(Iext)*np.nan)
    mask = weak_drive==False
    mu_max[mask] = get_mu_max(Iext[mask], D[mask], Delta[mask], K[mask], tm[mask], Vt=Vt, check_masks=False)
    if all_scalar:
      mu_max = np.float64(mu_max.item())
    return mu_max

def get_weak_coupling_limit(D, Delta, tm):
  ''' determine weak coupling limit below which no solution exists for mu_max.

  Parameters
  ----------
  D, Delta, tm:  network parameters.

  Returns
  -------
  Kmin sth no solution exists for K< Kmin
  '''
  return np.sqrt(2*pi*D)*np.exp(-Delta/tm)

def get_popspk_end(Iext, mumin, D, Delta, K, tm, Vt=1, check_masks=True):
  '''
  compute end of population spike toff (Eq 47)

  Parameters
  ----------
  Iext, mumin, D, Delta, K, tm
  mumin : float or np.array
    Membrane potential at the beginning of the cycle.
  Vt: Firing threshold. The default is 1.
  check_masks : bool, optional
    Check if drive or coupling are too weak enough s.th. no solution exists (no oscillations). In that case return toff=nan. The default is True.


  Returns
  -------
  toff (ms)
  '''
  if not check_masks:
    return Delta + tm*np.log((Iext-mumin)/(Iext-Vt+np.sqrt(2*D*np.log(K/np.sqrt(2*pi*D) * np.exp(Delta/tm)))))
  else:
    all_scalar = f_all_scalar(Iext, mumin, D, Delta, K, tm)
    Iext, mumin, D, Delta, K, tm = np.broadcast_arrays(Iext, mumin, D, Delta, K, tm)
    # define masks
    weak_drive = Iext <= Vt - np.sqrt(2*D*np.log(K/np.sqrt(2*pi*D)*np.exp(Delta/tm)))
    weak_coupling = K < get_weak_coupling_limit(D, Delta, tm)
#    diff_regime = mumin >  Vt - np.sqrt(2*D*np.log(K/np.sqrt(2*pi*D)*np.exp(Delta/tm)))
    # calculate end of popspk
    t_off = np.array(np.ones_like(Iext)*np.nan)
    mask = (weak_drive==False)  & (weak_coupling==False) #& (diff_regime==False)
    t_off[mask] = get_popspk_end(Iext[mask], mumin[mask], D[mask], Delta[mask], K[mask], tm[mask], Vt=Vt, check_masks=False)
    if all_scalar:
      t_off = np.float64(t_off.item())
    return t_off

  
def get_fnet(t_off, Delta, check_masks=True):
  '''
  compute network frequency (Eq 9)

  Parameters
  ----------
  t_off : float or np.array 
    pre-calculated end of population spike (ms)
  Delta : float or np.array
    syn delay (ms)
  check_masks : check_masks : bool, optional
    Check if drive or coupling are too weak enough s.th. no solution exists (no oscillations, toff=nan). In that case return fnet=nan. The default is True.

  Returns
  -------
  fnet (Hz)

  '''
  if not check_masks:
    return 1000/(t_off + Delta) # Hz
  else:
    all_scalar = f_all_scalar(t_off, Delta)
    t_off, Delta = np.broadcast_arrays(t_off, Delta)
    fnet = np.array(np.ones_like(t_off)*nan)
    mask = np.isnan(t_off)==False
    fnet[mask] = get_fnet(t_off[mask], Delta[mask], check_masks=False)
    if all_scalar:
      fnet = fnet.item()
    return fnet
  
def get_saturation(mu_max, D, Vt=1):
  '''
  compute saturation (Eq 48)

  Parameters
  ----------
  mu_max : float or np.array.
    Previously calculated mu_max.
  D : float or np.array.
    Noise intensity.
  Vt : Firing threshold. The default is 1.

  Returns
  -------
  Saturation s (between 0 and 1)
  '''
  return .5*(1-scipy.special.erf((Vt-mu_max)/(np.sqrt(2*D))))

def get_mu_reset(mu_max, D, Vt=1, Vr=0):
  '''
  Compute population reset (Eq 49)

  Parameters
  ----------
  mu_max : float or np.array.
    Previously calculated mu_max.
  D : float or np.array.
    Noise intensity.
  Vt : Firing threshold. The default is 1.
  Vr : float or np.array.
    Reset potential. Default is 0 (=rest).

  Returns
  -------
  mu_reset (dimless)
  '''
  sat = get_saturation(mu_max, D, Vt=Vt)
  return mu_max - (Vt-Vr)*sat


def get_funit(fnet, saturation):
  '''
  compute mean unit firing rate (Eq 10)

  Parameters
  ----------
  fnet : float or np.array.
    Previously calculated network frequency.
  saturation : float or np.array.
    Previously calculated saturation

  Returns
  -------
  funit (Hz)
  '''
  return saturation*fnet


def get_mu_up(t, mu0, Iext, tm):
  mu_t = Iext - (Iext-mu0)*np.exp(-t/tm) # (A1)
  return mu_t

def get_mu_down(t, mu0, Iext, D, Delta, K, mu_max, tm, Vt=1):
  ''' 
  Compute trajectory of mean membrane potential mu on its downstroke, after time toff (Eq 43 / 50)

  Parameters
  ----------
  t : float of np.array.
    Time since toff (ms): either one particular time point or an array of time points if an entire trajectory shall be computed/plotted.
  mu0 : float
    Initial condition at the end of the population spike: mu(toff) = mu0 (typically mu_max or mureset)
  Iext : float
    (constant) external drive
  D, Delta, K, tm: float
    network parameters.
  mu_max : float
    previously calculated mu_max for given parameters
  Vt : Firing threshold. The default is 1.

  Returns
  -------
  mu_t : float of np.array. Trajectory mu(t) for the requested time point(s) t

  '''
  phi = lambda tt: (Vt - mu_max + (Iext-mu_max)*(Delta-tt)/tm) / np.sqrt(2*D) # Eq 47
  psi = lambda tt: ( -(Iext-mu_max)*(np.exp(2*Delta/tm)+1)*(tm+Delta-tt) + tm*(Iext-Vt)*(np.exp(Delta/tm)+1)) / np.sqrt(2*D*(np.exp(2*Delta/tm)+1)) / tm # Eq 48
  
  c = (Vt-Iext)**2*(1-np.exp(Delta/tm))**2/(np.exp(2*Delta/tm)+1) # Eq 49
  # consider separately the 3 contributions to the flow:
  initial_condition = mu0*np.exp(-t/tm) # mV
  exc_drive = Iext*(1-np.exp(-t/tm)) # mV
  inh_feedback = 1/2*K*np.exp((Delta-t)/tm)*(scipy.special.erf(phi(0)) - scipy.special.erf(phi(t))) \
                - 1/2*K**2/np.sqrt(2*pi*D)*np.exp(-c/2/D)*np.exp((2*Delta-t)/tm)/np.sqrt(np.exp(2*Delta/tm) + 1)*(scipy.special.erf(psi(t)) - scipy.special.erf(psi(0)))
  mu_t = initial_condition + exc_drive - inh_feedback
  return mu_t

def get_mu_min(Iext, mu_max, D, Delta, K, tm, Vt=1, Vr=0, reset=False, check_masks=True):
  '''
  Copmute mumin reached at time toff+Delta (Eq 43 / 50).

  Parameters
  ----------
  Iext : float or np.array
    (constant) external drive
  mu_max : float or np.array
    previously calculated mu_max for given parameters
  D, Delta, K, tm: float or np.array
    network parameters 
  Vt : Firing threshold. The default is 1.
  Vr : Reset potential. The default is 0.
  reset : bool, optional
    Whether or not the population-level reset shall be applied (mu_max --> mu_reset). The default is False.
  check_masks : bool, optional
    Whether to return nan where theory is not applicable. The default is True.

  Returns
  -------
  mu_min: float or np.array
    mu_min at the end of the cycle
  mu_reset : float or np.array
    mu_reset, only relevant if reset=True, otherwise equal to mu_max

  '''
  if not check_masks:
    if reset:
      mu_reset = get_mu_reset(mu_max, D, Vt=Vt, Vr=Vr)
    else:
      mu_reset = mu_max
    mu_min = get_mu_down(Delta, mu_reset, Iext, D, Delta, K, mu_max, tm, Vt=Vt)
    return mu_min, mu_reset
  else:
    all_scalar = f_all_scalar(Iext, mu_max, D, Delta, K, tm, Vr)
    Iext, mu_max, D, Delta, K, tm, Vr = np.broadcast_arrays(Iext, mu_max, D, Delta, K, tm, Vr)
    # define masks
    weak_drive = Iext <= Vt - np.sqrt(2*D*np.log(K/np.sqrt(2*pi*D)*np.exp(Delta/tm)))
    mask = weak_drive==False
    mu_min, mu_reset = np.array(np.ones_like(Iext)*np.nan), np.array(np.ones_like(Iext)*np.nan)
    mu_min[mask], mu_reset[mask] \
      = get_mu_min(Iext[mask], mu_max[mask], D[mask], Delta[mask], K[mask], tm[mask], Vt=Vt, Vr=Vr[mask], reset=reset, check_masks=False)
    if all_scalar:
      mu_min, mu_reset = np.float64(mu_min.item()), np.float64(mu_reset.item())
    return mu_min, mu_reset  


def get_Iext_lower_bound(D, Delta, K, tm, Vr, reset=True, Iext_min=0, Iext_max=3.5, dI=.01, Vt=1):
  '''
  determine the lowest drive Iext for which condition (b) is fulfilled (Eq (52)), ie the "full" Gauss is subthreshold when mu=mu_min
  '''
  Iext_val = np.arange(Iext_min, Iext_max, dI) # test a range of input levels 
  
  mu_min, mu_max, f_net \
  = gaussian_drift_approx_stat(Iext_val, D, Delta, K, tm, Vr=Vr, reset=reset)[:3] # do the Gaussian-drift approx for this range
  
  # find the lowest level of drive, for which condition (b) is fulfilled (Eq (55)), ie the "full" Gauss is subthreshold when mu=mu_min
  if (mu_min + 3*np.sqrt(D) <= Vt).any():
    ix = np.where(mu_min + 3*np.sqrt(D) <= Vt)[0][0]
  else:
    raise ValueError('mumin + 3sqrt(D) is never subthreshold for Iext in [{},{}]. Increase Iext_max!'.format(Iext_min, Iext_max))
  return Iext_val[ix], mu_min[ix]



def get_pt_fullsynch(D, Delta, K, tm, Vt=1, gauss_extent=3, check_masks=True):
  ''' we approximate that all units are contained within mu +/- sqrt(D)*gauss_extent
  '''
  if not check_masks:
    return Vt + np.sqrt(D)*(gauss_extent+np.exp(-Delta/tm)*np.sqrt(2*np.log(K/np.sqrt(2*pi*D)*np.exp(Delta/tm)))) / (1-np.exp(-Delta/tm))
  else:
    all_scalar = f_all_scalar(D, Delta, K, tm)
    D, Delta, K, tm = np.broadcast_arrays(D, Delta, K, tm)
    weak_coupling = K < np.sqrt(2*pi*D)*np.exp(-Delta/tm)
    mask = weak_coupling==False
    Ifull = np.array(np.ones_like(D)*nan)
    Ifull[mask] = get_pt_fullsynch(D[mask], Delta[mask], K[mask], tm[mask], Vt=Vt, gauss_extent=gauss_extent, check_masks=False)
    if all_scalar:
      Ifull = Ifull.item()
    return Ifull

def get_popspk_start(Iext, mumin, D, tm, Vt=1, gauss_extent=3):
  '''
  estimate of beginning of population spike (t_on, not used in manuscript)

  Parameters
  ----------
  mumin : TYPE
    mu at beginning o cycle
  gauss_extent : int, optional
    How many standard deviations to use when defining the "boundaries" of the Gaussian distribtion. The default is 3, i.e. mu +/- 3 SD

  Returns
  -------
  t_on (ms) Approximate onset of population spike.
  '''
  return np.maximum(0, tm*np.log((Iext-mumin)/(Iext+gauss_extent*np.sqrt(D) -Vt)))

def get_popsynch(t_on, t_off, saturation, fnet):
  '''
  An estimate of population synchrony (not used in manuscript).

  Parameters
  ----------
  t_on, t_off : Beginning and end of population spike.

  Returns
  -------
  zeta_bounded : a synchrony index
  '''
  # zeta = saturation/((t_off-t_on)*fnet/1000) # dimless
  zeta_bounded = saturation*(1-(t_off-t_on)*fnet/1000)
  return zeta_bounded




def plot_gaussian_drift_approx(Iext, f_net, f_unit, mu_min, mu_max, zeta_bounded, t_on, t_off,  mu_reset, Delta, \
                            Ihopf=nan, Vt=1, El=0, linestyle='-', marker='', label='', unit='', fig=[], ax=[], reset=False, reduced=False):
  if reduced:
    nrows = 2
    figsize=(8,5)
  else:
    nrows=5
    figsize=(8,12)
  if not fig:
    fig, ax = plt.subplots(figsize=figsize, nrows=nrows, sharex=True)
    
  if (not unit) and (Vt != 1): # make everything dimensionless
    scaling = 1/(Vt-El)
    Iext, Ihopf, mu_min, mu_max \
    = Iext*scaling, Ihopf*scaling, (mu_min-El)*scaling, (mu_max-El)*scaling
    Vt, El = 1, 0
    # print(Iext, f_net)
  if not unit:
    unit = ''
  else:
    unit = ' ['+unit + ']'
  
  # mask for oscillatory regime
  if np.isnan(Ihopf): #label=='(analytical)': # show all (work in progress)
    osc = Iext >= 0
    # hopflabel = '(proxy)'
  else:
    osc = Iext >= Ihopf - 1e-3 
  if label=='(spiking net)':
    hopflabel = '(analytical)'
  else:
    hopflabel = '(proxy)'
  if (not reduced) and (not np.isnan(Ihopf)):  
    for i in range(nrows):
      if not i:
        ax[i].axvline(Ihopf, color='gray', lw=1, label='Hopf: {:.2f} '.format(Ihopf)+ hopflabel)
      else:
        ax[i].axvline(Ihopf, color='gray', lw=1)
        
  ax[0].axhspan(140, 220, facecolor='lightgray')   
  # ax[0].plot(Iext, f_net, 'b', marker=marker, linestyle='')
  ax[0].plot(Iext[osc], f_net[osc], 'b', linestyle=linestyle, marker=marker, label='network '+label)
  ax[0].plot(Iext, f_unit, 'k', marker=marker, linestyle=linestyle, label='unit '+label)
  ax[0].legend(handlelength=4, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.) 
  ax[0].set_ylabel('frequency [Hz]')
  ax[0].set_ylim([0, 300]) #np.nanmax([300, 20+np.nanmax(f_net[-1])])])
  
  ax[1].plot(Iext, mu_min, color=my_colors['min'], marker=marker, linestyle='')
  ax[1].plot(Iext[osc], mu_min[osc], linestyle=linestyle, marker=marker,color=my_colors['min'], label='$\mu_\mathrm{min}$ '+label)
  # if not label == '(spiking net)':
  if (np.isnan(mu_max)==False).any(): # do not plot the all-nan mu_max values for simulations
    ax[1].plot(Iext, mu_max, color=my_colors['max'], marker=marker, linestyle='')
    ax[1].plot(Iext[osc], mu_max[osc], linestyle=linestyle, marker=marker,color=my_colors['max'], label='$\mu_\mathrm{max}$ '+label)
  if reset and (np.isnan(mu_reset)==False).any():
    ax[1].plot(Iext[osc], mu_reset[osc], color='y', marker=marker, linestyle=linestyle, label='$\mu_\mathrm{reset}$ '+label)
  ax[1].axhline(Vt, color='gray', lw=1)
  ax[1].axhline(El, color='gray', lw=1)
  ax[1].set_yticks([El-4*Vt, El-2*Vt, El, Vt])
  ax[1].set_yticklabels(['{}'.format(El-4*Vt), '{}'.format(El-2*Vt), '$E_L={}$'.format(El), '$V_T={}$'.format(Vt)]) #'-4', '-2', '$E_L=0$', '$V_T=1$'])
  ax[1].set_ylabel('$\mu_\mathrm{min/max}$'+unit)
  ax[1].legend(handlelength=4, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.) 
  
  if not reduced:
    ax[2].plot(Iext, zeta_bounded, 'k', marker=marker, linestyle='')
    ax[2].plot(Iext[osc], zeta_bounded[osc], 'k', marker=marker, linestyle=linestyle, label=label)
    ax[2].legend(handlelength=4, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.) 
    ax[2].set_ylabel('synch $\zeta$')
    ax[2].set_ylim([0,1])
  
    ax[3].axhline(Delta, color='lightgray', lw=1)
    ax[3].axhline(0, color='gray', lw=1)
    ax[3].plot(Iext, t_on, 'y', marker=marker, linestyle='')
    ax[3].plot(Iext, t_off, 'g', marker=marker, linestyle='')
    ax[3].plot(Iext[osc], t_on[osc], 'y', linestyle=linestyle, marker=marker, label='start $t_\mathrm{on}$ '+label)
    ax[3].plot(Iext[osc], t_off[osc], 'g', linestyle=linestyle, marker=marker, label='end $t_\mathrm{off}$ '+label)
    ax[3].legend(handlelength=4, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.) 
    ax[3].set_ylabel('$\hat{t}$ [ms]')
    
    ax[4].axhline(Delta, color='lightgray', lw=1)
    ax[4].plot(Iext, t_off-t_on, 'k', marker=marker, linestyle='')
    ax[4].plot(Iext[osc], (t_off-t_on)[osc], 'k', marker=marker, linestyle=linestyle, label=label)
    ax[4].set_ylabel('$t_\mathrm{off}-t_\mathrm{on}$ [ms]')
    ax[4].set_yticks([0, 1, Delta, 3])
    ax[4].set_yticklabels(['0', '1', '$\Delta$={:.1f}'.format(Delta), '3'])
    ax[4].legend(handlelength=4, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.) 
  
  ax[-1].set_xlim([0, 1.05*np.max(Iext)])
  ax[-1].set_xlabel(str_Iext+unit)
  fig.tight_layout(h_pad=0.001)
  
  return fig, ax

#%% Gaussian-drift approx. analytical: pw constant drive

### constant drive, transient dynamics via initial condition ################################################################################################################################

def integrate_1cycle_wconstantdrive_analytically(mumin_start, Iext, D, Delta, K, tm, Vr, n_time = 10000,  Vt=1, reset=True): # iterate_constant_drive_analytical_1_cycle
  
  mumin_end, mu_max, fnet, funit, sat, zeta, ton, toff, mu_reset, _, _,  t, mu_t, rate_t, I_t \
  = gaussian_drift_approx_stat(Iext, D, Delta, K, tm, Vr=Vr, reset=reset, return_traces=True, mu_min_start=mumin_start)
  
  return toff, mu_max, mu_reset, mumin_end, fnet, funit, sat, ton, zeta, t, mu_t, rate_t, I_t

def iterate_constant_drive_analytical(mu_min_0, Iext, D, Delta, K, tm, Vr, Vt=1, reset=True, plot=True, \
                                      fmax=100, fmin=-100, fig=[], ax=[]): 
  '''
  Analytical approx of oscillation dynamics for piecewise constant drive of steps Iext. 
  INPUT:
    mumin0: initial mu_min for first cycle
    Iext: array of (constant) input drive for each cycle
  OUPUT:
    df: pandas dataframe with results (mumax, mureset, toff etc) in one row per cycle
    for compatibility with iteration for linear drive, df has spurious entries like slope m=0 or Iext at different points during the cycle
    traces: dict. Time-dependent traces of mu, rate, I across cycles.
  '''
  
  df = pd.DataFrame(columns=['m', 'mu_min_start', 'mu_max', 'mu_reset', 'mu_min_end', 'Iext_start', 'Iext_toff', \
                             'Iext_end', 't_off', 'f_inst', 'time_start', 'time_toff', 'time_end'], dtype=float)   
  # initialize
  df['Iext_start'] = df['Iext_toff'] = df['Iext_end'] = Iext
  df.loc[0,'mu_min_start'] = mu_min_0
  
  # add cyclostationary info
  df['mu_min_stat'], df['mu_max_stat'], df['f_net_stat'], df['f_unit_stat'], _, df['zeta_stat'], df['t_on_stat'], df['t_off_stat'], df['mu_reset_stat'] \
  = gaussian_drift_approx_stat(df.Iext_toff, D, Delta, K, tm, Vr=Vr, Vt=Vt, reset=reset)[:-2]
  
  traces = {}
  for c in range(len(df)):
    traces[c] = {}
    df.loc[c,'t_off'], df.loc[c,'mu_max'], df.loc[c,'mu_reset'], df.loc[c,'mu_min_end'], df.loc[c,'f_inst'], df.loc[c,'f_unit'], df.loc[c,'sat'], df.loc[c,'t_on'], df.loc[c,'zeta'], \
    traces[c]['t'], traces[c]['mu'], traces[c]['rate'], traces[c]['I'] \
    = integrate_1cycle_wconstantdrive_analytically(df.loc[c,'mu_min_start'], df.loc[c,'Iext_toff'], D, Delta, K, tm, Vr, reset=reset)    
    
    if c+1<len(df):
      df.loc[c+1,'mu_min_start'] = df.loc[c,'mu_min_end']

  # add additional info
  df.m = 0 # no slope, just for compatibility with linear drive integrations
  # add absolute time stamps (for consecutive cycles starting from time 0):
  df.time_end = np.cumsum(df.t_off+Delta) # time stamp for end of each cycle
  df.time_toff = df.time_end - Delta # time stamp for end of population spike in each cycle
  df.time_start = df.time_end - df.t_off - Delta # time stamp for start of each cycle
  df.loc[0,'time_start'] = 0 # avoid numerical inaccuracies
  
  # shift all time stamps in the resp. traces by the starting time of the cycle, so they can be plotted continuously 
  for c in range(len(df)):
    traces[c]['t'] += df.loc[c, 'time_start']

  if plot:
    fig, ax = plot_gaussian_drift_approx_transient(df, D, Delta, K, tm, Vr, reset=reset, fmax=fmax, fmin=fmin, fig=fig, ax=ax, traces=traces)
   
  return df, traces, fig, ax

def plot_convergence_to_asymptotic_dynamics(mu_min_high, mu_min_low, Iext, D, Delta, K, tm, Vr, T= 30, ax=[], reset=True):
  '''
  ax: 2x2 axis for plotting. Left: start with mu_min_high > mumin_stat, right: start with mu_min_low < mumin_stat. Top: rate, bottom: voltage
  T: [ms] total time to show
  '''
  # ---- ANALYTICAL DERIVATION OF DYNAMICS ------------------------------------------------------------------------------------------
  
  mu_min_stat = gaussian_drift_approx_stat(Iext, D, Delta, K, tm, Vr=Vr, reset=reset)[0] # asymptotic mumin
  
  # derive and store dynamics in one cycle starting with: mu_min_stat, mu_min_high, or mu_min_low resp.:
  asymp, high_init, low_init = {}, {}, {}
  
  # (asymptotic) dynamics for asymptotic mumin as initial value, analytically derived
  asymp['t_off'], asymp['mu_max'], asymp['mu_reset'], asymp['mu_min'], asymp['f_inst'], \
  asymp['f_unit'], _, _, _, asymp['t'], asymp['mu'], asymp['rate'], asymp['I'] \
  = integrate_1cycle_wconstantdrive_analytically(mu_min_stat, Iext, D, Delta, K, tm, Vr, reset=reset) 
  
  # dynamics for high initial membrane potential, analytically derived
  high_init['t_off'], high_init['mu_max'], high_init['mu_reset'], high_init['mu_min'], high_init['f_inst'], \
  high_init['f_unit'], _, _, _, high_init['t'], high_init['mu'], high_init['rate'], high_init['I'] \
  = integrate_1cycle_wconstantdrive_analytically(mu_min_high, Iext, D, Delta, K, tm, Vr, reset=reset) 
  
  # dynamics for low initial membrane potential, analytically derived
  low_init['t_off'], low_init['mu_max'], low_init['mu_reset'], low_init['mu_min'], low_init['f_inst'], \
  low_init['f_unit'], _, _, _, low_init['t'], low_init['mu'], low_init['rate'], low_init['I'] \
  = integrate_1cycle_wconstantdrive_analytically(mu_min_low, Iext, D, Delta, K, tm, Vr, reset=reset) 
  
  # ---- PLOTTING ------------------------------------------------------------------------------------------
  if not len(ax):
    fig, ax = plt.subplots(2,2,sharex=True, sharey='row')
    despine(ax, which=['top', 'bottom', 'right'])
  # gridline([0], [ax[0][0], ax[1][0]], 'y', zorder=1)
  gridline([Vr, 1], [ax[1][0], ax[1][1]], 'y', zorder=1)
  gridline(Iext, [ax[1][0], ax[1][1]], 'y', color=my_colors['iext'], lw=1, linestyle='-', zorder=1)
  
  # plot first cycle with mu_min_high
  # mark cycle 
  for row in range(2):
    ax[row][0].axvspan(0, high_init['t_off'] + Delta, alpha=0.15, color=my_colors['cyc0'], zorder=-20, ec=None)
  ax[0][0].plot(high_init['t'], high_init['rate'], color='k', zorder=10)
  ax[1][0].plot(high_init['t'], high_init['mu'], color=my_colors['mu'], zorder=10)
  
  # plot first cycle with mu_min_low
  # mark cycle 
  for row in range(2):
    ax[row][1].axvspan(0, low_init['t_off'] + Delta, alpha=0.15, color=my_colors['cyc0'], zorder=-20, ec=None)
  ax[0][1].plot(low_init['t'], low_init['rate'], color='k', zorder=10)
  ax[1][1].plot(low_init['t'], low_init['mu'], color=my_colors['mu'], zorder=10)
  
  # plot subsequent cycles with asymptotic dynamics:
  ncyc = int(np.ceil(T / (asymp['t_off'] + Delta))) # more cycles than needed to fill time T
  dt = np.mean(np.diff(asymp['t']))
  tstart_high = high_init['t_off'] + Delta # start time for next cycle
  tstart_low = low_init['t_off'] + Delta # start time for next cycle
  
  for i in range(ncyc):
    ax[0][0].plot(tstart_high + asymp['t'], asymp['rate'], 'k')
    ax[0][1].plot(tstart_low + asymp['t'], asymp['rate'], 'k')
    ax[1][0].plot(tstart_high + asymp['t'], asymp['mu'], color=my_colors['mu'])
    ax[1][1].plot(tstart_low + asymp['t'], asymp['mu'], color=my_colors['mu'])
    tstart_high += asymp['t'][-1] + dt
    tstart_low += asymp['t'][-1] + dt
  
  # mark initial value
  ax[1][0].plot(0, mu_min_high, '.', color=my_colors['min'], zorder=20)
  ax[1][1].plot(0, mu_min_low, '.', color=my_colors['min'], zorder=20)
  
  # scale bar
  scalebar = AnchoredSizeBar(ax[1][1].transData, 5, '5 ms', 'lower right', frameon=False, borderpad=.1 ) 
  ax[1][1].add_artist(scalebar)
  
  # formatting
  ax[0][0].set_xlim([-.1, T])
  ax[0][0].set_ylim([-20,1000])
  ax[1][0].set_ylim([mu_min_low-.5, Iext+.5])
  ax[0][0].set_ylabel('rate\n[spk/s]')
  ax[1][0].set_ylabel('voltage,\ndrive')

  return ax



def plot_frequency_difference_pw_constant_drive(D, Delta, K, tm, Vr, reset=False, Vt=1, Imin=None, Imax=None, \
                                           mu_margin=.3 , dmu=.01, dI=.1, fmax=100, fmin=-100, fig=[], ax=[]):
  '''
  Plot difference between instantaneous and asymptotic network frequency under pw constant drive, depending on drive and initial condition.
  (colorplot in Fig 5B).

  Parameters
  ----------
  D, Delta, K, tm, Vr : network parameters
  reset : bool, optional. The default is False.
  Vt : Firing threshold. The default is 1.
  Imin : float, optional. Lower boundary of range of applicability of theory. The default is None.
  Imax : float, optional. Upper boundary of range of applicability of theory. The default is None.
  mu_margin : float, optional. Extra margin of mumin values to explore above and below its asymptotic range. The default is .3.
  dmu : float, optional. Resolution for exploration of initial conditions mumin. The default is .01.
  dI : float, optional. Resolution for exploration of drive levels Iext. The default is .1.
  fmax, fmin : Cutoff-boundaries for frequency colorbar. The default is -100, +100.
  fig, ax : figure and axis to place plot in, optional. The default is [].

  Returns
  -------
  fig, ax

  '''
  # find boundaries for mu_min0 and Iext:
  if not Imax: # take theoretical pt of full synch
    Imax = get_pt_fullsynch(D, Delta, K, tm)
  if not Imin: # take smallest possible drive for which theory works
    Imin = get_Iext_lower_bound(D, Delta, K, tm, Vr, reset=reset, Iext_min=0, Iext_max=3, dI=dI)[0]
  mu_min_max = Vt - np.sqrt(2*D*np.log(K/np.sqrt(2*pi*D)*np.exp(Delta/tm)))
  # set exploration range for mu_min0 and Iext:
  Iext = np.arange(Imin, Imax+dI, dI) # range for Iext
  mu_min, mu_max, f_net, f_unit, sat, zeta, t_on, t_off \
  = gaussian_drift_approx_stat(Iext, D, Delta, K, tm, Vr=Vr, Vt=Vt, reset=reset)[:-3]  # get stationary values
  mu_min_min, mu_min_max = np.min(mu_min)-mu_margin, np.min([np.max(mu_min)+mu_margin, mu_min_max])
  mu_min0 = np.arange(mu_min_min, mu_min_max, dmu) # range for mumin

  # get transient values:
  II, mm = np.meshgrid(Iext, mu_min0)
  t_off0 = get_popspk_end(II, mm, D, Delta, K, tm, Vt=Vt)
  f_net0 = get_fnet(t_off0, Delta)
  
  # difference in network frequencies
  dfreq = f_net0 - f_net 
  norm = matplotlib.colors.TwoSlopeNorm(0, vmin=fmin, vmax=fmax)
  
  # plot
  if not fig:
    fig, ax = plt.subplots(1,2,figsize=(8,6), gridspec_kw={'width_ratios':[30,1]}) #plt.subplots(figsize=(8,6))
  despine(ax)
  im = ax[0].imshow(dfreq, origin='lower', extent=(Iext[0]-dI/2, Iext[-1]+dI/2, mu_min0[0]-dmu/2, mu_min0[-1]+dmu/2),  \
                 cmap=plt.cm.coolwarm, norm=norm, aspect='auto', interpolation=None)
  ax[0].plot(Iext, mu_min, color='k', label=r'$\mu_\mathrm{min}^\mathrm{\infty}($'+str_Iext+')', zorder=10)
  ax[0].set_xlabel(r'external drive $I_\mathrm{E}$')
  ax[0].set_ylabel(r'initial $\mu_\mathrm{min}$')
  ax[0].autoscale(False)
  ax[0].legend(loc='upper right')
  ax[0].set_title(r'$\Delta = {}$ ms, $\tau_m={}$ ms, $K={:.1f}$, D={:.2f}'.format(Delta, tm, K, D))
  cb= plt.colorbar(im, label=r'$f_\mathrm{net}^\mathrm{inst}-f_\mathrm{net}^\mathrm{\infty}$ [Hz]', cax=ax[1], extend='both',\
                   ticks=[fmin, int(fmin/2), 0, int(fmax/2), fmax])
  cb.ax.plot([-1e7, 1e7],[0]*2, 'w', lw=1) 
  ax[0].set_xlim([Imin, Imax])
  ax[0].set_ylim([np.min(mu_min)-mu_margin, np.max(mu_min)+mu_margin])
  
  # suppl fig showing separately: fnet_inst, fnet_stat, and difference between both
  vmin=150
  vmax=300
  fig2, ax2 = plt.subplots(3, figsize=(6,10), sharex=True, sharey=True)
  im = ax2[0].imshow(f_net0, origin='lower', extent=(Iext[0]-dI/2, Iext[-1]+dI/2, mu_min0[0]-dmu/2, mu_min0[-1]+dmu/2),  \
                     vmin=vmin, vmax=vmax, interpolation=None, aspect='auto')
  ax2[0].plot(Iext, mu_min, color='k', label=r'$\mu_\mathrm{min}^\mathrm{\infty}($'+str_Iext+')', zorder=10)
  fig.colorbar(im, ax=ax2[0], label=r'$f_\mathrm{net}^\mathrm{inst}$ [Hz]')
  
  im = ax2[1].imshow(np.broadcast_to(f_net, f_net0.shape), origin='lower', extent=(Iext[0]-dI/2, Iext[-1]+dI/2, mu_min0[0]-dmu/2, mu_min0[-1]+dmu/2),  \
                     vmin=vmin, vmax=vmax, interpolation=None, aspect='auto')
  fig.colorbar(im, ax=ax2[1], label=r'$f_\mathrm{net}^\mathrm{\infty}$ [Hz]')
  
    
  im2 = ax2[-1].imshow(dfreq, origin='lower', extent=(Iext[0]-dI/2, Iext[-1]+dI/2, mu_min0[0]-dmu/2, mu_min0[-1]+dmu/2),  \
                 cmap=plt.cm.coolwarm, norm=norm, aspect='auto')
  fig.colorbar(im2, ax=ax2[-1], label=r'$f_\mathrm{net}^\mathrm{inst}-f_\mathrm{net}^\mathrm{\infty}$ [Hz]')
  ax2[-1].plot(Iext, mu_min, color='k', label=r'$\mu_\mathrm{min}^\mathrm{\infty}($'+str_Iext+')', zorder=10)
  ax2[-1].set_xlabel(r'external drive $I_\mathrm{E}$')
  ax2[-1].set_ylabel(r'initial $\mu_\mathrm{min}$')

  fig2.savefig('fig_5_suppl.pdf', bbox_inches='tight')
  
  return fig, ax

def plot_gaussian_drift_approx_transient(df, D, Delta, K, tm, Vr, reset=True, dt=.01, Iext_min=None, ls_stim = '-', \
                                      fmax=100, fmin=-100, fig=[], ax=[], ms_freq=4.5, ms=3, traces={}, nrows=2):
  '''
  Plot oscillation dynamics under transient drive over time, across mulitple consecutive cycles.

  Parameters
  ----------
  df : pandas data frame. Result dynamics.
  D, Delta, K, tm, Vr : network parameters.
  reset : bool, optional. The default is True.
  dt : float, optional. The default is .01.
  Iext_min : float, optional. Lower boundary for plotting. The default is None.
  ls_stim : str, optional. Line style for stimulus. The default is '-'.
  fmax, fmin : [Hz] boundaries for freq colorbar, optional. The default is +/-100.
  fig, ax : Figure and axis for plot. The default is [].
  ms_freq : float, optional. Markersize for inst freqs. The default is 4.5.
  ms : float, optional. Markersize for other transient quantities (mumax, mumin..). The default is 3.
  traces : dict, optional. Time-dependent traces for mu, rate, I. The default is {}.
  nrows : int, optional. Number of plot panels. The default is 2 (rate not shown). If 3, the population rate is shown also.

  Returns
  -------
  fig, ax.
  '''
  # stimulus
  Iext, t  = aux_get_stimulus(df, dt=dt)
  if not Iext_min:
    Iext_min = Iext[0]
  
  # cyclostationary values
  mu_min_stat, mu_max_stat, f_net_stat, f_unit_stat, _, _, _, t_off_stat, mu_reset_stat \
  = gaussian_drift_approx_stat(Iext[Iext>=Iext_min], D, Delta, K, tm, Vr=Vr, reset=reset)[:-2]
  
  # --- construct figure
  if not fig:
    fig = plt.figure(figsize=(7,6))
    gs = gridspec.GridSpec(1, 1, figure=fig)
    gs_sub = gs[0,0].subgridspec(nrows,1)
    ax = gs_sub.subplots(sharex=True)
    ax=[None]*nrows
    tight_layout = True
  else:
    tight_layout = False
    nrows = len(ax)
  
  ymin = np.min(mu_min_stat)-1.2
  
  # mark every other cycle in grey 
  for i in range(nrows):
    ax[i].spines['bottom'].set_visible(False)
    ax[i].tick_params(axis = "x", which = "both", bottom = False, top = False)
    for c in range(0, len(df), 2):
      ax[i].axvspan(df.loc[c, 'time_start'], df.loc[c, 'time_end']-1e-3, alpha=0.15, color=my_colors['cyc0'], zorder=-20, ec=None)
      if c+1 < len(df):
        ax[i].axvspan(df.loc[c+1, 'time_start'], df.loc[c+1, 'time_end']-1e-3, alpha=0.35, color=my_colors['cyc1'], zorder=-20, ec=None)

  # insert cycle numbers
  cycle_center = df.time_start + (df.t_off + Delta)/2
  for c, cyc in enumerate(cycle_center):
    ax[-1].text(cyc, ymin+0.1, str(c+1), ha='center', va='bottom', color='dimgrey', fontsize=plt.rcParams['xtick.labelsize'])
  
  # freq
  norm=matplotlib.colors.TwoSlopeNorm(0, vmin=fmin, vmax=fmax)
  ax[0].plot(t[Iext>=Iext_min], f_net_stat, 'k', label='$f_\mathrm{net}^\mathrm{\infty}$', lw=.5)
  ax[0].plot(df.time_toff, df.f_inst, linestyle=':', color='dimgrey', zorder=2, lw=.5)
  df.plot.scatter(ax = ax[0], x='time_toff', y = 'f_inst', \
                  c = plt.cm.coolwarm(norm(((df.f_inst-df.f_net_stat)).values)), edgecolors='k', s=ms_freq**2, lw=.5, \
                  zorder=3, legend=False) 
  
  ax[0].set_ylabel('freq [Hz]')
  ax[0].set_ylim([df.f_inst.min()-25, df.f_inst.max()+25])
  handles, labels = ax[0].get_legend_handles_labels()
  extra_handle = Line2D([0], [0], linestyle=':', marker='o', color='dimgrey', lw=.5, markersize=ms_freq, markerfacecolor='w',\
                        markeredgecolor='k', markeredgewidth=.5, label='$f_\mathrm{net}^\mathrm{inst}$')
  handles.append(extra_handle)
  ax[0].legend(handles=handles, labelspacing=0.1, bbox_to_anchor=(1., 1), loc='lower right', borderaxespad=.1,handlelength=1, ncol=2)
  
  # mu
  gridline([0,1], ax[-1], 'y')
  # plot stimulus
  ax[-1].plot(t, Iext, color=my_colors['iext'], linestyle=ls_stim)#, label=str_Iext)
  if  (df.m!=0).any():
    txt = ax[-1].text(.97, .95, 'm={:.2f}/ms'.format(df.m.max()) , \
                      horizontalalignment='right', verticalalignment='top', transform=ax[-1].transAxes)
  ax[-1].plot(t[Iext>=Iext_min], mu_max_stat, color=my_colors['max'], label=r'$\mu_\mathrm{max}^\mathrm{\infty}$', lw=.5)
  ax[-1].plot(t[Iext>=Iext_min], mu_min_stat, color=my_colors['min'], label=r'$\mu_\mathrm{min}^\mathrm{\infty}$', lw=.5) 
  ax[-1].plot(df.time_toff, df.mu_max, markersize=ms, marker='.', linestyle='', color=my_colors['max'], zorder = 10, label=r'$\mu_\mathrm{max}^\mathrm{inst}$')
  ax[-1].plot(df.time_start, df.mu_min_start, markersize=ms, marker='.', linestyle='', color=my_colors['min'], zorder = 10, label=r'$\mu_\mathrm{min}^\mathrm{inst}$')
  
  ax[-1].plot(df.iloc[-1, :]['time_end'], df.iloc[-1, :]['mu_min_end'], marker='.', linestyle='', color=my_colors['min'], zorder = 10)
  
  for c in range(len(traces)):
    ax[-1].plot(traces[c]['t'], traces[c]['mu'], 'dimgrey', zorder = 9)


  ax[-1].set_ylabel('voltage, drive')#('$I_\mathrm{E}(t)$')
  ax[-1].set_xlim([0, t[-1]])
  ax[-1].set_ylim(bottom=ymin)
  ax[-1].set_xticks([]) 
  
  # scale bar:
  scalebar = AnchoredSizeBar(ax[-1].transData, 5, '5 ms', 'upper right', frameon=False, borderpad=.7) #,\
  ax[-1].add_artist(scalebar)

  ax[-1].legend(bbox_to_anchor=(0,0), loc='upper left', columnspacing=.8, borderaxespad=.1, handlelength=1, ncol=4)

  if nrows == 3:
    for c in range(len(traces)):
      ax[1].plot(traces[c]['t'], traces[c]['rate'], 'k')
    ax[1].set_ylabel('rate\n[spk/s]')
  
  if tight_layout:
    fig.tight_layout()
  return fig, ax

def analysis_IFA_constant_drive_analytical(Iext_min, Iext_max, nsteps, mu_min_0, D, Delta, K, tm, Vr, reset=True, \
                                           fig=[], ax_time = [], ax_phase=[], fmin=-100, fmax=100, ms=4.5):
  '''
  Analysis of transient oscillation dynamics for piecewise constant drive (Fig 5).

  Parameters
  ----------
  Iext_min : float. Drive in first cycle.
  Iext_max : float. Drive at plateau.
  nsteps : Number of steps in drive inbetween.
  mu_min_0 : mu at the beginning of the first cycle.
  D, Delta, K, tm , Vr : Network parameters.
  reset : bool, optional. The default is True.
  fig, ax_time, ax_phase : plotting axes, optional. The default is [].
  fmin, fmax : Hz, optional. Colorbar limits. The default is +/-100.
  ms : float, optional. Markersize. The default is 4.5.

  Returns
  -------
  2 figures and axes

  '''
  
  Iext_up = np.linspace(Iext_min, Iext_max, nsteps, endpoint=True)
  Iext_down = Iext_up[:-1][::-1]
  Iext = np.concatenate((Iext_up, Iext_down))
  
  # compute transient dynamics and create figure in time
  df, traces, fig_time, ax_time = iterate_constant_drive_analytical(mu_min_0, Iext, D, Delta, K, tm, Vr, reset=reset, plot=True,\
                                                            fmax=fmax, fmin=fmin, fig=fig, ax=ax_time)
  
  # create phase space figure and insert time trajectory
  fig_phase, ax_phase = plot_frequency_difference_pw_constant_drive(D, Delta, K, tm, Vr, reset=reset, fmax=fmax, fmin=fmin,\
                                                               fig=fig, ax=ax_phase)
  # insert path in analytical figure
  norm = matplotlib.colors.TwoSlopeNorm(0, vmin=fmin, vmax=fmax)
  for c in range(len(df)):
    ax_phase[0].plot(df.loc[c, 'Iext_toff'], df.loc[c,'mu_min_start'], 'o', markersize=ms, markeredgecolor='k', \
               markerfacecolor=plt.cm.coolwarm(norm(df.loc[c, 'f_inst']-df.loc[c, 'f_net_stat'])), markeredgewidth=0.5, zorder=10)
    if c+1<len(df):
      line = ax_phase[0].plot((df.loc[c, 'Iext_end'], df.loc[c+1, 'Iext_start']), (df.loc[c,'mu_min_end'], df.loc[c+1,'mu_min_start']), \
                         color='dimgrey', linestyle=':', zorder=3, lw=1)[0]
    if np.abs(df.loc[c,'mu_min_start']-df.loc[c,'mu_min_end'])>=0.5:
      ax_phase[0].arrow(df.loc[c, 'Iext_start'], df.loc[c, 'mu_min_start'], 0 , df.loc[c, 'mu_min_end']-df.loc[c, 'mu_min_start'],\
                      length_includes_head=True, head_length=.25, color='k', head_width=.15, lw=.5)
    else:
      ax_phase[0].arrow(df.loc[c, 'Iext_start'], df.loc[c, 'mu_min_start'], 0 , df.loc[c, 'mu_min_end']-df.loc[c, 'mu_min_start'],\
                      length_includes_head=True, head_width=.1, color='k', lw=.5)
      
    # annotate cycle numbers:
    dI = .2
    ax_phase[0].text(df.loc[c,'Iext_start']+dI, df.loc[c, 'mu_min_start'], str(c+1), ha='left', va='top', color='dimgrey', \
                     fontsize=plt.rcParams['xtick.labelsize'])

  if not fig:
    fig_phase.tight_layout()
  return fig_time, fig_phase, ax_time, ax_phase, df, traces


#%% Gaussian-drift approx. analytical: pw linear drive

def gaussian_drift_approx_transient(m_val, Iext_toff_val, D, Delta, K, tm, Vr,  \
                                    mu_min_start_val=[], reset=True,\
                                    n_mu=10, mu_margin=.3): # analysis_linear_drive_analytical_centered
  ''' 
  Calculate mu_max, t_off, f_inst, mu_min_end, Iext_start, Iext_end for a number of cycles with "center conditions":
    m: slope of drive, all positive
    Iext_toff: drive Iext at time t_off
    mu_min_start: mu_min at beginning of cycle (initial condition)
  return figure and data frame containing results for each cycle
  '''
  print('Compute Gaussian-drift approximation under linear drive for a range of slopes, initial conditions, and reference drives.')
  # make sure the input ranges are arrays and sorted by slope
  m_val, Iext_toff_val = np.array(m_val), np.array(Iext_toff_val)
  m_val = np.sort(m_val)[::-1]
  m_val = np.concatenate((m_val, -m_val[::-1])) # add negative slopes to exploration
  
  # initialize result container
  df = pd.DataFrame(columns=["m", "Iext_toff", "mu_min_start", "mu_max", "mu_reset", "Iext_end", "mu_min_end", "t_off", "f_inst", "Iext_start"])
  
  # add range of initial values to be explored, if not provided as input
  if not len(mu_min_start_val):
    # no range of initial values for mu provided: use asymptotic ones associated to resp. reference drive 
    mu_min_stat = gaussian_drift_approx_stat(Iext_toff_val, D, Delta, K, tm, Vr=Vr, reset=reset)[0]
    mu_min_start_val = np.linspace(np.nanmin(mu_min_stat)-mu_margin, np.nanmax(mu_min_stat), n_mu)
  
  # store the cycle configurations that will be explored (slope, drive, initial value)
  df['m'] = np.repeat(m_val, Iext_toff_val.size*mu_min_start_val.size)
  df['Iext_toff'] = np.tile(np.repeat(Iext_toff_val, mu_min_start_val.size), m_val.size)
  df['mu_min_start'] = np.tile(mu_min_start_val, m_val.size*Iext_toff_val.size)
  # the drive at the end of the cycle can already be inferred now
  df['Iext_end'] = get_linear_drive(Delta, df.m, df.Iext_toff)
  
  # do gaussian drift approximation for all cycle configurations:
  # we could just loop over all cycle configurations and use gaussian_drift_approx_transient_1cycle, but the following is more efficient:
  for m in m_val:
    print('Slope m={}'.format(m))
    for Iext_toff in Iext_toff_val:
      # mu_max, mu_reset, mu_min_end are independent of the initial condition mu_min_start:
      ix = (df.m==m) & (df.Iext_toff==Iext_toff) # indices of all cycles with same m and Iext_toff (variable mu_min_start)
      mu_max = get_mumax_transient(m, Iext_toff, D, Delta, K, tm)
      mu_min_end, mu_reset = get_mumin_transient(mu_max, Iext_toff, m, D, Delta, K, tm, Vr, reset=reset)
      df.loc[ix, 'mu_max'] = mu_max
      df.loc[ix, ['mu_min_end', 'mu_reset']] = mu_min_end, mu_reset
      
      # t_off depends on mu_min_start and needs to be calculated for every individual cycle
      for mu_min_start in mu_min_start_val:
        ix = (df.m==m) & (df.Iext_toff==Iext_toff) & (df.mu_min_start==mu_min_start)
        df.loc[ix, 't_off'] = get_toff_transient(m, Iext_toff, mu_max, mu_min_start, tm)
  # infer remaining variables:
  df['f_inst'] = 1000/(df.t_off + Delta)
  df['mu_min_stat'], df['mu_max_stat'], df['f_stat'] = gaussian_drift_approx_stat(df.Iext_toff, D, Delta, K, tm, Vr=Vr, reset=reset, check_mumin = True)[:3]
  df['Iext_start'] =  get_linear_drive(-df.t_off, df.m, df.Iext_toff)

  df.to_csv('results/gaussian_drift_approx_linear_drive_exploration_fig6-7.csv')
  return df

def gaussian_drift_approx_transient_1cycle(m, Iext_toff, mu_min_start, D, Delta, K, tm, Vr, \
                                           Vt=1, reset=False, n_time=1000, return_traces=False):
  '''
  Gaussian-drift approx for a single cycle with linear drive.

  Parameters
  ----------
  m : float. [dimless volt / ms]. Slope of drive.
  Iext_toff : Reference drive.
  mu_min_start : Initial mean membrane potential.
  D, Delta, K, tm , Vr : Network parameters.
  Vt : float, optional. Spike threshold. The default is 1.
  reset : bool, optional. Reset? The default is False.
  n_time : int, optional. Number of time steps for traces. The default is 1000.
  return_traces : bool, optional. Whether to compute time-dependent traces of mu, rate also. The default is False.

  Returns
  -------
  mu_max, mu_reset, mu_min_end, t_off, fnet, Iext_start, Iext_end: characterization of the oscillation cycle
  t, mu_t, rate_t: Time-dependent traces of mu, and rate if return_traces==True

  '''
  mu_max = get_mumax_transient(m, Iext_toff, D, Delta, K, tm) # Eq 65
  t_off = get_toff_transient(m, Iext_toff, mu_max, mu_min_start, tm) # Eq 66
  mu_min_end, mu_reset = get_mumin_transient(mu_max, Iext_toff, m, D, Delta, K, tm, Vr, reset=reset) # Eq
  fnet = 1000 / (t_off + Delta) # Hz, Eq. 68
  # i hindsight we can infer the drive at the beginning and end of the cycle:
  Iext_start = get_linear_drive(-t_off, m, Iext_toff)
  Iext_end = get_linear_drive(Delta, m, Iext_toff)
  
  if return_traces:
    # add traces for mu and rate for plotting purposes:
    t_up = np.linspace(0, t_off, n_time)  # time for upstroke of mu
    t_down = np.linspace(t_off, t_off+Delta, n_time) # time for downstroke of mu
    mu_up =  get_mu_trajectory_up_linear_drive(t_up, mu_min_start, Iext_start, tm, m) # trajectory of mu on upstroke 
    mu_down = get_mu_trajectory_down_linear_drive(t_down - t_down[0], mu_max, mu_reset, Iext_toff, D, Delta, K, tm, m)# trajectory of mu on downstroke
    Iext_up = get_linear_drive(t_up, m, Iext_start)
    
    window1 = np.where((t_up > t_off - 2*Delta) & (t_up <= t_off - Delta))[0] # second order feedback (2Delta before toff, influencing mu trajectory, and thus rate before toff)
    window2 =  np.where((t_up > t_off - Delta) & (t_up <= t_off))[0] # first order feedback (directly before toff, influencing inh feedback in next window)

    I_up, rate_up = np.zeros(t_up.size), np.zeros(t_up.size) # total current resulting from upstroke, arriving at times [Delta, toff+Delta]
    I_up[window1] = Iext_up[window1] # first, no inh current    
    # rate during time window1 as we integrate it for the feedback in Step 2:
    rate_up[window1] = (I_up[window1] - mu_up[window1])/(tm/1000)*get_gauss(Vt, mu_up[window1], np.sqrt(D), broadcast=False) # Hz
    # due to this (small) rate in window1 the current in window2 is slightly smaller than Iext (first inh feedback arrives):
    I_up[window2] = Iext_up[window2] - tm/1000*K*np.pad(rate_up[window1], (len(window2)-len(window1), 0)) # zero-pad rate, if window1 is shorter than window2 (Delta)
    rate_up[window2] = (I_up[window2]- mu_up[window2])*get_gauss(Vt, mu_up[window2], np.sqrt(D), broadcast=False)/(tm/1000) # Hz
    
    rate_down = np.zeros_like(t_down) # no rate on downstroke
    t = np.concatenate((t_up, t_down))
    mu_t = np.concatenate((mu_up, mu_down))
    rate_t = np.concatenate((rate_up, rate_down))
    return mu_max, mu_reset, mu_min_end, t_off, fnet, Iext_start, Iext_end, t, mu_t, rate_t
  else:
    return mu_max, mu_reset, mu_min_end, t_off, fnet, Iext_start, Iext_end


def get_linear_drive(t, m, Iext0):
  ''' linear fct Iext(t) with slope m '''
  return Iext0 + m*t

def get_toff_transient(m, Iext_toff, mu_max, mu_min0, tm): 
  ''' closed form solution for t_off using Labert W function (Eq (63)) '''
  W_arg = (Iext_toff-m*tm-mu_max)/m/tm*np.exp(-1+(Iext_toff-mu_min0)/m/tm)
  if np.isscalar(W_arg):
    if W_arg < -np.exp(-1):
      return nan
    else:
      W0 = np.real(scipy.special.lambertw(W_arg, k=0)) # bracnh 0
      W1 = np.real(scipy.special.lambertw(W_arg, k=-1)) # branch -1, use for negative slopes!
      toff = np.clip(-tm*W0 + (Iext_toff-m*tm-mu_min0)/m, 0, None)
      toff1 = np.clip(-tm*W1 + (Iext_toff-m*tm-mu_min0)/m, 0, None)
      if np.sign(m)<= 0:
        toff = toff1
  else:
    W0, W1 = np.ones(W_arg.size)*nan, np.ones(W_arg.size)*nan
    W0[W_arg >= -np.exp(-1)] = np.real(scipy.special.lambertw(W_arg[W_arg >= -np.exp(-1)], k=0)) # bracnh 0
    W1[W_arg >= -np.exp(-1)] = np.real(scipy.special.lambertw(W_arg[W_arg >= -np.exp(-1)], k=-1)) # branch -1, use for negative slopes!
    toff = np.clip(-tm*W0 + (Iext_toff-m*tm-mu_min0)/m, 0, None)
    toff1 = np.clip(-tm*W1 + (Iext_toff-m*tm-mu_min0)/m, 0, None)
    toff[m<=0] = toff1[m<=0]  
    
  return toff

def get_mumax_transient(m, Iext_toff, D, Delta, K, tm, Vt=1): # get_mumax_transient_backward_firstorder_m_analyt
  '''mu_max under linear drive (Eq (62))
  '''
  mu_max_stat = get_mu_max(Iext_toff, D, Delta, K, tm, Vt=Vt, check_masks=True) # mu_max under constant drive Iext_toff
  term1 = tm*(1-np.exp(-Delta/tm)) / ((Iext_toff-Vt)*np.sqrt(2/D*np.log(K/np.sqrt(2*pi*D)*np.exp(Delta/tm))) + 2*np.log(K/np.sqrt(2*pi*D)*np.exp(Delta/tm)))
  term2 = tm - (Delta+tm)*np.exp(-Delta/tm)
  mu_max_trans = mu_max_stat + m*(term1-term2)
  return mu_max_trans


def get_mu_trajectory_backward(t_back, Iext_toff, mu_max, tm):
  ''' backward approx of mu(t_off-t_back), independent of slope m ! '''
  # linear approx of mu(t_off - t_back)
  return Iext_toff - (Iext_toff- mu_max)*np.exp(t_back/tm)

def get_mu_trajectory_up_linear_drive(t, mu_min_start, Iext_start, tm, m):
  ''' trajectory of mu during upstroke, taking into account slope m of the linear drive (Eq. 58) 
  '''
  Iext_t = get_linear_drive(t, m, Iext_start)
  mu_t = Iext_t - m*tm + (m*tm + mu_min_start - Iext_start)*np.exp(-t/tm) # (A2)
  return mu_t

def get_mu_trajectory_down_linear_drive(t, mu_max, mu_reset, Iext_toff, D, Delta, K, tm, m, Vt=1):
  ''' trajectory of mu during downstroke (Eq. 66) 
  '''
  dt = np.mean(np.diff(t))
  # stationary part:
  mu_t_stat = get_mu_down(t, mu_reset, Iext_toff, D, Delta, K, mu_max, tm)  
  # new part of the current due to linear rise:
  mu_past = get_mu_trajectory_backward(Delta-t, Iext_toff, mu_max, tm) # no m dependence here!!!!
  mu_past2 = get_mu_trajectory_backward(2*Delta-t, Iext_toff, mu_max, tm) 
  p_past = get_gauss(Vt, mu_past, np.sqrt(D), broadcast=False)
  p_past2 = get_gauss(Vt, mu_past2, np.sqrt(D), broadcast=False)
  I_extra = m*t + K*p_past*m*(Delta-t) - K**2*p_past*p_past2*m*(2*Delta-t)
  integral_extra = np.cumsum(I_extra * np.exp(-(Delta-t)/tm))*dt/tm # numerical solution of an integral, this makes this approach semi-analytical!
  mu_t = mu_t_stat + integral_extra # Eq. (69)
  return mu_t

def get_mumin_transient(mu_max, Iext_toff, m, D, Delta, K, tm, Vr, reset=True, dt=0.01, Vt=1):
  ''' integrate inh feedback to infer the next mumin (Eq. 66)''' 
  # cyclostationary mumin
  mu_min_stat, mu_reset = get_mu_min(Iext_toff, mu_max, D, Delta, K, tm, Vr=Vr, reset=reset)
  # new part of the current due to linear rise:
  t = np.arange(0, Delta, dt)
  mu_past = get_mu_trajectory_backward(Delta-t, Iext_toff, mu_max, tm) # get_mu_forward(t_off - (Delta-t), m, Iext0, mumin0, tm)
  mu_past2 = get_mu_trajectory_backward(2*Delta-t, Iext_toff, mu_max, tm) # get_mu_forward(t_off - (2*Delta-t), m, Iext0, mumin0, tm)
  p_past = get_gauss(Vt, mu_past, np.sqrt(D), broadcast=False)
  p_past2 = get_gauss(Vt, mu_past2, np.sqrt(D), broadcast=False)
  I_extra = m*t + K*p_past*m*(Delta-t) - K**2*p_past*p_past2*m*(2*Delta-t)
  integral_extra = np.sum(I_extra * np.exp(-(Delta-t)/tm))*dt/tm # numerical solution of an integral, this makes this approach semi-analytical!
#  print(mu_min_stat, integral_extra)
  mu_min = mu_min_stat + integral_extra
  return mu_min, mu_reset



def aux_get_stimulus(df, dt = .01):
  '''
  auxiliary function reconstructing (piecewise) linear stimulus from cycle-wise results data frame.

  Parameters
  ----------
  df : pandas data frame with results for each cycle.
  dt : float, optional. Time step. The default is .01.

  Returns
  -------
  Iext, t: stimulus and resp. time points
  '''
  def linear_fct(m, f0, x0):
    return lambda x: f0 + m*(x-x0)
  
  t = np.arange(0, df.time_end.max(), dt)
  
  sections = [(t>=df.loc[c,'time_start']) & (t<df.loc[c,'time_end']) for c in range(len(df))]
  functions = [linear_fct(df.loc[c,'m'], df.loc[c,'Iext_start'], df.loc[c,'time_start']) for c in range(len(df))]
  Iext = np.piecewise(t, sections, functions)
  
  return Iext, t


def visualize_traces_wlineardrive(D, Delta, K, tm, Vr, m, mu_min_0, Iext_toff=nan, Iext_0=nan, i_ref=nan, \
                                  reset=True, dt=.01, tmax = 10, fig=None, ax=None, fmin=-100, fmax=100, traces_analytical={}, show_numerical_curves=False):
  '''
  plot individual example cycles, and show the analytically (and numerially) derived dynamics (Figs 5,6 A)
  Parameters
  ----------
  m : [1/ms]
    SLOPE OF EXTERNAL DRIVE. SLOPES 0, m, -m WILL BE COMPARED. The default is .4.
  Iext_toff : []
    EXTERNAL DRIVE AT END OF POPSPK. The default is 6
  dt : [ms], optional
    TIME STEP FOR NUMERICAL INTEGRATION. The default is .01.
  tmax : [ms], optional
    MAXIMAL TIME FOR NUMERICAL INTEGRATION. The default is 10.

  Returns
  -------
  figure comparing the traces of mu, current I and rate r for 3 different input slopes m
  '''
  # transform all scalar inputs to lists
  if np.isscalar(m):
    m = list([m])
  if np.isscalar(mu_min_0):
    mu_min_0 = list([mu_min_0])
  if np.isscalar(Iext_toff):
    Iext_toff = list([Iext_toff])
  if np.isscalar(Iext_0):
    Iext_0 = list([Iext_0])
    
  L = np.max([len(x) for x in [m, mu_min_0, Iext_toff, Iext_0]])
  
  if L>1:
    # broadcast all other lists that contain only one scalar value
    if len(m)==1:
      m = m*L
    if len(mu_min_0)==1:
      mu_min_0 = mu_min_0*L
    if len(Iext_toff)==1:
      Iext_toff = Iext_toff*L
    if len(Iext_0)==1:
      Iext_0 = Iext_0*L
      
  n_examples = len(m) # number of example cycles to be shown
  norm = matplotlib.colors.TwoSlopeNorm(0, vmin=fmin, vmax=fmax)
  res_num, res_analyt = {}, {}
  for i in range(n_examples):
    # numerical solution
    res_num[i] = integrate_1_cycle_wlineardrive_numerically(D, Delta, K, tm, Vr, m[i], mu_min_0[i], Iext_toff=Iext_toff[i], Iext_0=Iext_0[i], \
                                            reset=reset, dt=dt, tmax=tmax)
    # analytical solution
    if not m[i]:
      res_analyt[i] = {}
      res_analyt[i]['t_off'], res_analyt[i]['mu_max'], res_analyt[i]['mu_reset'], res_analyt[i]['mu_min'], res_analyt[i]['f_inst'], \
      res_analyt[i]['f_unit'], _, _, _, res_analyt[i]['t'], res_analyt[i]['mu'], res_analyt[i]['rate'], res_analyt[i]['I'] \
      = integrate_1cycle_wconstantdrive_analytically(mu_min_0[i], Iext_toff[i], D, Delta, K, tm, Vr, reset=reset) 
    else:
      res_analyt[i] = {}
      res_analyt[i]['mu_max'], res_analyt[i]['mu_reset'], res_analyt[i]['mu_min'], res_analyt[i]['t_off'], res_analyt[i]['f_inst'], \
      res_analyt[i]['Iext_start'], res_analyt[i]['Iext_end'], res_analyt[i]['t'], res_analyt[i]['mu'], res_analyt[i]['rate'] \
      = gaussian_drift_approx_transient_1cycle(m[i], Iext_toff[i], mu_min_0[i], D, Delta, K, tm, Vr, \
                                               reset=reset, return_traces=True)
  # boundaries for plotting:   
  Iext_max = np.max([np.max(res_num[i]['Iext']) for i in range(len(m))])
  t_max = 1.1*np.max([np.max(res_analyt[i]['t']) for i in range(len(m))])
  mu_min = np.min([np.min(res_analyt[i]['mu']) for i in range(len(m))])
  t0 = t_max / 2 # alignment for plotting: shift numerical and analytical t_off to t0 (middle of plot)
  
  # --- plot
  with plt.rc_context({"xtick.direction": "out", "ytick.direction": "out", 'xtick.major.pad': 1, 'ytick.major.pad': 1}):
    if not fig:
      # --- construct figure
      fig = plt.figure(figsize=(12, 5))#, constrained_layout=True)
      gs = gridspec.GridSpec(1, len(m)+1, figure=fig, wspace=.4, hspace=.4, width_ratios=[1]*len(m)+[.5])
      ax = [None]*len(m)
      
      gs_0 = gs[0,0].subgridspec(2,1, height_ratios=[1,2])
      ax0 = fig.add_subplot(gs_0[0])
      ax1 = fig.add_subplot(gs_0[1], sharex=ax0)
      ax[0] = [ax0, ax1]
      
      if len(m) > 1:
        for i in range(1,len(m)):
          gs_sub = gs[0,i].subgridspec(2,1, height_ratios=[1,2])
          ax[i] = [fig.add_subplot(gs_sub[0], sharex=ax0, sharey=ax1), \
                   fig.add_subplot(gs_sub[1], sharex=ax0, sharey=ax1)]
    elif len(ax)!=len(m):
      raise ValueError('Provided axes must be list of length len(m) containing sublists of length 2!')
    despine(ax)
    
    # --- fill figure
    for i in range(len(m)):      
      # shift numerical and analytical t_off to the same point:
      offset_analyt = -(res_analyt[i]['t_off']+Delta)/2 +t0 # set middle of trace into middle of plot
      if i==i_ref or np.isnan(i_ref):
        color = 'k' 
      else:
        color = plt.cm.coolwarm(norm(1000/(res_analyt[i]['t_off']+Delta) - 1000/(res_analyt[i_ref]['t_off']+Delta)))
      # mark cycle 
      for row in range(2):
        ax[i][row].axvspan(0 + offset_analyt, res_analyt[i]['t_off']+Delta+offset_analyt, alpha=0.15, color=my_colors['cyc0'], zorder=-20, ec=None)
        
      gridline(0, ax[i][0])  
      ax[i][0].plot(res_analyt[i]['t']+offset_analyt, res_analyt[i]['rate'], color=color, zorder=3) 
      
      
      gridline([0,1, Iext_toff[i]], ax[i][1], 'y', zorder=1)
      if m[i]:
        Iext_t = get_linear_drive(res_analyt[i]['t'], m[i], res_analyt[i]['Iext_start'])
      else:
        Iext_t = np.ones(len(res_analyt[i]['t']))*Iext_toff[i]
      ax[i][1].plot(res_analyt[i]['t']+offset_analyt, Iext_t, color=my_colors['iext'], label=r'$I_\mathrm{E}$', zorder=2)
      # mark stationary mu_max and mu_min
      if not np.isnan(i_ref):
        ax[i][1].axhline(res_analyt[i_ref]['mu_min'], color=my_colors['min'], linestyle='-', lw=.5 )
        ax[i][1].axhline(res_analyt[i_ref]['mu_max'], color=my_colors['max'], linestyle='-', lw=.5, zorder=0 )
      if not (np.array(m)==0).all(): # drive not constant, mark reference drive at time toff
        ax[i][1].plot(res_analyt[i]['t_off']+offset_analyt, Iext_toff[i], '.', ms=4, color=my_colors['iext'], zorder=2)
      ax[i][1].plot(res_analyt[i]['t']+offset_analyt, res_analyt[i]['mu'], color=my_colors['mu'], label=r'$\mu$', zorder=10)
      ax[i][1].plot(0+offset_analyt, mu_min_0[i], '.', color=my_colors['min'])
      ax[i][1].plot(res_analyt[i]['t_off']+offset_analyt, res_analyt[i]['mu_max'],  '.', color=my_colors['max'])

      if len(np.unique(m))>1:
        ax[i][0].set_title('m={:.1f}/ms'.format(m[i]))

      if not i:
        # yticks
        yticks = [0, int(Iext_toff[0])]#, int(Iext_toff[i])]
        ax[0][1].set_yticks(yticks)
      else:
        ax[i][0].tick_params(axis = "y", which = "both", left = False, right = False, labelleft=False)
        ax[i][1].tick_params(axis = "y", which = "both", left = False, right = False, labelleft=False)
          
      # xticks 
      ax[i][0].tick_params(axis = "x", which = "both", bottom = False, top = False, labelbottom=False)
      ax[i][1].tick_params(axis = "x", which = "both", bottom = False, top = False, labelbottom=False)
      ax[i][0].set_xlim([0, t_max]) #[-res_analyt[2]['t_off'], Delta])
      ax[i][0].set_ylim([-20, 1000]) # 100+np.max([np.max(res_analyt[i]['rate']) for i in range(len(m))])])
      ax[i][1].set_ylim([mu_min-.5, Iext_max+.5])
      
      if show_numerical_curves:
        offset_num = - res_num[i]['t_off'] + res_analyt[i]['t_off'] + offset_analyt # align numerical with analytical t_off and shift by same amount offset_analyt
        ax[i][0].plot(res_num[i]['t']+offset_num, res_num[i]['r'], color=color, linestyle='--', lw=.5, zorder=2) 
        ax[i][1].plot(res_num[i]['t']+offset_num, res_num[i]['Iext'], color=my_colors['iext'], label=r'$I_\mathrm{E}$', zorder=2)
        ax[i][1].plot(res_num[i]['t']+offset_num, res_num[i]['mu'], color=my_colors['mu'], linestyle='--', lw=.5, zorder=12)

    # scale bar:
    scalebar = AnchoredSizeBar(ax[-1][1].transData, 5, '5 ms', 'lower right', frameon=False, borderpad=.1 ) 
    ax[-1][1].add_artist(scalebar)
    # formatting
    ax[0][0].set_ylabel('rate \n[spk/s]')
    ax[0][1].set_ylabel('voltage,\ndrive')
    for i in [1,2]:
      plt.setp(ax[i][0].get_yticklabels(), visible=False)
  
  return fig, ax  

def find_trajectories_linear_drive(df, m_val, D, Delta, K, tm, Vr, Iext_min, Iext_plateau, Iext_start = [1, 1.3, 1.3], mu_min_start_first_cycle = .5, tolerance_Iext= .1, reset=True):
  ''' (for Fig 6/7)
  Based on a large "look-up table" df of the analytical results for SINGLE-cycle oscillation dynamics for a range of initial conditions, slopes and reference drives:
  Find by numerical search a number of consecutive cycles that match a symmetric double-ramp drive of slopes m_val, and plateau Iext_plateau.

  Parameters
  ----------
  df : pandas data frame. "Look-up table" of the analytical results for SINGLE-cycle oscillation dynamics for a range of initial conditions, slopes and reference drives
  m_val : np.array. Slopes of double-ramps for which to find consecutive cycles.
  D, Delta, K, tm , Vr : Network parameters.
  Iext_min : Lower bound for reference drive of each cycle.
  Iext_plateau : Drive during plateau phase: The last cycle of the upstroke should finish with Iext approx at this drive.
  Iext_start : Approx. drive at the beginning of the first cycle (for each of the slopes in m_val). The default is [1, 1.3, 1.3].
  mu_min_start_first_cycle : float, optional. Mu at beginning of first cycle. The default is .5.
  tolerance_Iext : float, optional. Deviation allowed between Iext(end of upstroke) and Iext_plateau. The default is .1.
  reset : bool, optional. The default is True.

  Returns
  -------
  trajectories_all : pandas data frame with info on all cycles (numbered by order)
  traces_all : dict with time-dependent traces of mu and rate across cycles.
  '''
  trajectories_all = pd.DataFrame(columns=list(df.columns) + ['cycle','time_start', 'time_toff', 'time_end'])
  traces_all = {}
  mu_min_stat_plateau = gaussian_drift_approx_stat(Iext_plateau, D, Delta, K, tm, Vr=Vr, reset=reset)[0]

  for i, m in enumerate(m_val):
    trajectory_up = find_upstroke_trajectory(df, m, Iext_plateau, Iext_min, Iext_start[i], mu_min_start_first_cycle, D, Delta, K, tm, Vr, tolerance_Iext=tolerance_Iext, reset=reset)
    trajectory_down = find_downstroke_trajectory(df, m, Iext_plateau, Iext_min, Delta, mu_min_start = mu_min_stat_plateau, tolerance_Iext=tolerance_Iext)
    trajectories_all = pd.concat([trajectories_all,trajectory_up, trajectory_down]) 
  
    # add traces for plotting:       
    traces_all[m] = {'t':np.array([]), 'mu':np.array([]), 'rate':np.array([])}
    traces_all[-m] = {'t':np.array([]), 'mu':np.array([]), 'rate':np.array([])}
    # upstroke:
    for c in range(len(trajectory_up)): # find traces for each cycle 
      Iext_toff = trajectory_up[trajectory_up.cycle==c].Iext_toff.squeeze()
      mu_min_start = trajectory_up[trajectory_up.cycle==c].mu_min_start.squeeze()
      t_c, mu_c, rate_c = gaussian_drift_approx_transient_1cycle(m, Iext_toff, mu_min_start, D, Delta, K, tm, Vr, \
                                                           reset=reset, return_traces=True)[-3:]
      if c>0: # adjust time stamps w.r.t previous cycles:
        t_c += traces_all[m]['t'][-1]
      # add this cycle to overall trajectory:
      traces_all[m]['t'] = np.concatenate((traces_all[m]['t'], t_c))
      traces_all[m]['mu'] = np.concatenate((traces_all[m]['mu'], mu_c))
      traces_all[m]['rate'] = np.concatenate((traces_all[m]['rate'], rate_c))
    # downstroke:
    for c in range(len(trajectory_down)): # find traces for each cycle 
      Iext_toff = trajectory_down[trajectory_down.cycle==c].Iext_toff.squeeze()
      mu_min_start = trajectory_down[trajectory_down.cycle==c].mu_min_start.squeeze()
      t_c, mu_c, rate_c = gaussian_drift_approx_transient_1cycle(-m, Iext_toff, mu_min_start, D, Delta, K, tm, Vr, \
                                                           reset=reset, return_traces=True)[-3:]
      if c>0: # adjust time stamps w.r.t previous cycles:
        t_c += traces_all[-m]['t'][-1]
      # add this cycle to overall trajectory:
      traces_all[-m]['t'] = np.concatenate((traces_all[-m]['t'], t_c))
      traces_all[-m]['mu'] = np.concatenate((traces_all[-m]['mu'], mu_c))
      traces_all[-m]['rate'] = np.concatenate((traces_all[-m]['rate'], rate_c))
  
  return trajectories_all, traces_all

# upstroke
def find_upstroke_trajectory(df, m, Iext_plateau, Iext_min, Iext_start, mu_min_start, D, Delta, K, tm, Vr, tolerance_Iext=.1, reset=True):
  ''' 
  find a trajectory of consecutive cycles 
  -- under linearly INcreasing drive with drive slope m 
  -- with initial condition mu= mu_min_start in the first cycle
  (!) -- with a last cycle ending precisely with drive = Iext_plateau (with error tolerance tolerance_Iext)
  -- start first try with Iext_start in beginning of first cycle and then retry with slightly higher Iext_start, until a matching trajectory is found
  
  Return:
    trajectory : pd data frame with one row per cycle 
  '''
  
  def find_upstroke_trajectory_1trial(df, m, Iext_plateau, Iext_min, Iext_start, mu_min_start, Delta, tolerance_Iext=tolerance_Iext):
    ''' 
    find a trajectory of consecutive cycles 
    -- under linearly INcreasing drive with drive slope m 
    -- with initial condition in the first cycle: mu= mu_min_start and Iext = Iext_start (with error tolerance tolerance_Iext) 
    Stop when the drive in the last cycle exceeds Iext_plateau.
    
    Return:
      trajectory : pd data frame with one row per cycle 
      drive at the end of the last cycle
    '''
    df_m =  df[df.m==m].copy()
    trajectory = pd.DataFrame(columns=list(df.columns) + ['cycle','time_start', 'time_toff', 'time_end'])
    cycle = 0
    while Iext_start < Iext_plateau:
      # print(cycle, end='--')
      # from the systematic parameter exploration in df, find the cycle that matches most closely m, mu_min_start, and Iext_start:
      try:
        ix = find_cycle(df_m, mu_min_start, Iext_start, tolerance_Iext=tolerance_Iext)
      except:
        print(m, cycle, mu_min_start, Iext_start)
        raise ValueError()
      if np.isnan(ix):
        return trajectory, trajectory.iloc[-1]['Iext_end']
      cycle_properties = df_m.loc[ix].to_dict() # extract the properties of this cycle which were already calculated 
      cycle_properties['cycle'] = cycle # add cycle number 
      # add new cycle to trajectory:
      trajectory = trajectory.append(cycle_properties, ignore_index=True)
      # add time stamps 
      if not cycle:
        trajectory.iloc[-1]['time_start'] = 0
      else:
        trajectory.iloc[-1]['time_start'] = trajectory.iloc[-2]['time_end']
      trajectory.iloc[-1]['time_toff'] = trajectory.iloc[-1]['time_start'] + trajectory.iloc[-1]['t_off']
      trajectory.iloc[-1]['time_end'] = trajectory.iloc[-1]['time_toff'] + Delta
      # take the final values of mu and Iext at end of this cycle as start values of next cycle
      cycle += 1
      mu_min_start = trajectory.iloc[-1]['mu_min_end']
      Iext_start = trajectory.iloc[-1]['Iext_end']
    
    return trajectory, trajectory.iloc[-1]['Iext_end'] # return the found trajectory and the final drive at the end of itt
  
  # use the above function repeatedly to find a trajectory that ends precisely at the plateau potential:  
  Iext_final = 1e3
  niter_max = 1e2 # safety exit from while 
  i = 0
  while (np.abs(Iext_final - Iext_plateau) > tolerance_Iext) and (i < niter_max):
    # print('Attempt #', i+1)
    trajectory, Iext_final = find_upstroke_trajectory_1trial(df, m, Iext_plateau, Iext_min, \
                                                      Iext_start = Iext_start, mu_min_start = mu_min_start, Delta=Delta, tolerance_Iext=tolerance_Iext)
    i+=1
    Iext_start += tolerance_Iext/2 # try a slightly higher initial drive 
  
  
  # delete any initial cycles that have a reference drive still below Iext_min (we want to have an asymptotic comparison): 
  if (trajectory.Iext_toff < Iext_min).any():
    trajectory.drop(trajectory.index[trajectory.Iext_toff < Iext_min], inplace=True)
    
    # run the same algorithm one last time with the desired mu_min_start in the initial cycle: 
    # Iext_start slightly shifted to keep the same Iext_toff, hence same Iext_end and rest of trajectory:
    Iext_start = gaussian_drift_approx_transient_1cycle(m, trajectory.iloc[0]['Iext_toff'], mu_min_start, D, Delta, K, tm, Vr, reset=reset)[-2]
    trajectory, Iext_final = find_upstroke_trajectory_1trial(df, m, Iext_plateau, Iext_min, \
                                                    Iext_start = Iext_start, mu_min_start = mu_min_start, Delta=Delta, tolerance_Iext=tolerance_Iext)
  # print('\n', trajectory[['cycle', 'Iext_start', 'Iext_toff', 'Iext_end', 'mu_min_start', 'mu_min_end', 'f_inst']])  
  return trajectory 


def find_downstroke_trajectory(df, m, Iext_plateau, Iext_min, Delta, mu_min_start, tolerance_Iext=.1):
  ''' 
  find a trajectory of consecutive cycles 
  -- under linearly DEcreasing drive with drive slope m 
  -- with initial condition in the first cycle: mu = mu_min_infty(Iext_plateau) 
  Stop when the reference drive in the next cycle would be below Iext_min.
  
  Return:
    trajectory : pd data frame with one row per cycle 
  '''
  # print('Find downstroke trajectory...')
  df_m =  df[df.m == -m].copy()
  trajectory = pd.DataFrame(columns=list(df.columns) + ['cycle','time_start', 'time_toff', 'time_end'])
  cycle = 0
  Iext_start = Iext_plateau # drive starts from plateau potential

  while Iext_start > Iext_min:
    # print(cycle, end='--')
    # from the systematic parameter exploration in df, find the cycle that matches most closely m, mu_min_start, and Iext_toff:
    ix = find_cycle(df_m, mu_min_start, Iext_start, tolerance_Iext=tolerance_Iext)
    if np.isnan(ix): # no more fitting cycles can be found, end of trajectory
      return trajectory
    cycle_properties = df_m.loc[ix].to_dict() 
    cycle_properties['cycle'] = cycle
    # add new cycle to trajectory:
    trajectory = trajectory.append(cycle_properties, ignore_index=True)
    # add time stamps 
    if not cycle:
      trajectory.iloc[-1]['time_start'] = 0
    else:
      trajectory.iloc[-1]['time_start'] = trajectory.iloc[-2]['time_end']
    trajectory.iloc[-1]['time_toff'] = trajectory.iloc[-1]['time_start'] + trajectory.iloc[-1]['t_off']
    trajectory.iloc[-1]['time_end'] = trajectory.iloc[-1]['time_toff'] + Delta
     
    # take the final values of mu and Iext at end of this cycle as starting va
    cycle += 1
    mu_min_start = trajectory.iloc[-1]['mu_min_end']
    Iext_start = trajectory.iloc[-1]['Iext_end']
  return trajectory

def find_cycle(df_m, mu_min_start, Iext_start, tolerance_Iext= .1):
  '''
  From the look-up table of SINGLE-cycle dynamics for slope m and diverse initial conditions and reference drives,
  find the cycle that most closely matches:
    mumin(0) = mu_min_start and
    Iext(0) = Iext_start (with error tolerance tolerance_Iext)

  Returns
  -------
  ix: index of the matching cycle.

  '''
  # resolution of df for initial values mu_min_start 
  d_mumin = np.min(np.diff(np.sort(df_m.mu_min_start.unique())))
  # all cycles with mu_min_start closeby
  ix_mumin = df_m.index[(df_m.mu_min_start - mu_min_start).abs() < d_mumin/2].tolist()
  # of those: the cycle with the closest Iext_start
  ix = (df_m.loc[ix_mumin, 'Iext_start'].astype(float) - Iext_start).abs().idxmin()
  if np.abs(df_m.loc[ix,  'Iext_start']-Iext_start) > tolerance_Iext :
    # print('Best match not good enough: looking for Iext_start={}, found {}, deviation: {}'.format(
    #   Iext_start, df_m.loc[ix,  'Iext_start'], df_m.loc[ix,  'Iext_start']-Iext_start))
    return nan
  else:
    return ix


#%% Gaussian-drift approx. numerical (DDE)

def integrate_dde_numerically(Iext, D, Delta, K, tm, Vt=1 , Vr = 0, tmax = 200, dt = 0.001, mu0=nan, reset= False, plot=True, rtol_conv=1e-2, \
                              keep_initial_history=False, illustrate_convergence=False):
  '''
  Numerical integration of DDE (Eq 30/31)

  Parameters
  ----------
  Iext : float. External drive.
  D, Delta, K, tm , Vr : Network parameters.
  Vt : float, optional. Spike threshold. The default is 1.
  tmax : float, optional. Simulation time. The default is 200.
  dt : float, optional. Time step. The default is 0.001.
  mu0 : float or np.array, optional. Initial value of mu at time 0 OR: entire history of mu during times [-Delta,0]. The default is nan.
  reset : bool, optional. Whether or not to add reset condition. The default is False.
  plot : bool, optional. Activate result plot. The default is True.
  rtol_conv : float, optional. Relative error tolerance (between consecutive mumax, mumin values) for diagnosing convergence. The default is 1e-2.
  keep_initial_history : bool, optional. Whether to keep the initial condition for [-Delta,0] in the result arrays. The default is False.
  illustrate_convergence : bool, optional. Whether to plot the convergence of the change in mumin/mumax over cycles. The default is False.

  Returns
  -------
  t : np.array of time points.
  mu : np.array. Mean membrane potential over time.
  r : np.array. Rate over time.
  ix_max : np.array. Indices of times toff in each cycle (when mu reaches its local maxima)
  ix_min_real : np.array. Indices of times when mu reaches its true local minima.
  mu_max : np.array. Local maxima mumax for each cycle.
  mu_reset : np.array. Reset value for each cycle.
  mu_min_real : np.array. Real local minima for each cycle.
  ix_min_theory : np.array. Indices of theoretical mumin (time Delta after each ix_max)
  mu_min_theory : np.array. Value of mu at time Delta after each mumax (estimate for mumin in Gaussian-drift approx)
  toff_theory : np.array. Time points when mumax is reached in each cycle.
  fig , ax : result plot.
  '''
  print('Integrating DDE numerically...')
  
  ### initialize 
  ndelay = int(Delta/dt) # synaptic delay in time steps 
  reset_clip = 1 # way to artificially clamp rate to 0 right after reset
  
  # match time array to input or vice versa (for transient input)
  if np.isscalar(Iext): # constant drive
    constant_drive=True
    t = np.arange(0, tmax, dt)
    Iext = np.ones_like(t)*Iext 
  else: # time-dependent drive 
    constant_drive=False
    tmax = Iext.size*dt
    t = np.arange(0, tmax, dt)
  
  # --- initialize result arrays
  t = np.concatenate((t, np.arange(-Delta, 0, dt) )) # add times before time 0 at the END of the time array
  T = len(t) 
  mu, mudot, r, Iinh, I = np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T)
  # set initial values:
  if np.isscalar(mu0):
    if np.isnan(mu0):
      mu0 = Vt - 6*np.sqrt(D) # if no initial value provided, start with the full gauss far below threshold
    elif mu0 > Vt - 3*np.sqrt(D):
      print('Note that initial value mu0={:.2f} is NOT consistent with the assumption of r(t<0) = 0!'.format(mu0) )
    mu[0] = mu0
    mu[-ndelay:] = mu0    
  elif len(mu0) == ndelay+1: 
    # non-zero initial values provided, insert at the END of the storage arrays (array will be rolled forward accordingly after the integration)
    mu[0] = mu0[-1]
    mu[-ndelay:] = mu0[:-1] # history of mu before time 0
    # infer the history of the slope mudot: 
    mudot[-ndelay: -1] = ( mu[-ndelay + 1: ] -  mu[-ndelay: -1]) / dt
    mudot[-1] = (mu[0]-mu[-1]) / dt
    # infer rate history:
    r[-ndelay:] =  np.clip(mudot[-ndelay:],0,None)*get_gauss(Vt, mu[-ndelay:], np.sqrt(D), broadcast=False) # population rate
    # infer current history: 
    I[-ndelay:] = tm*mudot[-ndelay:] + mu[-ndelay:]
    Iinh[-ndelay:] = np.nan # not relevant and not computed here, would depend on second order rate history.. 
  else:
    raise ValueError('Either provide scalar intial value mu0 < Vt-3sqrt(D), or provide full history array of length ndelay+1!')
  
  # also record discrete cycle data: mumin, mu_max, mureset
  mu_max, mu_reset, ix_max = [], [], [] # local maxima and index to retrieve them
  mu_min_real, ix_min_real = [mu[0]], [0] # local minima and index to retrieve them
  
  # --- numerical Euler integration over time 
  for i in range(int(tmax/dt)):
    # --- integrate next time step
    Iinh[i] = tm*K*r[i-ndelay] # inh feedback
    I[i] = Iext[i] - Iinh[i] # total input current
    mudot[i] = (I[i]-mu[i])/tm # derivative of mu
    r[i] = np.clip(mudot[i],0,None)*get_gauss(Vt, mu[i], np.sqrt(D))*reset_clip # population rate
    
    if i<t.size-1: # integrate next time step
      mu[i+1] = mu[i] + mudot[i]*dt # mu in next time step
      
      # --- check if local max/min was reached
      if i>1:
        is_max = (mu[i] > mu[i-1]) & (mu[i] > mu[i+1]) # mu has reached local maximum at step i
        is_min = (mu[i] < mu[i-1]) & (mu[i] < mu[i+1]) # mu has reached local minimum at step i
        # make sure a reset at step i, followed by short increase, is not counted as a minimum:
        if reset and len(ix_max):
          if i-1 == ix_max[-1]: # reset just happened mu[i] = mu_reset
            is_min = False # do not count reset as minimum
        # record relevant variables
        if is_max:
            if not reset:
              # simply record the local maximum
              mu_max.append(mu[i])
              ix_max.append(i)
            elif reset and reset_clip: 
              '''
              mu has reached an actual maximum mu_max
              record local maximum and perform reset, make sure rate stays at 0 despite the short rise in mu after the reset
              '''
              mu_max.append(mu[i])
              ix_max.append(i)
              mu[i+1] = get_mu_reset(mu[i], D, Vt=Vt, Vr=Vr)
              mu_reset.append(mu[i+1])
              reset_clip = 0 # rate will be clipped to 0 until mu reaches its next maximum
            elif reset and (not reset_clip):
              '''
              mu has just reached its second max (after reset) and will now decay again
              do not record this as a mu_max
              just release the rate from its artificial clip to 0
              '''
              reset_clip = 1
        elif is_min: 
            # mu has reached its true minimum, record:
            mu_min_real.append(mu[i])
            ix_min_real.append(i)
  
  # conversion to numpy arrays and rate into Hz
  mu_min_real, mu_max, mu_reset, ix_min_real, ix_max = np.array(mu_min_real), np.array(mu_max), np.array(mu_reset), np.array(ix_min_real), np.array(ix_max)
  r = 1000*r # Hz
  
  # the theoretical mu_min happens precisely delta after the maximum:
  ix_min_theory = np.append([0], ix_max+ndelay).astype(int)
  ix_min_theory = ix_min_theory[ix_min_theory< int(tmax/dt)] # keep only indices that lie inside the simulated time
  mu_min_theory = mu[ix_min_theory] # record the mu_min as defined in the theory ( mu(toff+Delta) ) 
  toff_theory = (ix_max - ix_min_theory[:len(ix_max)])*dt # length of upstroke, ms
  # --- plot everything unfiltered
  past = t<0 
  present = t>=0
  if plot:    
    fig, ax = plt.subplots(5 + 2*(constant_drive and illustrate_convergence), sharex=True, figsize=(width_a4_wmargin*.5, width_a4_wmargin*.6))
    despine(ax)
    # population rate
    for ix in [past, present]:
      ax[0].plot(t[ix], r[ix], color=my_colors['fnet'])
    ax[0].set_ylim(bottom=0)
    ax[0].set_ylabel('r(t)\n[spk/s]')
    
    # mean membrane potential mu
    gridline([Vr,1], [ax[1], ax[3]], axis='y')
    for ix in [past, present]:
      ax[1].plot(t[ix], mu[ix], 'k')
      ax[1].plot(t[ix], mu[ix]-3*np.sqrt(D), 'k--')
      ax[1].plot(t[ix], mu[ix]+3*np.sqrt(D), 'k--')
    # mark real local maxima, minima:
    if len(mu_max) and len(mu_min_real)>1:
      ax[1].plot(ix_max*dt, mu_max, '.', color=my_colors['max'])
      ax[1].plot(ix_min_real*dt, mu_min_real, '.', color=my_colors['min']) # real local minimum
      if reset:
        ax[1].plot(ix_max*dt, mu_reset, '.', color=my_colors['reset'])
      # mark theoretical mu_min:
      # for axi in ax:
      for i in ix_min_theory*dt:
        ax[1].axvline(i, color=my_colors['min'], lw=.5, linestyle='--')
    ax[1].set_ylabel(r'$\mu(t)$')
    
    # derivative of mu
    gridline([0], ax[2], axis='y')
    for ix in [past, present]:
      ax[2].plot(t[ix], mudot[ix], 'k')
    ax[2].set_ylabel(r'$\dot{\mu}(t)$')
    
    # mu compared to total input I: illustrate intersections 
    for ix in [past, present]:
      ax[3].plot(t[ix], mu[ix], 'k', label=r'$\mu(t)$' if (ix == present).all() else '')
      ax[3].plot(t[ix], I[ix], 'k--', label=r'$I(t)$' if (ix == present).all() else '')
    ax[3].set_ylabel('I, $\mu$')
    ax[3].legend(ncol=2, handlelength=1.5)
    
    # external drive
    gridline([0], ax[-2:], axis='y')
    ax[-1].plot(t[present], Iext, color=my_colors['iext'], label=r'$I_E$')
    for ix in [past, present]:
      ax[-1].plot(t[ix], -Iinh[ix], 'b', label=r'$I_\mathrm{inh}$' if (ix == present).all()  else '')
      ax[-1].plot(t[ix], I[ix], 'k--', label=r'$I(t) = I_E - I_\mathrm{inh}$' if (ix == present).all()  else '')
    ax[-1].set_xlabel('time [ms]')
    ax[-1].set_ylabel('input')
    ax[-1].set_xlim([np.min(t), np.max(t)])
    ax[-1].legend(ncol=3, handlelength=1.5)
    
    if constant_drive and illustrate_convergence:
      # illustrate convergence of mu_max, mu_min across cycles:
      if len(mu_max) and len(mu_min_real)>1:
        # absolute distance from asymptotic value
        ax[4].plot(ix_max*dt, mu_max - mu_max[-1], '.', color= my_colors['max'])
        ax[4].plot(ix_min_real[1:]*dt, mu_min_real[1:] - mu_min_real[-1], '.', color= my_colors['min'])
        ax[4].set_ylabel(r'$\Delta_\mathrm{abs}$')
        
        # relative distance from asymptotic value
        ax[5].plot(ix_max*dt, (mu_max - mu_max[-1])/mu_max[-1], '.', color= my_colors['max'])
        ax[5].plot(ix_min_real[1:]*dt, (mu_min_real[1:] - mu_min_real[-1])/mu_min_real[-1], '.', color= my_colors['min'])
        gridline([-rtol_conv, rtol_conv], ax[5], 'y')
        ax[5].set_ylabel(r'$\Delta_\mathrm{rel}$')

  else:
    fig, ax = [], []
  if not keep_initial_history:
    # cut off the end of the arrays, where the history for t in [-Delta,0] was recorded.
    t = t[present]
    mu = mu[present]
    r = r[present]
  return t, mu, r, ix_max, ix_min_real, mu_max, mu_reset, mu_min_real, ix_min_theory, mu_min_theory, toff_theory, fig, ax
      
def integrate_dde_numerically_until_convergence(Iext, D, Delta, K, tm, Vr, reset=False, mu0=0, dt=0.01, tmax = 0, plot=False, \
                                              ncyc_convergence = 100, rtol_convergence=1e-3, return_last_cycle=False):
  ''' 
  Integrate DDE numerically until convergence is reached. If necessary extend simlation time in steps of 1sec.
  only relevant for constant drive
  '''
  convergence = False
  while not convergence:
    tmax += 1000
    print('increasing tmax: '+str(tmax))
    # integrate DDE numerically over time tmax
    t, mu, r, ix_max, ix_min_real, mu_max, mu_reset, mu_min_real, ix_min_theory, mu_min_theory, toff_theory \
    = integrate_dde_numerically(Iext, D, Delta, K, tm, Vr = Vr, tmax = tmax, dt = dt, mu0=mu0, reset= reset, plot=plot)[:-2]
    # check for convergence 
    state, convergence = analyze_dde_dynamics_numerical(t, mu, r, ix_max, ix_min_real, mu_max, mu_reset, mu_min_real, Delta, \
                                                        ncyc_convergence = ncyc_convergence, rtol_convergence = rtol_convergence)
    print(state, convergence)
    
  if return_last_cycle:
    t, mu, r, mu_min_theory, mu_max, mu_reset, toff_theory, mu_min_real \
    = dde_extract_last_cycle(t, mu, r, ix_min_theory, mu_max, mu_min_theory, mu_min_real, mu_reset, toff_theory, reset_time = True)
  return t, mu, r, ix_max, ix_min_real, mu_max, mu_reset, mu_min_real, ix_min_theory, mu_min_theory, toff_theory, state, convergence
   

def analyze_dde_dynamics_numerical(t, mu, r, ix_max, ix_min, mu_max, mu_reset, mu_min, Delta, ncyc_convergence = 10, rtol_convergence =1e-2):
  '''
  analysis of the result of numerical DDE integration under constant drive 
  check for convergence of the dynamics, and if they have converged, define the network state:
    (1) 'fp': stable fixed point 
    (2) 'pathological': fast, low-amplitude oscillations of period 2Delta
    (3) 'period2': period-2 oscillations
    (4) 'period1': regular, period-1 oscillations
  '''
  state = nan
  dt = np.mean(np.diff(t))
  if (len(ix_min) < 2):
    # there was no local minimum except for the initial value 
    state = 'fp'
  elif (t.size - ix_min[-1]) > 2*np.mean(np.diff(ix_min)) : 
    # there were some initial local minima, but then no more oscillations for a time much longer than the periods of the initial transients
    state = 'fp'
  else: # some oscillatory state
    if np.isclose(mu_min[-2], mu_min[-1], rtol=1e-2): # last two local minima were "identical" (1% tolerance)
      state = 'period1'
      if np.diff(ix_min[-2:])*dt <= 2*Delta: # last period was shorter or equal 2*syn. delay
        state = 'pathological'
    elif np.isclose(mu[ix_min[-3]], mu[ix_min[-1]], rtol=1e-2): # last and third last local minima were "identical" (1% tolerance)
      state = 'period2'
    else:
      raise ValueError('So far this case never happened: No other dynamics?')
  # check if dde has really converged to the presumable state: 
  convergence = check_convergence(state, mu_max, mu_min, mu, ncyc_convergence, rtol=rtol_convergence)
  return state, convergence 

def check_convergence(state, mu_max, mu_min, mu, ncyc_conv, rtol=1e-2):
  '''
  Check if the numerical integration of the DDE has converged by the following definition:
    stable fixed point: mu has been (rtol-)close to its final value for the last quarter of the simulation time.
    oscillations: the last ncyc_conv cycles had the same local minima and maxima, except for relative deviations of rtol

  Parameters
  ----------
  state : Suspected dynamical state (fixed point, period 1/2 oscillations, pathological)
  mu_max, mu_min, mu: np.arrays. Results of numerical integration of DDE.
  ncyc_conv : int. Number of cycles with stable dynamics required to diagnose "convergence"
  rtol : float, optional. Tolerance for numerical deviations. The default is 1e-2.

  Returns
  -------
  convergence: bool. Whether convergence has been reached.

  '''
  if state == 'fp':
    # no change in mu in the last quarter of the simulation time (convergence to fixed point)
    convergence = np.isclose(mu[-int(mu.size/4):], mu[-1], rtol=rtol).all()
    if not convergence:
      print('largest relative divergence of mu from fixed point: ' , np.max((mu[-int(mu.size/4):] - mu[-1])/mu[-1]) , '\nPresumable fixed point: ', mu[-1]) 
  else:
    if mu_max.size < ncyc_conv:
      print('simulate longer to check convergence (at least {} cycles)'.format(ncyc_conv))
      return False 
    if state in ['period1', 'pathological']:
      # check that mu_max and mu_mmin have not changed significantly over the last ncyc_conv cycles 
      convergence = np.isclose(mu_max[-ncyc_conv:], mu_max[-1], rtol=rtol).all() and np.isclose(mu_min[-ncyc_conv:], mu_min[-1], rtol=rtol).all()
    elif state == 'period2':
      # check that the alternating mu_max and mu_mmin have not changed significantly over the last ncyc_conv cycles 
      convergence = np.isclose(mu_max[-1:-ncyc_conv:-2], mu_max[-1], rtol=rtol).all() and \
                    np.isclose(mu_min[-1:-ncyc_conv:-2], mu_min[-1], rtol=rtol).all() and \
                    np.isclose(mu_max[-2:-ncyc_conv:-2], mu_max[-2], rtol=rtol).all() and \
                    np.isclose(mu_min[-2:-ncyc_conv:-2], mu_min[-2], rtol=rtol).all() 
  
  return convergence


def dde_extract_last_cycle(t, mu, r, ix_min, mu_max_all, mu_min_all, mu_min_real_all, mu_reset_all, t_off_all, reset_time = True):
  ''' Extract last cycle of the oscillation dynamics after a numerical integration of the DDE ''' 
  cycle_start, cycle_end = ix_min[-2], ix_min[-1]+1 # include endpoint
  cycle_end = np.min([cycle_end, t.size]) # restrict to simulation time 
  t0 = t[cycle_start] # start time of last complete cycle
  
  # extract cycle properties:
  mu_min = mu_min_all[-1]
  mu_min_real = mu_min_real_all[-1]
  if len(mu_min_all) == len(mu_max_all) + 1 : # one more local minimum than maxima 
    mu_max = mu_max_all[-1]
    if len(mu_reset_all):
      mu_reset = mu_reset_all[-1]
    else:
      mu_reset = nan
    t_off = t_off_all[-1]
  elif len(mu_min_all) == len(mu_max_all): # the last cycle was incomplete, ie don't count the last maximum:
    mu_max = mu_max_all[-2]
    if len(mu_reset_all):
      mu_reset = mu_reset_all[-2]
    else:
      mu_reset = nan
    t_off = t_off_all[-2]
  
  return t[cycle_start:cycle_end]-t0*reset_time, mu[cycle_start:cycle_end], r[cycle_start:cycle_end], mu_min, mu_max, mu_reset, t_off, mu_min_real
  

def gaussian_drift_approx_stat_numerical(Iext, D, Delta, K, tm, Vt=1, Vr=0, reset=False, plot=False, tmax = 400, \
                                         ncyc_convergence = 100, rtol_convergence=1e-3): 
  '''
  Gaussian-drift approximation for constant drive, based on NUMERICAL integration of DDE (for comparison with analytical results)

  Parameters
  ----------
  Iext : externald drive. 
  D, Delta, K, tm , Vr : Network parameters.
  reset : bool, optional. Reset. The default is False.
  plot : bool, optional. Activate result plot. The default is False.
  tmax : float, optional. Initial simulation time (will automatically be extended until convergence is reached). The default is 400.
  ncyc_convergence : int, optional. Number of periodic cycles required for convergence. The default is 100.
  rtol_convergence : float, optional. Tolerance for numerical deviations. The default is 1e-3.

  Returns
  -------
  mu_min, mu_max, f_net, f_unit, zeta, t_on, t_off, mu_reset : Characterization of oscillation dynamics.
  mu_min_real : REAL local minimum of mean membrane potential (deviates from mu_min)
  fig, ax : plot.
  '''
  Iext, D, Delta, K, tm, Vr = np.broadcast_arrays(Iext, D, Delta, K, tm, Vr)   # only to avoid problems when using this within performance check
  
  mu_min, mu_max, f_net, t_on, t_off, mu_reset, mu_min_real \
  = np.ones(Iext.size)*nan, np.ones(Iext.size)*nan, np.ones(Iext.size)*nan, np.ones(Iext.size)*nan, \
    np.ones(Iext.size)*nan, np.ones(Iext.size)*nan, np.ones(Iext.size)*nan
  for i in range(Iext.size):
    print('{}/{}'.format(i+1, Iext.size))
    t, mu, r, ix_max_all, ix_min_real_all, mu_max_all, mu_reset_all, mu_min_real_all, ix_min_theory_all, mu_min_all, t_off_all, state, convergence\
    = integrate_dde_numerically_until_convergence(Iext[i], D[i], Delta[i], K[i], tm[i], Vr=Vr[i], reset=reset, plot=False, \
                                                  tmax=tmax, ncyc_convergence = ncyc_convergence, rtol_convergence = rtol_convergence)  
    if state == 'period1':
      mu_max[i], mu_min_real[i], mu_min[i], t_off[i] \
      = mu_max_all[-1], mu_min_real_all[-1], mu_min_all[-1], t_off_all[-1]
      if reset:
        mu_reset[i] = mu_reset_all[-1]
      # restrict traces to last cycle only:
      t, mu, r = dde_extract_last_cycle(t, mu, r, ix_min_theory_all, mu_max_all, mu_min_all, mu_min_real_all, mu_reset_all, t_off_all)[:3]
      t_on[i] = t[np.where(mu+3*np.sqrt(D[i])>Vt)[0][0]]
      f_net[i] = get_fnet(t_off[i], Delta[i])
  sat = get_saturation(mu_max, D, Vt=Vt)
  f_unit = get_funit(f_net, sat)
  zeta = get_popsynch(t_on, t_off, sat, f_net)
    
  if plot:
    fig, ax = plot_gaussian_drift_approx(Iext, f_net, f_unit, mu_min, mu_max, zeta, t_on, t_off, mu_reset, np.unique(Delta)[0], \
                                      Vt=Vt, linestyle=':', marker='', label='(numerical)', reset=reset)
  else:
    fig, ax = [], []
  return mu_min, mu_max, f_net, f_unit, zeta, t_on, t_off, mu_reset, mu_min_real, fig, ax


def integrate_1_cycle_wlineardrive_numerically(D, Delta, K, tm, Vr, m, mu_min_0, Iext_toff=nan, Iext_0=nan, \
                                               reset=True, dt=.01, tmax = 10, plot=False):    
  t = np.arange(0, tmax, dt)
  if np.isnan(Iext_0):
    if not m: 
      Iext_0 = Iext_toff
    else:
      # find initial value that will lead approx to the same Iext_toff
      mu_max = get_mumax_transient(m, Iext_toff, D, Delta, K, tm)
      t_off = get_toff_transient(m, Iext_toff, mu_max, mu_min_0, tm)
      Iext_0 = get_linear_drive(-t_off, m, Iext_toff)
    
  Iext = get_linear_drive(t, m, Iext_0)
  
  t_all, mu_all, r_all, _, _, mu_max_all, mu_reset_all, _, ix_min_theory_all, mu_min_theory_all, toff_theory_all \
  = integrate_dde_numerically(Iext, D, Delta, K, tm, Vr = Vr, dt = dt, tmax = tmax, mu0=mu_min_0, reset= reset, plot=plot)[:-2]
  d = {}
  # cut out first cycle
  end_of_first_cycle = ix_min_theory_all[1]
  d['t'], d['mu'], d['r'], d['Iext'] \
  = t_all[:end_of_first_cycle], mu_all[:end_of_first_cycle], r_all[:end_of_first_cycle], Iext[:end_of_first_cycle]
  
  d['mu_max'] = mu_max_all[0]
  d['mu_reset'] = mu_reset_all[0]
  d['mu_min'] = mu_min_theory_all[:2]
  d['t_off'] = toff_theory_all[0]
  # find Iext_toff
  d['Iext_toff'] = d['Iext'][np.argmax(d['mu'])]
  
  return d

def dde_bifurcation_analysis_numerical(D, Delta, K, tm, Vr, dI = 0.01, dt = 0.001):
  '''
  Numerical bifurcation analysis of DDE, assuming subthreshold initial conditions (Fig 8).

  Parameters
  ----------
  D, Delta, K, tm , Vr : Network parameters.
  dI : float, optional. Resolution in space of external drive. The default is 0.01.
  dt : float, optional. Numerical integration time step. The default is 0.001.

  Returns
  -------
  I_bifurcation : bifurcation point between stable fixed point and pathological oscillations.
  Imin_p2 : bifurcation point between pathological and period-2 oscillations.
  Imin_p1 : bifurcation point between period-2 and regular, period-1 oscillations.
  '''
  
  mu0 = np.min([0, 1-5*np.sqrt(D)])
  # boundaries of region of interest 
  Ifull = get_pt_fullsynch(D, Delta, K, tm)
  
  # find boundary between period2 and period 1
  print('\nFind transition between period2 and period1 oscillations...')
  Imin_p1 = find_transition('period2', 'period1', 0, Ifull, dI, D, Delta, K, tm, Vr, mu0=mu0, dt=dt)
  
  # find boundary between pathological and period2
  print('\nFind transition between pathological and period2 oscillations...')
  Imin_p2 = find_transition('pathological', 'period2', 0, Imin_p1, dI, D, Delta, K, tm, Vr, mu0=mu0, dt=dt)
  
  # find boundary between pathological and period2
  print('\nFind transition between fixed point and pathological oscillations...')
  I_bifurcation = find_transition('fp', 'pathological', 0, Imin_p2, dI, D, Delta, K, tm, Vr, mu0=mu0, dt=dt)
  
  
  return I_bifurcation, Imin_p2, Imin_p1


def find_transition(left, right, Imin, Imax, dI, D, Delta, K, tm, Vr, mu0, dt):
  ''' numerically integrate the DDE until the boundary between state "left" and state "right" is found up to a precision of dI
  '''
  while Imax-Imin > dI:
    # start in the middle
    I = np.mean([Imin, Imax])
    print('-------------------------------------------')
    print('I= ',I)
    
    state = integrate_dde_numerically_until_convergence(I, D, Delta, K, tm, Vr, mu0=mu0, dt=dt)[-2]
    
    if state == left:
      #new Imin:
      Imin = I
    else:
      # new Imax: 
      Imax = I
  return np.mean([Imin, Imax])






