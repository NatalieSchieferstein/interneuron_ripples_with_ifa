#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 17:31:20 2023

@author: natalie

This script prepares, runs and analyzes all simulations needed to produce all figures of the manuscript.
It relies on functions defined in methods_simulations.py and methods_analyze_simulations.py.

To produce the figures, run main_plot_figures.py.
"""

from matplotlib import font_manager
import numpy as np
import os
import pandas as pd

from methods_simulations import pypet_get_trajectoryPath, pypet_makeConfig_gauss_lif, pypet_makeConfig_lif_ifa_fromcyclostat, pypet_makeConfig_performance_check, \
  pypet_make_config_brunel_hakim99, pypet_run_simulation, pypet_setup_config, store_info_cyclostat_lif_simulation
from methods_analyze_simulations import do_ifa_analysis, performance_check_evaluation, pypet_analyse_membrane_potential_dynamics

path_to_simulations = './simulations/'

# check / create data paths
if not os.path.exists(path_to_simulations):
  os.makedirs(path_to_simulations)
  
# add Arial font path
try:
  font_path_arial = '/home/natalie/anaconda3/envs/{}/lib/python3.6/site-packages/matplotlib/mpl-data/fonts/ttf/Arial.ttf'.format(os.environ['CONDA_DEFAULT_ENV'])
  font_path_arial_bold = '/home/natalie/anaconda3/envs/{}/lib/python3.6/site-packages/matplotlib/mpl-data/fonts/ttf/Arialbd.ttf'.format(os.environ['CONDA_DEFAULT_ENV'])
  font_manager.fontManager.addfont(font_path_arial)
  font_manager.fontManager.addfont(font_path_arial_bold)
except:
  try:
    font_path_arial = '/home/schieferstein/anaconda3/envs/{}/lib/python3.6/site-packages/matplotlib/mpl-data/fonts/ttf/Arial.ttf'.format(os.environ['CONDA_DEFAULT_ENV'])
    font_path_arial_bold = '/home/schieferstein/anaconda3/envs/{}/lib/python3.6/site-packages/matplotlib/mpl-data/fonts/ttf/Arialbd.ttf'.format(os.environ['CONDA_DEFAULT_ENV'])
    font_manager.fontManager.addfont(font_path_arial)
    font_manager.fontManager.addfont(font_path_arial_bold)
  except:
    print('Path to Arial font could not be added, maybe provide correct path! Current path: ', font_path_arial)

#%% Small example simulation h0 for Fig 1 (only network size N=10_000)

traj_hash = 0
comment='Example simulation of Figure 1 type (N=10,000 only for different levels of constant drive).'
parameters={'D':0.04, 'Delta':1.2, 'K':5, 'tm':10, 'Vr':0}
pypet_makeConfig_gauss_lif(parameters, Nint=10_000, comment=comment, traj_hash = traj_hash,  v_recordwindow = [5000., 5050.], path_to_simulations = path_to_simulations) # setup configuration file
pypet_run_simulation(traj_hash = traj_hash, path_to_simulations = path_to_simulations) # run simulation
pypet_analyse_membrane_potential_dynamics(traj_hash, path_to_simulations = path_to_simulations) # analyse membrane potential dynamics                               
  
  
#%% Simulations h1 for Figure 1 
traj_hash = 1
comment='Figure 1: Spiking network dynamics for different levels of constant drive (including variation of network size Nint).'
parameters={'D':0.04, 'Delta':1.2, 'K':5, 'tm':10, 'Vr':0}
pypet_makeConfig_gauss_lif(parameters, Nint=1000, do_analysis_membrane_potential_dynamics = True, forExploration={'Nint':[100,1000,10000]}, comment=comment, traj_hash = traj_hash, path_to_simulations = path_to_simulations)  # setup configuration file
pypet_run_simulation(traj_hash = traj_hash, path_to_simulations = path_to_simulations) # run simulation


#%% Simulations h2 for Figure 2 (and 7)
traj_hash = 2 
comment='Vary ramp times to show: IFA depends on input slope. Slopes m=.1, .2, .4 to match theory. Vary network size to show that finite size effects dont matter much (Fig 2 of IFA paper).'
pypet_makeConfig_lif_ifa_fromcyclostat(traj_hash_stat=1, traj_hash=traj_hash, plateau_time = 20, slope_dimless = [.4, .2, .1],  Nint=[100, 1000, 10000], nreps=50, comment=comment, path_to_simulations =  path_to_simulations)  # setup configuration file
pypet_run_simulation(traj_hash = traj_hash, path_to_simulations = path_to_simulations) # run simulation
do_ifa_analysis(traj_hash=traj_hash, path_to_simulations = path_to_simulations) # analyze population rate for IFA
  
#%% Simulations for  Supplementary Figures S1A, S1B
traj_hash = 3 # 246
# parameter range to explore:
exploration = {'D': np.r_[.01, 0.0225, .04, 0.0625, .09, .16], 'K': np.r_[2, 3, 5, 10, 50], 'tm':10, 'Vr':0, 'Delta':1.2}
pypet_makeConfig_performance_check(exploration, traj_hash, Nint=25_000, n_sim=15, do_membrane_potential_dynamics_analysis=True, path_to_simulations = path_to_simulations)  # setup configuration file
pypet_run_simulation(traj_hash = traj_hash, path_to_simulations = path_to_simulations) # run simulation
performance_check_evaluation(traj_hash, reset=True, numerical_check=False, path_to_simulations=path_to_simulations) # compare simulation results to Gaussian-drift approximation:

#%% Supplementary Figure S2C
# CONSTANT DRIVE, different networks
## 1001 ---  network from Donoso et al. 2018 Fig 1
traj_hash = 1001
commentTraj = 'Donoso et al. 2018 default network under constant drive'
input_params = {'Tsim': 5050, 'shape': 'flat', 'level': 0.0, 'unit':'spks/sec'}
aux_params = {'analysis.unit': True, 'analysis.net': True, 'postproc.par0': 'parameters.input.level', 'postproc.v_recordwindow': [5025., 5050.]}
forExploration = {'level':  [2000., 2500., 3000., 3500., 4000., 5000., 6000.] + list(np.arange(8000., 16000., 2000.)) }
pypet_setup_config('Donoso2018', {}, forExploration, input_params, aux_params, commentTraj, cart_product=True, traj_hash=traj_hash, path_to_simulations = path_to_simulations) # setup configuration file
pypet_run_simulation(traj_hash = traj_hash, path_to_simulations = path_to_simulations) # run simulation

## 1002 ---  network from Donoso et al. 2018 but with INDEPENDENT Poisson inputs
traj_hash = 1002
commentTraj = 'Donoso et al. 2018 network with independent Gaussian white noise input (noise increases with drive as for Poisson input) under constant drive'
changeParams = {'inputmode':'DiffApprox'}
input_params = {'Tsim': 5050, 'shape': 'flat', 'level': 0.0, 'unit':'spks/sec'}
aux_params = {'analysis.unit': True, 'analysis.net': True, 'postproc.par0': 'parameters.input.level', 'postproc.v_recordwindow': [5025., 5050.]}
forExploration = {'level':  [2000., 2500., 3000., 3500., 4000., 5000., 6000.] + list(np.arange(8000., 16000., 2000.)) }
pypet_setup_config('Donoso2018', changeParams, forExploration, input_params, aux_params, commentTraj, cart_product=True, \
                    traj_hash=traj_hash, path_to_simulations = path_to_simulations) # setup configuration file
pypet_run_simulation(traj_hash = traj_hash, path_to_simulations = path_to_simulations) # run simulation

## 1003 ---  default reduced model with refrac period ------------------------------------------------------------------------------
traj_hash = 1003
commentTraj = 'Reduced model with refractory period (+ larger noise & delay to ensure unimodal voltage distributions, Nint=200 for comparability with Donoso2018). Constant drive.'
parameters={'D':0.2, 'Delta':1.5, 'K':5, 'tm':10, 'Vr':0}
pypet_makeConfig_gauss_lif(parameters, Nint=200, tref=1, do_analysis_membrane_potential_dynamics = False, \
                            comment=commentTraj, traj_hash = traj_hash, path_to_simulations = path_to_simulations) # setup configuration file
pypet_run_simulation(traj_hash = traj_hash, path_to_simulations = path_to_simulations) # run simulation

## 1004 --- Reduced network as in manuscript -----------------------------------------------------------------------
traj_hash = 1004
commentTraj = 'Reduced model as in Fig 1 (Nint=200 for comparability with Donoso2018). Constant drive.'
parameters={'D':0.04, 'Delta':1.2, 'K':5, 'tm':10, 'Vr':0}
pypet_makeConfig_gauss_lif(parameters, Nint=200, do_analysis_membrane_potential_dynamics = False, \
                            comment=commentTraj, traj_hash = traj_hash, path_to_simulations = path_to_simulations) # setup configuration file
pypet_run_simulation(traj_hash = traj_hash, path_to_simulations = path_to_simulations) # run simulation


# TRANSIENT DRIVE, different networks
# parameters for IFA simulations:
nreps = 30
Tinit, Tedge, t_pad = 200., 20., 20.
plateau_time = 5
ramp_time_up    = [20, 10, 30]
ramp_time_down  = [20, 30, 10]
aux_params = {'analysis.unit': True, 'analysis.net': True, 'analysis.offset': 50, 'analysis.getInstFreq':True, \
              'analysis.ifreq_baselinewindow':[50, 150], 'analysis.fmin': 70, 'analysis.fmax': 400}
input_params = {'unit':'spks/sec', 'shape' : 'ramp_asym', 'Tinit': Tinit, 'baseline': 500, 'peak': 14_000, 'plateau_time': plateau_time}
forExploration = {} # set up parameter explorations
forExploration['ramp_time_up'] = np.repeat(ramp_time_up, nreps)
forExploration['ramp_time_down'] = np.repeat(ramp_time_down, nreps)
forExploration['simulation.seed_brian'] = list(np.random.randint(2**32-1, size=nreps*len(ramp_time_up))) # vary the seed to simulate the same network and input configuration for multiple noise realizations
# --- add explorations that derive from the ramp_time exploration
forExploration['Tsim'] = list(Tinit + np.array(forExploration['ramp_time_up']) + np.array(forExploration['ramp_time_down']) + plateau_time + 2*Tedge + 2*t_pad)
forExploration['analysis.ifreq_targetwindow'] = [[Tinit + t_pad, Tsim - t_pad] for Tsim in forExploration['Tsim']]
forExploration['postproc.v_recordwindow'] = forExploration['postproc.v_recordwindow_for_simulation'] = forExploration['analysis.ifreq_targetwindow'] 
# --- enter default values for the explored parameters
input_params['ramp_time_up'], input_params['ramp_time_down'] = np.array(forExploration['ramp_time_up']).dtype.type(0), np.array(forExploration['ramp_time_down']).dtype.type(0)
input_params['Tsim'] =  np.array(forExploration['Tsim']).dtype.type(0)

# --- network from Donoso et al. 2018 Fig 1 ---------------------------------------------------------------------------------
traj_hash=1006
commentTraj = 'Donoso et al. 2018 default network: double-ramp drive, symmetric or asymmetric, to check IFA (compare to h1001).'
pypet_setup_config('Donoso2018', {}, forExploration, input_params, aux_params, commentTraj, cart_product=False, \
                    traj_hash=traj_hash, path_to_simulations = path_to_simulations, record_per_run={'runs': [0], 'record':['v']}) # setup configuration file
pypet_run_simulation(traj_hash = traj_hash, path_to_simulations =path_to_simulations) # run simulation
do_ifa_analysis(traj_hash=traj_hash, save2file=True, path_to_simulations =path_to_simulations) # analyze the population rate dynamics and check for IFA 


# --- network from Donoso et al. 2018 but with INDEPENDENT Poisson inputs ----------------------------------------------------
traj_hash=1007
changeParams = {'inputmode':'DiffApprox'}
commentTraj = 'Donoso et al. 2018 network with independent Gaussian white noise input (noise increases with drive as for Poisson input): double-ramp drive, symmetric or asymmetric, to check IFA (compare to h1002)'
pypet_setup_config('Donoso2018', changeParams, forExploration, input_params, aux_params, commentTraj, cart_product=False, \
                    traj_hash=traj_hash, path_to_simulations = path_to_simulations, record_per_run={'runs': [0], 'record':['v']}) # setup configuration file
pypet_run_simulation(traj_hash = traj_hash, path_to_simulations =path_to_simulations) # run simulation
do_ifa_analysis(traj_hash=traj_hash, save2file=True, path_to_simulations =path_to_simulations) # analyze the population rate dynamics and check for IFA 


# ## --- reduced model with refrac period ---------------------------------------------------------------------------------------
traj_hash=1008
commentTraj = 'Reduced model with refractory period (+ larger noise & delay to ensure unimodal voltage distributions, Nint=200 for comparability with Donoso2018): double-ramp drive, symmetric or asymmetric, to check IFA (compare to h1003)'
pypet_makeConfig_lif_ifa_fromcyclostat(traj_hash_stat=1003, nreps=nreps, traj_hash=traj_hash, comment=commentTraj, \
                                      shape='ramp_asym', ramp_time_up = ramp_time_up, ramp_time_down = ramp_time_down, plateau_time = plateau_time, Tinit = Tinit, Tedge=Tedge, t_pad = t_pad,  \
                                      path_to_simulations =path_to_simulations) # prepares an IFA simulation with parameters and drive matching the results of the simulations with constant drive
pypet_run_simulation(traj_hash = traj_hash, path_to_simulations =path_to_simulations) # run simulation
do_ifa_analysis(traj_hash=traj_hash, save2file=True, path_to_simulations =path_to_simulations) # analyze the population rate dynamics and check for IFA 

## --- reduced model as in manuscript ---------------------------------------------------------------------------------------
traj_hash=1009
commentTraj = 'Reduced model as in manuscript (Fig 2, Nint=200 for comparability with Donoso2018): double-ramp drive, symmetric or asymmetric, to check IFA (compare to h1004)'
pypet_makeConfig_lif_ifa_fromcyclostat(traj_hash_stat=1004, nreps=nreps, traj_hash=traj_hash, comment=commentTraj, \
                                      shape='ramp_asym', ramp_time_up = ramp_time_up, ramp_time_down = ramp_time_down, plateau_time = plateau_time, Tinit = Tinit, Tedge=Tedge, t_pad = t_pad,  \
                                      path_to_simulations =path_to_simulations) # prepares an IFA simulation with parameters and drive matching the results of the simulations with constant drive
pypet_run_simulation(traj_hash = traj_hash, path_to_simulations =path_to_simulations) # run simulation
do_ifa_analysis(traj_hash=traj_hash, save2file=True, path_to_simulations =path_to_simulations) # analyze the population rate dynamics and check for IFA 

#############################################################################################################################
# TRANSIENT DRIVE: SQUARE PULSE, different networks
# parameters for IFA simulations:
nreps = 30
strength = 0.8 # set amplitude of square pulse relative to full synchrony (here: 80%)
Tinit, Tedge, t_pad = 200., 20., 20.
plateau_time = 45
ramp_time = 0
Tsim = Tinit + 2*ramp_time + plateau_time + 2*Tedge + 2*t_pad

input_params = {'unit':'spks/sec', 'shape' : 'ramp', 'Tinit': Tinit, 'baseline': 500, 'peak': 14_000*strength, 'plateau_time': plateau_time, 'ramp_time': ramp_time,
                'Tsim': Tsim }
forExploration = {} # set up parameter explorations
forExploration['simulation.seed_brian'] = list(np.random.randint(2**32-1, size=nreps)) # vary the seed to simulate the same network and input configuration for multiple noise realizations
aux_params = {'analysis.unit': True, 'analysis.net': True, 'analysis.offset': 50, 'analysis.getInstFreq':True, \
              'analysis.ifreq_baselinewindow':[50, 150], 'analysis.fmin': 70, 'analysis.fmax': 400,
              'analysis.ifreq_targetwindow': [Tinit + t_pad, Tsim - t_pad]}
aux_params['postproc.v_recordwindow'] = aux_params['postproc.v_recordwindow_for_simulation'] = aux_params['analysis.ifreq_targetwindow'] 


# --- network from Donoso et al. 2018 Fig 1 ---------------------------------------------------------------------------------
traj_hash= 1010
commentTraj = 'Donoso et al. 2018 default network: show that square pulse drive (up to 80% of Ifull) does not yield IFA  (compare to h1001).'
pypet_setup_config('Donoso2018', {}, forExploration, input_params, aux_params, commentTraj, cart_product=False, \
                    traj_hash=traj_hash, path_to_simulations = path_to_simulations, record_per_run={'runs': [0], 'record':['v']}) # setup configuration file
pypet_run_simulation(traj_hash = traj_hash, path_to_simulations =path_to_simulations) # run simulation
do_ifa_analysis(traj_hash=traj_hash, save2file=True, path_to_simulations =path_to_simulations) # analyze the population rate dynamics and check for IFA


# --- network from Donoso et al. 2018 but with INDEPENDENT Poisson inputs ----------------------------------------------------
traj_hash= 1011
changeParams = {'inputmode':'DiffApprox'}
commentTraj = 'Donoso et al. 2018 network with independent Gaussian white noise input (noise increases with drive as for Poisson input): show that square pulse drive (up to 80% of Ifull) does not yield IFA (compare to h1002)'
pypet_setup_config('Donoso2018', changeParams, forExploration, input_params, aux_params, commentTraj, cart_product=False, \
                    traj_hash=traj_hash, path_to_simulations = path_to_simulations, record_per_run={'runs': [0], 'record':['v']}) # setup configuration file
pypet_run_simulation(traj_hash = traj_hash, path_to_simulations =path_to_simulations) # run simulation
do_ifa_analysis(traj_hash=traj_hash, save2file=True, path_to_simulations =path_to_simulations) # analyze the population rate dynamics and check for IFA 


# # ## --- reduced model with refrac period ---------------------------------------------------------------------------------------
traj_hash= 1012
commentTraj = 'Reduced model with refractory period (+ larger noise & delay to ensure unimodal voltage distributions, Nint=200 for comparability with Donoso2018): show that square pulse drive (up to 80% of Ifull) does not yield IFA (compare to h1003)'
# --- load point of full synchrony and scale peak input down to 80% 
info_stat = pd.read_csv(pypet_get_trajectoryPath(traj_hash = 1003, path_to_simulations=path_to_simulations) + 'info.csv', 
                        index_col=0, squeeze=True, header=None)
pypet_makeConfig_lif_ifa_fromcyclostat(traj_hash_stat=1003, nreps=nreps, traj_hash=traj_hash, comment=commentTraj, \
                                      shape='ramp', ramp_time = ramp_time, plateau_time = plateau_time, peak_nA = info_stat['I_full_nA']*strength,
                                      Tinit = Tinit, Tedge=Tedge, t_pad = t_pad, path_to_simulations =path_to_simulations) # prepares an IFA simulation with parameters and drive matching the results of the simulations with constant drive
pypet_run_simulation(traj_hash = traj_hash, path_to_simulations =path_to_simulations) # run simulation
do_ifa_analysis(traj_hash=traj_hash, save2file=True, path_to_simulations =path_to_simulations) # analyze the population rate dynamics and check for IFA 


## --- reduced model as in manuscript ---------------------------------------------------------------------------------------
traj_hash= 1013
commentTraj = 'Reduced model as in manuscript (Fig 2, Nint=200 for comparability with Donoso2018): show that square pulse drive (up to 80% of Ifull) does not yield IFA (compare to h1004)'
# --- load point of full synchrony and scale peak input down to 80% 
info_stat = pd.read_csv(pypet_get_trajectoryPath(traj_hash = 1004, path_to_simulations=path_to_simulations) + 'info.csv', 
                        index_col=0, squeeze=True, header=None)

pypet_makeConfig_lif_ifa_fromcyclostat(traj_hash_stat=1004, nreps=nreps, traj_hash=traj_hash, comment=commentTraj, \
                                      shape='ramp', ramp_time = ramp_time, plateau_time = plateau_time, peak_nA = info_stat['I_full_nA']*strength,
                                      Tinit = Tinit, Tedge=Tedge, t_pad = t_pad, path_to_simulations =path_to_simulations) # prepares an IFA simulation with parameters and drive matching the results of the simulations with constant drive
pypet_run_simulation(traj_hash = traj_hash, path_to_simulations =path_to_simulations) # run simulation
do_ifa_analysis(traj_hash=traj_hash, save2file=True, path_to_simulations =path_to_simulations) # analyze the population rate dynamics and check for IFA 



#%% Reviewer Figures 4, 5: Brunel 1999 network
traj_hash = 2001 
Jext = 0.1 # mV, syn weight of external inputs, Brunel, Hakim 99, Fig. 5
mu_ext_brunel = np.concatenate((np.arange(20, 45, 5), np.r_[50, 60, 70, 80, 90, 100, 125, 150, 200, 300, 400])) # mV, range of external mean input
sigma_ext_brunel = np.sqrt(Jext*mu_ext_brunel) # intensity of external Gaussian white noise, as expected from Poisson spiking input after diffusion approximation

pypet_make_config_brunel_hakim99(mu_ext_brunel, sigma_ext_brunel, traj_hash=traj_hash) # setup simulation config file, also produces Rev Fig 4
pypet_run_simulation(traj_hash = traj_hash, path_to_simulations = path_to_simulations) # run simulation

#%% Reviewer Figure 5 : recording membrane potentials for spectral analysis
traj_hash = 2002 # config generated by hand (copy of h0 with less runs and longer v_recordwindow)
pypet_run_simulation(traj_hash = traj_hash, path_to_simulations = path_to_simulations) # run simulation
   
#%% Reviewer Figures 1,2: wiggle dependence on initial condition at beginning of downwards ramp
traj_hash = 2003 
commentTraj = 'Investigation of wiggle in Fig 2D middle panel: simulate only downwards ramp with same initial condition for all trials: Gaussian centered at mumin, very subthreshold. Does the wiggle persist/occur reliably in all trials now?'
# copy parameters from h2 trajectory:  
nreps = 50
parent = 'mymicro'
aux_params = {'analysis.unit': True, 'analysis.net': True, 'analysis.offset': 50, 'analysis.getInstFreq':True, 
              'analysis.ifreq_baselinewindow':[50,70], 'analysis.ifreq_targetwindow': [0, 50], 'postproc.v_recordwindow' : [0,50], 
              'postproc.v_recordwindow_for_simulation' : [0,50], 'analysis.fmin': 70, 'analysis.fmax': 400,
              "linear_stability.A0crit": 15.541841101404215,
              "linear_stability.I0crit_nA": 0.09140000000000001,
              "linear_stability.Icrit_nA": 0.1923828125,
              "linear_stability.fcrit": 305.3}
changeParams = {"Nint": 10000,
                "IPSPint": 16.25,
                "IPSPshape": "delta",
                "Vreset": -65.0,
                "inputmode": "DiffApprox",
                "neuronmodel": "LIF",
                "p_ii": 1,
                "record_micro": "v",
                "tl": 1.2,
                "tref": 0,
                "Vinit": 'gaussian-4-revision'
                }
input_params = {'unit':'nA', 'Ie_sig': 0.037, 'shape' : 'ramp_asym', 'Tinit': 0., 'baseline': 0.096, 'peak': 1.16, \
                'ramp_time_up': 0., 'plateau_time': 0., 'ramp_time_down': 40.76, 'Tsim': 70., 't_on':0.}
forExploration = {'simulation.seed_brian': list(np.random.randint(2**32-1, size=nreps))}
pypet_setup_config('mymicro', changeParams, forExploration, input_params, aux_params, commentTraj, \
                    cart_product=False, record_per_run={'runs': [i for i in range(0,nreps,5)], 'record':['v']}, traj_hash=traj_hash, \
                    path_to_simulations = path_to_simulations)# setup configuration file
pypet_run_simulation(traj_hash = traj_hash, path_to_simulations =path_to_simulations) # run simulation  
  
