#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 17:34:44 2023

This file produces all figures of the manuscript, supplement, and response to the reviewer reports.
Hashes to simulation data plotted in each figure are provided as input arguments where applicable.

* Figures 3-5, and 8-10 can be reproduced without (re)running any simulations 
  (using the example simulation trajectory h0 that we provide in the repo)
  To restrict plotting to those Figures only, set all_simulations_available = False.
* Figures 1, 2, and 7 can only be reproduced after rerunning the simulations with hashes 1 and 2 (see main_run_simulations.py)
* For Supplementary and Reviewer Figures further simulation data are needed (hashes indicated below)
* All simulations can be reproduced by running main_run_simulations.py. Careful: result containers can be several GB large (especially the parameter exploration h3 is large)!
* You can also select only specific figures to be plotted (see dictionary plot_fig below)

@author: natalie
"""

###############################################################################################################################################################################
all_simulations_available = False # set to True if main_run_simulations.py has been run and all simulations are stored in /simulations/
###############################################################################################################################################################################

plot_fig = {1: True,
            2: True, 
            3: True,
            4: True,
            5: True,
            6: True,
            7: True,
            8: False,
            9: True,
            10: True,
            's1a': True,    # Supplementary Fig A in Appendix S1
            's1b' : True,   # Supplementary Fig B in Appendix S1
            's2c' : True,   # Supplementary Fig A in Appendix S2
            'r12' : True,   # Reviewer Figures 1, 2
            'r3' : True,    # Reviewer Figure 3
            'r45' : True,   # Reviewer Figures 4, 5
            'r6' : True,    # Reviewer Figure 6      
  }


if not all_simulations_available:
  for key in [1, 2, 6, 7, 's1a', 's1b', 's2c', 'r12', 'r3', 'r45', 'r6']:
    # disable plotting of figures for which simulation data has not been regenerated
    plot_fig[key] = False

########################################################################################################################################
path_to_simulations = './simulations/'
path_to_figures = './figures/'


import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import os

from methods_figures import plot_figure_1, plot_figure_2, plot_figure_3, plot_figure_4, plot_figure_5, plot_figures_6_7, plot_figure_8, plot_figure_9, plot_figure_10, \
                            plot_figure_S1A, plot_figure_S1B, plot_figure_S2C, \
                            plot_reviewer_figure_3, plot_reviewer_figure_6, plot_figure_1_fixN
from methods_analyze_simulations import do_ifa_analysis


# check / create data paths
if not os.path.exists(path_to_figures):
  os.makedirs(path_to_figures)

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
    
plt.close('all')

#%% Hashes to simulation data
# all simulations were done in main_run_simulations.py

traj_hash_ex = 0 # example simulation of reduced size provided in the repo (Fig 1 but only for N=10_000)
traj_hash_constant_drive = 1 # simulations under constant drive for Fig 1
traj_hash_spw_drive = 2 # simulations under time-dependent drive for Fig 2

# hashes to other simulations for supplementary Figures etc are provided where needed in the code below.

#%% Manuscript Figures 1-10 (Results, Methods)

plot_figure_1_fixN(traj_hash = traj_hash_ex, run_idx = [0, 5, 9, 11], path_to_simulations = path_to_simulations ) # plot example simulation h0

if plot_fig[1]:
  plot_figure_1(traj_hash= traj_hash_constant_drive, path_to_simulations = path_to_simulations, path_to_figures=path_to_figures) # 215, traj_name='Data_mymicro_LIF-delta-DiffApprox_SS___WITH_tref_p_ii_Nint_record_micro_tl_IPSPint_Vreset___FOR_Nint_level_h215', 
  
if plot_fig[2]:
  plot_figure_2(traj_trans_hash = traj_hash_spw_drive, traj_stat_hash = traj_hash_constant_drive, path_to_simulations = path_to_simulations, path_to_figures = path_to_figures)

if plot_fig[3]:
  plot_figure_3(traj_hash = traj_hash_ex, run=4, path_to_simulations = path_to_simulations, path_to_figures = path_to_figures)

if plot_fig[4]:
  plot_figure_4(traj_hash_ex, reset=True, run_max=9, path_to_simulations = path_to_simulations, path_to_figures = path_to_figures)

if plot_fig[5]:
  plot_figure_5(traj_hash_stat = traj_hash_ex, path_to_simulations = path_to_simulations, path_to_figures = path_to_figures)

if plot_fig[6] or plot_fig[7]:
  plot_figures_6_7(traj_stat_hash = traj_hash_ex, traj_trans_hash=traj_hash_spw_drive, m_val = np.r_[.4, .2, .1], reset=True, \
                  mu_min_start_first_cycle= 0.5, show_simulations = True,  plot_controls_fig7=True,\
                  path_to_simulations = path_to_simulations, path_to_figures = path_to_figures)

if plot_fig[8]:
  D, Delta, K, tm, Vr = 0.04, 1.2, 5, 10, 0
  plot_figure_8(D, Delta, K, tm, Vr, traj_hash = traj_hash_ex, path_to_simulations = path_to_simulations,  path_to_figures = path_to_figures)

if plot_fig[9]:
  plot_figure_9(parameters={'D':.04, 'Delta':1.2, 'K':5, 'tm':10, 'Vr':0, 'Iext':3.6}, path_to_figures = path_to_figures)

if plot_fig[10]:
  plot_figure_10(parameters={'D':.04, 'Delta':1.2, 'K':5, 'tm':10, 'Vr':0}, path_to_figures = path_to_figures)

#%% Supplementary Figures

'''
Supplementary Figures A and B of S1 Appendix
'''
traj_hash = 3 # We provide the results needed for plotting in >results/gaussian_drift_approx_constant_drive_performance_check/... These data are sufficient to regenerate the figures S1 A and B.

if plot_fig['s1a']:
  plot_figure_S1A(traj_hash = traj_hash, reset=True, path_to_figures = path_to_figures)
if plot_fig['s1b']:
  plot_figure_S1B(traj_hash = traj_hash, path_to_figures = path_to_figures)

'''
Supplementary Figure A of S2 Appendix
'''
if plot_fig['s2c']:
  plot_figure_S2C(traj_hash_ABCD_stat = [1001, 1002, 1003, 1004], traj_hash_ABCD_trans = [1006, 1007, 1008, 1009] , \
                 path_to_figures=path_to_figures, path_to_simulations= path_to_simulations) # all simulations done in main_run_simulations.py

#%% Reviewer Figures 1, 2
if plot_fig['r12']:
  do_ifa_analysis(traj_hash=2003, save2file=True, fmin=70, flim=400, path_to_simulations =path_to_simulations, t0_centered=False) # produces Reviewer Figs 1+2 in folder simulations/h2003/analysis_IFA_fmin70/

#%% Reviewer Figure 3
if plot_fig['r3']:
  plot_reviewer_figure_3(path_to_figures=path_to_figures)

#%% Reviewer Figure 4, 5
'''
Brunel, Hakim 1999 replication
'''
if plot_fig['r45']:
  plot_figure_1_fixN(traj_hash = 2001, run_idx = [0, 4, 12, 15], xmax=.8, vbins=30, fmax = 300, axis_limit_rate_upper=[20, 200, 1000, 1700], axis_limit_v_lower = [-60, -70, -150, -200], show_raster=True, eps_hopf_margin = 0, xlim_A=[0,40], path_to_simulations = path_to_simulations)

#%% Reviewer Figure 6
'''
Reviewer Figure
membrane potential power spectral density 
'''
if plot_fig['r6']:
  plot_reviewer_figure_6(traj_hash=2002, run_idx = [0,1,2,3], path_to_simulations=path_to_simulations, path_to_figures=path_to_figures)

