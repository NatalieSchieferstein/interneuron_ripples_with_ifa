#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 12:02:22 2023

This file contains the functions and auxiliary functions to generate all figures of the manuscript, supplementary, and response to the reviewer reports.

Run main_plot_figures.py to generate all figures.

@author: natalie
"""

### import packages
import json
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import ConnectionPatch
import matplotlib
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import pandas as pd
import string
from tools import add_ticks, custom_arrow, despine, df_heatmap, do_linear_regression, get_gauss, get_PSD, gridline, list_from_dict, match_axis_limits_new

### import functions from methods scripts
from methods_simulations import convert_input, f_oscillation_analysis_stationary, get_pfix_str, getVtraces,  pypet_find_runs, pypet_get_exploration_overview, pypet_get_from_runs,  pypet_get_runidx_from_runname, \
  pypet_get_trajectoryPath, pypet_load_trajectory,  recover_exploration, rescale_params_dimless, store_info_cyclostat_lif_simulation
from methods_analytical import analysis_IFA_constant_drive_analytical, dde_bifurcation_analysis_numerical, dde_extract_last_cycle, \
  find_trajectories_linear_drive, gaussian_drift_approx_stat, gaussian_drift_approx_stat_numerical, \
  gaussian_drift_approx_transient, get_Iext_lower_bound, get_pt_fullsynch, integrate_dde_numerically, integrate_dde_numerically_until_convergence, \
  plot_convergence_to_asymptotic_dynamics, visualize_traces_wlineardrive
from methods_analyze_simulations import pypet_gaussian_drift_approx_stat, plot_rate, plot_vhist, plot_raster, plot_ifa_example_run, \
  plot_ifa_linreg, get_error_gauss_drift_approx


pi = np.pi
inf = np.inf
nan = np.nan

my_rcParams = json.load(open('settings/matplotlib_my_rcParams.txt'))
matplotlib.rcParams.update(my_rcParams)

my_colors = json.load(open('settings/my_colors.txt'))
# custom colormap for histograms of voltage traces:
from matplotlib.colors import LinearSegmentedColormap
colors = [(1, 1, 1), (.5, .5, .5)]  # white -> gray
colormap_voltage_hist = LinearSegmentedColormap.from_list('my_cmap_grays', colors, N=100)

cm=1/2.54
width_a4 = 21*cm
width_a4_wmargin = width_a4 - 2*2*cm
# plos cb measurements
plos_width_text = 13.2*cm
plos_width_fullpage = 19.05*cm
plos_height_fullpage = 22.23*cm
panel_labelsize = 9

# labels for figures
str_Iext = '$I_\mathrm{E}$'
str_Iextmin = '$I_\mathrm{E}^\mathrm{min}$'
str_Iextfull = '$I_\mathrm{E}^\mathrm{full}$'
str_Iextcrit = '$I_\mathrm{E}^\mathrm{crit}$'

#%% Fig 1
def plot_figure_1(traj_hash, xmax=None, N_ex = 10_000, vbins=50, I_hopf=None, \
                                              axis_limit_rate_upper = [50, 1000, 1500, 1500], axis_limit_v_lower=[-70, -100, -130, -150] ,\
                                              show_overfull = True, show_raster = True, xlim_A=[0,20], \
                                              path_to_simulations = './simulations/', path_to_figures = './figures/'):
  '''
  Plot Fig 1.

  Parameters
  ----------
  traj_hash : hash pointing to simulation data.
  xmax : Upper limit for x-axis in Fig 1B. The default is None.
  N_ex : int, optional. Network size to show in A. The default is 10_000.
  vbins : int, optional. Number of bins for voltage histogram (in each time point). The default is 50.
  I_hopf : float, optional. Critical external drive. The default is None.
  axis_limit_rate_upper : list, optional. The default is [50, 1000, 1500, 1500].
  axis_limit_v_lower : list, optional. The default is [-70, -100, -130, -150].
  show_overfull : bool, optional. Whether to show the regime of "multiple spikes" after full synch in A. The default is True.
  show_raster : bool, optional. Include raster plot. The default is True.
  xlim_A : list, optional. Time axis limits in A. The default is [0,20].
  path_to_simulations : str, optional. The default is './simulations/'.
  path_to_figures : str, optional. The default is './figures/'.

  Returns
  -------
  None. (figure is stored in path_to_figures)
  '''  
  print('\n\nPlotting Fig 1...')
  traj = pypet_load_trajectory(traj_hash = traj_hash, path_to_simulations=path_to_simulations)
  res = traj.results.summary.scalarResults     
  
  linear_stability_available = 'linear_stability' in traj.parameters.f_get_children().keys()
  
  if linear_stability_available:
    I_hopf = traj.linear_stability.Icrit_nA
    res.loc[res['parameters.input.level'] < I_hopf - 1e-2, 'freq_net'] = nan
    
  res_pivot = res.pivot(index='parameters.input.level', columns='parameters.network.Nint', \
                        values=['freq_net', 'freq_unit_mean', 'freq_unit_std', 'saturation', 'CV'])
    
  # extract dimless D to set a ommon colorbar limit
  traj.v_idx=0
  sigma_v = np.sqrt(.5)*(traj.tm/(traj.C/1000)*traj.input.Ie_sig)# ms(ms/nF*nA)^2 = mV^2
  traj.f_restore_default()
  cbar_max = 1/(np.sqrt(2*np.pi)*sigma_v) # same limit as  FIg 3
  
  if not xmax:
    xmax = res.loc[(res.saturation>=1) & (res['parameters.network.Nint']==10000), 'parameters.input.level'].min()*1.05

  with plt.rc_context({"axes.labelsize": 8, "axes.titlesize": 8, "font.size": 8,\
                         "xtick.labelsize": 5, "ytick.labelsize": 5}):
    nruns = 3+show_overfull # number of example runs shown in A
    # --- construct figure -----------------------------------
    fig_width= plos_width_text # width_a4_wmargin*.2*nruns
    fig_height = plos_width_text*(1 +.125*show_raster) # width_a4_wmargin*(.8 +.1*show_raster) # 9*cm # 10.5*cm
    fig = plt.figure(figsize=(fig_width, fig_height))#, constrained_layout=True)
    
    width_ratios = [90, 1]
    wspace=.02
    hspace = .2
    height_ratios= [1,1.8]
    gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios = width_ratios, height_ratios=height_ratios, wspace=wspace, \
                           hspace=hspace)#, width_ratios=[5,2], height_ratios=[2,3])
    
    gs_A_sup = gs[0,:].subgridspec(2,2, width_ratios=width_ratios, height_ratios=[2,.6], wspace=wspace, hspace=.5)
    gs_A_sub = gs_A_sup[0,:].subgridspec(2+show_raster,2, width_ratios=width_ratios, wspace=wspace)
    gs_A_time = gs_A_sub[:,0].subgridspec(2+show_raster, nruns, wspace=.35, hspace=.2)
    gs_A_power = gs_A_sup[1,0].subgridspec(1, nruns, wspace=.3)
    
    ax_A_time = gs_A_time.subplots()
    ax_A_power = gs_A_power.subplots(sharey=True)
    ax_A_cb = fig.add_subplot(gs_A_sub[-1,1])
    
    despine(ax_A_power)
    despine(ax_A_time[:2,:], which=['top', 'right', 'bottom'])
    despine(ax_A_time[-1,:])
    
    gs_B = gs[1,0].subgridspec(3,1, height_ratios=[5,1,1])
    ax_B = gs_B.subplots(sharex=True)
    despine(ax_B)

    ax_B[0].text(-.1, 2.27, string.ascii_uppercase[0], transform=ax_B[0].transAxes, size=panel_labelsize*.8, weight='bold', ha='left')
    ax_B[0].text(-.1, 1, string.ascii_uppercase[1], transform=ax_B[0].transAxes, size=panel_labelsize*.8, weight='bold', ha='left')
    
    # different notations for drive depending on inputunit  
    if traj.input.unit=='nA':
      drive_str = '$I_\mathrm{ext}$'
    elif traj.input.unit=='spks/s':
      drive_str = '$\Lambda$'

      
    # --- fill figure -----------------------------------
    
    # find runs with Nint = Nex that represent the regimes AI, sparse, full synch, and pathological firing beyond full synch
    run_name_ai = res.loc[(res['parameters.network.Nint']==N_ex)].index[0]
    run_name_sparse =  res.loc[(res['parameters.network.Nint']==N_ex) & (res.saturation<=.5)].index[-1]
    run_name_full =  res.loc[(res['parameters.network.Nint']==N_ex) & (res.saturation<=1)].index[-1]
    run_name_overfull =  res.loc[(res['parameters.network.Nint']==N_ex) & (res.saturation>1)].index[-1]

    run_idx = [pypet_get_runidx_from_runname(run_name_ai),
               pypet_get_runidx_from_runname(run_name_sparse),
               pypet_get_runidx_from_runname(run_name_full),
               pypet_get_runidx_from_runname(run_name_overfull)]
    
    label= ['asynch irregular', 'sparse synch', 'full synch', 'multiple spikes']
    
    # A: plot example traces
    ax_A_time, ax_A_power, ax_cb \
    = auxplot_steadystate_examples(ax_A_time, ax_A_power, ax_A_cb, traj, run_idx[:nruns], drive_str, vbins=vbins, xlim_A=xlim_A, \
                                   cbar_max =cbar_max, show_raster=show_raster, axis_limit_rate_upper=axis_limit_rate_upper, axis_limit_v_lower =axis_limit_v_lower, label= label[:nruns])
      
    # B: plot network and unit frequency against drive
    ax_B[0].fill_between([0,xmax*1.1], 140, 220, facecolor='lightgray', edgecolor='face')  
    res_pivot.plot(ax=ax_B[0], y='freq_net', style= [':', '--', '-'], marker='^', color=my_colors['fnet'], legend=False)
    res_pivot.plot(ax=ax_B[0], y='freq_unit_mean', style= [':', '--', '-'], marker='o', color=my_colors['funit'], legend=False)  #yerr = 'freq_unit_std', 
      
    res_pivot.plot(ax=ax_B[1], y='saturation', style= [':', '--', '-'], marker='.', color='k', legend=False)
    res_pivot.plot(ax=ax_B[2], y='CV', style= [':', '--', '-'], marker='.', color='k', legend=False)
    
    # Hopf bifurcation
    if linear_stability_available:
      ax_B[0].plot(I_hopf, traj.linear_stability.fcrit, marker='^', linestyle='', fillstyle='none', markeredgecolor='r', zorder=3)
      ax_B[0].plot(I_hopf, traj.linear_stability.A0crit, marker='o', linestyle='', fillstyle='none', markeredgecolor='r', zorder=3)

    # legend by hand
    handle_fnet = Line2D([0], [0], marker='^', linestyle='-', color=my_colors['fnet'], label='$f_\mathrm{net}$')
    handle_funit = Line2D([0], [0], marker='o', linestyle='-', color=my_colors['funit'], label='$f_\mathrm{unit}$')
    handle_hopf = Line2D([0], [0], marker='^', linestyle='', fillstyle='none', markeredgecolor='r', label='Hopf')
    
    Nint = np.sort(res['parameters.network.Nint'].unique())
    handle_1 = Line2D([0], [0], linestyle=':', color=my_colors['fnet'], label='N={:.0f}'.format(Nint[0]))
    handle_2 = Line2D([0], [0], linestyle='--', color=my_colors['fnet'], label='N={:.0f}'.format(Nint[1]))
    handle_3 = Line2D([0], [0], linestyle='-', color=my_colors['fnet'], label='N={:.0f}'.format(Nint[2]))
    
    ax_B[0].legend(handlelength=2, handles=[handle_fnet, handle_funit, handle_hopf, handle_1, handle_2, handle_3], ncol=2, \
                  bbox_to_anchor=(1,1), loc='upper right', borderaxespad=0.)#, labelspacing=0.1, borderaxespad=.1)
  
    # formatting
    ax_B[0].set_ylim(bottom=0)
    ax_B[-1].set_xlim([0, xmax])
    ax_B[0].set_ylabel('frequency [Hz]')
    ax_B[1].set_ylabel('saturation')
    ax_B[2].set_ylabel('CV')
    ax_B[2].set_xlabel(r'external drive {} [{}]'.format(drive_str, traj.input.unit))
    
    ax_B[1].set_ylim([0,1.1])
    ax_B[2].set_ylim([0,1])
    
    for ax in [ax_B[1], ax_B[2], ax_A_power[0], ax_A_power[1], ax_A_power[2]]:
      ax.spines['right'].set_visible(False)
      ax.spines['top'].set_visible(False)
    
  traj.f_restore_default()
  fig.savefig(path_to_figures+'Fig1.pdf', bbox_inches='tight')
  fig.savefig(path_to_figures+'Fig1.tif', bbox_inches='tight')
  return 

def plot_figure_1_fixN(traj_hash, run_idx, xmax=None, vbins=50, I_hopf=None, fmax = 350, \
                                legend_loc='upper right', save2file=True, df=10, recalc_psd = [], letter='', number=False, cbar_max =.15, eps_hopf_margin=-1e-2, \
                                showC=False, show_raster=False, ncol_C=2, xlim_A=[0,20], axis_limit_rate_upper=[50, 1000, 1500, 1500], \
                                axis_limit_v_lower=[-70, -100, -130, -150],  \
                                label= ['asynch irregular', 'sparse synch', 'full synch', 'multiple spikes'], \
                                path_to_simulations = './simulations/'):
  ''' same as plot_figure_1, but using a smaller simulation file with only one network size (e.g. h0).
  '''

  traj = pypet_load_trajectory(traj_hash = traj_hash, path_to_simulations=path_to_simulations)
  res = traj.results.summary.scalarResults  
  
  if len(recalc_psd):
    freqs, power = {}, {}
    for i in recalc_psd:
      traj.v_idx = i
      freqs[i], power[i], fn \
      = f_oscillation_analysis_stationary(traj.results.crun.LFP_raw, traj.dt, traj.results.crun.spktrains, traj.scale, \
                                          df=df, k='max', fmin=30, sat_cyclewise=False, offset=50, freq_unit=traj.results.crun.freq_unit)[:3]
      res.iloc[i]['freq_net']=fn
      
  linear_stability_available = 'linear_stability' in traj.parameters.f_get_children().keys()
  
  if linear_stability_available:
    I_hopf = traj.linear_stability.Icrit_nA 
    res.loc[res['parameters.input.level'] < I_hopf + eps_hopf_margin, 'freq_net'] = nan


  # add new columns level, that either is a copy of the original level or is translated to the desired inputunit 
  res['level'] = res['parameters.input.level'].copy()
  if not xmax:
    xmax = res.loc[res.saturation>=1, 'level'].min()
    if np.isnan(xmax): # saturation never reached 1
      xmax = res['level'].max()    

  with plt.rc_context({"axes.labelsize": 8, "axes.titlesize": 8, "font.size": 8,\
                         "xtick.labelsize": 5, "ytick.labelsize": 5}):
    # --- construct figure -----------------------------------
    fig_width= width_a4_wmargin*.2*len(run_idx)
    fig_height = width_a4_wmargin*(.8 +.1*show_raster +.2*showC) # 9*cm # 10.5*cm
    fig = plt.figure(figsize=(fig_width, fig_height))#, constrained_layout=True)
    
    width_ratios = [90, 1]
    wspace=.02
    hspace = .3 if showC else .2
    height_ratios= [1, 1.5, 0.5] if showC else [1,1.8]
    gs = gridspec.GridSpec(2+showC, 2, figure=fig, width_ratios = width_ratios, height_ratios=height_ratios, wspace=wspace, \
                           hspace=hspace)#, width_ratios=[5,2], height_ratios=[2,3])
    
    gs_A_sup = gs[0,:].subgridspec(2,2, width_ratios=width_ratios, height_ratios=[2,.6], wspace=wspace, hspace=.5)
    gs_A_sub = gs_A_sup[0,:].subgridspec(2+show_raster,2, width_ratios=width_ratios, wspace=wspace)
    gs_A_time = gs_A_sub[:,0].subgridspec(2+show_raster, len(run_idx), wspace=.35, hspace=.2)
    gs_A_power = gs_A_sup[1,0].subgridspec(1, len(run_idx), wspace=.3)
    
    ax_A_time = gs_A_time.subplots()
    ax_A_power = gs_A_power.subplots(sharey=True)
    ax_A_cb = fig.add_subplot(gs_A_sub[-1,1])
    
    despine(ax_A_power)
    despine(ax_A_time[:2,:], which=['top', 'right', 'bottom'])
    despine(ax_A_time[-1,:])
    
    gs_B = gs[1,0].subgridspec(3,1, height_ratios=[5,1,1])
    ax_B = gs_B.subplots(sharex=True)
    despine(ax_B)
    
    if showC:
      ax_C = fig.add_subplot(gs[2,0])
    
    if number:
      if showC:
        ax_B[0].text(-.15, 2.3, '({}i)'.format(letter), transform=ax_B[0].transAxes, size=panel_labelsize*.8, weight='bold', ha='left')
        ax_B[0].text(-.15, 1, '({}ii)'.format(letter), transform=ax_B[0].transAxes, size=panel_labelsize*.8, weight='bold', ha='left')
        ax_B[0].text(-.15, -.87, '({}iii)'.format(letter), transform=ax_B[0].transAxes, size=panel_labelsize*.8, weight='bold', ha='left')
      else:
        ax_B[0].text(-.15, 2.27, '({}i)'.format(letter), transform=ax_B[0].transAxes, size=panel_labelsize*.8, weight='bold', ha='left')
        ax_B[0].text(-.15, 1, '({}ii)'.format(letter), transform=ax_B[0].transAxes, size=panel_labelsize*.8, weight='bold', ha='left')
    else:
      if showC:
        ax_B[0].text(-.15, 2.3, string.ascii_uppercase[0], transform=ax_B[0].transAxes, size=panel_labelsize*.8, weight='bold', ha='left')
        ax_B[0].text(-.15, 1, string.ascii_uppercase[1], transform=ax_B[0].transAxes, size=panel_labelsize*.8, weight='bold', ha='left')
        ax_B[0].text(-.15, -.87, string.ascii_uppercase[2], transform=ax_B[0].transAxes, size=panel_labelsize*.8, weight='bold', ha='left')
      else:
        ax_B[0].text(-.12, 2.27, string.ascii_uppercase[0], transform=ax_B[0].transAxes, size=panel_labelsize*.8, weight='bold', ha='left')
        ax_B[0].text(-.12, 1, string.ascii_uppercase[1], transform=ax_B[0].transAxes, size=panel_labelsize*.8, weight='bold', ha='left')
     
    # different notations for drive depending on inputunit  
    if traj.input.unit=='nA':
      drive_str = '$I_\mathrm{ext}$'
    elif traj.input.unit=='spks/s':
      drive_str = '$\Lambda$'
      
    # --- fill figure -----------------------------------
    # A: plot example traces
    ax_A_time, ax_A_power, ax_cb = auxplot_steadystate_examples(ax_A_time, ax_A_power, ax_A_cb, traj, run_idx, drive_str, vbins=vbins, fmax = np.round(res.freq_net.max()+50), \
                                                                cbar_max =cbar_max, show_raster=show_raster, xlim_A=xlim_A, axis_limit_rate_upper=axis_limit_rate_upper, axis_limit_v_lower =axis_limit_v_lower, label= label)
    
    # B: plot steadystate overview
    ax_B[0].fill_between([0,xmax*1.1], 140, 220, facecolor='lightgray', edgecolor='face')  
    res.plot(ax=ax_B[0], x= 'level', y='freq_net', marker='^', color=my_colors['fnet'], legend=False)
    res.plot(ax=ax_B[0], x= 'level', y='freq_unit_mean', yerr='freq_unit_std', marker='o', color=my_colors['funit'], legend=False)  #yerr = 'freq_unit_std', 
      
    res.plot(ax=ax_B[1], x='level', y='saturation',  marker='.', color='k', legend=False)
    res.plot(ax=ax_B[2], x='level', y='CV',  marker='.', color='k', legend=False)
    
    # Hopf bifurcation
    if linear_stability_available:
      ax_B[0].plot(I_hopf, traj.linear_stability.fcrit, marker='^', linestyle='', fillstyle='none', markeredgecolor='r', zorder=3)
      ax_B[0].plot(I_hopf, traj.linear_stability.A0crit, marker='o', linestyle='', fillstyle='none', markeredgecolor='r', zorder=3)
 
    # legend by hand    
    handle_fnet = Line2D([0], [0], marker='^', linestyle='-', color=my_colors['fnet'], label='$f_\mathrm{net}$')
    handle_funit = Line2D([0], [0], marker='o', linestyle='-', color=my_colors['funit'], label='$f_\mathrm{unit}$')
    if linear_stability_available:
      handle_hopf = Line2D([0], [0], marker='^', linestyle='', fillstyle='none', markeredgecolor='r', label='Hopf')
      ax_B[0].legend(handlelength=2, handles=[handle_fnet, handle_funit, handle_hopf], loc=legend_loc)#, labelspacing=0.1, borderaxespad=.1)
    else:
      ax_B[0].legend(handlelength=2, handles=[handle_fnet, handle_funit], loc=legend_loc)#, labelspacing=0.1, borderaxespad=.1)
  
    # formatting
    ax_B[0].set_ylim([0, fmax])
    ax_B[-1].set_xlim([0, xmax])
    ax_B[0].set_ylabel('frequency [Hz]')
    ax_B[1].set_ylabel('saturation')
    ax_B[2].set_ylabel('CV')
    ax_B[2].set_xlabel(r'external drive {} [{}]'.format(drive_str, traj.input.unit))
    
    ax_B[1].set_ylim([0,1.1])
    ax_B[2].set_ylim([0,1])
    
    
  if showC:
    unitrates = traj.results.summary.unitrates#).to_numpy()
    
    nruns = len(traj.f_get_run_names()) 
    colidx = np.linspace(.9,0,nruns) # only show every second run
    
    bins=10 #np.arange(0, 350, 20)
    unitrates.iloc[:, ::2].plot.hist(ax=ax_C, histtype='step',  bins=bins, color = plt.cm.inferno(colidx[::2]), legend=False)
    ax_C.set_ylabel('# units')
    ax_C.set_xlabel('firing rate [Hz]')
    
    level = traj.results.summary.scalarResults['parameters.input.level'].values
    if traj.input.unit=='spks/s':
      level = (level/1000).astype(int)
    handles = []
    for i in range(0,nruns,2):
      h = Patch(facecolor='w', edgecolor=plt.cm.inferno(colidx[i]), label= '{:.1f}'.format(level[i]))
      handles.append(h)
    
    unit_str = ' [nA]' if traj.input.unit=='nA' else ' [10³ spks/s]' 
    ax_C.legend(loc='upper right', labelspacing=.2, borderaxespad=0, handles=handles, handlelength=1, ncol=ncol_C, title=drive_str + unit_str)
    ax_C.spines['right'].set_visible(False)
    ax_C.spines['top'].set_visible(False)
    
    
  traj.f_restore_default()
  if save2file:
    if recalc_psd:
      fig.savefig(pypet_get_trajectoryPath(traj_hash=traj_hash, path_to_simulations=path_to_simulations)+'steadystates_psd-recalc-df{}_h{}.pdf'.format(df, traj_hash), bbox_inches='tight')
    else:
      fig.savefig(pypet_get_trajectoryPath(traj_hash=traj_hash, path_to_simulations=path_to_simulations)+'steadystates_h{}.pdf'.format(traj_hash), bbox_inches='tight')
  return fig

def auxplot_steadystate_examples(ax_time, ax_power, ax_cb, traj, run_idx, drive_str, inputunit='nA', vbins=30, fmax = 400, \
                                cbar_max =.15, show_raster=False, xlim_A=[], axis_limit_rate_upper=[50, 1100, 1100], axis_limit_v_lower = [], \
                                label= ['asynch irregular', 'sparse synch', 'full synch', 'multiple spikes']):
  ''' 12.3.22
  show steady-state dynamics of spiking network for constant drive
  INPUT:
    traj_hash: hash of pypet traj with exploration of network size N
    inputunit: 'nA' or None if figure should be shown for dimensionless units of input (problem: voltages still in volt! change that!)
    xmax: right limit of xaxis, if None: chosen to include pt of full synch 
    vbins: number of bins for voltage histograms
  '''
  with plt.rc_context({"axes.labelsize": 8, "axes.titlesize": 8, "font.size": 8, "xtick.labelsize": 5, "ytick.labelsize": 5}):
     # illustrate example runs:
    for i in range(len(run_idx)):
      traj.v_idx = run_idx[i]

      # plot population rate
      ax_time[0,i] = plot_rate(ax_time[0,i], traj.results.crun.raw.LFP_smooth, traj.dt, t0 = traj.v_recordwindow[0], label='rate \n[spk/s]'*(not i))
      ax_time[0,i].set_ylim([0, axis_limit_rate_upper[i]])
      ax_time[0,i].set_yticks([0, axis_limit_rate_upper[i]])

      # raster plot
      if show_raster:
        ax_time[1,i] = plot_raster(ax_time[1,i], traj.results.crun.spktrains, neuronview=[0,30], t0=traj.v_recordwindow[0], ms=2)        
        if i:
          ax_time[1,i].set_yticklabels([])

      # plot distribution of membrane potentials 
      v_hist, v_binedges \
      =  getVtraces(traj.scale, traj.results.crun.raw.v, Vthr=traj.Vthr, vbins=vbins, volt_max=traj.Vthr, density=True)
      ax_time[-1,i], ax_cb \
      = plot_vhist(ax_time[-1,i], ax_cb, traj.Vthr, traj.Vreset, Vdistr=v_hist, vbinedges=v_binedges, v_recordwindow=traj.v_recordwindow, t0 = traj.v_recordwindow[0],\
                   show_colorbar=True, cbar_label=r'$p(V,t)$', cbar_max=cbar_max)
      if len(axis_limit_v_lower):
        vmin = axis_limit_v_lower[i]
      else:
        vmin = v_binedges[0]
      ax_time[-1,i].set_ylim([vmin, traj.Vthr+2])       
    
      # label threshold and lower bound:
      ax_time[-1,i].set_yticks([vmin, traj.Vthr])
        
      # plot power spectral density
      ax_power[i].plot(traj.results.crun.network.freqs, traj.results.crun.network.power, 'k')
      ax_power[i].set_yscale('log')
      ax_power[i].set_yticks([1e4, 1e0, 1e-4])
      ax_power[i].set_xlim([0, fmax])
      if not i:
        ax_power[i].set_ylim(bottom=.5*np.min(traj.results.crun.network.power[traj.results.crun.network.freqs<350]))
      if i==2: 
        ax_power[i].set_ylim(top=1.1*traj.results.crun.network.peakpower)
      
      # same x axis limit for all rows
      if not len(xlim_A):
        xlim_A = [0, np.diff(traj.v_recordwindow)[0]]
      for row in range(ax_time.shape[0]):
        ax_time[row, i].set_xlim(xlim_A)
      
      # set title
      ax_time[0,i].set_title(label[i]+'\n({}={:.2f}{})'.format(drive_str, traj.level, traj.input.unit))
 
    # disable axis labels where not needed
    for column in range(ax_time.shape[1]):
      for row in range(ax_time.shape[0]):
        if column:
          ax_time[row,column].set_ylabel('')
          ax_time[row,column].set_xlabel('')
          if row < ax_time.shape[0]-1:
            ax_time[row,column].set_xticks([])
        else:  
          if row < ax_time.shape[0]-1:
            ax_time[row,column].set_xlabel('')
            ax_time[row,column].set_xticks([])
    
    # set correct axis labels
    ax_time[-1,0].set_xlabel('time [ms]', labelpad=0)          
    ax_power[0].set_xlabel('frequency [Hz]', labelpad=0)      
    ax_power[0].set_ylabel('power \n [a.u.]', labelpad=.1)      

  traj.f_restore_default()
  return ax_time, ax_power, ax_cb

#%% Fig 2
def plot_figure_2(traj_trans_hash, traj_stat_hash, fmin=70, path_to_simulations = './simulations/',  path_to_figures = './figures/'):
  '''
  Plot Fig 2.

  Parameters
  ----------
  traj_trans_hash : Hash to simulation data for constant drive.
  traj_stat_hash : Hash to simulation data for time-dependent drive.
  fmin : float, optional. The default is 70.
  path_to_simulations : str, optional. The default is './simulations/'.
  path_to_figures : str, optional. The default is './figures/'.

  Returns
  -------
  None. (figure is stored in path_to_figures)

  '''
  print('\n\nPlotting Fig 2...')
  # load transient drive data
  traj = pypet_load_trajectory(traj_hash=traj_trans_hash, path_to_simulations=path_to_simulations) 
  exploration_overview = pypet_get_exploration_overview(traj)
  ramp_time = exploration_overview['ramp_time'].unique()
  network_size = exploration_overview['Nint'].unique()
  col = ['royalblue', 'skyblue', 'dimgrey'] # ['skyblue', 'steelblue', 'dimgrey'] #color_gradient('skyblue', 'grey', 3) #['skyblue', 'steelblue', 'dimgrey'] # colors coding network size
  # linestyle= ['-', '-', '-']
  
  width_ratios = list(ramp_time + traj.plateau_time/2)
  ifa_stats = pd.read_csv(pypet_get_trajectoryPath(traj_hash = traj_trans_hash, path_to_simulations=path_to_simulations) \
                          # path_to_simulations=traj_trans_path)
                          + 'analysis_IFA_fmin-{}/data_ifa_h{}_summary.csv'.format(fmin, traj_trans_hash), index_col=0, squeeze=True)
  ifa_stats['slope'] = (traj.input.peak-traj.input.baseline)/ifa_stats.ramp_time
  
  # load constant drive data
  traj_stat = pypet_load_trajectory(traj_hash = traj_stat_hash, path_to_simulations=path_to_simulations)  
  res = traj_stat.results.summary.scalarResults
  level_stat = res['parameters.input.level']
  fnet_stat = res['freq_net']
  Ihopf = traj_stat.linear_stability.Icrit_nA
  
  
  with plt.rc_context({"axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 8, "font.size": 8, \
                       "xtick.labelsize": 6, "ytick.labelsize": 6}):
    fig_width= plos_width_fullpage # width_a4_wmargin 
    fig_height = fig_width*.7
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios = [1, 1], height_ratios=[2,1], wspace=.5, hspace=.35)
    
    gs_A = gs[0,0].subgridspec(5, 2, width_ratios=[15,1], height_ratios=[3,1,1,2,1], wspace=.05)
    gs_right = gs[0,1].subgridspec(2,2,width_ratios=[1,1], height_ratios=[2.5,1], hspace=.5)
    gs_B = gs_right[0,:].subgridspec(2,1, height_ratios=[3, 1]) # first one empty space
    ax_B = gs_B.subplots()#sharex=True)
    
    ax_C = fig.add_subplot(gs_right[1,0])
    despine(ax_C)
    
    gs_D = gs[1,:].subgridspec(2, len(ramp_time), width_ratios = width_ratios, height_ratios=[3,1])
    ax_D = gs_D.subplots()#sharey='row', sharex='col')
    
    ax_B[0].text(-1.75, 1, string.ascii_uppercase[0], transform=ax_B[0].transAxes, size=panel_labelsize, weight='bold')
    ax_B[0].text(-.25, 1, string.ascii_uppercase[1], transform=ax_B[0].transAxes, size=panel_labelsize, weight='bold')
    ax_B[0].text(-.25, -.9, string.ascii_uppercase[2], transform=ax_B[0].transAxes, size=panel_labelsize, weight='bold')
    ax_B[0].text(-1.75, -2.1, string.ascii_uppercase[3], transform=ax_B[0].transAxes, size=panel_labelsize, weight='bold')
#    ax_B[0].text(-1.75, -4, string.ascii_uppercase[4], transform=ax_B[0].transAxes, size=panel_labelsize, weight='bold')
    
    # --- fill figure
    Nex = np.max(network_size)
    # --- A ----
    run_idx = pypet_find_runs(traj, ['ramp_time', 'Nint'], lambda r, n: (r < np.sort(ramp_time)[1]) & (n==Nex)) # run indices for steepest ramp
    traj.v_idx = run_idx[0]
    stimulus = traj.derived_parameters.runs[traj.v_crun]['stim_plot']
    
    fig, gs_A = plot_ifa_example_run(traj, run_idx[0], fmin=fmin, fig=fig, gs=gs_A)
    traj.f_restore_default()
    
    # --- B ------
    ifreq_t = pypet_get_from_runs(traj, 'network.ifreq_discr_t', run_idx=run_idx) # dict 
    ifreq = pypet_get_from_runs(traj, 'network.ifreq_discr', run_idx=run_idx) # dict

    traj.v_idx = run_idx[0]
    ifa_slope = float(ifa_stats.loc[np.isclose(ifa_stats.ramp_time, np.min(ramp_time)) & (ifa_stats.Nint==Nex), 'ifa_slope'])
    ifa_intercept = float(ifa_stats.loc[np.isclose(ifa_stats.ramp_time, np.min(ramp_time)) & (ifa_stats.Nint==Nex), 'ifa_intercept'])
    
    ax_B = plot_ifa_linreg(ax_B, ifreq, ifreq_t, stimulus, traj.dt, traj.analysis.ifreq_targetwindow, \
                            fmin=fmin, inputunit=traj.input.unit, ifa_slope = ifa_slope, ifa_intercept = ifa_intercept, tpad=-10, flim=420, \
                            y_text=1, va_text='top')
    # add asymptotic network freqs:
    res_N = res[res['parameters.network.Nint'] == Nex].copy() # stationary simulation for network size N
    level_stat = res_N['parameters.input.level']
    fnet_stat = res_N['freq_net']
  
    # --- add asymptotic curve --- 
    fnet_stat_interp = np.interp(stimulus, level_stat, fnet_stat)
    fnet_stat_interp[stimulus<Ihopf] = nan
    t = np.arange(stimulus.size)*traj.dt - np.mean(traj.analysis.ifreq_targetwindow)
    ax_B[0].plot(t, fnet_stat_interp, 'k-', zorder=1e5)
      
    #--- C ---
    markers = ['x', 'o', '*']
    for ni, N in enumerate(network_size):
      if ni==1:
        ax_C.plot(ifa_stats.loc[ifa_stats.Nint==N, 'slope'], ifa_stats.loc[ifa_stats.Nint==N, 'ifa_slope'], \
                  marker=markers[ni], color=col[ni], linestyle='', ms=7-ni, zorder=ni, label='N={}'.format(N), fillstyle='none')
      else:
        ax_C.plot(ifa_stats.loc[ifa_stats.Nint==N, 'slope'], ifa_stats.loc[ifa_stats.Nint==N, 'ifa_slope'], \
                  marker=markers[ni], color=col[ni], linestyle='', ms=7-ni, zorder=ni, label='N={}'.format(N))
#    ifa_stats.pivot(index='slope', columns='Nint', values='ifa_slope').plot(ax=ax_C, linestyle='', marker='x', ms=6, color=col, legend=False)
    ax_C.set_xlabel('slope of drive [nA/ms]')
    ax_C.set_ylabel('$\chi_\mathrm{IFA}$ [Hz/ms]')
    ax_C.set_ylim([ifa_stats.ifa_slope.min()*1.1, 0.5])
    gridline(0, ax_C, 'y')
    ax_C.set_xlim([0,0.06])
    # ax_C.legend(labelspacing=.2, bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0., edgecolor='dimgrey')
    
    traj.f_restore_default()
      
    #--- D --- 
    handles_inst, handles_stat = [], []
    ls =  ['--', 'dashdot', '-']
    for i in range(ramp_time.size):
      for ni, N in enumerate(network_size):
        run_idx = pypet_find_runs(traj, ['ramp_time', 'Nint'], lambda r, n: np.isclose(r, ramp_time[i]) and np.isclose(n, N)) # run indices for steepest ramp
        ifreq_t = pypet_get_from_runs(traj, 'network.ifreq_discr_t', run_idx=run_idx) # dict 
        ifreq = pypet_get_from_runs(traj, 'network.ifreq_discr', run_idx=run_idx) # dict
        traj.v_idx = run_idx[0]
        stimulus = traj.derived_parameters.runs[traj.v_crun]['stim_plot']
        
        ifa_slope = float(ifa_stats.loc[np.isclose(ifa_stats.ramp_time, ramp_time[i]) & (ifa_stats.Nint==N), 'ifa_slope'])
        ifa_intercept = float(ifa_stats.loc[np.isclose(ifa_stats.ramp_time, ramp_time[i]) & (ifa_stats.Nint==N), 'ifa_intercept'])
        
        ax_D[:,i] = plot_ifa_linreg(ax_D[:,i], ifreq, ifreq_t, stimulus, traj.dt, traj.analysis.ifreq_targetwindow, \
                                      fmin=fmin, inputunit=traj.input.unit, ifa_slope = ifa_slope, ifa_intercept = ifa_intercept, \
                                      color_dots=col[ni], color_line=col[ni],\
                                      ms=1, tpad=-10, flim=420, print_slope=False)
        if not i:
          handles_inst.append(Patch(facecolor=col[ni], label=r'$N=10^{}$'.format(int(np.log10(N)))))       
          handles_stat.append(Line2D([0], [0], linestyle=ls[ni], color='k', label='$f_\mathrm{net}^\mathrm{\infty}$'*(not ni)))
        # add asymptotic curve 
        res_N = res[res['parameters.network.Nint'] == N].copy() # stationary simulation for network size N
        level_stat = res_N['parameters.input.level']
        fnet_stat = res_N['freq_net']
      
        # --- add asymptotic curve --- 
        fnet_stat_interp = np.interp(stimulus, level_stat, fnet_stat)
        fnet_stat_interp[stimulus<Ihopf] = nan
        t = np.arange(stimulus.size)*traj.dt - np.mean(traj.analysis.ifreq_targetwindow)
        ax_D[0,i].plot(t, fnet_stat_interp, 'k', linestyle=ls[ni], label='$f_\mathrm{net}^\mathrm{\infty}$', zorder=1e5)
#     # format
      if i:
        for j in range(2):
          plt.setp(ax_D[j,i].get_yticklabels(), visible=False)
          ax_D[j,i].set_ylabel('')        
    traj.f_restore_default()
        
    # ax_D[0,-1].legend(labelspacing=.2, bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0., handlelength=1)
    for i in range(2):
      match_axis_limits_new(ax_D[i,0], ax_D[i,1], ax_D[i,2], axis='y', set_to='max')
      
      
    # joint legend for C and D 
    # handles_D, labels = ax_D[0,-1].get_legend_handles_labels()
    # handles_D.append(Line2D([0], [0], linestyle='-', color=my_colors['fnet'], label=r'$f_\mathrm{net}^\mathrm{\infty}$'))
    ax_C.legend(handles=handles_inst + handles_stat, labelspacing=.2, bbox_to_anchor=(1.2, 1), loc='upper left', borderaxespad=0., handlelength=1.3, ncol=2, edgecolor='dimgrey')            
    
    fig.savefig(path_to_figures+'Fig2.pdf', bbox_inches='tight')
    fig.savefig(path_to_figures+'Fig2.tif', bbox_inches='tight')
    
  return fig

#%% Fig 3
def plot_figure_3(traj_hash, run=None, reset=True, n_hist=6, marked_cycle=1, \
                                        path_to_simulations='./simulations/', path_to_figures = './figures/'): 
  '''
  Plot Fig 3.

  Parameters
  ----------
  traj_hash : Hash for simulation data, the parameters of which should be used.
  run : int, optional. The parameters (drive) of which run of the simulation file should be used? The default is None.
  reset : bool, optional. Include population reset. The default is True.
  n_hist : bool, optional. Number of snapshots to show in Aiii, Biii. The default is 6.
  marked_cycle : int, optional. Placement of bar in Bi. The default is 1.
  path_to_simulations : str, optional. The default is './simulations/'.
  path_to_figures : str, optional. The default is './figures/'.

  Returns
  -------
  None. (figure is stored in path_to_figures)
  '''
  print('\n\nPlotting Fig 3...')
  # --- load data -----------------------------------------------------------------------------------------------
  traj =  pypet_load_trajectory(traj_hash = traj_hash, path_to_simulations = path_to_simulations)
  
  Ihopf_N = traj.linear_stability.Icrit_nA
  try:
    IPSPint = traj.IPSPint
  except:
    IPSPint = traj.derived_parameters.runs.run_00000000.IPSPint
  
  traj.v_idx = run
    
  if traj.level < Ihopf_N-1e-5:
    raise ValueError('Choose a run with exc drive larger than the bifurcation value!')
    
  # --- analytical traces -----------
  Delta, tm = traj.tl, traj.tm
  K, D, Iext, Vr \
  = rescale_params_dimless(traj.Vthr, traj.Vreset, traj.E_rest, IPSPint, traj.input.level, traj.input.Ie_sig, traj.Nint, traj.C, traj.tm)
  
  
  # --- numerical integration traces 
  t, mu, r, ix_max, ix_min_real, mu_max, mu_reset, mu_min_real, ix_min_theory, mu_min_theory, toff_theory \
  = integrate_dde_numerically(Iext, D, Delta, K, tm, Vr = Vr, tmax = 25, reset=reset, plot=False)[:-2]
  # cut off transients from first cycle: 
  t = t[ix_min_theory[1]:] - t[ix_min_theory[1]] # set new time 0
  mu = mu[ix_min_theory[1]:]
  r = r[ix_min_theory[1]:] 
  ix_min_theory = ix_min_theory[1:] - ix_min_theory[1]
  
  # restrict to one cycle: 
  t_1cyc, mu_1cyc, r_1cyc, mu_min_1cyc, mu_max_1cyc, mu_reset_1cyc, t_off_1cyc, mu_min_real_1cyc \
    = dde_extract_last_cycle(t, mu, r, ix_min_theory, mu_max, mu_min_theory, mu_min_real, mu_reset,  toff_theory)
  
  # --- prep figure -----------------------------------------------------------------------------------------------
  with plt.rc_context({"axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 8,"font.size": 8,\
                         "xtick.labelsize": 5, "ytick.labelsize": 5}):
    fig_width = plos_width_fullpage #width_a4_wmargin
    fig_height = fig_width*.5
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs_main = gridspec.GridSpec(1, 2, figure=fig, wspace=.6) #, height_ratios=[1.5,4,2], wspace=0.05, hspace=.6)
    gs_sub, gs_aux, gs = [None]*2, [None]*2, [[None]*3]*2
    ax = [[None, None, None], [None, None, None]]
    
    for i in range(2):
      gs_sub[i] = gs_main[0,i].subgridspec(1,2, width_ratios=[2,1], wspace=.6)
      gs_aux[i] = gs_sub[i][0].subgridspec(2,1, height_ratios=[1,2], hspace=.5)
      gs[i][0] = gs_aux[i][0].subgridspec(2,2, height_ratios=[1,2], width_ratios=[20,1], wspace=.05)
      gs[i][1] = gs_aux[i][1].subgridspec(3,2, height_ratios=[1,2,.5], width_ratios=[20,1], wspace=.05)
      gs[i][2] = gs_sub[i][1].subgridspec(n_hist,1)
      ax[i][0] = gs[i][0].subplots(sharex='col')
      ax[i][1] = gs[i][1].subplots(sharex='col')
      ax[i][2] = gs[i][2].subplots(sharex=True, sharey=True)
      # remove unneded colorbar axes:
      ax[i][0][0,1].remove()
      ax[i][1][0,1].remove()
      ax[i][1][2,1].remove()
    
      # remove spines 
      despine([ax[i][0][1,0], ax[i][1][1,0], ax[i][1][2,0]])
      despine(ax[i][2])
      despine(ax[i][0][0,0], which=['right', 'top', 'bottom'])
      despine(ax[i][1][0,0], which=['right', 'top', 'bottom'])
      
      offset = .6
      ax[i][0][0,0].text(-offset, 1.2, string.ascii_uppercase[i]+'i', transform=ax[i][0][0,0].transAxes, size=panel_labelsize, weight='bold')
      ax[i][1][0,0].text(-offset, 1.21, string.ascii_uppercase[i]+'ii', transform=ax[i][1][0,0].transAxes, size=panel_labelsize, weight='bold')
      ax[i][0][0,0].text(1.3, 1.2, string.ascii_uppercase[i]+'iii', transform=ax[i][0][0,0].transAxes, size=panel_labelsize, weight='bold')
    
    # --- fill figure -----------------------------------------------------------------------------------------------
    # common max for colorbar
    cbar_max = 1/np.sqrt(2*pi*D) # less than the max of the theoretical gaussian, previously 3 times
    # --- THEORY
    ax[1][0] = aux_plot_DDE_traces_whist(ax[1][0], t, r, mu, D, Vr, mu_min_theory, mu_max, pmax=cbar_max)
      
    
    ax[1][1] = aux_plot_DDE_traces_whist_1cyc(ax[1][1], t_1cyc, r_1cyc, mu_1cyc, t_off_1cyc, mu_max_1cyc, mu_reset_1cyc, mu_min_1cyc,\
                                              D, Delta, Vr, vmax=cbar_max)
    
    ax[1][2] = aux_plot_Vhist_theory(ax[1][2], t_1cyc, mu_1cyc, D, Vr, reset=reset)
    
    # --- SIMULATION
    ax[0], t_marked_cycle_sim \
    = aux_plot_gauss_analysis_spknet(ax[0], traj, n_hist, D, cbar_max=cbar_max, marked_cycle=marked_cycle, path_to_simulations=path_to_simulations)
    
    # match axis limits
    ax[1][0][0,0].set_ylim([0, np.max(r) + 150])
    ax[1][1][0,0].set_ylim([0, np.max(r) + 150])
    match_axis_limits_new(ax[0][0][0,0], ax[1][0][0,0], axis='y')
    match_axis_limits_new(ax[0][0][1,0], ax[1][0][1,0], axis='y')
    match_axis_limits_new(ax[0][0][0,0], ax[1][0][0,0], axis='x', set_to='min')
    # match_axis_limits_new(ax[0][0][1,0], ax[1][0][1,0], axis='x', set_to='min')
    match_axis_limits_new(ax[0][1][0,0], ax[1][1][0,0], axis='y')
    match_axis_limits_new(ax[0][1][1,0], ax[1][1][1,0], axis='y')
    match_axis_limits_new(ax[0][1][2,0], ax[1][1][2,0], axis='y')
    match_axis_limits_new(ax[0][2][0], ax[1][2][0], axis='xy')
    
    # add vertical gridlines to Aiii, Biii that end below text
    for ax3 in [ax[0][2], ax[1][2]]:
      for axi in ax3:
        axi.axvline(1, ymax=.8, zorder=9, lw=.5, linestyle=':', color='gray')
        axi.axvline(Vr, ymax=.8, zorder=9, lw=.5, linestyle=':', color='gray')
    
    
    # same labels
    for i in range(2):
      if not i:
        ax[i][0][0,0].set_ylabel('$r_N$')
        ax[i][1][0,0].set_ylabel('rate $r_N$\n[spk/s]\n')
      else:
        ax[i][0][0,0].set_ylabel(r'$r$')
        ax[i][1][0,0].set_ylabel('rate $r$\n[spk/s]\n')
      ax[i][0][1,0].set_ylabel(r'$V$')
      ax[i][0][1,0].set_xlabel('time [ms]')
      ax[i][1][1,0].set_ylabel('membrane \n potential $V$')
      ax[i][1][2,0].set_ylabel(r'SD($V$)')
      ax[i][1][2,0].set_xlabel('time within cycle [ms]')
      ax[i][2][-1].set_ylabel(r'$p(V,t)$')
      ax[i][2][-1].set_xlabel(r'$V$')
    
    dt = np.mean(np.diff(t))
    
    # with bar
    ax[1][0][0,0].fill_between([ix_min_theory[marked_cycle]*dt, ix_min_theory[marked_cycle+1]*dt], \
                                ax[1][0][0,0].get_ylim()[1]-50, ax[1][0][0,0].get_ylim()[1], color='k')
    ax[1][1][0,0].fill_between([0, toff_theory[-1]], \
                                ax[1][1][0,0].get_ylim()[1]-50, ax[1][1][0,0].get_ylim()[1], color='lightgrey')
    ax[1][1][0,0].fill_between([toff_theory[-1], toff_theory[-1]+Delta], \
                                ax[1][1][0,0].get_ylim()[1]-50, ax[1][1][0,0].get_ylim()[1], color='dimgrey')
    ax[1][1][0,0].text(toff_theory[-1]/2, ax[1][1][0,0].get_ylim()[1], r'$t_\mathrm{off}$', ha='center', va='bottom' )
    ax[1][1][0,0].text(toff_theory[-1] + Delta/2, ax[1][1][0,0].get_ylim()[1], r'$\Delta$', ha='center', va='bottom' )

  traj.f_restore_default()
  
  fig.savefig(path_to_figures + 'Fig3.pdf', bbox_inches = 'tight')
  fig.savefig(path_to_figures + 'Fig3.tif', bbox_inches = 'tight')
  
  return 

def aux_plot_DDE_traces_whist(ax, t, rate, mu, D, Vr, mu_min, mu_max, t_axislimit= 20, dv=1e-3, pmax=None):
  ax[0,0].plot(t, rate, 'k')
  
  v = np.arange(np.min(mu)-4*np.sqrt(D), np.max(mu)+4*np.sqrt(D), dv) # voltage array
  p = get_gauss(v, mu, np.sqrt(D)).T
  if not pmax:
    pmax = np.max(p)
  im = ax[1,0].imshow(p, origin='lower', extent=(t[0], t[-1], v[0], v[-1]),\
                    aspect='auto', cmap=colormap_voltage_hist, \
                    norm=matplotlib.colors.Normalize(vmin=0, vmax = pmax),\
                    interpolation='none')
  ax[1,0].plot(t, mu, color='k', lw=.7)
  gridline([Vr,1], ax[1,0],'y', zorder=10)
  # colorbar
  cb = plt.colorbar(im, cax= ax[1,1]) #ticks=[0, 1e-1, np.floor(100*np.max(p)/2)/100]
  cb.set_label(label=r'$p(V,t)$', labelpad=-.08)
  cb.ax.tick_params(axis='y', pad=1) #, direction='out')
  
  ax[0,0].set_ylim([0, np.max(rate)+50])
  ax[1,0].set_xlim([t[0], t_axislimit])
  ax[1,0].set_ylim([-1.5, 1.5]) #[np.floor(mu_min[-1]-3*np.sqrt(D)), np.ceil(mu_max[-1]+3*np.sqrt(D))]) 
  ax[0,0].set_ylabel(r'$r$')
  ax[1,0].set_ylabel(r'$V$')
  ax[1,0].set_xlabel('time [ms]')
  return ax

def aux_plot_DDE_traces_whist_1cyc(ax, t, r, mu, t_off, mu_max, mu_reset, mu_min, \
                                   D, Delta, Vr, dv=1e-3, vmax=None):
  ''' plot a single oscillation cycle and mark quantities of interest (Fig 3Bii) 
  '''
  v = np.arange(mu_min-4*np.sqrt(D), mu_max+4*np.sqrt(D), dv) # voltage array
  p = get_gauss(v, mu, np.sqrt(D)).T
  if not vmax:
    vmax = np.max(p)
  im = ax[1,0].imshow(p, origin='lower', extent=(t[0], t[-1], v[0], v[-1]),\
                      aspect='auto', cmap=colormap_voltage_hist, \
                      norm=matplotlib.colors.Normalize(vmin=0, vmax = vmax),\
#                      norm=matplotlib.colors.PowerNorm(gamma=.2, vmax = vmax),\
                      interpolation='none')
  # colorbar
  cb=plt.colorbar(im, cax= ax[1,1])#.set_label(label='$p(V,t)$', labelpad=-.1) # ticks=[0, 1e-1, np.floor(100*np.max(p)/2)/100]
  cb.set_label(label=r'$p(V,t)$', labelpad=-.08)
  cb.ax.tick_params(axis='y',  pad=1)#, direction='out')# add grid
      
  ## --- analytical traces
  ax[0,0].plot(t, r, color=my_colors['fnet'], label='analyt')
  ax[0,0].set_ylabel('rate $r$ \n [spks/sec]')
  ax[0,0].set_ylim(bottom=0)
  
  gridline([1, Vr], ax[1,0], 'y', zorder=10)
  ax[1,0].plot(t, mu, color='k', label='$\mu(t)$')
  ax[1,0].plot(t_off, mu_max, '.', ms=4, markeredgecolor=my_colors['max'], fillstyle='none',  label='$\mu_\mathrm{max}$')
  ax[1,0].plot(t_off, mu_reset, '.', ms=4, markeredgecolor=my_colors['reset'], fillstyle='none', label='$\mu_\mathrm{reset}$') #markeredgewidth= 2,
  ax[1,0].plot(0, mu_min, '.', ms=4, markeredgecolor=my_colors['min'], fillstyle='none',  label='$\mu_\mathrm{min}$')
  ax[1,0].plot(t_off+Delta, mu_min, '.', ms=4, markeredgecolor=my_colors['min'], fillstyle='none')
  
  ax[2,0].axhline(np.sqrt(D), color='k')
  
  if Vr != 0:
    yticks = [-1, Vr, 0, 1]
    yticklabels= ['-1', '$V_R=$'+'{:.0f}'.format(Vr), '$E_L=0$', '$V_T=1$']
  else:
    yticks = [-1, Vr, 1]
    yticklabels= ['-1', '$V_R=$'+'{:.0f}'.format(Vr), '$V_T=1$']
  ax[1,0].set_yticks(yticks)
  ax[1,0].set_yticklabels(yticklabels)
  ax[1,0].set_ylabel('membrane \npotential $V$')#, labelpad=-.1)
  ax[1,0].set_ylim([mu_min-4*np.sqrt(D), mu_max+4*np.sqrt(D)]) #mu_max+3*np.sqrt(D)+.2]) #bottom=1.1*(mu_min-3*np.sqrt(D))) 
  ax[2,0].set_ylim([0, 2*np.sqrt(D)])
  ax[2,0].set_yticks([0,np.sqrt(D)])
  ax[2,0].set_yticklabels([0,r'$\sqrt{D}=$'+'{:.1f}'.format(np.sqrt(D))])
  ax[2,0].set_ylabel('SD($V$)')
  ax[2,0].set_xlabel('time within cycle [ms]')
  ax[2,0].set_xlim([0, t[-1]])
  
  return ax

def aux_plot_Vhist_theory(ax, t, mu, D, Vr, dv=1e-3, reset=True, eps=0):
  # show 3 examples on downstroke, rest on upstroke:
  t_off_ix = np.argmax(mu)
  ix_ex = np.round(np.concatenate((np.linspace(0, t_off_ix, len(ax)-3, endpoint=False) , np.linspace(t_off_ix, len(t)-1, 3, endpoint=True)))).astype(int)
  v = np.arange(np.min(mu)-3*np.sqrt(D), np.max(mu)+3*np.sqrt(D), dv) # v array
  
  for i, ix in enumerate(ix_ex):
    p = get_gauss(v, mu[ix], np.sqrt(D))
    ax[i].fill_between(v, p, color='darkgrey')
    ax[i].text(1,1,'t={:.2f}ms'.format(t[ix]),transform=ax[i].transAxes, va='top', ha='right', fontsize=plt.rcParams['xtick.labelsize'])
    # txt.set_bbox(dict(facecolor='w', alpha=1, edgecolor='none'))
    if reset and ix == t_off_ix: # show reset
      ax[i].plot(v, p, color=my_colors['max'])  
      p_r = get_gauss(v, mu[ix+1], np.sqrt(D))
      ax[i].plot(v, p_r, color=my_colors['reset'])
      ax[i].arrow(mu[ix]-eps, np.max(p)/2, mu[ix+1]-mu[ix]+2*eps, 0, color='k', head_width=0.2, head_length=(mu[ix]-mu[ix+1]-2*eps)/2,\
                  length_includes_head=True, zorder=10)
    if i in [0, len(ix_ex)-1]:
      ax[i].plot(v, p, color=my_colors['min'])
  
  ax[-1].set_ylim([0, 1.3*np.max(p)])    
  ax[-1].set_xlabel('$V$')
  ax[-1].set_ylabel(r'$p(V,t)$', labelpad=-.1)
  for i in range(len(ax)-1):
    plt.setp(ax[i].get_yticklabels(), visible=False)
  return ax

def aux_plot_gauss_analysis_spknet(ax, traj, n_hist, D, cbar_max=None, marked_cycle=None, path_to_simulations = './simulations/'):

  # --- load data
  df_N = pd.read_hdf(pypet_get_trajectoryPath(traj_hash = traj.hash, path_to_simulations=path_to_simulations) \
                     + 'analysis_membrane_potential_dynamics/analysis_membrane_potential_dynamics_h{}'.format(traj.hash)+'.hdf5')
  mu_min_N = (df_N.loc[traj.v_idx,'mu_min']-traj.E_rest)/(traj.Vthr-traj.E_rest)
  # mu_min_N = -.5
  Vr = (traj.Vreset-traj.E_rest)/(traj.Vthr-traj.E_rest)
  # --- numerical traces
  # averaged in one cycle
  sample, v_hist, v_std, v_bincenters, freq_net_N, popspk_av \
  = traj.results.crun.gauss.sample, traj.results.crun.gauss.v_hist, traj.results.crun.gauss.v_std, \
    traj.results.crun.gauss.v_bincenters, traj.results.crun.network.freq_net, traj.results.crun.gauss.popspk_av
  
  # over full time
  v_hist_full, v_binedges_full = getVtraces('micro', traj.results.crun.raw.v, Vthr=traj.Vthr, volt_max = traj.Vthr, \
                                            vbins=30, density=True) 
  
  # --- rescale to make dimensionless  
  v_bincenters = (v_bincenters-traj.E_rest)/(traj.Vthr-traj.E_rest)
  v_hist = v_hist*(traj.Vthr-traj.E_rest) # preserve integral = 1 in rescaled density 
  v_std = v_std/(traj.Vthr-traj.E_rest)
  v_hist_full = v_hist_full*(traj.Vthr-traj.E_rest) # preserve integral = 1 in rescaled density
  v_binedges_full = (v_binedges_full-traj.E_rest)/(traj.Vthr-traj.E_rest)
  
  # --- add phase 0 to end as well
  popspk_av_pad = np.append(popspk_av, popspk_av[0])
  v_hist_pad = np.vstack((v_hist, v_hist[0,:]))
  v_std_pad = np.append(v_std, v_std[0])
  
  # loaction of approx mu_max after rotation s.th. mu_min at position 0
  n = sample.shape[0]
  
  period_N = 1000/freq_net_N
  t_samples = np.linspace(0, period_N, n+1, endpoint=True)
  dt_samples = np.mean(np.diff(t_samples))
  dv = np.mean(np.diff(v_bincenters))
  v_hist_sim_extent = [-dt_samples/2, period_N+dt_samples/2, v_bincenters[0]-dv/2, v_bincenters[-1]+dv/2]  
  
  
  # --- (i) full simulation  
  ax[0][0,0] = plot_rate(ax[0][0,0], traj.results.crun.raw.LFP_smooth, traj.dt, t0 =traj.v_recordwindow[0])
  ax[0][1,:] = plot_vhist(ax[0][1,0], ax[0][1,1], Vthr=1, Vreset=Vr, Vdistr=v_hist_full, vbinedges=v_binedges_full,\
                          v_recordwindow=traj.v_recordwindow,  t0 =traj.v_recordwindow[0], cbar_max=cbar_max, cbar_label=r'$p(V,t)$')
    
  gridline(1,ax[0][1,0],'y', zorder=10)
  
  # --- (ii) 1 cycle
  ax[1][0,0].plot(t_samples, popspk_av_pad, '.-', color=my_colors['fnet'])
  if not cbar_max:
    cbar_max = np.max(v_hist_pad) 
  
  im = ax[1][1,0].imshow(v_hist_pad.T, origin='lower', extent=v_hist_sim_extent,\
                            aspect='auto', cmap=colormap_voltage_hist, 
                            norm=matplotlib.colors.Normalize(vmin=0, vmax = cbar_max),\
                            interpolation='none') 
  # colorbar
  if np.max(v_hist_pad) > cbar_max:
    cb = plt.colorbar(im, cax= ax[1][1,1], extend='max')#.set_label(label='$p(V,t)$', labelpad=-.1)
  else:
    cb = plt.colorbar(im, cax= ax[1][1,1])#.set_label(label='$p(V,t)$', labelpad=-.1)
  cb.set_label(label=r'$p(V,t)$', labelpad=-.08)
  cb.ax.tick_params(axis='y', pad = 1)
  # cb.ax.set_ylim([0, np.max(v_hist_pad)])
  gridline([Vr,1], ax[1][1,0], 'y', zorder=10)
  
  if Vr != 0:
    yticks = [-1, Vr, 0, 1]
    yticklabels= ['-1', '$V_R=$'+'{:.0f}'.format(Vr), '$E_L=0$', '$V_T=1$']
  else:
    yticks = [-1, Vr, 1]
    yticklabels= ['-1', '$V_R=$'+'{:.0f}'.format(Vr), '$V_T=1$']
  ax[1][1,0].set_yticks(yticks)
  ax[1][1,0].set_yticklabels(yticklabels)
  
  
  ax[1][2,0].plot(t_samples, v_std_pad, 'k.-')
  gridline(np.sqrt(D), ax[1][2,0], 'y', linestyle='--')
  ax[1][2,0].set_yticks([0, np.sqrt(D)])
  ax[1][2,0].set_yticklabels([0,r'$\sqrt{D}=$'+'{:.1f}'.format(np.sqrt(D))])
  ax[1][2,0].set_ylim([0, 1.1*np.max([np.max(v_std_pad), np.sqrt(D)])])
  
  # --- (iii) vhist movies
  ix_ex = np.round(np.linspace(0, n, n_hist, endpoint=True)).astype(int)
  
  for i, ix in enumerate(ix_ex):
    ax[2][i].fill_between(v_bincenters, v_hist_pad[ix,:], color='darkgrey')
    ax[2][i].text(1,1,'t={:.2f}ms'.format(t_samples[ix]),transform=ax[2][i].transAxes, va='top', ha='right', fontsize=plt.rcParams['xtick.labelsize'])
    # gridline([1, Vr], ax[2][i], 'x', zorder=9)
  ax[2][-1].set_ylim([0, 1.3*np.max(v_hist_pad)])    
  ax[2][-1].set_xlabel('$V$')
  ax[2][-1].set_ylabel(r'$p(V,t)$', labelpad=-.1)
  for i in range(len(ax[2])-1):
    plt.setp(ax[2][i].get_yticklabels(), visible=False)
  
  # --- mark one cycle and zoom in:
  if marked_cycle:
    # beginning and end of cycle "marked_cycle" (counted back from end of sim)  
    cycle_start = (traj.results.crun.gauss.sample[0,:]*traj.dt)[traj.results.crun.gauss.sample[0,:]*traj.dt>=traj.v_recordwindow[0]]
    t_cycle = cycle_start[marked_cycle:marked_cycle+2]-traj.v_recordwindow[0] #traj.results.crun.gauss.sample[0,marked_cycle:marked_cycle+2]*traj.dt-traj.v_recordwindow[0
  
  ax[1][1,0].plot([t_samples[0], t_samples[-1]], [mu_min_N]*2, '.', ms=4, markeredgecolor=my_colors['min'], fillstyle='none')
  
  for i in range(3):
    ax[1][i,0].set_xlim([-dt_samples/2, t_samples[-1]+dt_samples/2])
  
  return ax, t_cycle

#%% Fig 4
def plot_figure_4(traj_hash, reset=True, run_max=None, path_to_simulations='./simulations/', path_to_figures = './figures/'): 
  '''
  Plot Fig 4.

  Parameters
  ----------
  traj_hash : Hash to simulation data that shall be compared to Gaussian-drift approx.
  reset : bool, optional. Reset. The default is True.
  run_max : int, optional. Last run of simulation data to include. The default is None.
  path_to_simulations : str, optional. The default is './simulations/'.
  path_to_figures : str, optional. The default is './figures/'.

  Returns
  -------
  None. (figure is stored in path_to_figures)

  '''
  print('\n\nPlotting Fig 4...')
  # oad parameters of reference simulation:
  traj_path = pypet_get_trajectoryPath(traj_hash = traj_hash, path_to_simulations = path_to_simulations)
  path_stat_info = traj_path + 'info.csv'
  if not os.path.exists(path_stat_info):
    store_info_cyclostat_lif_simulation(traj_hash=traj_hash, path_to_simulations=path_to_simulations)
  info = pd.read_csv(traj_path + 'info.csv', index_col=0, squeeze=True, header=None)
  D, Delta, K, tm, Vr \
  = info['D'], info['Delta'], info['K'], info['tm'], info['Vr']
  
  file_gaussian_drift_approx = traj_path + 'gaussian_drift_approx/gaussian_drift_approx_analytical_reset-{}_h{}'.format(reset, traj_hash)+'.hdf5'
  # load results of gaussian drift approximation
  if not os.path.exists(file_gaussian_drift_approx):
    pypet_gaussian_drift_approx_stat(traj_hash, reset=reset, save2file=True)
  df_T = pd.read_hdf(file_gaussian_drift_approx) # theoretical results

  # load results of spiking net
  traj = pypet_load_trajectory(traj_hash=traj_hash, path_to_simulations=path_to_simulations)
  df_N = pd.read_hdf(traj_path + 'analysis_membrane_potential_dynamics/analysis_membrane_potential_dynamics_h{}'.format(traj_hash)+'.hdf5')
  # add freq_net etc to df_N data frame
  res = traj.results.summary.scalarResults
  df_N['f_net'], df_N['f_unit'], df_N['Iext'], df_N['sat']\
  = res['freq_net'].values, res['freq_unit_mean'].values, df_N['level']*traj.tm/(traj.C/1000), res['saturation'].values
  Ihopf = traj.linear_stability.Icrit_nA*traj.tm/(traj.C/1000)  # mV, 
  # rescale voltages
  scaling_factor = 1/(traj.Vthr-traj.E_rest)
  Ihopf, df_N['Iext'] = Ihopf*scaling_factor, df_N['Iext']*scaling_factor
  df_N['mu_min'] = (df_N['mu_min']-traj.E_rest)*scaling_factor
  
  osc = df_N.Iext >= Ihopf - 1e-2 # subtract epsilon to avoid numerical errors in <=
  
  # --- indicate Iext_min ?
  Iext_min = get_Iext_lower_bound(D, Delta, K, tm, Vr, reset=True, Iext_min=0, Iext_max=3.5, dI=.1)[0]
  Iext_full = get_pt_fullsynch(D, Delta, K, tm)
  
  # --- construct figure -------
  fig_width = plos_width_text #10*cm
  fig_height = fig_width #10*cm #10.5*cm
  fig, ax = plt.subplots(nrows=3, figsize=(fig_width, fig_height), sharex=True, gridspec_kw={'height_ratios':[2,.5,1]})
  despine(ax)
  # analytical
  ix = df_T.Iext >= Iext_min
  gridline(Iext_min, ax, 'x')
  gridline(Iext_full, ax, 'x', linestyle='--')
  ax[0].plot(df_T[ix]['Iext'], df_T[ix]['f_net'], linestyle= '-', color=my_colors['fnet'], label='network (analyt)')
  ax[0].plot(df_T[ix]['Iext'], df_T[ix]['f_unit'], linestyle= '-', color=my_colors['funit'], label='unit (analyt)')
  ax[1].plot(df_T[ix]['Iext'], df_T[ix]['sat'], linestyle= '-', color='k', label='analyt')
  ax[-1].plot(df_T[ix]['Iext'], df_T[ix]['mu_max'], linestyle= '-', color=my_colors['max'], label='$\mu_\mathrm{max}$ (analyt)')
  ax[-1].plot(df_T[ix]['Iext'], df_T[ix]['mu_reset'], linestyle= '-', color=my_colors['reset'], label='$\mu_\mathrm{reset}$ (analyt)')
  ax[-1].plot(df_T[ix]['Iext'], df_T[ix]['mu_min'], linestyle= '-', color=my_colors['min'], label='$\mu_\mathrm{min}$ (analyt)')
  
  # numerical
  ax[0].fill_between(np.arange(0,1.5*df_N['Iext'].max(), .1), 140, 220, facecolor='lightgray', edgecolor='face', zorder=-20)
  ax[0].plot(df_N[osc]['Iext'], df_N[osc]['f_net'], linestyle= '--', marker='^', color=my_colors['fnet'], label='network (spiking net)')
  ax[0].errorbar(df_N['Iext'], df_N['f_unit'], yerr=res['freq_unit_std'], linestyle= '--', marker='o', color=my_colors['funit'], label='unit (spiking net)')
  ax[0].plot(Ihopf, traj.linear_stability.fcrit, marker='^', linestyle='', fillstyle='none', markeredgecolor='r', zorder=3, label='Hopf bifurcation')# markersize=9,
  ax[0].plot(Ihopf, traj.linear_stability.A0crit, marker='o', linestyle='', fillstyle='none', markeredgecolor='r', zorder = 3)
  ax[1].plot(df_N[osc]['Iext'], df_N[osc]['sat'], linestyle= '--', marker='o', color='k', label='spiking net')
  ax[-1].plot(df_N[osc]['Iext'], df_N[osc]['mu_min'], linestyle= '--', marker='o', color=my_colors['min'], label='$\mu_\mathrm{min}$ (spiking net)')

  # formatting
  gridline([0,1], ax[-1], 'y')
  if run_max:
    Iext_max = np.max([Iext_full,df_N.loc[run_max, 'Iext']])
  else:
    Iext_max = np.max([Iext_full,df_N.iloc[-1]['Iext']])
  ax[0].set_xlim([0, 1.05*Iext_max])
  ax[0].set_ylim([0, np.max([traj.linear_stability.fcrit, df_T[ix]['f_net'].max()]) + 10])
  ax[-1].set_ylim([df_T['mu_min'].min()-.2, df_T['mu_max'].max()+.2])
  ax[0].set_ylabel('frequency [Hz]')
  ax[1].set_ylabel('s')
  ax[1].set_ylim([0,1.2*df_N[:run_max+1]['sat'].max()])
  ax[-1].set_ylabel('membrane \npotential')
  ax[-1].set_xlabel('external drive ' +str_Iext)
  
#  ax[1].legend(loc='upper left', labelspacing=.2, borderaxespad=.1)
#  ax[-1].legend(loc='lower left', labelspacing=.2, borderaxespad=.2)
  ax[-1].set_yticks([-4, -2, Vr, 1])
#  ax[-1].set_yticklabels(['-4', '-2', '$V_R=${:.0f}'.format(Vr), '$V_T=1$'])

  xticks = list(np.arange(0,Iext_max,2)) + [Iext_min,Iext_full]
  xticklabels = [str(int(x)) for x in list(np.arange(0,Iext_max,2))] + ['$I_\mathrm{E}^\mathrm{min}$','$I_\mathrm{E}^\mathrm{full}$']
  ax[-1].set_xticks(xticks)
  ax[-1].set_xticklabels(xticklabels)
  
  for axi in [ax[1], ax[2]]:
    axi.spines['right'].set_visible(False)
    axi.spines['top'].set_visible(False)
  
  # legends
  hnet = Patch(facecolor=my_colors['fnet'], label=r'$f_\mathrm{net}$')
  hunit = Patch(facecolor=my_colors['funit'], label=r'$f_\mathrm{unit}$')
  hhopf = Line2D([], [], color='red', marker='^', linestyle='None', fillstyle='none', markeredgecolor='r',  label='Hopf')
  l2 = ax[0].legend(bbox_to_anchor=(1, 0.9), loc='upper right', labelspacing=.2, borderaxespad=0, handles=[hnet, hunit, hhopf],\
         handlelength=1, framealpha=1, facecolor='w', ncol=3)
  ax[0].add_artist(l2)
  
  handle_1 = Line2D([0], [0], linestyle='-', color='dimgrey', label='theory')
  handle_2 =  Line2D([0], [0], linestyle='--', color='dimgrey', label='simulation ($N=10^{:.0f}$)'.format(np.log10(traj.Nint)))
  ax[0].legend(bbox_to_anchor=(1, 1.), loc='upper right', borderaxespad=0., handlelength=2, \
                handles=[handle_1, handle_2], framealpha=1, facecolor='w', ncol=2)
  
  hmax = Patch(facecolor=my_colors['max'], label=r'$\mu_\mathrm{max}$')
  hreset = Patch(facecolor=my_colors['reset'], label=r'$\mu_\mathrm{reset}$')
  hmin = Patch(facecolor=my_colors['min'], label=r'$\mu_\mathrm{min}$')
  ax[-1].legend(loc='lower left', labelspacing=.2, borderaxespad=.1, handles=[hmax, hreset, hmin], handlelength=1)
  
  fig.savefig(path_to_figures + 'Fig4.pdf', bbox_inches='tight')
  fig.savefig(path_to_figures + 'Fig4.tif', bbox_inches='tight')
  
  return 

#%% Fig 5
def plot_figure_5(traj_hash_stat, example_cycle=1, nsteps=4, \
                                reset=True, fmax=200, fmin=-200, \
                                path_to_simulations = './simulations/', path_to_figures = './figures/'):
  '''
  Plot Fig 5.

  Parameters
  ----------
  traj_hash_stat : Hash to simulation data the parameters of which should be used (D, Delta, K...)
  example_cycle : int optional. Which cycle to show in A. The default is 1.
  nsteps : int, optional. Number of steps in pw constant drive (up and down resp.). The default is 4.
  reset : bool, optional. Reset. The default is True.
  fmax, fmin : float, optional. [Hz] Limit for freq colorbar. The default is +/-200.
  path_to_simulations : str, optional. The default is './simulations/'.
  path_to_figures : str, optional. The default is './figures/'.

  Returns
  -------
  None (figure is stored in path_to_figures)

  '''
 
  print('\n\nPlotting Fig 5...')

  # load parameters from the provided simulationn
  traj_path_stat = pypet_get_trajectoryPath(traj_hash = traj_hash_stat, path_to_simulations = path_to_simulations)
  path_stat_info = traj_path_stat + 'info.csv'
  if not os.path.exists(path_stat_info):
    store_info_cyclostat_lif_simulation(traj_hash=traj_hash_stat, path_to_simulations=path_to_simulations)
  info = pd.read_csv(path_stat_info, index_col=0, squeeze=True, header=None)
  D, Delta, K, tm, Vr, I_full \
  = info['D'], info['Delta'], info['K'], info['tm'], info['Vr'],  info['I_full']
  print('D: {}, Delta: {}, K: {}, tm: {}, Vr: {}, I_full: {}'.format(D, Delta, K, tm, Vr, I_full))
  
  # --- CONSTANT DRIVE 
  Iext_min, mu_min_max = get_Iext_lower_bound(D, Delta, K, tm, Vr, reset=reset, Iext_min=0, Iext_max=3, dI=.1)
  
  # summary IFA figure -- constant drive
  # --- construct figure
  with plt.rc_context({"axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 8,"font.size": 8,\
                       "xtick.labelsize": 6, "ytick.labelsize": 6,\
                        "xtick.direction": "out", "ytick.direction": "out", 'xtick.major.pad': 1, 'ytick.major.pad': 1}):
    fig_width= plos_width_fullpage # width_a4_wmargin # 21*cm
    fig_height = fig_width*.6 #9*cm #10.5*cm
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios = [1, 1, 1], hspace=.35, wspace=.5)#, width_ratios=[5,2], height_ratios=[2,3])
  
    gs_0 = gs[0,:].subgridspec(2,2, height_ratios = [.5,2], hspace=.2, wspace=.1)
    gs_A_sup = gs[1,0].subgridspec(1,2, width_ratios=[30,1])
    gs_A = gs_A_sup[0,0].subgridspec(2, 3, height_ratios = [.5,2], wspace=.5, hspace=.2)
    gs_B = gs[2,0].subgridspec(1, 3, width_ratios = [20, .5, 2], wspace=.1)
    gs_C = gs[1:,1].subgridspec(3, 1, height_ratios=[1,.5,2], hspace=.1)
    
    ax_0 = gs_0.subplots(sharey='row', sharex=True)
    ax_A = gs_A.subplots(sharex=True).T
    ax_B = gs_B.subplots() #[fig.add_subplot(gs_B[0,0]), fig.add_subplot(gs_B[0,1])]
    ax_B[-1].remove()
    ax_C = gs_C.subplots(sharex=True)
    despine(ax_C)
    despine(ax_0, which=['top', 'right', 'bottom'])
    despine(ax_A, which=['top', 'right', 'bottom'])
    despine(ax_A[1:,:], which=['left'])
    
    ax_0[0][0].text(-1.2, 9, 'Ai', transform=ax_A[0][0].transAxes, size=panel_labelsize, weight='bold')
    ax_0[0][0].text(5.2, 9.2, 'Aii', transform=ax_A[0][0].transAxes, size=panel_labelsize, weight='bold')
    ax_A[0][0].text(-1.2, 1.6, 'Aiii', transform=ax_A[0][0].transAxes, size=panel_labelsize, weight='bold')
    ax_B[0].text(-1.2, -.6, 'B', transform=ax_A[0][1].transAxes, size=panel_labelsize, weight='bold')
    ax_C[0].text(2.38, 1.7, 'C', transform=ax_A[-1][0].transAxes, size=panel_labelsize, weight='bold')
    
    # --- fill figure
    # --- B -------------------------------------
    fig, ax_C, ax_B, df, traces_analytical \
    = analysis_IFA_constant_drive_analytical(Iext_min+.6, I_full, nsteps, mu_min_max, D, Delta, K, tm, Vr, reset=True, \
                                             fmax=fmax, fmin=fmin, fig=fig, ax_time = ax_C, ax_phase=ax_B)[1:]
    ax_B[0].set_title('')
    ax_B[0].set_ylim(top=mu_min_max)
    
    # --- new top panel ------------------------------------------------------------
    mu_min_high = df.loc[example_cycle, 'mu_min_start']
    mu_min_low = df.iloc[-1-example_cycle, :].mu_min_start
    Iext_toff = df.loc[example_cycle, 'Iext_toff']
    ax_0 = plot_convergence_to_asymptotic_dynamics(mu_min_high, mu_min_low, Iext_toff, D, Delta, K, tm, Vr, T= 40, ax=ax_0, reset=reset)
    
    # --- A ------------------------------------
    # show traces for constant Iext which was reached in the example_cycle (up vs downstroke)
    mu_min_0_stat = gaussian_drift_approx_stat(Iext_toff, D, Delta, K, tm, Vr=Vr, reset=reset)[0]
    fig, ax_A = visualize_traces_wlineardrive(D, Delta, K, tm, Vr, m = 0, \
                                              mu_min_0=[mu_min_high, mu_min_0_stat, mu_min_low], \
                                              Iext_toff = Iext_toff, \
                                              fig=fig, ax=ax_A, reset=reset, i_ref=1, fmin=fmin, fmax=fmax)
    # add cycle numbers for reference
    ax_A[0][0].set_title('up')# + r'$\mu_\mathrm{min}^0 > \mu_\mathrm{min}(I_\mathrm{E})$')#('upstroke\n' + r'$\mu_\mathrm{min}^0 > \mu_\mathrm{min}(I_\mathrm{E})$'+'\n(cycle {})'.format(example_cycle+1))
    ax_A[0][0].text(0.05,1,'c{}'.format(example_cycle+1), ha='left', va='top', color='dimgrey', fontsize=plt.rcParams['xtick.labelsize'], \
                    transform = ax_A[0][0].transAxes)
    ax_A[2][0].set_title('down')# + r'$\mu_\mathrm{min}^0 < \mu_\mathrm{min}(I_\mathrm{E})$')#('upstroke\n' + r'$\mu_\mathrm{min}^0 > \mu_\mathrm{min}(I_\mathrm{E})$'+'\n(cycle {})'.format(example_cycle+1))
    ax_A[2][0].text(0.05,1,'c{}'.format(2*nsteps-1-example_cycle), ha='left', va='top', color='dimgrey', fontsize=plt.rcParams['xtick.labelsize'], \
                    transform = ax_A[2][0].transAxes)
    ax_A[1][0].set_title('asymptotic') # + r'$\mu_\mathrm{min}^0 = \mu_\mathrm{min}(I_\mathrm{E})$')#('upstroke\n' + r'$\mu_\mathrm{min}^0 > \mu_\mathrm{min}(I_\mathrm{E})$'+'\n(cycle {})'.format(example_cycle+1))
    
    # --- legends
    handles = ax_A[-1,-1].get_legend_handles_labels()[0]
    ax_A[-1,-1].legend(handles=handles, handlelength=1, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    fig.savefig('test.pdf', bbox_inches='tight')
    fig.savefig(path_to_figures+'Fig5.pdf', bbox_inches='tight')
    fig.savefig(path_to_figures+'Fig5.tif', bbox_inches='tight')
    fig.savefig(path_to_figures+'Fig5.svg', bbox_inches='tight')
  return 

def plot_figure_5_abc(traj_hash_stat, example_cycle=1, nsteps=4, \
                                reset=True, fmax=200, fmin=-200, \
                                path_to_simulations = './simulations/', path_to_figures = './figures/'):
  '''
  Plot Fig 5.

  Parameters
  ----------
  traj_hash_stat : Hash to simulation data the parameters of which should be used (D, Delta, K...)
  example_cycle : int optional. Which cycle to show in A. The default is 1.
  nsteps : int, optional. Number of steps in pw constant drive (up and down resp.). The default is 4.
  reset : bool, optional. Reset. The default is True.
  fmax, fmin : float, optional. [Hz] Limit for freq colorbar. The default is +/-200.
  path_to_simulations : str, optional. The default is './simulations/'.
  path_to_figures : str, optional. The default is './figures/'.

  Returns
  -------
  None (figure is stored in path_to_figures)

  '''
 
  print('\n\nPlotting Fig 5...')

  # load parameters from the provided simulationn
  traj_path_stat = pypet_get_trajectoryPath(traj_hash = traj_hash_stat, path_to_simulations = path_to_simulations)
  path_stat_info = traj_path_stat + 'info.csv'
  if not os.path.exists(path_stat_info):
    store_info_cyclostat_lif_simulation(traj_hash=traj_hash_stat, path_to_simulations=path_to_simulations)
  info = pd.read_csv(path_stat_info, index_col=0, squeeze=True, header=None)
  D, Delta, K, tm, Vr, I_hopf, I_full \
  = info['D'], info['Delta'], info['K'], info['tm'], info['Vr'], info['I_hopf'], info['I_full']
  print('D: {}, Delta: {}, K: {}, tm: {}, Vr: {}, I_hopf: {}, I_full: {}'.format(D, Delta, K, tm, Vr, I_hopf, I_full))
  
  # --- CONSTANT DRIVE 
  Iext_min, mu_min_max = get_Iext_lower_bound(D, Delta, K, tm, Vr, reset=reset, Iext_min=0, Iext_max=3, dI=.1)
  
  # summary IFA figure -- constant drive
  # --- construct figure
  with plt.rc_context({"axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 8,"font.size": 8,\
                       "xtick.labelsize": 6, "ytick.labelsize": 6,\
                        "xtick.direction": "out", "ytick.direction": "out", 'xtick.major.pad': 1, 'ytick.major.pad': 1}):
    fig_width= plos_width_fullpage # width_a4_wmargin # 21*cm
    fig_height = fig_width*.43 #9*cm #10.5*cm
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios = [1, 1], hspace=.3, wspace=.5)#, width_ratios=[5,2], height_ratios=[2,3])
  
    gs_A_sup = gs[0,0].subgridspec(1,2, width_ratios=[30,1])
    gs_A = gs_A_sup[0,0].subgridspec(2, 3, height_ratios = [.5,2], wspace=.5, hspace=.2)
    gs_B = gs[1,0].subgridspec(1, 3, width_ratios = [20, .5, 2], wspace=.1)
    gs_C = gs[:,1].subgridspec(3, 1, height_ratios=[1,.5,2], hspace=.1)
    
    ax_A = gs_A.subplots(sharex=True).T
    ax_B = gs_B.subplots() #[fig.add_subplot(gs_B[0,0]), fig.add_subplot(gs_B[0,1])]
    ax_B[-1].remove()
    ax_C = gs_C.subplots(sharex=True)
    despine(ax_C)
    despine(ax_A, which=['top', 'right', 'bottom'])
    despine(ax_A[1:,:], which=['left'])
    
    ax_A[0][0].text(-1.2, 1.5, string.ascii_uppercase[0], transform=ax_A[0][0].transAxes, size=panel_labelsize, weight='bold')
    ax_B[0].text(-1.2, -.65, string.ascii_uppercase[1], transform=ax_A[0][1].transAxes, size=panel_labelsize, weight='bold')
    ax_C[0].text(2.38, 1.5, string.ascii_uppercase[2], transform=ax_A[-1][0].transAxes, size=panel_labelsize, weight='bold')
    
    # --- fill figure
    # --- B -------------------------------------
    fig, ax_C, ax_B, df, traces_analytical \
    = analysis_IFA_constant_drive_analytical(Iext_min+.6, I_full, nsteps, mu_min_max, D, Delta, K, tm, Vr, reset=True, \
                                             fmax=fmax, fmin=fmin, fig=fig, ax_time = ax_C, ax_phase=ax_B)[1:]
    ax_B[0].set_title('')
    ax_B[0].set_ylim(top=mu_min_max)
    
    # --- A ------------------------------------
    # show traces for constant Iext which was reached in the example_cycle (up vs downstroke)
    Iext_toff = df.loc[example_cycle, 'Iext_toff']
    mu_min_0_stat = gaussian_drift_approx_stat(Iext_toff, D, Delta, K, tm, Vr=Vr, reset=reset)[0]
    fig, ax_A = visualize_traces_wlineardrive(D, Delta, K, tm, Vr, m = 0, \
                                              mu_min_0=[df.loc[example_cycle, 'mu_min_start'], mu_min_0_stat, df.iloc[-1-example_cycle, :].mu_min_start], \
                                              Iext_toff = Iext_toff, \
                                              fig=fig, ax=ax_A, reset=reset, i_ref=1, fmin=fmin, fmax=fmax)
    # add cycle numbers for reference
    ax_A[0][0].set_title('up')# + r'$\mu_\mathrm{min}^0 > \mu_\mathrm{min}(I_\mathrm{E})$')#('upstroke\n' + r'$\mu_\mathrm{min}^0 > \mu_\mathrm{min}(I_\mathrm{E})$'+'\n(cycle {})'.format(example_cycle+1))
    ax_A[0][0].text(0.05,1,'c{}'.format(example_cycle+1), ha='left', va='top', color='dimgrey', fontsize=plt.rcParams['xtick.labelsize'], \
                    transform = ax_A[0][0].transAxes)
    ax_A[2][0].set_title('down')# + r'$\mu_\mathrm{min}^0 < \mu_\mathrm{min}(I_\mathrm{E})$')#('upstroke\n' + r'$\mu_\mathrm{min}^0 > \mu_\mathrm{min}(I_\mathrm{E})$'+'\n(cycle {})'.format(example_cycle+1))
    ax_A[2][0].text(0.05,1,'c{}'.format(2*nsteps-1-example_cycle), ha='left', va='top', color='dimgrey', fontsize=plt.rcParams['xtick.labelsize'], \
                    transform = ax_A[2][0].transAxes)
    ax_A[1][0].set_title('asymptotic') # + r'$\mu_\mathrm{min}^0 = \mu_\mathrm{min}(I_\mathrm{E})$')#('upstroke\n' + r'$\mu_\mathrm{min}^0 > \mu_\mathrm{min}(I_\mathrm{E})$'+'\n(cycle {})'.format(example_cycle+1))
    
    # --- legends
    handles = ax_A[-1,-1].get_legend_handles_labels()[0]
    ax_A[-1,-1].legend(handles=handles, handlelength=1, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    
    fig.savefig(path_to_figures+'Fig5.pdf', bbox_inches='tight')
    fig.savefig(path_to_figures+'Fig5.tif', bbox_inches='tight')
    fig.savefig(path_to_figures+'Fig5.svg', bbox_inches='tight')
  return 

#%% Fig 6, 7
def plot_figures_6_7(traj_stat_hash=0, traj_trans_hash=2,  m_val = np.r_[.4, .2, .1], \
                              reset=True, fmax=200, fmin=-200, dIext_resolution=.05, n_mu=500, dt_num = .001, \
                              plateau_time_fig6 = 6, ms= 4, mu_min_start_first_cycle= 0.5, show_simulations = True, plot_controls_fig7=False, \
                              N_sim=10000, path_to_figures = './figures/', path_to_simulations='./simulations/' ):
  '''
  Plot Figs 6 and 7. Generate a large look-up table of Gaussian-drift SINGLE-cycle results for linear drive.

  Parameters
  ----------
  traj_stat_hash : int, optional. Hash to simulation data for constant drive. The default is 0.
  traj_trans_hash : int, optional. Hash to simulation data for double-ramp drive. The default is 2.
  m_val : list, optional. Slopes of double-ramp input explored in traj_trans_hash. The default is np.r_[.4, .2, .1].
  reset : bool, optional. Reset. The default is True.
  fmax, fmin : float, optional. [Hz] Limit for freq colorbar. The default is +/-200.
  dIext_resolution : float, optional. Resolution of look-up table in drive-space. The default is .05.
  n_mu : int, optional. Resolution of look-up table in mu_min_0-space (number of data points to compute). The default is 500.
  dt_num : float, optional. Time step for traces. The default is .001.
  plateau_time_fig6 : float, optional. Plateau time to plot in Fig 6 (ms). The default is 6.
  ms : float, optional. Markersize. The default is 4.
  mu_min_start_first_cycle : float. optional. Initial membrane potential in first cycle of each trajectory. The default is 0.5.
  show_simulations : bool, optional. Add simulation data to Fig 7. The default is True.
  plot_controls_fig7 : bool, optional. Store a copy of Fig 7 with sanity checks included. The default is False.
  N_sim : float, optional. Which simulated network size to show for comparison in Fig 7. The default is 10000.
  path_to_figures : str, optional. The default is './figures/'.
  path_to_simulations : str, optional. The default is './simulations/'.

  Returns
  -------
  None. (figure is stored in path_to_figures)

  '''

  print('\n\nPreparing Figs 6,7: Parameter-sweep of Gaussian-drift approximation for linear drive...')
  
  #### load and use parameters from the stationary reference  simulation indicated by traj_hash_stat:
  traj_path_stat = pypet_get_trajectoryPath(traj_hash = traj_stat_hash,  path_to_simulations = path_to_simulations)
  path_stat_info = traj_path_stat + 'info.csv'
  if not os.path.exists(path_stat_info):
    store_info_cyclostat_lif_simulation(traj_hash=traj_stat_hash, path_to_simulations=path_to_simulations)
  info = pd.read_csv(path_stat_info, index_col=0, squeeze=True, header=None)
  D, Delta, K, tm, Vr, I_hopf, I_full \
  = info['D'], info['Delta'], info['K'], info['tm'], info['Vr'], info['I_hopf'], info['I_full']
  # D, Delta, K, tm, Vr, I_hopf, I_full = 0.04, 1.2, 5, 10, 0, 1.5, 8.9
  print('D: {}, Delta: {}, K: {}, tm: {}, Vr: {}, I_hopf: {}, I_full: {}'.format(D, Delta, K, tm, Vr, I_hopf, I_full))
  
  ### set up the range of drives to be explored: from the theoretical lower bound of Iext to the same peak level that was used in the transient simulations (should be the pont of full synch of the stationary simulation)
  # lower bound Iext_min:
  Iext_min, mu_min_max = get_Iext_lower_bound(D, Delta, K, tm, Vr, reset=reset)
  if show_simulations:
    # load transient simulation and take the same peak level as the upper bound for the explored drives (simulated pt of full synchrony):
    traj_trans = pypet_load_trajectory(traj_hash=traj_trans_hash,  path_to_simulations = path_to_simulations) 
    Iext_plateau = convert_input(traj_trans.input.peak, traj_trans.tm, traj_trans.C, traj_trans.Vthr, traj_trans.E_rest, unit=False) # convert from nA to dimensionless voltage 
    t0_ramp = traj_trans.plateau_time / 2
  else:
    traj_trans = None
    t0_ramp = 10
    Iext_plateau = I_full # use pt of full synch as plateau level, as estimated in constant-drive simulation
    
  Iext_toff_val = np.arange(1, Iext_plateau + dIext_resolution, dIext_resolution) # range of drives to be explored 
  # start exploring drives from below Iext_min already (here from Vt on). No asymp. solu for those, but transient in case of decreasing drive are possible.

  # gaussian-drift approx for linear drive (analytical): parameter exploration for slope, initial mu, reference drive 
  try: # reload pre-calculated exploration of gauss-drift-approx for linear drive
    gauss_approx_exploration  = pd.read_csv('results/gaussian_drift_approx_linear_drive_exploration_fig6-7.csv', index_col=0, squeeze=True)
  except:
    print('Performing parameter exploration for gaussian-drift exploration under linear drive...')
    gauss_approx_exploration = gaussian_drift_approx_transient(m_val, Iext_toff_val, D, Delta, K, tm, Vr, n_mu=n_mu, reset=reset)
  
    
  # put together trajectories of consecutive cycles for a ramp up or down, analytically:
  trajectories_all, traces_all \
  = find_trajectories_linear_drive(gauss_approx_exploration, m_val, D, Delta, K, tm, Vr, Iext_min, Iext_plateau, Iext_start = [1,1.3,1.5],\
                                   mu_min_start_first_cycle = mu_min_start_first_cycle, tolerance_Iext= .1, reset=reset)
  trajectories_all.to_csv('results/gaussian_drift_approx_linear_drive_trajectories_fig6-7.csv') # store
  
  # plot example with strongest slope, same layout as for constant drive
  fig6 = plot_figure_6(gauss_approx_exploration, trajectories_all, traces_all, D, Delta, K, tm, Vr, Iext_min, Iext_plateau, m=.4, \
                       reset=reset, I_hopf=I_hopf, fmax=fmax, fmin=fmin, dt_num = dt_num, t0_ramp = plateau_time_fig6/2, ms= ms, \
                       path_to_figures = path_to_figures, path_to_simulations =path_to_simulations)
  
  # show results for all slopes in one plot
  fig7 = plot_figure_7(traj_trans, gauss_approx_exploration, trajectories_all, traces_all,  D, Delta, K, tm, Vr, reset=reset, \
                       traj_stat_hash = traj_stat_hash, plot_controls= False, \
                       fmax=fmax, fmin=fmin, dt_num = dt_num, t0_ramp = t0_ramp, ms= 3, N_sim=N_sim, show_simulations=show_simulations, path_to_figures = path_to_figures, path_to_simulations =path_to_simulations)
  if plot_controls_fig7:
    fig7_wcontrols = plot_figure_7(traj_trans, gauss_approx_exploration, trajectories_all, traces_all,  D, Delta, K, tm, Vr, reset=reset, \
                         traj_stat_hash = traj_stat_hash, plot_controls= True, \
                         fmax=fmax, fmin=fmin, dt_num = dt_num, t0_ramp = t0_ramp, ms= 3, N_sim=N_sim, show_simulations=show_simulations, path_to_figures = path_to_figures, path_to_simulations =path_to_simulations)
    
# traj_stat_name = traj_stat_name, 
  fig6.savefig(path_to_figures+'Fig6.pdf', bbox_inches='tight')
  fig6.savefig(path_to_figures+'Fig6.tif', bbox_inches='tight')
  fig6.savefig(path_to_figures+'Fig6.svg', bbox_inches='tight')
  fig7.savefig(path_to_figures+'Fig7.tif', bbox_inches='tight')
  fig7.savefig(path_to_figures+'Fig7.pdf', bbox_inches='tight')
  fig7.savefig(path_to_figures+'Fig7.svg', bbox_inches='tight')
  fig7_wcontrols.savefig(path_to_figures+'Fig7_wcontrols.tif', bbox_inches='tight')
  fig7_wcontrols.savefig(path_to_figures+'Fig7_wcontrols.pdf', bbox_inches='tight')
  return fig6, fig7
  
def plot_figure_6(df_all, trajectories_all, traces, D, Delta, K, tm, Vr, Iext_min, Iext_plateau, m=.4, reset=True, I_hopf=None, \
                                fmax=200, fmin=-200, dt_num = .001, \
                                t0_ramp = 3, plateau_time_for_linreg = 20, ms= 3, ms_freq=4, path_to_figures = './figures/', path_to_simulations='./simulations/'): #,rcParams=dict(plt.rcParamsDefault)):
  print('\n\nPlotting Fig 6...')
  linestyle_plateau = '--'
  # restrict info to slope m
  df = df_all[df_all.m.abs()==m].copy()
  trajectory  = trajectories_all[trajectories_all.m.abs()==m].copy()

  
  # construct a double-ramp stimulus from Iext_min up to Iext_plateau with slope +/- m:
  ramp_time = (Iext_plateau-Iext_min)/m
  t_Iext_stat = np.arange(0, 2*ramp_time+2*t0_ramp, dt_num) - t0_ramp - ramp_time
  Iext_stat = np.piecewise(t_Iext_stat, \
                            [t_Iext_stat<-t0_ramp, (-t0_ramp<=t_Iext_stat) & (t_Iext_stat<=t0_ramp), t_Iext_stat>t0_ramp], \
                            [lambda x: Iext_plateau + m*(x+t0_ramp), Iext_plateau, lambda x: Iext_plateau - m*(x-t0_ramp)])
  # infer the asymptotic reference dynamics:
  mu_min_stat, mu_max_stat, f_net_stat, f_unit_stat, _, _, _, t_off_stat, mu_reset_stat\
  = gaussian_drift_approx_stat(Iext_stat, D, Delta, K, tm, Vr=Vr, reset=reset)[:-2] 
  
  # the trajectory on the downstroke can end at a drive < Iext_min. Pad the stimulus accordingly:
  Iext_min_pad = trajectory.Iext_end.min() - .5
  ramp_time_pad = (Iext_plateau-Iext_min_pad)/m
  t_Iext_stat_pad = np.arange(0, 2*ramp_time_pad+2*t0_ramp, dt_num) - t0_ramp - ramp_time_pad
  Iext_stat_pad = np.piecewise(t_Iext_stat_pad, \
                            [t_Iext_stat_pad<-t0_ramp, (-t0_ramp<=t_Iext_stat_pad) & (t_Iext_stat_pad<=t0_ramp), t_Iext_stat_pad>t0_ramp], \
                            [lambda x: Iext_plateau + m*(x+t0_ramp), Iext_plateau, lambda x: Iext_plateau - m*(x-t0_ramp)])
    
  # for plotting: 
  # indices to target ramp up, down or plateau phase resp.:
  ix_up, ix_plateau, ix_down = t_Iext_stat<-t0_ramp, (t_Iext_stat>-t0_ramp)&(t_Iext_stat<t0_ramp), t_Iext_stat>t0_ramp
  upstroke_duration = trajectory[trajectory.m == m].time_end.max()
  # downstroke_duation = trajectory[trajectory.m == -m].time_end.max()
  offset = {m:- t0_ramp - upstroke_duration, -m:  t0_ramp } # beginning of upstroke and downstroke resp.
  norm = matplotlib.colors.TwoSlopeNorm(0, vmin=fmin, vmax=fmax) # color norm for freq differences     
  ymin = np.min(mu_min_stat)-1.2 # lower limit for yaxis of axC1
  ncycles_upstroke = len(trajectory[trajectory.m == m])
  
  # --- construct figure
  with plt.rc_context({"axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 8,"font.size": 8,\
                       "xtick.labelsize": 6, "ytick.labelsize": 6,\
                      "xtick.direction": "out", "ytick.direction": "out", 'xtick.major.pad': 1, 'ytick.major.pad': 1}):
    fig_width= plos_width_fullpage # width_a4_wmargin #21*cm
    fig_height = fig_width*0.4 # 9*cm # 10.5*cm
    fig = plt.figure(figsize=(fig_width, fig_height))#, constrained_layout=True)
    
    gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios = [1, 1], hspace=.4, wspace=.5)#, width_ratios=[5,2], height_ratios=[2,3])
  
    gs_A_sup = gs[0,0].subgridspec(1,2, width_ratios=[30,1])
    gs_A = gs_A_sup[0,0].subgridspec(2, 3, height_ratios = [.5,2], wspace=.5, hspace=.2)
    gs_B_sup = gs[1,0].subgridspec(1, 3, width_ratios = [30, .5, 1], wspace=.1)
    gs_B = gs_B_sup[0,0].subgridspec(1,2)
    ax_B_cb = fig.add_subplot(gs_B_sup[0,1])
    gs_C = gs[:,1].subgridspec(3, 1, height_ratios=[1,.5,2], hspace=.1)
    
    ax_A = gs_A.subplots(sharex='col').T
    ax_B = gs_B.subplots(sharey=True, sharex=True) #[fig.add_subplot(gs_B[0,0]), fig.add_subplot(gs_B[0,1])]
    ax_C = gs_C.subplots(sharex=True)
    
    despine(ax_A, which=['top', 'right', 'bottom'])
    despine(ax_A[1:,:], which=['left'])
    despine(ax_B)
    despine(ax_C, which=['top', 'right', 'bottom'])


    
    ax_A[0][0].text(-1.2, 1.55, string.ascii_uppercase[0], transform=ax_A[0][0].transAxes, size=panel_labelsize, weight='bold')
    ax_B[0].text(-1.2, -.65, string.ascii_uppercase[1], transform=ax_A[0][1].transAxes, size=panel_labelsize, weight='bold')
    ax_C[0].text(2.38, 1.55, string.ascii_uppercase[2], transform=ax_A[-1][0].transAxes, size=panel_labelsize, weight='bold')
    
    # --- fill figure
    # --- A ------------------------------------
    # show traces for constant Iext which was reached in the example_cycle (up vs downstroke)
    Iext_toff_A = 5 #df.loc[example_cycle, 'Iext_toff']
    m_A = [m, 0, -m]
    mu_min_0_stat = gaussian_drift_approx_stat(Iext_toff_A, D, Delta, K, tm, Vr=Vr, reset=reset)[0]
    mu_min_0_A = [0.35, mu_min_0_stat, -3.45]
    fig, ax_A = visualize_traces_wlineardrive(D, Delta, K, tm, Vr, \
                                              m = m_A, \
                                              mu_min_0 = mu_min_0_A, \
                                              Iext_toff = Iext_toff_A, \
                                              fig=fig, ax=ax_A, reset=reset, i_ref=1, fmin=fmin, fmax=fmax)
    ax_A[0][0].set_title('up\n \u2605 m=+{}'.format(m))# + r'$\mu_\mathrm{min}^0 > \mu_\mathrm{min}(I_\mathrm{E})$')#('upstroke\n' + r'$\mu_\mathrm{min}^0 > \mu_\mathrm{min}(I_\mathrm{E})$'+'\n(cycle {})'.format(example_cycle+1))
    ax_A[2][0].set_title('down\n \u2606 m={}'.format(-m))# + r'$\mu_\mathrm{min}^0 < \mu_\mathrm{min}(I_\mathrm{E})$')#('upstroke\n' + r'$\mu_\mathrm{min}^0 > \mu_\mathrm{min}(I_\mathrm{E})$'+'\n(cycle {})'.format(example_cycle+1))
    ax_A[1][0].set_title('asymptotic\n m=0') # + r'$\mu_\mathrm{min}^0 = \mu_\mathrm{min}(I_\mathrm{E})$')#('upstroke\n' + r'$\mu_\mathrm{min}^0 > \mu_\mathrm{min}(I_\mathrm{E})$'+'\n(cycle {})'.format(example_cycle+1))
      #  --- legends
    handles = ax_A[-1,-1].get_legend_handles_labels()[0]
    ax_A[-1,-1].legend(handles=handles, handlelength=1, bbox_to_anchor=(1.1, 1), loc='upper left', borderaxespad=0)
    
                                  
    # --- B ------------------------------------
    ax_B[0].text(1,1,'m=+{:.1f}/ms'.format(m), ha='right', va='bottom', transform= ax_B[0].transAxes)
    ax_B[1].text(1,1,'m=-{:.1f}/ms'.format(m), ha='right', va='bottom', transform= ax_B[1].transAxes)

    dmu = np.unique(np.diff(df.mu_min_start.unique()))[0] # resolution of exploration in mumin_start
    dI = np.unique(np.diff(df.Iext_toff.unique()))[0] # resolution of exploration in Iext_toff
    extent = (np.min(df.Iext_toff.unique())-dI/2, np.max(df.Iext_toff.unique())+dI/2, \
              np.min(df.mu_min_start.unique())-dmu/2, np.max(df.mu_min_start.unique())+dmu/2)
    for i, mm in enumerate([m, -m]):
      df_freq = (df[df.m==mm]).pivot(index='mu_min_start', columns='Iext_toff', values=['f_inst', 'f_stat']).astype(float)
      df_abs = np.abs(df_freq.f_inst-df_freq.f_stat)  
      im = ax_B[i].imshow(df_freq.f_inst.to_numpy()-df_freq.f_stat.to_numpy(), origin='lower', extent=extent,  \
                          aspect='auto', cmap=plt.cm.coolwarm, norm=norm, interpolation=None)
      # plot the approx 0-line of df_freq
      ax_B[i].plot(Iext_stat[ix_up], mu_min_stat[ix_up], color='k', zorder=3) #label=r'$\mu_\mathrm{min}^\mathrm{\infty}\left(\hat{I}_\mathrm{E}\right)$', 
      ax_B[i].plot(df_abs.idxmin().index.values, df_abs.idxmin().values, 'w', zorder=2)#, label=label_stat[3])
      ax_B[i].set_xlabel(r'drive $\hat{I}_\mathrm{E}$')
      ax_B[i].autoscale(False)
    ax_B[1].legend(loc='upper right', handlelength=1)
    ax_B[0].set_ylabel(r'initial $\mu_\mathrm{min}$')
    cb = plt.colorbar(im, cax=ax_B_cb, \
                      ticks=[fmin, int(fmin/2), 0, int(fmax/2), fmax])
    cb.set_label(label=r'$f_\mathrm{net}^\mathrm{inst}-f_\mathrm{net}^\mathrm{\infty}\left(\hat{I}_\mathrm{E}\right)$ [Hz]',labelpad=-.1)
    cb.ax.plot([-1e7, 1e7],[0]*2, 'w', lw=1) 
    plt.setp(ax_B[1].get_yticklabels(), visible=False)
    
    # insert example trajectories in B:
    for mm in [m, -m]:
      trajectory_m = trajectory[trajectory.m == mm] # select up or downstroke   
      ix_B = 0 if mm>0 else 1
      ax_B[ix_B].plot(trajectory_m.Iext_toff, trajectory_m.mu_min_start, \
                                    linestyle='-', color='k', zorder=2, lw=.5) # marker='o', markersize=ms, markerfacecolor='None', \
                                    # markeredgecolor='k')
      colors = plt.cm.coolwarm(norm(((trajectory_m.f_inst-trajectory_m.f_stat)).values))     
      ax_B[ix_B].scatter(trajectory_m.Iext_toff, trajectory_m.mu_min_start, \
                  c = colors, edgecolors='k', s=ms**2, zorder=5, lw=.5) 
      ax_B[ix_B] = custom_arrow(ax_B[ix_B], trajectory_m.Iext_toff.values, trajectory_m.mu_min_start.values, head_width=.3, point_to=.6, alpha=20, lw=.5)
      ax_B[ix_B].set_ylim([extent[2], trajectory.mu_min_start.max()+.2])

      # annotate cycle numbers
      for c in range(len(trajectory_m)):
        if mm>0:
          ax_B[ix_B].text(trajectory_m.loc[trajectory_m.cycle==c, 'Iext_toff'], trajectory_m.loc[trajectory_m.cycle==c, 'mu_min_start']-.3, str(c+1), \
                        ha='center', va='top', color='dimgrey', fontsize=plt.rcParams['xtick.labelsize'])
        else:
          ax_B[ix_B].text(trajectory_m.loc[trajectory_m.cycle==c, 'Iext_toff'], trajectory_m.loc[trajectory_m.cycle==c, 'mu_min_start']+.25, \
                          str(c+1+ncycles_upstroke), ha='center', va='bottom', color='dimgrey', fontsize=plt.rcParams['xtick.labelsize'])
    # mark the A examples
    ax_B[0].plot(Iext_toff_A, mu_min_0_A[0], marker="$\u2605$", markersize=4, markeredgecolor='k', markeredgewidth=.2, markerfacecolor='k')
    ax_B[1].plot(Iext_toff_A, mu_min_0_A[-1], marker="$\u2605$", markersize=4, markeredgecolor='k', markeredgewidth=.2, markerfacecolor='w')
        
    # --- C -----------------------------------    
    # mark reset and threshold
    gridline([Vr, 1], ax_C[-1])
    # plot asymptotic reference dynamics in time  
    # plateau phase:
    ax_C[0].plot(t_Iext_stat[ix_plateau], f_net_stat[ix_plateau], color='k', linestyle=linestyle_plateau, lw=0.5) # if mm>0 else '')
    ax_C[-1].plot(t_Iext_stat[ix_plateau], Iext_stat[ix_plateau], color=my_colors['iext'], linestyle=linestyle_plateau, lw=0.5) # if mm>0 else '') 
    ax_C[-1].plot(t_Iext_stat[ix_plateau], mu_max_stat[ix_plateau], color=my_colors['max'], linestyle=linestyle_plateau, lw=0.5) # if mm>0 else '')
    ax_C[-1].plot(t_Iext_stat[ix_plateau], mu_min_stat[ix_plateau], color=my_colors['min'], linestyle=linestyle_plateau, lw=0.5) # if mm>0 else '')  
    # ramp phase, drive padded: 
    ax_C[-1].plot(t_Iext_stat_pad[t_Iext_stat_pad<-t0_ramp], Iext_stat_pad[t_Iext_stat_pad<-t0_ramp], color=my_colors['iext'])  
    ax_C[-1].plot(t_Iext_stat_pad[t_Iext_stat_pad>t0_ramp], Iext_stat_pad[t_Iext_stat_pad>t0_ramp], color=my_colors['iext'])
    # ramp phase, asymptotic dynamics:
    for ix_ramp in [ix_up, ix_down]:      
      ax_C[0].plot(t_Iext_stat[ix_ramp], f_net_stat[ix_ramp], color='k', lw=0.5, label=r'$f_\mathrm{net}^\mathrm{\infty}$' if ix_ramp[0] else '')
      ax_C[-1].plot(t_Iext_stat[ix_ramp], mu_max_stat[ix_ramp], color=my_colors['max'], lw=0.5, label=r'$\mu_\mathrm{max}^\mathrm{\infty}$' if ix_ramp[0] else '')
      ax_C[-1].plot(t_Iext_stat[ix_ramp], mu_min_stat[ix_ramp], color=my_colors['min'], lw=0.5, label=r'$\mu_\mathrm{min}^\mathrm{\infty}$' if ix_ramp[0] else '')
    
    # add transient dynamics, keypoints:
    for mm in [m, -m]:
      trajectory_m = trajectory[trajectory.m == mm] # select up or downstroke 
      print('m= ', mm, '. Iext at time toff for each cycle: ', trajectory_m.Iext_toff) # to cite in main text
      ax_C[0].plot(trajectory_m.time_toff + offset[mm], trajectory_m.f_stat, 'k.')
      ax_C[0].plot(trajectory_m.time_toff + offset[mm], trajectory_m.f_inst, linestyle=':', color='dimgrey', zorder=2, lw=.5)
      ax_C[0].scatter(trajectory_m.time_toff + offset[mm], trajectory_m.f_inst, \
                  c = plt.cm.coolwarm(norm(((trajectory_m.f_inst-trajectory_m.f_stat)).values)), \
                  edgecolors='k', s=ms_freq**2, zorder=3, lw=.5) 
        
      ax_C[-1].plot(trajectory_m.time_toff + offset[mm], trajectory_m.Iext_toff , '.', markersize=ms, color=my_colors['iext'])
      ax_C[-1].plot(trajectory_m.time_toff + offset[mm], trajectory_m.mu_max, '.', markersize=ms, color=my_colors['max'], label=r'$\mu_\mathrm{max}^\mathrm{inst}$' if mm>0 else '')
      ax_C[-1].plot(trajectory_m.time_start + offset[mm], trajectory_m.mu_min_start, '.', markersize=ms, color=my_colors['min'], label=r'$\mu_\mathrm{min}^\mathrm{inst}$' if mm>0 else '')
      ax_C[-1].plot(trajectory_m.time_end + offset[mm], trajectory_m.mu_min_end, '.', markersize=ms, color=my_colors['min'])
      
      # mark every other cycle in grey 
      for i in range(len(ax_C)):
        for c in range(0, len(trajectory_m), 2):
          # gridline([df_num.loc[(df_num.m==mm) & (df_num.cycle==c), 'time_start'], df_num.loc[(df_num.m==mm) & (df_num.cycle==c), 'time_end']], ax_C[1], 'x')
          ax_C[i].axvspan(float(trajectory_m.loc[trajectory_m.cycle==c, 'time_start'])+offset[mm], \
                          float(trajectory_m.loc[trajectory_m.cycle==c, 'time_end'])+offset[mm], \
                          alpha=0.15, color=my_colors['cyc0'], zorder=-20, ec=None)
          if c+1 < len(trajectory_m):
            ax_C[i].axvspan(float(trajectory_m.loc[trajectory_m.cycle==c+1, 'time_start'])+offset[mm], \
                            float(trajectory_m.loc[trajectory_m.cycle==c+1, 'time_end'])+offset[mm], \
                            alpha=0.35, color=my_colors['cyc1'], zorder=-20, ec=None)
      # insert cycle numbers
      cycle_center = trajectory_m.time_start + (trajectory_m.time_end - trajectory_m.time_start) / 2
      for c, cyc in enumerate(cycle_center):
        if mm>0:
          ax_C[-1].text(cyc+ offset[mm], ymin+0.1, str(c+1), ha='center', va='bottom', color='dimgrey', fontsize=plt.rcParams['xtick.labelsize'])
        else:
          ax_C[-1].text(cyc+ offset[mm], ymin+0.1, str(c+1+ncycles_upstroke), ha='center', va='bottom', color='dimgrey', fontsize=plt.rcParams['xtick.labelsize'])
      
      # add traces for rate and mu
      ax_C[1].plot(traces[mm]['t']+ offset[mm], traces[mm]['rate'], color= 'k')
      ax_C[-1].plot(traces[mm]['t']+ offset[mm], traces[mm]['mu'], color= my_colors['mu'])
    
    # linear regression to quantify IFA
    # pool together time and frequency values from up- and down-stroke, assuming again a fixed plateau_time as in Fig 2 (20 ms)
    offset_plateau = (plateau_time_for_linreg-2*t0_ramp)/2
    time = list(trajectory[trajectory.m == m].time_toff + offset[m] - offset_plateau) + list(trajectory[trajectory.m == -m].time_toff + offset[-m] + offset_plateau) 
    frequencies = list(trajectory[trajectory.m == m].f_inst) + list(trajectory[trajectory.m == -m].f_inst) 
    ifa_slope, ifa_intercept = do_linear_regression(time, frequencies)
    ax_C[0].text(.99,.97, r'$\chi_\mathrm{IFA}=$'+'{:.1f} Hz/ms'.format(ifa_slope), transform=ax_C[0].transAxes, va='top', ha='right', \
        fontsize=plt.rcParams['legend.fontsize'])  
    
    # scale bar:
    scalebar = AnchoredSizeBar(ax_C[-1].transData, 5, '5 ms', 'upper left', frameon=False, borderpad=.7 ) 
    ax_C[-1].add_artist(scalebar)
    
    # labels & ticks 
    ax_C[0].set_ylim([np.min([100, trajectory.f_inst.min()-20]), 400]) # trajectory.f_inst.max()+20])
    ax_C[0].set_yticks([100,200,300, 400])
    ax_C[0].set_yticklabels(['100','','','400'])
    ax_C[-1].set_xlim([t_Iext_stat_pad[0], t_Iext_stat_pad[-1]])
    ax_C[-1].set_ylim(bottom=ymin)
       
    ax_C[0].set_ylabel('freq [Hz]')
    ax_C[1].set_ylabel('rate\n[spk/s]')
    ax_C[-1].set_ylabel('voltage, drive')
          
    # legend
    handles, labels = ax_C[0].get_legend_handles_labels()
    extra_handle = Line2D([0], [0], marker='o', color='dimgrey', lw=.5, linestyle=':', markersize=ms_freq, markerfacecolor='w',\
                          markeredgewidth=.5, markeredgecolor='k', label='$f_\mathrm{net}^\mathrm{inst}$')
    handles.append(extra_handle)
    ax_C[0].legend(handles=handles, handlelength=1, labelspacing=0.1, bbox_to_anchor=(1., 1), loc='lower right', borderaxespad=.1, ncol=2)
    ax_C[-1].legend(bbox_to_anchor=(0,0), loc='upper left', columnspacing=.8, borderaxespad=.1, handlelength=1, ncol=4)
  
  fig.savefig(path_to_figures+'Fig6.pdf', bbox_inches='tight')
  return fig


def plot_figure_7(traj_trans, df_all, trajectories_all, traces, D, Delta, K, tm, Vr, reset=True,  \
                  traj_stat_hash=np.nan, traj_stat_name=None, plot_controls = False, \
                  fmax=200, fmin=-200, dt_num = .001,  ylim_panels_ii = [100, 430], \
                  t0_ramp = 3, ms= 5, N_sim=10000, show_simulations=True, path_to_figures='./figures/', path_to_simulations='./simulations/'):
  '''
  t0_ramp: time [ms] away from 0 where ramp should begin (artificial plateau time will be 2*t0_ramp)
  '''   
  print('\n\nPlotting Fig 7...')
  # extract
  m_val = df_all.m.abs().unique() # esplored slopes
  Iext_min, Iext_max = np.min(df_all.Iext_toff.unique()), np.max(df_all.Iext_toff.unique())
  Iext_min_pad = trajectories_all.Iext_end.min()-.3 # lowest drive at the end of a transient cycle (under linear drive, this drive can be lower than theoretical lower bound Iext_min)
  dmu = np.unique(np.diff(df_all.mu_min_start.unique()))[0] # resolution in mumin0
  dI = np.unique(np.diff(df_all.Iext_toff.unique()))[0] # resolution in Iext(toff)
  extent = (np.min(df_all.Iext_toff.unique())-dI/2, np.max(df_all.Iext_toff.unique())+dI/2, \
            np.min(df_all.mu_min_start.unique())-dmu/2, np.max(df_all.mu_min_start.unique())+dmu/2)
  
  # --- infer wspace for top plots
  ramp_time = (Iext_max-Iext_min_pad)/m_val
  width_ratios = list(ramp_time + t0_ramp)
  width_ratios_top_middle = width_ratios[0]/np.array(width_ratios)
  width_ratios_top_sides = (1-width_ratios_top_middle)/2
  width_ratios_top = list(np.array([width_ratios_top_middle, width_ratios_top_sides, width_ratios_top_sides]).T)

  norm=matplotlib.colors.TwoSlopeNorm(0, vmin=fmin, vmax=fmax)

  # prepare table 2 with summary statistics:
  table1 = pd.DataFrame(columns=['m', 'ifa_sim', 'ifa_theory', 'error'])
    
  with plt.rc_context({"axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 8, "font.size": 8, \
                       "xtick.labelsize": 6, "ytick.labelsize": 6}):
    # --- construct figure ---------------------------------------------
    fig_width= plos_width_fullpage #width_a4_wmargin # 21*cm
    fig_height = fig_width*.6  # 8*cm
    fig = plt.figure(figsize=(fig_width, fig_height))#, constrained_layout=True)
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios = width_ratios, wspace=.2)#, width_ratios=[5,2], height_ratios=[2,3])
    
    gs_sub, gs_top, gs_bottom = [None]*len(m_val), [None]*len(m_val), [None]*len(m_val)
    ax_top, ax_bottom = [[None]*2]*len(m_val), [[None]*3]*len(m_val)
      
    for i in range(len(m_val)):
      gs_sub[i] = gs[0,i].subgridspec(2,1, height_ratios=[.7,1], hspace=.4)
      gs_top[i] = gs_sub[i][0].subgridspec(2, 3, width_ratios = width_ratios_top[i], hspace=.4, wspace=0) # subgridspec(1,2, width_ratios=[1,1], wspace=wspace[i])
      gs_bottom[i] = gs_sub[i][1].subgridspec(3,1, height_ratios=[1,.5,1], hspace=.2)
      ax_top_aux = fig.add_subplot(gs_top[i][0,0]) #fig.add_subplot(gs_top[i][0,1])
      # ax_top[i] = [ax_top_aux, fig.add_subplot(gs_top[i][1,1], sharex=ax_top_aux, sharey=ax_top_aux)] 
      ax_top[i] = [ax_top_aux, fig.add_subplot(gs_top[i][1,0], sharex=ax_top_aux, sharey=ax_top_aux)] 
      ax_bottom[i] = gs_bottom[i].subplots(sharex=True)
      despine(ax_top[i])
      
      
      ax_top[i][0].text(1,1,'m=+{}'.format(m_val[i]), ha='right', va='bottom', transform = ax_top[i][0].transAxes )
      ax_top[i][1].text(1,1,'m=-{}'.format(m_val[i]), ha='right', va='bottom', transform = ax_top[i][1].transAxes )
      
      # ax_top[i][0].set_title('m=+{}'.format(m_val[i]), pad=1.1)
      # ax_top[i][1].set_title('m=-{}'.format(m_val[i]), pad=0.01)#, pad=titlepad)
      ax_bottom[i][0].set_title('$m=\pm${}'.format(m_val[i]), pad=-.3)
      
      despine(ax_bottom[i], which=['bottom', 'top', 'right'])
  
    # colorbar axis 
    ax_cbar = ax_top[-2][-1].inset_axes([1.1, 0, .05, 1], transform=ax_top[-2][-1].transAxes)

    xx, yy = [0.05, .26, .51], [.9, .5]
    for i in range(3):
      ax_top[i][0].text(xx[i], yy[0], string.ascii_uppercase[i]+string.ascii_lowercase[8], transform=fig.transFigure, size=panel_labelsize, weight='bold')
      ax_top[i][0].text(xx[i], yy[1], string.ascii_uppercase[i]+string.ascii_lowercase[8]*2, transform=fig.transFigure, size=panel_labelsize, weight='bold')

    # --- fill figure
    # t = {} # time stamps
    offset = {}   
    for i, m in enumerate(m_val): # loop over slopes (columns A,B,C)
      # --- theory: bottom panels ------------------------------------------------------------------------------------------------------
      # reconstruct double ramp stimulus
      ramp_time_pad = (Iext_max-Iext_min_pad)/m
      t_Iext_stat_pad = np.arange(0, 2*ramp_time_pad+2*t0_ramp + dt_num, dt_num) - t0_ramp - ramp_time_pad
      Iext_stat_pad = np.piecewise(t_Iext_stat_pad, \
                               [t_Iext_stat_pad<-t0_ramp, (-t0_ramp<=t_Iext_stat_pad) & (t_Iext_stat_pad<=t0_ramp), t_Iext_stat_pad>t0_ramp], \
                               [lambda x: Iext_max + m*(x+t0_ramp), Iext_max, lambda x: Iext_max - m*(x-t0_ramp)])
      # stationary reference
      ramp_time = (Iext_max-Iext_min)/m
      t_Iext_stat = np.arange(0, 2*ramp_time+2*t0_ramp + dt_num, dt_num) - t0_ramp - ramp_time
      Iext_stat = np.piecewise(t_Iext_stat, \
                               [t_Iext_stat<-t0_ramp, (-t0_ramp<=t_Iext_stat) & (t_Iext_stat<=t0_ramp), t_Iext_stat>t0_ramp], \
                               [lambda x: Iext_max + m*(x+t0_ramp), Iext_max, lambda x: Iext_max - m*(x-t0_ramp)])
      mu_min_stat, mu_max_stat, f_net_stat, f_unit_stat, _, _, _, t_off_stat, mu_reset_stat\
      = gaussian_drift_approx_stat(Iext_stat, D, Delta, K, tm, Vr=Vr, reset=reset, check_mumin=True)[:-2] # gaussian drift approx under constant drive for cont range of reference drives
      
      # plot stationary reference values during plateau phase:
      ix_plateau = (-t0_ramp<=t_Iext_stat) & (t_Iext_stat<=t0_ramp)
      ax_bottom[i][0].plot(t_Iext_stat[ix_plateau], f_net_stat[ix_plateau], color='k', linestyle=':', lw=.5) # if mm>0 else '')
      ax_bottom[i][-1].plot(t_Iext_stat[ix_plateau], Iext_stat[ix_plateau], color=my_colors['iext'], linestyle=':', lw=.5) # if mm>0 else '') 
      ax_bottom[i][-1].plot(t_Iext_stat[ix_plateau], mu_max_stat[ix_plateau], color=my_colors['max'], linestyle=':', lw=.5) # if mm>0 else '')
      ax_bottom[i][-1].plot(t_Iext_stat[ix_plateau], mu_min_stat[ix_plateau], color=my_colors['min'], linestyle=':', lw=.5) # if mm>0 else '')  
      # plot ramping stimulus:
      ax_bottom[i][-1].plot(t_Iext_stat_pad[t_Iext_stat_pad<-t0_ramp], Iext_stat_pad[t_Iext_stat_pad<-t0_ramp], color=my_colors['iext'], lw=.5, label=str_Iext)  
      ax_bottom[i][-1].plot(t_Iext_stat_pad[t_Iext_stat_pad>t0_ramp], Iext_stat_pad[t_Iext_stat_pad>t0_ramp], color=my_colors['iext'], lw=.5)
      # plot stationary reference values during ramp phase:
      for ix_ramp in [(-t0_ramp>t_Iext_stat), (t_Iext_stat>t0_ramp)]:
        ax_bottom[i][0].plot(t_Iext_stat[ix_ramp], f_net_stat[ix_ramp], color='k', lw=.5, label=r'$f_\mathrm{net}^\mathrm{\infty}$ (theory)' if ix_ramp[0] else '')
        ax_bottom[i][-1].plot(t_Iext_stat[ix_ramp], mu_max_stat[ix_ramp], color=my_colors['max'], lw=.5, label=r'$\mu_\mathrm{max}^\mathrm{\infty}$' if ix_ramp[0] else '')
        ax_bottom[i][-1].plot(t_Iext_stat[ix_ramp], mu_min_stat[ix_ramp], color=my_colors['min'], lw=.5, label=r'$\mu_\mathrm{min}^\mathrm{\infty}$' if ix_ramp[0] else '')
      
      # transient dynamics
      # ------- theory ------------------------------------------------------------      
      # use offsets for plotting up- and downstroke to mimick an itermediate plateau time:
      T_up = traces[m]['t'][-1] # duration of upstroke;
      offset[m] = - t0_ramp - T_up
      offset[-m] = t0_ramp 
      
      for mm in [m, -m]: # first up, then downstroke
        # select analytically inferred trajectory with slope mm:
        traj_mm = trajectories_all[trajectories_all.m==mm]
        # --- top panels ----------------------------------------------------------------------------------------------------
        # parameter exploration for single cycles under linear drive:
        ix_top = 0 if mm>0 else 1
        df_freq = (df_all[df_all.m==mm]).pivot(index='mu_min_start', columns='Iext_toff', values=['f_inst', 'f_stat']).astype(float)
        df_abs = np.abs(df_freq.f_inst-df_freq.f_stat)  
        im = ax_top[i][ix_top].imshow(df_freq.f_inst.to_numpy()-df_freq.f_stat.to_numpy(), origin='lower', \
                   extent=extent, aspect='auto', cmap=plt.cm.coolwarm, norm=norm, interpolation=None)
        # plot the approx 0-line of df_freq
        ax_top[i][ix_top].plot(Iext_stat[ix_ramp], mu_min_stat[ix_ramp], color='k', label=r'$\mu_\mathrm{min}^\mathrm{\infty}$', zorder=3)
        ax_top[i][ix_top].plot(df_abs.idxmin().index.values, df_abs.idxmin().values, 'w', lw=1, zorder=2)#, label=label_stat[3])
        ax_top[i][ix_top].autoscale(False)
        # axis limits
        ax_top[i][ix_top].set_xlim([trajectories_all.Iext_toff.min()-.3, extent[1]])
        ax_top[i][ix_top].set_ylim([extent[2], trajectories_all.mu_min_start.max()+.2])
        
        # transient trajectories (analytically inferred) in bottom panel:
        ax_bottom[i][-1].plot(traj_mm.time_toff + offset[mm], traj_mm.Iext_toff, '.', markersize=ms, color=my_colors['iext'], label=r'$\hat{I}_\mathrm{E}$' if mm>0 else '')
        ax_bottom[i][-1].plot(traj_mm.time_toff + offset[mm], traj_mm.mu_max, '.', markersize=ms, color=my_colors['max'], label= r'$\mu_\mathrm{max}^\mathrm{inst}$' if mm>0 else '')  
        ax_bottom[i][-1].plot(traj_mm.time_start + offset[mm], traj_mm.mu_min_start, '.', markersize=ms, color=my_colors['min'], label= r'$\mu_\mathrm{min}^\mathrm{inst}$' if mm>0 else '')
        ax_bottom[i][-1].plot(traj_mm.time_end + offset[mm], traj_mm.mu_min_end, '.', markersize=ms, color=my_colors['min'])
        
        # add traces
        ax_bottom[i][1].plot(traces[mm]['t'] + offset[mm], traces[mm]['rate'], color='k')
        ax_bottom[i][-1].plot(traces[mm]['t'] + offset[mm], traces[mm]['mu'], color=my_colors['mu'])
        
        
        # Shift inst. frequencies back in time by Delta for better comparability with simulation (where each freq is plotted in the middle between the underlying populaiton spikes)
        ax_bottom[i][0].scatter(traj_mm.time_toff + offset[mm] - Delta, traj_mm.f_inst, \
                    c = plt.cm.coolwarm(norm(((traj_mm.f_inst-traj_mm.f_stat)).values)), \
                    edgecolors='k', s=ms**2, linewidths=.5, zorder=3) 
        # insert same trajectories in in top panels:
        ax_top[i][ix_top].plot(traj_mm.Iext_toff, traj_mm.mu_min_start, \
                                     linestyle='-', lw = .5, color='k', zorder=2) # marker='o', markersize=ms, markerfacecolor='None', \
                                     # markeredgecolor='k')
        ax_top[i][ix_top].scatter(traj_mm.Iext_toff, traj_mm.mu_min_start, \
                    c = plt.cm.coolwarm(norm(((traj_mm.f_inst-traj_mm.f_stat)).values)), lw=.5,\
                    edgecolors='k', s=ms**2, zorder=3) 
        ax_top[i][ix_top] = custom_arrow(ax_top[i][ix_top], traj_mm.Iext_toff.values, traj_mm.mu_min_start.values, head_width=.2, point_to=.6, alpha=20)

        # formatting
        if not ix_top:
          plt.setp(ax_top[i][ix_top].get_xticklabels(), visible=False)
        if i:
          plt.setp(ax_top[i][0].get_yticklabels(), visible=False)
          plt.setp(ax_top[i][1].get_yticklabels(), visible=False)
        gridline([Vr, 1], ax_bottom[i][-1], axis='y')  
        plt.setp(ax_bottom[i][-1].get_xticklabels(), visible=False)
        if i:
          ax_bottom[i][0].set_ylabel('')
          plt.setp(ax_bottom[i][0].get_yticklabels(), visible=False)
          plt.setp(ax_bottom[i][1].get_yticklabels(), visible=False)
          plt.setp(ax_bottom[i][-1].get_yticklabels(), visible=False)

      # scale bar:
      scalebar_label = '20 ms' if i==2 else ''
      scalebar = AnchoredSizeBar(ax_bottom[i][-1].transData, 20, scalebar_label, 'lower right', frameon=False, borderpad=0, \
                                 fontproperties = {'size':plt.rcParams["legend.fontsize"]}, label_top=True) #,\
                                 # bbox_to_anchor=(ax_C[1].get_xlim()[-1], ax_C[1].get_ylim()[0]), bbox_transform=ax_C[1].transData)
      ax_bottom[i][-1].add_artist(scalebar)
      # axis limits
      ax_bottom[i][0].set_ylim(ylim_panels_ii)    
      ax_bottom[i][1].set_ylim([0, 1700])    
      ax_bottom[i][-1].set_ylim([1.5*np.nanmin(mu_min_stat) , 1.2*Iext_max])
      ax_bottom[i][-1].set_xlim([t_Iext_stat_pad[0], t_Iext_stat_pad[-1]])
      # ------- simulation ------------------------------------------------------------
      if show_simulations:
        # load data
        ifa_stats = pd.read_csv(pypet_get_trajectoryPath(traj_hash = traj_trans.hash, path_to_simulations = path_to_simulations) \
                                + 'analysis_IFA_fmin-{}/data_ifa_h{}_summary.csv'.format(traj_trans.fmin, traj_trans.hash), \
                                index_col=0, squeeze=True)
        # extract ramp times
        exploration_overview = pypet_get_exploration_overview(traj_trans)
        ramp_time_sim = exploration_overview['ramp_time'].unique().astype(float)
        peak_sim = convert_input(traj_trans.input.peak, traj_trans.tm, traj_trans.C, traj_trans.Vthr, traj_trans.E_rest, unit=False)
        baseline_sim = convert_input(traj_trans.input.baseline, traj_trans.tm, traj_trans.C, traj_trans.Vthr, traj_trans.E_rest, unit=False)
        slope_sim= (peak_sim - baseline_sim)/ramp_time_sim
        
        # pick simulations with correct ramp time
        ramp_time = ramp_time_sim[np.isclose(slope_sim, m)].item()
        run_idx = pypet_find_runs(traj_trans, ['ramp_time', 'Nint'], lambda r, n: np.isclose(r, ramp_time) & (n==N_sim)) # run indices with given ramp_time and N=N_sim
        # load ifreq and IFA data
        ifreq_t = pypet_get_from_runs(traj_trans, 'network.ifreq_discr_t', run_idx=run_idx) # dict 
        ifreq = pypet_get_from_runs(traj_trans, 'network.ifreq_discr', run_idx=run_idx) # dict
        traj_trans.v_idx = run_idx[0]
        stimulus = traj_trans.derived_parameters.runs[traj_trans.v_crun]['stim_plot']
        stimulus = convert_input(stimulus, traj_trans.tm, traj_trans.C, traj_trans.Vthr, traj_trans.E_rest, unit=False)
        ifreq_t_all = list_from_dict(ifreq_t, output_numpy=True) - np.mean(traj_trans.analysis.ifreq_targetwindow)
        ifreq_all  = list_from_dict(ifreq, output_numpy=True)
        ifa_slope_sim = float(ifa_stats.loc[np.isclose(ifa_stats.ramp_time, np.min(ramp_time)) & (ifa_stats.Nint==N_sim), 'ifa_slope'])
        
        
        # plot simulated inst. freqs:
        ax_bottom[i][0].plot(ifreq_t_all, ifreq_all, '.', color='lightgrey', ms=1, label='$f_\mathrm{net}^\mathrm{inst}$ (sim)', zorder=0)
        
        # compute IFA slopes for theory trajectories
        # shift the time stamps of the theory inst freqs backwards by Delta to approx the middle between peaks that we use in simulation
        finst_t_theory = np.array(list(trajectories_all[trajectories_all.m==m]['time_toff'] + offset[m] - Delta) \
                                + list(trajectories_all[trajectories_all.m==-m]['time_toff'] + offset[-m] - Delta))
        finst_theory = np.array(list(trajectories_all[trajectories_all.m==m]['f_inst']) \
                              + list(trajectories_all[trajectories_all.m==-m]['f_inst']))
        ifa_slope_theory, ifa_intercept_theory = do_linear_regression(finst_t_theory, finst_theory)
        
        # error between theory and simulation:
        finst_interp_sim = interp_finst_discr(finst_t_theory, ifreq_t_all, ifreq_all, t_range=1.5)  
        
        error_mean = np.mean(np.abs(finst_theory - finst_interp_sim)/finst_interp_sim)
        
        # controls:
        if plot_controls:
          t_linreg = np.arange(np.min(finst_t_theory), np.max(finst_t_theory), 0.01)
          ax_bottom[i][0].plot(t_linreg, t_linreg*ifa_slope_theory + ifa_intercept_theory, 'k:', lw=.5, zorder=4)
          ax_bottom[i][0].plot(finst_t_theory, finst_interp_sim, 'r.', ms=1,  zorder=4)
          ax_bottom[i][-1].plot(np.arange(len(stimulus))*traj_trans.dt - np.mean(traj_trans.analysis.ifreq_targetwindow), \
                                stimulus, 'r--',  zorder=4)
        
        
        # add comparison between theory and simulation to table 2
        table1 = table1.append({'m':m, 
                                'ifa_theory': ifa_slope_theory,
                                'ifa_sim': ifa_slope_sim, 
                                'error': error_mean}, ignore_index=True)
        table1.to_csv('results/' + 'table1_theory_vs_simulation.csv')
        traj_trans.f_restore_default()
      if not np.isnan(traj_stat_hash): # also show the constant-drive reference for the simulations!
        # load constant drive data
        traj_stat = pypet_load_trajectory(traj_hash = traj_stat_hash, path_to_simulations=path_to_simulations)
        res = traj_stat.results.summary.scalarResults
        level_stat = res['parameters.input.level']
        fnet_stat = res['freq_net']
        if not traj_stat.Nint == N_sim:
          print('Simulations for constant drive loaded for network size: ', traj_stat.Nint)
        
        # add asymptotic network freqs:
        level_stat = convert_input(res['parameters.input.level'], traj_stat.tm, traj_stat.C, traj_stat.Vthr, traj_stat.E_rest, unit=False)
        Ihopf = convert_input(traj_stat.linear_stability.Icrit_nA, traj_stat.tm, traj_stat.C, traj_stat.Vthr, traj_stat.E_rest, unit=False)
        fnet_stat = res['freq_net']
      
        # --- add asymptotic curve --- 
        fnet_stat_interp = np.interp(Iext_stat, level_stat, fnet_stat)
        fnet_stat_interp[Iext_stat<Ihopf] = nan
        ax_bottom[i][0].plot(t_Iext_stat, fnet_stat_interp, '--', color='dimgrey', lw=.5, zorder=1)
    
    # common colorbar
    cb = plt.colorbar(im, cax=ax_cbar, label=r'$f_\mathrm{net}^\mathrm{inst}-f_\mathrm{net}^\mathrm{\infty}$ [Hz]',\
                      ticks=[fmin, int(fmin/2), 0, int(fmax/2), fmax])
    cb.ax.plot([-1e7, 1e7],[0]*2, 'w', lw=1) 
    
    # formatting 
    ax_top[0][1].set_ylabel(r'$\mu_\mathrm{min}$', labelpad=-.2)
    ax_top[0][1].set_xlabel(r'drive $\hat{I}_\mathrm{E}$', labelpad=-.2)
    ax_bottom[0][0].set_ylabel('freq [Hz]')
    ax_bottom[0][1].set_ylabel('rate\n[spk/s]')
    ax_bottom[0][-1].set_ylabel('voltage,\ndrive')

    # common legend
    h_stat_sim = Line2D([0], [0], linestyle='--', lw=.5, color='dimgrey', label=r'$f_\mathrm{net}^\mathrm{\infty}$')
    h_inst_sim = Line2D([0], [0], marker='.', color='lightgrey',  linestyle='', label='$f_\mathrm{net}^\mathrm{inst}$')
    h_stat_th = Line2D([0], [0], linestyle='-', lw=.5, color='k', label=r'$f_\mathrm{net}^\mathrm{\infty}$')
    h_inst_th = Line2D([0], [0], marker='o', color='dimgrey', markersize=ms, markerfacecolor='None', markeredgecolor='k', \
                          markeredgewidth=.5, linestyle='', label='$f_\mathrm{net}^\mathrm{inst}$')
    
    ax_bottom[-1][0].legend(bbox_to_anchor=(.5, 3.2), bbox_transform = ax_bottom[-1][0].transAxes, loc='lower left', borderaxespad=0.,\
                            handlelength=1., handles=[h_stat_sim, h_inst_sim], labelspacing=0.4, ncol=2, title='simulation')

    
    handles, labels = ax_bottom[-1][-1].get_legend_handles_labels()
    # extra_handle = Line2D([0], [0], marker='.', color='k', linestyle='', label='inst')
    handles_theory = [h_stat_th] + handles[:3] + [h_inst_th] + handles[3:]
    ax_bottom[-1][-1].legend(handles = handles_theory, bbox_to_anchor=(.5, 1.6), bbox_transform = ax_bottom[-1][0].transAxes, loc='lower left', borderaxespad=0.,\
                             handlelength=1., labelspacing=0.4, ncol=2, title='theory')
      
    print(table1)
  return fig


def interp_finst_discr(t, t_sim, f_sim, t_range=1.5):
  '''
  average simulated inst freqs f_sim around time point t (+/- t_range ms)
  compare this estimate to theoretical estimate of f_inst at time t
  '''
  if not np.isscalar(t):
    f_est = np.array([interp_finst_discr(ti, t_sim, f_sim, t_range=t_range)  for ti in t])
  else:
    # find sim points close to theory time point
    ix = np.where(np.abs(t_sim - t) < t_range)[0]
    # average sim freqs in that window and compare theory to that 
    f_est = np.nanmean(f_sim[ix])
  return f_est

#%% Fig 8
def plot_figure_8(D, Delta, K, tm, Vr, traj_hash,\
                              title = ['stable \n fixed point', 'pathological \noscillation', 'period-2 \noscillation', 'regular \n oscillation'], \
                              dI = 0.01, reset=False, mu0=0, dt=0.01, tmax=300, \
                              path_to_simulations = './simulations/',  path_to_figures = './figures/'): #make_fig_DDE_dynspectrum
  '''
  show dynamical spectrum of the DDE system 

  Parameters
  ----------
  D : noise intensity
  Delta : syn delay, ms
  K : coupling strength
  tm : membrane time constant, ms
  Vr : reset
  Iext_val : 4 example drives, typically: before bifurcation, just after bifurcation, sparse synch, full synch
    (see also title)
    The default is [0.5, 0.8, 4.24, 8.9].
  title : description of dynamical states for the drives in Iext_val
    The default is ['stable FP', 'pathological \noscillation', 'sparse synch', 'full synch'].
  xmax_zoom : right xlimit for time plots that need zoom, optional
    The default is 25.
  tmax : max time for numerical integration of DDEs, optional
    The default is 200.
  dt : integration time step, optional
    The default is 0.01.
  mu0 : initial value for mean membrane potential, optional
    The default is 0.
  reset : should be false, otherwise numerical integration errors around bifurcation, optional
    The default is False.

  Returns
  -------
  None. (figure is stored in path_to_figures)
  '''
  print('\n\nPlotting Fig 8...')
  # load Hopf bifurcation from spk network simulation with same parameters
  info = pd.read_csv(pypet_get_trajectoryPath(traj_hash = traj_hash, path_to_simulations=path_to_simulations) + 'info.csv', index_col=0, squeeze=True, header=None)
  if not (np.isclose(D,info['D'], atol=1e-3) and np.isclose(Delta,info['Delta']) and np.isclose(K,info['K']) and np.isclose(tm, info['tm']) \
          and np.isclose(Vr,info['Vr'])):
    raise ValueError('Spiking network did not have same parameters as used here!', info['D'], info['Delta'], info['K'], info['tm'], info['Vr'])
  Ihopf = info['I_hopf']
  
  # from here on only theory:
  Ifull = get_pt_fullsynch(D, Delta, K, tm)  
  I_bifurcation, Imin_p2, Imin_p1 \
    = dde_bifurcation_analysis_numerical(D, Delta, K, tm, Vr, dI = dI, dt = dt)
  col = ['royalblue','orangered', 'y', 'mediumseagreen']
  
  # add dde info:
  info['dde_I_bifurcation'], info['dde_Imin_p2'], info['dde_Imin_p1'] \
  = I_bifurcation, Imin_p2, Imin_p1
  info.to_csv(pypet_get_trajectoryPath(traj_hash = traj_hash, path_to_simulations=path_to_simulations) + 'info.csv', header=None)
  
  # pick example drive levels for each regime in its center:
  Iex_fp = np.mean([0,I_bifurcation])
  Iex_pathological = np.mean([I_bifurcation, Imin_p2])
  Iex_p2 = np.mean([Imin_p2, Imin_p1])
  Iex_p1 = np.mean([Imin_p1, Ifull])
  
  Iext_val = [Iex_fp, Iex_pathological, Iex_p2, Iex_p1]
  
  Iext_min = get_Iext_lower_bound(D, Delta, K, tm, Vr, reset=reset, Iext_max=Iex_p1, dI=.01)[0]

  # --- construct figure
  fig_width= plos_width_fullpage # width_a4_wmargin #21*cm
  fig_height = fig_width*0.5 # 9*cm # 10.5*cm
  fig = plt.figure(figsize=(fig_width, fig_height))#, constrained_layout=True)
  
  ncols = len(Iext_val)
  gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[20,1], hspace=.3)#, width_ratios=[5,2], height_ratios=[2,3])
  
  ax_Iext = fig.add_subplot(gs[-1,:])
  for dir in ['right', 'top', 'left']:
    ax_Iext.spines[dir].set_visible(False)
  
  yI = [1,1]
  ax_Iext.fill_between([0, I_bifurcation], yI, color= col[0])
  ax_Iext.fill_between([I_bifurcation, Imin_p2], yI, color= col[1])
  ax_Iext.fill_between([Imin_p2, Imin_p1], yI, color= col[2])
  ax_Iext.fill_between([Imin_p1, Ifull*1.5], yI, color= col[3])
  
  add_ticks(ax_Iext, [Ihopf, Iext_min, Ifull], [str_Iextcrit, str_Iextmin, str_Iextfull], 'x')
  
  ax_Iext.set_yticks([])
  ax_Iext.set_ylim([0,1])
  ax_Iext.set_xlim([0,Ifull*1.01])
  ax_Iext.set_xlabel('external drive '+str_Iext)
  
  gs_frames = gs[0].subgridspec(1, ncols, wspace=.4)
  ax_t = [None]*ncols
  ax_p = [None]*ncols
  tlim = [200, 35, 35, 35]
  for i in range(ncols):

    ax_Iext.plot(Iext_val[i], .4, marker='v', markerfacecolor=col[i], markeredgecolor='k')
    # inner
    gs_sub = gs_frames[i].subgridspec(2,1, height_ratios=[2,1], hspace=.5)
    gs_t = gs_sub[0].subgridspec(2,1)
    ax_t[i] = gs_t.subplots(sharex=True)
    ax_p[i] = fig.add_subplot(gs_sub[1])

    # numerically integrate DDE system
    t, mu, r, ix_max, ix_min_real, mu_max, mu_reset, mu_min_real, ix_min_theory, mu_min_theory, toff_theory \
    = integrate_dde_numerically(Iext_val[i], D, Delta, K, tm, Vt=1 , Vr = Vr, tmax=tmax, dt=dt, mu0=mu0, reset=reset, \
                                        plot=False, rtol_conv=1e-3)[:-2]
    # plot in time
    ax_t[i][0].plot(t, r, color = col[i])
    gridline([Vr, 1], ax_t[i][1], axis='y')
    ax_t[i][1].plot(t, mu, color = col[i])
    ax_t[i][0].set_title(title[i])
    # remove spines
    for direction in ['right', 'top']:
      ax_t[i][0].spines[direction].set_visible(False)
      ax_t[i][1].spines[direction].set_visible(False)
    ax_t[i][1].set_xlim([0, tlim[i]])
    
    if i==1:
      # mark period of pathological oscillation
      ax_t[i][1].arrow(ix_min_real[7]*dt, .2, -2*Delta, 0, color='k', head_width=.3, head_length=.5, length_includes_head=True) 
      ax_t[i][1].arrow(ix_min_real[6]*dt, .2, 2*Delta, 0, color='k', head_width=.3, head_length=.5, length_includes_head=True)
      ax_t[i][1].text(ix_min_real[7]*dt-Delta, -.10, "$2\Delta$", va='top', ha='center')
    
    # plot in phase space
    ax_p[i].plot(mu, r, color = col[i])#, lw=.8)
    if not i:
      ax_p[i].plot(Iext_val[i], 0, 'ko')
    else:
      ax_p[i].plot(Iext_val[i], 0, 'o', markerfacecolor='w', markeredgecolor='k')
  
  mu_lim = ax_t[-1][1].get_ylim()
  for i in range(ncols-1): 
    ax_t[i][1].set_ylim(mu_lim)
  
  # labels  
  ax_t[0][0].set_ylabel('r [spks/s]')
  ax_t[0][1].set_ylabel('$\mu$')
  ax_t[0][1].set_xlabel('time [ms]', labelpad=-.2)
  ax_t[1][1].set_xlabel('time [ms]', labelpad=-.2)
  ax_p[0].set_ylabel('r [spks/s]')
  ax_p[0].set_xlabel('$\mu$', labelpad=-.8, zorder=10)

  fig.savefig(path_to_figures + 'Fig8.pdf', bbox_inches = 'tight')
  fig.savefig(path_to_figures + 'Fig8.tif', bbox_inches = 'tight')
  return 

#%% Fig 9
def plot_figure_9(parameters, path_to_figures = './figures/'):
  print('\n\nPlotting Fig 9...')
  D, Delta, K, tm, Vr, Iext \
  = parameters['D'], parameters['Delta'], parameters['K'], parameters['tm'], parameters['Vr'], parameters['Iext']
  
  # Gaussian-drift approx with reset:
  mu_min, mu_max, f_net, f_unit, sat, zeta, t_on, t_off_a, mu_reset, _, _, t_a, mu_a, r_a, I_a \
  = gaussian_drift_approx_stat(Iext, D, Delta, K, tm, Vr=Vr, reset=True, return_traces = True)
  
  # Gaussian-drift approx without reset:
  mu_min_nr, mu_max_nr, f_net_nr, f_unit_nr, sat_nr, zeta_nr, t_on_nr, t_off_a_nr, mu_reset_nr, _, _,t_a_nr, mu_a_nr, r_a_nr, I_a_nr \
  = gaussian_drift_approx_stat(Iext, D, Delta, K, tm, Vr=Vr, reset=False, return_traces = True)
  
  # --- numerical integration of DDE
  # with reset:
  t_n, mu_n, r_n, _, _,_,_, mumin_real_n, _, _, t_off_n, state, _ \
  = integrate_dde_numerically_until_convergence(Iext, D, Delta, K, tm, Vr, reset=True, return_last_cycle=True, rtol_convergence=1e-2)  
  
  # without reset:
  t_n_nr, mu_n_nr, r_n_nr, _, _,_,_, mumin_real_n_nr, _, _, t_off_n_nr, _, _ \
  = integrate_dde_numerically_until_convergence(Iext, D, Delta, K, tm, Vr, reset=False, return_last_cycle=True, rtol_convergence=1e-2, plot=True) 
  
  # --- plot -----------------------------------------------------------------------------------------------  
  # --- construct figure -----------------------------------
  with plt.rc_context({"axes.labelsize": 8, "axes.titlesize": 8, "font.size": 8}):
    fig_width= width_a4_wmargin*.5
    fig_height = width_a4_wmargin*.8 
    fig = plt.figure(figsize=(fig_width, fig_height)) #, constrained_layout=True)
    
    gs = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[3,3,1], hspace=.6)
    
    gs_A = gs[0].subgridspec(2,2, height_ratios=[1,3], width_ratios=[1.5,1], wspace=.3)
    gs_B = gs[1].subgridspec(2,2, height_ratios=[1,3], width_ratios=[1.5,1], wspace=.3)
    gs_C = gs[2].subgridspec(1,2, width_ratios=[2,1])
    
    ax_A = gs_A.subplots(sharex='col')
    ax_B = gs_B.subplots(sharex='col')
    ax_C = fig.add_subplot(gs_C[0])
    
    ax_A[0,1].remove()
    ax_B[0,1].remove()
    
    
    ax_A[0,0].text(-.37, 1.4, string.ascii_uppercase[0], transform=ax_A[0,0].transAxes, size=panel_labelsize, weight='bold')
    ax_B[0,0].text(-.37, 1.4, string.ascii_uppercase[1], transform=ax_B[0,0].transAxes, size=panel_labelsize, weight='bold')
    ax_B[0,0].text(-.37, -5.5, string.ascii_uppercase[2], transform=ax_B[0,0].transAxes, size=panel_labelsize, weight='bold')
    
      
    # --- fill figure -----------------------------------
    ax_A = aux_plot_traces_analyt_vs_num_wzoom(fig, ax_A, t_a_nr, r_a_nr, mu_a_nr, t_off_a_nr, I_a_nr, t_n_nr, r_n_nr, mu_n_nr, t_off_n_nr, \
                                                mu_max_nr, nan, mu_min_nr, Iext, D, Delta, Vr, legend=True, t0= t_off_a_nr-t_off_a,\
                                                zy0 = mumin_real_n_nr - .1, zy1 = np.max([mu_max + .1, 1.05]))
    ax_B = aux_plot_traces_analyt_vs_num_wzoom(fig, ax_B, t_a, r_a, mu_a, t_off_a, I_a, t_n, r_n, mu_n, t_off_n, \
                                               mu_max, mu_reset, mu_min, Iext, D, Delta, Vr, legend=False, \
                                               zy0 = mumin_real_n - .2, zy1 = np.max([mu_max + .1, 1.05]))
    ax_C = aux_plot_reset_visualized(ax_C, mu_max, mu_reset, D, Vr, legendloc='side')
    
    
    #plt.setp(ax_A[1,0].get_xticklabels(), visible=False)
    ax_A[1,0].set_xlabel('')
    ax_A[1,1].set_xlabel('')
    
    # -- remove spines -- 
    for ax in [ax_A[0,0], ax_A[1,0], ax_A[1,1], ax_B[0,0], ax_B[1,0], ax_B[1,1], ax_C]:
      ax.spines['right'].set_visible(False)
      ax.spines['top'].set_visible(False)

  fig.savefig(path_to_figures + 'Fig9.pdf', bbox_inches = 'tight')
  fig.savefig(path_to_figures + 'Fig9.tif', bbox_inches = 'tight')
    
  return 

def aux_plot_traces_analyt_vs_num_wzoom(fig, ax, t_a, r_a, mu_a, t_off_a, I_a, \
                                          t_n, r_n, mu_n, t_off_n, \
                                          mu_max, mu_reset, mu_min, Iext, D, Delta, Vr, \
                                          legend=True, label=True, t_margin=.1, t0=0, zy0=None, zy1=None, ms=5):
  offset = t_off_a - t_off_n
  
  # add grid
  for l in [t_off_a-2*Delta, t_off_a-Delta,t_off_a, t_off_a+Delta]:
    for i in range(2):
      gridline(l, ax[i], 'x', zorder=100)
      
  ## --- analytical traces
  ax[0,0].plot(t_a, r_a, color=my_colors['fnet'], label='r')
  ax[0,0].plot(t_n + offset, r_n, ':', color=my_colors['fnet'], label='r (num)')
  if label:
    ax[0,0].set_ylabel('rate $r$ \n [spk/s]')
  ax[0,0].set_ylim(bottom=0)
  
  for ax1 in [ax[1,0], ax[1,1]]:
    gridline([Vr, 1], ax1, 'y', zorder=100)
    ax1.fill_between(t_a[0<=t_a], mu_a[0<=t_a]-3*np.sqrt(D), mu_a[0<=t_a]+3*np.sqrt(D), color='lightgray', zorder=1)#, label=r'$\mu\pm3\sqrt{D}$')
    if np.isnan(mu_reset):
      ax1.plot(t_a[t_a>=0], mu_a[t_a>=0], color=my_colors['mu'], label='$\mu$') # do not necessarily show all to toff-2Delta if that is before time 0
    else:
      ax1.plot(t_a[(0<=t_a) & (t_a<t_off_a)], mu_a[(0<=t_a) & (t_a<t_off_a)], color=my_colors['mu'], label='$\mu$')
    ax1.plot(t_off_a, mu_max, '.', ms=ms, zorder=1e10, color=my_colors['max'],  label='$\mu_\mathrm{max}$') #, fillstyle='none'
    ax1.plot(t_off_a, mu_reset, '.', ms=ms,zorder=1e10, color=my_colors['reset']) #markeredgewidth= 2,
    ax1.plot(0, mu_min, '.',ms=ms, zorder=1e10, color=my_colors['min'],  label='$\mu_\mathrm{min}$')
    ax1.plot(t_off_a+Delta, mu_min, '.', ms=ms,zorder=1e10, color=my_colors['min'])# fillstyle='none')
    ax1.axhline(Iext, color=my_colors['iext'], label=r'$I_\mathrm{E}$')
    ax1.plot(t_a, I_a, 'k--', label=r'$I$')
    if np.isnan(mu_reset):
      ax1.plot(t_n + offset, mu_n, ':', color=my_colors['mu'], label='DDE')
    else:
      ax1.plot(t_a[t_a>=t_off_a], mu_a[t_a>=t_off_a], color=my_colors['mu'])
      ax1.plot((t_n + offset)[t_n<t_off_n], mu_n[t_n<t_off_n], ':', color=my_colors['mu'], label='DDE')
      ax1.plot((t_n + offset)[t_n>=t_off_n], mu_n[t_n>=t_off_n], ':', color=my_colors['mu'], label='DDE')
    
    ax1.set_xticks([0, t_off_a, t_off_a+Delta])
    ax1.set_xticklabels([r'$t_\mathrm{on}=0$', r'$t_\mathrm{off}$', r'$t_\mathrm{off} + \Delta$'])
    ax1.set_xlabel('time within cycle [ms]')

    
  if label:
    ax[1,0].set_ylabel('voltage,\ndrive', labelpad=-.1)
  
  # determine zoom-in region
  zx0 = t_off_a - Delta/3 #np.where(np.abs(mu_a - mu_n)) # where error gets bigger
  zx1 = t_off_a+Delta+ t_margin
  if not zy0:
    zy0 = mu_min - .1
  if not zy1:
    zy1 = np.max([mu_max + .1, 1.02])
  
  ax[1,0].set_xlim([t0-t_margin, t_off_a+Delta+2*t_margin])
  ax[1,0].set_ylim([np.min(I_a)-.3, Iext+.5]) #mu_max+3*np.sqrt(D)+.2]) #bottom=1.1*(mu_min-3*np.sqrt(D)))
  # inset zoomed in
  ax[1,1].set_xlim([zx0, zx1])
  ax[1,1].set_ylim([zy0, zy1])  
    

  if legend:
    ax[1,0].legend(loc='lower left', bbox_to_anchor=(1.02, -.1), bbox_transform=ax[0,0].transAxes, ncol=2, handlelength=1.5, borderaxespad=0.)
  else: # hack for thesis
    hreset = Line2D([0], [0], linestyle='', marker='.', color=my_colors['reset'], markersize=ms, label='$\mu_\mathrm{reset}$')
    ax[1,1].legend(handles = [hreset], loc='lower right', bbox_to_anchor=(1, 1.1), bbox_transform=ax[1,1].transAxes, handlelength=1.5, borderaxespad=0.)
  # annotations
  rmax, T, dr = 1.2*np.nanmax(r_a), t_off_a+Delta, 25
  ax[0,0].fill_between([0, t_off_a], [rmax+dr, rmax+dr], [rmax, rmax], color='lightgray',  hatch='//')
  ax[0,0].fill_between([ t_off_a, T], [rmax+dr, rmax+dr], [rmax, rmax], color='dimgray',  hatch="\\" )
  ax[0,0].text(T/2, rmax+dr, 'T={:.2f}ms'.format(T), va='bottom', ha='center', transform=ax[0,0].transData)
  ax[0,0].set_ylim([0, rmax+100])
  
  ax[0,0].arrow(t0-t_margin, 50, t_off_a-2*Delta-t0+t_margin, 0, head_width=35, head_length=0.2, lw=.5, length_includes_head=True, color='k')
  ax[0,0].text(t0, 51, r'$r\approx 0$', va='bottom', ha='left', transform=ax[0,0].transData)

  ax[1,0].arrow(t0-t_margin, Iext*.85, t_off_a-Delta-t0+t_margin, 0, head_width=0.5, head_length=0.2, lw=.5, length_includes_head=True, color='k')
  ax[1,0].text(t0, Iext*.8, r'$I\approx$ '+str_Iext, va='top', ha='left', transform=ax[1,0].transData)
  ax[1,0].plot([0,0], [ax[1,0].get_ylim()[0],mu_min], lw=1, color=my_colors['min'], linestyle=':')
  ax[0,0].plot([0,0], [0,rmax+dr], lw=1, color=my_colors['min'], linestyle=':')
  
  # draw inset box
  ax[1,0].plot([zx0, zx1, zx1, zx0, zx0], [zy1, zy1, zy0, zy0, zy1], 'k', lw=.5, zorder=1e50)  
  
  # add inset lines 
  con_top = ConnectionPatch(xyA=(zx1,zy1), coordsA=ax[1,0].transData, 
                            xyB=(0,1), coordsB=ax[1,1].transAxes, color = 'k', lw=.5, zorder=1)
  con_bottom = ConnectionPatch(xyA=(zx1,zy0), coordsA=ax[1,0].transData, 
                              xyB=(0,0), coordsB=ax[1,1].transAxes, color = 'k', lw=.5, zorder=1)
  fig.add_artist(con_top)
  fig.add_artist(con_bottom)
  
  return ax

def aux_plot_reset_visualized(ax, mu_max, mu_reset, \
                              D, Vr, \
                              dv = 0.01, legendloc='side'):
  v = np.arange(Vr-1, mu_max+4*np.sqrt(D), dv)
  p_max = get_gauss(v, mu_max, np.sqrt(D))
  p_reset = get_gauss(v, mu_reset, np.sqrt(D))
  
  gridline([Vr, 1], ax, 'x', zorder=1e3)
  ax.plot(v, p_max, color=my_colors['max'], zorder=2, label='before')
  ax.plot(v, p_reset, color=my_colors['reset'], zorder=3, label='after')
  ax.plot(v[v>=1]-(1-Vr), p_max[v>=1], color='gray', zorder=1)
  ax.fill_between(v[v>=1], p_max[v>=1], facecolor='w', hatch='///', edgecolor=my_colors['max'], label='saturation', lw=0)
  ax.fill_between(v[v>=1]-(1-Vr), p_max[v>=1], color='w', hatch='///', edgecolor='gray', lw=0)
  
  if legendloc=='side':
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.) 
  elif legendloc=='top':
    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
              ncol=2, mode="expand", borderaxespad=0.)
  ax.set_xlim([Vr-.1, mu_max+4*np.sqrt(D)])
  ax.set_ylim(bottom=0)
  ax.set_xlabel('membrane potential $V$')
  ax.set_ylabel('$p(V, t_\mathrm{off})$')
  ax.set_xticks([Vr, mu_reset, mu_max, 1])
  ax.set_xticklabels(['$V_R$', '$\mu_\mathrm{reset}$', '$\mu_\mathrm{max}$', '$V_T$'], rotation=-25) # adjust in case Vr not int
  return ax

#%% Fig 10
def plot_figure_10(parameters, n_num = 10, n_analyt = 100, show_real_mumin=False, path_to_figures = './figures/'):
  '''
  compare the analytical approx for fnet, mumax etc with the result from numerical integration of the DDE
  (show that error coming from analyt approx is small)
  '''
  print('\n\nPlotting Fig 10...')
  D, Delta, K, tm, Vr \
  = parameters['D'], parameters['Delta'], parameters['K'], parameters['tm'], parameters['Vr']

  # --- data -------------------------------------------------------------------------------------------------------------
  # Iext arrays from lower bound to pt of full synch, with or without reset (_r)
  Iext_min =  get_Iext_lower_bound(D, Delta, K, tm, Vr, reset=False, Iext_max=10)[0]
  Iext_min_r =  get_Iext_lower_bound(D, Delta, K, tm, Vr, reset=True, Iext_max=6)[0]
  Iext_full = get_pt_fullsynch(D, Delta, K, tm)
  
  Iext_analyt_r = np.linspace(Iext_min_r, Iext_full, n_analyt, endpoint=True)
  Iext_num_r = np.linspace(Iext_min_r, Iext_full, n_num, endpoint=True)
  
  Iext_analyt = np.linspace(Iext_min, Iext_full, n_analyt, endpoint=True)
  Iext_num = np.linspace(Iext_min, Iext_full, n_num, endpoint=True)
  
  # analytical, no reset
  mu_min_analyt, mu_max_analyt, f_net_analyt, f_unit_analyt, sat_analyt, zeta_analyt, t_on_analyt, t_off_analyt, mu_reset_analyt \
  = gaussian_drift_approx_stat(Iext_analyt, D, Delta, K, tm, Vr=Vr, reset=False)[:-2]
  
  # numerical, no reset
  mu_min_num, mu_max_num, f_net_num, f_unit_num, zeta_num, t_on_num, t_off_num, mu_reset_num \
  = gaussian_drift_approx_stat_numerical(Iext_num, D, Delta, K, tm, Vr=Vr, reset=False, tmax = 400, rtol_convergence=1e-2)[:-3]
  
  # analytical, with reset
  mu_min_analyt_r, mu_max_analyt_r, f_net_analyt_r, f_unit_analyt_r, sat_analyt_r, zeta_analyt_r, t_on_analyt_r, t_off_analyt_r, mu_reset_analyt_r \
  = gaussian_drift_approx_stat(Iext_analyt_r, D, Delta, K, tm, Vr=Vr, reset=True)[:-2]
  
  # numerical, with reset
  mu_min_num_r, mu_max_num_r, f_net_num_r, f_unit_num_r, zeta_num_r, t_on_num_r, t_off_num_r, mu_reset_num_r, mu_min_num_r_real \
  = gaussian_drift_approx_stat_numerical(Iext_num_r, D, Delta, K, tm, Vr=Vr, reset=True, tmax = 400, rtol_convergence=1e-2)[:-2]

  # --- construct figure -------------------------------------------------------------------------------------------------------------
  fig_width = plos_width_text #width_a4_wmargin*.4
  fig_height =  fig_width*1.3
  
  fig = plt.figure(figsize=(fig_width, fig_height))
  gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1,3])
  gs_plot = gs[1].subgridspec(2,1, height_ratios=[1,1])
  ax = gs_plot.subplots(sharex=True)
  
  gridline([Iext_min, Iext_min_r, Iext_full], ax, 'x')
  gridline([Iext_full], ax, 'x', linestyle='--')
  
  ax[0].plot(Iext_analyt, f_net_analyt, '--', color=my_colors['fnet'])
  ax[0].plot(Iext_num, f_net_num, 's', markersize=1.5, color=my_colors['fnet'])
  ax[0].plot(Iext_analyt_r, f_net_analyt_r, '-', color=my_colors['fnet'])
  ax[0].plot(Iext_num_r, f_net_num_r, '.', color=my_colors['fnet'])
  
  ax[1].plot(Iext_analyt, mu_max_analyt, '--', color=my_colors['max'])
  ax[1].plot(Iext_analyt, mu_min_analyt, '--', color=my_colors['min'])
  ax[1].plot(Iext_num, mu_max_num, 's', markersize=1.5, color=my_colors['max'])
  ax[1].plot(Iext_num, mu_min_num, 's', markersize=1.5, color=my_colors['min'])
  
  ax[1].plot(Iext_analyt_r, mu_max_analyt_r, color=my_colors['max'])
  ax[1].plot(Iext_analyt_r, mu_reset_analyt_r, color=my_colors['reset'])
  ax[1].plot(Iext_analyt_r, mu_min_analyt_r, color=my_colors['min'])
  ax[1].plot(Iext_num_r, mu_max_num_r, '.', color=my_colors['max'])
  ax[1].plot(Iext_num_r, mu_reset_num_r, '.', color=my_colors['reset'])
  ax[1].plot(Iext_num_r, mu_min_num_r, '.', color=my_colors['min'])
  if show_real_mumin:
    ax[1].plot(Iext_num_r, mu_min_num_r_real, 'x', color=my_colors['min'])
  
  # formatting
  gridline([0,1], ax[1], 'y')
  ax[0].set_xlim([0, 1.05*Iext_full])
  ax[0].set_ylim(bottom = 0)
  
  if np.min(mu_min_analyt_r) < -2:
    ax[1].set_yticks([-4, -2, Vr, 1])
#    ax[1].set_yticklabels(['-4', '-2', '$V_R=${:.0f}'.format(Vr), '$V_T=1$'])
  else:
    ax[1].set_ylim(bottom=-1.5)
  
  ax[0].set_ylabel('frequency [Hz]\n')
  ax[1].set_ylabel('membrane potential')
  ax[1].set_xlabel('external drive $I_\mathrm{E}$')
  # ax[0].legend(loc='upper right', labelspacing=.2, borderaxespad=.1)
  # ax[1].legend(loc='lower left', labelspacing=.2, borderaxespad=.2)
  
  # legend by hand 
  har = Line2D([0], [0], linestyle='-', color='darkgrey', label='analytical, with reset')
  hnr = Line2D([0], [0], marker = '.', linestyle='', color='darkgrey', label='numerical, with reset')
  
  ha = Line2D([0], [0], linestyle='--', color='darkgrey', label='analytical, without reset')
  hn = Line2D([0], [0], marker = 's', markersize=1.5, linestyle='', color='darkgrey', label='numerical, without reset')
  
  hnet = Patch(facecolor=my_colors['fnet'], label=r'$f_\mathrm{net}$') #edgecolor='r', 
  hmax = Patch(facecolor=my_colors['max'], label=r'$\mu_\mathrm{max}$') #edgecolor='r', 
  hreset = Patch(facecolor=my_colors['reset'], label=r'$\mu_\mathrm{reset}$') #edgecolor='r', 
  hmin = Patch(facecolor=my_colors['min'], label=r'$\mu_\mathrm{min}$') #edgecolor='r', 
  hnr_real = Line2D([0], [0], marker = 'x', linestyle='', color=my_colors['min'], label='$min(\mu)$ (num, with reset)')
  if show_real_mumin:
    handles2 = [har, ha, hn, hnr, hnr_real]
  else:
    handles2 = [har, ha, hn, hnr]
    
  l1 = ax[0].legend(handlelength=1, handles=[hnet, hmax, hreset, hmin],
                bbox_to_anchor=(0, 0), loc='lower left')
  ax[0].add_artist(l1)
  
  ax[0].legend(handlelength=1.5, handles=handles2,
                bbox_to_anchor=(1, 1), loc='lower right', borderaxespad=0., ncol=2)
  
  for axx in ax:
    axx.spines['right'].set_visible(False)
    axx.spines['top'].set_visible(False)
  
  fig.savefig(path_to_figures + 'Fig10.pdf', bbox_inches = 'tight')
  fig.savefig(path_to_figures + 'Fig10.tif', bbox_inches = 'tight')
  
  return 

#%% Supplementary Figs S1A, S1B
def plot_figure_S1A(traj_hash, reset=True, path_to_figures = './figures/'):
  
  datapath = 'results/gaussian_drift_approx_constant_drive_performance_check/'
  df_runs = pd.read_csv(datapath+'df_parameters_per_run_h{}_evaluated.csv'.format(traj_hash), index_col=0, squeeze=True)
  df_net = pd.read_csv(datapath+'df_network_configurations_h{}_evaluated.csv'.format(traj_hash), index_col=0, squeeze=True)
  
  exploration, p_fix, p_var = recover_exploration(df_runs) 
  
  # recompute interpolated theory curves for plotting:
  n_config = int(df_runs.config.max() + 1) # number of parameter configurations
  theory = {} # dict of dicts: interpolated theory curves for plotting
  # loop over parameter configurations
  for c in range(n_config):    
    theory[c] \
    = get_error_gauss_drift_approx(df_runs.loc[df_runs.config==c, 'Iext'].values, df_runs.loc[df_runs.config==c, 'freq_net'].values, \
                                   df_runs.loc[df_runs.config==c, 'freq_unit_mean'].values, df_net.loc[c,'Ifull_sim'], df_net.loc[c,'Ifull'], \
                                   df_net.loc[c, 'D'], df_net.loc[c, 'tl'], df_net.loc[c, 'K'], df_net.loc[c, 'tm'], df_net.loc[c, 'Vr'], \
                                   reset=reset)[-1]
  
  # --- construct figure  
  fig_width = width_a4_wmargin
  fig_height = width_a4_wmargin*1.4
  fig = plt.figure(figsize=(fig_width, fig_height))#, constrained_layout=True)
  gs = gridspec.GridSpec(3, 2, figure=fig, width_ratios = [1,2.5], height_ratios=[1,3,3], hspace=.3, wspace=.3)
  
  gs_B_sup = gs[0,1].subgridspec(1,3, wspace=.4)
  gs_B = [None]*3
  ax_B = [[None]*2]*3
  for i in range(3):
    gs_B[i] = gs_B_sup[0,i].subgridspec(1,2,width_ratios=[20,1], wspace=.05)
    ax_B[i] = gs_B[i].subplots()
  ax_C = fig.add_subplot(gs[1,:])
  ax_D = fig.add_subplot(gs[2,:])
  
  # --- fill figure
  ax_A = fig.add_subplot(gs[0,0]) 
  if 'D' in p_var:
    ax_A = plot_visualize_D(exploration['D'][[0,2,4,5]], ax=ax_A)
  else:
    ax_A = plot_visualize_D(np.array([exploration['D']]), ax=ax_A)
  
  ax_B = plot_performance_idx_manuscript(df_net, p_var, ax=ax_B, cbar_labelsize= matplotlib.rcParams['ytick.labelsize']-1)
  ax_C = plot_performance_details_frequencies(df_runs, df_net, theory, external_ax=ax_C)
  ax_D = plot_performance_details_gaussdynamics(df_runs, df_net, theory, reset=reset, external_ax=ax_D)
  # add labels
  ax_C.text(-.1, 1.6, 'a', transform=ax_C.transAxes, size=panel_labelsize, weight='bold')
  ax_C.text(.29, 1.6, 'b', transform=ax_C.transAxes, size=panel_labelsize, weight='bold')
  ax_C.text(-.1, 1, 'c', transform=ax_C.transAxes, size=panel_labelsize, weight='bold')
  ax_C.text(-.1, -.25, 'd', transform=ax_C.transAxes, size=panel_labelsize, weight='bold')
  
  fig.savefig(path_to_figures+'FigS1A.pdf', bbox_inches='tight')
  fig.savefig(path_to_figures+'FigS1A.tif', bbox_inches='tight')
  
  return 

def plot_figure_S1B(traj_hash, path_to_figures = './figures/'):
  
  datapath = 'results/gaussian_drift_approx_constant_drive_performance_check/'
  df_runs = pd.read_csv(datapath+'df_parameters_per_run_h{}_evaluated.csv'.format(traj_hash), index_col=0, squeeze=True)
  df_net = pd.read_csv(datapath+'df_network_configurations_h{}_evaluated.csv'.format(traj_hash), index_col=0, squeeze=True)
  exploration, p_fix, p_var = recover_exploration(df_runs)  
  
  # --- construct figure
  fig_width= width_a4_wmargin# 15*cm
  fig_height = 7*cm 
  fig = plt.figure(figsize=(fig_width, fig_height))#, constrained_layout=True)
  gs = gridspec.GridSpec(2, 4, figure=fig, hspace=.2, wspace=.6, width_ratios=[.2, 1,1, 1]) #, height_ratios=[1,30], hspace=.2, wspace=.4)
  
  i0=1
  gs_ffull_t = gs[0,i0].subgridspec(1,2, width_ratios=[10,.5], wspace=.1)
  ax_ffull_t = gs_ffull_t.subplots()
  ax_ffull_t[0].set_title('theory')
  ax_ffull_t[0].text(-1.1, .5, 'network\nfrequency\n$f_\mathrm{net}^\mathrm{full}$', transform=ax_ffull_t[0].transAxes, ha='left')

  gs_ffull_s = gs[0,i0+1].subgridspec(1,2, width_ratios=[10,.5], wspace=.1)
  ax_ffull_s = gs_ffull_s.subplots()
  ax_ffull_s[0].set_title('simulation')

  gs_ffull_e = gs[0,i0+2].subgridspec(1,2, width_ratios=[10,.5], wspace=.1)
  ax_ffull_e = gs_ffull_e.subplots()
  ax_ffull_e[0].set_title('error')
  
  gs_Ifull_t = gs[1,i0].subgridspec(1,2, width_ratios=[10,.5], wspace=.1)  
  ax_Ifull_t = gs_Ifull_t.subplots()
  ax_Ifull_t[0].text(-1.1, .5, 'external\ndrive\n'+str_Iextfull, transform=ax_Ifull_t[0].transAxes, ha='left')

  gs_Ifull_s = gs[1,i0+1].subgridspec(1,2, width_ratios=[10,.5], wspace=.1)
  ax_Ifull_s = gs_Ifull_s.subplots()
  # ax_Ifull_s[0].set_title(r'$I_\mathrm{full}^\mathrm{sim}$')
  
  gs_Ifull_e = gs[1,i0+2].subgridspec(1,2, width_ratios=[10,.5], wspace=.1)
  ax_Ifull_e = gs_Ifull_e.subplots()
  # ax_Ifull_e[0].set_title(r'$I_\mathrm{full}^\mathrm{T}-I_\mathrm{full}^\mathrm{sim}$')

  # --- plot
  if p_var[0] == 'Delta':
    ylabel = r'$\Delta$'
  else:
    ylabel = p_var[1]
  ffull_t = df_net.pivot(index=p_var[1], columns=p_var[0], values='fnet_full_theory').astype(float)
  df_heatmap(ffull_t, fig, ax_ffull_t[0], ax_ffull_t[1], xtick_rotation='vertical', cbar_orientation= 'vertical', cmap = plt.cm.Greys, \
             ylabel=ylabel, cbar_label='frequency [Hz]') #'$f_F^\mathrm{th}$ [Hz]'
    
  ffull_s = df_net.pivot(index=p_var[1], columns=p_var[0], values='fnet_full_sim').astype(float)
  df_heatmap(ffull_s, fig, ax_ffull_s[0], ax_ffull_s[1], xtick_rotation='vertical', cbar_orientation= 'vertical', cmap = plt.cm.Greys, \
             cbar_label='frequency [Hz]') #'$f_F^\mathrm{sim}$ [Hz]'
  
  ffull_e = (ffull_t - ffull_s)/ffull_s*100
  vmin = ffull_e.min().min()
  vmax = ffull_e.max().max()
  if vmin > 0:
    vmin = -1e-3
  if vmax < 0 :
    vmax = 1e-3
  norm = matplotlib.colors.TwoSlopeNorm(0, vmin=vmin, vmax=vmax)
  df_heatmap(ffull_e, fig, ax_ffull_e[0], ax_ffull_e[1], xtick_rotation='vertical', cbar_orientation= 'vertical', cmap = plt.cm.coolwarm, \
             norm=norm, cbar_label='relative error [%]') #'$(f_F^\mathrm{th}-f_F^\mathrm{sim}) / f_F^\mathrm{sim}$ [%]'
  
  Ifull_t = df_net.pivot(index=p_var[1], columns=p_var[0], values='Ifull').astype(float)
  df_heatmap(Ifull_t, fig, ax_Ifull_t[0], ax_Ifull_t[1], xtick_rotation='vertical', cbar_orientation= 'vertical', cmap = plt.cm.Greens, \
             xlabel=p_var[0], ylabel=ylabel, cbar_label='voltage [-]')#'$I_F^\mathrm{th}$'
    
  Ifull_s = df_net.pivot(index=p_var[1], columns=p_var[0], values='Ifull_sim').astype(float)
  df_heatmap(Ifull_s, fig, ax_Ifull_s[0], ax_Ifull_s[1], xtick_rotation='vertical', cbar_orientation= 'vertical', cmap = plt.cm.Greens, \
             xlabel=p_var[0], cbar_label='voltage [-]')  #'$I_F^\mathrm{sim}$'
  
  Ifull_e = (Ifull_t - Ifull_s)/Ifull_s*100
  vmin = Ifull_e.min().min()
  vmax = Ifull_e.max().max()
  if vmin > 0:
    vmin = -1e-3
  if vmax < 0 :
    vmax = 1e-3
  norm = matplotlib.colors.TwoSlopeNorm(0, vmin=vmin, vmax=vmax)
#  norm = matplotlib.colors.TwoSlopeNorm(0, vmin=Ifull_e.min().min(), vmax=Ifull_e.max().max())
  df_heatmap(Ifull_e, fig, ax_Ifull_e[0], ax_Ifull_e[1], xtick_rotation='vertical', cbar_orientation= 'vertical', cmap = plt.cm.coolwarm, \
             norm=norm, xlabel=p_var[0], cbar_label='relative error [%]') #'$(I_F^\mathrm{th}-I_F^\mathrm{sim}) / I_F^\mathrm{sim}$ [%]'
    
  # set labels from D to sqrt(D) if necessary
  for ax in [ax_Ifull_t[0], ax_Ifull_s[0], ax_Ifull_e[0]]:
    if p_var[0] == 'D':
      ax.set_xticklabels(np.sqrt(ffull_t.columns), rotation='vertical');
      ax.set_xlabel(r'$\sqrt{D}$')
    elif p_var[0] == 'Delta':
      ax.set_xlabel(r'$\Delta$')
  for ax in [ax_ffull_s[0], ax_ffull_e[0]]:
    ax.set_xticklabels([])
    ax.set_yticklabels([])
  for ax in [ax_Ifull_s[0], ax_Ifull_e[0]]:
    ax.set_yticklabels([])  
  ax_ffull_t[0].set_xticklabels([])
  
  fig.savefig(path_to_figures+'FigS1B.pdf', bbox_inches='tight')
  fig.savefig(path_to_figures+'FigS1B.tif', bbox_inches='tight')
  
  return fig

def plot_visualize_D(D_val, displace=1.5, n = 1000, ax=[], pad=1e-2):
  if not ax:
    external_fig = False
    fig, ax = plt.subplots()
  else:
    external_fig = True
  gridline([-1,0,1], ax, axis='y')
  gauss = np.zeros((D_val.size, n))
  for i in range(D_val.size):
    v = np.linspace(-3*np.sqrt(D_val[i]), 3*np.sqrt(D_val[i]), n)
    gauss[i,:] = get_gauss(v, 0, np.sqrt(D_val[i]), broadcast=False)
    ax.plot(-gauss[i,:]+i*displace, v, 'k')
    ax.annotate(r'$\sqrt{D}=$'+'{:.2f}'.format(np.sqrt(D_val[i])), (-gauss[i,0]+i*displace, v[0]-i*pad), \
                horizontalalignment='right', verticalalignment='top', fontsize=matplotlib.rcParams['ytick.labelsize'])
  ax.set_ylabel('voltage')
  ax.set_yticks([-1, 0,1])
  ax.set_yticklabels(['-1', '$E_L=0$','$V_T=1$'])
  
  # Hide the right and top spines
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)
  ax.spines['bottom'].set_visible(False)
  ax.tick_params(
      axis='x',          # changes apply to the x-axis
      which='both',      # both major and minor ticks are affected
      bottom=False,      # ticks along the bottom edge are off
      top=False,         # ticks along the top edge are off
      labelbottom=False)
  if external_fig:
    return ax
  else:
    fig.tight_layout()
    return fig

def plot_performance_idx_manuscript(df_net, p_var, ax=[], cbar_labelsize=8):
  '''
  plot in 1 row: error in network freq, applicability, performance index
  ax: contains 3 axes of length 2 (for each of 3 plots: plot axis and colorbar axis)  
  '''
  if not ax:
    raise ValueError('provide external axis to put plot in')
  #NMAE
  # --- network frequency ---------------------------------------
  cbar_max = df_net['error_net_nmae'].max()*100
  norm = matplotlib.colors.Normalize(vmin=0, vmax=cbar_max)
  df = df_net.pivot(index=p_var[1], columns=p_var[0], values='error_net_nmae').astype(float)*100
  df_heatmap(df, None, ax[0][0], ax[0][1], ylabel=p_var[1], cmap=plt.cm.Greys, \
             cbar_orientation= 'vertical', norm=norm, xtick_rotation='vertical', cbar_label='(%)',\
             cbar_labelsize=cbar_labelsize)
  ax[0][0].set_title('error in network freq.')
  
  # --- applicability ---------------------------------------
  norm = matplotlib.colors.Normalize(vmin=0, vmax=100)
  df = df_net.pivot(index=p_var[1], columns=p_var[0], values='applicability').astype(float)*100
  df_heatmap(df, None, ax[1][0], ax[1][1], cmap=plt.cm.Greens, \
             cbar_orientation= 'vertical', norm=norm, xtick_rotation='vertical', cbar_label='(%)', \
             cbar_labelsize=cbar_labelsize, cbar_labelpad=-.8)
  ax[1][0].set_title('applicability')  
  plt.setp(ax[1][0].get_yticklabels(), visible=False)
  
  # --- performance ---------------------------------------
  norm = matplotlib.colors.Normalize(vmin=0, vmax=100)
  df = df_net.pivot(index=p_var[1], columns=p_var[0], values='performance').astype(float)*100
  df_heatmap(df, None, ax[2][0], ax[2][1], \
             cbar_orientation= 'vertical', norm=norm, xtick_rotation='vertical', cbar_label='(%)', \
             cbar_labelsize=cbar_labelsize,cbar_labelpad=-1)
  ax[2][0].set_title('performance')
  plt.setp(ax[2][0].get_yticklabels(), visible=False)
  
  
  
  
  # --- formatting
  ax[0][0].set_ylabel(p_var[1])
  for i in range(3):
    if p_var[0] == 'D':
      ax[i][0].set_xticklabels(np.sqrt(df.columns), rotation='vertical');
      ax[i][0].set_xlabel(r'$\sqrt{D}$')
    elif p_var[0] == 'Delta':
      ax[i][0].set_xticklabels(df.columns, rotation='vertical');
      ax[i][0].set_xlabel(r'$\Delta$')
    else:
      ax[i][0].set_xticklabels(df.columns, rotation='vertical');
      ax[i][0].set_xlabel(p_var[0])
  if p_var[1] == 'Delta':
    ax[0][0].set_ylabel(r'$\Delta$')
  return ax

def plot_performance_details_gaussdynamics(df_runs, df_net, theory, Vt=1, reset=False,\
                                           show_gauss_numerical=False, external_ax=[]):
  exploration, p_fix, p_var = recover_exploration(df_runs)   
  n_config = int(df_runs.config.max() + 1)
  
  # plotting parameters
  # dmu = .1
  xpad = .3 if p_var[0]=='Delta' else .05
  ypad = .15
  width, height = 1-xpad, 1-ypad
  ms=1
  
  x_val = exploration[p_var[0]]
  if len(p_var) == 1:
    raise ValueError('Function must be extended for 1D case!')
  elif len(p_var) == 2:
    y_val = exploration[p_var[1]]
    
  if p_var[0]=='D':
    x_val = np.sqrt(x_val)
  elif p_var[1]=='D':
    raise ValueError('If D was explored it should appear as the first explored parameter!')
    
  # --- construct figure
  if external_ax:
    ax = external_ax
  else:
    fig_width = width_a4_wmargin
    fig_height = width_a4_wmargin*.52
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=200)
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)
  ax.set_xlim([-.1, len(x_val)+width/2+xpad])
  ax.set_ylim([-.4, len(y_val)+1])
  
  ax.set_xticks(np.arange(len(x_val))+1)
  ax.set_xticklabels(['{:.2f}'.format(x) for x in x_val])
  if p_var[0] == 'D':
    ax.set_xlabel('noise intensity ' + r'$\sqrt{D}$')
  elif p_var[0] == 'Delta':
    ax.set_xlabel('synaptic delay ' + r'$\Delta$')
  else:
    ax.set_xlabel(p_var[0])
  
  ax.set_yticks(np.arange(len(y_val))+1)
  ax.set_yticklabels(['{:.1f}'.format(y) for y in y_val])
  if p_var[1]=='K': # paper
    ax.set_ylabel('coupling strength ' + p_var[1])
  elif p_var[1] == 'Delta':
    ax.set_ylabel('synaptic delay ' + r'$\Delta$')
  else:
    ax.set_ylabel(p_var[1])
  
  ax_sub = [[None]*len(y_val)]*len(x_val)
  
  for c in range(n_config):
#    print(c)
    # --- extract relevant runs
    df_temp = df_runs[df_runs.config==c]
    
    # --- deduce plot position and create axis
    x = df_net.loc[c,p_var[0]]
    y = df_net.loc[c,p_var[1]]
    if p_var[0] == 'D':
      x = np.sqrt(df_net.loc[c,p_var[0]])
    ix = np.argwhere(x_val == x).item()
    iy = np.argwhere(y_val == y).item()

    x0, y0 = 1+ix-width/2, 1+iy-height/2
    
    if c==0:
      ax_root = ax.inset_axes([x0, y0, width, height], transform=ax.transData)
      ax_sub[ix][iy] = ax_root
      ax_root.set_xlim([0, .1+np.ceil(df_net['Ifull_sim'].max())])
#      ax_root.set_ylim([0, np.ceil(df_runs.loc[df_runs['Iext'] <= df_net['Ifull_sim'].max(), 'gauss_f_net'].max())])
    else:
      if ix:
        ax_sub[ix][iy] = ax.inset_axes([x0, y0, width, height], transform=ax.transData, sharex=ax_root, sharey=ax_sub[0][iy])
      else:
        ax_sub[ix][iy] = ax.inset_axes([x0, y0, width, height], transform=ax.transData, sharex=ax_root)
    legend = ix==iy==0

    # --- fill axis
    gridline([0,Vt], ax_sub[ix][iy], 'y')
    theory_applies = (theory[c]['Iext'] >= df_net.loc[c,'Imin_theory']) & (theory[c]['Iext'] <= df_net.loc[c,'Imax_theory'])
    ax_sub[ix][iy].plot(theory[c]['Iext'], theory[c]['mu_min'], color=my_colors['min'], lw=.5, linestyle='--')
    ax_sub[ix][iy].plot(theory[c]['Iext'], theory[c]['mu_max'], color=my_colors['max'], lw=.5, linestyle='--')
    # add mumin as estimated in spiking network
    if 'gauss_mu_min_sim' in df_temp.columns:
      df_temp.plot(ax=ax_sub[ix][iy], x='Iext', y='gauss_mu_min_sim', color=my_colors['min'], marker='o', linestyle='', \
                   markersize=ms, label=r'$\mu_\mathrm{min}$ (sim)', legend=legend)
    if reset:
      ax_sub[ix][iy].plot(theory[c]['Iext'], theory[c]['mu_reset'], color=my_colors['reset'], lw=.5, linestyle='--')
    if theory_applies.any():
      ax_sub[ix][iy].plot(theory[c]['Iext'][theory_applies], theory[c]['mu_min'][theory_applies], color=my_colors['min'], label=r'$\mu_\mathrm{min}$')
      ax_sub[ix][iy].plot(theory[c]['Iext'][theory_applies], theory[c]['mu_max'][theory_applies], color=my_colors['max'], label=r'$\mu_\mathrm{max}$')
      if reset:
        ax_sub[ix][iy].plot(theory[c]['Iext'][theory_applies], theory[c]['mu_reset'][theory_applies], color=my_colors['reset'], label=r'$\mu_\mathrm{reset}$')
    if show_gauss_numerical:
      df_temp.plot(ax=ax_sub[ix][iy], x='Iext', y='gauss_mu_min_num', color='m',  linestyle=':', \
                   label=r'$\mu_\mathrm{min}$ (num)', legend=legend)
      df_temp.plot(ax=ax_sub[ix][iy], x='Iext', y='gauss_mu_max_num', color='m',  linestyle='--',  \
                   label=r'$\mu_\mathrm{max}$ (num)', legend=legend)
      if reset:
        df_temp.plot(ax=ax_sub[ix][iy], x='Iext', y='gauss_mu_reset_num', color=my_colors['reset'], marker='.', linestyle=':',  label=r'$\mu_\mathrm{reset}$ (num)', legend=legend)
    # --- formatting
    if c:
      ax_sub[ix][iy].set_xlabel('')
      plt.setp(ax_sub[ix][iy].get_xticklabels(), visible=False)
      if ix:
        plt.setp(ax_sub[ix][iy].get_yticklabels(), visible=False)
      
  ax_root.legend(loc='lower left', bbox_to_anchor=(1, len(y_val)+height/2+ypad), bbox_transform=ax.transData, \
                 ncol= 3 + 1*('v_av_mumin' in df_temp.columns), \
                 handlelength=1)# borderaxespad=0.) #1-width/2
  ax_root.set_xlabel('drive ' +str_Iext)#, labelpad=0.1)
  ax_root.set_ylabel('$\mu$',  rotation=0) #labelpad=0.3,
  if external_ax:
    return ax
  else:
    fig.suptitle(get_pfix_str(exploration))
    return fig

def plot_performance_details_frequencies(df_runs, df_net, theory, show_gauss_numerical=False, external_ax=[]):
  
  exploration, p_fix, p_var = recover_exploration(df_runs)   
  n_config = int(df_runs.config.max() + 1)
  
  # plotting parameters
  xpad = .3 if 'Delta' in p_var else .05
  ypad = .15 # 0.1
  width, height = 1-xpad, 1-ypad
  ms=1
  sharey = False if 'Delta' in p_var else True  
  
  x_val = exploration[p_var[0]]
  if len(p_var) == 1:
    raise ValueError('Function must be extended for 1D case!')
  elif len(p_var) == 2:
    y_val = exploration[p_var[1]]
    
  if p_var[0]=='D':
    x_val = np.sqrt(x_val)
  elif p_var[1]=='D':
    raise ValueError('If D was explored it should appear as the first explored parameter!')
    
  # --- construct figure  
  if external_ax:
    ax = external_ax
  else:
    fig_width = width_a4_wmargin
    fig_height = width_a4_wmargin*.52
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=200)
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)
  ax.set_xlim([-.1, len(x_val)+width/2+xpad])
  ax.set_ylim([-.4, len(y_val)+1])
  
  ax.set_xticks(np.arange(len(x_val))+1)
  ax.set_xticklabels(['{:.2f}'.format(x) for x in x_val])
  if p_var[0] == 'D':
    ax.set_xlabel('noise intensity ' + r'$\sqrt{D}$')
  elif p_var[0] == 'Delta':
    ax.set_xlabel('synaptic delay ' + r'$\Delta$')
  else:
    ax.set_xlabel(p_var[0])
  
  ax.set_yticks(np.arange(len(y_val))+1)
  ax.set_yticklabels(['{:.1f}'.format(y) for y in y_val])
  if p_var[1]=='K': # paper
    ax.set_ylabel('coupling strength ' + p_var[1])
  elif p_var[1] == 'Delta':
    ax.set_ylabel('synaptic delay ' + r'$\Delta$')
  else:
    ax.set_ylabel(p_var[1])
  
  ax_sub = [[None]*len(y_val)]*len(x_val)
  
  fmax = np.zeros(n_config)
  for c in range(n_config):
#    print(c)
    # --- extract relevant runs
    df_temp = df_runs[df_runs.config==c]
    
    # --- deduce plot position and create axis
    x = df_net.loc[c,p_var[0]]
    y = df_net.loc[c,p_var[1]]
    if p_var[0] == 'D':
      x = np.sqrt(df_net.loc[c,p_var[0]])
    ix = np.argwhere(x_val == x).item()
    iy = np.argwhere(y_val == y).item()

    x0, y0 = 1+ix-width/2, 1+iy-height/2
    
    if c==0:
      ax_root = ax.inset_axes([x0, y0, width, height], transform=ax.transData)
      ax_sub[ix][iy] = ax_root
      ax_root.set_xlim([0, .1+np.ceil(df_net['Ifull_sim'].max())])
#      ax_root.set_ylim([0, np.ceil(df_runs.loc[df_runs['Iext'] <= df_net['Ifull_sim'].max(), 'gauss_f_net'].max())])
    else:
      if sharey==True:
        ax_sub[ix][iy] = ax.inset_axes([x0, y0, width, height], transform=ax.transData, sharex=ax_root, sharey=ax_root)
      else:
        ax_sub[ix][iy] = ax.inset_axes([x0, y0, width, height], transform=ax.transData, sharex=ax_root) #, sharey=ax_sub[ix][0])
        if 'Delta' == p_var[0]:
          freq_net_delta_max = df_runs[df_runs[p_var[0]] == exploration[p_var[0]][ix]].freq_net.max() # max in this columns
        else:
          freq_net_delta_max = df_runs[df_runs[p_var[1]] == exploration[p_var[1]][iy]].freq_net.max() # max in this columns
        ax_sub[ix][iy].set_ylim([0, np.max([250, 10+freq_net_delta_max])])
    legend = ix==iy==0

    # --- fill axis
    # show only results up to Imax
    Imax = df_net.loc[c,['Ifull','Ifull_sim']].max()
    theory_applies = (theory[c]['Iext'] >= df_net.loc[c,'Imin_theory']) & (theory[c]['Iext'] <= df_net.loc[c,'Imax_theory'])
    ax_sub[ix][iy].axhspan(140, 220, facecolor='lightgray')    

    df_temp[df_temp.Iext <= Imax].plot(ax=ax_sub[ix][iy], x='Iext', y='freq_net', color=my_colors['fnet'], marker='^', linestyle='',  \
                                       label='net (sim)', legend=legend, markersize=ms, zorder=3)
    df_temp[df_temp.Iext <= Imax].plot(ax=ax_sub[ix][iy], x='Iext', y='freq_unit_mean', color=my_colors['funit'], marker='o', linestyle='', \
                                       label='unit (sim)', legend=legend, markersize=ms, zorder=2)
    if theory_applies.any():
      ax_sub[ix][iy].plot(theory[c]['Iext'][theory_applies], theory[c]['fnet'][theory_applies], color=my_colors['fnet'], label='net (analyt)', zorder=3, lw=1)
      ax_sub[ix][iy].plot(theory[c]['Iext'][theory_applies], theory[c]['funit'][theory_applies], color=my_colors['funit'], label='unit (analyt)', zorder=2, lw=1)
      fmax[c] = np.nanmax(theory[c]['fnet'][theory_applies])
    if show_gauss_numerical:
      df_temp.plot(ax=ax_sub[ix][iy], x='Iext', y='gauss_f_unit_num', color='m',  linestyle=':', label='unit (num)', legend=legend, zorder=1e4)
      df_temp.plot(ax=ax_sub[ix][iy], x='Iext', y='gauss_f_net_num', color='m',  linestyle='--', label='net (num)', legend=legend, zorder=1e4)
    ax_sub[ix][iy].plot(df_net.loc[c,'Ifull_sim'], df_net.loc[c,'fnet_full_sim'], marker='^', linestyle='', markerfacecolor='none', \
                        markersize= ms+2, markeredgecolor='deepskyblue', label=r'$f_\mathrm{net/unit}^\mathrm{full}$ (sim)', zorder=3, markeredgewidth=.5)# 
    ax_sub[ix][iy].plot(df_net.loc[c,'Icrit'], df_net.loc[c,'fcrit'], marker='^', linestyle='', fillstyle='none', markeredgecolor='r', \
                        markersize= ms+2, label=r'$f_\mathrm{net/unit}^\mathrm{crit}$ (analyt)', zorder=3, markeredgewidth=.5)
    ax_sub[ix][iy].plot(df_net.loc[c,'Icrit'], df_net.loc[c,'A0crit'], marker='o', linestyle='', fillstyle='none', markeredgecolor='r', \
                        markersize= ms+2, zorder=3, markeredgewidth=.5) #label=r'$f_\mathrm{unit}^\mathrm{crit}$ (analyt)', 
    # --- formatting
    if c:
      ax_sub[ix][iy].set_xlabel('')
    # if not ix:
    #   ax_sub[ix][iy].set_yticks([0,100,200, 300])
    # else:
      if sharey==True:
        plt.setp(ax_sub[ix][iy].get_yticklabels(), visible=False)
    # if iy:
      plt.setp(ax_sub[ix][iy].get_xticklabels(), visible=False)
      
  # ax_root.legend(fontsize=6)  
  if 'Delta' not in p_var:
    ax_root.set_ylim([0, 1000/(2*exploration['Delta'])]) #np.ceil(np.max(fmax))])
    ax_root.set_yticks([0,100,200, 300])
 
  ax_root.legend(loc='lower left', bbox_to_anchor=(0, len(y_val)+height/2+ypad), bbox_transform=ax.transData, ncol=7, \
                 handlelength=1, columnspacing=1.3)# borderaxespad=0.) #1-width/2
  
  ax_root.set_xlabel('drive '+str_Iext)#, labelpad=0)
  ax_root.set_ylabel('freq [Hz]')#, labelpad=0)
  if external_ax:
    return ax
  else:
#    fig.suptitle(get_pfix_str(exploration))
    return fig, ax    


#%% Supplementary Fig S2C
def plot_figure_S2C(traj_hash_ABCD_stat = [1001, 1003, 1002, 1004], traj_hash_ABCD_trans = [1006, 1007, 1008, 1009], 
                    traj_hash_ABCD_squarepulse = [1010, 1011, 1012, 1013], 
                    path_to_figures = './figures/', path_to_simulations = './simulations/'):
  print('\n\nPlotting Fig S2C...')
  # extract simulation hashes:
  traj_hash_stat_A, traj_hash_stat_B, traj_hash_stat_C, traj_hash_stat_D = traj_hash_ABCD_stat # stationary sims for models A-D
  traj_hash_trans_A, traj_hash_trans_B, traj_hash_trans_C, traj_hash_trans_D = traj_hash_ABCD_trans # double ramp sims for models A-D
  traj_hash_sqp_A, traj_hash_sqp_B, traj_hash_sqp_C, traj_hash_sqp_D = traj_hash_ABCD_squarepulse # square pulse sims for models A-D
  
  fig_width= width_a4_wmargin
  fig_height = fig_width #*.8
  
  fig = plt.figure(figsize=(fig_width, fig_height))#, constrained_layout=True
  
  gs = gridspec.GridSpec(2,1, figure=fig, hspace=.4, height_ratios=[1,2.5*4/3])
  gs1 = gs[0].subgridspec(1,4, wspace=.3)
  gs2 = gs[1].subgridspec(4,4, wspace=.3, hspace=.3)
  
  ax_top = gs1.subplots(sharey=True)
  ax_bottom = gs2.subplots(sharey='row', sharex='col')
  
  despine(ax_top)
  despine(ax_bottom)
  
  ax_top[0].set_ylabel('frequency [Hz]')
  ax_bottom[-1,0].set_ylabel('frequency [Hz]')
  
  dx, dy = 0, 1.05
  for i in range(4):
    ax_top[i].text(dx, dy, string.ascii_lowercase[i]+'1', transform=ax_top[i].transAxes, weight='bold')
    ax_bottom[0,i].text(dx, dy, string.ascii_lowercase[i]+'2', transform=ax_bottom[0,i].transAxes, weight='bold')
    ax_bottom[1,i].text(dx, dy, string.ascii_lowercase[i]+'3', transform=ax_bottom[1,i].transAxes, weight='bold')
    ax_bottom[2,i].text(dx, dy, string.ascii_lowercase[i]+'4', transform=ax_bottom[2,i].transAxes, weight='bold')
    ax_bottom[3,i].text(dx, dy, string.ascii_lowercase[i]+'5', transform=ax_bottom[3,i].transAxes, weight='bold')
  
  # --- data ---------------------------------------------
  # --- Donoso2018 original network ---------------------------------------------------------
  ax_top[0], ax_bottom[:,0] = plot_freqs_stat_vs_trans(traj_stat_hash = traj_hash_stat_A, traj_trans_hash = traj_hash_trans_A,
                                                       traj_sqp_hash = traj_hash_sqp_A, ax_stat = ax_top[0], ax_trans = ax_bottom[:,0],
                                                       level_crit = 1500, xmax=16_000, path_to_simulations=path_to_simulations)
  
  
  # --- Donoso2018 network with independent Poisson inputs ---------------------------------------------------------
  ax_top[1], ax_bottom[:,1] = plot_freqs_stat_vs_trans(traj_stat_hash = traj_hash_stat_B, traj_trans_hash = traj_hash_trans_B, 
                                                       traj_sqp_hash = traj_hash_sqp_B, ax_stat = ax_top[1], ax_trans = ax_bottom[:,1], 
                                                       level_crit = 1500, xmax=16_000, path_to_simulations=path_to_simulations) # try 1005 & 10051 for a larger network 
  
  # --- Reduced network with refractory period ---------------------------------------------------------
  ax_top[2], ax_bottom[:,2] = plot_freqs_stat_vs_trans(traj_stat_hash = traj_hash_stat_C, traj_trans_hash = traj_hash_trans_C, 
                                                       traj_sqp_hash = traj_hash_sqp_C, ax_stat = ax_top[2], ax_trans = ax_bottom[:,2], 
                                                       xmax=2.2, path_to_simulations=path_to_simulations)
  
  # --- Reduced network as in manuscript -----------------------------------------------------------------------
  ax_top[3], ax_bottom[:,3] = plot_freqs_stat_vs_trans(traj_stat_hash = traj_hash_stat_D, traj_trans_hash = traj_hash_trans_D, 
                                                       traj_sqp_hash = traj_hash_sqp_D, ax_stat = ax_top[3], ax_trans = ax_bottom[:,3], 
                                                       xmax=1.3, path_to_simulations=path_to_simulations, show_second_yaxis=True, label_finst=True)

  ax_top[-1].legend(bbox_to_anchor=(1.2, 1.2), handlelength=1, edgecolor='dimgrey', loc='upper right', borderaxespad=0., fontsize = 6, facecolor='white')
  
  fig.savefig(path_to_figures + 'FigS2C.pdf', bbox_inches='tight')
  fig.savefig(path_to_figures + 'FigS2C.tif', bbox_inches='tight')
  return

def plot_figure_S2C_ramponly(traj_hash_ABCD_stat = [1001, 1003, 1002, 1004], traj_hash_ABCD_trans = [1006, 1007, 1008, 1009], 
                             path_to_figures = './figures/', path_to_simulations = './simulations/'):
  print('\n\nPlotting Fig S2C...')
  # extract simulation hashes:
  traj_hash_stat_A, traj_hash_stat_B, traj_hash_stat_C, traj_hash_stat_D = traj_hash_ABCD_stat # stationary sims for models A-D
  traj_hash_trans_A, traj_hash_trans_B, traj_hash_trans_C, traj_hash_trans_D = traj_hash_ABCD_trans # double ramp sims for models A-D
  
  fig_width= width_a4_wmargin
  fig_height = fig_width*.8
  
  fig = plt.figure(figsize=(fig_width, fig_height))#, constrained_layout=True
  
  gs = gridspec.GridSpec(2,1, figure=fig, hspace=.4, height_ratios=[1,2.5])
  gs1 = gs[0].subgridspec(1,4, wspace=.3)
  gs2 = gs[1].subgridspec(3,4, wspace=.3, hspace=.3)
  
  ax_top = gs1.subplots(sharey=True)
  ax_bottom = gs2.subplots(sharey='row', sharex='col')
  
  despine(ax_top)
  despine(ax_bottom)
  
  ax_top[0].set_ylabel('frequency [Hz]')
  ax_bottom[-1,0].set_ylabel('frequency [Hz]')
  
  dx, dy = 0, 1.05
  for i in range(4):
    ax_top[i].text(dx, dy, string.ascii_lowercase[i]+'1', transform=ax_top[i].transAxes, weight='bold')
    ax_bottom[0,i].text(dx, dy, string.ascii_lowercase[i]+'2', transform=ax_bottom[0,i].transAxes, weight='bold')
    ax_bottom[1,i].text(dx, dy, string.ascii_lowercase[i]+'3', transform=ax_bottom[1,i].transAxes, weight='bold')
    ax_bottom[2,i].text(dx, dy, string.ascii_lowercase[i]+'4', transform=ax_bottom[2,i].transAxes, weight='bold')

  
  # --- data ---------------------------------------------
  # --- Donoso2018 original network ---------------------------------------------------------
  ax_top[0], ax_bottom[:,0] = plot_freqs_stat_vs_trans_ramponly(traj_stat_hash = traj_hash_stat_A, traj_trans_hash = traj_hash_trans_A, ax_stat = ax_top[0], ax_trans = ax_bottom[:,0],\
                                                                level_crit = 1500, xmax=16_000, path_to_simulations=path_to_simulations)
  
  
  # --- Donoso2018 network with independent Poisson inputs ---------------------------------------------------------
  ax_top[1], ax_bottom[:,1] = plot_freqs_stat_vs_trans_ramponly(traj_stat_hash = traj_hash_stat_B, traj_trans_hash = traj_hash_trans_B, ax_stat = ax_top[1], ax_trans = ax_bottom[:,1],\
                                                     level_crit = 1500, xmax=16_000, path_to_simulations=path_to_simulations) # try 1005 & 10051 for a larger network 
  
  # --- Reduced network with refractory period ---------------------------------------------------------
  ax_top[2], ax_bottom[:,2] = plot_freqs_stat_vs_trans_ramponly(traj_stat_hash = traj_hash_stat_C, traj_trans_hash = traj_hash_trans_C, ax_stat = ax_top[2], ax_trans = ax_bottom[:,2], xmax=2.2, path_to_simulations=path_to_simulations)
  
  # --- Reduced network as in manuscript -----------------------------------------------------------------------
  ax_top[3], ax_bottom[:,3] = plot_freqs_stat_vs_trans_ramponly(traj_stat_hash = traj_hash_stat_D, traj_trans_hash = traj_hash_trans_D, ax_stat = ax_top[3], ax_trans = ax_bottom[:,3], xmax=1.3, path_to_simulations=path_to_simulations, show_second_yaxis=True, label_finst=True)

  ax_top[-1].legend(bbox_to_anchor=(1.2, 1.2), handlelength=1, edgecolor='dimgrey', loc='upper right', borderaxespad=0., fontsize = 6, facecolor='white')
  
  fig.savefig(path_to_figures + 'FigS2C.pdf', bbox_inches='tight')
  fig.savefig(path_to_figures + 'FigS2C.tif', bbox_inches='tight')
  return

def plot_freqs_stat_vs_trans(traj_stat_hash, traj_trans_hash, traj_sqp_hash, ax_stat, ax_trans, 
                             level_crit = None, xmax=100, ylim=[0,400], tpad=10, fmin=70, 
                             show_second_yaxis=False, path_to_simulations='./simulations/', label_finst=False):
  # load stationary data 
  traj_stat = pypet_load_trajectory(traj_hash = traj_stat_hash, path_to_simulations=path_to_simulations)
  res_stat = traj_stat.results.summary.scalarResults  
  res_stat['level'] = res_stat['parameters.input.level'].copy()
  if not level_crit:
    level_crit = traj_stat.linear_stability.Icrit_nA - 1e-3 # subtract epsilon to avoid numerical errors in comparison with critical level
  
  # --- plot stationary frequencies over drive
  ax_stat.fill_between([0,xmax*1.1], 140, 220, facecolor='lightgray', edgecolor='face', zorder=0)  
  ax_stat.plot(res_stat[res_stat.level>=level_crit]['level'], res_stat[res_stat.level>=level_crit]['freq_net'], marker='^', color=my_colors['fnet'], zorder=4, label=r'$f_\mathrm{net}^\mathrm{\infty}$')
  # res_stat.plot(ax=ax_stat, x= 'level', y='freq_net', marker='^', color=my_colors['fnet'], zorder=3, legend=False, label=r'$f_\mathrm{net}^\mathrm{\infty}$')
  res_stat.plot(ax=ax_stat, x= 'level', y='freq_unit_mean', marker='o', color=my_colors['funit'], zorder=3, legend=False, label=r'$f_\mathrm{unit}^\mathrm{\infty}$')  #yerr = 'freq_unit_std', 
  ax_stat.set_xlabel('drive [{}]'.format(traj_stat.input.unit.replace('sec','s')))
  ax_stat.set_xlim([0,xmax])
  ax_stat.set_ylim(ylim)
  
  # --- load transient drive data (double ramp) ----------------------------
  traj_trans = pypet_load_trajectory(traj_hash = traj_trans_hash, path_to_simulations=path_to_simulations) # double-ramps
  
  # load results from IFA analysis
  ifa_stats = pd.read_csv(pypet_get_trajectoryPath(traj_hash = traj_trans_hash, path_to_simulations=path_to_simulations) + 'analysis_IFA_fmin-{}/data_ifa_h{}_summary.csv'.format(fmin, traj_trans_hash), index_col=0, squeeze=True)
  ax_trans2 = list(np.zeros(len(ifa_stats) +1))
  
  for i in range(len(ifa_stats)):      
      # --- find run indices of interest (the simulations belonging to the network configuration specified by "exploration"-----------------------------------------------
      run_idx = pypet_find_runs(traj_trans, 'ramp_time_up', lambda x: x==ifa_stats.loc[i]['ramp_time_up'])             
      # nreps = len(run_idx) # number of simulation repetitions done for this network configuration
      
      # load correct runs:
      ifreq_t = pypet_get_from_runs(traj_trans, 'network.ifreq_discr_t',run_idx=run_idx) # dict 
      ifreq = pypet_get_from_runs(traj_trans, 'network.ifreq_discr',run_idx=run_idx) # dict
      traj_trans.v_idx = run_idx[0]
      stimulus = traj_trans.derived_parameters.runs[traj_trans.v_crun]['stim_plot']

      # important time stamps
      tw = traj_trans.analysis.ifreq_targetwindow
      t0 = np.mean(tw) # middle of ramp stimulation
      t_stim_on = tw[0] + (np.diff(tw) - (traj_trans.ramp_time_up + traj_trans.plateau_time + traj_trans.ramp_time_down)) / 2 # reinferring Tedge here
      t_stim_off = t_stim_on + traj_trans.ramp_time_up + traj_trans.plateau_time + traj_trans.ramp_time_down # end of ramp
      # t_mid_plateau = t_stim_on + traj_trans.ramp_time_up + traj_trans.plateau_time/2 # middle of plateau
      # color markers depending on TIME of instfreq
      norm = matplotlib.colors.TwoSlopeNorm(t0, vmin=t_stim_on, vmax=t_stim_off)
      
      # restrict ifreq analysis to time points DURING RAMP:
      ifreq_t_all = list_from_dict(ifreq_t, output_numpy=True) 
      ifreq_all  = list_from_dict(ifreq, output_numpy=True)    
      ix_keep = (ifreq_t_all >= t_stim_on) & (ifreq_t_all <= t_stim_off)
      ifreq_t_all = ifreq_t_all[ix_keep]
      ifreq_all = ifreq_all[ix_keep]
      
      # current at time of ifreq measurement
      Iext_trans = stimulus[(ifreq_t_all/traj_trans.dt).astype(int)]  
  
      # --- plot instantanous frequencies over drive in ax_stat: ONLY FOR THE SYMMETRIC RAMP!
      if traj_trans.ramp_time_up == traj_trans.ramp_time_down:
        color = plt.cm.coolwarm(1-norm(ifreq_t_all))[:,:3]
        ax_stat.scatter(Iext_trans, ifreq_all, c = color, marker='.', s=2.5, zorder=2, label=r'$f_\mathrm{net}^\mathrm{inst}$')
        # mark point of full synch in stat plot
        gridline(traj_trans.input.peak, ax_stat, axis='x', zorder=1)
      else:
        color = 'grey'
        
      # --- plot instantanous frequencies over time
      ax_trans[i].scatter(ifreq_t_all-t0, ifreq_all, marker='.', s=2.5, c = color, zorder=2) #, label=r'$f_\mathrm{net}^\mathrm{inst}$'*label_finst)
  
      # --- plot stationary frequencies over time
      # interpolate stationary frequencies from stat simulation result
      fnet_stat_interp = np.interp(stimulus, res_stat['parameters.input.level'].values, res_stat['freq_net'])
      fnet_stat_interp[stimulus < level_crit] = nan
      t = np.arange(stimulus.size)*traj_trans.dt - t0
      ax_trans[i].plot(t, fnet_stat_interp, 'k', zorder=3)
  
      # limit yaxis
      ax_trans[i].set_ylim(ylim)
      ax_trans[i].set_xlim([t_stim_on-tpad-t0, t_stim_off+tpad-t0])
  
      # mark transient stimulus (normalized)
      ax_trans2[i] = ax_trans[i].twinx()
      time = np.arange(len(stimulus)) * traj_trans.dt
      ax_trans2[i].plot(time-t0, stimulus/np.max(stimulus), color='gray')#, label='drive')
      ax_trans2[i].set_ylim([0,4.5])
      Ifull_nA = np.max(stimulus) # save max stim amplitude to correctly scale square pulse input later 
      
      # if show_second_yaxis and (i==len(ifa_stats)-1):
      #   ax_trans2[i].set_yticks([0,1])
      #   ax_trans2[i].tick_params(axis='y', labelcolor='gray', color='gray')
      #   ax_trans2[i].spines['right'].set_color('gray')
      #   ax_trans2[i].set_ylabel('drive [normalized]', color='gray')
      #   ax_trans2[i].spines['top'].set_visible(False)
      # else:
      despine(ax_trans2[i])
      ax_trans2[i].set_yticks([])    
        
      # add linear regression line 
      ifa_slope = ifa_stats.loc[i]['ifa_slope']
      ifa_intercept = ifa_stats.loc[i]['ifa_intercept']
      time = np.arange(t_stim_on, t_stim_off+traj_trans.dt, traj_trans.dt) -t0
      ax_trans[i].plot(time, ifa_slope*time + ifa_intercept, color='grey', linestyle='-', lw=.5, label=r'$\chi_\mathrm{IFA}$='+'{:.1f} Hz/ms'.format(ifa_slope))  
      ax_trans[i].text(1,1, r'$\chi_\mathrm{IFA}$='+'{:.1f} Hz/ms'.format(ifa_slope), ha='right',  va='bottom', fontsize=6, transform=ax_trans[i].transAxes)
      # ax_trans[i].legend(bbox_to_anchor=(1,1), loc='lower right', handlelength=1, borderaxespad=0., fontsize=6)
  
  # ----- bottom row: add square pulse simulations -------------------------------------------------------------------------------------
  traj_sqp = pypet_load_trajectory(traj_hash = traj_sqp_hash, path_to_simulations=path_to_simulations) # square pulse data
  # --- plot instantanous frequencies over time
  ifreq_t = pypet_get_from_runs(traj_sqp, 'network.ifreq_discr_t') # dict 
  ifreq = pypet_get_from_runs(traj_sqp, 'network.ifreq_discr') # dict
  traj_sqp.v_idx = 0
  stimulus = traj_sqp.derived_parameters.runs[traj_sqp.v_crun]['stim_plot'] # extract stimulus
  
  # important time stamps
  tw = traj_sqp.analysis.ifreq_targetwindow
  t0 = np.mean(tw) # middle of ramp stimulation
  t_stim_on = tw[0] + (np.diff(tw) - (traj_sqp.plateau_time)) / 2 # reinferring Tedge here
  t_stim_off = t_stim_on + traj_sqp.plateau_time # end of ramp
  
  # restrict ifreq analysis to time points DURING square pulse:
  ifreq_t_all = list_from_dict(ifreq_t, output_numpy=True) 
  ifreq_all  = list_from_dict(ifreq, output_numpy=True)    
  ix_keep = (ifreq_t_all >= t_stim_on) & (ifreq_t_all <= t_stim_off)
  ifreq_t_all = ifreq_t_all[ix_keep]
  ifreq_all = ifreq_all[ix_keep]
  
  # current at time of ifreq measurement
  Iext_trans = stimulus[(ifreq_t_all/traj_sqp.dt).astype(int)]  
  
  # --- plot instantanous frequencies over time
  color = 'grey'
  ax_trans[-1].scatter(ifreq_t_all-t0, ifreq_all, marker='.', s=2.5, c = color, zorder=2) #, label=r'$f_\mathrm{net}^\mathrm{inst}$'*label_finst)
  
  # --- plot stationary frequencies over time
  # interpolate stationary frequencies from stat simulation result
  fnet_stat_interp = np.interp(stimulus, res_stat['parameters.input.level'].values, res_stat['freq_net'])
  fnet_stat_interp[stimulus < level_crit] = nan
  t = np.arange(stimulus.size)*traj_sqp.dt - t0
  ax_trans[-1].plot(t, fnet_stat_interp, 'k', zorder=3)
  
  # limit yaxis
  ax_trans[-1].set_ylim(ylim)
  ax_trans[-1].set_xlim([t_stim_on-tpad-t0, t_stim_off+tpad-t0])
  
  # mark transient stimulus (normalized)
  ax_trans2[-1] = ax_trans[-1].twinx()
  time = np.arange(len(stimulus)) * traj_sqp.dt
  ax_trans2[-1].plot(time-t0, stimulus/Ifull_nA, color='gray') # hack: hardcoded stimulus amplitude here!
  ax_trans2[-1].set_ylim([0,4.5])
  if show_second_yaxis:
    ax_trans2[-1].set_yticks([0,1])
    ax_trans2[-1].tick_params(axis='y', labelcolor='gray', color='gray')
    ax_trans2[-1].spines['right'].set_color('gray')
    ax_trans2[-1].set_ylabel('drive [normalized]', color='gray')
    ax_trans2[-1].spines['top'].set_visible(False)
  else:
    despine(ax_trans2[-1])
    ax_trans2[-1].set_yticks([])   
    
  # add linear regression line 
  with np.load(pypet_get_trajectoryPath(traj_hash = traj_sqp_hash, path_to_simulations=path_to_simulations) + 'analysis_IFA_fmin-{}/data_ifa_hash{}.npz'.format(fmin, traj_sqp_hash)) as data:
    ifa_slope = data['ifa_slope']
    ifa_intercept = data['ifa_intercept']
  time = np.arange(t_stim_on, t_stim_off+traj_sqp.dt, traj_sqp.dt) -t0
  ax_trans[-1].plot(time, ifa_slope*time + ifa_intercept, color='grey', linestyle='-', lw=.5, label=r'$\chi_\mathrm{IFA}$='+'{:.1f} Hz/ms'.format(ifa_slope))  
  ax_trans[-1].text(1,1, r'$\chi_\mathrm{IFA}$='+'{:.1f} Hz/ms'.format(ifa_slope), ha='right',  va='bottom', fontsize=6, transform=ax_trans[-1].transAxes)
    
  ax_trans[-1].set_xlabel('time [ms]')
  return ax_stat, ax_trans

def plot_freqs_stat_vs_trans_ramponly(traj_stat_hash, traj_trans_hash, ax_stat, ax_trans, level_crit = None, xmax=100, ylim=[0,400], tpad=10, fmin=70, show_second_yaxis=False, path_to_simulations='./simulations/', label_finst=False):
  # load stationary data 
  traj_stat = pypet_load_trajectory(traj_hash = traj_stat_hash, path_to_simulations=path_to_simulations)
  res_stat = traj_stat.results.summary.scalarResults  
  res_stat['level'] = res_stat['parameters.input.level'].copy()
  if not level_crit:
    level_crit = traj_stat.linear_stability.Icrit_nA - 1e-3 # subtract epsilon to avoid numerical errors in comparison with critical level
  
  # --- plot stationary frequencies over drive
  ax_stat.fill_between([0,xmax*1.1], 140, 220, facecolor='lightgray', edgecolor='face', zorder=0)  
  ax_stat.plot(res_stat[res_stat.level>=level_crit]['level'], res_stat[res_stat.level>=level_crit]['freq_net'], marker='^', color=my_colors['fnet'], zorder=4, label=r'$f_\mathrm{net}^\mathrm{\infty}$')
  # res_stat.plot(ax=ax_stat, x= 'level', y='freq_net', marker='^', color=my_colors['fnet'], zorder=3, legend=False, label=r'$f_\mathrm{net}^\mathrm{\infty}$')
  res_stat.plot(ax=ax_stat, x= 'level', y='freq_unit_mean', marker='o', color=my_colors['funit'], zorder=3, legend=False, label=r'$f_\mathrm{unit}^\mathrm{\infty}$')  #yerr = 'freq_unit_std', 
  ax_stat.set_xlabel('drive [{}]'.format(traj_stat.input.unit.replace('sec','s')))
  ax_stat.set_xlim([0,xmax])
  ax_stat.set_ylim(ylim)
  
  # --- load transient drive data ----------------------------
  traj_trans = pypet_load_trajectory(traj_hash = traj_trans_hash, path_to_simulations=path_to_simulations)
  
  # load results from IFA analysis
  ifa_stats = pd.read_csv(pypet_get_trajectoryPath(traj_hash = traj_trans_hash, path_to_simulations=path_to_simulations) + 'analysis_IFA_fmin-{}/data_ifa_h{}_summary.csv'.format(fmin, traj_trans_hash), index_col=0, squeeze=True)
  ax_trans2 = list(np.zeros(len(ifa_stats)))
  
  for i in range(len(ifa_stats)):      
      # --- find run indices of interest (the simulations belonging to the network configuration specified by "exploration"-----------------------------------------------
      run_idx = pypet_find_runs(traj_trans, 'ramp_time_up', lambda x: x==ifa_stats.loc[i]['ramp_time_up'])             
      # nreps = len(run_idx) # number of simulation repetitions done for this network configuration
      
      # load correct runs:
      ifreq_t = pypet_get_from_runs(traj_trans, 'network.ifreq_discr_t',run_idx=run_idx) # dict 
      ifreq = pypet_get_from_runs(traj_trans, 'network.ifreq_discr',run_idx=run_idx) # dict
      traj_trans.v_idx = run_idx[0]
      stimulus = traj_trans.derived_parameters.runs[traj_trans.v_crun]['stim_plot']

      # important time stamps
      tw = traj_trans.analysis.ifreq_targetwindow
      t0 = np.mean(tw) # middle of ramp stimulation
      t_stim_on = tw[0] + (np.diff(tw) - (traj_trans.ramp_time_up + traj_trans.plateau_time + traj_trans.ramp_time_down)) / 2 # reinferring Tedge here
      t_stim_off = t_stim_on + traj_trans.ramp_time_up + traj_trans.plateau_time + traj_trans.ramp_time_down # end of ramp
      # t_mid_plateau = t_stim_on + traj_trans.ramp_time_up + traj_trans.plateau_time/2 # middle of plateau
      # color markers depending on TIME of instfreq
      norm = matplotlib.colors.TwoSlopeNorm(t0, vmin=t_stim_on, vmax=t_stim_off)
      
      # restrict ifreq analysis to time points DURING RAMP:
      ifreq_t_all = list_from_dict(ifreq_t, output_numpy=True) 
      ifreq_all  = list_from_dict(ifreq, output_numpy=True)    
      ix_keep = (ifreq_t_all >= t_stim_on) & (ifreq_t_all <= t_stim_off)
      ifreq_t_all = ifreq_t_all[ix_keep]
      ifreq_all = ifreq_all[ix_keep]
      
      # current at time of ifreq measurement
      Iext_trans = stimulus[(ifreq_t_all/traj_trans.dt).astype(int)]  
  
      # --- plot instantanous frequencies over drive in ax_stat: ONLY FOR THE SYMMETRIC RAMP!
      if traj_trans.ramp_time_up == traj_trans.ramp_time_down:
        color = plt.cm.coolwarm(1-norm(ifreq_t_all))[:,:3]
        ax_stat.scatter(Iext_trans, ifreq_all, c = color, marker='.', s=2.5, zorder=2, label=r'$f_\mathrm{net}^\mathrm{inst}$')
        # mark point of full synch in stat plot
        gridline(traj_trans.input.peak, ax_stat, axis='x', zorder=1)
      else:
        color = 'grey'
        
      # --- plot instantanous frequencies over time
      ax_trans[i].scatter(ifreq_t_all-t0, ifreq_all, marker='.', s=2.5, c = color, zorder=2) #, label=r'$f_\mathrm{net}^\mathrm{inst}$'*label_finst)
  
      # --- plot stationary frequencies over time
      # interpolate stationary frequencies from stat simulation result
      fnet_stat_interp = np.interp(stimulus, res_stat['parameters.input.level'].values, res_stat['freq_net'])
      fnet_stat_interp[stimulus < level_crit] = nan
      t = np.arange(stimulus.size)*traj_trans.dt - t0
      ax_trans[i].plot(t, fnet_stat_interp, 'k', zorder=3)
  
      # limit yaxis
      ax_trans[i].set_ylim(ylim)
      ax_trans[i].set_xlim([t_stim_on-tpad-t0, t_stim_off+tpad-t0])
  
      # mark transient stimulus (normalized)
      ax_trans2[i] = ax_trans[i].twinx()
      time = np.arange(len(stimulus)) * traj_trans.dt
      ax_trans2[i].plot(time-t0, stimulus/np.max(stimulus), color='gray')#, label='drive')
      ax_trans2[i].set_ylim([0,5])
      if show_second_yaxis and (i==len(ifa_stats)-1):
        ax_trans2[i].set_yticks([0,1])
        ax_trans2[i].tick_params(axis='y', labelcolor='gray', color='gray')
        ax_trans2[i].spines['right'].set_color('gray')
        ax_trans2[i].set_ylabel('drive [normalized]', color='gray')
        ax_trans2[i].spines['top'].set_visible(False)
      else:
        despine(ax_trans2[i])
        ax_trans2[i].set_yticks([])    
        
      # add linear regression line 
      ifa_slope = ifa_stats.loc[i]['ifa_slope']
      ifa_intercept = ifa_stats.loc[i]['ifa_intercept']
      time = np.arange(t_stim_on, t_stim_off+traj_trans.dt, traj_trans.dt) -t0
      ax_trans[i].plot(time, ifa_slope*time + ifa_intercept, color='grey', linestyle='-', lw=.5, label=r'$\chi_\mathrm{IFA}$='+'{:.1f} Hz/ms'.format(ifa_slope))  
      ax_trans[i].text(1,1, r'$\chi_\mathrm{IFA}$='+'{:.1f} Hz/ms'.format(ifa_slope), ha='right',  va='bottom', fontsize=6, transform=ax_trans[i].transAxes)
      # ax_trans[i].legend(bbox_to_anchor=(1,1), loc='lower right', handlelength=1, borderaxespad=0., fontsize=6)
  
  ax_trans[-1].set_xlabel('time [ms]')
  return ax_stat, ax_trans

def plot_freqs_stat_vs_trans_reduced(traj_stat_hash, traj_trans_hash, ax_stat, ax_trans, level_crit = None, xmax=100, ylim=[0,400], tpad=10, fmin=70, show_second_yaxis=False, path_to_simulations='./simulations/', label_finst=False):
  # load stationary data 
  traj_stat = pypet_load_trajectory(traj_hash = traj_stat_hash, path_to_simulations=path_to_simulations)
  res_stat = traj_stat.results.summary.scalarResults  
  res_stat['level'] = res_stat['parameters.input.level'].copy()
  if not level_crit:
    level_crit = traj_stat.linear_stability.Icrit_nA - 1e-3 # subtract epsilon to avoid numerical errors in comparison with critical level
    
  # --- load transient drive data ----------------------------
  traj_trans = pypet_load_trajectory(traj_hash = traj_trans_hash, path_to_simulations=path_to_simulations)
  ifreq_t = pypet_get_from_runs(traj_trans, 'network.ifreq_discr_t') # dict 
  ifreq = pypet_get_from_runs(traj_trans, 'network.ifreq_discr') # dict
  traj_trans.v_idx = 0
  stimulus = traj_trans.derived_parameters.runs[traj_trans.v_crun]['stim_plot']
  t0 = np.mean(traj_trans.analysis.ifreq_targetwindow) # middle of ramp stimulation
  ifreq_t_all = list_from_dict(ifreq_t, output_numpy=True) 
  ifreq_all  = list_from_dict(ifreq, output_numpy=True)
  
  # restrict to time points DURING RAMP:
  t_ramp_on = t0 - traj_trans.plateau_time/2 - traj_trans.ramp_time # beginning of ramp
  t_ramp_off = t0 + traj_trans.plateau_time/2 + traj_trans.ramp_time # end of ramp
  ix_keep = (ifreq_t_all >= t_ramp_on) & (ifreq_t_all <= t_ramp_off)
  ifreq_t_all = ifreq_t_all[ix_keep]
  ifreq_all = ifreq_all[ix_keep]
  
  # current at time of ifreq measurement
  Iext_trans = stimulus[(ifreq_t_all/traj_trans.dt).astype(int)]  
  
  # --- plot instantanous frequencies over time or drive resp.
  # color markers depending on TIME of instfreq
  # tmin, tmax = np.min(ifreq_t_all), np.max(ifreq_t_all)
  norm = matplotlib.colors.TwoSlopeNorm(t0, vmin=t_ramp_on, vmax=t_ramp_off)
  ax_stat.scatter(Iext_trans, ifreq_all, c = plt.cm.coolwarm(1-norm(ifreq_t_all))[:,:3], marker='.', s=4, zorder=2, label=r'$f_\mathrm{net}^\mathrm{inst}$')
  ax_trans.scatter(ifreq_t_all-t0, ifreq_all, marker='.', s=3, c = plt.cm.coolwarm(1-norm(ifreq_t_all))[:,:3], zorder=2) #, label=r'$f_\mathrm{net}^\mathrm{inst}$'*label_finst)
  ax_trans.set_xlabel('time [ms]')
  
  # --- plot stationary frequencies over drive
  ax_stat.fill_between([0,xmax*1.1], 140, 220, facecolor='lightgray', edgecolor='face', zorder=0)  
  ax_stat.plot(res_stat[res_stat.level>=level_crit]['level'], res_stat[res_stat.level>=level_crit]['freq_net'], marker='^', color=my_colors['fnet'], zorder=4, label=r'$f_\mathrm{net}^\mathrm{\infty}$')
  # res_stat.plot(ax=ax_stat, x= 'level', y='freq_net', marker='^', color=my_colors['fnet'], zorder=3, legend=False, label=r'$f_\mathrm{net}^\mathrm{\infty}$')
  res_stat.plot(ax=ax_stat, x= 'level', y='freq_unit_mean', marker='o', color=my_colors['funit'], zorder=3, legend=False, label=r'$f_\mathrm{unit}^\mathrm{\infty}$')  #yerr = 'freq_unit_std', 
  ax_stat.set_xlabel('drive [{}]'.format(traj_stat.input.unit.replace('sec','s')))
  ax_stat.set_xlim([0,xmax])
  
  # --- plot stationary frequencies over time
  # interpolate stationary frequencies from stat simulation result
  fnet_stat_interp = np.interp(stimulus, res_stat['parameters.input.level'].values, res_stat['freq_net'])
  fnet_stat_interp[stimulus < level_crit] = nan
  t = np.arange(stimulus.size)*traj_trans.dt - np.mean(traj_trans.analysis.ifreq_targetwindow)
  ax_trans.plot(t, fnet_stat_interp, 'k', zorder=3)
  
  # mark point of full synch
  gridline(traj_trans.input.peak, ax_stat, axis='x', zorder=1)
  
  # limit yaxis
  ax_stat.set_ylim(ylim)
  ax_trans.set_ylim(ylim)
  ax_trans.set_xlim([t_ramp_on-tpad-t0, t_ramp_off+tpad-t0])
  
  # mark transient stimulus (normalized)
  ax_trans2 = ax_trans.twinx()
  time = np.arange(len(stimulus)) * traj_trans.dt
  ax_trans2.plot(time-t0, stimulus/np.max(stimulus), color='lightgray')#, label='drive')
  ax_trans2.set_ylim([0,6])
  if show_second_yaxis:
    ax_trans2.set_yticks([0,1])
    ax_trans2.tick_params(axis='y', labelcolor='gray', color='gray')
    ax_trans2.spines['right'].set_color('gray')
    ax_trans2.set_ylabel('drive [normalized]', color='gray')
    ax_trans2.spines['top'].set_visible(False)
  else:
    despine(ax_trans2)
    ax_trans2.set_yticks([])    
    
  # add linear regression line 
  with np.load(pypet_get_trajectoryPath(traj_trans_hash, path_to_simulations=path_to_simulations) + 'analysis_IFA_fmin-{}/data_ifa_hash{}.npz'.format(fmin, traj_trans_hash)) as data:
    ifa_slope = data['ifa_slope']
    ifa_intercept = data['ifa_intercept']
  time = np.arange(t_ramp_on, t_ramp_off+traj_trans.dt, traj_trans.dt) -t0
  ax_trans.plot(time, ifa_slope*time + ifa_intercept, color='gray', linestyle='-', lw=.5, label=r'$\chi_\mathrm{IFA}$='+'{:.1f} Hz/ms'.format(ifa_slope))  
  
  ax_trans.legend(handlelength=1, loc='upper right', borderaxespad=0., fontsize=6)
  return ax_stat, ax_trans

def plot_figure_S2C_reduced(path_to_figures = './figures/', path_to_simulations = './simulations/'):
  fig_width= width_a4_wmargin
  fig_height = fig_width*.5
  
  fig = plt.figure(figsize=(fig_width, fig_height))#, constrained_layout=True
  
  gs = gridspec.GridSpec(2,1, figure=fig, hspace=.7)
  gs1 = gs[0].subgridspec(1,4)
  gs2 = gs[1].subgridspec(1,4)
  
  ax_top = gs1.subplots(sharey=True)
  ax_bottom = gs2.subplots(sharey=True)
  
  despine(ax_top)
  despine(ax_bottom)
  
  ax_top[0].set_ylabel('frequency [Hz]')
  ax_bottom[0].set_ylabel('frequency [Hz]')
  
  dx, dy = 0, 1.1
  for i in range(4):
    ax_top[i].text(dx, dy, string.ascii_uppercase[i]+'i', transform=ax_top[i].transAxes, weight='bold')
    ax_bottom[i].text(dx, dy, string.ascii_uppercase[i]+'ii', transform=ax_bottom[i].transAxes, weight='bold')
  
  # --- data ---------------------------------------------
  # --- Donoso2018 original network ---------------------------------------------------------
  ax_top[0], ax_bottom[0] = plot_freqs_stat_vs_trans(traj_stat_hash = 1001, traj_trans_hash = 10011, ax_stat = ax_top[0], ax_trans = ax_bottom[0],\
                                                      level_crit = 1500, xmax=16_000, path_to_simulations=path_to_simulations)
  
  
  # --- Donoso2018 network with independent Poisson inputs ---------------------------------------------------------
  ax_top[1], ax_bottom[1] = plot_freqs_stat_vs_trans(traj_stat_hash = 1003, traj_trans_hash = 10031, ax_stat = ax_top[1], ax_trans = ax_bottom[1],\
                                                      level_crit = 1500, xmax=16_000, path_to_simulations=path_to_simulations) # try 1005 & 10051 for a larger network 
  
  # --- Reduced network with refractory period ---------------------------------------------------------
  ax_top[2], ax_bottom[2] = plot_freqs_stat_vs_trans(traj_stat_hash = 1002, traj_trans_hash = 10021, ax_stat = ax_top[2], ax_trans = ax_bottom[2], xmax=2.2, path_to_simulations=path_to_simulations)
  
  # --- Reduced network as in manuscript -----------------------------------------------------------------------
  ax_top[3], ax_bottom[3] = plot_freqs_stat_vs_trans(traj_stat_hash = 1004, traj_trans_hash = 10041, ax_stat = ax_top[3], ax_trans = ax_bottom[3], xmax=1.3, path_to_simulations=path_to_simulations, show_second_yaxis=True, label_finst=True)

  ax_top[-1].legend(bbox_to_anchor=(1.2, 1.2), handlelength=1, edgecolor='dimgrey', loc='upper right', borderaxespad=0., fontsize = 6, facecolor='white')
  
  fig.savefig(path_to_figures + 'FigsX.pdf', bbox_inches='tight')
  return



#%% Reviewer Figures
def plot_reviewer_figure_3(path_to_figures='./figures/'):
  from methods_analytical import integrate_dde_numerically
  
  # parameters:
  Iext = 1
  D = 0.04
  Delta = 1.2
  K = 5 
  tm = 10
  Vr = 0
  tmax = 30
  dt = 0.01
  reset= False
  
  # initialization:
  mu_00 = 0
  t = np.arange(0, Delta+dt, dt)
  Iext_aux = 6.5
  mu_history = Iext_aux - (Iext_aux-mu_00)*np.exp(-t/tm)
  
  # integrate DDE numerically
  t, mu, r, ix_max, ix_min_real, mu_max, mu_reset, mu_min_real, ix_min_theory, mu_min_theory, toff_theory, fig, ax \
  = integrate_dde_numerically(Iext, D, Delta, K, tm, Vr = Vr, tmax = tmax, dt =dt, mu0= mu_history, reset=reset,  plot=True, rtol_conv=1e-2, \
                                keep_initial_history=True)
    
  fig.savefig(path_to_figures+'fig_R3_left_dde_initialization.pdf', bbox_inches='tight')
  
  # zoom into panels and save figure again
  ax[0].set_ylim([0, 50])
  ax[2].set_ylim([-1,1])
  ax[3].set_ylim(ax[1].get_ylim())
  ax[-1].set_ylim([-2, Iext_aux*1.1])   
  fig.savefig(path_to_figures+'fig_R3_right_dde_initialization_zoom.pdf', bbox_inches='tight')

  return

def plot_reviewer_figure_6(traj_hash=1, run_idx = [0,5,9,11], path_to_simulations = './simulations/', path_to_figures='./figures/'):
  ''' 
  calculate and plot the average power spectral density of the membrane potentials for
  simulation traj_hash and the selected runs in run_idx.
  '''  
  traj = pypet_load_trajectory(traj_hash = traj_hash, path_to_simulations = path_to_simulations)
  len_power = int(1000/2/traj.dt / (1000/np.diff(traj.v_recordwindow)))
  power_average = np.zeros((len_power, len(run_idx)))
  
  fig, ax = plt.subplots(figsize=(10,5))
  despine(ax)
  color = ['b', 'c', 'g', 'gold']
  
  for r, run in enumerate(run_idx):
    traj.v_idx = run 
    # load membrane potentials
    v = traj.results.crun.raw.v # time x neurons
    power = np.zeros((traj.N_record_max, len_power))
    for i in range(traj.N_record_max):
      freqs, power[i,:] = get_PSD(v[:,i], traj.dt/1000, df='min', k=1, offset=0)
    power_average[:,r] = np.mean(power, axis=0)
    
    ax.plot(freqs, power_average[:,r], label='{:.2f}'.format(traj.level), color=color[r])
  
  traj.f_restore_default()
  ax.legend(title=r'$I_\mathrm{ext}$')
  ax.set_xlabel('frequency [Hz]')
  ax.set_ylabel('power [a.u.]')
  ax.set_xlim([0, 400])
  ax.set_ylim([1e-5, 1e6])
  ax.set_yscale('log')
  
  fig.savefig(path_to_figures+'fig_R6_rev_psd-V.pdf', bbox_inches='tight')
  return


#%% Spielplatz

with plt.rc_context({"axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 8,"font.size": 8,\
                     "xtick.labelsize": 6, "ytick.labelsize": 6,\
                      "xtick.direction": "out", "ytick.direction": "out", 'xtick.major.pad': 1, 'ytick.major.pad': 1}):
  fig_width= plos_width_fullpage # width_a4_wmargin # 21*cm
  fig_height = fig_width*.6 #9*cm #10.5*cm
  fig = plt.figure(figsize=(fig_width, fig_height))
  gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios = [1, 1, 1], hspace=.35, wspace=.5)#, width_ratios=[5,2], height_ratios=[2,3])

  gs_0 = gs[0,:].subgridspec(2,2, height_ratios = [.5,2], hspace=.2, wspace=.1)
  gs_A_sup = gs[1,0].subgridspec(1,2, width_ratios=[30,1])
  gs_A = gs_A_sup[0,0].subgridspec(2, 3, height_ratios = [.5,2], wspace=.5, hspace=.2)
  gs_B = gs[2,0].subgridspec(1, 3, width_ratios = [20, .5, 2], wspace=.1)
  gs_C = gs[1:,1].subgridspec(3, 1, height_ratios=[1,.5,2], hspace=.1)
  
  ax_0 = gs_0.subplots(sharey=True)
  ax_A = gs_A.subplots(sharex=True).T
  ax_B = gs_B.subplots() #[fig.add_subplot(gs_B[0,0]), fig.add_subplot(gs_B[0,1])]
  ax_B[-1].remove()
  ax_C = gs_C.subplots(sharex=True)
  despine(ax_C)
  despine(ax_0, which=['top', 'right', 'bottom'])
  despine(ax_A, which=['top', 'right', 'bottom'])
  despine(ax_A[1:,:], which=['left'])
  
  ax_0[0][0].text(-1.2, 9, 'Ai', transform=ax_A[0][0].transAxes, size=panel_labelsize, weight='bold')
  ax_0[0][0].text(5.2, 9, 'Aii', transform=ax_A[0][0].transAxes, size=panel_labelsize, weight='bold')
  ax_A[0][0].text(-1.2, 1.5, string.ascii_uppercase[1], transform=ax_A[0][0].transAxes, size=panel_labelsize, weight='bold')
  ax_B[0].text(-1.2, -.65, string.ascii_uppercase[2], transform=ax_A[0][1].transAxes, size=panel_labelsize, weight='bold')
  ax_C[0].text(2.38, 1.5, string.ascii_uppercase[3], transform=ax_A[-1][0].transAxes, size=panel_labelsize, weight='bold')
  
  fig.savefig('test.pdf', bbox_inches='tight')

plt.close('all')