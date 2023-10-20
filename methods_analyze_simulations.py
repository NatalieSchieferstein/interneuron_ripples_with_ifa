#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 19:16:57 2023

This file contains all code for analyses of the network dynamics that go beyond what has already been done during the simulation:
* analysis of instantaneous frequency dynamics and IFA (do_ifa_analysis)
* analysis of membrane potential dynamics to compare to the Gaussian approximation (pypet_analyse_membrane_potential_dynamics)
* application of the analytical Gaussian-drift approximation to a particular network simulation (extracting the parameters from a simulation file) (pypet_gaussian_drift_approx_stat)
* plotting of network dynamics

These functions are partially called in main_plot_figures.py

@author: natalie
"""
import json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import numpy as np
import os
import pandas as pd 
import string

from methods_simulations import find_full_synch_interpolated, getVtraces, pypet_load_trajectory, pypet_find_runs, \
  pypet_get_from_runs, pypet_get_exploration_overview, pypet_get_trajectoryPath, pypet_shortcut, rescale_params_dimless
from methods_analytical import get_stationary_rate_lif, get_stationary_fpe_lif, gaussian_drift_approx_stat_numerical, \
  gaussian_drift_approx_stat, get_pt_fullsynch
from tools import add_ticks, despine, dict2str, do_linear_regression, frame_subplot,  get_gauss,  \
                  gridline, list_from_dict, plot_multicolor_line
pi=np.pi
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
panel_labelsize = 9


#%% IFA analysis
def get_instfreq_continuous(wspec, wspec_extent, Pthr=np.nan, fmin=70, set_fmin_na = True):
  '''
  extract continuous estimate of inst. frequency from wavelet spectrogram

  Parameters
  ----------
  wspec : np.array
    wavelet spectrogram (trials x freq x time).
  wspec_extent : list
    boundaries of wspec: [time start, time end, frequency lowest, frequency highest].
  Pthr : float, optional
    Minimal power required for inst. freq. to be significant. The default is np.nan.
  fmin : float (Hz), optional
    Minimum for instantaneous frequency. The default is 70 (Hz).
  set_fmin_na : bool, optional
    Set all inst. frequencies below fmin to nan? The default is True.

  Returns
  -------
  instfreq : np.array. Inst. frequencies.
  instpower : np.array. Inst. power
  instfreq_significant : Inst. frequencies with nan values where inst. power does not exceed Pthr.

  '''
  freq = np.linspace(wspec_extent[0,2], wspec_extent[0,3], wspec.shape[1], endpoint=False)
  fmin_idx = np.where(freq>=fmin)[0][0]
  inst_idx = np.argmax(wspec[:, freq>=fmin, :], axis=1) + fmin_idx
  
  instfreq = freq[inst_idx] # instantaneous frequency estimate, consider only freqs > fmin
  instpower = wspec[np.arange(wspec.shape[0])[:,None], inst_idx, np.arange(wspec.shape[-1])] #F[ifmin+imax,np.arange(0,F.shape[1])]

  instfreq_significant = instfreq.copy()

  if set_fmin_na:
    instfreq_significant[instfreq_significant==fmin] = nan
    # instpower[instfreq==fmin] = nan
  if not np.isnan(Pthr):
    instfreq_significant[instpower < Pthr] = nan  
  
  return instfreq, instpower, instfreq_significant

def prepare_ifa_analysis(traj, exploredParameters=[]):
  '''
  checks whether additional parameters were explored and if yes, extracts all explorations that need to be analyzed for IFA separately.
  for multidimensional explorations (see h296), one can also provide the relevant exploredParameters as an input 
  it will be assumed then that during the repetitions of configurations of these parameters, all additionally varied parameters were kept constant
  (nreps trials per parameter configuration)
  

  Parameters
  ----------
  traj : loaded pypet trajectory
  exploredParameters : list of strings, optional
    complete list of relevant, explored Parameters in case of more complicated explorations, where they cannot be inferred automatically. The default is [].

  Returns
  -------
  all_configs : pandas data frame
    list of explored network configurations for which IFA analysis will be done separately.
  exploredParameters : list of str
    list of explored parameters (other than brian_seed etc and aux parameters) that distinguish the above network configurations.
  nreps : int.
    number of repetitions that were simulated per network configuration.
  '''
  if not len(exploredParameters):
    exploredParameters_all = pypet_shortcut(list(traj.f_get_explored_parameters()))
    # get rid of any parameters that just had to be COvaried given the actual explored parameters
    exploredParameters = list(set(exploredParameters_all) - set(['seed_brian', 'simulation.seed_brian', 'ifreq_targetwindow', 'v_recordwindow', 'v_recordwindow_for_simulation', 'Tsim', 'record_micro']))
  
  exploration_overview = pypet_get_exploration_overview(traj)
  if len(exploredParameters):
    all_configs = exploration_overview[exploredParameters].drop_duplicates()
    all_configs.reset_index(drop=True, inplace=True)
    n_configs = len(all_configs) # number of simulation configs each of which was simulated nreps times
    print('Found {} additional explored Parameter(s): {}.\nWill do {} separate IFA analyses plus a final comparison:\n'.format(len(list(all_configs.columns)), list(all_configs.columns), n_configs), all_configs)
  else:
    all_configs = []
    n_configs = 1
  nreps = len(exploration_overview)//n_configs
  return all_configs, exploredParameters, nreps

def do_ifa_analysis_1batch(traj, exploration={}, ex_run = 0, figPath='', Pthr=nan, fmin=70, flim=400, Tedge=20., t0_centered=True, save2file=False):
  '''
  

  Parameters
  ----------
  traj : TYPE
    DESCRIPTION.
  exploration : dict, optional
    specifies setting of additional explored Parameters (other than brian_seed). The default is {}.
  ex_run : int, optional
    specify which run to use for the example plot. The default is 0.
  ncols : TYPE, optional
    DESCRIPTION. The default is 5.
  figPath : TYPE, optional
    DESCRIPTION. The default is ''.
  Pthr : float, optional
    Power threshold.  If not nan, ripple detection will be redone for the given Pthr The default is nan.
  fmin : TYPE, optional
    DESCRIPTION. The default is 30.
  flim : TYPE, optional
    DESCRIPTION. The default is 350.
  save2file : TYPE, optional
    DESCRIPTION. The default is False.

  Returns
  -------
  None.

  '''

  # --- find run indices of interest (the simulations belonging to the network configuration specified by "exploration"-----------------------------------------------
  if not len(exploration): # take all runs, only one network configuration was simulated
      nruns = len(traj.f_get_run_names())
      run_idx = list(range(nruns))
      label = ''
  else: # filter out the runs that belong to the required network configuration
      label = dict2str(exploration, delimiter=', ')
      if len(exploration)==1:
        my_filter = lambda x1: x1==val1
        ep1, val1 = list(exploration.items())[0]
        run_idx = pypet_find_runs(traj, ep1, my_filter)        
      elif len(exploration)==2:
        my_filter = lambda x1, x2: (x1==val1) and (x2==val2)
        ep1, val1 = list(exploration.items())[0]
        ep2, val2 = list(exploration.items())[1]
        run_idx = pypet_find_runs(traj, [ep1, ep2], my_filter)
      else:
        raise ValueError('Implement 3D IFA analyses!')
  nreps = len(run_idx) # number of simulation repetitions done for this network configuration
  
  
  # --- extract data -----------------------------------------------------------
  wspec = pypet_get_from_runs(traj, 'network.wspec', run_idx=run_idx)
  wspec_extent = pypet_get_from_runs(traj, 'network.wspec_extent', run_idx=run_idx)
  ifreq_discr_t = pypet_get_from_runs(traj, 'network.ifreq_discr_t', run_idx=run_idx, return_nparray_if_possible=False) # force dict format
  ifreq_discr = pypet_get_from_runs(traj, 'network.ifreq_discr', run_idx=run_idx, return_nparray_if_possible=False)
  stimulus_check = pypet_get_from_runs(traj, 'stim_plot', run_idx=run_idx) # runs x time
  # all simulations should have had the same stimulus:
  if sum(np.std(stimulus_check, axis=0))>1e-5:
    raise ValueError('all trials in one IFA_1batch should have received the same stimulus!!')
  if np.isnan(Pthr):
    Pthr_all = pypet_get_from_runs(traj, 'network.Pthr', run_idx=run_idx)
    Pthr = np.mean(Pthr_all)
  
  # extract necessary parameters about input from a random run (input same for all runs)
  traj.v_idx = run_idx[0]
  stimulus_all = traj.derived_parameters.runs[traj.v_crun]['stim_plot']
  tw = traj.analysis.ifreq_targetwindow  # time window for which inst. freq. was analyzed (symmetric around stimulus)
  t0 = np.mean(tw) # center of stimulus time window
  stimulus = stimulus_all[int(tw[0]/traj.dt):int(tw[1]/traj.dt)]   # keep only the part of the stimulus within ifreq_targetwindow
  # determine (approx) beginning and end of stimulation for restricting the IFA analysis below
  if traj.input.shape == 'ramp':
    t_stim_on = t0 - traj.plateau_time/2 - traj.ramp_time # beginning of ramp
    t_stim_off = t0 + traj.plateau_time/2 + traj.ramp_time # end of ramp
  elif traj.input.shape == 'ramp_asym':
    t_stim_on = tw[0] + (np.diff(tw) - (traj.ramp_time_up + traj.plateau_time + traj.ramp_time_down)) / 2 # reinferring Tedge here
    t_stim_off = t_stim_on + traj.ramp_time_up + traj.plateau_time + traj.ramp_time_down # end of ramp
  else:
    t_stim_on = tw[0] + Tedge
    t_stim_off = tw[-1] - Tedge  
  traj.f_restore_default()

  # --- analysis ---------------------------------------------------------------
  # continuous estimate of instantaneous frequency. Re-determine inst freq for the given fmin and Pthr (instead of using analysis results from pypet run)
  ifreq, ipower, ifreq_significant = get_instfreq_continuous(wspec, wspec_extent, Pthr= Pthr, fmin=fmin, set_fmin_na = True)  
  
  # linear regression to quantify IFA:
  # all data in one array:
  ifreq_discr_t_all = list_from_dict(ifreq_discr_t, output_numpy=True)
  ifreq_discr_all  = list_from_dict(ifreq_discr, output_numpy=True)
  mask_f = ifreq_discr_all >= fmin # only freqs above a minimum
  mask_t = (ifreq_discr_t_all >= t_stim_on) & (ifreq_discr_t_all <= t_stim_off) # only consider frequencies DURING the stimulation?
  ifa_slope, ifa_intercept = do_linear_regression(ifreq_discr_t_all[mask_f & mask_t] - t0, ifreq_discr_all[mask_f & mask_t]) # intercept w.r.t. time t0
    
    
  # --- plotting ----------------------------------------------------------------
  # --- plot example run
  fig_example = plot_ifa_example_run(traj, run_idx[ex_run], Pthr=Pthr, fmin=fmin, t0_centered=t0_centered)[0]
  fig_example.suptitle(label)

  # --- plot all runs
  fig_all = plot_ifa_all_runs(traj, run_idx, Pthr, label, fmin=fmin, t0_centered=t0_centered)

  # --- plot summary + linear regression over all runs
  fig_mean_discr_regress \
  = plot_ifa_linreg([], ifreq_discr, ifreq_discr_t, stimulus_all, traj.dt, tw, inputunit=traj.input.unit, \
                      ifa_slope=ifa_slope, ifa_intercept = ifa_intercept, color_line='k', color_dots='dimgrey', flim=flim, t0_centered=t0_centered)[0]
  
  # --- save ----------------------------------------------------------------------
  if save2file:
    figname = 'fig_ifa_xxx_hash{}'.format(traj.hash)
    file_name = 'data_ifa_hash{}.npz'.format(traj.hash)
    if label:
      figname += '_{}'.format(dict2str(exploration, equals='-'))
      file_name += '_{}'.format(dict2str(exploration, equals='-'))
    fig_mean_discr_regress.savefig(os.path.join(figPath + figname.replace('xxx', 'mean_discrete_linregress')+'.pdf'), bbox_inches='tight')
    fig_all.savefig(os.path.join(figPath + figname.replace('xxx', 'all_runs')+'.pdf'), bbox_inches='tight')
    fig_example.savefig(os.path.join(figPath + figname.replace('xxx', 'example_run{}'.format(run_idx[ex_run]))+'.pdf'), bbox_inches='tight')
    # save mean data for merging with cyclostat results
    np.savez(figPath+file_name, ifreq=ifreq, ipower=ipower, stimulus=stimulus, \
             ifa_slope=ifa_slope, ifa_intercept=ifa_intercept, fmin=fmin, inputunit=traj.input.unit, dt=traj.dt)
          
  return ifa_slope, ifa_intercept, stimulus, Pthr, label, tw 

def do_ifa_analysis(traj_hash, ex_run = 0, Pthr=nan, fmin=70, flim=350, Tedge=20., save2file=True, colors=[], exploredParameters=[],\
                    path_to_simulations = './simulations/', t0_centered=True):
  print('IFA analysis for traj {}...'.format(traj_hash))
  traj = pypet_load_trajectory(traj_hash=traj_hash, path_to_simulations = path_to_simulations) # load data
  # prepare path for analysis results
  datapath = pypet_get_trajectoryPath(traj_hash, path_to_simulations = path_to_simulations) + 'analysis_IFA_fmin-{}/'.format(int(fmin))
  if not os.path.exists(datapath):
    os.makedirs(datapath)
  
  # recover the different network configurations that were explored 
  all_configs, exploredParameters, nreps = prepare_ifa_analysis(traj, exploredParameters=exploredParameters)
  
  # do one IFA analysis per network configuration 
  if not len(exploredParameters):
    # only one network was simulated
    do_ifa_analysis_1batch(traj, ex_run = ex_run, figPath=datapath, Pthr=Pthr, fmin=fmin, flim=flim, Tedge=Tedge, t0_centered=t0_centered, save2file=save2file)
  else:
    # multiple network configurations were explored
    # --- collect average IFA data for subsequent comparison
    label = ['']*len(all_configs)
    ifa_stats = all_configs.copy() # summarize ifa properties for the different configurations
    tw = np.zeros((len(all_configs), 2)) # time window symmetric around stimulus 
    ifreq_av, ifreq_std, snr, stimulus, ripple_av, Pthr_av \
    = {}, {}, {}, {}, np.zeros((len(all_configs), 2)), np.zeros(len(all_configs))
    
    freq_diff, ifreq_onset, ifreq_offset, level_onset, level_offset\
    =  np.zeros((len(all_configs), nreps)),  np.zeros((len(all_configs), nreps)),  np.zeros((len(all_configs), nreps)),  np.zeros((len(all_configs), nreps)),  np.zeros((len(all_configs), nreps))
    
    # do separate IFA analyses per network configuration
    for i in range(len(all_configs)):
      exploration = all_configs.loc[i].to_dict() # parameter settings of this network
      # IFA analysis:
      ifa_stats.loc[i,'ifa_slope'], ifa_stats.loc[i,'ifa_intercept'], stimulus[i], Pthr_av[i], label[i], tw[i,:] \
      = do_ifa_analysis_1batch(traj, exploration= exploration, ex_run = ex_run, figPath=datapath, Pthr=Pthr, fmin=fmin, flim=flim, Tedge=Tedge, \
                               t0_centered=t0_centered, save2file=save2file)
      plt.close('all') # close figs
    
    print('Results:')
    print(ifa_stats)   
    if save2file:
       ifa_stats.to_csv(datapath+'data_ifa_h{}_summary.csv'.format(traj_hash))  
  print('[done]')    
  return 

#%% plot IFA
from tools import map_1d_idx_2d_grid

def get_str_drive(inputunit, linebreak=False):
    if inputunit == 'nA':
      if linebreak:
        str_drive = '$I_\mathrm{ext}$ \n[nA]'
        str_drive_val = str_drive+'= {}\nnA'
      else:
        str_drive = '$I_\mathrm{ext}$ [nA]'
        str_drive_val = str_drive+'= {} nA'
    elif inputunit=='':
      str_drive = '$I_\mathrm{E}$'
      str_drive_val = str_drive+'= {}'
    elif inputunit == 'spks/sec':
      str_drive = '$\Lambda$ [spk/s]'
      str_drive_val = str_drive+'= {} spk/s'
    return str_drive, str_drive_val


def plot_ifa_example_run(traj, run_idx, Pthr=nan, maxpower=nan, fmin=70, neuronview=[0,10], fig=None, gs=None, cbar_max=None, t0_centered=True):
  # load data from run "run_idx"
  traj.v_idx = run_idx
  if traj.scale=='micro':
    # load voltage traces if available
    if 'v' in traj.record_micro:
      v = traj.results.crun.raw.v
    else:
      v = []
  Iext = traj.derived_parameters.crun.stim_plot    
  wspec, wspec_extent = traj.results.crun.network.wspec, traj.results.crun.network.wspec_extent
  if np.isnan(Pthr):
    Pthr = traj.results.crun.network.Pthr
  
  str_drive, str_drive_val = get_str_drive(traj.input.unit, linebreak=True)
  # with plt.rc_context({"ytick.direction": "out", "xtick.direction": "out",}):
  # --- construct figure
  if not fig:
    fig_width= 8*cm
    fig_height = 10*cm #10.5*cm
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(5, 2, figure=fig, width_ratios=[15,1], height_ratios=[3,1,1,2,1], wspace=.05)
  gs_cbar = gs[:,1].subgridspec(5,2, height_ratios=[3,1,1,2,1], wspace=.5)
  
  
  ax_freq = fig.add_subplot(gs[0,0])
  ax_freq_cbar = fig.add_subplot(gs_cbar[0,0])
  ax_freq_cbar_inst = fig.add_subplot(gs_cbar[0,1])
  ax_rate = fig.add_subplot(gs[1,0], sharex=ax_freq)
  ax_raster = fig.add_subplot(gs[2,0], sharex=ax_freq)
  ax_volt = fig.add_subplot(gs[3,0], sharex=ax_freq)
  ax_volt_cbar = fig.add_subplot(gs_cbar[3,0])
  ax_input = fig.add_subplot(gs[4,0], sharex=ax_freq)

  despine([ax_rate, ax_raster, ax_volt, ax_input])

  ax_rate.spines['bottom'].set_visible(False)
  
  ax_raster.spines['bottom'].set_visible(False)
  
  # --- fill figure
  # call the middle of stimulation time 0
  if t0_centered:
    t0 = (wspec_extent[0] + wspec_extent[1])/2 # wspec is centered around the stimulus
  else:
    t0 = wspec_extent[0]
  
  fig, ax_freq, ax_freq_cbar, ax_freq_cbar_inst\
  = plot_wspec(fig, ax_freq, wspec, wspec_extent, traj.results.crun.network.instfreq, traj.results.crun.network.instpower, traj.dt, \
               ax_cb=ax_freq_cbar, ax_cb_inst=ax_freq_cbar_inst, Pthr=traj.results.crun.network.Pthr, fmin=fmin, maxpower=maxpower, t0=t0, label_cbar='')
  ax_freq.plot(traj.results.crun.network.ifreq_discr_t - t0, traj.results.crun.network.ifreq_discr, 'wo', ms=2, label='discrete')
  
  # add continuous legend handle
  handles, labels = ax_freq.get_legend_handles_labels() # plt.gca().get_legend_handles_labels()
  line = Line2D([0], [0], label='continuous', color='w')
  handles.extend([line])
  # ax_freq.legend(handles=handles, handlelength=1, loc='lower left', labelcolor='w', borderaxespad=0.01) #title='inst. freq.', 
  ax_rate = plot_rate(ax_rate, traj.results.crun.raw.LFP_smooth, traj.dt, t0=t0)
  ax_raster = plot_raster(ax_raster, traj.results.crun.raw.spktrains, neuronview=[0,10], t0=t0, ms=2)
  if len(v):
    ax_volt, ax_cb = plot_vhist(ax_volt, ax_volt_cbar, traj.Vthr, traj.Vreset, v=v, v_recordwindow=traj.v_recordwindow, t0=t0, cbar_max=cbar_max)
  
  ax_input.plot(np.arange(Iext.size)*traj.dt-t0, Iext, 'g')
#  ax_input.set_yticks([np.min(Iext), np.max(Iext), (np.min(Iext)+np.max(Iext))/2])
  ax_input.set_ylim(bottom=0)
  ax_input.set_ylabel(str_drive)
  
  for ax in [ax_freq, ax_rate, ax_raster, ax_volt]:
    plt.setp(ax.get_xticklabels(), visible=False)
  
  ax_raster.tick_params(axis = "x", which = "both", bottom = False, top = False)
  ax_rate.tick_params(axis = "x", which = "both", bottom = False, top = False)
  ax_rate.set_ylim(bottom=-20)
  
  ax_input.set_xlabel('time [ms]')
  ax_input.set_xlim([wspec_extent[0]-t0, wspec_extent[1]-t0])
  
  traj.f_restore_default()
  return fig, gs


def plot_ifa_all_runs(traj, run_idx, Pthr, suptitle, ncols=5, fmin=0, t0_centered=True):  
  
  # --- extract data -----------------------------------------------------------
  LFP = pypet_get_from_runs(traj, 'raw.LFP_smooth', run_idx=run_idx)
  ripple = pypet_get_from_runs(traj, 'network.ripple', run_idx=run_idx)
  wspec = pypet_get_from_runs(traj, 'network.wspec', run_idx=run_idx)
  wspec_extent = pypet_get_from_runs(traj, 'network.wspec_extent', run_idx=run_idx)
  ifreq_discr_t = pypet_get_from_runs(traj, 'network.ifreq_discr_t', run_idx=run_idx, return_nparray_if_possible=False) # force dict format
  ifreq_discr = pypet_get_from_runs(traj, 'network.ifreq_discr', run_idx=run_idx, return_nparray_if_possible=False)
  dt = traj.dt
  tw = traj.analysis.ifreq_targetwindow  # time window for which inst. freq. was analyzed (symmetric around stimulus)
  # extract stimulus
  traj.v_idx = run_idx[0]
  stimulus_all = traj.derived_parameters.runs[traj.v_crun]['stim_plot']
  tw = traj.analysis.ifreq_targetwindow  # time window for which inst. freq. was analyzed (symmetric around stimulus)
  stimulus = stimulus_all[int(tw[0]/traj.dt):int(tw[1]/traj.dt)]   # keep only the part of the stimulus within ifreq_targetwindow
  traj.f_restore_default()
  
  # continuous estimate of instantaneous frequency. Re-determine inst freq for the given fmin and Pthr (instead of using analysis results from pypet run)
  ifreq, ipower, ifreq_significant = get_instfreq_continuous(wspec, wspec_extent, Pthr= Pthr, fmin=fmin, set_fmin_na = True)  

  nreps = LFP.shape[0]
  # --- adjust figure layout  -----------------------------------------------------------
  if nreps%ncols != 0:
    if nreps%4 == 0:
      ncols = 4
    elif nreps%3 == 0:
      ncols = 3
    else:
      ncols = 2
  nrows = int(nreps/ncols)
  
  # --- construct figure
  with plt.rc_context({"xtick.labelsize": 8, "ytick.labelsize": 8,"font.size": 8}):
    fig = plt.figure(figsize=(12,nrows*2))#, constrained_layout=True)
    gs = gridspec.GridSpec(nrows, ncols+1, figure=fig, width_ratios=[30]*ncols+[1])
    gs_sub, ax_rate, ax_freq = [None]*nreps, [None]*nreps, [None]*nreps
    
    row, col = map_1d_idx_2d_grid(np.arange(nreps), nrows, ncols, start='lower left', direction='vertical')
    
    for i in range(nreps):
      gs_sub[i] = gs[row[i], col[i]].subgridspec(2,1, height_ratios=[1,4])#, hspace=0.1)
      if not i:
        ax_rate[i] = fig.add_subplot(gs_sub[i][0])
        ax_freq[i] = fig.add_subplot(gs_sub[i][1], sharex=ax_rate[0])
      else:
        ax_rate[i] = fig.add_subplot(gs_sub[i][0], sharex=ax_rate[0], sharey=ax_rate[0])
        ax_freq[i] = fig.add_subplot(gs_sub[i][1], sharex=ax_rate[0], sharey=ax_freq[0])
      
    gs_cb = gs[-1, -1].subgridspec(2,1, height_ratios=[1,4])  
    ax_cb = fig.add_subplot(gs_cb[1,0])
      
    # --- fill figure --------------------------------------------------------------------
    if not len(tw):
      tw = [wspec_extent[0,0] + wspec_extent[0,1]]
    if t0_centered:
      t0 = np.mean(tw)
    else:
      t0 = tw[0]
    t = dt*np.arange(stimulus.size) - t0 #np.arange(tw[0], tw[1], dt) - t0
    cb_max= np.max(wspec) # maximum for colorbar
    for i in range(nreps):
      if not i: # add colorbar and labels
        fig, ax_freq[i], ax_cb = plot_wspec(fig, ax_freq[i], wspec[i], wspec_extent[i], ifreq[i], ipower[i], dt, \
                                            ax_cb = ax_cb, Pthr=Pthr, maxpower=cb_max, t0 = t0, label=True, fmin=fmin)[:-1]
        ax_freq[i].plot(ifreq_discr_t[i]-t0, ifreq_discr[i], 'w.')
        ax_rate[i] = plot_rate(ax_rate[i], LFP[i], dt, t0=t0, label='[Hz]', col='k')
        # add the stimulus in a second axis TO DO
        ax_stim = ax_rate[i].twinx()
        ax_stim.plot(t, stimulus, 'g', lw=1, zorder=0)
        ax_stim.set_ylim(bottom=0)
        ax_stim.set_yticks([np.min(stimulus), np.max(stimulus)])
        ax_stim.spines['top'].set_visible(False)
        ax_stim.set_ylabel('[{}]'.format(traj.input.unit), color='g')
        ax_stim.tick_params(axis='y', labelcolor='g')
        
        ax_rate[i].set_xlim([tw[0]-t0, tw[1]-t0])
        ax_freq[i].set_xlabel('time [ms]')
      else:
        fig, ax_freq[i] = plot_wspec(fig, ax_freq[i], wspec[i], wspec_extent[i], ifreq[i], ipower[i], dt, \
                                        Pthr=Pthr, maxpower=cb_max, t0 = t0, label=False, fmin=fmin)[:-2]
        ax_freq[i].plot(ifreq_discr_t[i]-t0, ifreq_discr[i], 'w.')
        plt.setp(ax_freq[i].get_xticklabels(), visible=False)
        plt.setp(ax_freq[i].get_yticklabels(), visible=False)
        ax_rate[i] = plot_rate(ax_rate[i], LFP[i], dt, t0=t0, label='', col='k')
        ax_rate[i].axis('off')
      plt.setp(ax_rate[i].get_xticklabels(), visible=False) 
        
    if suptitle:
      fig.suptitle(suptitle)
  return fig


def plot_ifa_linreg(ax, ifreq, ifreq_t, Iext, dt, tw, fmin=0, flim=350, inputunit='', ifa_slope = None, ifa_intercept = None, \
                      color_dots='grey', color_line='dimgrey', tpad=0, print_slope=True, label='', ms=2, ymaxrel=1.1, show_drive=True,\
                      empty_marker=False, linestyle='-',zorder=2, y_text=1.1, va_text='bottom', t0_centered=True):
  if not len(ax):
    fig = plt.figure(figsize=(width_a4_wmargin*.4, width_a4_wmargin*.35))
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[3,1])
    ax = gs.subplots(sharex=True)
    despine(ax)
    return_fig=True
  else:
    return_fig=False
  str_drive, str_drive_val = get_str_drive(inputunit, linebreak=True)    
  if t0_centered:
    t0 = np.mean(tw)
  else:
    t0 = tw[0]
  time = dt*np.arange(Iext.size) - t0

  ifreq_t_all = list_from_dict(ifreq_t, output_numpy=True)
  ifreq_all  = list_from_dict(ifreq, output_numpy=True)

  ax[0].plot(ifreq_t_all-t0, ifreq_all, '.', color=color_dots, ms=ms, label=label, zorder=zorder)
  # linear regression line:
  ax[0].plot(time, ifa_slope*time + ifa_intercept, color=color_line, linestyle=linestyle, lw=.5, label='{:.2f}'.format(ifa_slope))
  if print_slope:
    ax[0].text(.5, y_text, '$\chi_\mathrm{IFA}$'+'={:.2f} Hz/ms'.format(ifa_slope),  va=va_text, ha='center', transform = ax[0].transAxes,\
             fontsize=plt.rcParams["legend.fontsize"], backgroundcolor='w', zorder=0)
  
  # format
  xlim = [tw[0]-t0-tpad, tw[1]-t0+tpad]
  ax[0].set_ylabel('freq [Hz]')
  ax[0].set_xlim(xlim)
  ax[0].set_ylim([fmin, flim]) #np.max([flim, np.max(ifreq_all)*ymaxrel])])
  plt.setp(ax[0].get_xticklabels(), visible=False)
  
  despine(ax)
  
  if show_drive:
    ax[1].plot(time, Iext, color=my_colors['iext'])  
    ax[1].set_ylabel(str_drive)
    ax[1].set_xlabel('time [ms]')
    ax[1].set_xlim(xlim)
    ax[1].set_ylim(bottom=0)#
    ax[1].tick_params(which='minor', length=0)

    gridline([0], ax, 'x')
  if return_fig:
    return fig, ax
  else:
    return ax

#%% basic plotting components
def plot_raster(ax, spktrain, neuronview=[0,20], t0=0, ms=3):
  if not len(neuronview):
    N = len(spktrain.keys())
    neuronview = [0, N]
  for i in range(neuronview[0], neuronview[1]):
    ax.plot(spktrain[str(i)]-t0, np.ones(spktrain[str(i)].size)*i, 'k|', ms=ms)#, markeredgewidth=2)
  ax.set_ylabel('unit')
  ax.set_ylim([neuronview[0]-.5, neuronview[1]+.5])
  return ax
  
def plot_rate(ax, rate, dt, t0=0, label='rate \n[spk/s]', col=my_colors['fnet'], spines=False, linestyle='-', zorder=1):
  ax.plot(np.arange(rate.size)*dt-t0, rate, color=col, linestyle=linestyle, zorder=10)
  if label:
    ax.set_ylabel(label)
  if not spines:
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
  ax.set_ylim(bottom=0)
  return ax
  
def plot_wspec(fig, ax, wspec, wspec_extent, ifreq, ipower, dt, \
               ax_cb=None, ax_cb_inst=None, Pthr=0, maxpower=nan, t0=0, label=True, fmin=nan, \
               label_cbar='power \n[a.u.]', label_cbar_inst='power [a.u.]', show_ifreq=True, alpha=1, cmap=plt.cm.viridis, vmin=0):
  if np.isnan(maxpower):
    maxpower = np.max(wspec)
  norm = matplotlib.colors.Normalize(vmin=vmin, vmax=maxpower)
  im = ax.imshow(wspec, norm=norm, extent = (wspec_extent[0]-t0, wspec_extent[1]-t0, wspec_extent[2], wspec_extent[3]), origin='lower', aspect='auto',\
                 alpha=alpha, cmap = cmap)
  gridline(fmin, ax, 'y', zorder=2)
  ax.autoscale(False)
  t = np.arange(ifreq.size)*dt + wspec_extent[0] - t0 #np.arange(wspec_extent[0], wspec_extent[1], dt)-t0
  # plot inst freq with color graded according to inst. power
  if show_ifreq:
    fig, ax, ax_cb_inst = plot_multicolor_line(t, ifreq, ipower, fig, ax, cmap='gray', norm=norm,  ax_cbar=ax_cb_inst, cbar_label=label_cbar_inst)#, lw=2.5)
  if ax_cb:
    if maxpower<np.max(wspec):
      cb = fig.colorbar(im, cax=ax_cb, extend='max')
    else:
      cb = fig.colorbar(im, cax=ax_cb)
    if not len(label_cbar):
      cb.ax.set_yticklabels([])
    cb.ax.plot([0,10000],[np.mean(Pthr)]*2, 'r')#, lw=1.5) 
    cb.set_label(label=label_cbar, labelpad=-.1)
    if ax_cb_inst:
      ax_cb_inst.plot([0,10000],[np.mean(Pthr)]*2, 'r')#, lw=1.5) 
  if label:
    ax.set_ylabel('freq [Hz]')
  return fig, ax, ax_cb, ax_cb_inst

def plot_vhist(ax, ax_cb, Vthr, Vreset, v=[], Vdistr=[], vbinedges=[], v_recordwindow=[], volt_min=None, volt_max=None, t0=0, \
               show_colorbar=True, cbar_label=r'$p(v,t)$', cbar_max_relative=None, cbar_max=None, vbins=30, color_norm='linear'): # cbar_max_relative=3
  despine(ax)
  if not len(Vdistr):
    Vdistr, vbinedges = getVtraces('micro', v, Vthr=Vthr, volt_min=volt_min, volt_max=volt_max, density=True, vbins=vbins)
  if not cbar_max:
    if cbar_max_relative:
      cbar_max = np.max(Vdistr)*cbar_max_relative
    else:
      cbar_max = np.max(Vdistr)
  
  if color_norm=='linear':
    norm=matplotlib.colors.Normalize(vmin=0, vmax=cbar_max)
  elif color_norm=='log':
    print('log norm')
    vmin = np.min(Vdistr[Vdistr>0])
    norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=cbar_max)
    Vdistr[Vdistr<vmin] = vmin
  
  im = ax.imshow(Vdistr, origin='lower', extent=(v_recordwindow[0]-t0, v_recordwindow[-1]-t0, vbinedges[0], vbinedges[-1]), \
                 norm=norm, aspect='auto', cmap=colormap_voltage_hist, interpolation=None) # norm=matplotlib.colors.PowerNorm(gamma=.2, vmax=cbar_max)
  # mark threshold and reset
  gridline([Vthr, Vreset], ax, axis='y', zorder=100)
  ax.set_ylabel('$v$\n[mV]')
  ax.set_xlim([v_recordwindow[0]-t0, v_recordwindow[-1]-t0])
  
  if show_colorbar:
    if np.max(Vdistr) > cbar_max:
      cb = plt.colorbar(im, cax=ax_cb,  extend='max')
#      cb = plt.colorbar(im, cax=ax_cb,  extend='max', ticks=np.round(np.floor(np.linspace(0, cbar_max, 3)*10)/10, decimals=1)).set_label( label=cbar_label, labelpad=-.1)
    else:
      cb = plt.colorbar(im, cax=ax_cb)
    cb.set_label(label=cbar_label, labelpad=-.08)
    cb.ax.tick_params(axis='y', pad=1)
  return ax, ax_cb

def f_plotTraces(ax, v_hist, v_binedges, rate, dt, Vthr, Vreset, \
                 v_timewindow= [], rate_raw=[], xlim=[], ylim_rate=[], \
                 show_colorbar=True, ax_cb=None, cbar_label='$p(v,t)$', cbar_max_relative=None, cbar_max=None, t0=0):
  '''
  v_timewindow: [T0, T1] [ms]: for which time window was v_hist recorded (relevant if vhist comes from pypet sim with reduced trace memory)
  '''
  if not len(v_timewindow):
    if v_hist.shape[1] == rate.size:
      v_timewindow = [0, rate.size*dt]
    else:
      raise ValueError('Please provide v_timewindow, indicating in which time frame the voltage traces were recorded!')
  if not len(ylim_rate):
    if sum(rate[int(v_timewindow[0]/dt): int(v_timewindow[-1]/dt)]):
      ylim_rate = [0,1.1*np.max(rate[int(v_timewindow[0]/dt): int(v_timewindow[-1]/dt)])]
    else:
      ylim_rate = [0,50]
  if not len(xlim):
    xlim = [0, (rate.size+1)*dt]
  if len(rate_raw):
    ax[0].plot(np.arange(0, rate_raw.size)*dt, rate_raw, 'lightgray', lw=.5)

  ax[0] = plot_rate(ax[0], rate, dt, t0=t0)
  ax[0].set_ylim(ylim_rate)
  
  ax[1], ax_cb \
  = plot_vhist(ax[1], ax_cb, Vthr, Vreset, Vdistr=v_hist, vbinedges=v_binedges, v_recordwindow=v_timewindow, t0=t0,\
               show_colorbar=show_colorbar, cbar_label=cbar_label, cbar_max_relative=cbar_max_relative, cbar_max=cbar_max)
  ax[1].set_xlabel('time [ms]')
  plt.setp(ax[0].get_xticklabels(), visible=False)
  
  return ax, ax_cb

#%% Gaussian-drift approximation for simulation trajectory

def pypet_gaussian_drift_approx_stat(traj_hash, reset=True, save2file=True, dI = .1, \
                                     path_to_simulations = './simulations/'):
  ''' perform Gaussian-drift approx for all runs of a pypet simulation trajectory
  '''
  print('Gaussian-drift approximation for constant drive, for all runs of simulation trajectory h{}..'.format(traj_hash), end='')
  traj = pypet_load_trajectory(traj_hash = traj_hash, path_to_simulations = path_to_simulations)  
  datapath = pypet_get_trajectoryPath(traj_hash = traj_hash, path_to_simulations=path_to_simulations) + 'gaussian_drift_approx/'
  if not os.path.exists(datapath):
    os.makedirs(datapath)
    
  # load results and extract dimensionless parameters
  res = traj.results.summary.scalarResults
  Iext_nA = np.sort(res['parameters.input.level']) 
  try:
    IPSPint = traj.IPSPint
  except:
    IPSPint = traj.derived_parameters.runs.run_00000000.IPSPint
    
  K, D, _, Vr = rescale_params_dimless(traj.Vthr, traj.Vreset, traj.E_rest, IPSPint, Iext_nA, traj.Ie_sig, traj.Nint, traj.C, traj.tm)
  
  # Ihopf = get_hopf(D, traj.tl, K, traj.tm, exc=exc, gauss_extent=gauss_extent)
  Ifull = get_pt_fullsynch(D, traj.tl, K, traj.tm)
  Iext = np.arange(dI, Ifull+dI, dI)

  df_A = pd.DataFrame(columns= ['Iext', 'mu_min', 'mu_max', 'mu_reset', 'zeta', 't_on', 't_off'])
  df_A['Iext'] = Iext

  df_A['mu_min'], df_A['mu_max'], df_A['f_net'], df_A['f_unit'], df_A['sat'], df_A['zeta'], df_A['t_on'], df_A['t_off'], df_A['mu_reset'] \
  = gaussian_drift_approx_stat(Iext, D, traj.tl, K, traj.tm, Vr=Vr, reset=reset)[:-2]

  if save2file:
    df_A.to_hdf(datapath+'gaussian_drift_approx_analytical_reset-{}_h{}'.format(reset, traj.hash)+'.hdf5',key='res',format='table', data_columns=True)
   
  print('[done]')
  return


#%% Analysis of membrane potential dynamics in simulation for comparison with Gaussian-drift approximation

def pypet_analyse_membrane_potential_dynamics(traj_hash, mse_max=.5, plot=True, path_to_simulations='./simulations/'):
  ''' 
  analyse the membrane potential dynamics of all runs of a pypet simulation trajectory
  check how "gaussian" the distribution is over time
  '''
  print('Analysis of membrane potential dynamics in spiking network simulation h{} for comparison with Gaussian-drift approximation...'.format(traj_hash), end='')
  traj = pypet_load_trajectory(traj_hash=traj_hash, path_to_simulations=path_to_simulations)
  
  datapath = pypet_get_trajectoryPath(traj_hash, path_to_simulations=path_to_simulations) + 'analysis_membrane_potential_dynamics/'
  if not os.path.exists(datapath):
    os.makedirs(datapath)
  # initialize data frame with overview of explored parameters:
  df = pypet_get_exploration_overview(traj)
  # add columns for analysis results
  for col in ['mu_min', 'mu_max', 'sigma_min', 'sigma_max', 'zeta', 't_on', 't_off', 'r0', 'I0', 'v_std_mean', 'v_std_std']:
    df[col] = nan
  # do analysis of gauss features
  nruns = len(traj.f_get_run_names())
  for idx in range(nruns):
    traj.v_idx=idx 
    df.loc[idx, 'Iext'] = traj.level
    # add analysis results:
    df.loc[idx, 'mu_min'] = traj.results.crun.network.v_av_mumin
    df.loc[idx, 'sigma_min'] = traj.results.crun.network.v_av_sigmin
    df.loc[idx, 't_on'] = traj.results.crun.gauss.t_on
    df.loc[idx, 't_off'] = traj.results.crun.gauss.t_off
    # df.loc[idx, 'zeta'] = get_popsynch(traj.results.crun.gauss.t_on, traj.results.crun.gauss.t_off, \
    #                                     traj.results.crun.network.saturation, traj.results.crun.network.freq_net)
      

    df.loc[idx, ['v_std_mean', 'v_std_std']] = traj.results.crun.network.v_std_mean, traj.results.crun.network.v_std_std
    # characterize AI state in case no oscillation was detected
    if traj.results.crun.gauss.gauss_fit_params.shape[0]==1:
      try:
        IPSPint = traj.IPSPint
      except:
        IPSPint = traj.derived_parameters.runs.run_00000000.IPSPint
      df.loc[idx, ['r0', 'I0']] = get_stationary_rate_lif(traj.level*traj.tm/(traj.C/1000), -IPSPint*traj.Nint, traj.Vthr-traj.E_rest, traj.Vreset-traj.E_rest,\
                                                  traj.input.Ie_sig*traj.tm/(traj.C/1000), traj.tm, traj.tref, plot=False)[:2] 
    
    # plot
    if plot:
      D = .5*(traj.input.Ie_sig*traj.tm/(traj.C/1000))**2 # mV^2
      fig = plot_analysis_membrane_potential_dynamics(traj.results.crun.raw.v, traj.results.crun.raw.LFP_smooth, traj.results.crun.gauss.popspk_av, \
                                      traj.results.crun.gauss.v_av, traj.results.crun.gauss.v_std, traj.results.crun.gauss.v_hist, \
                                      traj.results.crun.gauss.v_bincenters, df.loc[idx,'mu_min'], traj.results.crun.gauss.gauss_fit_params, \
                                      traj.results.crun.gauss.sample, traj.results.crun.network.freq_net, traj.level, traj.dt, traj.Vthr, \
                                      traj.Vreset, traj.E_rest, D, traj.tm, traj.postproc.v_recordwindow, \
                                      r0 = df.loc[idx,'r0'], I0=df.loc[idx,'I0'])
      
      # fig, ax_trace, ax_fit, ax_sum = plot_gauss_analysis_4lif(traj.results.crun.raw.v, traj.dt, traj.results.crun.raw.LFP_smooth, traj.Vthr, traj.Vreset, traj.E_rest, D, traj.tm, \
      #                                traj.results.crun.gauss.sample, traj.results.crun.gauss.gauss_fit_params, traj.results.crun.gauss.v_hist, \
      #                                traj.results.crun.gauss.v_bincenters, df.loc[idx,'mu_min'], df.loc[idx,'mu_max'], df.loc[idx,'r0'], df.loc[idx,'I0'], traj.postproc.v_recordwindow)
      # ax_trace[0].set_title(r'$I_\mathrm{E}=$'+'{:.2f}nA'.format(traj.level))
      fig.savefig(datapath+'membrane_potential_dynamics_h{}_run{}.pdf'.format(traj.hash, idx), bbox_inches='tight')
  
  # store data frame in readable table format in the same folder as trajectory
  df.to_hdf(datapath+'analysis_membrane_potential_dynamics_h{}'.format(traj.hash)+'.hdf5', key='df', format='table', data_columns=True)
  
  # add-on for simplified gauss approximation:
  if plot:
    D = .5*(traj.input.Ie_sig*traj.tm/(traj.C/1000))**2 # mV^2
    fig, ax = plt.subplots()
    ax.axhline(np.sqrt(D), lw=1, color='gray')
    df.plot(x='level', y='v_std_mean', yerr= 'v_std_std', ax=ax, marker='o', color='k', legend=False)
    add_ticks(ax, [np.sqrt(D)],[r'$\sqrt{D}$'], 'y')
    ax.set_xlabel(r'$I_\mathrm{E}$ [nA]')
    ax.set_ylabel('std(V) [mV]')
    ax.set_xlim(right=1.1*np.max(df.level))
    fig.tight_layout()
    fig.savefig(datapath+'voltage_standard_deviations_h{}.pdf'.format(traj.hash), bbox_inches='tight')
  
  traj.f_restore_default() 
  print('[done]')
  return

def plot_analysis_membrane_potential_dynamics(v, LFP_smooth, popspk_av, v_av, v_std, hist, hist_bincenters, mu_min, fit_params, sample, f_net, Iext, \
                             dt, Vthr, Vreset, El, D, tm, v_timewindow, \
                             r0 = nan, I0=nan, nrows=3, ms=3):
  '''
  plot the result of the gauss analysis for an LIF unit, i.e. the extraction of the "average" cyclostationary cycle in terms of
  - population rate
  - mean membrane potential
  - standard deviation of membrane potential

  Parameters
  ----------
  v : mV, time x neurons
    membrane potentials over time (can be just the last 25ms of the simulation).
  LFP_smooth : Hz
    population rate (smoothed)
  popspk_av : Hz, size n
    average population spike over one cycle (sample size n).
  v_av : mV, size n
    average membrane potential over one cycle (sample size n).
  v_std : mV
    average std from mean membrane potential over one cycle (sample size n).
  hist : density
    histogram of membrane potentials over one cycle (sample size n)..
  hist_bincenters : mV
    bincenters to plot hist over.
  mu_min : mV
    estimate for mu_min: minimum of v_av.
  fit_params : 
    result of the fitting of a gauss to hist (mean, std, mse).
  sample : 
    indices of all the time points used as samples for the cycle-averages
  f_net : Hz
    network frequency estimated from PSD.
  Iext : nA
    external drive of this simulation.
  dt : ms
    simulation time step.
  Vthr : mV
    threshold.
  Vreset : mV
    reset.
  El : mV
    resting/leak potential.
  D: [mV] variance of unbounded membrane potential 
  tm: [ms] membrane time constant
  v_timewindow : ms, list size 2
    beginning and end of the time window for which voltages v are given
  r0 : Hz
    Stationary population rate in case of non-oscillatory AI regime. The default is nan.
  I0 : mV
    Total input in case of non-oscillatory AI regime. The default is nan.
  nrows : optional
    Number of rows in the plot of the voltage histograms. The default is 3.

  Returns
  -------
  fig 
  '''
  from methods_analyze_simulations import f_plotTraces
  # recover additional parameters
  n = fit_params.shape[0] # number of steps in ripple cycle
  oscillatory = n!=1
  
  if n < nrows: # happens in AI state
    nrows = 1
    
  # if oscillatory:
  #   argmin = (np.argmin(np.abs(v_av - mu_min)))%(n-1)
  # else:
  #   argmin = 0
    
  # --- plot
#  with plt.rc_context({"xtick.labelsize": 12, "ytick.labelsize": 12,"font.size": 12}):
  col = plt.cm.Wistia(np.linspace(1,0,n)) # plt.cm.cool(np.linspace(0,1,n)) # sequential colormap starting at argmin
  # --- construct figure
  if oscillatory:
    rc_context = {"axes.labelsize": 7, "axes.titlesize": 7, "legend.fontsize": 5,"font.size": 7,\
                       "xtick.labelsize": 5, "ytick.labelsize": 5}
  else:
    rc_context={}
  with plt.rc_context(rc_context):
    if oscillatory:
      fig = plt.figure(figsize=(width_a4_wmargin, width_a4_wmargin*.6))#, constrained_layout=True)
      gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[5,2], height_ratios=[1,1], hspace=.5, wspace=.4)
      
      gs_trace = gs[0,0].subgridspec(3,1, height_ratios=[3,3,1])
      ax_trace=[None]*3
      ax_trace[0] = fig.add_subplot(gs_trace[0,:])
      ax_trace[1] = fig.add_subplot(gs_trace[1,:], sharex=ax_trace[0])
      ax_trace[2] = fig.add_subplot(gs_trace[2,:], sharex=ax_trace[0])
      ax_trace[0].text(-.15, 1.1, string.ascii_uppercase[0], transform=ax_trace[0].transAxes, size=panel_labelsize, weight='bold')
      ax_trace[0].text(-.15, -3, string.ascii_uppercase[1], transform=ax_trace[0].transAxes, size=panel_labelsize, weight='bold')
      
      gs_hist = gs[1,:].subgridspec(nrows,n//nrows, hspace=.3)
      ax_fit = [None]*n
      for i in range(n):
        if not i:
          ax_fit[i] = fig.add_subplot(gs_hist[i//int(n/nrows),i%int(n/nrows)])
        else:
          ax_fit[i] = fig.add_subplot(gs_hist[i//int(n/nrows),i%int(n/nrows)], sharex=ax_fit[0], sharey=ax_fit[0])
#    if oscillatory:
      gs_av = gs[0,1].subgridspec(3,2, height_ratios=[3,3,1], width_ratios=[20,.5], wspace=.05)
      ax_av = [None]*3
      ax_av[0] = fig.add_subplot(gs_av[0,0], sharey = ax_trace[0])
      ax_av[1] = fig.add_subplot(gs_av[1,0], sharex = ax_av[0], sharey = ax_trace[1])
      ax_av[2] = fig.add_subplot(gs_av[2,0], sharex = ax_av[0], sharey = ax_trace[2])
      ax_av_cbar = fig.add_subplot(gs_av[1,1])
      ax_av[0].text(-.4, 1.1, string.ascii_uppercase[2], transform=ax_av[0].transAxes, size=panel_labelsize, weight='bold')
      
      
    else:
      fig = plt.figure(figsize=(width_a4_wmargin, width_a4_wmargin*.3))#, constrained_layout=True)
      gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[4,2], wspace=.4)
      
      gs_trace = gs[0,0].subgridspec(3,1, height_ratios=[3,3,1])
      ax_trace=[None]*3
      ax_trace[0] = fig.add_subplot(gs_trace[0,:])
      ax_trace[1] = fig.add_subplot(gs_trace[1,:], sharex=ax_trace[0])
      ax_trace[2] = fig.add_subplot(gs_trace[2,:], sharex=ax_trace[0])
      ax_trace[0].text(-.17, 1.1,  string.ascii_uppercase[0], transform=ax_trace[0].transAxes, size=panel_labelsize, weight='bold')
      ax_trace[0].text(1.15, 1.04, string.ascii_uppercase[1], transform=ax_trace[0].transAxes, size=panel_labelsize, weight='bold')
    
      ax_fit = [fig.add_subplot(gs[0,1])]
      despine(ax_fit[0])
     
    
    # --- fill figure
    ## --- traces 
    v_hist, v_binedges = getVtraces('micro', v, Vthr=Vthr, volt_max = Vthr, vbins=100, density=True) # vbins=hist.shape[1]
    ax_trace, im = f_plotTraces(ax_trace, v_hist, v_binedges, LFP_smooth, dt, Vthr, Vreset, v_timewindow = v_timewindow, show_colorbar=False,\
                                t0 = v_timewindow[0])
    t0 = v_timewindow[0]
    time = np.arange(0, LFP_smooth.size)*dt - t0
    # v_av = np.mean(v, axis=1) # average voltage
    if oscillatory:
      for i in range(n):
        # j = i#(i-argmin)% (n-1)
        t = time[sample[i,:]] # time points
        ax_trace[1].plot(t, v_av[i]*np.ones(t.size), '.', markersize = ms, color=col[i]) 
      # ax_trace[1].plot(time[sample[sample[:,i]<time.size,i]], theta_av[sample[sample[:,i]<time.size,i]], 'o', color=col[j])
    if not np.isnan(r0): # AI state
      ax_trace[0].axhline(r0, color='k', linestyle='--') #, label='$r_0$={:.2f}Hz'.format(r0))
      ax_trace[0].legend(loc='upper right', borderaxespad=0, borderpad=0)
    ax_trace[0].set_title(r'$I_\mathrm{ext}=$'+'{:.2f}nA'.format(Iext))
    ax_trace[0].set_xlim(right = ax_trace[0].get_xlim()[1]-5) # so last cycle is more or less cut off
    
    ax_trace[-1].axhline(np.sqrt(D), color='k', linestyle='--')
    ax_trace[-1].plot(np.arange(v_timewindow[0],  v_timewindow[-1], dt) - t0, np.std(v, axis=1), 'k')
    plt.setp(ax_trace[1].get_xticklabels(), visible=False)
    ax_trace[1].set_xlabel('')
    ax_trace[-1].set_ylabel('SD(v)')
    ax_trace[-1].set_xlabel('time [ms]')
    ax_trace[-1].spines['right'].set_visible(False)
    ax_trace[-1].spines['top'].set_visible(False)
#    ax_trace[0].set_xticks(list(np.arange(v_timewindow[0], v_timewindow[1], 5)))
    
    ## --- histogram
    gauss = np.zeros(hist.shape)
    for i in range(n):
      if oscillatory:
        frame_subplot(ax_fit[i], color=col[i], lw=1)
      gauss[i,:] = get_gauss(hist_bincenters, fit_params[i,0], fit_params[i,1]) # consider plotting against separate phase vector?
      plot_gauss_fit(hist[i,:], gauss[i,:], fit_params[i,2], hist_bincenters, Vthr, Vreset, ax=ax_fit[i], show_legend= i == 6, \
                     axlabel= (i==14) or (not oscillatory), show_fit = oscillatory)
      # plot_gauss_fit(hist[i,:], gauss[i,:], fit_params[i,2], hist_bincenters, Vthr, Vreset, ax=ax_fit[ix], show_legend= ix == n-1, \
      #                xlabel= ix==n-1, show_fit = oscillatory)
      
    if not oscillatory:
      p0 = get_stationary_fpe_lif(hist_bincenters-El, r0, I0, Vthr-El, Vreset-El, D, tm)
      ax_fit[0].plot(hist_bincenters, p0, 'k', label='$p_0(v)$')
      ax_fit[0].legend(loc='upper left')
      ax_fit[0].set_xticks([Vreset, Vthr])
      ax_fit[0].set_xticklabels([r'$V_\mathrm{reset}=$'+str(Vreset), r'$V_\mathrm{thr}=$'+str(Vthr)])
      ax_fit[0].set_ylabel('$p_0(v)$')
      # ax_fit[0].set_xlabel('V [mV]')
      # ax_fit[0].set_ylabel('p(V)')
    else:
      p_max = np.max(hist[:, hist_bincenters > Vreset + np.mean(np.diff(hist_bincenters))])
      ax_fit[0].set_ylim([0, 1.1*p_max])
    
    ## --- average cycle
    if oscillatory:
      period = 1000/f_net
      t_samples = np.linspace(0,1,n+1, endpoint=True)*period
      dt_samples = np.mean(np.diff(t_samples))
      popspk_av_pad = np.append(popspk_av, popspk_av[0])
      hist_pad = np.vstack((hist, hist[0,:]))
      v_std_pad = np.append(v_std, v_std[0])
      dv = np.mean(np.diff(hist_bincenters))
      ax_av[0].plot(t_samples, popspk_av_pad, '.-', color=my_colors['fnet'])
      ax_av[0].set_ylabel('rate\n[spk/s]')
      ax_av[0].spines['right'].set_visible(False)
      ax_av[0].spines['top'].set_visible(False)
      ax_av[0].set_title('average cycle')
      plt.setp(ax_av[0].get_xticklabels(), visible=False)
      plt.setp(ax_av[1].get_xticklabels(), visible=False)
      
      cbar_max = np.max(hist_pad)*.8
      im = ax_av[1].imshow(hist_pad.T, origin='lower', extent=[-dt_samples/2, period+dt_samples/2, hist_bincenters[0]-dv/2, hist_bincenters[-1]+dv/2],\
                  aspect='auto', cmap=my_colors['vhist'], norm=matplotlib.colors.Normalize(vmin=0, vmax=cbar_max),\
                  interpolation='none')
      for i in range(n+1):
        j = i#(i-argmin)% (n-1)
        ax_av[1].plot(t_samples[i], v_av[i%n], '.', markersize = ms, color=col[j%n]) 
      ax_av[-1].plot(t_samples, v_std_pad, 'k.-')  
      ax_av[-1].axhline(np.sqrt(D), color='k', linestyle='--')
      ax_av[-1].set_ylim([.6*np.min([np.sqrt(D), np.min(v_std_pad)]),  1.2*np.max([np.sqrt(D), np.max(v_std_pad)])])
      ax_trace[-1].set_ylim([.6*np.min([np.sqrt(D), np.min(v_std_pad)]),  1.2*np.max([np.sqrt(D), np.max(v_std_pad)])])
      ax_av[-1].spines['right'].set_visible(False)
      ax_av[-1].spines['top'].set_visible(False)
      ax_av[1].set_ylabel('v [mV]')
      ax_av[-1].set_ylabel('SD(v)\n')
      ax_av[-1].set_xlabel('time [ms]')
      # colorbar
  #      cbar_max = np.floor(100*np.max(hist)*.75)/100
      fig.colorbar(im, cax= ax_av_cbar, orientation='vertical' ).set_label(label='$p(v,t)$') #ticks=[0, cbar_max/2, cbar_max]
        
  return fig

def plot_gauss_fit(hist, fit, mse, bin_centers, Vthr, Vr, color='k', ax=None, show_legend=True, axlabel=True, show_fit=True):
  '''
  fit: calculated for same phase as hist (so no edges 0, 2pi!) (maybe think about better implementation?)
  '''
  if not ax:
    fig, ax = plt.subplots()
  # ax.plot(phase, hist, color='gray', label='hist')
  ax.bar(bin_centers, hist, facecolor='gray', color= 'gray', align='center', width = np.mean(np.diff(bin_centers)), label='hist')
  # ax.fill_between(phase, hist, facecolor='lightgray', edgecolor= 'lightgray', label='hist')
  if show_fit:
    ax.plot(bin_centers, fit, color=color, label='fit', lw=1)
    ax.set_title('MSE: {:.0f} x'.format(np.round(mse*1e6)) + '$10^{-6}$', loc='right', pad=-.1, fontsize=plt.rcParams['legend.fontsize'])
  if axlabel:
    ax.set_xticks([Vr, Vthr])
    # ax.set_xticklabels(['$V_R=${}'.format(Vr), '$V_T=${}'.format(Vthr)])
    ax.set_xlabel('v [mV]')
    ax.set_ylabel('p(v)')
      
  else:
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    # ax.set_yticks([])
  ax.set_xlim([bin_centers[0]-np.mean(np.diff(bin_centers))/2, Vthr])
  ax.set_ylim(bottom=0)
  if show_legend:
    ax.legend(handlelength=1, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
  return 

#%% Gaussian-drift approx vs simulation
def performance_check_evaluation(traj_hash, Vt=1, reset=True, numerical_check=False, path_to_simulations='/simulations/'):
  
  print('\n\nPerformance check for Gaussian-drift approx under constant drive (for Figs S1A, S1B)...')
  
  from methods_simulations import recover_exploration, map_pstr, get_pfix_str
  
  traj = pypet_load_trajectory(traj_hash = traj_hash, path_to_simulations=path_to_simulations) 
  datapath = 'results/gaussian_drift_approx_constant_drive_performance_check/'
  if not os.path.exists(datapath):
    os.makedirs(datapath)
    
  df_runs = pd.read_csv(datapath+'df_parameters_per_run_h{}.csv'.format(traj_hash), index_col=0, squeeze=True)
  df_net = pd.read_csv(datapath+'df_network_configurations_h{}.csv'.format(traj_hash), index_col=0, squeeze=True)
  df_net['Delta'] = df_net.tl
  print('The following network configurations were explored:')
  print(df_net)
  # from the run configurations extract the exploration dictionary and lists of fixed, and explored parameters:
  exploration, p_fix, p_var = recover_exploration(df_runs)  

  exploredParams_full = list(traj.f_get_explored_parameters()) # full parameter names (tree structure in names)
  exploredParams = pypet_shortcut(list(traj.f_get_explored_parameters())) # shortcut names (final leaf of parameter tree)
  exploredParams_sim = list(set(exploredParams)-set(['level'])) # all explored params EXCEPT input level
  
  # sanity check:
  if not set(map_pstr(exploredParams_sim, unit=False)) == set(p_var):
    raise ValueError('Simulation explored parameters {}, not {}!'.format(exploredParams_sim, p_var))
  # load simuation results  
  sim = traj.results.summary.scalarResults  
  sim = sim.sort_index()
  sim.index = np.arange(len(sim)).astype(int)
  
  # rename columns by short names
  pars_label = {}
  for par in list(exploredParams_full):
      pars_label[par] = par.split('.')[-1]
  sim.rename(columns=pars_label, inplace=True)
  
  # should deprecate soon, I forgot to store scalar results of gauss analysis in traj.results.summary.scalarResults
  if traj.analysis.v_dynamics and ('v_av_mumin' not in sim.columns):
    print('Adding mumin estimate from spiking simulation to summary df sim')
    sim[['v_av_mumin', 'v_av_mumin_sd', 'v_av_sigmin']] = np.nan
    for i in range(len(sim)):
      traj.v_idx = i
      sim.iloc[i]['v_av_mumin'] = traj.results.crun.v_av_mumin
      sim.iloc[i]['v_av_mumin_sd'] = traj.results.crun.v_av_mumin_sd
      sim.iloc[i]['v_av_sigmin'] = traj.results.crun.v_av_sigmin
  else:
    print('No mumin estimate from spiking simulation available (or already stored in scalarResults). Delete above code passage!')
  
  if 'v_av_mumin' in sim.columns:
    # rescale to dimless voltage
    sim['gauss_mu_min_sim'] = (sim['v_av_mumin']-traj.E_rest)/(traj.Vthr-traj.E_rest)
    sim['gauss_mu_min_sd_sim'] = (sim['v_av_mumin_sd']-traj.E_rest)/(traj.Vthr-traj.E_rest)
    sim['gauss_sig_min_sim'] = (sim['v_av_sigmin']-traj.E_rest)/(traj.Vthr-traj.E_rest)
    
    # print(sim['gauss_mu_min_sim'])
  
  if not np.isclose(sim[exploredParams], df_runs[exploredParams]).all().all():
    raise ValueError('Mismatch between data frames df_runs and sim!')
  
  # merge the two data frames into one:
  if 'v_av_mumin' in sim.columns:
    df_runs = pd.concat([df_runs, sim[['freq_net', 'freq_unit_mean', 'freq_unit_std', 'saturation', 'CV', 'ampl_mean', 'qfac', \
                                       'v_std_mean', 'v_std_std', 'v_av_mumin', 'v_av_mumin_sd', 'gauss_mu_min_sim', \
                                       'gauss_mu_min_sd_sim', 'gauss_sig_min_sim']]], axis=1)
  else:
    df_runs = pd.concat([df_runs, sim[['freq_net', 'freq_unit_mean', 'freq_unit_std', 'saturation', 'CV', 'ampl_mean', 'qfac', \
                                       'v_std_mean', 'v_std_std']]], axis=1)
    
  # --- Gaussian-drift approximation for all runs:
  # rescale external drive to dimless voltage:
  df_runs['Iext'] = rescale_params_dimless(traj.Vthr, df_runs.Vreset.values, traj.E_rest, df_runs.IPSPint.values, df_runs.level.values, \
                                      df_runs.Ie_sig.values, traj.Nint, traj.C, df_runs.tm.values)[2]
  df_runs['gauss_mu_min'], df_runs['gauss_mu_max'], df_runs['gauss_f_net'], df_runs['gauss_f_unit'], df_runs['gauss_sat'], df_runs['gauss_zeta'], \
  df_runs['gauss_t_on'], df_runs['gauss_t_off'], df_runs['gauss_mu_reset'] \
  = gaussian_drift_approx_stat(df_runs.Iext.values, df_runs.D.values, df_runs.tl.values, df_runs.K.values, df_runs.tm.values, Vr=df_runs.Vr.values, \
                               reset=reset)[:-2]
  
  if numerical_check:
    # compare the analytical Gaussian-drift approx to numerical integration of the DDE
    df_runs['gauss_mu_min_num'], df_runs['gauss_mu_max_num'], df_runs['gauss_f_net_num'], df_runs['gauss_f_unit_num'], \
    df_runs['gauss_zeta_num'], df_runs['gauss_t_on_num'], df_runs['gauss_t_off_num'], df_runs['gauss_mu_reset_num'] \
    = gaussian_drift_approx_stat_numerical(df_runs.Iext.values, df_runs.D.values, df_runs.tl.values, df_runs.K.values, df_runs.tm.values, \
                                       Vr = df_runs.Vr.values, reset=reset)[:-2]
    
  n_config = int(df_runs.config.max() + 1) # number of parameter configurations
  theory_interp = {} # dict of dicts: interpolated theory curves for plotting
  # loop over parameter configurations
  for c in range(n_config):
    # --- take network frequency of first run for this network (at critical input level) as simulated frequency at bifurcation
    df_net.loc[c,'fcrit_sim'] = df_runs.loc[df_runs.config==c, 'freq_net'].values[0]
    
    # --- theoretical freq in pt of full synch
    df_net.loc[c, 'fnet_full_theory'] \
    = gaussian_drift_approx_stat(df_net.loc[c, 'Ifull'], df_net.loc[c, 'D'], df_net.loc[c, 'tl'], df_net.loc[c, 'K'], df_net.loc[c, 'tm'], \
                                 Vr=df_net.loc[c, 'Vr'], reset=reset)[2]
    
    # --- estimate point of full synch in simulation
    df_net.loc[c,'Ifull_nA_sim'], df_net.loc[c,'fnet_full_sim'] \
    = find_full_synch_interpolated(df_runs.loc[df_runs.config==c, 'level'].values, df_runs.loc[df_runs.config==c, 'saturation'].values, df_runs.loc[df_runs.config==c, 'freq_net'].values)
    df_net.loc[c,'Ifull_sim'] = df_net.loc[c,'tm']/(traj.C/1000)*df_net.loc[c,'Ifull_nA_sim']/(traj.Vthr-traj.E_rest) 
    
    # --- errors between simulation and theory
    df_net.loc[c, 'error_net_nmae'], df_net.loc[c, 'error_unit_nmae'], df_net.loc[c, 'Imin_theory'], df_net.loc[c, 'Imax_theory'], theory_interp[c] \
    = get_error_gauss_drift_approx(df_runs.loc[df_runs.config==c, 'Iext'].values, df_runs.loc[df_runs.config==c, 'freq_net'].values, \
                                   df_runs.loc[df_runs.config==c, 'freq_unit_mean'].values, df_net.loc[c,'Ifull_sim'], df_net.loc[c,'Ifull'], \
                                   df_net.loc[c, 'D'], df_net.loc[c, 'tl'], df_net.loc[c, 'K'], df_net.loc[c, 'tm'], df_net.loc[c, 'Vr'], \
                                   reset=reset)
  
  # --- region where theory applies (fraction within [0,1])
  df_net['applicability'] \
  = (df_net[['Imax_theory', 'Ifull_sim']].min(axis=1)-df_net['Imin_theory']).clip(lower=0)/(df_net['Ifull_sim']-df_net['Icrit'])

  df_net['performance'] = df_net['applicability']*(1-df_net['error_net_nmae'])
  
  # nan for fnet_full_theory, if Ifull > Imax_theory
  df_net.loc[df_net.Ifull > df_net.Imax_theory, 'fnet_full_theory'] = np.nan
  df_net.loc[df_net.applicability.isna(), 'fnet_full_theory'] = np.nan
  
  
  # store results
  df_net.to_csv(datapath+'df_network_configurations_h{}_evaluated.csv'.format(traj_hash))
  df_runs.to_csv(datapath+'df_parameters_per_run_h{}_evaluated.csv'.format(traj_hash))  
  
  traj.f_restore_default()
  print('[done]')
  return 

def get_error_gauss_drift_approx(Iext, fnet_sim, funit_sim, Ifull_sim, Ifull_theory, D, Delta, K, tm, Vr, \
              dI=0.1, reset=True, Vt=1):
  # interpolate up to whatever is smaller: the simulated or the theoretically predicted pt of full synch
  Imax = np.nanmax([Ifull_sim, Ifull_theory])
  if Imax < Iext[0]+dI:
    return nan, nan, 1e10, {}
  else:
    # --- interpolate simulation results up to pt of full synch  
    Iext_interp = np.arange(Iext[0], Imax+dI, dI)
    fnet_sim_interp = np.interp(Iext_interp, Iext, fnet_sim)
    funit_sim_interp = np.interp(Iext_interp, Iext, funit_sim)
    
    # --- calculate theoretical approximation at that resolution
    mu_min_theory, mu_max_theory, fnet_theory, funit_theory, _, _, _, t_off, mu_reset_theory \
    = gaussian_drift_approx_stat(Iext_interp, D, Delta, K, tm, Vr=Vr, reset=reset)[:-2]
    
    # --- keep interpolated theory curves for plotting
    theory_plots = {'Iext': Iext_interp, 'fnet':fnet_theory, 'funit':funit_theory, 'mu_max':mu_max_theory, 'mu_min':mu_min_theory, 'mu_reset':mu_reset_theory}
  
    # --- determine beginning of region where theory applies:
    # --- criterion now: mumin+3sqrt(D) must be below threhold 
    # --- before I used: popspk must be longer than delay
    # based on the code for calculating t_off, these 2 conditions should be the same:
    ix_theory_applies = np.where(mu_min_theory + 3*np.sqrt(D) <= Vt)[0]
    if not len(ix_theory_applies):
      return nan, nan, nan, nan, theory_plots
    # check that region where theory applies is one piece (no intermediate "holes")
    if (np.diff(ix_theory_applies) != 1).any():
      raise ValueError('I expected that the region where theory applies is one coherent piece in Iext-space!')
    Imin_theory = Iext_interp[ix_theory_applies[0]] 
    Imax_theory = np.min([Iext_interp[ix_theory_applies[-1]], Ifull_theory])
    
    
    # --- calculate error (normalized mean absolute error)
    # make sure no simulated nan values are included:
    ix0 = ix_theory_applies[0]
    ix1 = np.where(Iext_interp <= np.nanmin([Ifull_sim, Imax_theory]))[0][-1] + 1
    no_nan = np.isnan(fnet_sim_interp[ix0:ix1]) == False
    nMAE_net = np.mean(np.abs(fnet_sim_interp[ix0:ix1][no_nan] - fnet_theory[ix0:ix1][no_nan]) / fnet_sim_interp[ix0:ix1][no_nan] )
    nMAE_unit= np.mean(np.abs(funit_sim_interp[ix0:ix1] - funit_theory[ix0:ix1])) / np.mean(funit_sim_interp[ix0:ix1])
        
    return nMAE_net, nMAE_unit, Imin_theory, Imax_theory, theory_plots