#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 17:28:22 2023

This file contains all code used to 
* setup the parameter configuration files for simulations
* run simulations (see Class RippleNetwork and pypet wrapper function pypet_run_simulation)
* perform analyses of the network dynamics which are done immediately after simulation and stored in the same result container

To run simulations run main_run_simulations.py

@author: natalie
"""
import brian2 as b2
import json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import pandas as pd 
from pypet import Environment, cartesian_product, Trajectory 
import random
from scipy.signal import hilbert
import scipy
import time as Time
from tools import convert_numpy_2_python, despine, dict2str,  findPeaks,  get_amplitude,  get_gauss, get_PSD, list2str,  lorentzian

### aux functions
from methods_analytical import get_pt_fullsynch, linear_stability_analysis

pi=np.pi
nan = np.nan

mV, ms, nA, pF, nS, Hz, infty = b2.mV, b2.ms, b2.nA, b2.pF, b2.nS, b2.Hz, b2.infty
TimedArray = b2.TimedArray
SpikeMonitor, StateMonitor, PopulationRateMonitor = b2.SpikeMonitor, b2.StateMonitor, b2.PopulationRateMonitor 
Synapses, Network, NeuronGroup, PoissonGroup = b2.Synapses, b2.Network, b2.NeuronGroup, b2.PoissonGroup


my_rcParams = json.load(open('settings/matplotlib_my_rcParams.txt'))
matplotlib.rcParams.update(my_rcParams)

my_colors = json.load(open('settings/my_colors.txt'))

cm=1/2.54
width_a4 = 21*cm
width_a4_wmargin = width_a4 - 2*2*cm
panel_labelsize = 9

#%% Parameter house keeping

def loadStandardParams(parent, use='normal', dropna=True):
  pars = pd.read_csv('./settings/StandardParams_all_values.csv', index_col='parent') 
  if use=='normal':
    pars_dict = pars.loc[parent].dropna().to_dict()
    pars_dict['parent'] = parent # keep track of parent
    return pars_dict
  elif use=='pypet':
    if dropna:
      pars_val = pars.loc[parent].dropna() # drop nan parameters 
    else:
      pars_val = pars.loc[parent]      
    pars_val = pars_val.append(pd.Series([parent], index=['parent'])) # keep track of parent
    pars_meta = pd.read_csv('./settings/StandardParams_all_meta.csv', index_col='metadata')  # load meta data
    a = pars_meta.loc[:,pars_val.index] # keep only meta data for nonan-parameters, if nans were dropped
    b = pars_val.rename('value')
    pars_df = a.append(b) # pandas data frame: columns: parameter names, index: value and metadata
    return pars_df 
  
def convert_coupling(x, tm, Nint, Vthr, El, p_ii=1, unit=True, default=False):
  if default:
    stp = loadStandardParams('mymicro')
    p_ii, Nint = stp['p_ii'], stp['Nint']
  if unit:
    return x/Nint/p_ii*(Vthr-El)*tm # mVms
  else:
    return x*Nint*p_ii/(Vthr-El)/tm # dimless voltage

def convert_input(x, tm, C, Vthr, El, unit=True):
  if unit:
    return x/tm*(C/1000)*(Vthr-El) # nA
  else:
    return tm/(C/1000)*x/(Vthr-El)

def rescale_params_dimless(Vthr, Vreset, El, IPSPint, Iext_nA, Ie_sig, Nint, C, tm, default_coupling=False):
  Vr = (Vreset-El)/(Vthr-El)
  K = convert_coupling(IPSPint, tm, Nint, Vthr, El, unit=False, default=default_coupling) 
  D = .5*(tm/(C/1000)*Ie_sig/(Vthr-El))**2 # ms(ms/nF*nA)^2 = mV^2
  Iext = tm/(C/1000)*Iext_nA/(Vthr-El)
  return K, D, Iext, Vr

def rescale_params_units(D, Delta, Iext, K, tm, Vr, Vthr, El, Nint, C, default_coupling=False):
  IPSPint = convert_coupling(K, tm, Nint, Vthr, El, unit=True, default=default_coupling) #K/Nint*(Vthr-El)*tm
  Ie_sig_nA = np.sqrt(2*D)/tm*(C/1000)*(Vthr-El)
  Iext_nA = Iext/tm*(C/1000)*(Vthr-El)
  Vreset = Vr*(Vthr-El)+El
  return IPSPint, Ie_sig_nA, Vreset, Iext_nA

def getParams_pypet(parent, **changeParams):
  '''
  load default parameters, update by changeParams and return all parameters in a pd Data Frame for use in pypet simulation
  '''
  print('loading simulation parameters...')
  # get default Parameters:
  dfParams = loadStandardParams(parent, use='pypet', dropna=False)   
  # replace parameter values where desired
  d = dfParams.to_dict(orient='index') # export df to dict
  d['value'].update(changeParams) # update dict
  dfParams = pd.DataFrame.from_dict(d, orient='index') # make new df from updated dict
  dfParams = dfParams.dropna(axis='columns', subset=['value']) # drop nans
  return dfParams

def convertParams2Dict(group, key_format='endnode', condition = lambda x: '.input.' not in x and '.analysis.' not in x):
  '''
  convert the parameters of a pypet trajectory to a dictionary
    group: for example: traj.parameters or traj.derived_parameters
    condition: if True for a parameter, it will be included in the output dictionary
  '''
  d_traj = group.f_to_dict()
  d_out = {}
  for key in d_traj:
    if condition(key):
      if key_format == 'endnode':
        key_out = key.split('.')[-1] # extract parameter name (delete nested part)
      elif key_format == 'full':
        key_out = key
      val_out = d_traj[key].f_get()  # extract parameter value
      d_out[key_out] = val_out
  return d_out # dictionary

#%% Pypet tools
def pypet_get_trajectoryPath(traj_hash, path_to_simulations = './simulations/'):
  return path_to_simulations + 'h{}/'.format(traj_hash)
    
def pypet_load_trajectory(traj_hash = None, pypet_loadmode = (2,1,2), run=None, path_to_simulations = './simulations/'):
  ''' load trajectory from hdf5 storage
  INPUT:
    path: full path to file
    pypet_loadmode: 0: load nothing, 1: load skeleton, 2: load content (given for parameters, results, derived pars)
    run: set trajectory to one specific run
          either specify run number directly or give dictionary of desired parameter setting to look at    
  '''
  (lp, lr, ldp) = pypet_loadmode
  
  trajectoryPath = pypet_get_trajectoryPath(traj_hash, path_to_simulations = path_to_simulations)
  
  traj_name = 'Data_h{}'.format(traj_hash)
  
  traj = Trajectory(traj_name, add_time=False)
  traj.f_load(filename=trajectoryPath+traj_name+'.hdf5', load_parameters=lp, load_results=lr, load_derived_parameters=ldp, force=True)
  traj.v_auto_load = True
  if run!=None:
    if type(run)==dict: # run is a dictionary specifying the parameter settings of the run of interest
      # find the run-idx corresponding to the parameter setting specified in run
      exploredParameters = traj.f_get_explored_parameters()
      if len(exploredParameters)==1:
        parsetting = lambda x: x==list(run.values())[0]
        idx_iterator = traj.f_find_idx(list(run.keys())[0], parsetting)
      elif len(exploredParameters)==2:
        runitems = list(run.items())
        parsetting = lambda x, y: x==runitems[0][1] and y==runitems[1][1]
        idx_iterator = traj.f_find_idx([runitems[0][0],runitems[1][0]] , parsetting)
      else: # NO parameter exploration:
        raise ValueError('Implement reload for >=3d parameter exploration!')
      idx = [x for x in idx_iterator]
      if len(idx)!=1:
        raise ValueError('Too many or no runs found matching the requested parameter setting!')
      else:
        idx = idx[0]
      traj.v_idx=idx # trajectory now behaves as single run with parameters set as in 'run'
    else: # run is already the number of the run of interest
      traj.v_idx=run
  return traj

def pypet_find_runs(traj, params, condition):
  ''' Find runs of a pypet trajectory that satisfy a certain condition
  INPUT:
    params: (list of) parameter-strings
    condition: lambda function taking as many arguments as len(params), returning True for the configurations we want to extract
  OUTPUT:
    list of run indices
  '''
  idx_iterator = traj.f_find_idx(params, condition) 
  return [x for x in idx_iterator]  

def pypet_get_from_runs(traj, name, run_idx=[], return_nparray_if_possible=True):
  '''   
  returns dict or numpy array with values of "name" for each run of traj
  INPUT:
    name: string
  OUTPUT:
    dimensions:
    0: run-index
    1-x: dimension of "name" (e.g. time)
  '''
  
  d = traj.f_get_from_runs(name, auto_load=True, fast_access=True, include_default_run=False, use_indices=True, shortcuts=False) # ordered dict #  
  if return_nparray_if_possible:
    try: # return numpy array
      #print([len(v) for v in d.values()])
      array_out = np.stack(list(d.values()))
      if len(run_idx):
        array_out = array_out[run_idx]
      return array_out
    except: # return dictionary
      pass
  # keep only the runs from run_idx and rename all dict keys 0,1,...  
  new_idx = 0
  dict_out = {}
  for r in list(d.keys()):
    keep = True
    if len(run_idx):
      if r not in run_idx:
        keep = False
    if keep:
      dict_out[new_idx] = d[r]
      new_idx += 1
  if len(run_idx) and not (len(dict_out) == len(run_idx)):
    raise ValueError('fix pypet_get_from_runs!')
  # for run_idx in list(d.keys()):
  #   d[new_idx] = d.pop(run_idx)
  #   new_idx += 1
  if return_nparray_if_possible:
    # if for the given runs all arrays had the same shape, try making an array from the dict now:
    try: # return numpy array
      array_out = np.stack(list(dict_out.values()))
      return array_out
    except:
      pass
  print('pypet f_get_from_runs: returns {} (variable shape) as dictionary, renumbering runs to start from 0'.format(name))
  return dict_out

def pypet_shortcut(params):
  ''' return only name of last child in parameter tree '''
  if type(params) == list:   
    for i in range(len(params)):
      params[i] = params[i].split('.')[-1]
  else:
    params = params.split('.')[-1]
  return params

def pypet_get_exploration_overview(traj):
  '''
  return pandas data frame 
  '''
  exploredParameters = pypet_shortcut(list(traj.f_get_explored_parameters()))
  nruns = len(traj.f_get_run_names())
  df = pd.DataFrame(index = np.arange(nruns), columns = exploredParameters).rename_axis(index='run')

  for i in range(nruns):
      traj.v_idx = i
      for ep in exploredParameters:
        df.loc[i,ep] = traj.f_get(ep).data 
  if ('level' in df.columns) and (traj.input.shape == 'flat'):
    df['level'] = df['level'].astype(float) # force data type to make sure it can be stored
  traj.f_restore_default() 
  return df

def pypet_get_runidx_from_runname(run_name):
  '''
  Parameters
  ----------
  run_name : str of form "run_00000XX"

  Returns
  -------
  int: run index XX

  '''
  return int(run_name.split('_')[1].replace('0',''))

#%% Pypet Configuration Files
def get_aux_params_default(input_params):
  np.random.seed()
  aux_params = {'analysis.getInstFreq':False, 'analysis.getSynch':False, 'analysis.getSat_cyclewise':False, \
                'postproc.discard':['Cii', 'Cie', 'Pspktrains'], 'analysis.unit': False, 'analysis.net': False, \
                'analysis.offset': 50, 'analysis.v_dynamics': False, 'analysis.fmin': 10, 'analysis.fmax': 350, 'analysis.wavelet_fexp': 200,\
                'simulation.seed_brian': np.random.randint(2**31-1), 'simulation.seed_network':False, 'simulation.track_connectivity' : False } 
  aux_params['analysis.ifreq_targetwindow'] = [type(input_params['Tsim'])(200), input_params['Tsim']]
  aux_params['analysis.ifreq_baselinewindow'] = [50, 150]
  aux_params['postproc.v_recordwindow'] = [0.0, float(input_params['Tsim'])]
  aux_params['postproc.v_recordwindow_for_simulation'] = aux_params['postproc.v_recordwindow']
  return aux_params


def pypet_setup_config(parent, changeParams, forExploration, input_params, aux_params_change, commentTraj, 
                 traj_hash=np.nan, multiproc = True, CoresToUse=50, cart_product=True, \
                 record_per_run={}, path_to_simulations = './simulations/'):
  ''' 
  Setup a configuration file for a pypet simulation and store it in path_to_simulations/h{traj_hash}/config_h{traj_hash}.txt
  '''
  if np.isnan(traj_hash):
    traj_hash=np.random.randint(1e10)
  trajectoryPath = pypet_get_trajectoryPath(traj_hash, path_to_simulations = path_to_simulations)
  trajectoryName = 'Data_h{}'.format(traj_hash)
  
  aux_params_default = get_aux_params_default(input_params)
  aux_params = {**aux_params_default, **aux_params_change}
  if ('postproc.v_recordwindow' in aux_params_change.keys()) and ('postproc.v_recordwindow_for_simulation' not in aux_params_change.keys()):
    aux_params['postproc.v_recordwindow_for_simulation'] = aux_params['postproc.v_recordwindow']
  
  # all possible parameter combinations:
  if cart_product:
    forExploration = cartesian_product(forExploration)  
 
  if len(record_per_run):
    changeParams['record_micro'] = ['NONE'] # add record_micro as explorable parameter
    nruns = len(forExploration[list(forExploration.keys())[0]])
    forExploration['record_micro'] = [["NONE"]]*nruns
    for r in record_per_run['runs']:
      forExploration['record_micro'][r] = record_per_run['record']
  
  # parameter adjustments:
  if ('inputmode' in changeParams.keys()) and (changeParams['inputmode'] == 'DiffApprox'):
    changeParams['integrationmethod'] = 'milstein' # adjust integrationmethod used by Brian
  
  
  config = \
  {   'commentTraj' : commentTraj,
      'parent' : parent,
      'changeParams' : convert_numpy_2_python(changeParams),
      'input_params' : convert_numpy_2_python(input_params), 
      'forExploration' : convert_numpy_2_python(forExploration),  
      'trajectoryPath' : trajectoryPath,
      'trajectoryName' : trajectoryName,
      'CoresToUse' : CoresToUse,
      'multiproc' : multiproc,
      'aux_params' : convert_numpy_2_python(aux_params),
      'hash': traj_hash
  }
  
  # check forExploration length
  n = len(forExploration[list(forExploration.keys())[0]])
  for i in forExploration.keys():
    if len(forExploration[i]) != n:
      print(i,len(forExploration[i]), n)
      raise ValueError("Lengths of explored parameters don't match!")
  
  if not os.path.exists(trajectoryPath):
    os.makedirs(trajectoryPath)
  json.dump(config, open(trajectoryPath+'config_h{}.txt'.format(traj_hash),'w'), sort_keys=True, indent=2, separators=(',', ': '))
  print('Configuration stored for trajectory: \n{}. Will perform {} runs'.format(trajectoryName, n))
  return 

def pypet_makeConfig_gauss_lif(parameters, Nint=1000, tref=0, do_analysis_membrane_potential_dynamics = True, comment='', forExploration={}, traj_hash=np.nan,\
                               v_recordwindow = [4000., 4025.], path_to_simulations = './simulations/'):
  ''' creates config file for a pypet simulation with constant drive of varying levels
  Parameters
  ----------
  parameters : dict. Network parameters D, Delta, K, tm, Vr.
  Nint : float, optional. Network size. The default is 1000.
  do_analysis_membrane_potential_dynamics : bool, optional. The default is True.
  comment : str, optional. The default is ''.
  forExploration : dict, optional. The default is {}.

  Returns
  -------
  traj_hash 

  '''
  print('Preparing Configuration file for Pypet simulation (Fig. 1: variation of constant drive input)...')
  # --- unpack (dimensionless) parameters and rescale to derive parameters with standard units
  D, Delta, K, tm, Vr = parameters['D'], parameters['Delta'], parameters['K'], parameters['tm'], parameters['Vr']
  stp = loadStandardParams('mymicro')
  Vthr, E_rest, C = stp['Vthr'], stp['E_rest'], stp['C']
  
  # --- upper end of input range: approximate pt of full synch analytically
  Ifull = get_pt_fullsynch(D, Delta, K, tm)
  
  IPSPint, Ie_sig, Vreset, Ifull_nA \
  = rescale_params_units(D, Delta, Ifull, K, tm, Vr, Vthr, E_rest, Nint, C, default_coupling=True) 
  # returns IPSPint for N=200, sth it can be scaled in the code
  
  # --- lower end of input range: find bifurcation
  sig = np.sqrt(2*D)*(Vthr-E_rest) # mV
  Icrit_nA, f_crit, I0_crit_nA, A0_crit, fig_ls \
  = linear_stability_analysis('delta', Vthr-E_rest, Vreset-E_rest, tm, tref, Delta, C, \
                              Nint, IPSPint = IPSPint*stp['p_ii']*stp['Nint']/Nint , lif_sigma=sig, plot=True)[:5]
    
  # --- set up the dictionaries changeParams and forExploration for the pypet simulation
  input_params = {'Tsim': 5050, 'shape': 'flat', 'level': 0.0, 'unit':'nA', 'Ie_sig': np.round(Ie_sig, decimals=3)}
  aux_params = {'analysis.unit': True, 'analysis.net': True, 'postproc.par0': 'parameters.input.level', \
                'analysis.v_dynamics': do_analysis_membrane_potential_dynamics, 'postproc.v_recordwindow': v_recordwindow,
                'linear_stability.Icrit_nA':Icrit_nA, 'linear_stability.fcrit': f_crit, 'linear_stability.I0crit_nA': I0_crit_nA, \
                'linear_stability.A0crit': A0_crit}
  if do_analysis_membrane_potential_dynamics:
    aux_params['postproc.v_recordwindow_for_simulation'] = [4000., 5000.] # record longer voltage traces to get good analysis of mean dynamics, but only save shorter traces and average histogram
  commentTraj = 'Constant drive simulations with drive level ranging from below critical level to point of full synchrony. ' + comment
  forExploration['level'] = list(np.round([Icrit_nA/2] + list(np.linspace(Icrit_nA, Ifull_nA, 10)) + [1.5*Ifull_nA], decimals=3))
  changeParams = {'neuronmodel': 'LIF', 'IPSPshape' : 'delta', 'inputmode': 'DiffApprox',
                  'tref': tref, 'p_ii': 1, 'Nint': Nint, 'record_micro': ('v')} #, 'scaleWeights':False}
                   
  for key, val in zip(['tl', 'IPSPint', 'tm', 'Vreset'], [Delta, IPSPint, tm, Vreset]):
      if (not val == stp[key] and (not key in forExploration.keys())):
        changeParams[key] = float(val)
      
  pypet_setup_config('mymicro', changeParams, forExploration, input_params, aux_params, commentTraj, \
                     cart_product=True, traj_hash=traj_hash, path_to_simulations = path_to_simulations)
  
  # save results of linear stability analysis
  # datapath = 'Pypet/Analysis/linear_stability/'
  # if not os.path.exists(datapath):
  #   os.makedirs(datapath)
  # np.savez(datapath+'linear_stability_h{}.npz'.format(traj_hash), I_crit_nA = Icrit_nA, f_crit = f_crit, 
  #           A0_crit = A0_crit, I0_crit_nA = I0_crit_nA,
  #           doc='analytically calculated bifurcation parameter Icrit [nA] and onset frequency f_crit [Hz].\
  #               A0_crit [Hz]: unit rate / stationary population rate in bifurcation point, \
  #               I0_crit [nA]: total input in bifurcation point.')    
  # fig_ls.savefig(datapath+'linear_stability_h{}.pdf'.format(traj_hash), bbox_inches='tight')
  print('Config for traj h{} stored! \n Will explore input levels {}'.format(traj_hash, forExploration['level'])) 
  return 

def pypet_makeConfig_lif_ifa_fromcyclostat(traj_hash_stat=None, Nint=None, nreps=50, comment='', traj_hash=None, \
                             shape='ramp', ramp_time = 20, plateau_time = 20, peak_nA=None, slope_dimless=None, Tinit = 200., Tedge=20., t_pad = 20.,\
                             ramp_time_up = None, ramp_time_down=None, std_gauss = 7, \
                             fmin=70, fmax=400, path_to_simulations = './simulations/'):
  '''
  Based on an existing network simulation for constant drive, setup a simulation with time-dependent drive that
  - starts below the bifurcation level
  - ends at the approx. point of full synchrony found in traj_hash_stat
  - has ramp_time, plateau_time etc as specified in inputs
  '''
  
  print('Preparing Pypet Config for IFA simulation w.r.t. cyclostat reference simulation hash {}...'.format(traj_hash_stat))
  comment +=  ' Cyclostat reference simulation: {}'.format(traj_hash_stat)
  
  # load config file of constant-drive simulation
  changeParams, _, _, aux_params = pypet_loadConfig(traj_hash_stat, path_to_simulations=path_to_simulations)[6:]

  # load constant-drive trajectory
  traj = pypet_load_trajectory(traj_hash = traj_hash_stat, path_to_simulations=path_to_simulations)
  
  # --- load info with and without units from stationary simulation
  path_stat_info = pypet_get_trajectoryPath(traj.hash, path_to_simulations=path_to_simulations) + 'info.csv'
  if not os.path.exists(path_stat_info):
    store_info_cyclostat_lif_simulation(traj=traj)
  info = pd.read_csv(path_stat_info, index_col=0, squeeze=True, header=None)
  D, Delta, K, tm, Vr, I_hopf, I_hopf_nA, I_full, I_full_nA, IPSPint_default \
  = info['D'], info['Delta'], info['K'], info['tm'], info['Vr'], info['I_hopf'], info['I_hopf_nA'], info['I_full'], info['I_full_nA'], info['IPSPint_default']
  print('D: {}, Delta: {}, K: {}, tm: {}, Vr: {}'.format(D, Delta, K, tm, Vr))

  # --- prepare configuration file
  stp = loadStandardParams('mymicro')
  baseline, baseline_nA = I_hopf/2, I_hopf_nA/2
  if not peak_nA:
    peak, peak_nA = I_full, I_full_nA
  else:
    Vthr, E_rest, C = stp['Vthr'], stp['E_rest'], stp['C']
    peak = convert_input(peak_nA, tm, C, Vthr, E_rest, unit=False)
  aux_params['analysis.unit'] = True
  aux_params['analysis.net'] = True
  aux_params['analysis.offset'] = 50
  aux_params['analysis.getInstFreq'] = True
  aux_params['analysis.ifreq_baselinewindow'] = [50, 150]
  aux_params['analysis.fmin'] = fmin
  aux_params['analysis.fmax'] = fmax
  aux_params['analysis.v_dynamics'] = False
  aux_params['postproc.v_recordwindow_for_simulation'] = aux_params['postproc.v_recordwindow']
  # aux_params['linear_stability.Icrit_nA']=traj.linear_stability.Icrit_nA
  # aux_params['linear_stability.fcrit']= traj.linear_stability.fcrit
  # aux_params['linear_stability.I0crit_nA']= traj.linear_stability.I0crit_nA
  # aux_params['linear_stability.A0crit']= traj.linear_stability.A0crit
    
  # --- set up the dictionaries changeParams and forExploration for the pypet simulation
  forExploration = {}
  input_params = {'unit':'nA', 'Ie_sig': np.round(traj.Ie_sig, decimals=3), 'shape': shape, 'Tinit':Tinit, 'baseline':baseline_nA, 'peak':peak_nA}

  # set network size and coupling strength
  if type(Nint) == list: # vary network size
    forExploration['Nint'] = [int(n) for n in Nint]
    # set default weight for Nint=200 and let weights be scaled internally depending on resp. network size
    changeParams['IPSPint'] = IPSPint_default
    # remove Nint and scaleWeights as changed Parameters
    for key in ['scaleWeights', 'Nint']:
      if key in changeParams.keys():
        changeParams.pop(key)
  elif Nint: # network size fixed as input argument
    if ('Nint' in changeParams.keys()) and (changeParams['Nint'] == Nint):
        pass
    else: # update: different network size here as in stationary simulation
      changeParams['Nint'] = Nint
      changeParams['IPSPint'] = IPSPint_default
      if 'scaleWeights' in changeParams.keys():
        changeParams.pop('scaleWeights')
  else: # no network size specified
    pass # use same network size (and coupling strength) as in stationary simulation (already indicated in changeParams, if it deviates from default)

  # --- prepare (exploration of) input parameters
  if shape == 'ramp':
    input_params['plateau_time'] = plateau_time
    if slope_dimless: # slopes are given (from theory), infer ramp_time:
      if type(slope_dimless)==list:
        ramp_time = [(peak-baseline)/m for m in slope_dimless] 
      else:
        ramp_time = (peak-baseline)/slope_dimless
    if type(ramp_time) == list:
      forExploration['ramp_time'] = ramp_time
      # --- take cartesian product if also the network size was varied: 
      if len(forExploration.keys()) > 1:
        forExploration = cartesian_product(forExploration)
      # --- add explorations that derive from the ramp_time exploration
      forExploration['Tsim'] = list(Tinit + 2*np.array(forExploration['ramp_time']) + plateau_time + 2*Tedge + 2*t_pad)
      forExploration['analysis.ifreq_targetwindow'] = [[Tinit + t_pad, Tsim - t_pad] for Tsim in forExploration['Tsim']]
      forExploration['postproc.v_recordwindow'] = forExploration['postproc.v_recordwindow_for_simulation'] =  forExploration['analysis.ifreq_targetwindow'] 
      # --- enter default values for the explored parameters
      input_params['ramp_time'] = np.array(forExploration['ramp_time']).dtype.type(0)
      input_params['Tsim'] =  np.array(forExploration['Tsim']).dtype.type(0)
    else:
      input_params['ramp_time'] = ramp_time
      input_params['Tsim'] =  Tinit + 2*ramp_time + plateau_time + 2*Tedge + 2*t_pad 
      aux_params['analysis.ifreq_targetwindow'] = [Tinit + t_pad, input_params['Tsim'] - t_pad]
      aux_params['postproc.v_recordwindow'] = aux_params['analysis.ifreq_targetwindow'] 
  elif shape == 'ramp_asym': 
    # double-ramp with asymmetric ramp_times 
    input_params['plateau_time'] = plateau_time
    if type(ramp_time_up) == list:
      forExploration['ramp_time_up'] = ramp_time_up
      forExploration['ramp_time_down'] = ramp_time_down
      # --- take cartesian product if also the network size was varied: 
      if 'Nint' in forExploration.keys(): # other co-explorations would need to be implemented here
        forExploration['Nint'] = np.repeat(forExploration['Nint'], len(ramp_time_up))
        forExploration['ramp_time_up'] = np.tile(forExploration['ramp_time_up'], len(Nint) )
        forExploration['ramp_time_down'] = np.tile(forExploration['ramp_time_down'], len(Nint) )
      # --- add explorations that derive from the ramp_time exploration
      forExploration['Tsim'] = list(Tinit + np.array(forExploration['ramp_time_up']) + np.array(forExploration['ramp_time_down']) + plateau_time + 2*Tedge + 2*t_pad)
      forExploration['analysis.ifreq_targetwindow'] = [[Tinit + t_pad, Tsim - t_pad] for Tsim in forExploration['Tsim']]
      forExploration['postproc.v_recordwindow'] = forExploration['postproc.v_recordwindow_for_simulation'] = forExploration['analysis.ifreq_targetwindow'] 
      # --- enter default values for the explored parameters
      input_params['ramp_time_up'], input_params['ramp_time_down'] = np.array(forExploration['ramp_time_up']).dtype.type(0), np.array(forExploration['ramp_time_down']).dtype.type(0)
      input_params['Tsim'] =  np.array(forExploration['Tsim']).dtype.type(0)
    else:
      input_params['ramp_time_up'] = ramp_time_up
      input_params['ramp_time_down'] = ramp_time_down
      input_params['Tsim'] =  Tinit + ramp_time_up + ramp_time_down + plateau_time + 2*Tedge + 2*t_pad 
      aux_params['analysis.ifreq_targetwindow'] = [Tinit + t_pad, input_params['Tsim'] - t_pad]
      aux_params['postproc.v_recordwindow'] = aux_params['analysis.ifreq_targetwindow'] 
  elif shape == 'gaussian_shifted':
    if type(std_gauss) == list:
      forExploration['sig'] = std_gauss
      # --- take cartesian product if also the network size was varied: 
      if len(forExploration.keys()) > 1:
        forExploration = cartesian_product(forExploration)
      # --- add explorations that derive from the ramp_time exploration
      forExploration['Tsim'] = list(Tinit + 6*np.array(forExploration['sig']) + 2*Tedge + 2*t_pad)
      forExploration['analysis.ifreq_targetwindow'] = [[Tinit + t_pad, Tsim - t_pad] for Tsim in forExploration['Tsim']]
      forExploration['postproc.v_recordwindow'] = forExploration['postproc.v_recordwindow_for_simulation'] = forExploration['analysis.ifreq_targetwindow'] 
      # --- enter default values for the explored parameters
      input_params['Tsim'] =   np.array(forExploration['Tsim']).dtype.type(0)
      input_params['sig'] =  np.array(forExploration['sig']).dtype.type(0)
    else:
      input_params['sig'] = std_gauss
      input_params['Tsim'] =  Tinit + 6*std_gauss + 2*Tedge + 2*t_pad # Tinit + 2*ramp_time + plateau_time + Tpost
      aux_params['analysis.ifreq_targetwindow'] = [Tinit + t_pad, input_params['Tsim'] - t_pad]
      aux_params['postproc.v_recordwindow'] = aux_params['analysis.ifreq_targetwindow'] 
  else:
    raise ValueError('unknown stimulus shape {}!'.format(shape))
    
  
  # number of settings we want to explore so far
  if len(forExploration.keys()):
    n_trials = len(forExploration[list(forExploration.keys())[0]])
  else:
    n_trials = 1
    
  # --- now repeat each exploration list nreps times and add the brian seed as an additional explored parameter
  # (we dont simply take cartesian product because brian seeds should not repeat in different settings, even though that probably doesnt matter)
  for key in list(forExploration.keys()):
    all_items = forExploration[key].copy()
    forExploration[key] = [item for item in all_items for i in range(nreps)]
    #   forExploration[key] = list(np.repeat(forExploration[key], nreps))
    
  forExploration['simulation.seed_brian'] = list(np.random.randint(2**32-1, size=nreps*n_trials))
  
  commentTraj = 'LIF transient stimulation with {}. '.format(shape) + comment
                  
  # for key, val in zip(['tl', 'IPSPint', 'tm', 'Vreset'], [Delta, IPSPint_default, tm, traj.Vreset]):
  #     if not val == stp[key]:
  #       changeParams[key] = val
      
  pypet_setup_config('mymicro', changeParams, forExploration, input_params, aux_params, commentTraj, \
                     cart_product=False, record_per_run={'runs': [i*nreps for i in range(n_trials)], 'record':['v']}, traj_hash=traj_hash, path_to_simulations = path_to_simulations)    
  
  print('Config for traj h{} stored! \n Will stimulate with a {} between {:.2f}-{:.2f}nA'.format(traj_hash, shape, input_params['baseline'], input_params['peak'])) 
  return

def pypet_make_config_brunel_hakim99(mu_ext, sigma_ext, traj_hash, path_to_simulations = './simulations/'):
  ''' Prepare simulation configuration file to replicate the Brunel, Hakim 1999 network with parameters as in their Fig. 3,5.
  The network activity will be simulated for the range of external Gaussian white noise inputs specified by mean mu_ext and noise intensity sigma_ext.  

  Parameters
  ----------
  mu_ext : TYPE
    DESCRIPTION.
  sigma_ext : TYPE
    DESCRIPTION.

  Returns
  -------
  None.

  '''
  parent = 'mymicro' # start from the default parameters of our network models
  stp = loadStandardParams(parent)
  # now adjust parameters as given in Brunel, Hakim 99 Fig. 3+5:
  changeParams = {'neuronmodel': 'LIF', 'IPSPshape' : 'delta', 'inputmode': 'DiffApprox', 'record_micro': ('v'), 
                  'Nint': 5000, 
                  'Vreset': stp['E_rest']+10,
                  'Vthr': stp['E_rest']+20,
                  'tm': 20,
                  'tl': 2,
                  'tref' : 0, 
                  'p_ii': 0.2, # indegree C=1000 -> pii = C/N
                  'IPSPint': 2, # =J*tm = 0.1mV*20ms = 2 mVms
                  'scaleWeights' : False # disable internal weight scaling              
                  }
    
  # translate into my existing brian eqs that are formulated in terms of currents (assuming fixed default capacitance C=100pF)
  # my Iext = capacitance/tm*mu_ext (mu_ext Brunel)
  # my Ie_sig = capacitance/tm*sigma_ext (sigma_ext Brunel)
  capacitance = stp['C']/1000 # 0.1 nano Farad, aux variable, no real significance
  Iext_range = capacitance/changeParams['tm']*mu_ext # nF*ms*mV = nA
  Iesig_range = capacitance/changeParams['tm']*sigma_ext # nF*ms*mV = nA
  
  input_params = {'Tsim': 5050, 'shape': 'flat', 'level': 0.0, 'Ie_sig': 0.0, 'unit':'nA'}
  aux_params = {'analysis.unit': True, 'analysis.net': True, 'postproc.par0': 'parameters.input.level', 'postproc.v_recordwindow': [5010., 5050.]}
  commentTraj = 'Brunel, Hakim 1999 network with parameters as in Fig. 3-5. Vary external Gaussian white noise input in mean and intensity (Poisson-like). Cf. Brunel, Hakim 99, Fig. 5 right, long-dashed line.'
  forExploration = {'level': list(Iext_range), 'Ie_sig': list(Iesig_range)} # co-vary mean input and noise level 
  
  # linear stability analysis, assumes full connectivity, so syn strength is scaled down
  from methods_analytical import linear_stability_analysis
  sigma_fix = 1.5 # assume fixed sigma
  Icrit_nA, f_crit, I0_crit_nA, A0_crit, fig_ls \
  = linear_stability_analysis('delta', changeParams['Vthr']-stp['E_rest'], changeParams['Vreset']-stp['E_rest'], changeParams['tm'], changeParams['tref'], \
                              changeParams['tl'], 100, changeParams['Nint'], IPSPint = changeParams['IPSPint']*changeParams['p_ii'], lif_sigma=sigma_fix, Imax_nA=0.2, dIext_min_nA=0.02, plot=True)[:5]
  aux_params['linear_stability.Icrit_nA'] = Icrit_nA
  aux_params['linear_stability.fcrit'] = f_crit
  aux_params['linear_stability.I0crit_nA'] = I0_crit_nA
  aux_params['linear_stability.A0crit'] = A0_crit
    
  pypet_setup_config(parent, changeParams, forExploration, input_params, aux_params, commentTraj, cart_product=False, traj_hash=traj_hash, path_to_simulations = path_to_simulations)
    
  # illustrate inputs in terms of current and voltage
  fig = plt.figure(figsize=(width_a4_wmargin*.7, 4.5*cm))
  gs = gridspec.GridSpec(1,2,figure=fig, wspace=.6)
  gs1 = gs[0,0].subgridspec(2,1)
  ax1 = gs1.subplots(sharex=True)
  ax2 = fig.add_subplot(gs[0,1])
  despine(ax1)
  
  ax1[0].plot(Iext_range, mu_ext, 'k')
  ax1[0].set_ylabel(r'$\mu_\mathrm{ext}$ [mV]')
  ax1[1].plot(Iext_range, sigma_ext, 'k')
  # mark range of inputs explored by Brunel in Fig. 5
  ix = [np.where(mu_ext>=20)[0][0], np.where(mu_ext<=30)[0][-1]]
  ax1[0].plot(Iext_range[[ix[0], ix[0], ix[1], ix[1], ix[0]]], mu_ext[[ix[0], ix[1], ix[1], ix[0], ix[0]]], lw=.5, color='gray')
  ax1[1].plot(Iext_range[[ix[0], ix[0], ix[1], ix[1], ix[0]]], sigma_ext[[ix[0], ix[1], ix[1], ix[0], ix[0]]], lw=.5, color='gray')
  ax1[1].set_xlim([0, 0.8])
  ax1[0].set_ylim(top=190)
  ax1[1].set_ylabel(r'$\sigma_\mathrm{ext}$ [mV]')
  ax1[1].set_xlabel('drive [nA]')
  
  ax2.plot(sigma_ext, mu_ext, 'k--')
  mu_ext_crit = Icrit_nA/capacitance*changeParams['tm']
  ax2.plot(sigma_fix, mu_ext_crit, 'r^', label='approx \nbifurcation')
  ax2.set_xlim([0,5])
  ax2.set_ylim([20,30])
  ax2.set_xticks([0,1,2,3,4,5])
  ax2.set_yticks(list(np.arange(20,31,2).astype(int)))
  ax2.set_xlabel(r'noise $\sigma_\mathrm{ext}$')
  ax2.set_ylabel(r'drive $\mu_\mathrm{ext}$')
  
  fig.savefig(pypet_get_trajectoryPath(traj_hash=traj_hash, path_to_simulations=path_to_simulations)+'input_current_vs_volt_h{}.pdf'.format(traj_hash), bbox_inches='tight')
  return

def pypet_makeConfig_performance_check(exploration, traj_hash, Nint=1000, n_sim=15, do_membrane_potential_dynamics_analysis=True, 
                                       path_to_simulations='./simulations/', CoresToUse=25): 
  '''
  prepare a large pypet network simulation config to compare with the analytical gauss approximation over a range of parameters
  INPUT:
    exploration: dictionary with keys:
      D, Delta, K, tm, Vr: dimensionless parameters of the analytical gauss approximation (any of them can be an array!)
    Nint: number of neurons to use in network
    n_sim: number of input levels to check between bifurcation and full synch
  OUTPUT: 
    changeParams, forExploration: dicts to be used in pypet_makeConfig.py 
  '''
  # --- unpack exploration dictioanry:
  D, Delta, K, tm, Vr = exploration['D'], exploration['Delta'], exploration['K'], exploration['tm'], exploration['Vr']
  p_fix = get_pfix(exploration) # names of parameters that are kept fixed in the exploration
  p_var = get_pvar(exploration) # names of parameters that are varied in the exploration
  
  # --- take cartesian product of all parameter variations and setup data frame with parameters for each NETWORK configuration
  D_all, Delta_all, K_all, tm_all, Vr_all = np.meshgrid(D, Delta, K, tm, Vr)
  
  df_net = pd.DataFrame(columns=['D', 'Delta', 'tl', 'K', 'tm', 'Vr', 'Ifull', 'IPSPint', 'Vreset', 'Ie_sig', 'Ifull_nA', 'Icrit_nA'])
  df_net['D'] = D_all.flatten()
  df_net['tl'] = Delta_all.flatten()
  df_net.Delta = df_net.tl
  df_net['K'] = K_all.flatten()
  df_net['tm'] = tm_all.flatten()
  df_net['Vr'] = Vr_all.flatten()

  # --- approximate pt of full synchrony for each network
  df_net['Ifull'] = get_pt_fullsynch(df_net.D, df_net.tl, df_net.K, df_net.tm)
  
  # --- convert parameters to units of simulated ripple network
  ## load default network values
  stp = loadStandardParams('mymicro')
  Vthr, E_rest, C = stp['Vthr'], stp['E_rest'], stp['C']
  
  df_net.IPSPint, df_net.Ie_sig, df_net.Vreset, df_net.Ifull_nA \
  = rescale_params_units(df_net.D, df_net.tl, df_net.Ifull, df_net.K, df_net.tm, df_net.Vr, Vthr, E_rest, Nint, C)

  # --- linear stability analysis to approximate point of bifurcation  
  for i in range(len(df_net)):
    print(i,'/',len(df_net), end='\n----------------------------------------------------------\n')
    sig = df_net.loc[i,'tm']*df_net.loc[i,'Ie_sig']/(C/1000) # ms*nA/nF = mV
    df_net.loc[i,'Icrit_nA'], df_net.loc[i,'fcrit'], df_net.loc[i,'I0crit'], df_net.loc[i,'A0crit'] \
    = linear_stability_analysis('delta', Vthr-E_rest, df_net.loc[i,'Vreset']-E_rest, df_net.loc[i,'tm'], 0, df_net.loc[i,'tl'], C, \
                                Nint, IPSPint = df_net.loc[i,'IPSPint'], lif_sigma=sig, plot=False, fmax=1000/(2*df_net.loc[i,'tl']))[:4]
  
  df_net['Icrit'] = df_net['tm']/(C/1000)*df_net['Icrit_nA']/(Vthr-E_rest)
      
  if df_net[['Icrit_nA', 'Ifull_nA']].isna().any().any():
    print('Nan was found for some critical or full synch input level! I will still do a simulation!')
    print(df_net[df_net.isnull().any(axis=1)])
    
  # --- go from the data frame per NETWORK to a data frame per RUN 
  # --- every network will be run n_sim times with input levels ranging (linearly) from Icrit to Ifull
  df_runs = pd.concat([df_net]*n_sim)
  df_runs = df_runs.sort_index()
  for i in range(len(df_net)):
    if np.isnan(df_net.loc[i,'Icrit_nA']):
      I0 = 0.1
    else:
      I0 = df_net.loc[i,'Icrit_nA'] # set lowest input level depending on bifurcation point
    if np.isnan(df_net.loc[i,'Ifull_nA']): # set highest input level depending on approx point of full synch
      I1 = I0+2
    else:
      I1 = df_net.loc[i,'Ifull_nA']
    # range of input levels to explore: from critical input level to the point of full synch, plus one level beyond that in case the pt of full synch was underestimated  
    df_runs.loc[df_runs.index==i, 'level'] = np.append(np.linspace(I0, I1, n_sim-1), 1.5*I1)
  df_runs['config'] = df_runs.index.copy()  # give each network configuration (D, Delta, K, Vr, tm) a "config" number
  df_runs.index = np.arange(len(df_runs)).astype(int)
  
  # --- force data types for storage
  df_net = df_net.astype(float)
  df_runs = df_runs.astype(float)
  
  # --- set up the dictionaries changeParams and forExploration for the pypet simulation
  input_params = {'Tsim': 5050, 'shape': 'flat', 'level': 0.0, 'unit':'nA', 'Ie_sig': 0.0}
  aux_params = {'analysis.unit': True, 'analysis.net': True, 'postproc.par0': 'parameters.input.level', \
                'postproc.v_recordwindow': [4000.0, 4025.0], \
                'analysis.v_dynamics': do_membrane_potential_dynamics_analysis}
  if do_membrane_potential_dynamics_analysis:
    aux_params['postproc.v_recordwindow_for_simulation'] = [4000.0, 5000.0] # record at least 1sec of membrane potentials to get a good estimate of mean dynamics
  else:
    aux_params['postproc.v_recordwindow_for_simulation'] = aux_params['postproc.v_recordwindow']
  commentTraj = 'Compare analytical Gaussian-drift approx to simulation of LIF spiking network. Explored Params: {}'.format(list2str(p_var))
  
  changeParams = {'neuronmodel': 'LIF', 'IPSPshape' : 'delta', 'inputmode': 'DiffApprox', 'dt' : 0.05,
                  'tref': 0, 'p_ii': 1, 'Nint': Nint, 'record_micro': 'v', 'scaleWeights':False}
  
  # sort the parameter settings stored in df_net into explored and changed Parameters
  for key in p_fix:
    if key == 'D':
      input_params['Ie_sig'] = df_net.loc[0,'Ie_sig'] 
    else:
      changeParams[map_pstr(key, unit=True)] = df_net.loc[0,map_pstr(key,unit=True)]
  # translate names of explored parameters from dimless names to the names used in network simulation (e.g. tl not Delta etc..)
  # and add the input level as another explored parameter
  exploredParams = map_pstr(p_var, unit=True) + ['level']  

  # extract lists of values for the explored parameters for each run  
  forExploration = df_runs[exploredParams].to_dict(orient='list')

  # setup pypet trajectory for simulating all the desired network configurations and input levels
  pypet_setup_config('mymicro', changeParams, forExploration, input_params, aux_params, commentTraj, 
                                           traj_hash = traj_hash, cart_product=False, CoresToUse=CoresToUse, 
                                           path_to_simulations=path_to_simulations)
  datapath = 'results/gaussian_drift_approx_constant_drive_performance_check/'
  if not os.path.exists(datapath):
    os.makedirs(datapath)
  df_net.to_csv(datapath+'df_network_configurations_h{}.csv'.format(traj_hash))
  df_runs.to_csv(datapath+'df_parameters_per_run_h{}.csv'.format(traj_hash))
  return 

################################################################ aux functions for  performance check
def recover_exploration(df):
  exploration={}
  p_var, p_fix = [], []
  for key in ['D', 'Delta', 'K', 'tm', 'Vr']:     
    try:
      val = df[key].unique()
    except:
      val = df[map_pstr(key)].unique()
    if len(val)==1:
      val = val.item()
      p_fix.append(key)
    else:
      val = np.array(list(val))
      p_var.append(key)
    exploration[key] = val
  return exploration, p_fix, p_var

def get_pfix(exploration):
  p = []
  for k in sorted(exploration):
    if np.isscalar(exploration[k]):
      p += [k]
  return p

def get_pvar(exploration):
  p = []
  for k in sorted(exploration):
    if not np.isscalar(exploration[k]):
      p += [k]
  return p

def get_pfix_str(exploration):
  map2latex = {'D':'D', 'Delta':r'$\Delta$', 'K':'K', 'tm':  r'$\tau_m$', 'Vr':'$V_R$'}
  unit = {'D':'', 'Delta':'ms', 'K':'', 'tm':  'ms', 'Vr':''}
  p_fix = get_pfix(exploration)
  pfix_str = ''
  for p in p_fix:
    pfix_str += map2latex[p] +'={:.2f}{}, '.format(exploration[p], unit[p])
  return pfix_str[:-2]

def map_pstr(x, unit=True):
  map2unit = {'D':'Ie_sig', 'Delta':'tl', 'K':'IPSPint', 'tm':'tm', 'Vr':'Vreset'}
  map2dimless = {v: k for k, v in map2unit.items()} # inverse map
  if type(x) == list:
    result = [map_pstr(xi, unit=unit) for xi in x]
    return result
  else:
    if unit:
      return map2unit[x]
    else:
      return map2dimless[x]
    
def get_config_str(df, p_var):
  s = ''
  for p in p_var:
    if p=='D':
      s += r'$\sqrt{D}$'+'={:.2f}, '.format(np.sqrt(df[p].unique().item()))
    else:
      s += '{}={}, '.format(p, df[p].unique().item())
  return s[:-2]



def pypet_loadConfig(traj_hash, path_to_simulations = './simulations/'):
  ''' extract simulation settings from configuration file
  '''
  trajectoryPath = pypet_get_trajectoryPath(traj_hash, path_to_simulations = path_to_simulations)
  config = json.load(open(trajectoryPath+'config_h{}.txt'.format(traj_hash))) # load configuration file
  # extract content
  commentTraj = config['commentTraj']
  parent = config['parent']
  changeParams = config['changeParams']
  input_params = config['input_params']
  forExploration = config['forExploration']
  # trajectoryPath = config['trajectoryPath']
  trajectoryName = config['trajectoryName']
  CoresToUse = config['CoresToUse']
  multiproc = config['multiproc']
  aux_params = config['aux_params']
  
  print('Configuration loaded and saved for later reference')
  return trajectoryName, trajectoryPath, commentTraj, CoresToUse, multiproc, \
          parent, changeParams, forExploration, input_params, aux_params
          

#%% Pypet simulation wrapper
def pypet_run_simulation(traj_hash, path_to_simulations = './simulations/'):
  '''
  Run a pypet simulation of an interneuron network. (see also pypet.readthedocs.io/)

  Parameters
  ----------
  traj_hash : Hash for simulation. Configuration file will be loaded from path_to_simulations/h{traj_hash}/config_h{traj_hash}.txt
  path_to_simulations : str, optional. The default is './simulations/'.

  Returns
  -------
  pypet trajectory

  '''
  start = Time.time()
  # load Configuration
  trajectoryName, trajectoryPath, commentTraj, CoresToUse, multiproc, parent, changeParams, forExploration, input_params, aux_params \
  = pypet_loadConfig(traj_hash, path_to_simulations = path_to_simulations)
  
  env = Environment(trajectory = trajectoryName,
                    comment = commentTraj,
                    add_time = False, # We don't want to add the current time to the name,
                    log_config = 'DEFAULT',
                    multiproc = multiproc, 
                    ncores = CoresToUse, 
                    filename = trajectoryPath,
                    pandas_format = 't',
                    use_pool= False,
                    continuable = False,
                    memory_cap = 50.,
                    swap_cap = 99.9, #70.,
                    wrap_mode = 'LOCK'
                    )
  traj = env.trajectory
  
  # add technical parameters
  traj.f_add_parameter('hash', traj_hash, comment = 'hash to link this trajectory to figures etc') 
  for key in aux_params.keys():
    traj.f_add_parameter(key, aux_params[key], comment='flag used within pypet_run_simulation()')
    
  # get network parameters
  dfParams = getParams_pypet(parent, **changeParams)
  
  # assign parameters to trajectory
  for p in dfParams.columns:
    comment = dfParams.loc['comment',p]
    if (type(comment)==str) and type(dfParams.loc['unit',p])==str: # add unit if any
      comment = '['+dfParams.loc['unit',p]+'] '+comment
    if p in changeParams.keys():
      comment = '* '+comment # mark changed parameters with star in comments
    traj.f_add_parameter(dfParams.loc['name_pypet',p], dfParams.loc['value',p], comment = comment)
    
  # assign input parameters
  for key in input_params:
    traj.f_add_parameter('input.'+key, input_params[key], comment = 'input parameter')

  # Add parameters to be explored
  traj.f_explore(forExploration)
  
  def postproc_current_path(traj, result_list):
    return postproc(traj, result_list, path_to_simulations = path_to_simulations)
  
  env.add_postprocessing(postproc_current_path)
  env.f_run(run)
  
  env.f_disable_logging()
  end = Time.time()
  print('Total Pypet Simulation Time (min): ' + str((end - start)/60))
  return traj
  
  
def run(traj):  
  # initialize network and transfer parameters
  print('running trajectory...')
  random.seed()
  np.random.seed()
  net = RippleNetwork(AllParams=convertParams2Dict(traj)) # create a RippleNetwork instance !
  
  # run network
  input_params = convertParams2Dict(traj, condition = lambda x: '.input.' in x) # recover input_params dict
  net.run(input_params, seed_brian = traj.simulation.seed_brian, seed_network=traj.simulation.seed_network, \
          track_connectivity=traj.simulation.track_connectivity, record_window = traj.postproc.v_recordwindow_for_simulation)
  
  # add derived parameters to pypet trajectory
  traj.f_add_derived_parameter('stim_plot', net.stim_plot, comment='input stimulus (for plotting purposes)')
  for key in net.pars_deriv:
    if 'TA' in key:
      traj.f_add_derived_parameter(key, vars(net)[key.split('.')[-1]].values, comment='parameter derived within RippleNetwork.run')
    else:
      traj.f_add_derived_parameter(key, vars(net)[key.split('.')[-1]], comment='parameter derived within RippleNetwork.run')
    
  # analyse unit & network activity
  if traj.analysis.unit:
    net.UnitAnalysis(getSynch=traj.analysis.getSynch, offset=traj.analysis.offset)
  if traj.analysis.net:
    net.NetAnalysis(fmin = traj.analysis.fmin, fmax=traj.analysis.fmax, offset=traj.analysis.offset, getInstFreq=traj.analysis.getInstFreq, \
                    ifreq_targetwindow= traj.analysis.ifreq_targetwindow, ifreq_baselinewindow= traj.analysis.ifreq_baselinewindow, \
                    sat_cyclewise=traj.analysis.getSat_cyclewise, wavelet_f_expected=traj.analysis.wavelet_fexp)
  # analyze membrane potential dynamics:
  if traj.analysis.v_dynamics:
    try:
      oscillatory = net.input_params['level'] >= traj.linear_stability.Icrit_nA - 1e-3 # avoid numerical errors
    except:
      print('No linear stability analysis result available, will run get_average_cycle with oscillatory=True.')
      oscillatory = True
    net.get_average_cycle(oscillatory, offset = traj.analysis.offset) # averatge voltage histogram over one cycle was extracted and saved
  
  # now we can save memory by reducing the time window for which voltages are saved to just the default 25 ms:
  if not np.isclose(traj.postproc.v_recordwindow_for_simulation ,  traj.postproc.v_recordwindow).all(): 
    print('reducing voltage recording from ', traj.postproc.v_recordwindow_for_simulation, 'to ', traj.postproc.v_recordwindow)
    # reduce the voltage array that is saved to traj.postproc.v_recordwindow
    start = int((traj.postproc.v_recordwindow[0]  - traj.postproc.v_recordwindow_for_simulation[0]) / traj.dt)
    end   = int((traj.postproc.v_recordwindow[-1] - traj.postproc.v_recordwindow_for_simulation[0]) / traj.dt)
    net.v = net.v[start:end, :]
    

  net_data = vars(net)
  
  # --- store results ----------------------------------------------------------------
  res_meta = pd.read_csv('settings/pypet_results_metadata.csv', index_col='metadata').fillna('')  # load meta data
  for r in res_meta.columns:
    store = False
    if (r  in net_data.keys()) and (r not in traj.postproc.discard):
        if np.isscalar(net_data[r]):
          store = True
        elif len(net_data[r]):
          store = True
        if store:
          comment = res_meta.loc['comment',r]
          if len(res_meta.loc['unit',r]): # add unit if any
              comment = '['+res_meta.loc['unit',r]+'] '+comment
          traj.f_add_result('$.'+res_meta.loc['name_pypet', r], net_data[r], comment=comment)

  #-------------------------

  analysis_results = pypetOutput(net, traj.postproc.discard) # object with all measures assigned as attributes

  return analysis_results 

def postproc(traj, result_list, path_to_simulations = './simulations/'):
  if traj.input.shape=='flat' and traj.analysis.net:
      ''' postprocessing for stimulation with constant drive
      '''
      exploredParameters = traj.f_get_explored_parameters()
      
      if len(exploredParameters): # was there ANY parameter exploration?
        pars = list(exploredParameters)
        if 'parameters.postproc.par0' in traj.parameters.f_to_dict().keys():
          par0 = traj.postproc.par0 # lead parameter that becomes index of pandas data frames
          # put par0 first in the parameter list
          idx0 = pars.index(par0)
          pars[0], pars[idx0] = par0, pars[0]
        else: # pick randomly
          par0 = pars[0]       
        # fill dict with parameter ranges that were explored
        pars_range = {}
        for par in pars:
          pars_range[par] = traj.f_get(par).f_get_range()
    
        # initialize panda data frames    
        # scalar results
        measures = ['freq_net', 'freq_unit_mean', 'freq_unit_std', 'saturation', 'CV', 'ampl_mean', 'ampl_std',\
                    'coh_donoso', 'oS', 'coh_lindner', 'qfac', 'synchIdx', 'synchIdxCorr', 'STS', 'v_std_mean', 'v_std_std']
        column_labels = pars + measures
        scalarResults = pd.DataFrame(index= traj.f_get_run_names(), columns= column_labels, dtype=float).rename_axis(index='run')
        # if network size was not varied, sort unitrates into data frame as well:
        if traj.scale == 'micro' and ('parameters.network.Nint' not in exploredParameters.keys()): 
          unitrates_df = pd.DataFrame(index=np.arange(0,int(np.min([traj.Nint, traj.N_record_max]))), columns=traj.f_get_run_names())    
        else:
          unitrates_df = []
        
        # fill data frames
        for result_tuple in result_list: # loop over all runs of the given simulation
            # load results of resp. run
            run_idx = result_tuple[0]
            run_name = traj.f_get_run_names()[run_idx]
            traj.v_idx = run_idx
            totalResults = result_tuple[1]
            
            # fill in values of explored parameters in resp. run
            for par in pars: 
              scalarResults.loc[run_name, par] = pars_range[par][run_idx]
            # fill in results for all measures of resp. run
            unitrates_std = np.std(totalResults.unitrates)
            scalarResults.loc[run_name, measures] \
            = [totalResults.freq_net, totalResults.freq_unit, unitrates_std, totalResults.saturation, totalResults.CV] \
              + list(totalResults.ampl) + [totalResults.coh_donoso, totalResults.oS, totalResults.coh_lindner, totalResults.qfac]\
              + list(totalResults.synchIdx) + [totalResults.v_std_mean, totalResults.v_std_std ]

            if 'parameters.network.Nint' not in exploredParameters.keys(): 
              unitrates_df.loc[:,run_name] = totalResults.unitrates
        # sort data frame according to input level if applicable
        if par0 == 'parameters.input.level': #steady states
          scalarResults.sort_values(by=[par0], inplace=True)     
        
        # store data frame in readable table format in the same folder as trajectory
        scalarResults.to_hdf(pypet_get_trajectoryPath(traj.hash, path_to_simulations = path_to_simulations)+'results_overview_hash'+str(traj.hash)+'.hdf5',key='res',format='table', data_columns=True)
        # store data frame in trajectory
        traj.f_add_result('summary', scalarResults=scalarResults, unitrates=unitrates_df, comment='All scalar measures in one table + unitrates data frame if applicable')
        # release belief to be in last run
        traj.f_restore_default() 
  else:
      ''' postprocessing for transient stimulation
      '''    
      exploredParameters = traj.f_get_explored_parameters()
      iteration_key = 'parameters.simulation.seed_brian' 
      if (iteration_key in exploredParameters.keys()) and ('parameters.analysis.ifreq_targetwindow' not in exploredParameters.keys()):
        ifreq_n = int((traj.analysis.ifreq_targetwindow[-1]-traj.analysis.ifreq_targetwindow[0])/traj.dt)
        ifreq_all = np.zeros((len(exploredParameters[iteration_key]), ifreq_n))
        ipow_all = np.zeros((len(exploredParameters[iteration_key]), ifreq_n))
        icoh_all = np.zeros((len(exploredParameters[iteration_key]), ifreq_n))
        ripple_all = np.zeros((len(exploredParameters[iteration_key]), 2))
        ifreq_onset_all = np.zeros(len(exploredParameters[iteration_key]))
        Pthr = np.zeros(len(exploredParameters[iteration_key]))
        # fill data frames
        for result_tuple in result_list:
            # load results of resp. run
            run_idx = result_tuple[0]
            run_name = traj.f_get_run_names()[run_idx]
            traj.v_idx = run_idx
            totalResults = result_tuple[1]
            
            ifreq_all[run_idx,:] = totalResults.instfreq
            ipow_all[run_idx,:] = totalResults.instpower
            ripple_all[run_idx,:] = totalResults.ripple
            ifreq_onset_all[run_idx] = totalResults.freq_onset_inst
            Pthr[run_idx] = totalResults.Pthr
            if sum(totalResults.instcoherence):
              icoh_all[run_idx,:] = totalResults.instcoherence
        if not np.sum(np.sum(icoh_all)):
          icoh_all = []
        
        traj.f_add_result('summary', instfreq=ifreq_all, instpower=ipow_all, instcoh = icoh_all, ripple=ripple_all, \
                          ifreq_onset = ifreq_onset_all, Pthr=Pthr, \
                          comment='instantaneous frequency measures aligned in arrays for subsequent averaging')



class pypetOutput():
  def __init__(self, net, discard): 
    ''' 
    transfer result data from RippleNetwork object to pypetOutput container
    discard: list of variables that should be discarded to save memory
    '''
    # all possible measures that should potentially be stored
    measures_all = set(['freq_unit', 'unitrates', 'CV', 'synchIdx', 'freqs', 'power', 'freq_net', 'peakpower', 'power_0', 'power_fwhm', 'coh_donoso', 'oS', 'coh_lindner', 'qfac', \
                    'ampl', 'sat_t', 'sat', 'saturation', 'wspec', 'wspec_extent', 'instfreq', 'instpower', 'instcoherence', 'ripple',\
                    'freq_onset_inst', 'Pthr',  'ifreq_discr_t', 'ifreq_discr', 'Cii', 'Cie', 'v_std_mean', 'v_std_std'])
    
    # extract the Analysis results that are available (depends on the type of stimulation (constant or transient))
    measures_given = list(measures_all.intersection(set(vars(net).keys()))) # which are the measures that were actually computed and assigned to net?
    measures_missing = list(measures_all-set(vars(net).keys()))
    
    for attr in measures_given:
      setattr(self, attr, vars(net)[attr]) 
    for attr in measures_missing+discard:
      setattr(self, attr, []) 



#%% Brian2 tools

def drawSynapses(mode, Npre, Npost, p, autapses=True):
  ''' 
  If the synaptic connections should be non-random in some way (see mode), we generate pre- and postsynaptic indices here for subsequent use in Brian.
  mode: fix_indegree or fix_inoutdegree
  Npre: size of presyn population
  Npost: size of postsyn population
  p: connection probability
  autapses: allow connections from index i  to itself? (only relevant if pre and post-syn populations are identical)
  returns:
    i: list of presynaptic indices
    j: list of postsynaptic indices
    (to be used in Brians Synapse.connect(i=, j=))
    
  Note:
    CAREFUL WITH MODE 'fix_inoutdegree'! 
    My algorithm seems to introduce some unwanted structure around the edges of the connectivity matrix!!
  '''
  # in case of several modes:
  if 'fix_indegree' in mode:
    mode = 'fix_indegree'
  elif 'fix_inoutdegree' in mode:
    mode = 'fix_inoutdegree'
  
  Npre = int(Npre)
  Npost = int(Npost)  
  filename = 'C_'+mode+'_autap-'+str(autapses)+'_'+str(Npre)+'-'+str(Npost)+'-'+str(p)
  # check if connectivity matrix has already been computed
  if os.path.isfile('./ConnectivityMatrices/'+filename+'.npz'):
    print('loading saved connectivity data...', end='')
    with np.load('./ConnectivityMatrices/'+filename+'.npz', 'r') as data:
      i = data['ix_presyn']
      j = data['ix_postsyn']
      print('[done]')
    return i, j
  else: # compute from scratch and save for later
    print('drawSynapses...')
    if mode == 'fix_indegree':
      K_in = int(p*Npre) # in-degree
      j = np.repeat(np.arange(Npost), K_in) # postsyn indices
      i = np.zeros((Npost, K_in))+nan
      sources_all = np.arange(Npre) # list of all source neurons
      for m in range(Npost):
        mask = np.ones(Npre).astype(bool)
        if not autapses:
          mask[m] = False
        i[m,:] = np.random.choice(sources_all[mask], K_in, replace=False)
      i = i.flatten()
    elif mode=='fix_inoutdegree':
      print('initial synapse distribution...', end='')
      K_in = int(p*Npre) # in-degree
      K_out = int(p*Npost) # out-degree
      i = np.repeat(np.arange(Npre), K_out) # presyn indices
      j = np.zeros((Npre, K_out))+nan # postsyn indices
      targets_all = np.arange(Npost) # list of all target neurons
      counts = np.zeros(Npost) # count how often each target was hit already
      for n in range(Npre):
        if n%(Npre/10)==0:
          print(str(n)+'/'+str(Npre))
        # check which target cells are still available:
        mask = counts < K_in
        if autapses == False: # avoid autapses
          mask[n] = False
        targets = targets_all[mask] # still available as targets
        # randomly pick an output set for presyn neuron n:
        if targets.size >= K_out:
          j[n,:] = np.random.choice(targets, K_out, replace=False)
          counts[j[n,:].astype(int)] += 1
        else: # less than K_out different target neurons available, we need to switch in the end
          j[n,:targets.size] = targets # assign each of the left-over neurons once, leave the rest as nan for now
          counts[targets.astype(int)] += 1
      print('[done]')
      print('eliminating final nans by switching...', end=' ')
      mask = counts < K_in
      while sum(mask): # as long as there are target neurons left
        print(np.sum(np.isnan(j)), end='-')
        n = np.where(np.isnan(j))[0][0] # idx of presynaptic neuron that still needs a target (just not m)
        m = targets_all[mask][0] # target neuron to give to an existing pair
        # evaluate conditions that must apply for valid switching options
        if autapses == False:
          cond0 = (i!=m)*((j!=n).flatten()) # the presyn neuron cannot have index m and its current target cannot be n 
        else: 
          cond0 = np.ones(i.size).astype(bool)
        cond1 = np.isin(j, targets_all[mask], invert=True).flatten() # the switching target should be different from the leftover targets
        cond2 = np.isin(j, j[n,:], invert=True).flatten() # the switching target should be different from the targets already given in row n
        cond3 = np.repeat(np.sum(j==m, axis=1)==0, K_out)#.astype(bool) # m should not yet appear in the row where it gets placed
        # given that cond0 and cond1 are satisfied, pick a switching partner that also avoids duplicates:
        idx_switch = np.where(cond0*cond1*cond2*cond3)[0][0]
        n_switch = i[idx_switch].copy()
        # perform the switch
        m_switch = j[n_switch, idx_switch%K_out].copy() #.flatten()[idx_switch]
        j[n_switch, idx_switch%K_out] = m
        j[n, np.where(np.isnan(j))[1][0]] = m_switch       
        # update the counter
        counts[m] += 1
        mask = counts < K_in
      j = j.flatten()
      print('[done]')
    
    i = i.astype(int)
    j = j.astype(int)
    # test the result
    if np.sum(np.isnan(j)):
      raise ValueError('all nans should be gone')
    C = np.zeros((Npost, Npre))
    C[j,i] = 1
    indeg = np.sum(C, axis=1)
    outdeg = np.sum(C, axis=0)
    indeg_ok = np.unique(indeg).size == 1 and np.unique(indeg)==K_in
    if not indeg_ok:
        raise ValueError('Indegree wrong!')
    if mode=='fix_inoutdegree':
      outdeg_ok = np.unique(outdeg).size == 1 and np.unique(outdeg)==K_out
      if not outdeg_ok:
        raise ValueError('Outdegree wrong!')
    if not autapses:
      if sum(np.abs(i-j)<1):
        raise ValueError('Autapses found!')
    # save for REusage:
    np.savez('./ConnectivityMatrices/'+filename+'.npz', ix_presyn = i, ix_postsyn = j, \
           doc='Pre- and postsyn. indices to build connectivity matrix via C[post,pre] = 1')
    print('[done]')
  return i, j

def get_BrianEqs(neuronmodel, inputmode, IPSPshape, EPSPshape=None):
  ''' Setup the differential equations describing the model neuron and the synaptic coupling for use in Brian.
  '''
  record = ['v']
  
  # ---------- neuron-eqs -----------------------------------
  if neuronmodel == 'IF':
    neuron_eqs = '''
    dv/dt = Isyn/(C*pfarad) : volt (unless refractory)
    Isyn = Ie - Ii : amp
    '''
  else:
    neuron_eqs = '''
    dv/dt = (E_rest*mV - v)/(tm*ms) + Isyn/(C*pfarad) : volt (unless refractory) 
    Isyn = Ie - Ii : amp
    '''
    
  ## ---------- inh current -----------------------------------
  ode, on_pre_ii, record = get_synapse_ODE(IPSPshape, 'i', record, 'cb' in neuronmodel)
  neuron_eqs += ode
  syn_eqs_ii = '''J : 1'''  
  ## ---------- exc current -----------------------------------
  syn_eqs_ie = '' 
  on_pre_ie = ''
  if inputmode == 'spike':
    ode, on_pre_ie, record = get_synapse_ODE(EPSPshape, 'e', record, 'cb' in neuronmodel)
    neuron_eqs += ode
    syn_eqs_ie = '''J : 1''' 
  elif inputmode == 'DiffApprox_colored':
    neuron_eqs += '''
    dIe/dt = 1/(td_e*ms)*(-Ie + TA_Iext(t)*nA + sqrt(td_e*ms)*TA_Ie_sig(t)*nA*xi_indep) :amp''' # mean and noise are filtered
  elif inputmode == 'DiffApprox_coloredN':
    neuron_eqs += '''
    Ie = TA_Iext(t)*nA + Inoise : amp  
    dInoise/dt = 1/(td_e*ms)*(-Inoise + sqrt(td_e*ms)*TA_Ie_sig(t)*nA*xi_indep) :amp''' # only noise is filtered, Iext = mu*lambda
  elif inputmode == 'DiffApprox':
    # add white noise to voltage ode
    v_ode = neuron_eqs.split(': volt')[0]
    v_ode += ' + sqrt(tm*ms)*TA_Ie_sig(t)*nA/(C*pfarad)*xi_indep'
    # put neuron eqs back together and add input current 
    neuron_eqs = v_ode + ': volt' + neuron_eqs.split(': volt')[1]
    neuron_eqs += 'Ie = TA_Iext(t)*nA  :amp' # add constant current input
  elif inputmode == 'current':
    neuron_eqs += '''
    Ie = TA_Iext(t)*nA : amp'''
  elif inputmode == 'current_filt':
    neuron_eqs += '''
    dIe/dt = 1/(td_e*ms)*(-Ie + TA_Iext(t)*nA) :amp''' # 1exp filtering of input current
    record += ['Ie']
  elif inputmode == 'conductance':
    neuron_eqs += '''
    Ie =  TA_ge(t)*nS*(Ee*mV-v) : amp'''
  else:
    neuron_eqs += '''
    Ie = 0 :amp'''
    
  # ---------- thresh and reset-eqs -----------------------------------
  thr_eqs = 'v>Vthr*mV'
  reset_eqs = 'v=Vreset*mV'
  
  return neuron_eqs, syn_eqs_ii, syn_eqs_ie, thr_eqs, reset_eqs, on_pre_ii, on_pre_ie, record   

def get_synapse_ODE(PSPshape, syn, record, conductance = False):
  '''
  PSPshape: (string) which shape should the postsyn potential have
  syn: (string) e or i: exc or inh synapse?
  conductance: if True, return ODEs for conductance g, otherwise for current I
  '''
  if conductance:
    if PSPshape == 'delta':
      raise ValueError('delta PSPshape not defined for conductance-based synapse!')
    record += ['g'+syn]
  else:
    record += ['I'+syn]
  if PSPshape =='2exp':
    ode = '''
    dIsyn/dt = (-Isyn + x_syn)/(td_syn*ms) : amp
    dx_syn/dt = -x_syn/(tr_syn*ms) : amp
    '''
    on_pre = '''x_syn_post += J*nA'''
  elif PSPshape == '1exp':
    ode = '''
    dIsyn/dt = -Isyn/(td_syn*ms) : amp 
    '''
    on_pre = '''Isyn_post += J*nA'''
  elif PSPshape =='alpha':
    ode = ''' 
    dIsyn/dt = -Isyn/(ta_syn*ms) + x_syn/((ta_syn*ms)**2) : amp
    dx_syn/dt = -x_syn/(ta_syn*ms) : amp*second
    '''
    on_pre = '''x_syn_post += J*nA*ms'''
  elif PSPshape =='delta':
    record.remove('I'+syn)
    if syn=='e':
      ode = '''
      Ie = 0*nA : amp
      '''
      on_pre = 'v_post += J*mV'
    elif syn=='i':
      ode = '''
      Ii = 0*nA : amp 
      '''
      on_pre = 'v_post -= J*mV'
  
  if conductance:
    ode = ode.replace('I', 'g')
    ode = ode.replace('amp', 'siemens')
    if syn=='e':
      ode += '''
      Ie = ge*(Ee*mV-v) : amp
      '''
    else:
      ode += '''
      Ii = -gi*(Ei*mV-v) : amp
      '''
    on_pre = on_pre.replace('nA','nS')
  
  ode = ode.replace('syn',syn)
  on_pre = on_pre.replace('syn',syn)

  return ode, on_pre, record

def getStep(x, knots, level):
  '''
  generate Step Stimulus
  '''
  step = np.zeros(x.size)
  knots = np.asarray(knots)
  level = np.asarray(level)
  if not knots[0] == x[0]:
    knots = np.insert(knots, 0, x[0]) # add x(0) as first knot
  if not level.size == knots.size:
    raise ValueError('With x(0) as first knot, knots.size should equal levels.size!')
  for i in range(knots.size):
    step[x>= knots[i]] = level[i]
  return step  

def getRamp(x, knots = None, baseline = None, peak = None, slope = None, disp = True):
    '''
    x: array of all time points with desired time step
    knots: points of x where: 
            - ramp starts
            - plateau starts (*optional)
            - plateau ends (*)
            - ramp ends
    baseline: value at baseline level
    peak: value of plateau level (*)
    slope: slope of ramp (*) array of length 1 or 2
    
    returns a ramp with the specified baseline/peak levels, slopes and knot points
    '''   
    knots = np.asarray(knots)   
    
    # slope
    if slope==None: # slope not given 
      k1, k2, k3, k4 = knots
      m_up = (peak-baseline)/(k2-k1)
      m_down = (baseline-peak)/(k4-k3)    
    elif np.isscalar(slope): # different slopes for up/down
        m_up = slope # slope of rising part
        m_down = -slope
    else: # same slope for up and down
        m_up = slope[0] # slope of rising part
        m_down = slope[1] # slope of falling part
        
    # knots
    if knots.size < 4:
        if (not peak == None) and (not slope == None):
            k1, k4 = knots # only start of rise/end of fall given
            k2 = k1 + (peak-baseline)/m_up
            k3 = k4 + (peak-baseline)/m_down 
        else:
            raise ValueError('Either provide all 4 knots or provide peak and slope!')
    else:
        k1, k2, k3, k4 = knots
    
    # peak
    if peak==None:
        peak = baseline+m_up*(k2-k1)
        print(peak)
        
    ramp = np.piecewise(x, [x<k1, (k1<=x)&(x<k2), (k2<=x)&(x<k3), (k3<=x)&(x<k4), k4<=x], 
                         [baseline, lambda x: m_up*(x-k1)+baseline, peak, lambda x: m_down*(x-k3)+peak, baseline])
                         
    if disp:
        plt.figure()
        plt.plot(x,ramp)
        plt.ylim([0, 1.1*peak])
        
    return knots, baseline, peak, slope, ramp

def getStimulus(dt, Tsim_is_total_time = True, **params):
  ''' construct input stimulus profile (can be conductance or rate or anything)
  OPTIONS:
    Tinit: initialization time will be filled with constant "baseline" values, all remaining parameters refer to "time after initialization = 0"
    none: const. 0
    gaussian: params: peak, sig
    flat: params: level
    ramp: params: knots, base, peak (or slope)
    step: params: knots, levels
    Tsim_is_total_time: if True: Tsim = Tinit + simulation time (RippleNetwork setting), else: Tsim = simulation time (PCO network setting)
  returns s (same length as time) 
  '''
  if 'Tinit' in params.keys():
    if Tsim_is_total_time:
      params['Tsim'] -= params['Tinit'] # now Tsim is the pure simulation time only
    params_stim = params.copy()
    params_stim.pop('Tinit', None)
    s_stim = getStimulus(dt, **params_stim)
    if params['shape']=='flat':
      return s_stim # scalar only
    else:
      s_init = np.ones(int(params['Tinit']/dt))*params['baseline']
      s = np.concatenate((s_init, s_stim))
  else:
    time = np.arange(0, params['Tsim'], dt)
    if params['shape'] == 'flat':
      s = params['level'] # scalar
    elif params['shape'] == 'none':
      s = np.zeros(time.size)
    elif params['shape'] == 'other':
      # chance to provide special stimulus, already preconstructed in ripple_main
      s = params['stim']
    elif params['shape'] == 'gaussian':
      mu = params['Tsim']//2 #int(np.round(np.mean(time)))
      s = params['peak']*np.exp(-(time-mu)**2/(2*params['sig']**2))
    elif params['shape'] == 'gaussian_shifted':
      mu = int(np.round(np.mean(time)))
      s = (params['peak']-params['baseline'])*np.exp(-(time-mu)**2/(2*params['sig']**2)) + params['baseline']
    elif params['shape'] == 'ramp':
      if 'knots' not in params.keys():
        t0 = params['Tsim']/2
        knots = [t0-0.5*params['plateau_time']-params['ramp_time'], \
                 t0-0.5*params['plateau_time'], \
                 t0+0.5*params['plateau_time'],\
                 t0+0.5*params['plateau_time']+params['ramp_time'] ]
        if knots[0] < 0:
          raise ValueError('Increase simulation time to have room for the full ramp!')
      else:
        knots = params['knots']
      s = getRamp(time, knots = knots, baseline = params['baseline'], peak = params['peak'])[4]
    elif params['shape'] == 'ramp_asym':
      if 'knots' not in params.keys():
        if 't_on' in params.keys():
          t0 = params['t_on']
        else:
          t0 = (params['Tsim']-(params['ramp_time_up']+params['plateau_time']+params['ramp_time_down'])) / 2 # beginning of ramp up
        knots = [t0, \
                 t0+params['ramp_time_up'], \
                 t0+params['ramp_time_up'] + params['plateau_time'],\
                 t0+params['ramp_time_up'] + params['plateau_time']+params['ramp_time_down'] ]
        if knots[0] < 0:
          raise ValueError('Increase simulation time to have room for the full ramp!')
      else:
        knots = params['knots']  
      s = getRamp(time, knots = knots, baseline = params['baseline'], peak = params['peak'])[4]
    elif params['shape'] == 'step':
      s = getStep(time, knots=params['knots'], level=params['level'])
    elif params['shape'] == 'cosine':
      s = params['mean'] + params['amp']*np.cos(2*pi*params['freq']*time/1000)
    elif params['shape'] == 'pulse_malerba':
      s = params['peak']/(1+np.exp(-(time-params['t_on'])/params['tau']))/(1+np.exp((time-params['t_off'])/params['tau']))
  return s

def make_TimedArray(x, Tsim, dt):
  if np.isscalar(x):
    x = b2.TimedArray(np.array([x]), dt=Tsim*b2.ms)
  else:
    x = b2.TimedArray(x, dt=dt*b2.ms)
  return x

#%% Ripple Network Simulations

class RippleNetwork(): 
  ''' This class is used to setup and simulate a wide range of inhibitory network models and analyse their dynamics. '''
  def __init__(self, parent=None, changeParams={}, AllParams={}): 
    '''
    parent: str. Parent-network (Donoso2018, Brunel2003, mymicro) determining the default parameter setting (see settings/StandardParams_all_values.csv)
    changeParams: dict. Parameters to be changed w.r.t. default setting.
    AllParams: dict. Alternative to changeParams: Pass a complete dictionary of parameters (used within pypet wrapper function).
    '''
    if not len(AllParams.items()): # we need to piece together the full parameter list
      par_default = loadStandardParams(parent) # load default parameters for parent network
      AllParams = {**par_default, **changeParams} # replace parameters that should be changed
    else: 
      pass  # AllParams are provided by pypet (changeParams was already done)
    # assign all parameter values as attributes to the class
    for key in AllParams:
      setattr(self, key, AllParams[key])  
    # derive additional parameters based on the ones that have been set
    self.pars_deriv = self.deriveParams()
    self.name= self.parent+'___WITH_'+dict2str(changeParams, equals='-') 

  def deriveParams(self):
    '''
    -- scale synaptic weights if required 
    -- derive postsynaptic response to presynaptic spiking for Brian (increment J used in Brian Synapses)
    '''
    pypet_pars_deriv = [] # keep track of new parameters for later storage in pypet trajectory 
    # --- EPSPshape ------------------------------------------------
    if 'EPSPshape' not in self.__dict__.keys():
        if self.inputmode=='spike':
          self.EPSPshape = self.IPSPshape # if no EPSPshape specified, take the same as for IPSP
          pypet_pars_deriv += ['neuron.synapse.EPSPshape']
        else:
          self.EPSPshape = '' # no EPSPshape needed (e.g. if inputmode=current)
          
    # --- scaling inhibitory weights ------------------------------------------------
    if self.scaleWeights: # set to False if syn weights should remain unchanged w.r.t network size and connectivity
      # universal scaling of the inhibitory synaptic strength: 
      if self.p_ii: # --> no scaling in cases where we look at uncoupled interneurons
        # load standard values for p_ii and Nint, for which the syn. strength was defined
        stp = loadStandardParams(self.parent)
        scaling_factor = stp['p_ii']*stp['Nint']/(self.p_ii*self.Nint) # this scaling factor is 1 unless we change p_ii or Nint from the default values
        if scaling_factor != 1:
          print('scaling inhibitory weights by {}'.format(scaling_factor))
          if 'cb' in self.neuronmodel: # conductance-based synapses
            if self.neuronmodel== 'Brunel2003':
              self.gsyn_i *= scaling_factor
              pypet_pars_deriv += ['gsyn_i']
            else:
              self.gpeak_i *= scaling_factor
              pypet_pars_deriv += ['gpeak_i']
          else: # current-based synapses
            self.IPSPint *= scaling_factor
            self.IPSCint *= scaling_factor
            self.Ipeak_i *= scaling_factor
            pypet_pars_deriv += ['IPSPint', 'IPSCint', 'Ipeak_i']
          
    # --- derive PSP increments used in brian equations ------------------------------------------------
    self.J_i, pypet_pars_deriv = self.deriveJ('i', pypet_pars_deriv)
    if (self.inputmode=='spike'):# and ('J_e' not in self.__dict__.keys()):
      self.J_e, pypet_pars_deriv = self.deriveJ('e', pypet_pars_deriv)
      
    return pypet_pars_deriv
  
  def deriveJ(self, syn, pypet_pars_deriv):
    '''
    derive the increment of postsynaptic conductance/current/voltage (depends on model architecture) 
    in response to a presynaptic spike
    INPUT:
      syn: 'e' or 'i' for excitatory or inhibitory
      pypet_pars_deriv: list. Keep track of newly derived parameters for subsequent storage in pypet trajectory.
    '''      
    if syn=='e':
      PSPint = self.EPSPint
      PSCint = self.EPSCint
      PSPshape = self.EPSPshape
    elif syn=='i':
      PSPshape = self.IPSPshape
      if not 'cb' in self.neuronmodel:
        PSPint = self.IPSPint
        PSCint = self.IPSCint
        
      
    if PSPshape: # EPSPshape can be '', then no Je needs to be calculated
        print('deriving J_{}...'.format(syn), end='')
        pypet_pars_deriv += ['J_'+syn] # for pypet: store keys of derived parameters that should be stored in the trajectory
        td = vars(self)['td_'+syn]
        tr = vars(self)['tr_'+syn]
          
        if self.neuronmodel in ['IF', 'LIF', 'GIF']: # current based add if
          if PSPshape=='2exp': #nA
            J = PSCint/tr # = Ipeak*s*(td-tr)/tr #Ipeak*s
          elif PSPshape == '1exp': #nA
            J = PSCint/td #Ipeak*s*(td-tr)/td
          elif PSPshape == 'alpha': # nA*ms
            J = PSCint 
          elif PSPshape == 'delta': #mV
            J = PSPint/self.tm 
        elif self.neuronmodel in ['cbLIF', 'cbGIF', 'cbLIFrandthr', 'cbAdExp']: # conductance-based
          if PSPshape != '2exp':
            raise ValueError('Increment J only implemented for 2exp conductances!')
          if self.parent == 'Brunel2003': # nS
            s = self.tm/(td-tr) 
            gsyn = vars(self)['gsyn_'+syn]
            J = gsyn*s*(td-tr)/tr  #self.gsyn*self.tm/tr, nS
          else: # nS
            s = 1/((td/tr)**(tr/(tr-td))-(td/tr)**(td/(tr-td))) # normalization of 2exp conductance
            gpeak = vars(self)['gpeak_'+syn]                  
            J = gpeak*s*(td-tr)/tr #gpeak*s
    else: 
        J = nan
    return J, pypet_pars_deriv
  


  
  def run(self, input_params, seed_brian=None, seed_network=None, track_connectivity= False, record_window=[]):
    '''
    run the network simulation
    INPUTS:
      input_params: dict. Specify input properties (shape, strength etc)
      seed_brian:   int. Seed for REALIZATION of noise, etc
      seed_network: int. Seed for network ARCHITECTURE, can be specified separately to run several noise realizations (different seed_brian) for the same I network 
      track_connectivity: bool. Whether or not to record the I-I connectivity matrix (only relevant if p_ii < 1).
      record_window: list. Time window for which state variables should be recorded.
    '''
    # setup network
    self.track_connectivity= track_connectivity
    print('constructing network...')
    start = Time.time()
    Tsim = input_params['Tsim']
    self.Tsim = Tsim
    self.input_params = input_params
    b2.start_scope()
    b2.defaultclock.dt = self.dt*ms
    time = np.arange(0, Tsim, self.dt)
    self.time = time
    if not len(record_window):
      self.record_window = [0, Tsim]
    else:
      self.record_window = record_window
    
    # get Brian Equations
    neuron_eqs, syn_eqs_ii, syn_eqs_ie, thr_eqs, reset_eqs, on_pre_ii, on_pre_ie, record \
    = self.get_BrianEqs()   
    
    print(self.brianeqs,'\nJ_i: ', self.J_i)
    if 'J_e' in vars(self).keys():
      print('J_e: ', self.J_e)
      
    # overwrite record, if provided as input parameter:
    if 'record_micro' in self.__dict__.keys():
      if 'NONE' in self.record_micro:
        record = ()
      else:
        record = self.record_micro
    else:
      self.record_micro = tuple(record) # default variables to be recorded depend on Brian Eqs
      
    # width of initial voltage distribution
    if self.Vinit == 'narrow':
      v0 = [self.E_rest-5, self.E_rest+5] # 5
    elif self.Vinit == 'wide':
      v0 = [np.min([self.Vreset,self.E_rest]), self.Vthr]
    elif self.Vinit == 'identical':
      v0 = [self.E_rest, self.E_rest]
    
    # --- NETWORK hard structure -----------------------------------------------------------------------------------------------------
    if seed_network:
      b2.seed(seed = seed_network)
    else:
      b2.seed(seed = seed_brian)
#      np.random.seed(seed_brian)
    # interneurons
    I = b2.NeuronGroup(self.Nint, model=neuron_eqs, threshold=thr_eqs, reset=reset_eqs, method=self.integrationmethod, refractory=self.tref*ms, name='I')
    
    # synapses I-I
    S=[]
    S.append(b2.Synapses(I, I, model=syn_eqs_ii, on_pre=on_pre_ii, delay = self.tl*ms, name='Sii'))
    if self.p_ii:
        if ('fix_indegree' in self.mode) or ('fix_inoutdegree' in self.mode): #=='fix_inoutdegree':
          i,j = drawSynapses(self.mode, self.Nint, self.Nint, self.p_ii, autapses=self.autapses)
          S[-1].connect(i=i, j=j)
        else: # random connectivity
          if self.autapses:
            if self.p_ii==1:
              S[-1].connect() # save time for large networks
            else:
              S[-1].connect(p=self.p_ii)
          else:
            if self.p_ii==1:
              S[-1].connect(condition='i!=j') # save time for large networks
            else:
              S[-1].connect(p=self.p_ii, condition='i!=j')
        if self.CV_Ji: # vary strength, but clip to 0 to avoid violation of Dale's law
          from brian2 import clip, Inf
          J_i_aux = self.J_i # bring values to current namespace such that synapse equation below can be evaluated
          CV_Ji_aux = self.CV_Ji
          S[-1].J = 'clip(J_i_aux  + J_i_aux *CV_Ji_aux *randn(), 0, Inf)'
        else:
          S[-1].J = self.J_i
    else:
      S[-1].active = False
      
    # --- INITIALIZATION (realization of network with hard structure as defined above) ------------------------------------------------------
    if seed_network: # change now to the seed of the current simulation (or pick a new one, if none was given)
      b2.seed(seed = seed_brian)
      np.random.seed(seed_brian)
    if self.Vinit == 'gaussian-4-revision':
      I.v = np.random.normal(loc = self.E_rest - 3*(self.Vthr-self.E_rest), scale = 2.62, size=self.Nint)*mV # hard-coded for just one experiment
      print('v0 min, max: ', np.min(I.v/mV), np.max(I.v/mV))
    else:
      I.v = np.random.uniform(v0[0], v0[1], size=self.Nint)*mV
    if self.neuronmodel == 'cbLIFrandthr':
      I.vthr = np.random.normal(loc=self.Vthr, scale=self.thr_sig_cbLIFrandthr, size=self.Nint)*mV
    if 'I_dc' in self.brianeqs: # constant dc current, normally distributed across units
      self.I_dc = np.random.normal(loc=input_params['Idc_mean'], scale= input_params['Idc_std'], size=self.Nint)
      I.I_dc = self.I_dc.copy()
      
    # add interneuron Monitors
    print('Recording spikes and {} for {} units.'.format(record, int(np.min([self.Nint, self.N_record_max]))))
    Ispike = SpikeMonitor(I[:int(np.min([self.Nint, self.N_record_max]))], name='Ispike')
    if len(record):
      Istate = StateMonitor(I, record, name='Istate', record=list(np.arange(np.min([self.Nint, self.N_record_max]))))
    LFP = PopulationRateMonitor(I, name='LFP')
    # minimal network components
    if len(record):
      sim = b2.Network(I, Ispike, Istate, LFP)
    else:
      sim = b2.Network(I, Ispike, LFP)
    
    self.get_BrianInputs() # create TimedArrays and all necessary input parameter settings and add them as attributes
    
    # add excitation and assemble all network components
    if self.inputmode=='spike':
      if self.tref_pyr:
        P = b2.SpikeGeneratorGroup(self.Npyr, self.Pspk_indices, self.Pspk_times*ms)
      else:
        P = PoissonGroup(self.Npyr, rates = 'TA_lambda(t)*Hz', name='P') 
      Pspike = SpikeMonitor(P, name='Pspike')
      LFP_P = PopulationRateMonitor(P, name='LFP_P')
      S.append(Synapses(P, I, model=syn_eqs_ie, on_pre=on_pre_ie, delay = self.tl*ms, name='Sie'))
      if 'independent-inputs' in self.mode: #== :
        S[-1].connect(j='i') # 1 input cell per interneuron, no overlaps
      elif 'fix_indegree' in self.mode: # fixed in-degree
        i,j = drawSynapses(self.mode, self.Npyr, self.Nint, self.p_ie)
        S[-1].connect(i=i, j=j)
      elif 'fix_inoutdegree' in self.mode:
        i,j = drawSynapses(self.mode, self.Npyr, self.Nint, self.p_ie)
        S[-1].connect(i=i, j=j)
      else:
        S[-1].connect(p=self.p_ie)
      S[-1].J = self.J_e
      if 'globalsignal' in self.mode: # Npyr=p_ie=1
        # add private noise neurons:
        Pnoise = PoissonGroup(self.Nint, 1200*Hz, name='Pnoise')  
        S.append(Synapses(Pnoise, I, model=syn_eqs_ie, on_pre=on_pre_ie, delay = self.tl*ms, name='Sinoise'))
        S[-1].connect(j='i') # private noise neuron for each interneuron
        S[-1].J = self.J_e
        sim.add(Pnoise, P, S, Pspike)
      else: # most cases
        sim.add(P, S, Pspike, LFP_P)
    else:
      sim.add(S)
    
    # check in-degree
    if not self.test:
      if self.track_connectivity:
        print('inh in-degree of interneurons:\n',np.mean(sim['Sii'].N_incoming), np.std(sim['Sii'].N_incoming))
        print('out-degree of interneurons:\n' , np.mean(sim['Sii'].N_outgoing), np.std(sim['Sii'].N_outgoing))
        if self.inputmode == 'spike':
          print('exc in-degree of interneurons:\n',np.mean(sim['Sie'].N_incoming), np.std(sim['Sie'].N_incoming))
          print('out-degree of pyramids:\n' , np.mean(sim['Sie'].N_outgoing), np.std(sim['Sie'].N_outgoing))
        Npreset = self.Nint*self.autapses + (self.Nint-1)*(self.autapses==False) # size of inh presynaptic pool
        if np.abs(np.mean(sim['Sii'].N_incoming)/Npreset-self.p_ii)>0.02:  # ACHTUNG: hier zählt N_incoming alle in-degrees so oft wie ein Neuron postsyn. Target ist! Funktioniert trotzdem für grobe Fehlerabschätzung
          # I added Nint-1 for the case that N=2 only, then avoiding autapses means, p_ii -> 1 connection only
          raise ValueError('In-degree deviating from target value by >2% !')
        if self.inputmode=='spike' and ('independent-inputs' not in self.mode):
          if np.abs(np.mean(sim['Sie'].N_incoming)/self.Npyr-self.p_ie)>0.02:
              raise ValueError('In-degree deviating from target value by >2% !')
    
    if 'fix_indegree' in self.mode:
      print('fixed: inh in-degree: ', np.unique(sim['Sii'].N_incoming))
      if (self.p_ii and np.std(sim['Sii'].N_incoming)!=0):
        raise ValueError('Fixing of inh in-degree did not work!')
      if self.inputmode=='spike':
        print('fixed: exc in-degree: ', np.unique(sim['Sie'].N_incoming))
        if (self.p_ie and np.std(sim['Sie'].N_incoming)!=0):
          raise ValueError('Fixing of exc in-degree did not work!')
        
    
    end = Time.time()
    print('time for building network (s): ' + str(end - start))
    
    # run simulation
    print('running simulation...')
    namespace = vars(self) # any attributes of the RippleNetwork object are available in the Brian run namespace
    
    if len(record):
      if self.record_window[0] > self.dt: # initial period without recording
        sim['Istate'].active = False
        sim.run(self.record_window[0]*ms, namespace=namespace)
        
      # simulate with recording
      sim['Istate'].active = True
      sim.run((self.record_window[1]-self.record_window[0])*ms, namespace=namespace)
      
      
      if self.record_window[1] < Tsim: # final period without recording
        sim['Istate'].active = False
        sim.run((Tsim-self.record_window[1])*ms, namespace=namespace)
    
    else:
      sim.run(Tsim*ms, namespace=namespace)
    
    
    print('evaluating simulation...')
    self.evaluate_simulation(sim)
    return 
  
  def get_BrianEqs(self):
    '''
    setup the equations that define the model network and are used to integrate its dynamics in Brian2
    '''
    if 'EPSPshape' in vars(self).keys():
      EPSPshape = self.EPSPshape
    else:
      EPSPshape = ''
    neuron_eqs, syn_eqs_ii, syn_eqs_ie, thr_eqs, reset_eqs, on_pre_ii, on_pre_ie, record\
    = get_BrianEqs(self.neuronmodel, self.inputmode, self.IPSPshape, EPSPshape=EPSPshape)
      
    # save Brian Eqs in one long string for storage through pypet
    self.brianeqs = 'Neuron Equations:\n'+neuron_eqs+'\n\nSynaptic Eqs:\n' + syn_eqs_ii+'\n'+ syn_eqs_ie+'\n'+on_pre_ii+'\n'+on_pre_ie+\
                           '\n\nThreshold Eqs:\n'+thr_eqs+'\n\nReset Eqs:\n'+reset_eqs+'\n\nVariables available for recording:\n'+str(record)
    self.pars_deriv += ['brianeqs']
    return neuron_eqs, syn_eqs_ii, syn_eqs_ie, thr_eqs, reset_eqs, on_pre_ii, on_pre_ie, record
  

  def get_BrianInputs(self):
    ''' 
    setup the external drive depending on the settings given by inputmode and input_params
    -- also takes care of rescaling since the Brian Eqs are all written assuming an input in current units (nA)
    '''
    # get stimulus
    s = getStimulus(self.dt, **self.input_params)
    self.stim_plot = s # remember this output for plotting purposes
    if self.inputmode == 'spike':
      if not (('globalsignal' in self.mode) or ('independent-inputs' in self.mode)):
          s = s/(self.Npyr*self.p_ie) # scale to get intensity of a single pyr input cell
      if self.tref_pyr: # implement refrac period in input Poisson spiking: calculate all spike times and hand them to Brian
        raise ValueError('deprecated')
      else:
        self.TA_lambda = make_TimedArray(s, self.Tsim, self.dt)
    # --- mean current ------------------------------------------
    if 'TA_Iext' in self.brianeqs:
      if self.input_params['unit'] == 'spks/sec': # Diff Approx
        Iext = self.C/1000*self.EPSPint/self.tm*s/1000 # nF*mVms/ms/ms = nA
      elif self.input_params['unit'] == 'nA':
        Iext = s
      else:
        raise ValueError('Input must be provided either directly in nA or in spks/sec (internal DiffApprox)!')
      self.TA_Iext = make_TimedArray(Iext, self.Tsim, self.dt)
      self.pars_deriv += ['input.TA_Iext']
    # --- noise ------------------------------------------
    if 'TA_Ie_sig' in self.brianeqs:
      if 'Ie_sig' in self.input_params.keys():
        self.TA_Ie_sig = make_TimedArray(self.input_params['Ie_sig'], self.Tsim, self.dt)
      elif self.input_params['unit'] == 'spks/sec': # derive noise intensity from spiking intensity
        if self.inputmode == 'DiffApprox':
          noise = 'white'
        elif self.inputmode == 'DiffApprox_colored':
          noise = 'colored'
        
        if noise=='white': #(self.inputmode == 'DiffApprox') or ('chizhov0' in self.hazardmode): # white noise
          Ie_sig = self.C/1000*self.EPSPint/self.tm/np.sqrt(self.tm)*np.sqrt(s/1000) # nF*mVms/ms/sqrt(ms)/sqrt(ms) = nFmV/ms = nA
          print(Ie_sig)
        elif noise=='colored': #('DiffApprox_colored' in self.inputmode) or ('chizhov1' in self.hazardmode): # colored noise
          Ie_sig = self.EPSCint/np.sqrt(self.td_e)*np.sqrt(s/1000) # nAms/sqrt(ms)/sqrt(ms) = nA
        self.TA_Ie_sig = make_TimedArray(Ie_sig, self.Tsim, self.dt)
      else:
        raise ValueError('Please provide noise intensity Ie_sig as input parameter!')
      self.pars_deriv += ['input.TA_Ie_sig']
    elif 'Ie_sig' in self.brianeqs:
      self.Ie_sig = self.input_params['Ie_sig'] # constant scalar value for all units
    # --- conductance ------------------------------------------
    if 'TA_ge' in self.brianeqs:
      if self.input_params['unit'] == 'nS':
        self.TA_ge = make_TimedArray(s, self.Tsim, self.dt)
      else:
        raise ValueError('Provide input conductance in nS!')
    return  
  
  def evaluate_simulation(self, sim):
    '''
    remove Brian units before storage of the simulation results
    keep only variables that were indicated for storage
    INPUT:
      sim: Brian2 Network object after the simulation has been run.
    '''
    # start = Time.time()
    LFP_pyr, Pspktrains, v, ge, gi, Ie, Ii, Cii, Cie, v_std_mean, v_std_std \
    = [],[],[],[],[],[],[],[],[],nan, nan
    
    # --- absolute minimum: record population activity-----------------------------------------
    Ispike = sim['Ispike'] # monitor, incl its functions and attributes
    LFP = sim['LFP']
    
    self.spktrains = Ispike.spike_trains()  # maybe use get_states() also to turn Brian object into dictionary??
    for i in range(int(np.min([self.Nint, self.N_record_max]))):    
      self.spktrains[i] = self.spktrains[i]/ms # remove unit   
      self.spktrains[str(i)] = self.spktrains.pop(i) # make keys to strings for pypet use
    self.Icount = Ispike.count[:] # array of total # spikes from each neuron
    self.LFP_raw = LFP.rate/Hz
    self.LFP_smooth = LFP.smooth_rate(window='gaussian', width=.3*ms)/Hz
    
    # record connectivity matrices
    if self.track_connectivity:
      if self.p_ii:
        Cii = np.zeros((int(self.Nint), int(self.Nint))) # row: post, column: pre, i.e. row-sum=in-degree
        Cii[sim['Sii'].j, sim['Sii'].i] = sim['Sii'].J #1
      else:
        Cii = []
      if self.inputmode=='spike':
        Cie = np.zeros((int(self.Nint), int(self.Npyr)))
        try:
          Cie[sim['Sie'].j, sim['Sie'].i] = sim['Sie'].J #1
        except: 
          pass
      else:
        Cie = []
    
    # --- if required, store additional state variables --------------------------------------
    if 'NONE' not in self.record_micro:
      Istate = sim.get_states()['Istate'] # dictionary
      
      # store voltage traces etc
      if 'v' in self.record_micro:
        v = Istate['v']/mV # time x neurons
        if self.input_params['shape'] == 'flat':
          v_std = np.std(v[int(50/self.dt):, :], axis=1) # std across neurons
          v_std_mean = np.mean(v_std)
          v_std_std = np.std(v_std)
      if 'ge' in self.record_micro:
        ge = Istate['ge']/nS
      if 'gi' in self.record_micro:
        gi = Istate['gi']/nS
      if 'Ie' in self.record_micro:
        Ie = Istate['Ie']/nA
      if 'Ii' in self.record_micro:
        Ii = Istate['Ii']/nA
      if 'pspk' in self.record_micro:
        self.pspk = Istate['pspk']
      if 'rnd' in self.record_micro:
        self.rnd = Istate['rnd']
      if 'A' in self.record_micro:
        self.A = Istate['A']
      if 'B' in self.record_micro:
        self.B = Istate['B']
      if 'f' in self.record_micro:
        self.f = Istate['f']/b2.Hz
      if 'T' in self.record_micro:
        self.T = Istate['T']
      if 'Tdot' in self.record_micro:
        self.Tdot = Istate['Tdot']*b2.ms     
      if 'sx2' in self.record_micro:
        self.sx2 = Istate['sx2']/(b2.mV**2)  
      if 'w' in self.record_micro:
        self.w = Istate['w']/nA # axis0: time                      

      if self.inputmode == 'spike' and not self.test: # record exc input spike trains
        Pspike = sim['Pspike']
    #    self.Praster = [Pspike.t/ms, Pspike.i]
        Pspktrains = Pspike.spike_trains() 
        for i in range(int(self.Npyr)):    
          Pspktrains[i] =  Pspktrains[i]/ms # remove unit
          Pspktrains[str(i)] = Pspktrains.pop(i) # make keys to strings for pypet use
        LFP_pyr = sim['LFP_P'].rate/Hz  #smooth_rate(window='gaussian', width=.3*ms)/Hz


    # assign to network object
    self.LFP_pyr, self.Pspktrains, self.v, self.ge, self.gi, \
    self.Ie, self.Ii, self.Cii, self.Cie, self.v_std_mean, self.v_std_std \
    = LFP_pyr, Pspktrains, v, ge, gi, Ie, Ii, Cii, Cie, v_std_mean, v_std_std
    return  
    
 
  def UnitAnalysis(self, tau=1, binsize=1, getSynch=1, offset=50):
    ''' 
    Analyse the spiking statistics.
    INPUTS:
      tau: size of sliding window used for analysis of spike synchrony
      binsize: used for analysis of spike synchrony
      getSynch: bool. Whether spike synchrony should be analyzed.
      offset: float. [ms] Initial time window to exclude from analysis.
    OUTPUTS (stored as new attributes):
      unitrates: firing rates of all neurons
      freq_unit: mean unit firing rate in Hz
      CV: average CV of ISIs 
      synchIdx: various indices quantifying spike synchrony
    '''
    if not self.input_params['shape']=='flat':
      getSynch = False # the synch idx only makes sense in stationary regimes
    if offset>self.Tsim:
      offset=0
    self.freq_unit, self.unitrates, self.CV, self.synchIdx \
    = f_UnitAnalysis(self.spktrains, self.Tsim, self.dt, tau=tau, binsize=binsize, getSynch=getSynch, offset=offset)   
  
  def NetAnalysis(self, fmin = 30 ,offset=50, fmax=350, getInstFreq=False, df=1, k='max', ifreq_targetwindow=[], ifreq_baselinewindow=[50, 150], \
                  sat_cyclewise=False, Pthr=nan, coh_thr=1e-2, wavelet_f_expected=200):
    ''' 
    Analyze network activity.
    INPUTS:
      fmin: float [Hz]. Minimum for dominant frequency in power spectral density.
      offset: float. [ms] Initial time window to exclude from analysis.
      fmax: float [Hz]. Maximal frequency for wavelet spectrogram.
      getInstFreq: bool. [optional] Whether to analyse the instantaneous frequency over time.
      df: float [Hz]. Resolution for power spectral density of population activity.
      k: str. How many snippets to cut the population activity into when calculating the power spectral density (see get_PSD)
      ifreq_targetwindow, ifreq_baselinewindow: lists. Target and baseline windows for analysis of instantaneous frequency.
      sat_cyclewise: bool. Whether to compute cyclewise saturation.
      Pthr: power threshold.
      coh_thr: coherence threshold (DEPRECATED)
      wavelet_f_expected: float [Hz]. Rough estimate of the expected oscillation frequency. Used to set wavelet spectrogram parameter sig.
    OUTPUTS:
      analysis for constant drive:
        freqs, power: power spectral density of population activity 
        freq_net: network frequency
        saturation
        ampl: average amplitude of the population rate oscillation
        diverse coherence measures
      analysis of instantaneous frequency for time-dependent drive:
        wspec: np.array. wavelet spectrogram
        wspec_extent: boundaries of wavelet spectrogram for plotting
        instfreq, instpower: continuous instantaneous frequency and power
        ifreq_discr_t, ifreq_discr: discrete instantaneous frequency 
    '''
    if self.scale=='micro':
      spktrain = self.spktrains
    else:
      spktrain = {}
    stationary = self.input_params['shape']=='flat'
    
    try:
      funit_aux = self.freq_unit
    except:
      funit_aux = nan
      
    if stationary:
      self.freqs, self.power, self.freq_net, self.peakpower, self.power_0, self.power_fwhm, self.coh_donoso, self.oS, self.coh_lindner, self.qfac, \
      self.ampl, self.sat_t, self.sat, self.saturation \
      = f_oscillation_analysis_stationary(self.LFP_raw, self.LFP_smooth, self.dt, spktrain, self.scale, \
                                          df=df, k=k, fmin=fmin, sat_cyclewise=sat_cyclewise, offset=offset, freq_unit=funit_aux, coh_thr=coh_thr)
    if not stationary or getInstFreq:
      if stationary:
        expected_freq = self.freq_net
      else:
        expected_freq = wavelet_f_expected
      self.wspec, self.wspec_extent, self.instfreq, self.instpower, self.ripple, self.freq_onset_inst, \
      self.instcoherence, self.Pthr, self.ifreq_discr_t, self.ifreq_discr \
      = f_oscillation_analysis_transient(self.LFP_smooth, self.dt, fmax= fmax, df=df, \
                                         baseline_window=ifreq_baselinewindow, target_window = ifreq_targetwindow, Pthr=Pthr, \
                                         expected_freq = expected_freq, fmin= fmin, stationary=stationary)
    return
  
  def getSaturation(self, offset=50, cyclewise=False):
    '''
    Compute saturation.
    INPUTS:
      offset: float. [ms] Initial time window to exclude from analysis.
      cyclewise: bool. If True, a cyclewise estimate is calculated, checking separately in each population spike, how many units spiked.
    OUTPUT:
      saturation: average saturation (freq_unit / freq_net)
      sat_t, sat: time points of individual cycles and associated, cycle-wise, saturation 
    '''
    self.sat_t, self.sat, self.saturation \
    = f_getSaturation(self.LFP_smooth, self.spktrains, self.Tsim, self.dt, self.scale, freq_unit=self.freq_unit, freq_net=self.freq_net, offset=offset, cyclewise=cyclewise)
    return 
  
  def get_average_cycle(self, oscillatory, offset=50, dv=.1, n=21): 
    
    self.popspk_av, self.v_av, self.v_hist, self.v_bincenters, self.gauss_fit_params, self.sample, \
    self.v_av_mumin, self.v_av_sigmin, self.v_std, self.t_on, self.t_off, self.v_av_mumin_sd \
    = get_average_cycle(self.v, self.LFP_smooth, self.freq_net, self.Vthr, self.E_rest, self.dt, oscillatory, self.record_window, offset=offset, dv=dv, n=n)
    return

#%% Analysis of ripple simulations 
def getVtraces(scale, Vrec, Vthr=-52, volt_min=None, volt_max=None, vbins=30, density=False, dv=None):
  '''
  Vrec: 
    micro: record of all membrane potentials, axis 0: time, 1: neurons

  '''
  print('Extracting voltage histogram...', end="")
  if not volt_min:
    volt_min = np.floor(np.min(Vrec.flatten()))-1 
  if not volt_max:
    volt_max = np.ceil(np.max(Vrec.flatten())) +1
    if volt_max<Vthr:
      volt_max=Vthr+2
  if dv:
    vbinedges = np.arange(volt_min, volt_max+dv, dv)
    vbins=len(vbinedges)-1
  else:
    vbinedges = np.linspace(volt_min, volt_max, vbins+1, endpoint=True)
  Vdistr = np.zeros((vbins,Vrec.shape[0]))
  for t in range(Vrec.shape[0]):
    volt = Vrec[t,:] # voltages in all history/neuron bins
    Vdistr[:,t] = np.histogram(volt,  vbinedges, density=density)[0]
  Vdistr[np.isnan(Vdistr)] = 0
  print('[done]')
  return Vdistr, vbinedges

def get_oscillation_cycles(signal_in, fnet, dt, offset= 50, max_period_deviation = .15, plot=False):
  '''
  using Hilbert transform, extract the beginning and end of all regular oscillation cycles!

  Parameters
  ----------
  signal_in : [Hz]
    signal, typically smoothed population rate
  fnet : [Hz]
    frequency of signal as determined via PSD
  dt : [ms]
    time step of signal
  offset : [ms]
    Initial period to ignore. The default is 50.
  max_period_deviation : [%]
    maximal relative deviation from the period (1/fnet) to allow. The default is 0.15.
  plot : 
    The default is False.

  Returns
  -------
  cyc_idx : shape (2, number of accepted cycles)
    indicies of start and beginning of each accepted oscillation cycle
    (i.e. with a period within 1/fnet +/- 0.15*1/fnet)

  '''
  
  # cut off initial period, subtract mean
  signal = signal_in.copy()
  signal = signal[int(offset/dt):]
  signal = signal - np.mean(signal)
  time = np.arange(signal.size)*dt + offset
  
  # take hilbert transform to derive phase
  analytic_signal = hilbert(signal)
  amplitude_envelope = np.abs(analytic_signal)
  phase = np.angle(analytic_signal)
  # instantaneous_phase = np.unwrap(np.angle(analytic_signal))
  # instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0*np.pi) * fs)
  
  # extract cycles
  idx = np.where(np.diff(phase) < -pi*.9)[0]
  cyc_idx_all = np.array([idx[:-1], idx[1:]]) # axis 0: start, end indices of each cycle, axis 1: cycles
  cyc_length = np.squeeze(np.diff(cyc_idx_all, axis=0))*dt # ms
  period = 1000/fnet # ms
  accept = np.abs(cyc_length - period)/period < max_period_deviation
  
  cyc_idx = cyc_idx_all[:,accept]
  
  print('Accepted {}/{} cycles, corresponding to {:.2f}ms/{:.2f}ms total signal time.\n Cycle length: {:.2f} +/- {:.2f}ms'.format(\
         np.sum(accept), cyc_idx_all.shape[1], np.sum(cyc_length[accept]), time[-1]-time[0], np.mean(cyc_length[accept]), np.std(cyc_length[accept])))
  
  if plot:
    plt.figure()
    plt.axvspan(period - max_period_deviation*period, period + max_period_deviation*period, facecolor='palegreen', alpha=0.5, zorder=0, label='15% deviation')
    plt.hist(cyc_length, bins=30)
    plt.axvline(period, color='g', label='$1/f_{net}$')
    plt.legend()
    plt.xlabel('cycle length [ms]')
    plt.ylabel('#')
    plt.tight_layout()
    
    fig, ax = plt.subplots(2, sharex=True)
    for i in range(2):
      for c in range(cyc_idx.shape[1]):
        ax[i].axvspan(time[cyc_idx[0,c]], time[cyc_idx[1,c]], facecolor='palegreen', alpha=0.5, zorder=0)
    ax[0].plot(time, signal)
    ax[0].plot(time, amplitude_envelope)
    ax[0].set_ylabel('signal [Hz]')
    ax[1].axhline(pi,lw=1,color='k')
    ax[1].axhline(-pi,lw=1,color='k')
    ax[1].plot(time, phase)
    ax[1].set_yticks([-pi, 0, pi])
    ax[1].set_yticklabels(['$-\pi$', '0', '$\pi$'])
    ax[-1].set_xlabel('time [ms]')
    ax[1].set_ylabel('phase [rad]')
   
  return cyc_idx + int(offset/dt)

def get_average_cycle(v_in, rate, f_net, Vthr, El, dt, oscillatory, record_window, offset=50, dv=.1, n=21):
  '''
  Compute average oscillation cycle by averaging over all detected individual cycles.
  '''
  print('Find rate and voltage of average cycle...')
  v = v_in.copy() # time x Nint 
  # only take offset if the entire simulation time series was passed over
  if record_window[0] > offset:
    print('Taking full record_window: {} ms (no additional offset).'.format(record_window))
    offset=0
  
  if oscillatory:
    # extract "good" oscillation cycles
    cyc_idx = get_oscillation_cycles(np.mean(v, axis=1), f_net, dt, offset=offset, max_period_deviation = .15, plot=False) # 2 x ncyc, before I used rate, not mean v
    
    # take n equally spaced samples in each cycle
    # samples 0 and n are equivalent (start/end of cycle)
    cyc_length = np.squeeze(np.diff(cyc_idx, axis=0))
    sample_v = np.round(cyc_idx[0,:] + np.linspace(0,1,n, endpoint=False)[:,None]*cyc_length[None,:]).astype(int) # n x ncyc
  else:
    # network is in AI state, simply average once over the full time axis
    n = 1
    sample_v = np.arange(v.shape[0])[None,:] #np.arange(rate.size)[None,:]
  # rate was recorded over entire simulation, v only in record_window!
  sample_r = sample_v + int(record_window[0]/dt)
  
  # average population spike
  popspk_av = np.mean(rate[sample_r], axis=1)
  # average voltage density
  # collect voltage data in n samples and fit n gauss-distributions
  bin_edges = np.arange(int(np.min(v)), Vthr+dv, dv)
  bin_centers = bin_edges[:-1] + dv/2
  nbins = bin_centers.size
  hist = np.zeros((n,nbins)) # average voltage density in each of n time bins
  fit_params = np.zeros((n,3)) # mean, std, mean squared error
  
  def gauss_fit(x, x0, sig):
    return get_gauss(x, x0, sig, broadcast=False)
  
  v_data = v[sample_v,:] # n x ncyc x Nint (3D array)
  v_std_phase = np.zeros(n) # average standard deviation in each of n phases during one ripple cycle
  for i in range(n):
    # merge all samples of phase i into one histogram
    hist[i,:] = np.histogram(v_data[i,:,:][~np.isnan(v_data[i,:,:])], bins = bin_edges, density=True)[0]
    v_std_phase[i] = np.std(v_data[i,:,:][~np.isnan(v_data[i,:,:])])
    # fit a gaussian
    x = bin_centers
    y = hist[i,:][~np.isnan(hist[i,:])]
    fit_params[i,:2] = scipy.optimize.curve_fit(gauss_fit, x, y, p0=[np.nanmean(v_data[i,:,:]), (Vthr-El)/2])[0]
  
  fit_params[:,2] = np.mean((hist-get_gauss(bin_centers, fit_params[:,0], fit_params[:,1]))**2, axis=1) # mse

  # use average voltage data to get an estimate for mu_min, sig_min
  
  dt_av = 1000/f_net/n
  v_av = np.mean(v_data, axis=(1,2)) # size n
  v_std = np.std(v_data, axis=(1,2)) # size n
  if oscillatory:
    v_av_mumin= np.min(v_av)
    v_av_sigmin = v_std[np.argmin(v_av)]  
    if (popspk_av > 1).any():
      t_on, t_off = np.where(popspk_av > 1)[0][[0,-1]]*dt_av
    else:
      t_on, t_off = nan, nan
    
    
    # roll all arrays such that sample 0 corresponds to the beginning of the cycle where v_av is in its minimum
    argmin = np.argmin(v_av)
    v_av = np.roll(v_av, -argmin)
    popspk_av = np.roll(popspk_av, -argmin)
    hist = np.roll(hist, -argmin, axis=0)
    fit_params = np.roll(fit_params, -argmin, axis=0)
    sample_v = np.roll(sample_v, -argmin, axis=0)
    v_std_phase = np.roll(v_std_phase, -argmin)
    
    all_mumins = np.mean(v[sample_v,:], axis=2)[0,:]
    v_av_mumin_sd = np.std( all_mumins )  # standard deviation of mumin over different cycles
    ''' v[sample_v,:]: n x ncyc x Nint (3D array) 
        --> average over neurons 
        --> take only the mean voltages in the resp. first bin of the cycle 
    '''
    # check that these are the right points to look at:
    if not np.isclose(np.mean(all_mumins), v_av_mumin):
      print(np.mean(all_mumins), v_av_mumin)
      raise ValueError('Mean over all mu_mins should equal minimum of mean voltage over average cycle!')
  else:
     v_av_mumin, v_av_sigmin, t_on, t_off, v_av_mumin_sd = nan, nan, nan, nan, nan
    
  return popspk_av, v_av, hist, bin_centers, fit_params, sample_v, v_av_mumin, v_av_sigmin, v_std_phase, t_on, t_off, v_av_mumin_sd


def get_spkstats(spktrain, offset=0):
  ''' 
  r: mean firing rate
  CV: measure for regularity of single unit firing, returns coefficient of variation of ISIs
  spktrain: spike train dictionary from Brian
  N: #neurons
  '''
  N = len(spktrain.keys())
  CV = np.ones(N)*nan
  r = np.zeros(N)
  isi_mean = np.ones(N)*np.nan
  isi_sd = np.ones(N)*np.nan
  for i in range(N):
    t_spk = spktrain[str(i)]
    t_spk = t_spk[t_spk>=offset]
    if len(t_spk) > 1:
      ISI = np.diff(t_spk)
      if ISI.size > 1:
        isi_mean[i] = np.mean(ISI)
        isi_sd[i] = np.std(ISI)
        CV[i] = np.std(ISI)/np.mean(ISI)
        r[i] = 1000/np.mean(ISI)
      elif ISI.size==1:
        r[i] = 1000/np.mean(ISI)
  CVmean = np.nanmean(CV)
  rmean = np.mean(r)  
  return r, rmean, CV, CVmean, isi_mean, isi_sd

def getSpkMatrix(spktrain, Tsim, dt, binsize=0, timewindow=[], squeeze=False, keys_as_str=True):
  '''
  produce binary matrix "spks" indicating spike times (axis 0: neurons, 1: time) 
  0: no spike, 1: spike
  spktrain: dictionary with keys=str(neuron idx), value: list of spike times in ms
  timewindow = [start, end]: only create spike matrix for this time window
  dt: time step of brian simulation [ms]
  '''
  if keys_as_str:
    def key(i):
      return str(i)
  else:
    def key(i):
      return i
  if not binsize:
    binsize = dt # by default: one bin per timestep
  elif binsize<dt:
    print('minimal binsize: dt={}ms'.format(dt))
    binsize = dt
  time = np.arange(0, Tsim, dt)
  if not type(spktrain)==dict: # spktrain is array of spike TIMES of a single neuron only
    spktrain = {key(0): spktrain}
  N = len(spktrain.keys())
  if not len(timewindow):
    timewindow=[0, time[-1]] # keep full simulation time
  t = time[(timewindow[0]<=time) & (time<=timewindow[-1])] # time of interest
  S = np.zeros((N, t.size))
  for i in range(N):
    spktimes = spktrain[key(i)][(t[0]<=spktrain[key(i)]) & (spktrain[key(i)]<=t[-1])] # keep only the spikes that are in the timewindow of interest
    if spktimes.size: # any spikes left
      spktimes -= t[0] # shift: map t[0] onto 0
      S[i, (spktimes/dt).astype(int)] = 1
  if binsize != dt:
    bins = np.arange(0,Tsim+binsize, binsize) # binedges
    idx = np.digitize(time, bins)  
    nbins = bins.size-1
    S_binned = np.zeros((N,nbins))
    for i in np.arange(1, nbins):
      # if i in np.linspace(1, nbins, 10).astype(int):
        # print(i,'/',nbins)
      S_binned[:,i] = np.sum(S[:,idx==i], axis=1) # sum all spks from all neurons in time bin i
    S = S_binned
  if squeeze:
    S= np.squeeze(S)
  return S

def getAnet_binned(S, binsize, dt):
  ''' 
  returns network activity (fraction of active neurons) in bins of size 'binsize'
  S: binary spike/no spike matrix, axis 0: neurons, axis 1: time
  binsize: (ms)
  time: time array (ms) (without units)
  '''
  Tsim = S.shape[-1]*dt
  time = np.arange(0, Tsim, dt)
  bins = np.arange(0,Tsim+binsize, binsize)
  idx = np.digitize(time, bins)  
  nbins = bins.size-1
  
  Anet_binned = np.zeros(nbins)
  for i in np.arange(1, nbins):
      if i in np.linspace(1, nbins, 10):
        print(i,'/',nbins)
      Anet_binned[i] = np.sum(np.sum(S[:,idx==i])) # sum all spks from all neurons in time bin i
  return bins[:-1], Anet_binned # Anet : # spikes

def getSpkSynchIdx(S0, dt, binsize=1, tau=1, meanrate=None):
  ''' compute Spike Train Synch Idx 
  (a) as described in Andre's Dissertation and detailed in my handwritten notes (10.01.19)
      tau: half-width of coincidence window [ms]
  (b) as described in Brunel, Wang 2003 and in my handwritten notes (~2018)
      binsize: size of time-bins [ms]
  S0: binary spike matrix (1: spike, 0: no spike), dimensions: 0: neurons, 1: time 
  dt: time step of spike matrix [ms]
  meanrate: mean unit firing rate (can be provided as input, or it will be computed here from the spike matrix)
  '''
  # (a)
  # remove empty rows, i.e. discard neurons that did not spike
  S = S0[np.sum(S0, axis=1)!=0,:]
  N, nstep = S.shape # number of neurons that fired >=1 spike, number of time steps
  T = dt*nstep # total simulation time [ms]
  if not meanrate: 
    meanrate = np.mean(np.sum(S, axis=1)/(T/1000)) # [1/ms] mean firing rate of neurons across the whole time, to improve estimate for short simulation times, use ISIs instead!
  w = int(tau/dt) # windowwidth in units of timesteps
  A = 0
  for i in range(N): # all neurons
    for j in range(int(np.sum(S[i,:]))): # all spikes of neuron i
      spkidx= np.nonzero(S[i,:])[0][j]
      # slice the array around spike j of neuron i
      A += np.sum(np.sign(np.sum(S[:,np.max(spkidx-w+1,0):np.min((spkidx+w,nstep))], axis=1))) # take sign to make sure each neuron can only contribute 1 coincident spike at most!
  A -= np.sum(S)
  synchIdx = A/((N-1)*np.sum(S))
  synchIdxCorr = synchIdx - 2*tau/1000*meanrate
  
  #(b) STS from Brunel
  Sbinned = getAnet_binned(S0, binsize, dt)[1] # number of spikes in bins of binsize
  STS = T/binsize*np.sum(Sbinned**2)/(np.sum(Sbinned)**2)
  return synchIdx, synchIdxCorr, STS

def get_fwhm(x, y):
  '''
  Full-width-half-maximum.
  graph y = y(x)
  finds peak of y, determines fwhm around the peak
  '''
  imax = np.argmax(y)
  ypeak = y[imax]
  idx = np.where(y<=ypeak/2)[0]
  if len(idx):
    if (idx<imax).any():
      left = x[idx[idx<imax][-1]]
    else:
      left = 0
    if (idx>imax).any():
      right = x[idx[idx>imax][0]]
    else:
      right = np.infty
    fwhm = right-left
  else:
    fwhm, left, right = nan, nan, nan
  return fwhm, left, right

  
def fitLorentz(f, p, par0, df_interp):
    ''' Fit Lorentz distribution to power spectral density to estimate peak position.
    f: frequencies
    p: power
    first guess of Lorentz parameters
    '''
    fpeak_put, hwhm_put, a_put = par0
    # fit a lorentzian around that putative dominant frequency
    # exclude small frequencies and additional peaks at higher harmonics 
    fitting_range = (f>fpeak_put-4*hwhm_put) & (f<fpeak_put+4*hwhm_put) # (f>fmin) & (f<1.6*faux[imax])
    f_crop = f[fitting_range] 
    p_crop = p[fitting_range]
    print('fitting lorentzian in the range {}-{}Hz. #samples: {}'.format(f_crop[0], f_crop[-1], f_crop.size))
    #  lower_bd = [.5*fpeak_put, .02, .5*a_put]
    #  upper_bd = [1.5*fpeak_put, np.inf, 1.5*a_put]
    popt, pcov = scipy.optimize.curve_fit(lorentzian, f_crop, p_crop, p0=par0)#, bounds=(lower_bd, upper_bd))
    # store dominant frequency, peakpower and fwhm
    fpeak = popt[0]
    ppeak = popt[2]/(pi*popt[1]) #p[peak_idx]
    # interpolate to get more accurate fwhm:
    f_interp = np.arange(f[0], f[-1]+df_interp, df_interp)
    p_interp = lorentzian(f_interp, popt[0], popt[1], popt[2])
    return fpeak, ppeak, (f_interp, p_interp)
  
def gabor(x, w, sig= 1/np.sqrt(2), x0=0):
    '''
    sig: width
    w: frequency (beachte: 2pi faktor ist hier im gabor wavelet enthalten)
    '''    
    return 1/(sig*np.sqrt(2*pi))*np.exp(-(x-x0)**2/(2*sig**2))*np.exp(-1j*2*pi*w*(x-x0))  
  
def waveletspec(signal, freq, dt, wavelet=gabor, sig=None, fmin=30):
    ''' creates wavelet spectrogram (cf. Donoso IFA paper)
    signal: signal to be analysed
    freq: [Hz] array of frequencies to be tested
    dt: [sec] time step of discrete signal measurements t (in seconds)
    wavelet: so far just gabor used, has to take arguments time, freq, center x0
    fmin: [Hz] for instantaneous freq, only look at freqs > fmin
    '''
    if not sig: # no temporal window width given
      expPer = 1/np.mean(freq) # expected period of the signal
      sig = expPer # choose default time window such that it contains ~3 cycles of the signal: 3sig~ Width of Gaussian Window in Time
      # sig_frq = np.sqrt(pi/2)/sig # just for reference: the std in freq domain (note: was bigger (=248) before, maybe make sig smaller?)
    t = np.arange(signal.size)*dt
    t0 = np.mean(t) # center wavelet array w.r.t time array
    F = np.zeros((freq.size, signal.size))
    for i in range(freq.size):
        w = freq[i]
        g = wavelet(t, w, sig=sig, x0=t0)
        F[i,:] = np.abs(scipy.signal.fftconvolve(signal,g,mode='same')*dt)
    ifmin = np.where(freq>=fmin)[0][0] # index of first frequency point larger than fmin
    imax = np.argmax(F[ifmin:,:], axis=0) # at each point in time, find frequency with maximal power
    instfreq = freq[ifmin+imax] # instantaneous frequency estimate, consider only freqs > fmin
    instpower = F[ifmin+imax,np.arange(0,F.shape[1])]
#    plt.figure()
#    plt.imshow(F, origin='lower', aspect='auto', extent=[t[0], t[-1], freq[0], freq[-1]])
#    plt.plot(t, instfreq, 'w')
    return F, instfreq, instpower

def get_instfreq_discrete(signal, dt, freq_av=180, ampl_min=0):
  '''
  dt: [ms]
  freq_av: [Hz] initial guess for frequencies that we are looking for
  ampl_min: minimal signal amplitude to be considered a significant peak
  '''
  print('InstFreq discrete...', end='')
  T = len(signal)*dt 
  mindiff = 1000/(freq_av+200)/dt # minimal distance between peaks
  maxpeaks = (freq_av+100)*T/1000
  period = 1000/freq_av # expected period in ms  #period = 1/self.freq_av*1000 
  hw = period/dt/10
    
  peak_idx = findPeaks(signal, maxpeaks = maxpeaks, minabs=ampl_min, mindiff = mindiff, halfwidth = hw)[0] 

  ifreq_discr_t = (peak_idx[:-1]+np.diff(peak_idx)/2)*dt # mid points between signal peaks
  ifreq_discr = 1/(np.diff(peak_idx)*dt/1000) #  inst freq as 1/peak distance
  print('[done]')
  return ifreq_discr_t, ifreq_discr

def f_UnitAnalysis(spktrain, Tsim, dt, tau=1, binsize=1, getSynch=1, offset=0):
  ''' Analysis of single unit firing '''
  unitrates, freq_unit, _, CV = get_spkstats(spktrain, offset=offset)[:-2]
  if getSynch:
    start = Time.time()
    if Tsim > 1000 + offset:
      offset = Tsim - 1000
    S = getSpkMatrix(spktrain, Tsim, dt, timewindow=[offset, Tsim])
    synchIdx, synchIdxCorr, STS = getSpkSynchIdx(S, dt, binsize=binsize, tau=tau, meanrate=freq_unit)
    end = Time.time()
    print('time for getSynchIdx (s): ' + str(end - start))
  else:
    synchIdx, synchIdxCorr, STS = nan, nan, nan
  return freq_unit, unitrates, CV, (synchIdx, synchIdxCorr, STS)

def f_getSaturation(LFP_smooth, spktrain, Tsim, dt, scale, freq_unit=nan, freq_net=nan, offset=0, cyclewise=False):
  sat_t =  [] # saturation for each cycle: time points
  sat = [] # saturation for each cycle: tuple (spike-based, cell-based). Cell-based is more important
  saturation = [] # mean saturation of the whole simulation
  
  if not cyclewise:
    saturation=freq_unit/freq_net
  else:
    time = np.arange(0, Tsim, dt)
    # else find the cycles in the signals and count spikes or active cells 
    signal = LFP_smooth[int(offset/dt):] # exclude initial period
    # find signal peaks
    minabs = np.max(signal)/10 # minimal peak heigth
    mindiff = 1000/(600)/dt # minimal distance between peaks
    maxpeaks = 400*Tsim/1000
    period = 1000/400 # in ms  #period = 1/freq_net*1000 
    hw = period/dt/10
    
    peak_idx = findPeaks(signal, maxpeaks = maxpeaks, minabs=minabs, mindiff = mindiff, halfwidth = hw)[0] 
    
    if peak_idx.size>1: # only leaves out single peak (which is not an oscillation anyway)
      binedges = (np.append((peak_idx[0]-np.diff(peak_idx)[0]/2), np.append((peak_idx[:-1] + np.diff(peak_idx)/2), (peak_idx[-1]+np.diff(peak_idx)[-1]/2)))).astype(int)
      binedges[binedges<= 0] = 0 # smallest left binedge = 0
      binedges[binedges>= signal.size] = signal.size-1
      
      sat_t = time[peak_idx + int(offset/dt)] # centers of oscillation cycles
      
  #    plt.figure()
  #    plt.plot(time, signal)
  #    plt.plot(time[peak_idx], signal[peak_idx], 'ro')
  #    plt.axhline(minabs)  
  #    for j in range(binedges.size):
  #      plt.axvline(time[binedges[j]], color='r')
  
      if scale == 'micro':
        N = len(spktrain.keys())
        spks01 = getSpkMatrix(spktrain, Tsim, dt) # binary matrix with 0: no spike, 1: spike
        spksT = spks01*time[None, :] # matrix with 0: no spike, <time of spike in ms>: when there was a spike
        # saturation based on number of spikes in a cycle
        sat_spk = np.histogram(spksT, time[binedges+int(offset/dt)])[0]/N
        
        # saturation based on number of active cells in a cycle
        spksT[spksT==0] = -1 # replace 0s coding for "no spike", by "-1"
        # assign each spike time a bin index (which cycle-bin does it belong to?)
        binidx = np.digitize(spksT, time[binedges+int(offset/dt)]) # all "no spike (-1)" values will be assigned bin 0, which is left from the lowest binedge and will not be counted subsequently
        # in each neuron's row, take only unique values, i.e. in every cycle-bin count at most 1 spike from each neuron, not more!
        # from each neuron collect the indices of the cycles in which they spiked (once or more)
        cycnum = np.array([])
        for i in range(N):
          cycnum = np.append(cycnum,np.unique(binidx[i,:])) # unique gets rid of multiple spikes from 1 cell in a cycle
        sat_cell = np.histogram(cycnum, np.arange(sat_t.size+1)+0.5)[0]/N # how many cells spiked in each cycle bin? we neglect bin "0", since it has the no-spike instances only
        
      elif scale=='meso': # meso
        sat_spk = np.zeros(sat_t.size)
        for j in range(sat_t.size):
          sat_spk[j] = np.sum(signal[binedges[j]:binedges[j+1]])*(dt/1000) 
        sat = sat_spk
        sat_cell = np.zeros(sat_t.size)
        
      sat = np.array([sat_spk, sat_cell])
      saturation = np.mean(sat_spk[1:-1]) # exclude boundary effects (incomplete cycles) 
      # number of cycles with >1 spk from at least one cell = np.nonzero(np.abs(RIN.sat[0]-RIN.sat[1]))[0].size
    else:
      print('No oscillations/peaks could be detected, hence saturation=None! Consider lowering the threshold.')
  return sat_t, sat, saturation

def oscillation_analysis(f, p, fmin = 30, df_interp = 0.01, empirical=True, fig=False):
  ''' oscillation analysis for constant drive
  fits a Lorentzian to power to determine dominant frequency and fwhm
  determines coherence based on
  Input: 
    f: frequencies
    p: power (raw! df=1 Hz, now computed in get_PSD!)
    fmin: all frequencies lower than fmin will be disregarded for the dominant frequency computation
    empirical:  True: the input PSD is a NOISY estimate from simulation data (use Lorentz fit)
                False: the input PSD results from an analytical calculation and is sufficiently smooth (no need for Lorentz fit)
    df_interp: [Hz] resolution of interpolation in frequency space for fwhm computation (only applies for empirical=True)
  Output:
    dominant frequency
    peakpower at dominant frequency
    power at 0 (estimated from power values closeby)
    fwhm: full width at half maximum of the power (fitted by Lorentzian to avoid numerical inacccuracies)
    coherence (as in Donoso2018)
    oscillation strength = peakpower*FWHM (see Andre Holzbecher)    
    coherence (Lindner2005)
    qfac: (Benjamin) find english name (q-factor???)
  '''
  
  if np.isnan(f).any(): # 
    print('No PSD, hence no oscillation analysis!')
    return nan, nan, nan, nan, nan, nan, nan, nan 
  
  # recover parameters used for PSD calc:
  df = np.mean(np.diff(f))
  faux = f[f>fmin]
  paux = p[f>fmin]
  # get a first estimate of the dominant frequency
  imax = np.argmax(paux)
  
  # estimate power at 0 by averaging power values slightly above 0Hz
  p0 = np.mean(p[(f>0)&(f<=df*4)])

  # check if there is any "valid" peak at all  
  if not imax:
      print('No peak detectable above {} Hz!'.format(fmin))
      return nan, nan, nan, nan, nan, nan, nan, nan 
  else: 
      ''' little hacky part to avoid output of too high dominant frequencies due to numerical errors, 
         this solu is not very clean and only makes sense in the context of ripple oscillations
         find other power peaks that are at least 90% as high as the max'''
      if paux[imax]>1e4: # still not calibrated well 
        minabs = 0.1*paux[imax]
      else:
        minabs =  0.4*paux[imax] #0.6*paux[imax]
      peak_idx = findPeaks(paux, maxpeaks = 200, minabs= minabs, mindiff = int(20/df), halfwidth = int(2/df))[0] # minabs = 0.6*paux[imax]
      #    plt.figure()
      #    plt.plot(faux, paux)
      #    plt.plot(faux[peak_idx], paux[peak_idx], 'ro')
      #    plt.plot(faux[imax], paux[imax], 'ko')
      i = 0
      while peak_idx[i] < imax:
        if np.abs( faux[imax]/faux[peak_idx[i]] - np.round(faux[imax]/faux[peak_idx[i]]) ) < .05: # if the imax-freq is a multiple of the lower freq
          imax = peak_idx[i]     
  #        plt.plot(faux[imax], paux[imax], 'go')
          print('Artificial correction of dominant frequency to a lower harmonic in >>oscillation_analysis.')
          break
        else:
          i+=1
          
  if empirical: # fit a Lorentzian and compute fwhm on that
      # determine putative parameters of lorentzian
      fpeak_put = faux[imax] # putative peak location
      if fpeak_put < 70:
        a_put = np.sum(p[f<300])*df # estimate of integral over power
      else:
        a_put = np.sum(p[f<1.5*fpeak_put])*df # estimate of integral over power (higher harmonics excluded)
      hwhm_put = np.max([a_put/(pi*np.max(p)), 3*df]) # for sharp peaks, make sure that we take at least hwhm=df
      try:
        fpeak, ppeak, PSDfit =  fitLorentz(f, p, [fpeak_put, hwhm_put, a_put], df_interp)
        if fpeak < fmin: # if lorentz fit yields peak below fmin, set peak=nan
          fpeak, ppeak, fwhm, left, right = nan, nan, nan, nan, nan 
        else:
          fwhm, left, right = get_fwhm(PSDfit[0], PSDfit[1])
      except:
        if hwhm_put < 5: # we suspect high synchrony (a power peak so sharp, that it cannot be fitted by lorentzian anymore)
          # work with the raw power trace, without fitting
          empirical = False
        else: # we suspect very low synchrony (fitting did not work because of absence of obvious peak)
          fpeak, ppeak, fwhm, left, right = nan, nan, nan, nan, nan        
  if not empirical: # assume that the power curve is nice and smooth
      ppeak = paux[imax]
      fpeak = faux[imax]
      fwhm, left, right = get_fwhm(f,p)
    
  # compute different coherence measures
  coh_donoso = np.sqrt(ppeak/p0)
  oS = ppeak*fwhm  
  coh_lindner = fpeak*ppeak/fwhm # coherence from Lindner2005
  qfac = ppeak/fwhm # is this really called q-factor??? according to wikipedia, it should be q = fpeak/fwhm
    
  if fig:
    plt.figure()
    plt.plot(f,p, 'gray')
    plt.plot(fpeak, ppeak, 'ro')
    if not np.isnan(left):
      plt.plot([left, right], [ppeak/2, ppeak/2], 'g')
    if empirical:
      plt.plot(PSDfit[0], PSDfit[1], 'k', lw=1)
    plt.xlim([0,np.min([400, np.max(f)])])
    plt.ylim(bottom=0)
    plt.xlabel('freq [Hz]')
    plt.ylabel('power [spks²/sec]')
  print('[done]')
  return fpeak, ppeak, p0, fwhm, coh_donoso, oS, coh_lindner, qfac

def f_oscillation_analysis_stationary(LFP_raw, LFP_smooth, dt, spktrain, scale, df=1, k='max', fmin=30, \
                                      sat_cyclewise=False, offset=50, freq_unit=nan, coh_thr=1e-2):
  print('Oscillation analysis for stationary stimulation...')    
  Tsim = LFP_raw.size*dt
  
  # PSD of population rate
  freqs, power = get_PSD(LFP_raw, dt/1000, df=df, k=k, offset=offset/1000, subtract_mean=True, fig=False) 
  # find peak of PSD 
  freq_net, peakpower, power_0, power_fwhm, coh_donoso, oS, coh_lindner, qfac = oscillation_analysis(freqs, power, fmin=fmin)
  
  # find amplitude of rate oscillations
  if qfac < coh_thr: # set amplitude of weak/insignificant oscillations to mean(signal)
    ampl = (np.mean(LFP_smooth[int(offset/dt):]), np.std(LFP_smooth[int(offset/dt):]))
  else: 
    if scale=='micro':
      ampl = get_amplitude(LFP_smooth, dt, offset=np.max((offset, Tsim-500))) # tuple (mean, std) of signal amplitude  # analyze max 500ms for stronger oscillations, to save time
    elif scale=='meso':
      ampl = get_amplitude(LFP_raw, dt, offset=np.max((offset, Tsim-500))) 
  
  # saturation
  sat_t, sat, saturation = f_getSaturation(LFP_smooth, spktrain, Tsim, dt, scale, freq_unit=freq_unit, freq_net=freq_net, offset=offset, cyclewise=sat_cyclewise)

  return freqs, power, freq_net, peakpower, power_0, power_fwhm, coh_donoso, oS, coh_lindner, qfac, ampl, sat_t, sat, saturation

def f_oscillation_analysis_transient(signal, dt, fmax= 350, df=1, baseline_window=[50,150], target_window = [], Pthr=nan, \
                                     sig = None, expected_freq = 200, t_pad = 20, \
                                     fmin= 30, stationary=False, plot=False):
  '''
  INPUT
    signal: time series with signal to be analyzed
    dt: [ms] time step
    fmax: [Hz] waveletspectrogram will be computed for frequency range [0, fmax] with resolution df
    df: [Hz] resolution of wavelet spectrogram in frequency space
    baseline_window: [ms], list len 2, beginning and end of time window to use as baseline activity to determine threshold for power and amplitude
    target_window: [ms], list len 2, beginning and end of time window for which we want to analyse instantaneous frequency 
    Pthr: user-defined power threshold, if nan it will be determined based on the activity during the baseline_window
    sig: [ms] width of gaussian window to use for the waveletspectrogram, if None, sig will be defined based on the expected mean frequency.
              lower sig -> finer temporal resolution of waveletspec but coarser frequency resolution and vice versa
    expected_freq: [Hz] expected mean frequency of signal, used to infer optimal sig 
    t_pad: [ms] padding of baseline and target window to avoid boundary effects in the instantaneous frequency estimates
    fmin: [Hz] minimum for instantaneous frequency 
    stationary: bool, does the signal exhibit persistent oscillations? 
    plot: bool, produce result plot  
  
  OUTPUT
    wspec: 2D array (freq x time), instantaneous power in frequencies [0, fmax] over time
    wspec_extent: freq and time boundaries of wspec to be used for plotting with imshow
    instfreq: [Hz], instantaneous frequency (freq of maximal power, that is larger than fmin over time)
    instpower: power associated to the instfreq
    ripple: index of beginning and end of ripple event
    freq_onset_inst: onset frequency at beginning of ripple event
    instcoherence: instantaneous coherence associated to inst. freq.
    Pthr: power threshold
    ifreq_discr_t: time points associated to discrete inst. freq estimates 
    ifreq_discr: [Hz] discrete estimate of inst. frequency based on peak-to-peak distances in signal
  '''
  
  print('Oscillation analysis for transient stimulation...')    
  # --- initialization ---------------------------------------------------------
  wspec, wspec_extent, instfreq, instpower, instcoherence, ripple, freq_onset_inst = [],[],[],[],[],[nan, nan],nan
  freq = np.arange(0, fmax, df)
  Tsim = signal.size*dt
  if not len(target_window): # take inst freq of the full simulation
    target_window = [baseline_window[1], Tsim]
  # --- pad the baseline and target window to avoid boundary effects
  target_window_pad = [np.max([0,target_window[0]-t_pad]), np.min([Tsim,target_window[1]+t_pad])]
  baseline_window_pad = [np.max([0,baseline_window[0]-t_pad]), np.min([Tsim,baseline_window[1]+t_pad])]

  # --- setting the wavelet window width ---------------------------------------------------------
  if not sig: # no temporal window width given
    sig = 1/expected_freq # period of the expected onset freq in sec, 3sig~ Width of Gaussian Window in Time
    # sig_frq = np.sqrt(pi/2)/sig 
  
  # --- determine significant power threshold ---------------------------------------------------------
  if not stationary and np.isnan(Pthr):
    # --- establish baseline power threshold
    start = int(baseline_window_pad[0]/dt)
    end = int(baseline_window_pad[1]/dt)
    signal0 = signal[start:end]
      
    wspec0 = waveletspec(signal0, freq, dt/1000, sig = sig, fmin=0)[0]
    wspec0_extent = (baseline_window_pad[0], baseline_window_pad[1], 0, fmax)
    
    # take the average power at 0 ONLY from the baseline window (which is in the middle of the signal I used)
    power0_mean = np.mean(wspec0[0,int((baseline_window[0]-baseline_window_pad[0])/dt):int((baseline_window[1]-baseline_window_pad[0])/dt)])
    power0_std = np.std(wspec0[0,int((baseline_window[0]-baseline_window_pad[0])/dt):int((baseline_window[1]-baseline_window_pad[0])/dt)])
    # define threshold power
    Pthr = power0_mean + 4*power0_std
    print('Pthr: ', Pthr)
  else: 
    wspec0 = []  
  # --- actual analysis ---------------------------------------------------------
  # pad to avoid boundary effects
  start = int(target_window[0]/dt)
  end = int(target_window[1]/dt)
  start_pad = int(target_window_pad[0]/dt)
  end_pad = int(target_window_pad[1]/dt)
  
  # --- discrete
  if not stationary:
    ampl_min = np.mean(signal[int(baseline_window[0]/dt):int(baseline_window[1]/dt)]) \
            + 4*np.std(signal[int(baseline_window[0]/dt):int(baseline_window[1]/dt)])
  else:
    ampl_min = np.mean(signal)
  ifreq_discr_t, ifreq_discr = get_instfreq_discrete(signal[start : end], dt, freq_av=expected_freq, ampl_min=ampl_min)
  ifreq_discr_t += target_window[0]
    
  # --- continuous
  signal_target_pad = signal[start_pad : end_pad]
  
  wspec, instfreq, instpower = waveletspec(signal_target_pad, freq, dt/1000, sig = sig, fmin=fmin)
  # keep only the target window
  wspec_extent = (target_window[0], target_window[1], 0, fmax)
  wspec, instfreq, instpower \
  = wspec[:,start-start_pad:end-start_pad], instfreq[start-start_pad:end-start_pad], \
    instpower[start-start_pad:end-start_pad]
  
  if plot:
      t = np.arange(signal.size)*dt
      t_target = np.arange(target_window[0], target_window[1], dt)
      fig, ax = plt.subplots(2, figsize=(10,6), sharex=True)
      cbar_ax = fig.add_axes([0.92, 0.55, 0.025, 0.3])
      norm = matplotlib.colors.Normalize(vmin=0, vmax=np.max(wspec))
      if len(wspec0):
        ax[0].imshow(wspec0, extent = wspec0_extent, origin='lower', aspect='auto', norm=norm)
      im = ax[0].imshow(wspec, extent = wspec_extent, origin='lower', aspect='auto', norm=norm)
      ax[0].plot(ifreq_discr_t, ifreq_discr, 'wo', label='inst.freq. (discrete)')
      ax[0].autoscale(False)
      # identify snippets of significant inst freq (to avoid that continuous line is plotted through the gaps)
      ix = np.where(instpower>=Pthr)[0]
      jumps = np.where(np.diff(ix)>1)[0]
      jumps = np.concatenate([[-1], jumps, [ix.size-1]])
      for j in range(jumps.size-1):
        if not j:
          ax[0].plot(t_target[ix[jumps[j]+1] : ix[jumps[j+1]]+1], instfreq[ix[jumps[j]+1] : ix[jumps[j+1]]+1], 'w', label='inst.freq. (continuous)')
        else:
                    ax[0].plot(t_target[ix[jumps[j]+1] : ix[jumps[j+1]]+1], instfreq[ix[jumps[j]+1] : ix[jumps[j+1]]+1], 'w')
      cb = fig.colorbar(im, cax=cbar_ax, label='power [signal-unit*sec]')
      cb.ax.plot([0,1e5],[Pthr]*2, 'r', lw=2)
      ax[0].legend(loc='best')
      ax[1].plot(t, signal)  
      ax[1].set_xlabel('time [ms]')
      ax[0].set_ylabel('freq [Hz]')
      ax[1].set_ylabel('signal')

  instcoherence = np.sqrt(instpower/wspec[0,:])
  
  # ripple detection
  if not stationary:
    ripple_idx = np.where(instpower>=Pthr)[0] # indices of transient response
    if ripple_idx.any():
        ripple_gaps = np.diff(ripple_idx)>1 # any intermediate periods where power was NOT significant?
        if not ripple_gaps.any():
            freq_onset_inst = instfreq[ripple_idx][0]
        else:
            if np.max(np.diff(ripple_idx))*dt > 5: # there is a gap larger than 5ms, where power is actually not significant
                ripple_idx = ripple_idx[np.argmax(np.diff(ripple_idx))+1:] # this assumes that the gap was BEFORE the actual ripple, maybe refine this
            freq_onset_inst = nan
            # we do not assign any inst. onset freq, since there are several ripple periods, 
            # onset is not clearly defined
        ripple = np.array([ripple_idx[0], ripple_idx[-1]]) +start 
            
  print('[done]')
  return wspec, wspec_extent, instfreq, instpower, ripple, freq_onset_inst, instcoherence, Pthr, ifreq_discr_t, ifreq_discr

def find_full_synch_extrapolated(level, saturation, fnet, ix_back=-4):
  funit = saturation*fnet
  m_unit = (funit[-1] - funit[ix_back]) / (level[-1] - level[ix_back])
  m_net = (fnet[-1] - fnet[ix_back]) / (level[-1] - level[ix_back])
  Ifull = (fnet[-1] - funit[-1]) / (m_unit-m_net) + level[-1]
  fnet_full = fnet[-1] + m_net*(Ifull - level[-1])
  # print(fnet[-1], fnet[ix_back])
  # print(m_unit, m_net, Ifull, fnet_full)
  return Ifull, fnet_full

def find_full_synch_interpolated(level, saturation, fnet, level_res=0.01, plot=False):
  ''' for a given cyclo-stationary simulation, interpolate the saturation values
  to approximate the input level Ifull where full synch is readched
  
  level_res: (higher) resolution of input levels for interpolation
  '''
  # ignore nans
  if (saturation<1).all():
    print('simulation never reached full synchrony, will extrapolate linearly!')
    return find_full_synch_extrapolated(level, saturation, fnet)
  else:
    if np.isnan(saturation).any():
      idx = np.where(np.isnan(saturation))[0]
      saturation = np.delete(saturation, idx)
      fnet = np.delete(fnet, idx)
      level = np.delete(level, idx)
    level_highres = np.arange(level[0], level[-1]+level_res, level_res)
    sat_interp = np.interp(level_highres, level, saturation)
    fnet_interp = np.interp(level_highres, level, fnet)
    idx = np.argmin(np.abs(sat_interp-1))
    if plot:
      plt.figure()
      plt.plot(level_highres, sat_interp)
      plt.plot(level_highres[idx], sat_interp[idx], 'ro')
    
    if sat_interp[idx] < 0.9:
      raise ValueError('Check what went wrong with saturation interpolation!')
    return level_highres[idx], fnet_interp[idx]
  

def store_info_cyclostat_lif_simulation(traj=None, traj_hash=None, path_to_simulations='./simulations/'):
  '''
  Store short summary of most important parameters and results of a constant-drive simulation.
  Most importantly: the location of the point of full synchrony is approximated by interpolation and stored.

  Parameters
  ----------
  traj : pypet trajectory (loaded externally)
  traj_hash : trajectory hash (simulaiton will be loaded here)
  path_to_simulations : str, optional. The default is './simulations/'.

  Returns
  -------
  run index for which full synchrony was reached approximately.

  '''
  if not traj:
    traj= pypet_load_trajectory(traj_hash = traj_hash, path_to_simulations=path_to_simulations)
  path_stat_info = pypet_get_trajectoryPath(traj_hash = traj.hash, path_to_simulations=path_to_simulations) + 'info.csv'
  res = traj.results.summary.scalarResults  
  info = pd.Series(dtype='float64')
  info['I_full_nA'], info['fnet_full'] = find_full_synch_interpolated(res['parameters.input.level'].to_numpy(), res.saturation.to_numpy(), res.freq_net.to_numpy())
  info['run_idx_full'] = int(np.argmin(np.abs(res['parameters.input.level'] - info['I_full_nA'])))
  info['I_hopf_nA'] = traj.linear_stability.Icrit_nA
  info['fnet_hopf'] = traj.linear_stability.fcrit
  info['I0_hopf_nA'] = traj.linear_stability.I0crit_nA
  info['A0_hopf'] = traj.linear_stability.A0crit
  # --- extract dimensionless parameters
  # info['IPSPint_default'] = traj.IPSPint # convert_coupling(info['K'], info['tm'], nan, traj.Vthr, traj.E_rest, unit=True, default=True)
  traj.v_idx = 0
  IPSPint = traj.derived_parameters.crun.IPSPint
  Nint = traj.Nint
  info['K'], info['D'], [info['I_hopf'], info['I_full']], info['Vr'] \
  = rescale_params_dimless(traj.Vthr, traj.Vreset, traj.E_rest, IPSPint, np.r_[info['I_hopf_nA'], info['I_full_nA']], traj.input.Ie_sig, \
                           Nint, traj.C, traj.tm) # return coupling 
  info['Delta'], info['tm'] = traj.tl, traj.tm
  info['IPSPint_default'] = convert_coupling(info['K'], info['tm'], nan, traj.Vthr, traj.E_rest, unit=True, default=True)
  info.sort_index(inplace=True)
  info.to_csv(path_stat_info, header=None)
  print(info['K'], info['D'], info['Delta'], info['IPSPint_default'])
  traj.f_restore_default()
  return int(info['run_idx_full'])


