#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
General tools independent of repository
(mostly copied from methods.py (ecxept for project-specific functions))

Created on Fri Feb 16 13:14:11 2018

@author: schieferstein
"""
import copy
import json
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import scipy
from scipy.optimize import minimize
from scipy import signal as scisig
from scipy.ndimage import gaussian_filter1d
from time import gmtime, strftime#, time
nan = np.nan
pi = np.pi

my_rcParams = json.load(open('settings/matplotlib_my_rcParams.txt'))
matplotlib.rcParams.update(my_rcParams)


def euler(f, x0, t_max, dt, parameters={}):
    ''' 
    f: the function on the right hand side of the ODE.
                The first two arguments must be x and t!
                Note that x can be a vector -> you can use it to integrate systems of ODE
    x0: initial value
    t_max: total simulation time
    dt: integration time step
    parameters: other parameters for the functions can be passed as a dictionary
    '''
    time = np.arange(0, t_max+dt, dt)
    nstep = time.size
    x = np.zeros(nstep)
    x[0]=x0
    for i in range(nstep-1):
        x[i+1] = x[i] + f(x[i], time[i], **parameters)*dt
    return time, x
  
def get_gauss(x, x0, sigma, broadcast = True):
  if broadcast and ((not np.isscalar(x0)) or (not np.isscalar(sigma))):
    if np.isscalar(x0):
      x0 = np.ones(sigma.shape)*x0
    if np.isscalar(sigma):
      sigma = np.ones(x0.shape)*sigma
    return 1/np.sqrt(2*pi)/sigma[:,None]*np.exp(-(x[None,:]-x0[:,None])**2/(2*sigma[:,None]**2)) # dimensions: variation in mean/var x phase
  else:
    return 1/np.sqrt(2*pi)/sigma*np.exp(-(x-x0)**2/(2*sigma**2))

def heaviside(x):
  return (x>=0)*1

def lorentzian(x, x0, hwhm, a):
  '''
  hwhm: scale parameter (half-width at half-maximum)
  '''
  return a/(pi*hwhm)*(hwhm**2/((x-x0)**2+hwhm**2))

def pm_pi(x):
  ''' convert rad variable from 0-2pi range to -pi to pi range'''
  x = x%(2*pi)
  if np.isscalar(x):
    if x>pi:
      x -= 2*pi
  else:
    x[x>pi] -= 2*pi    
  return x

def integrate(f, lower, upper, dx=None):
  ''' computes integral over function f from lower to upper boundary'''
  if not dx:
    dx = (upper-lower)/1e5
  x = np.arange(lower, upper, dx)
  integral = np.sum(f(x))*dx
  return integral

def SpkTrainGenerator(f, L, ts, refrac=0):  # add t as input for ISI histo
    ''' function that generates homogeneous or inhomogeneous Poisson process
        all inputs in SECONDS!
    INPUT:
        f: intensity (array: inhomo, scalar: homo)  must have resolution of ts!
        L: length of time interval (0,L) in SECONDS
        ts: time step in SECONDS
        refrac: refractory period in SECONDS (if >0 ISIs smaller than refrac will be eliminated in an extra step)
    OUTPUT:
        binary array of all time STEPS (length: L/ts):
            0: no spike
            1: spike
    '''
    np.random.seed()    
    nstep = int(np.round(L/ts))
#    nstep = int(np.round(L/ts)) # recover nstep, length of output vector
    spkmatrix = np.zeros(nstep) # initialize output vector
    
    if type(f) is np.ndarray : # INhomogeneous not homogeneous process
        Lam = int(np.ceil(np.max(f))) # pick upper bound
    else: 
        Lam = f # homogeneous

    # generate homogeneous Poisson process
    n = np.random.poisson(Lam*L)  # number of spikes in time interval (0,L)    
    u = np.random.choice(np.arange(0, nstep), size=n, replace=False)
    
    spk = np.sort(u)   
    
    # get inhomogeneous process by THINNING
    if type(f) is np.ndarray :         
        v = np.random.rand(n)
        keep = ( v <= f[spk]/Lam )
        spk = spk[keep]
    
    # impose refractory period if desired
    if refrac:
      isi = np.diff(spk*ts) # sec
      done = (isi<refrac).any()==False
      while not done:
        # find spikes that are too close and remove them
        ix_remove = np.where(isi<refrac)[0]
        spk = np.delete(spk, ix_remove)
        # determine where new spikes can NOT be added
        not_here = np.min(np.abs(np.arange(0, nstep)-spk[:,None]), axis=0)<refrac/ts # true if no spike should be there
        # add new spikes somewhere else instead
        spks_new = np.random.choice(np.arange(0, nstep)[not_here==False], size=ix_remove.size, replace=False)
        spk = np.sort(np.append(spk,spks_new))
        isi = np.diff(spk*ts) # sec
        done = (isi<refrac).any()==False
        
    spkmatrix[spk] = 1 # binary spike time matrix (0: no spike, 1: spike)
    spktimes = (np.arange(0,L,ts)*1000)[spk] # spike times in ms 
    
    #check ISI distribution
#    x = np.arange(0, np.max(isi), ts/1000)
#    plt.figure()
#    plt.hist(isi, bins=20, density=1)
#    plt.plot(x, Lam*np.exp(-Lam*x), 'r')   # Lam: 40/s --> 40/1000 pro ms
#    plt.title('ISI distribution of homogeneous Poisson with exponential fit')
#    plt.xlabel('ISI [s]')
    
    return spkmatrix, spktimes
  
def get_SpkTrain_refracPoisson(N, r, tref, Tsim, dt):
  ''' 07.02.20 
  generate Poisson spike times for N neurons with rate r and refractory period tref
  Tsim, dt, tref: [ms] sim time, time step, refrac period
  '''   
  print('drawing Pyr spike times with refractory period...')
  spktrains = np.zeros((int(N), int(np.round(Tsim/dt))))
  # generate Npyr refractory Poisson Spike Trains
  for i in range(int(N)):
    if not i%100:
      print(i, end='.')
    spktrains[i,:] = SpkTrainGenerator(r, Tsim/1000, dt/1000, refrac=tref/1000)[0]
  # extract spike times and indices of spiking neurons
  spk_indices = np.nonzero(spktrains)[0]
  spk_times = np.arange(0,Tsim,dt)[np.nonzero(spktrains)[1]] # unsorted!
  print('[done]')
  return spk_indices, spk_times

def ISIhist(t, spikes):
    ''' gives ISI histogram from time and spikes vectors
    '''
    isi = np.diff(t[spikes==1])
    plt.figure()
    plt.hist(isi, bins=20, normed=1)
    plt.xlabel('ISI (ms)')
    plt.title('ISI distribution inhomogeneous Poisson')
    return isi

def findPeaks_diffbased(f, mode='max'):
  if mode=='max':
    if (np.diff(np.sign(np.diff(f)))==-2).any():
      peak_idx = np.where(np.diff(np.sign(np.diff(f)))==-2)[0]+1
    else:
      peak_idx = []
  elif mode=='min':
    if (np.diff(np.sign(np.diff(f)))==2).any():
      peak_idx = np.where(np.diff(np.sign(np.diff(f)))==2)[0]+1
    else:
      peak_idx = []
  return peak_idx

def findPeaks(f, maxpeaks=1e10, minabs = -1e10, mindiff = 2, halfwidth = 1):
    ''' finds local maxima of array f (excluding boundaries)
    maxpeaks: maximal number of peaks to be found
    minabs: minimal absolute value of peaks
    mindiff: minimal distance between peaks (in step units (dimensionless)), if 2 peaks are too close, just the latter one is included
    halfwidth: minimal halfwidth of one peak (in step units)
    '''
    sortidx = np.flipud(np.argsort(f))
    peak_idx = np.array([]).astype(int)
    peakcounter = 0
    hw = int(halfwidth)
    if hw> mindiff/2:
      raise ValueError('Halfwidth cannot be bigger than half the minimal peak distance!')
    for i in sortidx:
      if (i-1>=0) and (i+1<f.size): # exclude boundary values
        if f[i]>=np.max(np.append(f[i-hw:i+hw+1], minabs)):
#        if f[i]>=np.max((f[i-1], f[i+1], minabs)): # local maximum, >= also possible, possible extension to the function to choose this
          peak_idx = np.append(peak_idx,i)
          peakcounter += 1
          if peakcounter > maxpeaks-1:
            break
    if peakcounter:
      peak_idx = np.sort(peak_idx)
      too_close = np.where(np.diff(peak_idx) < mindiff)[0]
      keep = np.array(list(set(np.arange(peak_idx.size)) - set(too_close) - set(too_close+1))).astype(int) # at first take out BOTH peaks that are too close to each other
      # kick out peaks that are too close to each other, keep larger one:
#      print(keep)
      for i in too_close:
        ix_keep = i + np.argmax([ f[peak_idx[i]], f[peak_idx[i+1]] ])
        if ix_keep not in keep:
          keep = np.append(keep, ix_keep)
#        print(keep)
      peak_idx = peak_idx[keep]
      peak_idx = np.sort(peak_idx)
      if len(np.unique(peak_idx)) != len(peak_idx):
        raise ValueError('some peaks still counted twice!')
      peakcounter = peak_idx.size
    else: 
      print('No peaks detected!')
    return peak_idx, peakcounter
  
def find_cos_phase_ampl(s, dt, f):
  ''' 14.09.20
  assuming a cosinusoidal signal of zero mean and frequency f
  s(t) = a*cos(w*t + phi)
  determine the amplitude a and phase offset phi via Fourier transform
  
  INPUT:
    s: signal
    dt: time step [arbitrary unit]
    f: [1/(unit of dt)] suspected frequency of the cosinus signal
  OUTPUT:
    a: amplitude [units of s]
    phi: phase offset [1 / (units of dt)]
  '''
  # compute fft only for frequency f
  time = np.arange(s.size)*dt
  exp_vec = np.exp(-1j*2*pi*f*time)
  
  fft = np.dot(s,exp_vec)*dt/(s.size*dt) # normalize by length of signal in units of time

  a = 2*np.abs(fft)
  phi = np.angle(fft)
  return a, phi
  
def bin_matrix(A, new_shape, take_mean=False, take_sum=True):
  ''' bin the array A along one or two axes, such that the new matrix has shape "new_shape"
  perform the binning by either summing up the elements or averaging them.
  

  Parameters
  ----------
  A : numpy matrix 1 or 2D
  new_shape : scalar (if A is 1D), tuple or list (if A is 2D)
    shape for the new, binned matrix
  take_mean : boolean, optional
    TAKE THE AVERAGE OF ALL ELEMENTS OF THE SAME BIN. The default is False.
  take_sum : boolean, optional
    TAKE THE SUM OF ALL ELEMENTS OF THE SAME BIN. The default is True.

  Returns
  -------
  Anew: binned matrix of new_shape

  '''
  if len(A.shape) == 1:
    A = A[:,None]
    new_shape = (new_shape, 1)
  if take_mean and take_sum:
    raise ValueError('Decide on either summing or averaging!')
  if (np.array(A.shape) % np.array(new_shape) != 0).any():
    raise ValueError('Current matrix shape must be multiple of new shape!')
    
  ar = A.reshape((new_shape[0], A.shape[0]//new_shape[0], new_shape[1], A.shape[1]//new_shape[1]))
  if take_sum:
    Anew = np.sum(np.sum(ar,axis=-1), axis=1)
  elif take_mean:
    Anew = np.mean(np.mean(ar,axis=-1), axis=1)
  if not Anew.shape == new_shape:
    raise ValueError('something wrong in the implementation')
  
  return Anew.squeeze()

def convert_numpy_2_python(x):
  '''
  convert all numpy dtypes (int64, float64) in a given scalar, list or dict into python natives (int, float)
  returns lists instead of arrays
  used before storing parameters with json, since json cannot deal with int64, float64
  '''
  if type(x) in [int, float, str, bool]:
    return x
  elif type(x) in [list, np.ndarray]:
    return [convert_numpy_2_python(xx) for xx in x]
  elif type(x) == dict:
    for key in list(x.keys()):
      x[key] = convert_numpy_2_python(x[key])
    return x
  else:
    return x.item()
  
# test = {'a':'some string', 'b':np.arange(3,3.5,.1), 'c': [100, 4.5], 'd':np.r_[2.,4.,6.4]}
# for key in test.keys():
#   if type(test[key]==list):
#     print([type(x) for x in test[key]])
#   else:
#     print(type(test[key]))

# test = convert_numpy_2_python(test)

def do_linear_regression(x,y):
  '''
  analytical solution of linear regression
  '''
  if type(x)==list:
    x = np.array(x)
  if type(y)==list:
    y = np.array(y)
  # kick out nans
  x = x[np.isnan(y) == False]
  y = y[np.isnan(y) == False]
  
  slope = np.sum((y-np.mean(y))*(x-np.mean(x))) / np.sum((x-np.mean(x))**2)
  intercept = np.mean(y) - slope*np.mean(x)
  return slope, intercept

def list_from_dict(d, output_numpy=False):
  d_list = []
  for i in d.keys():
    if type(d[i]) in [list, np.ndarray]:
      d_list += list(d[i])
    else:
      d_list.append(d[i])
  if output_numpy:
    d_list = np.array(d_list)
  return d_list

#%% PLOTTING
def fig_adjust(ax, lw=2.5): # adjusts all line width within all subplots at once
  for i in range(ax.size):  
    for ln in ax[i].lines:
      ln.set_linewidth(lw)
      
def plot_array(x, y, ax, idx=[], labels=[], colors=[], linestyle = '-', marker='', legend=True, lw=2):
  ''' plot a family of graphs x-y into ax with colors and labels
  INPUT:
    x, y: dimension 0: trials, dimension 1: plotting-dimension
  '''
  if not x.shape[-1] == y.shape[-1]: 
    raise ValueError('x and y must be aligned in last (plotting) dimension!')
  # broadcast if necessary:
  if len(x.shape)<len(y.shape):
    x = np.repeat(x[None,:], y.shape[0], axis=0)
  elif len(y.shape)<len(x.shape):
    y = np.repeat(y[None,:], x.shape[0], axis=0)
  if not len(labels):
    labels = [None]*x.shape[0]
    legend = False
  if not len(colors):
    colidx = np.linspace(1,0,x.shape[0], endpoint=True)
    colors = plt.cm.viridis(colidx)
  #--- plotting ---------------------------------------------------------------
  if len(idx): #restrict all curves to a specific region
    if not idx.shape == (x.shape[0],2):
      raise ValueError('wrong indexing dimensions!')
    for x, y, c, l, ix in zip(x,y, colors, labels, idx):  
        ax.plot(x[ix[0]:ix[-1]+1],y[ix[0]:ix[-1]+1], color=c, label=l, linestyle =linestyle, marker=marker, lw=lw)
        if legend:
          ax.legend()
  else:
    for x, y, c, l in zip(x,y, colors, labels):  
        ax.plot(x,y,color=c,label=l, linestyle =linestyle, marker=marker, lw=lw)
        if legend:
          ax.legend()
  return


def frame_subplot(ax, color='k', lw=5):
  ''' put frame around a subplot '''
  for side in ['bottom', 'top', 'right', 'left']:
    ax.spines[side].set_color(color)
    ax.spines[side].set_linewidth(lw)  
    
def add_ticks(ax, val, label, axis):
  if axis=='x':
    t = ax.get_xticks() 
  elif axis=='y':
    t = ax.get_yticks() 
    
  t=np.append(t,val)
  if len(label):
    tl=t.tolist()
    tl[-len(label):]=label
  
  if axis=='x':
    ax.set_xticks(t)
    if len(label):
      ax.set_xticklabels(tl)
  elif axis=='y':
    ax.set_yticks(t)
    if len(label):
      ax.set_yticklabels(tl) 
  return


def add_arrow(line, start_ind=None, pos_x=None, direction='right', arrow_at_end = False, size=15, color=None, ls=None, lw=None):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
      color = line.get_color()
    if ls is None:
      ls = line.get_linestyle()
    if lw is None:
      lw = line.get_lw()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    # straight line between 2 points    
    if len(xdata)==2:
      start_ind = 0 if direction=='right' else 1
      if arrow_at_end:
        xy_end = (xdata[1], ydata[1])
      else:
        xy_end = (np.mean(xdata), np.mean(ydata))
    else:
      # line with many points
      if start_ind is None:
        if pos_x is None:
          pos_x = xdata.mean()
        # find closest index
        start_ind = np.argmin(np.absolute(xdata - pos_x))
      end_ind = start_ind + 1 if direction == 'right' else start_ind - 1
      xy_end = (xdata[end_ind], ydata[end_ind])
      
    xy_start = (xdata[start_ind], ydata[start_ind])
    
    line.axes.annotate('',
        xytext= xy_start,
        xy= xy_end,
        arrowprops=dict(arrowstyle="->", color=color, ls=ls, lw=lw),
        size=size
    )
    
def df_heatmap(df, fig, ax, ax_cbar, xlabel='', ylabel='', cbar_label='', cmap = plt.cm.viridis, norm=None, \
               cbar_labelsize=matplotlib.rcParams["ytick.labelsize"], cbar_orientation= 'horizontal', cbar_labelpad=0, show_colorbar=True, \
               color_nan='k', xtick_rotation='horizontal'):
  '''
  plot and label a heatmap for a pandas data frame (pivoted before to correct index, columns, values)
  '''
  current_cmap = copy.copy(cmap)
#  current_cmap.set_bad(color=color_nan)
  
  im = ax.pcolor(df, cmap=current_cmap, norm=norm)
  # ax[0,1].set_yticks(np.arange(0.5, len(Ifull_theory.index), 1))
  ax.set_yticks(np.arange(0.5, len(df.index), 1));
  ax.set_yticklabels(df.index);
  ax.set_xticks(np.arange(0.5, len(df.columns), 1));
  ax.set_xticklabels(df.columns, rotation=xtick_rotation);
  if show_colorbar:
    cb = plt.colorbar(im, cax= ax_cbar, orientation=cbar_orientation) #fig.colorbar(im, cax= ax_cbar, orientation=cbar_orientation)
    cb.set_label(label=cbar_label, size=cbar_labelsize, labelpad=cbar_labelpad)
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  
  # mark nan with black cross: (not tested thoroughly)
  y_nan, x_nan = np.where(np.asanyarray(np.isnan(df)))
  if len(x_nan):
    for i in range(len(x_nan)):
      ax.plot([x_nan[i], x_nan[i]+1], [y_nan[i], y_nan[i]+1], 'k')
      ax.plot([x_nan[i], x_nan[i]+1], [y_nan[i]+1, y_nan[i]], 'k')
  return

#import seaborn as sns

# fig, ax = plt.subplots()
# ax_cbar = fig.add_axes([0.92, 0.1, 0.01, 0.5])
# ax = sns.heatmap(df.sort_index(ascending=False), ax=ax, cmap=plt.cm.Reds, cbar_ax = ax_cbar , \
#                  annot=True, square=True, yticklabels=[np.round(np.sqrt(x), decimals=2) for x in df.index])

def add_diagonal(ax, pt=(0,0)):
  ax.axline(pt, slope=1, zorder=1, lw=1, color='gray') 
  return

def map_1d_idx_2d_grid(idx, nrows, ncols, start='lower left', direction='vertical'):
  ''' for making 2D plots from 1D lists that contain 2 parameters '''
  if start == 'lower left':
    row = nrows- 1 - idx%nrows
  if direction=='vertical':
    col = idx//nrows
  return row, col

def plot_multicolor_line(x, y, z, fig, ax, ax_cbar=None, cmap='viridis', lw=matplotlib.rcParams['lines.linewidth'], max_color=1, \
                         cbar_label='', norm=None):
  from matplotlib.collections import LineCollection
  # Create a set of line segments so that we can color them individually
  # This creates the points as a N x 1 x 2 array so that we can stack points
  # together easily to get the segments. The segments array for line collection
  # needs to be (numlines) x (points per line) x 2 (for x and y)
  points = np.array([x, y], dtype=object).T.reshape(-1, 1, 2)
  segments = np.concatenate([points[:-1], points[1:]], axis=1)
  # Create a continuous norm to map from data points to colors
  if not norm:
    norm = plt.Normalize(z.min(), z.max()*max_color)
  lc = LineCollection(segments, cmap=cmap, norm=norm)
  # Set the values used for colormapping
  lc.set_array(z)
  lc.set_linewidth(lw)
  line = ax.add_collection(lc)
  if ax_cbar:
    if max_color<1:
      cb= fig.colorbar(line, cax=ax_cbar, label=cbar_label, extend='max')
    else:
      cb= fig.colorbar(line, cax=ax_cbar, label=cbar_label)
  return fig, ax, ax_cbar


def gridline(coords, ax, axis='y', zorder=-10, color='gray', lw=.5, label='', linestyle=':'):
  if np.isscalar(coords):
    coords = list([coords])
  for c in coords:
    if axis=='y':
      if type(ax) in [list, np.ndarray]:
        for axi in ax:
          axi.axhline(c, lw=lw, linestyle=linestyle, color=color, zorder=zorder, label=label)
      else:
        ax.axhline(c, lw=lw, linestyle=linestyle, color=color, zorder=zorder, label=label)
    else:
      if type(ax) in [list, np.ndarray]:
        for axi in ax:
          axi.axvline(c, lw=lw, linestyle=linestyle, color=color, zorder=zorder, label=label)
      else:
        ax.axvline(c, lw=lw, linestyle=linestyle, color=color, zorder=zorder, label=label)
  return

def get_aspect(ax=None):
    if ax is None:
        ax = plt.gca()
    fig = ax.figure

    ll, ur = ax.get_position() * fig.get_size_inches()
    width, height = ur - ll
    axes_ratio = height / width
    aspect = axes_ratio / ax.get_data_ratio()

    return aspect

def get_quadrant(x,y):
  if np.isscalar(x):
    if x>0:
      if y>0:
        return 1
      elif y<0: 
        return 4
    else:
      if y>0:
        return 2
      elif y<0: 
        return 3
  else:
    result = np.empty(x.size)
    for i, xx, yy in zip(range(x.size), x,y):
      result[i] = get_quadrant(xx,yy)
    return result

def custom_arrow(ax, x, y, color='k', alpha = 10, head_width = .5, lw=plt.rcParams['lines.linewidth'], point_to = .5):
  # x, y = np.array(x), np.array(y)
  alpha *= 1/360*2*pi
  
  aspect_ratio = get_aspect(ax)

  head_length = head_width/(2*np.sin(alpha))
  
  dx, dy = np.diff(x), np.diff(y)
  xm, ym = x[:-1]+dx*point_to, y[:-1]+dy*point_to
  quadrant = get_quadrant(dx,dy)
  theta = np.arctan(dy*aspect_ratio/dx)%(pi/2)
  
  angle_left =theta-alpha+pi+pi/2*(quadrant-1)
  angle_right =theta+alpha+pi+pi/2*(quadrant-1)
  
  # print(angle_left, angle_right)
  
  left = [xm + head_length*np.cos(angle_left), ym+ head_length*np.sin(angle_left)/aspect_ratio] # coordinates of left arrow point
  right = [xm + head_length*np.cos(angle_right), ym+ head_length*np.sin(angle_right)/aspect_ratio] # coordinates of left arrow point
  
  for i in range(x.size-1):
    ax.plot([left[0][i], xm[i], right[0][i]], [left[1][i], ym[i], right[1][i]], color=color, lw=lw)

  return ax

#def match_axis_limits(args):
#  ''' takes 1D figure axes
#  makes ylimits of all subplots the same
#  extend to be more flexible (xaxis, 2D subplots, etc)
#  '''
#  s = args[0].shape
#  yl = np.zeros((len(args),np.product(s), 2))
#  for a, ax in enumerate(args):
#    if ax.shape != s:
#      raise ValueError('All input axes must be of same shape!')
#    for i in range(ax.size):
#      yl[a,i,:] = ax[i].get_ylim()
#  ymin = np.min(yl[:,:,0], axis=0)
#  ymax = np.max(yl[:,:,1], axis=0)
#  for a, ax in enumerate(args):
#    for i in range(ax.size):
#      ax[i].set_ylim([ymin[i], ymax[i]])
      
def match_axis_limits_new(*args, axis='y', set_to='max'):
  ''' 
  args: arbitrarily many axes 
  axis: 'x', 'y' or 'xy'
  '''
  yl, xl = np.zeros((len(args),2)), np.zeros((len(args),2))
  for i, ax in enumerate(args):
    yl[i,:] = ax.get_ylim()
    xl[i,:] = ax.get_xlim()
  if set_to == 'max':
    ymin, ymax = np.min(yl[:,0]), np.max(yl[:,1])
    xmin, xmax = np.min(xl[:,0]), np.max(xl[:,1])
  elif set_to == 'min':
    ymin, ymax = np.max(yl[:,0]), np.min(yl[:,1])
    xmin, xmax = np.max(xl[:,0]), np.min(xl[:,1])
  for i, ax in enumerate(args):
    if 'y' in axis:
      ax.set_ylim([ymin, ymax])
    if 'x' in axis:
      ax.set_xlim([xmin, xmax])
      
def despine(ax, which=['top', 'right']):
  if type(ax) in [list, np.ndarray]:
    for axi in ax:
      despine(axi, which=which)
  else:
    for spine in which:
      ax.spines[spine].set_visible(False)
      if spine == 'bottom': # also remove xticks 
        ax.tick_params(axis = "x", which = "both", bottom = False, top = False, labelbottom=False)

def color_gradient(color_i, color_f, n):
    """
    calculates array of intermediate colors between two given colors

    Args:
        color_i: initial color
        color_f: final color
        n: number of colors

    Returns:
        color_array: array of n colors
    """

    if n > 1:
        rgb_color_i = np.array(colors.to_rgb(color_i))
        rgb_color_f = np.array(colors.to_rgb(color_f))
        color_array = [None] * n
        for i in range(n):
            color_array[i] = colors.to_hex(rgb_color_i*(1-i/(n-1)) + rgb_color_f*i/(n-1))
    else:
        color_array = [color_f]

    return color_array


#%% OTHER
def cut(x, center, halfwidth):
    ''' x: 1d array (ideally sorted)
        cuts out portion of x that lies within center +/- halfwidth
        returns cut out values and indices corresp. to sorted array
    '''
    x = np.sort(x)
    a = center -halfwidth
    b = center + halfwidth
    idx = np.where((x>= a) & (x<=b))[0]
    xcut = x[idx]
    return xcut, idx
       
def discPlot(x,y,thr):
    ''' return x and y for plotting with nans inserted at jumps, such that line gets interrupted at jumps of the function y
    '''
    x = np.sort(x)
    y = y[np.argsort(x)]
    idx = np.where(np.abs(np.diff(y)) >= thr)[0]+1
    x = np.insert(x, idx, np.nan)
    y = np.insert(y, idx, np.nan)
    return x,y

def getPowerSpec(signal, dt, sdSmooth=0, fig=0, returnpositive=0):
    """ compute Power Spectogram
    input: signal array, sampling of signal dt (in SECONDS!), smoothing factor (optional), fig=1 if plot desired
    output: freqs: frequencies, power: raw power values, power_smooth: smoothed power, fft: unshifted FFT result (for further filtering)
    """
    n = signal.size
    cutoff = n - n%2 # if n is uneven, only use n-1 points
    
    fft = np.fft.fft(signal, n = cutoff) 
    
    # ACHTUNG: HIER NORMALISIERUNG ÜBER ARRAY SIZE (~SIMULATION TIME) EINGEFÜGT!!!
    power = np.abs((np.fft.fftshift(fft)))**2/signal.size
    freqs = np.fft.fftshift(np.fft.fftfreq(cutoff, dt))

    if not (freqs==0).any():
        raise ValueError('Error in frequency computation! freqs is not centered around 0!')
    
    if fig:
        plt.figure()
        plt.plot(freqs, power, 'b')
        plt.title('raw Powerspec')
        
    if sdSmooth:
        power_smooth = gaussian_filter1d(power, sdSmooth)
        if returnpositive:
          power_smooth = power_smooth[freqs>=0]
    else:
        power_smooth = 0
    
#    print('power at 0: ' , power[freqs==0])
#    print('squared integral over signal: ' ,  (np.sum(signal)*dt)**2/signal.size)
#    print('squared integral over signal: ' ,  (np.sum(signal))**2/signal.size)
    check = (np.abs(np.sum(signal[:cutoff])**2/signal.size - power[freqs==0]) < 1e-4)
    if not check:
      raise ValueError('Error in getPowerSpec (tools.py)')
    
    if returnpositive:
      power = power[freqs>=0]
      freqs=freqs[freqs>=0]
    
    return(freqs, power, power_smooth, fft)
    
#%% applied in ripple_methods
def getCov(t, var, tau):
  '''
  returns covariance matrix for gaussian process
  t : time
  var: variance at fixed time t (entries on diagonal)
  tau: time constant of decay of correlation in time, for delta-correlation put tau = 0, this returns identity matrix*var
  '''
  if not tau: # delta function --> white noise, no correlation in time
    return var*np.identity(t.size)
  else:
    return var*np.exp(-1/tau*(t[:,None]-t[None,:])**2)
  

#dt=0.00001
#t = np.arange(0,2,dt)
#x = np.sin(2*pi*t*40)+1

# def getAutoCorr_DEPRECATED(signal, dt, norm1=1):
#   ''' USE get_xcorr(a,a) INSTEAD!
#   input: signal + sampling interval (dt)
#   dt: [seconds]
#   '''
#   T = signal.size*dt
#   ac = np.correlate(signal, signal, mode='full')*dt/T
#   ac_t = (np.arange(0, ac.size)-np.argmax(ac))*dt  
  
#   if norm1:
#     ac = ac/((np.sum(signal**2)*dt)/T)
  
#   asymmetric = np.sum(ac-ac[::-1])
#   if asymmetric:
#     raise ValueError('Autocorrelation is not symmetric!!')
# #  ac = np.correlate(signal, signal[:signal.size//2], mode='valid')
# #  ac_t = np.arange(0, ac.size)*dt
#   return ac_t, ac

def get_xcorr(a, b, dt=1, mode='full', norm_overlap=True, subtract_mean=False, scale_std=True, norm_AC0=False, alignment='left', plot=True):
  ''' 24.09.20 
  cross-correlation between a and b (b will be slided over a from left to right)
  INPUT: 
    a, b: two signals (can be of different sizes)
    dt [ms, sec or none]: sampling time step
    mode: mode for numpy.correlate, currently only "full" is supported in this code
    norm_overlap: [default: True] normalize each entry of the correlation by the respective amount of overlap between a and b
    subtract_mean: subtract mean of both signals before correlating
    scale_std: scale both signals by their standard deviation before correlating
    alignment: [left (default), center, right] which alignment of a and b is considered "0 lag"
    norm_AC0: normalize by the signal norms such that autocorrelation xcorr(a,a)=1 at 0 lag
    plot: plot the result?
  OUTPUT:
    tau: array of time lags [same unit as dt: ms, sec or none]
         negative: shift b backwards (correlation of a with future b)
         positive: shift b forwards (correlation of a with past b)
    C: cross-correlation [a-unit*b-unit] (unit depends on normalization also!)
  '''
  n_a = a.size
  n_b = b.size
  
  if subtract_mean:
    a, b = a-np.mean(a), b-np.mean(b)
  if scale_std:
    a, b = a/np.std(a), b/np.std(b)
  
  C = np.correlate(a,b, mode=mode)*dt # unit_a * unit_b * unit_dt
  if mode=='full':
    tau_0 = n_b # number of shifts corresponding to 0 lag
  else:
    raise ValueError('Not yet implemented: For modes other than "full" think about how to determine the 0-lag position for general array sizes!')
    
  tau = (np.arange(C.size)+1 - tau_0) # lags belonging to the correlation values in C
  if norm_overlap: # account for input arrays a,b of different sizes!
    overlap = np.zeros(tau.size)
    overlap[tau<=0] = n_b+tau[tau<=0] # overlaps when b is shifted to the left
    overlap[tau>0] = np.minimum(n_b, n_a-tau[tau>0]) # overlaps when b is shifted to the right
    overlap *= dt
    
    C = C/overlap # unit_a * unit_b 
  
  tau = tau*dt # translate lags to units of time 
  
  if norm_AC0:
    norm_a = np.sqrt(np.sum(a**2)*dt) 
    norm_b = np.sqrt(np.sum(b**2)*dt) 
    # when dividing by BOTH autocorrelations, maybe take the norm, i.e. SQRT(sum(a**2)*dt) ? 
    if norm_overlap:
      norm_a /= np.sqrt(n_a*dt)
      norm_b /= np.sqrt(n_b*dt)
    C /= (norm_a*norm_b)
  
  if plot:
    fig, ax = plt.subplots()
    ax.plot(tau, C)
    ax.set_xlabel('lag')
    ax.set_ylabel('cross-correlation')
    
  return tau, C

# def getPSD(signal, dt, fig=False, sdSmooth=2.5, returnpositive=True, substract_mean=False):
#   ''' Power Spectral Density 
#   signal: array with signal values
#   dt: sampling interval of signal (SECONDS!)
#   sdSmooth: std of gaussian smoothing window (Hz)
#   '''
#   if signal.size*dt<1:
#     print('Warning: Resolution of PSD > 1Hz! Provide 1 sec signal to get resolution of 1Hz!')
  
#   power = np.abs(np.fft.fftshift(np.fft.fft(signal)*dt))**2
#   freqs = np.fft.fftshift(np.fft.fftfreq(signal.size, dt))
  
# #  # Alternative computation using Wiener Khinchin Theorem: (for stationary stochastic processes)
# #  '''PSD = (Fourier Transform(Autocorrelation))**2/(length of autocorr array)'''
# #  ac_t, ac = getAutoCorr(signal, dt, norm1=0)
# #  plt.figure()
# #  plt.plot(ac_t, ac)
# #  fft = np.fft.fft(ac)#, n = cutoff) 
# #  power2 = np.real(np.fft.fftshift(fft)) 
# #  freqs2 = np.fft.fftshift(np.fft.fftfreq(n, dt))
      
#   if sdSmooth:
#       sdSmooth = sdSmooth/np.mean(np.diff(freqs)) # convert from Hz to Hz-steps
#       power_smooth = gaussian_filter1d(power, sdSmooth)
#   else:
#       power_smooth = 0
  
#   if fig:
#     plt.figure()
#     plt.plot(freqs, power, 'lightblue')
#     plt.plot(freqs, power_smooth, 'b')
#     plt.title('raw Powerspec')
#     plt.xlim([0,500])
    
# #  print('power at 0: ' , power[freqs==0])
# #  print('squared integral over signal: ' ,  (np.sum(signal)*dt)**2)
#   check = ((np.sum(signal)*dt)**2 - power[freqs==0]) < 1e-4
#   if not check:
#     raise ValueError('Error in getPSD (tools.py)')
  
#   if returnpositive:
#     power = power[freqs>=0]
#     power_smooth = power_smooth[freqs>=0]
#     freqs=freqs[freqs>=0]
  
#   return(freqs, power, power_smooth)


# def getPSD_averaged(signal, dt, fig=False, returnpositive=True, ls=1, sdSmooth = 5, offset = 0.05):
#   '''
#   dt: [SECONDS!]
#   offset: [sec] initial period of signal that will be ignored for PSD (default 50 ms)
#   ls: [sec] length of snippets for which to calculate PSD separately
#   '''
#   Tsim = signal.size*dt # signal length in sec
#   ns = int((Tsim-offset)/ls) # number of snippets to get from signal
#   cutoff = np.min([signal.size+1, int((offset+ns*ls)/dt)])
#   if signal.size*dt<ls*3+offset:
#     raise ValueError('For averaged PSD please provide at least 3,05 seconds of signal!')
    
#   signal_snip = np.reshape(signal[int(offset/dt):cutoff], (ns, int(np.round(ls/dt))))
  
#   freqs = np.fft.fftshift(np.fft.fftfreq(signal_snip.shape[1], dt))
#   if returnpositive:
#     freqs = freqs[freqs>=0]
#   power_snip = np.zeros((ns, freqs.size))
#   for i in range(ns):
#     power_snip[i,:] = getPSD(signal_snip[i,:], dt, returnpositive=returnpositive)[1]
#   power_mean = np.mean(power_snip, axis=0)
#   if sdSmooth:
#     power_mean_smooth = gaussian_filter1d(power_mean, sdSmooth)
#   else:
#     power_mean_smooth = 0
#   if fig:
#     plt.figure()
#     plt.plot(freqs, power_snip.transpose(), 'gray', lw=0.5)
#     plt.plot(freqs, power_mean, 'k')
#     if not np.isscalar(power_mean_smooth):
#       plt.plot(freqs, power_mean_smooth, 'b--')
#     plt.title('raw Powerspec')
#     plt.xlim([0,500])
  
#   return (freqs, power_mean, power_mean_smooth)

def get_fft(signal, dt):
  ''' 22.10.19
  returns scipy fast fourier transform with correction for the time step dt
  dt: [sec]
  '''
  return dt*scipy.fft.fft(signal)

def get_PSD(signal, dt, df='min', k=1, offset=0, subtract_mean=True, fig=False, eps=0.1): # formerly get_PSD_new
    """
    Computes the power spectrum of the signal
    Input:
        signal: typically the LFP 
        dt: [sec] time step
        df: [Hz] desired frequency resolution 
            OR: set df='min' to get best PSD resolution possible for the given signal length (only for k=1)
        k (int): The signal is split into k_repetitions which are FFT'd independently and then averaged in frequency domain.
        offset: [sec] signal in the time interval [0, offset] is removed before doing the Fourier transform. 
                Use this parameter to ignore the initial transient signals of the simulation.
        subtract_mean (bool): If true, the mean value of the signal is subtracted.

    Returns:
        freqs, ps, average_population_rate
    """
    
    print('calculating PSD...', end='')
    if k==1:
      print('(no averaging)', end='')     
    f_max = np.around(1/(2 * dt)) # maximal frequency of PSD
    N_init = int(offset/dt) # offset in array steps
    N_data = len(signal)
    if df == 'min': # compute PSD of maximal freq resolution that is possible for given signal length
      N_signal = (N_data-N_init)//k # use the maximal signal length possible
      df = 2 * f_max / N_signal
    else: # use only as much of the signal as needed for desired resolution df
      N_signal = int(2 * f_max / df) # required signal length for resolution df (in array length)
      if k=='max': # see how many snippets we can average
        k = (N_data-N_init)//N_signal
        if not k: 
          print('Signal length not sufficient for PSD calculation!')
          return (nan, nan)
    N_required = k * N_signal + N_init
    
    T = N_signal*dt # length of snippets in units of time [seconds]
    # crop the signal if necessary
    if (N_data < N_required):
        err_msg = "Inconsistent parameters. k={} repetitions require {} samples." \
                  " signal contains {} samples.".format(k, N_required, N_data)
        raise ValueError(err_msg)
    if N_data > N_required:
        print("PSD: drop samples")
        signal = signal[:N_required]
#        signal = signal[-N_required:]
#     print("length after dropping end:{}".format(len(signal)))
    signal = signal[N_init:]
    if np.sum(np.abs(signal)) == 0:
      print('signal = 0 everywhere, returning nan for PSD')
      return nan, nan
    print("\n  Now using {} / {} seconds for PSD calculation (df={}Hz, k={})".format(len(signal)*dt, N_data*dt, df, k))
    average_signal = np.mean(signal)
    D = np.var(signal)/2 
    if subtract_mean:
        signal = signal - average_signal
    signal = signal.reshape(k, N_signal)  # reshape into one row per repetition (k)
    k_ps = np.abs(get_fft(signal, dt))**2
    ps = np.mean(k_ps, axis=0)
    # normalize
    ps /= T  
    # crop
    ps = ps[:int(N_signal/2)]
    freqs = np.arange(0, f_max, df)
#    freqs = np.fft.fftfreq(N_signal, dt)[:int(N_signal/2)]
    if fig:
      plt.figure()
#      plt.plot(freqs, k_ps[:, :int(N_signal/2)].transpose()* dt / N_signal, 'gray', lw=1)
      plt.plot(freqs, k_ps[:, :int(N_signal/2)].transpose()/T, 'gray', lw=1)
      plt.plot(freqs, ps, 'k--', label='fft')
      plt.xlabel('freqs')
      plt.ylabel('power')
      plt.title('k={}, df={}Hz, T={}s'.format(k, df, T))
    check = np.abs(np.sum(ps)*df/D - 1) < eps #
    if not check:
      raise ValueError('Integral over PSD (={}) not equal to D=0.5var(signal)={}'.format(np.sum(ps)*df, D))
    
    print('[done]')
    return freqs, ps

def get_amplitude(signal, dt, offset=0, period=3):
  ''' returns signal amplitude
  dt, offset in ms
  period: expected period of the signal, default: 3ms < ripple
  '''
  peak_idx = findPeaks(signal[int(offset/dt):], maxpeaks = 1000, minabs= 0.1*np.max(signal), mindiff = int(period/2/dt), halfwidth = int(.5/dt))[0] 
  if peak_idx.size:
    peak_heights = signal[int(offset/dt):][peak_idx]
    ampl, ampl_std = np.mean(peak_heights), np.std(peak_heights)
#    plt.figure()
#    plt.plot(time[time>=offset], signal[time>=offset])
#    plt.plot(time[time>=offset][peak_idx], signal[time>=offset][peak_idx], 'ro')
  else:
    print('No clear oscillation, hence amplitude=mean(signal).')
    ampl, ampl_std = np.mean(signal[int(offset/dt):]), np.std(signal[int(offset/dt):])
  return (ampl, ampl_std)


def makealpha(J):
  def alphafct(t, tau=1):
    return J*t/tau**2*np.exp(-t/tau) # J=integral over alpha fct
  return alphafct


def f_all_scalar(*args):
  is_scalar = np.array([np.isscalar(x) for x in args])
  return is_scalar.all()

def num2str(x):
  '''
  convert decimal numbers to format:
    1.24 = 124e-2
    0.01 = 1e-2
  '''
  if np.abs(x-int(x))<1e-9:
    return str(int(x))
  else:
    x = np.round(x, decimals=2)
    s_out = str(x).replace('.','c')
#  elif x>= 1e-4:
#    s = str(x)
#    dec = s.split('.')[1] # nachkommastellen
#    n_dec = len(dec) 
#    s_out = str(int(x*10**n_dec))+'e-'+str(n_dec)
#  else:
#    s_out = str(x) # here str() automatically converts correctly
  # check
  if '.' in s_out:
    raise ValueError('sth wrong')
  return s_out

def list2str(l):
  s = str(l)
  s = s.replace('[', '')  
  s = s.replace(']', '') 
  s = s.replace("'", '') 
  s = s.replace(', ', '-') 
  s = s.replace('.', 'c')  # replace decimal comma by c
  return s

def dict2str(d, equals='=', delimiter='_', keys_only = False):
  '''
  from dictionary, generate a string of the form "key1=value1_key2=value2_"
  '''
  s = ''
  for key in d.keys():
    s += str(key)
    if not keys_only:
      values = d[key]
      if type(values)==list:
        values = list2str(values)
      elif type(values)==str:
        pass
      elif type(values)==tuple:
        values = list2str(list(values))
      else: # number
        values = num2str(values)
      s += equals+values
    s += delimiter
  s = s[:-len(delimiter)]
  return s



      
      
def F(mu, tm, tref, vr, vt, Er=0):
  ''' deterministic fI curve
  mu [mV] can be array
  tref, tm: [ms]
  '''
  if Er:
    vr = vr-Er
    vt = vt-Er
  if np.isscalar(mu):
    mu = np.array(mu.copy())
  r = np.zeros(mu.size)
  r[mu>vt] = 1000/(tref+tm*np.log((mu[mu>vt]-vr)/(mu[mu>vt]-vt)))
  return r
  
def Fdot(mu, tm, tref, vr, vt, Er=0):
  '''
  derivative of deterministic fI curve
  mu [mV]
  Er: rest potential
  '''
  if Er:
    vr = vr-Er
    vt = vt-Er
  if np.isscalar(mu):
    mu = np.array(mu.copy())
  rdot = np.zeros(mu.size)
  rdot[mu>vt] = 1000/((tref+tm*np.log((mu[mu>vt]-vr)/(mu[mu>vt]-vt)))**2)*tm*(vt-vr)/(mu[mu>vt]-vr)/(mu[mu>vt]-vt)
#  (mu[mu>vt]-vt)/(mu[mu>vt]-vr)*(vt-vr)/((mu[mu>vt]-vr)**2)
  return rdot

#%% OLD
#ac_t, ac = getAutoCorr(x,dt)
#f_PSD, P_PSD = getPSD(x, np.mean(np.diff(t)))[:2]
#f_welch, P_welch = scisig.welch(x, 1/dt)
#f_welchAC, P_welchAC = scisig.welch(ac, 1/dt)
#f_pspec, P_pspec = getPowerSpec(x, dt, returnpositive=1)[:2]
#
#plt.figure()
#plt.subplot(311)
#plt.plot(t,x)
#plt.subplot(312)
#plt.plot(ac_t, ac)
#plt.subplot(313)
#plt.plot(f_PSD, P_PSD, label='getPSD')
#plt.plot(f_welch, P_welch, '*-', label='welch')
#plt.plot(f_welchAC, P_welchAC, label='welch(AC)')
#plt.plot(f_pspec, P_pspec, label='old')
#plt.xlim([0,50])
#plt.legend()


#def writeDict2File(d, file):
#  ''' writes content of dictionary into file
#  d: dictionary
#  file: path to file where it should be stored
#  '''
#  fo = open(file, "w+")
#  fo.write('Parameters\n'+strftime("%a, %d %b %Y %H:%M:%S", gmtime()))
#  
#  for k, v in d.items():
#    fo.write(str(k) + ' :  ' + str(v) + '\n')
#  
#  fo.close()
#
##import json
#params = {'z':'hello', 'a':5}
##json.dump(params, open("params.txt",'w'), sort_keys=True, indent=2, separators=(',', ': '))
##check = json.load(open('params.txt'))
##
#def f(**simparams):
#  d = {'a':'sgrgh', 'b':2}
#  print({**d,**simparams})
#  
#f(f=4, **params)
#
#json.dump(test, open("0.txt",'w'), sort_keys=True, indent=2, separators=(',', ': '))





