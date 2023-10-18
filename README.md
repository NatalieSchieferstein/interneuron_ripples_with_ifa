# README #

This repository contains all code to produce the simulations, analyses and figures of the manuscript 
"Intra-ripple frequency accommodation in an inhibitory network model for hippocampal ripple oscillations"
by Natalie Schieferstein, Tilo Schwalger, Benjamin Lindner, Richard Kempter (bioRxiv preprint: 10.1101/2023.01.30.526209).

All code was written by Natalie Schieferstein.

### What is this repository for? ###

This repository can be used to

* regenerate all simulation data and analyses discussed in the manuscript
* regenerate all figures shown in the manuscript, supplement and response to reviewers

### Requirements ###

* Python (3.6.8)
* Brian2 (2.2.2.1, https://brian2.readthedocs.io/)
* Pypet (0.5.1, https://pypet.readthedocs.io/)
* Numpy (1.19.2)
* Scipy (1.5.2)
* Pandas (1.1.5)
* Matplotlib (3.3.3)
The code was run using the versions indicated in brackets.

### File types ###

* main_XXX.py files contain executable code
* methods_XXX.py files contain functions used in the main files
* tools.py: low level tools for plotting etc
* config_hX.txt Parameter configuration files for each simulation in /simulations/hX/

### Usage of main files ###

* main_plot_figures.py produces all figures (Results, Methods, Supplement, and Reviewer Figures)
  (Figures 3-5, and 8-10 can be reproduced right away, without rerunning any simulations)
* main_run_simulations.py sets up configurations files for, runs, and analyzes all simulations.

### Data structure for simulation results ###

The core of our simulation routines is contained in the class RippleNetwork (see methods_simulations.py).
We used pypet (https://pypet.readthedocs.io/) to run large simulations in parallel and organize the storage of parameter explorations (variations of drive, network size etc). Every simulation is stored as a pypet "trajectory" and is assigned a unique hash that can be used to access the data (pypet_load_trajectory). 

### Example simulation h0 ###

An example simulation (hash 0) can be downloaded here: https://box.hu-berlin.de/d/8feca38cef894f71b3e5/ 
and added to the simulations folder (>simulations/h0/Data_h0.hdf5, files are too large for GitHub). This simulation corresponds to the network of size N=10,000 seen in Fig 1 (the other network sizes are omitted to reduce the file size). Using the data from this example simulation Figures 3-5, and 8-10 can be reproduced right away by running main_plot_figures.py, without rerunning any simulations.

### Parameter settings ###

All default parameter settings can be found in >settings/StandardParams_all_values.csv.

