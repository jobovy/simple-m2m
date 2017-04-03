# simple-m2m

Extensions to the standard made-to-measure (M2M) algorithm for for full modeling of observational data.

## Overview

M2M is a standard technique for modeling the dynamics of astrophysical
systems in which the system is modeled with a set of *N* particles
with weights that are slowly optimized to fit a set of constraints
while integrating these particles forward in the gravitational
potential. The code in this repository extends this standard technique
to allow parameters of the system other than the particle weights to
be fit as well: nuisance parameters that describe the observer's
relation to the dynamical system (e.g., the inclination) or parameters
describing an external potentia. We also introduce a method for
sampling the uncertainty distribution of the particle weights and
combine it in a Gibbs sampler with a method for sampling the
uncertainty distribution of nuisance and potential parameters. All of
this is explored using a simple harmonic-oscillator potential and this
code implements Harmonic Oscillator M2M (HOM2M).

## AUTHORS

Jo Bovy - bovy at astro dot utoronto dot edu

Daisuke Kawata - d dot kawata at ucl dot ac dot uk 

## Code

## 1. [py/hom2m.py](hom2m.py)

This module contains all of the functions defining the HOM2M setup:
the dynamics of a harmonic oscillator, density and velocity
observables, M2M forces of change for various parameters, and the
general *fit_m2m* and *sample_m2m* functions that fit and sample the
parameters of a model, respectively.

## 2. [py/HOM2M.ipynb](HOM2M.ipynb)

(render this notebook on [nbviewer](http://nbviewer.ipython.org/github/jobovy/simple-m2m/py/HOM2M.ipynb), where you can toggle the code and also watch an embeedded movie)

This is the basic notebook for all of the mock data tests. Generates
mock data and fits it with particle weights, nuisance parameters, and
potential parameters in various combinations. Also performs MCMC
sampling of all parameters.

## 3. [py/HOM2M_v2.ipynb](HOM2M_v2.ipynb)

(render this notebook on [nbviewer](http://nbviewer.ipython.org/github/jobovy/simple-m2m/py/HOM2M_v2.ipynb), where you can toggle the code and also watch an embeedded movie)

Same as [py/HOM2M.ipynb](HOM2M.ipynb), except that the velocity
observable is the mean-squared velocity rather than the
density-weighted mean-squared velocity. Only briefly discussed in the
HOM2M paper.

## 4. [py/HOM2M_Fstar.ipynb](HOM2M_Fstar.ipynb)

(render this notebook on [nbviewer](http://nbviewer.ipython.org/github/jobovy/simple-m2m/py/HOM2M_Fstar.ipynb), where you can toggle the code)

Similar to [py/HOM2M.ipynb](HOM2M.ipynb), but applying the method to
F-star data from Gaia DR1.

## 5. [py/Fstar-kinematics.ipynb](Fstar-kinematics.ipynb)

(render this notebook on [nbviewer](http://nbviewer.ipython.org/github/jobovy/simple-m2m/py/Fstar-kinematics.ipynb), where you can toggle the code)

Notebook in which the velocity dispersion of F-type dwarfs as a
function of vertical height from the mid-plane is measured using the
*Gaia* DR1 *TGAS* data alone.

## 5. [py/simplex.py](simplex.py)

Functions to transform a set of *N* variables that sums to one (a
simplex) to a set of variables that span all of *N-1* dimensional real
space. Also transforms the derivatives in a fast manner. Not used in
the HOM2M paper, but described in an appendix.