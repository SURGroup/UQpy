
*******************************************
Uncertainty Quantification with python (UQpy)
*******************************************

|logo|
:align: center

====

:Authors: Michael D. Shields, Dimitris G. Giovanis
:Contributors: Aakash Bangalore Satish, Mohit Singh Chauhan, Lohit Vandanapu, Jiaxin Zhang
:Contact: michael.shields@jhu.edu, dgiovan1@jhu.edu
:Version: 0.1.0


Description
===========

UQpy (Uncertainty Quantification with python) is a general purpose Python toolbox for modeling uncertainty in physical and mathematical systems..

Supported methods
===========

For sampling:
           1. Monte Carlo simulation (MCS), 
           2. Latin Hypercube Sampling (LHS), 
           3. Markov Chain Monte Carlo simulation (MCMC) 
           4. Partially Stratified Sampling (PSS).

For reliability analysis:
           1. Subset Simulation
           
For surrogate modeling:
           1. Stochastic Reduced Order Models (SROM).


Dependencies
===========

            * ::
            
                Python >= 3.6
                Git >= 2.13.1


Installation
===========

From PyPI

            * ::

                        $pip install UQpy 

In order to uninstall it

            * ::

                        $pip uninstall UQpy

Clone your fork of the UQpy repo from your GitHub account to your local disk (to get the latest version): 

            * ::

                        $git clone https://github.com/SURGroup/UQpy.git
                        $cd UQpy/
                        $python setup.py install  (user installation)
                        $python setup.py develop (developer installation)


Help and Support
===========

Documentation:
           http://uqpy-docs-v0.readthedocs.io/en/latest/

Website:
           www.ce.jhu.edu/surg



.. |logo| image:: logo2.jpg
    :target: https://gihub.com/SURGroup/UQpy
