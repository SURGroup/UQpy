
*******************************************
Uncertainty Quantification using python (UQpy)
*******************************************

|logo|

:Authors: Dimitris G. Giovanis, Michael D. Shields
:Contributors: Jiaxin Zhang, Aakash Bangalore Satish, Lohit Vandanapu, Mohit Singh Chauhan
:Contact: dgiovan1@jhu.edu, michael.shields@jhu.edu
:Web site: www.ce.jhu.edu/surg
:Documentation: http://uqpy-docs-v0.readthedocs.io/en/latest/
:Copyright: SURG 
:License:
:Date: May 2018
:Version: 0.1.0

Note
====

UQpy is currently in initial development and therefore should not be
considered as a stable release.

Description
===========

UQpy is a numerical tool for performing uncertainty quantification
using python.

Supported methods
-----------------

Sampling methods:
           1. Monte Carlo simulation (MCS), 
           2. Latin Hypercube Sampling (LHS), 
           3. Markov Chain Monte Carlo simulation (MCMC) 
           4. Partially Stratified Sampling (PSS).

Reliability methods:
           1. Subset Simulation
           
Surrogate methods:
           1. Stochastic Reduced Order Models (SROM).


Requirements
------------

            * ::
            
                Python >= 3.6.2
                Git >= 2.13.1


Installation
------------

As user through github: 

            * ::

                        $git clone https://github.com/SURGroup/UQpy.git
                        $cd UQpy/
                        $python setup.py install  

As developer: 

            * ::

                        $git clone https://github.com/SURGroup/UQpy.git
                        $cd UQpy/
                        $python setup.py develop 

 

In order to uninstall it

            * ::

                        $pip uninstall UQpy


.. |logo| image:: logo.jpg
    :target: https://gihub.com/SURGroup/UQpy
