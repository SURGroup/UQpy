
*******************************************
Uncertainty Quantification using python (UQpy)
*******************************************

|logo|

Note
====

UQpy (Uncertainty Quantification with python) is a general purpose Python toolbox for modeling uncertainty in physical and mathematical systems.


====

:Authors: Dimitris G. Giovanis, Michael D. Shields
:Contributors: Jiaxin Zhang, Aakash Bangalore Satish, Lohit Vandanapu, Mohit Singh Chauhan
:Contact: dgiovan1@jhu.edu, michael.shields@jhu.edu 
:License: Apache 2.0
:Date: May 2018
:Version: 0.1.0


Description
===========

UQpy is a numerical tool for performing uncertainty quantification
using python.

Supported methods
===========

Sampling methods:
           1. Monte Carlo simulation (MCS), 
           2. Latin Hypercube Sampling (LHS), 
           3. Markov Chain Monte Carlo simulation (MCMC) 
           4. Partially Stratified Sampling (PSS).

Reliability methods:
           1. Subset Simulation
           
Surrogate methods:
           1. Stochastic Reduced Order Models (SROM).


Dependencies
===========

            * ::
            
                Python >= 3.6
                Git >= 2.13.1


Installation
===========

Clone your fork of the UQpy repo from your GitHub account to your local disk (to get the latest version): 

            * ::

                        $git clone https://github.com/SURGroup/UQpy.git
                        $cd UQpy/
                        $python setup.py install  (user installation)
                        $python setup.py develop (developer installation)

From PyPI

            * ::

                        $pip install UQpy 

In order to uninstall it

            * ::

                        $pip uninstall UQpy


Documentation
===========

http://uqpy-docs-v0.readthedocs.io/en/latest/

Web Site of Group
===========

www.ce.jhu.edu/surg



.. |logo| image:: logo.jpg
    :target: https://gihub.com/SURGroup/UQpy
