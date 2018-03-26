
*******************************************
Uncertainty Quantification using python (UQpy)
*******************************************

|logo|

:Authors: Michael D. Shields, Dimitris G. Giovanis
:Contributors: Jiaxin Zhang, Aakash Bangalore Satish, Lohit Vandanapu, Mohit Singh Chauhan
:Contact: michael.shields@jhu.edu, dgiovan1@jhu.edu
:Web site: www.ce.jhu.edu/surg
:Documentation: http://uqpy-docs-v0.readthedocs.io/en/latest/
:Copyright: SURG 
:License:
:Date: March 2018
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

1. Monte Carlo simulation (MCS), 
2. Latin hypercube sampling (LHS), 
3. Markov Chain Monte Carlo simulation (MCMC) 
4. Partially stratified sampling (PSS).
5. Stochastic Reduced order Models (SROM).


Requirements
------------

            * ::
            
                Python >= 3.6.2
                Git >= 2.13.1


Installation
------------

This will check for all the necessary packages. It will create the required virtual environment and install all its dependencies. 

            * ::

                        $git clone https://github.com/SURGroup/UQpy.git
                        $cd UQpy/
                        $pip install -r requirements.txt
                        $python setup.py install  

 

In order to uninstall it

            * ::

                        $pip uninstall UQpy


.. |logo| image:: logo.jpg
    :target: https://gihub.com/SURGroup/UQpy
