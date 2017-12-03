
*******************************************
Uncertainty Quantification using python (UQpy)
*******************************************

:Date: December 2017
:Authors: Michael Shields, Dimitris G.. Giovanis, Aakash Bangalore
:Contact: michael.shields@jhu.edu, dgiovan1@jhu.edu
:Web site: https://github.com/SURGroup/UQpy.git
:Documentation:  
:Copyright: This document has been placed in the public domain.
:License: UQpy is released under the GNU General Public Licence.
:Version: 0.0.1

Description
===========

UQpy is a numerical tool for performing uncertainty quantification using
using python.

Supported methods
-----------------

1. Monte Carlo simulation (MCS), 
2. Latin hypercube sampling (LHS), 
3. Markov Chain Monte Carlo simulation (MCMC) 
4. Partially stratified sampling (PSS).


Requirements
------------

            * ::
            
                Python >= 3.6.2
                Git >= 2.13.1


Installation
------------

This will check for all the necessary packages. It will create the virtual environment SURG_UQenv and install  its dependencies. In order to deactivate the virtual environment type

            * ::

                        $git clone https://github.com/SURGroup/UQpy.git
                        $cd UQpy
                        $python3 setup_.py   
                        $source SURG_UQenv/bin/activate
                        $pip3 install -r requirements.txt
 

In order to use matplotlib within SURG_UQenv

            * ::
            
                      $cd ~/.matplotlib
                      $nano matplotlibrc
                      type: "backend: TkAgg"


Installed packages:
------------------

1. numpy
2. scipy
3. matplotlib
4. pyDOE     
5. scikit-learn

