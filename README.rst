
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
advanced Monte Carlo simulation methods using python 3.

Note
----

   Requirements: Python >= 3.6.2
   Git >= 2.13.1

Description
===========

UQpy is a numerical tool for performing uncertainty quantification using
using python 3. 

Note
----

   At the moment only Monte Carlo simulation (MCS), Latin hypercube sampling (LHS), 
   Markov Chain Monte Carlo simulation (MCMC) and partially stratified sampling (PSS).
   More methods are coming soon.

Installation
------------

            * item text::

                        git clone https://github.com/SURGroup/UQpy.git
                        cd UQpy
                        python3 setup_.py   
                        source SURG_UQenv/bin/activate
                        pip3 install -r requirements.txt



This will check for all the necessary packages. It will create the virtual environment SURG_UQenv and install  its dependencies. In order to deactivate the virtual environment type 

SURG_UQenv environment has the following packages installed:
------------------------------------------------------------

   1. numpy
   2. scipy
   3. matplotlib
   4. pyDOE     
   5. scikit-learn

In order to use matplotlib within SURG_UQenv

    cd ~/.matplotlib
    nano matplotlibrc

And type:
    backend: TkAgg

