UQpy
=======


UQpy is a numerical tool for performing uncertainty quantification using
advanced Monte Carlo simulation methods using python 3.

# Requirements: Python >= 3.6.2
# Git >= 2.13.1

Installation of the virtual environment 
------------

    git clone https://github.com/SURGroup/UQpy.git
    cd UQpy
    python3 setup_.py   
    source SURG_UQenv/bin/activate
    pip3 install -r requirements.txt

This will check for all the necessary packages. It will create the virtual environment SURG_UQenv and install  its dependencies. In order to deactivate the virtual environment type 

# SURG_UQenv environment has the following packages installed:

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


