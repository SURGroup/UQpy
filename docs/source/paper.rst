UQpy paper
==========

The following repository contains all examples used for showcasing UQpy version 3.0.1 for needs of the paper.

https://github.com/SURGroup/UQpy_paper

Olivier, A., Giovanis, D.G., Aakash, B.S., Chauhan, M., Vandanapu, L., and Shields, M.D. (2020). "UQpy: A general
purpose Python package and development environment for uncertainty quantification". Journal of Computational Science.
https://doi.org/10.1016/j.jocs.2020.101204

The notebooks contain all the pieces of code necessary to run the various examples presented in the manuscript.
The project structure follows the structure of the paper, i.e., folders and sub-folders correspond to various sections
of the manuscript.

Note that certain scripts require the use of third-party software Abaqus. Several examples that require high
performance computing were run on the MARCC cluster; as an illustration we provide the scripts necessary to submit the
calculations related to the subset simulation example (in folder Section5_Reliability/Section5.2_SubsetSimulation/MARCC_files).

Binder link
-----------
All the examples except the ones using third-party software can run in following Binder environment:

.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/SURGroup/UQpy_paper/master
