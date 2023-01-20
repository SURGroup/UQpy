|AzureDevops| |PyPIdownloads| |PyPI| |CondaPlatforms| |GithubRelease| |Binder| |Docs| |bear-ified|

.. |Docs| image:: https://img.shields.io/readthedocs/uqpy?style=plastic  :alt: Read the Docs
.. |CondaPlatforms| image:: https://img.shields.io/conda/pn/SURG_JHU/uqpy?style=plastic   :alt: Conda
.. |GithubRelease| image:: https://img.shields.io/github/v/release/SURGroup/UQpy?style=plastic   :alt: GitHub release (latest by date)
.. |AzureDevops| image:: https://img.shields.io/azure-devops/build/UQpy/5ce1851f-e51f-4e18-9eca-91c3ad9f9900/1?style=plastic   :alt: Azure DevOps builds
.. |PyPIdownloads| image:: https://img.shields.io/pypi/dm/UQpy?style=plastic   :alt: PyPI - Downloads
.. |PyPI| image:: https://img.shields.io/pypi/v/UQpy?style=plastic   :alt: PyPI
.. |Binder| image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/SURGroup/UQpy/master

.. |bear-ified| image:: https://raw.githubusercontent.com/beartype/beartype-assets/main/badge/bear-ified.svg
   :align: top
   :target: https://beartype.rtfd.io
   :alt: bear-ified


*******************************************
Uncertainty Quantification with python (UQpy)
*******************************************

|logo|

====

+-----------------------+------------------------------------------------------------------+
| **Product Owner:**    | Michael D. Shields                                               |
+-----------------------+------------------------------------------------------------------+
| **Lead Developers:**  | Dimitris Giovanis, Audrey Olivier, Dimitris Tsapetis             |
+-----------------------+------------------------------------------------------------------+
| **Development Team:** | Aakash Bangalore Satish, Mohit Singh Chauhan, Lohit Vandanapu,   |
+                       +                                                                  +
|                       | Ketson RM dos Santos, Katiana Kontolati, Dimitris Loukrezis,     |
+                       +                                                                  +
|                       | Promit Chakroborty, Lukáš Novák, Andrew Solanto                  |
+-----------------------+------------------------------------------------------------------+
| **Contributors:**     | Michael Gardner, Prateek Bhustali, Julius Schultz, Ulrich Römer  |
+-----------------------+------------------------------------------------------------------+

Contact
===========

To engage in conversations about uncertainty quantification, or ask question about UQpy usage and functionality refer to the UQpy's discussions tab:

`Discussions <https://github.com/SURGroup/UQpy/discussions>`_

Description
===========

UQpy (Uncertainty Quantification with python) is a general purpose Python toolbox for modeling uncertainty in physical and mathematical systems.

Documentation
================

Website:
           https://uqpyproject.readthedocs.io

Dependencies
===========

            * ::
            
                Python >= 3.9
                Git >= 2.13.1

License
===========
UQpy is distributed under the MIT license

Copyright (C) <2018> <Michael D. Shields>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


Installation
===========

From PyPI

            * ::

                        pip install UQpy

Using Conda

            * ::

                        conda install -c conda-forge uqpy

Clone your fork of the UQpy repo from your GitHub account to your local disk (to get the latest version): 

            * ::

                        git clone https://github.com/SURGroup/UQpy.git
                        cd UQpy/
                        python setup.py {version} install  (user installation)
                        python setup.py {version} develop (developer installation)

You will need to replace {version} with the latest version.

Referencing UQpy
=================

If you are using this software in a work that will be published, please cite this paper:

Olivier, A., Giovanis, D.G., Aakash, B.S., Chauhan, M., Vandanapu, L., and Shields, M.D. (2020). "UQpy: A general purpose Python package and development environment for uncertainty quantification". Journal of Computational Science, DOI:  10.1016/j.jocs.2020.101204.


Help and Support
===========

For assistance with the UQpy software package, please raise an issue on the Github Issues page. Please use the appropriate labels to indicate which module you are specifically inquiring about.

.. |logo| image:: logo.jpg
    :scale: 25 %
    :target: https://gihub.com/SURGroup/UQpy
    
    

