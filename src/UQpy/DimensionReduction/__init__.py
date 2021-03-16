#UQpy is distributed under the MIT license.

#Copyright (C) 2018  -- Michael D. Shields

#Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
#documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
#rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
#persons to whom the Software is furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
#Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
#WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
#COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
``DimensionReduction`` is the module for ``UQpy`` to perform the dimensionality reduction of high-dimensional data.

This module contains the classes and methods to perform pointwise and multi point data-based dimensionality reduction
via projection onto Grassmann manifolds and Diffusion Maps, respectively. Further, interpolation in the tangent space
centered at a given point on the Grassmann manifold can be performed, as well as the computation of kernels defined on
the Grassmann manifold to be employed in techniques using subspaces.

The module currently contains the following classes:

* ``Grassmann``: Class for for analysis of samples on the Grassmann manifold.

* ``DiffusionMaps``: Class for multi point data-based dimensionality reduction.

* ``POD``: Class for dimension reduction of solution snapshots.

"""

from UQpy.DimensionReduction.DiffusionMaps import DiffusionMaps
from UQpy.DimensionReduction.Grassmann import Grassmann
from UQpy.DimensionReduction.DirectPOD import DirectPOD
from UQpy.DimensionReduction.SnapshotPOD import SnapshotPOD
from UQpy.DimensionReduction.HOSVD import HOSVD



