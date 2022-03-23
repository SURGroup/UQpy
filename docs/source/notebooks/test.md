---
jupytext:
  encoding: '# -*- coding: utf-8 -*-'
  text_representation:
    extension: .mb
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Continuous 1D Distribution

+++

We'll be using UQpy's lognormal distribution class to exemplify how distribution classes work in UQpy. As well, we'll use Numpy for its math functionalities and Matplotlib to display results graphically.

```{code-cell} ipython3
from UQpy.distributions import Lognormal

import numpy as np
import matplotlib.pyplot as plt
```

Let's start by constructing a lognormal distribution `dist` with parameters of shape `s` equal to one, location `loc` equal to zero, and scale `scale` equal to $e^5$.

We can access the parameters we've set after construction via the `parameters` attribute.

```{code-cell} ipython3
dist = Lognormal(s=1,
                 loc=0,
                 scale=np.exp(5))

dist.parameters
```
