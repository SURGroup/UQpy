"""
This module is used to post-process the sensitivity analysis results. Currently it 
supports plotting the sensitivity results and comparing the sensitivity results
(such first order index v/s total order index) using the following two methods:

    1. plot_index
    2. compare_index

"""

import itertools

import numpy as np
import matplotlib.pyplot as plt
from beartype import beartype

from UQpy.utilities.ValidationTypes import (
    NumpyFloatArray,
)


@beartype
def plot_sensitivity_index(
    indices: NumpyFloatArray,
    confidence_interval: NumpyFloatArray = None,
    plot_title: str = None,
    variable_names: list = None,
    **kwargs,
):

    """

    This function plots the sensitivity indices (with confidence intervals)
    in a bar plot.

    **Inputs:**

    * **indices** (list or ndarray):
        list/array of sensitivity indices
        Shape: (num_vars)

    * **confidence_interval** (list or ndarray):
        list/array of confidence interval for the sensitivity indices.
        Shape: (num_vars, 2)

    * **plot_title** (str):
        Title of the plot
        Default: "Sensitivity index"

    * **variable_names** (list):
        List of variable names
        Default: [r"$X_{}$".format(i) for i in range(num_vars)]

    * **kwargs (dict):
        Keyword arguments for the plot to be passed to matplotlib.pyplot.bar

    """

    num_vars = len(indices)

    if variable_names is None:
        variable_names = [r"$X_{}$".format(i + 1) for i in range(num_vars)]

    # Check if confidence intervals are available
    if confidence_interval is not None:
        conf_int_flag = True
        error = confidence_interval[:, 1] - indices
    else:
        conf_int_flag = False

    # x and y data
    _idx = np.arange(num_vars)

    indices = np.around(indices, decimals=2)  # round to 2 decimal places

    # Plot one index
    fig, ax = plt.subplots()
    width = 0.3
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    index_bar = ax.bar(
        _idx,  # x-axis
        indices,  # y-axis
        width=width,  # bar width
        yerr=error if conf_int_flag else None,  # error bars
        ecolor="k",  # error bar color
        capsize=5,  # error bar cap size in pt
        **kwargs,
    )

    ax.bar_label(index_bar, label_type="edge", fontsize=10)
    ax.set_xticks(_idx, variable_names)
    ax.set_xlabel("Model inputs")
    ax.set_ylim(top=1)  # set only upper limit of y to 1
    ax.set_title(plot_title)

    plt.show()

    return fig, ax


@beartype
def plot_index_comparison(
    indices_1: NumpyFloatArray,
    indices_2: NumpyFloatArray,
    confidence_interval_1: NumpyFloatArray = None,
    confidence_interval_2: NumpyFloatArray = None,
    label_1: str = None,
    label_2: str = None,
    plot_title: str = "Sensitivity index",
    variable_names: list = None,
    **kwargs,
):

    """

    This function plots two sensitivity indices (with confidence intervals)
    in a bar plot for comparison.
    For example:
    first order Sobol indices and total order Sobol indices
    OR
    first order Sobol indices and Chatterjee indices.

    **Inputs:**

    * **indices_1** (list or ndarray):
        list/array of sensitivity indices
        Shape: (num_vars)

    * **indices_2** (list or ndarray):
        list/array of sensitivity indices
        Shape: (num_vars)

    * **confidence_interval_1** (list or ndarray):
        list/array of confidence interval for the sensitivity indices.
        Shape: (num_vars, 2)
        Default: None

    * **confidence_interval_2** (list or ndarray):
        list/array of confidenceiInterval for the sensitivity indices.
        Shape: (num_vars, 2)
        Default: None

    * **plot_title** (str):
        Title of the plot

    * **variable_names** (list):
        List of variable names
        Default: [r"$X_{}$".format(i) for i in range(num_vars)]

    * **kwargs (dict):
        Keyword arguments for the plot to be passed to matplotlib.pyplot.bar

    """

    if indices_1 is None and indices_2 is None:
        raise ValueError("Please provide two indices to plot")

    if len(indices_1) != len(indices_2):
        raise ValueError("indices_1 and indices_2 should have the same length")

    num_vars = len(indices_1)

    if variable_names is None:
        variable_names = [r"$X_{}$".format(i + 1) for i in range(num_vars)]

    # Check if confidence intervals are available
    if confidence_interval_1 is not None:
        conf_int_flag_1 = True
        error_1 = confidence_interval_1[:, 1] - indices_1
    else:
        conf_int_flag_1 = False

    if confidence_interval_2 is not None:
        conf_int_flag_2 = True
        error_2 = confidence_interval_2[:, 1] - indices_2
    else:
        conf_int_flag_2 = False

    # x and y data
    _idx = np.arange(num_vars)

    indices_1 = np.around(indices_1, decimals=2)  # round to 2 decimal places

    if indices_2 is not None:
        indices_2 = np.around(indices_2, decimals=2)  # round to 2 decimal places

    # Plot two indices side by side
    fig, ax = plt.subplots()
    width = 0.3
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    bar_indices_1 = ax.bar(
        _idx - width / 2,  # x-axis
        indices_1,  # y-axis
        width=width,  # bar width
        color="C0",  # bar color
        # alpha=0.5,  # bar transparency
        label=label_1,  # bar label
        yerr=error_1 if conf_int_flag_1 else None,
        ecolor="k",  # error bar color
        capsize=5,  # error bar cap size in pt
        **kwargs,
    )

    bar_indices_2 = ax.bar(
        _idx + width / 2,  # x-axis
        indices_2,  # y-axis
        width=width,  # bar width
        color="C1",  # bar color
        # alpha=0.5,  # bar transparency
        label=label_2,  # bar label
        yerr=error_2 if conf_int_flag_2 else None,
        ecolor="k",  # error bar color
        capsize=5,  # error bar cap size in pt
        **kwargs,
    )

    ax.bar_label(bar_indices_1, label_type="edge", fontsize=10)
    ax.bar_label(bar_indices_2, label_type="edge", fontsize=10)
    ax.set_xticks(_idx, variable_names)
    ax.set_xlabel("Model inputs")
    ax.set_title(plot_title)
    ax.set_ylim(top=1)  # set only upper limit of y to 1
    ax.legend()

    plt.show()

    return fig, ax


@beartype
def plot_second_order_indices(
    indices: NumpyFloatArray,
    num_vars: int,
    confidence_interval: NumpyFloatArray = None,
    plot_title: str = "Second order indices",
    variable_names: list = None,
    **kwargs,
):

    """

    This function plots second order indices (with confidence intervals)
    in a bar plot.

    **Inputs:**

    * **indices** (list or ndarray):
        list/array of second order indices
        Shape: (n_parameters)

    * **confidence_interval** (list or ndarray):
        list/array of confidence interval for the second order indices.
        Shape: (n_p, 2)

    * **label** (str):
        Label of the plot

    * **plot_title** (str):
        Title of the plot

    * **variable_names** (list):
        List of variable names
        Default: (Assumes that the indices are in lexicographic order.)
        [r"$X_{}$".format(i) for i in range(n_parameters)]

    * **kwargs (dict):
        Keyword arguments for the plot to be passed to matplotlib.pyplot.bar

    """

    num_second_order_terms = len(indices)

    if variable_names is None:
        variable_names = [r"$X_{}$".format(i + 1) for i in range(num_vars)]

    # All combinations of variables
    all_combs = list(itertools.combinations(variable_names, 2))

    # # Create a list of all combinations of variables
    all_combs_list = [" ".join(comb) for comb in all_combs]

    # Check if confidence intervals are available
    if confidence_interval is not None:
        conf_int_flag = True
        error = confidence_interval[:, 1] - indices
    else:
        conf_int_flag = False

    # x and y data
    _idx = np.arange(num_second_order_terms)

    indices = np.around(indices, decimals=2)  # round to 2 decimal places

    # Plot one index
    fig, ax = plt.subplots()
    width = 0.3
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    index_bar = ax.bar(
        _idx,  # x-axis
        indices,  # y-axis
        width=width,  # bar width
        yerr=error if conf_int_flag else None,  # error bars
        ecolor="k",  # error bar color
        capsize=5,  # error bar cap size in pt
        **kwargs,
    )

    ax.bar_label(index_bar, label_type="edge", fontsize=10)

    ax.set_xticks(_idx, all_combs_list)
    # generally, there are many second order terms
    # so we need to make sure that the labels are
    # not overlapping. We do this by rotating the labels
    plt.setp(
        ax.get_xticklabels(),
        rotation=30,
        horizontalalignment="right",
    )
    ax.set_xlabel("Model inputs")
    ax.set_ylim(top=1)  # set only upper limit of y to 1
    ax.set_title(plot_title)

    plt.show()

    return fig, ax
