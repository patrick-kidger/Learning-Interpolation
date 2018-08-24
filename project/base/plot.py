"""Provides for plotting results."""

# Pretty pictures!
#
#          ;M";::;; 
#         ,':;: ""'. 
#        ,M;. ;MM;;M: 
#        ;MMM::MMMMM: 
#       ,MMMMM'MMMMM: 
#       ;MMMMM MMMMMM 
#       MMMMM::MMMMMM: 
#       :MM:',;MMMMMM' 
#       ':: 'MMMMMMM: 
#         '; :MMMMM" 
#            ''"""' 
#             . 
#             M 
#             M 
# .           M           . 
# 'M..        M        ,;M' 
#  'MM;.      M       ;MM: 
#   :MMM.     M      ;MM: 
#   'MMM;     M     :MMM: 
#    MMMM.    M     MMMM: 
#   :MMMM:    M     MMMM: 
#   :MMMM:    M    :MMMM: 
#   :MMMMM    M    ;MMMM: 
#   'MMMMM;   M   ,MMMMM: 
#    :MMMMM.  M   ;MMMMM' 
#     :MMMM;  M  :MMMMM" 
#      'MMMM  M  ;MMMM" 
# -hrr- ':MM  M ,MMM:' 
#         "": M :""' 

import numpy as np

import ipympl
import matplotlib.pyplot as plt
# Has side effects allowing for 3D plots
from mpl_toolkits.mplot3d import Axes3D

from . import evalu
from . import factory as fac
from . import grid


def make_3d_ax_for_grid_plotting(fig, loc=(1, 1, 1)):
    """Creates an axis to start plotting on."""
    ax = fig.add_subplot(*loc, projection='3d')
    t_list, x_list = list(zip(*grid.coarse_grid((0, 0))))
    ax.set_ylim(min(t_list), max(t_list))
    ax.set_xlim(min(x_list), max(x_list))
    return ax


def grid_plot(ax, data, cg_or_fg=None, label=''):
    """Plots some data according to either a coarse or fine grid."""
    
    if cg_or_fg == 'cg':
        grid_ = grid.coarse_grid((0, 0))
    elif cg_or_fg == 'fg':
        grid_ = grid.fine_grid((0, 0))
    else:
        raise ValueError("Argument cg_or_fg must be 'cg' or 'fg'.")
    ax.scatter(*zip(*grid_), data, label=label)
    
    
# No idea what the actual name of this type of plot is called, but I'm calling it an
# over plot.
def over_plot(xvals, uvals, step=1, axis=True, offset=0.01):
    """Creates an over plot of :uvals[i]: against :xvals:, for
    i in some range.
    
    Optional arguments:
    :int step: Specifying the step size that i should take.
        Defaults to 1. (i.e. plot everything in uvals.)
    :bool axis: Whether or not to display the axes. Defaults to
        True.
    :float offset: How much to offset each plot from each other.
    """
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    if not axis:
        ax.axis('off')
    for i in range(0, len(uvals), step):
        zorder = len(uvals) - round(i / step)
        offset_ = offset * i
        ax.fill_between(xvals, offset_ + uvals[i], offset_, color='white', zorder=zorder)
        ax.plot(xvals, offset_ + uvals[i], color='black', zorder=zorder)

        
### Visualising the results of regressors

# Only plots fine grid style stuff at the moment
def plot_regressors(regressors, *args, **kwargs):
    """Plots the results of some regressors using the given data."""
    
    regressor_factories = [fac.RegressorFactory(regressor=regressor) 
                           for regressor in regressors]
    return plot_regressor_factories(regressor_factories, *args, **kwargs)


def plot_regressor_factories(regressor_factories, names, X, y, plot_size=8, 
                             legend=True, ticklabels=True):
    """Plots the results of some :regressor_factories: using the 
    given data :X: :y:.
    Other arguments:
    :int plot_size: How large to plot the results. Defaults to 8.
    :bool legend: Whether or not to put a legend on the plots. Defaults to
        True.
    :bool ticklabels: Whether or not to put ticklabels on the plots. 
        Defaults to True.
    """
    
    fig = plt.figure(figsize=(plot_size, 
                              plot_size * len(regressor_factories)))
    
    X = np.array([X])
    y = np.array([y])
    results = evalu._eval_regressors(regressor_factories, X, y)
    
    for i, (result, name) in enumerate(zip(results, names)):
        ax = make_3d_ax_for_grid_plotting(fig, (len(regressor_factories), 1, i + 1))
        grid_plot(ax, X, 'cg', '_nolegend_')
        grid_plot(ax, result.prediction, 'fg', name)
        if legend:
            ax.legend()
        if not ticklabels:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
        
    return results


# Slightly hacky convenience function
def plot_reg_and_fac(reg_or_fac, *args, **kwargs):
    """Plots the results of either regressors or their factories using
    the given data.
    """
    
    regressor_factories = []
    for rf in reg_or_fac:
        if isinstance(rf, fac.RegressorFactoryBase):
            regressor_factories.append(rf)
        else:
            regressor_factories.append(fac.RegressorFactory(regressor=rf))
            
    return plot_regressor_factories(regressor_factories, *args, **kwargs)
