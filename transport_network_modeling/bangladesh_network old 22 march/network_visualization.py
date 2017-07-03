from __future__ import division
from matplotlib import pyplot as plt
from matplotlib.pylab import *
import matplotlib.colors as colors

import pandas as pd
import numpy as np

import geopandas as gp

def plot_network_admcolmap_betweenness(gdf,gdf2, colname,betweenness_string,
                                       cmap='OrRd', linewidth=1.25, edgecolor='grey',
                                       maxbetweenness=0, maxpop=0, thres1=0.1, thres2=0.2):
    fig, ax = plt.subplots(figsize=(12,9))

    ax.set_aspect('equal')

    valmin1 = min(list(gdf2[betweenness_string]))
    valmax1 = max(list(gdf2[betweenness_string]))
    gdf2.plot(ax=ax, column=betweenness_string, cmap=cmap,vmin=valmin1, vmax=valmax1, linewidth=linewidth)

    #adjust linewidth based on betweenness
    betweenness_list = list(gdf2[betweenness_string])
    #change small betweenness values to 0.1 so that they are still visible in the figure
    betweenness_list = [1 if x < thres1 else 2 if x >= thres1 and x < thres2 else 3.5 for x in betweenness_list]
    betweenness_list = [0.1 if x < 0.1 else x for x in betweenness_list]
    i = 0
    for ln in ax.lines:
        ln.set_linewidth(betweenness_list[i]*1)
        ln.set_linewidth(betweenness_list[i])
        i +=1

    valmin2 = min(list(gdf[colname]))
    valmax2 = max(list(gdf[colname]))
    gdf.plot(ax=ax, column=colname, cmap='Greys',vmin=valmin2, vmax=valmax2, linewidth=0.5, edgecolor=edgecolor, alpha=0.3)

    ax.set_title(colname)
    #remove the lon-lat in the x-y axis of the plot
    ax.axis('off')

    # add colorbar1
    fig = ax.get_figure()
    cax = fig.add_axes([0.85, 0.45, 0.02, 0.43])
    sm = plt.cm.ScalarMappable(cmap='Greys')
    columnlist = list(gdf[colname])
    columnlist.append(0)
    columnlist.append(maxpop) #hardcoded, not good
    cbmin, cbmax = min(columnlist), max(columnlist)
    sm.set_array(columnlist)
    cb = plt.colorbar(sm, cax=cax, label = colname, alpha=0.3)
    labels = [0, cbmax/4, cbmax/4*2, cbmax/4*3, cbmax/4*4]
    loc = labels
    cb.set_ticks(loc)
    cb.set_ticklabels(labels)
    cb.ax.yaxis.label.set_font_properties(matplotlib.font_manager.FontProperties(size=16))
    cb.ax.tick_params(labelsize=16)

    #add colorbar2
    fig = ax.get_figure()
    cax = fig.add_axes([0.7, 0.45, 0.02, 0.43])
    sm = plt.cm.ScalarMappable(cmap=cmap)
    columnlist = list(gdf2[betweenness_string])
    columnlist.append(0)
    columnlist.append(maxbetweenness)
    cbmin, cbmax = min(columnlist), max(columnlist)
    cbmin, cbmax = round(cbmin,3), round(cbmax,3)
    sm.set_array(columnlist)
    cb = plt.colorbar(sm, cax=cax, label=betweenness_string)
    labels = [0, cbmax/4, cbmax/4*2, cbmax/4*3, cbmax/4*4]
    loc = labels
    cb.set_ticks(loc)
    cb.set_ticklabels(labels)
    cb.ax.yaxis.label.set_font_properties(matplotlib.font_manager.FontProperties(size=16))
    cb.ax.tick_params(labelsize=16)

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap