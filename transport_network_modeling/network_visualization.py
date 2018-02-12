###################################################################################################
# Module: network_visualization.py
# Description: Visualizing results of criticality analysis
# Author: Bramka Arga Jafino
# Web: https://github.com/bramkaarga/transcrit
###################################################################################################

from __future__ import division
from matplotlib import pyplot as plt
from matplotlib.pylab import *
import matplotlib.colors as colors

import pandas as pd
import math
import numpy as np

import geopandas as gp

import seaborn as sns
from scipy.stats import gaussian_kde
import scipy

__all__ = ['plot_network_admcolmap_betweenness',
           'plot_socioeconomic_attribute',
           'truncate_colormap',
           'plot_network_admcolmap_betweenness_new',
           'plot_od_heatmap',
		   'plot_network_multimodal',
           'plot_interactive',
           'overlap_distribution',
           'correlation_plot',
           'two_correlation_plot']

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
    #betweenness_list = [0.1 if x < 0.1 else x for x in betweenness_list]
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

def plot_socioeconomic_attribute(gdf, colname,cmap='OrRd', linewidth=1.25, edgecolor='grey', maxpop=0):
    '''
    Function to draw socioeconomic attribute of regions

    Parameters
    ------------
    gdf: GeoDataFrame
        GeoDataFrame of region (e.g. districts) boundaries
    colname: basestring
        String of socioeconomic attribute from gdf that wants to be presented
    cmap: Colormap Object / string
        Fill colormap of regions
    linewidth: float
        Default width of network links
    edgecolor: Colormap Object / string
        Colormap of region boundaries
    maxpop: float
        Maximum value of the socioeconomic attribute that wants to be presented

    Returns
    ------------
    matplotlib figure
    '''


    print('maximum number of '+colname+' is',max(list(gdf[colname])))
    fig, ax = plt.subplots(figsize=(12,9))

    ax.set_aspect('equal')

    valmin2 = min(list(gdf[colname]))
    valmax2 = max(list(gdf[colname]))
    gdf.plot(ax=ax, column=colname, cmap=cmap,vmin=valmin2, vmax=valmax2, linewidth=0.5, edgecolor=edgecolor, alpha=0.3)

    ax.set_title(colname)

    #remove the lon-lat in the x-y axis of the plot
    ax.axis('off')

    # add colorbar1
    fig = ax.get_figure()
    cax = fig.add_axes([0.7, 0.45, 0.02, 0.43])
    sm = plt.cm.ScalarMappable(cmap=cmap)
    columnlist = list(gdf[colname])
    columnlist.append(0)
    columnlist.append(maxpop)
    cbmin, cbmax = min(columnlist), max(columnlist)
    sm.set_array(columnlist)
    cb = plt.colorbar(sm, cax=cax, label = colname, alpha=0.3)
    labels = [0, cbmax/4, cbmax/4*2, cbmax/4*3, cbmax/4*4]
    loc = labels
    cb.set_ticks(loc)
    cb.set_ticklabels(labels)
    cb.ax.yaxis.label.set_font_properties(matplotlib.font_manager.FontProperties(size=16))
    cb.ax.tick_params(labelsize=16)

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    '''
    Truncate upper and lower limit of a standard colormap

    Parameters
    ------------
    cmap: Colormap Object / string
        Standard colormap that wants to be truncated
    minval: float
        Lower limit of truncation (0-1)
    maxval: float
        Upper limit of truncation (0-1)

    Returns
    ------------
    Colormap object
    '''

    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def _get_percentile(gdf2, col, n):
    #get n-th percentile of a DataFrame column
    get_col = list(gdf2[col])
    get_col = [x for x in get_col if x > 0]
    nth_percentile = np.percentile(get_col, n)

    return nth_percentile

def plot_network_admcolmap_betweenness_new(gdf, gdf2, colname, criticality_string,
                                       cmap='OrRd', linewidth=1.25, edgecolor='grey',
                                       maxcrit=0, maxpop=0, perc1=60, perc2=90):
    '''
    Function to draw criticality results

    Parameters
    ------------
    gdf: GeoDataFrame
        GeoDataFrame of region (e.g. districts) boundaries
    gdf2: GeoDataFrame
        GeoDataFrame of transport network
    colname: basestring
        String of socioeconomic attribute from gdf that wants to be presented
    criticality_string: string
        String of criticality metric's column name from gdf2 that wants to be presented
    cmap: Colormap Object / string
        Colormap of network links
    linewidth: float
        Default width of network links
    edgecolor: Colormap Object / string
        Colormap of region boundaries
    maxcrit: float
        Maximum value of the corresponding criticality metrics, act as upper limit of the colorbar
    maxpop: float
        Maximum value of the socioeconomic attribute that wants to be presented
    perc1: float
        Lower threshold for changing the links' linewidth
    perc2: float
        Upper threshold for changing the links' linewidth

    Returns
    ------------
    matplotlib figure
    '''

    fig, ax = plt.subplots(figsize=(12,9))

    ax.set_aspect('equal')

    valmin1 = min(list(gdf2[criticality_string]))
    valmax1 = max(list(gdf2[criticality_string]))
    thres1 = _get_percentile(gdf2, criticality_string, perc1)
    thres2 = _get_percentile(gdf2, criticality_string, perc2)

    gdf2.plot(ax=ax, column=criticality_string, cmap=cmap,vmin=valmin1, vmax=valmax1, linewidth=linewidth)

    #adjust linewidth based on betweenness
    betweenness_list = list(gdf2[criticality_string])
    #change the linewidth based on the percentile
    betweenness_list = [1 if x < thres1 else 2 if x >= thres1 and x < thres2 else 3 for x in betweenness_list]
    i = 0
    for ln in ax.lines:
        ln.set_linewidth(betweenness_list[i]*1)
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
    columnlist = list(gdf2[criticality_string])
#     columnlist.append(0)
    columnlist.append(maxcrit)
    cbmin, cbmax = min(columnlist), max(columnlist)
#     cbmin, cbmax = round(cbmin,3), round(cbmax,3)
    sm.set_array(columnlist)
    cb = plt.colorbar(sm, cax=cax, label=criticality_string)
    poin1 = cbmin+(cbmax-cbmin)/4
    poin2 = cbmin+(cbmax-cbmin)/4*2
    poin3 = cbmin+(cbmax-cbmin)/4*3
    labels = [cbmin, poin1, poin2, poin3, cbmax]
    loc = labels
    cb.set_ticks(loc)
    cb.set_ticklabels(labels)
    cb.ax.yaxis.label.set_font_properties(matplotlib.font_manager.FontProperties(size=16))
    cb.ax.tick_params(labelsize=16)

def _log_base_n(x,logn):
    try:
        return math.log(x,logn)
    except:
        return 0

def plot_od_heatmap(OD_df, gdf_points, log=False, logn=100, division=False):

    #adopted from http://nbviewer.jupyter.org/gist/joelotz/5427209

    #Scale data logarithmically if we want to dampen the Chittagong effect (tremendous amount of goods
    #is transported to Chittagong)
    if log:
        OD_df = OD_df.applymap(lambda x: _log_base_n(x, logn))

    # Plot it out
    fig, ax = plt.subplots()

    #if we don't want to aggregate to division level
    if not division:
        heatmap = ax.pcolor(OD_df, cmap=plt.cm.Blues, alpha=0.8)

        ##################################################
        ## FORMAT ##
        ##################################################

        fig = plt.gcf()
        fig.set_size_inches(14,14)

        # turn off the frame
        ax.set_frame_on(False)

        # put the major ticks at the middle of each cell
        ax.set_yticks(np.arange(OD_df.shape[0])+0.5, minor=False)
        ax.set_xticks(np.arange(OD_df.shape[1])+0.5, minor=False)

        # want a more natural, table-like display
        ax.invert_yaxis()
        ax.xaxis.tick_top()

        # Set the labels
        ax.set_xticklabels(gdf_points.District, minor=False)
        ax.set_yticklabels(gdf_points.District, minor=False)

    #if we want to aggregate to division level
    else:
        OD_dummy = OD_df.copy()
        gdf_points_dummy = gdf_points.copy()

        node_division_dict = dict(zip(list(gdf_points_dummy['Node']), list(gdf_points_dummy['Division'])))

        OD_dummy.index = [node_division_dict[x] for x in OD_dummy.columns]
        OD_dummy.columns = [node_division_dict[x] for x in OD_dummy.columns]

        OD_dummy = OD_dummy.groupby(OD_dummy.index).sum().groupby(OD_dummy.columns, axis=1).sum()

        heatmap = ax.pcolor(OD_dummy, cmap=plt.cm.Blues, alpha=0.8)

        ##################################################
        ## FORMAT ##
        ##################################################

        fig = plt.gcf()
        fig.set_size_inches(14,14)

        # turn off the frame
        ax.set_frame_on(False)

        # put the major ticks at the middle of each cell
        ax.set_yticks(np.arange(OD_dummy.shape[0])+0.5, minor=False)
        ax.set_xticks(np.arange(OD_dummy.shape[1])+0.5, minor=False)

        # want a more natural, table-like display
        ax.invert_yaxis()
        ax.xaxis.tick_top()

        # Set the labels
        ax.set_xticklabels(OD_dummy.columns, minor=False, fontsize=18)
        ax.set_yticklabels(OD_dummy.index, minor=False, fontsize=18)

    # rotate the labels
    plt.xticks(rotation=90)

    # give the x and y label
    plt.xlabel('To', fontsize=18)
    ax.xaxis.set_label_position('top')
    plt.ylabel('From', fontsize=18)

    ax.grid(False)

    # Turn off all the ticks
    ax = plt.gca()

    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

def plot_network_multimodal(gdf, gdf2, colname,betweenness_string, cmaps, maxvals=[0,0], minvals=[],
                            linewidth=1.25, edgecolor='grey', perc1=60, perc2=90,
                            modes=['road', 'water']):
    
    fig, ax = plt.subplots(figsize=(12,9))

    ax.set_aspect('equal')

    cmap_pos = [0.72, 0.45, 0.02, 0.43]
    
    ax.set_title(colname)
    
    gdf3 = gdf2.copy()
    gdf3.sort_values(by='mode', inplace=True)
    
    c = 0

    for i, mode in enumerate(modes):
        
        gdf_mode = gdf3.loc[gdf3['mode']==mode]

        valmin1 = min(list(gdf_mode[betweenness_string]))
        valmax1 = max(list(gdf_mode[betweenness_string]))
        
        gdf_mode.plot(ax=ax, column=betweenness_string, cmap=cmaps[i], vmin=valmin1, vmax=valmax1, linewidth=linewidth)
        
        #add colorbar2
        fig = ax.get_figure()
        axes_pos = cmap_pos
        axes_pos[0] = axes_pos[0] + (0.1*i)
        cax = fig.add_axes(axes_pos)
        sm = plt.cm.ScalarMappable(cmap=cmaps[i])
        columnlist = list(gdf_mode[betweenness_string])
        maxbetweenness = max(columnlist)
        try:
            columnlist.append(maxvals[i])
        except:
            columnlist.append(maxbetweenness)
        try:
            columnlist.append(minvals[i])
        except:
            pass
        cbmin, cbmax = min(columnlist), max(columnlist)
        sm.set_array(columnlist)
        cb = plt.colorbar(sm, cax=cax, label=betweenness_string)
        poin1 = cbmin+(cbmax-cbmin)/4
        poin2 = cbmin+(cbmax-cbmin)/4*2
        poin3 = cbmin+(cbmax-cbmin)/4*3
        labels = [cbmin, poin1, poin2, poin3, cbmax]
        loc = labels
        cb.set_ticks(loc)
        cb.set_ticklabels(labels)
        cb.ax.tick_params(labelsize=16)

    try:
        thres1 = _get_percentile(gdf3, betweenness_string, perc1)
    except:
        thres1 = 99999
    try:
        thres2 = _get_percentile(gdf3, betweenness_string, perc2)
    except:
        thres2 = 999999

    #adjust linewidth based on betweenness
    betweenness_list = list(gdf3[betweenness_string])
    #change the linewidth based on the percentile
    betweenness_list = [1 if x < thres1 else 2 if x >= thres1 and x < thres2 else 3 for x in betweenness_list]
    j = 0
    for ln in ax.lines:
        ln.set_linewidth(betweenness_list[j]*1)
        if betweenness_list[j]*1 > 1:
            c +=1
        j +=1
        
    valmin2 = min(list(gdf[colname]))
    valmax2 = max(list(gdf[colname]))
    gdf.plot(ax=ax, column=colname, cmap='Greys',vmin=valmin2, vmax=valmax2, linewidth=0.5, edgecolor=edgecolor, alpha=0.3)
    
    ax.axis('off')
		

	
def plot_interactive(rank, metric, show_division, result_gdf, cmaps, district_gdf):
    
    #record the metric names
    metric_names = {'m1_01' : 'Change in unweighted daily accessibility', 
                    'm1_02' : 'Change in number of nodes accessible within daily reach', 
                    'm2_01': 'Change in unweighted total travel cost', 
                    'm2_02' : 'Change in network average efficiency', 
                    'm3_01' : 'Unweighted link betweenness centrality', 
                    'm3_02': 'Change in region-based unweighted total travel cost', 
                    'm4_01' : 'Minimum link cut centrality', 
                    'm4_02' : 'OD k-connectivity', 
                    'm5_01' : 'Nearby alternative links (simplified)', 
                    'm6_01' : 'Change in weighted accessibility',
                    'm7_01' : 'Change in weighted total travel cost', 
                    'm7_02' : 'Change in expected user exposure', 
                    'm7_03' : 'Change in worst-case user exposure', 
                    'm8_01' : 'Traffic flow data', 
                    'm8_02' : 'Weighted link betweenness centrality', 
                    'm8_03' : 'Volume over capacity', 
                    'm9_01' : 'Unsatisfied demand', 
                    'm10' : 'Exposure to disaster', 
                    'wa_crit' : 'Weighted average criticality score'}
    
    print(str(rank) + ' top links are highlighted from metric ' + metric + ' (' + metric_names[metric] + ')')
    linewidth=1.25
    edgecolor='grey'
    modes=['road', 'water']
    
    fig, ax = plt.subplots(figsize=(12,9))

    ax.set_aspect('equal')
    gdf3 = result_gdf.copy()
    gdf3 = gdf3.sort_values(by='mode')

    for i, mode in enumerate(modes):
        
        gdf_mode = gdf3.loc[gdf3['mode']==mode]

        valmin1 = min(list(gdf_mode[metric]))
        valmax1 = max(list(gdf_mode[metric]))

        gdf_mode.plot(ax=ax, column=metric, cmap=cmaps[i], vmin=valmin1, vmax=valmax1, linewidth=linewidth)
        
    for i, mode in enumerate(modes):
        #adjust linewidth based on betweenness
        betweenness_list = list(gdf3[metric+'_rank'])
        #change the linewidth based on the percentile
        betweenness_list = [1 if x > rank else 3 for x in betweenness_list]
        j = 0
        for ln in ax.lines:
            ln.set_linewidth(betweenness_list[j]*1)
            j +=1
    
    if show_division:
        district_gdf.plot(ax=ax, color='white', edgecolor=edgecolor, alpha=0.7)
    
    ax.axis('off')
    
    plt.show()
	
def overlap_distribution(crit_df, all_metric, save=False, alls=True, name='fig.jpeg'):
    # create the general figure
    import copy
    #either 18,18 or 24,20
    fig1 = plt.figure(figsize=(20,26))
    c = 0
    all_metric2 = copy.deepcopy(all_metric)
    for metric1 in all_metric:
        all_metric2.remove(metric1)
        for metric2 in all_metric2:
            if metric1 != metric2:
                
                data1 = crit_df[metric1]
                data2 = crit_df[metric2]
                min1 = data1.min()
                min2 = data2.min()
        
                max1 = data1.max()
                max2 = data2.max()
                
                if not alls:
                    min1=data1.min()
                    min2=data2.min()

                density1 = gaussian_kde(data1)
                xs1 = np.linspace(min1,max1,200)
                density1.covariance_factor = lambda : .25
                density1._compute_covariance()
                # plt.plot(xs1,density1(xs1))

                # and the first axes using subplot populated with data 
                #either 16,10 or 18,9
                ax1 = fig1.add_subplot(22, 7, c+1)
                line1 = ax1.plot(xs1,density1(xs1), color='Blue', alpha=4)
                d = scipy.zeros(len(density1(xs1)))
                ax1.fill_between(xs1, density1(xs1), where=density1(xs1)>=d, interpolate=True, color='blue', alpha=0.2)
                ax1.set_title(label=metric1 + ' - ' + metric2, fontsize=20)
                ax1.set_axis_bgcolor('white')
                

                density2 = gaussian_kde(data2)
                xs2 = np.linspace(min2,max2,200)
                density2.covariance_factor = lambda : .25
                density2._compute_covariance()

                ax2 = fig1.add_subplot(22, 7, c+1, frameon=False)
                line2 = ax2.plot(xs2,density2(xs2), color='red', alpha=4)
                ax2.xaxis.tick_top()
                ax2.yaxis.tick_right()
                ax2.set_xlabel(metric2, color="red", fontsize=20) 
                ax2.set_ylabel("KDE {}".format(metric2), color="red", fontsize=20)
                ax2.fill_between(xs2, density2(xs2), where=density2(xs2)>=d, interpolate=True, color='red', alpha=0.2)
                ax2.set_axis_bgcolor('white')

                ax1.axes.get_xaxis().set_visible(False)
                ax1.axes.get_yaxis().set_visible(False)
                ax2.axes.get_xaxis().set_visible(False)
                ax2.axes.get_yaxis().set_visible(False)

                c += 1
    fig1.tight_layout()
    if save:
        plt.savefig(name, dpi=80)
    plt.draw()
    plt.show()
	
def correlation_plot(df, title='', center=None, vmin=None, vmax=None, save=False, filename=''):
    sns.set(style="white", font_scale=1.5)
#     sns.set(font_scale=1.3)

    # Generate a mask for the upper triangle
    mask = np.zeros_like(df, dtype=np.bool)
    mask[np.triu_indices_from(mask,1)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(12, 12))

    # Draw the heatmap using seaborn
    sns.heatmap(df, mask=mask, square=True,
                linewidths=.5, cbar_kws={"shrink": .8}, ax=ax, annot=True,
               center=center, vmin=vmin, vmax=vmax, annot_kws={"size":12})

    plt.yticks(rotation=0)
    plt.xticks(rotation=60)
    ax.set_title(title)
    
    if save:
        plt.savefig(filename, dpi=100)

    plt.show()
    
def two_correlation_plot(df1, df2, title='', center=None, vmin=None, vmax=None):
    sns.set(style="white", font_scale=1.5)

    # Generate a mask for the upper triangle
    mask = np.zeros_like(df, dtype=np.bool)
    mask[np.triu_indices_from(mask,1)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(12, 12))

    # Draw the heatmap using seaborn
    sns.heatmap(df, mask=mask, square=True,
                linewidths=.5, cbar_kws={"shrink": .8}, ax=ax, annot=True,
               center=center, vmin=vmin, vmax=vmax, annot_kws={"size":12})

    plt.yticks(rotation=0)
    plt.xticks(rotation=60)
    ax.set_title(title)

    plt.show()

def corr_plot_rank_robustness(df, title='', center=None, vmin=None, vmax=None, save=False, filename=''):
    sns.set(style="white", font_scale=1.5)
#     sns.set(font_scale=1.3)

    # Generate a mask for the upper triangle
    mask = np.zeros_like(df, dtype=np.bool)
    mask[np.triu_indices_from(mask,1)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(12, 12))

    # Draw the heatmap using seaborn
    sns.heatmap(df, mask=mask, square=True,
                linewidths=.5, cbar_kws={"shrink": .8}, ax=ax, annot=False, cmap='RdBu_r',
               center=center, vmin=-1, vmax=1)

    plt.yticks(rotation=0, fontsize=6)
    plt.xticks(rotation=60, fontsize=6)
    ax.set_title(title)

    if save:
        plt.savefig(filename, dpi=200)

    plt.show()

def corr_plot_distribution_robustness(df, title='', center=None, vmin=None, vmax=None, save=False, filename='', cmap='viridis'):
    sns.set(style="white", font_scale=1.5)
#     sns.set(font_scale=1.3)

    # Generate a mask for the upper triangle
    mask = np.zeros_like(df, dtype=np.bool)
    mask[np.triu_indices_from(mask,1)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(12, 12))

    # Draw the heatmap using seaborn
    sns.heatmap(df, mask=mask, square=True,
                linewidths=.5, cbar_kws={"shrink": .8}, ax=ax, annot=False,
               center=center, vmin=0, vmax=1, cmap=cmap)

    plt.yticks(rotation=0, fontsize=6)
    plt.xticks(rotation=60, fontsize=6)
    ax.set_title(title)

    if save:
        plt.savefig(filename, dpi=200)

    plt.show()

def plot_val_sen(df, mad='mad', sd='std', title=''):

    n_groups = len(df.index)

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8

    #set right and top border invisible
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    rects1 = plt.bar(index, df[mad], bar_width,
                     alpha=opacity,
                     color='b')

    rects2 = plt.bar(index + bar_width, df[sd], bar_width,
                     alpha=opacity,
                     color='g')

    plt.legend(['MAD', 'StDev'], bbox_to_anchor=(1, 1))

    plt.axhline(y=0.2, xmin=0, xmax=1, hold=None, color='black', linestyle='dashed')

    plt.xlabel('Metric')
    plt.ylabel('Average')
    plt.xticks(index , tuple(df['metric']), rotation=60)
    plt.legend()
    plt.title(title, y=1.07)

    plt.tight_layout()
    plt.show()

def plot_nodes_multimodal(background, links, nodes, back_col, cmap, maxval=0, minval=0,
                            linewidth=0.5, edgecolor='grey', perc1=60, perc2=90,
                            modes=['road', 'water']):
    
    fig, ax = plt.subplots(figsize=(12,9))

    ax.set_aspect('equal')

    cmap_pos = [0.72, 0.45, 0.02, 0.43]
    
    links2 = links.copy()
    links2.sort_values(by='mode', inplace=True)
    
    link_colors = ['maroon', 'blue']
    alphas= [0.3, 0.2]
    
    c = 0

    #draw roads and waterways
    for i, mode in enumerate(modes):
        
        gdf_mode = links2.loc[links2['mode']==mode]
        gdf_mode.plot(ax=ax, color=link_colors[i], linewidth=linewidth, alpha=alphas[i])
        
    #add colorbar2
    fig = ax.get_figure()
    axes_pos = cmap_pos
    axes_pos[0] = axes_pos[0]
    cax = fig.add_axes(axes_pos)
    sm = plt.cm.ScalarMappable(cmap=cmap)
    columnlist = list(nodes['centrality'])
    maxbetweenness = max(columnlist)
    try:
        columnlist.append(maxval)
    except:
        columnlist.append(maxbetweenness)
    try:
        columnlist.append(minval)
    except:
        pass
    cbmin, cbmax = min(columnlist), max(columnlist)
    sm.set_array(columnlist)
    cb = plt.colorbar(sm, cax=cax, label='Nodes centrality')
    poin1 = cbmin+(cbmax-cbmin)/4
    poin2 = cbmin+(cbmax-cbmin)/4*2
    poin3 = cbmin+(cbmax-cbmin)/4*3
    labels = [cbmin, poin1, poin2, poin3, cbmax]
    loc = labels
    cb.set_ticks(loc)
    cb.set_ticklabels(labels)
    cb.ax.tick_params(labelsize=16)
        
    valmin2 = min(list(background[back_col]))
    valmax2 = max(list(background[back_col]))
    background.plot(ax=ax, column=back_col, cmap='Greys',vmin=valmin2, vmax=valmax2, linewidth=0.5, 
                    edgecolor=edgecolor, alpha=0.2)
    
    #change the size of the nodes based on the percentile  
    try:
        thres1 = _get_percentile(nodes, 'centrality', perc1)
    except:
        thres1 = 99999
    try:
        thres2 = _get_percentile(nodes, 'centrality', perc2)
    except:
        thres2 = 999999
    
    nodes['markersize'] = nodes['centrality'].apply(lambda c: 5 if c < thres1 else 15 if c >= thres1 and c < thres2 else 30)
    
    #draw points
    nd = nodes.plot(ax=ax, column='centrality', cmap=cmap, markersize= nodes['markersize'], alpha=4)
    
    ax.axis('off')