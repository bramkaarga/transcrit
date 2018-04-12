###################################################################################################
# Module: criticality.py
# Description: Network assignments of a transport model and other criticality metrics calculation
# Author: Bramka Arga Jafino
# Web: https://github.com/bramkaarga/transcrit
###################################################################################################

from __future__ import division

import collections
import copy
import time
import os
import sys

from heapq import heappush, heappop
from itertools import count

import pandas as pd
import numpy as np
import networkx as nx
import geopandas as gp

from matplotlib import pyplot as plt
import matplotlib.colors as colors
from matplotlib.pylab import *
from shapely.geometry import LineString, Point, Polygon, MultiPolygon
from scipy.stats.stats import pearsonr
from scipy.stats import spearmanr, ks_2samp

from astropy.stats import median_absolute_deviation

from numpy import std

import ema_workbench

try:
	from transport_network_modeling.od_prep import od_aggregation
except:
	from od_prep import od_aggregation

__all__ = ['aon_assignment',
           'probit_assignment',
           'edge_betweenness_centrality',
           'edge_betweenness_subset_od',
           'betweenness_to_df',
           'edge_betweenness_subset_od_ema',
           'ema_betweenness',
           'k_shortest_paths',
           'ksp_edge_betweenness_subset_od',
           'sp_dict_graph_creation',
           'interdiction_single_edge',
           'min_edge_cut',
           'interdiction_m1',
           'interdiction_m2',
           'interdiction_m6',
           'interdiction_user_exposure',
           'interdiction_m3_02',
           'metric_m5_01',
           'correlate_metrics_pearson',
           'correlate_metrics_spearman',
           'correlate_metrics_ks']


def aon_assignment(G, sources, targets, weight, od):
    '''
    Function to do All-or-Nothing assignment on transport network

    Parameters
    ------------
    G: Graph
        Transport network Graph Networkx object that will be analyzed
    sources: list
        List of nodes (integer) that will be used as sources. The integer should correspond to
        node id in G Graph
    targets: list
        List of nodes (integer) that will be used as targets. The integer should correspond to
        node id in G Graph
    weight: str
        String which corresponds to attribute of G Graph's edges that will be used as penalty for each
        edge. In most cases this is defined as 'length' of the edge.
    od: DataFrame
        OD matrix dataframe

    Returns
    ------------
    d: dict
        Dictionary with edge tuple as keys (e.g. (2,3) ) and flow value as values
    '''

    #create empty dict
    d={}

    #iterate over all sources
    for i in range(len(sources)):
        source = sources[i]
        #iterate over all edges
        for j in range(len(targets)):
            target = targets[j]
            #it is assumed that there is no self-loop on the node
            #e.g. there is no flow from node A to node A
            if source != target :
                #determine shortest path between the OD pair
                sp_dijk_all = nx.dijkstra_path(G, source=source, target=target, weight=weight)
                #update the betweenness value of all edges in the shortest path
                flow = od[source][target]
                for j in range(len(sp_dijk_all)-1):
                    lst = [sp_dijk_all[j],sp_dijk_all[j+1]]
                    lst = [min(lst), max(lst)]
                    tup = tuple(lst)
                    if tup in d.keys():
                        d[tup]+=1*flow
                    else:
                        d.update({tup:1*flow})

    #assign 0 to all edges which don't belong to any shortest path
    #at the same time, record all the correct order of edges name
    edges_list = []
    for u,v in G.edges():
        elst = [u,v]
        elst = [min(elst), max(elst)]
        etup = tuple(elst)
        if not etup in d.keys():
            d.update({etup:0})
        tup = tuple([u,v])
        edges_list.append(tup)

    #alter the tuple(u,v) to tuple(v,u) if the order is inconsistent with the original graph's order
    d1 = {}
    for key, val in d.items():
        if not key in edges_list:
            tup = tuple([key[1], key[0]])
            d1.update({tup:val})
        else:
            d1.update({key:val})

    return d1


def probit_assignment(G, sources, targets, weight, od, N=5, sd=10, penalty=0):
    '''
    Function to do stochastic probit assignment on transport network. The weight of the transport network
    is sampled by normal distribution with the original link weight as the mean.

    Parameters
    ------------
    G: Graph
        Transport network Graph Networkx object that will be analyzed
    sources: list
        List of nodes (integer) that will be used as sources. The integer should correspond to
        node id in G Graph
    targets: list
        List of nodes (integer) that will be used as targets. The integer should correspond to
        node id in G Graph
    weight: str
        String which corresponds to attribute of G Graph's edges that will be used as penalty for each
        edge. In most cases this is defined as 'length' of the edge.
    od: DataFrame
        OD matrix dataframe
    N: int
        Number of probit iterations that want to be performed
    sd: int
        Percentage of the link's weight that will be used as standard deviation of the normal distribution (e.g.
        if 10 is inputted, then the standard deviation is 10% of the link's weight). If you don't want to sample
        over the normal distribution, set sd to 0.
    penalty: double
        Penalty that is given to links which have been part of shortest paths set. If set, the value should be higher
        than 1. The intention is to force the code to find distinguished shortest paths between each probit iteration
        by increasing the weight of links that have been part of shortest paths in previous iterations.

    Returns
    ------------
    d: dict
        Dictionary with edge tuple as keys (e.g. (2,3) ) and flow value as values
    '''

    #create empty dict
    d = collections.defaultdict(float)

    #create copy of original network to avoid changing the attributes of the original network
    G1 = G.copy()
    
    #iterate N times
    #in each iteration, sample the link's weight using a normal distribution
    for i in np.arange(N):
        length_dict = {}
        for u, v, data in G1.edges(data=True):
            tup = tuple([u,v])
            if sd > 0:
                length_mean = data[weight]
                stdev = sd/100
                length_sd = length_mean * stdev
                try:
                    length = np.random.normal(length_mean, length_sd)
                except:
                    length = length_mean
                if length < 0:
                    length = 0
            else:
                length = data[weight]
            length_dict[tup] = length

        #create a copy of G1 since we want to work the penalty on G1 later
        G2 = G1.copy()

        #set the attribute of G2, we'll work the assignment based on G2's weight information
        nx.set_edge_attributes(G2, name=weight, values=length_dict)

        #iterate over all sources
        penalty_set = set()
        for i, source in enumerate(sources):
            #iterate over all edges
            for j, target in enumerate(targets):
                #it is assumed that there is no self-loop on the node
                #e.g. there is no link from node A to node A
                if source == target:
                    continue
                
                #determine shortest path between the OD pair
                sp_dijk_all = nx.dijkstra_path(G2, source=source, target=target, 
                                               weight=weight)


                #update the betweenness value of all edges in the shortest path
                flow = od.ix[source, target]
                
                #divide the flow over the number of iteration
                flow = flow/N
                for j in range(len(sp_dijk_all)-1):
                    lst = [sp_dijk_all[j], sp_dijk_all[j+1]]
                    lst = [min(lst), max(lst)]
                    tup = tuple(lst)
                    d[tup] += flow

                    #if we want to work with penalty, record the shortest paths
                    if penalty:
                        penalty_set.add(tup)
                        tup = tup[::-1]
                        penalty_set.add(tup)

        #if work with penalty, update the weight of the links which belong to the shortest paths
        if penalty:
            penalty_dict = {}
            for u, v,data in G1.edges(data=True):
                if tuple([u,v]) in penalty_set:
                    length = data[weight] * penalty
                else:
                    length = data[weight]
                penalty_dict[tuple([u,v])] = length

            nx.set_edge_attributes(G1, name=weight, values=penalty_dict)

    #assign 0 to all edges which don't belong to any shortest path
    #at the same time, record all the correct order of edges name
    edges_set = set()
    for u,v in G.edges():
        elst = [u,v]
        elst = [min(elst), max(elst)]
        etup = tuple(elst)
        if not etup in d:
            d[etup] = 0.0
        tup = tuple([u,v])
        edges_set.add(tup)

    #alter the tuple(u,v) to tuple(v,u) if the order is inconsistent with the original graph's order
    d1 = {}
    for key, val in d.items():
        if not key in edges_set:
            tup = tuple([key[1], key[0]])
            d1[tup] = val
        else:
            d1[key] = val

    return d1


def correlate_metric_spearman(df, m_a, m_b):

    df2 = df[df[m_a] + df[m_b] != 0]
    r, p = spearmanr(df2[m_a], df2[m_b])

    return r, p, len(df2)


def edge_betweenness_centrality(flow, od):
    '''
    Function to do stochastic probit assignment on transport network. The weight of the transport network
    is sampled by normal distribution with the original link weight as the mean

    Parameters
    ------------
    flow: dict
        Flow dictionary obtained from assignment function (e.g. from aon_assignment or probit_assignment)
    od: DataFrame
        OD matrix dataframe

    Returns
    ------------
    d: dict
        Dictionary with edge tuple as keys (e.g. (2,3) ) and betweenness value as values
    '''

    #record the total flow in the network
    totalval = (sum(od.sum()))

    #copy the flow to avoid changing the original flow dictionary
    flow2 = flow.copy()

    #normalize the flow
    for key, val in flow2.items():
        flow2[key] = val / totalval

    return flow2


def edge_betweenness_subset_od(G, sources, targets, weight, od):
    '''
    Old function before betweenness centrality and flow assignment were separated.
    Calculating edge betweenness centrality between only subset of nodes in the network (e.g. between districts)

    Parameters
    ------------
    G: Graph
        Transport network Graph Networkx object that will be analyzed
    sources: list
        List of nodes (integer) that will be used as sources. The integer should correspond to
        node id in G Graph
    targets: list
        List of nodes (integer) that will be used as targets. The integer should correspond to
        node id in G Graph
    weight: str
        String which corresponds to attribute of G Graph's edges that will be used as penalty for each
        edge. In most cases this is defined as 'length' of the edge.
    od: DataFrame
        OD matrix dataframe

    Returns
    ------------
    d: dict
        Dictionary with edge tuple as keys (e.g. (2,3) ) and betweenness value as values
    '''

    #create empty dict
    d={}

    #iterate over all sources
    for i in range(len(sources)):
        source = sources[i]
        #iterate over all edges
        for j in range(len(targets)):
            target = targets[j]
            #it is assumed that there is no self-loop on the node
            #e.g. there is no flow from node A to node A
            if source != target :
                #determine shortest path between the OD pair
                sp_dijk_all = nx.dijkstra_path(G, source=source, target=target, weight=weight)
                #update the betweenness value of all edges in the shortest path
                flow = od[source][target]
                for j in range(len(sp_dijk_all)-1):
                    lst = [sp_dijk_all[j],sp_dijk_all[j+1]]
                    lst = [min(lst), max(lst)]
                    tup = tuple(lst)
                    if tup in d.keys():
                        d[tup]+=1*flow
                    else:
                        d.update({tup:1*flow})

    #normalize the betweenness value
    totalval = (sum(od.sum()))
    for key, val in d.items():
        d[key] = val / totalval

    #assign 0 to all edges which don't belong to any shortest path
    for u,v in G.edges():
        elst = [u,v]
        elst = [min(elst), max(elst)]
        etup = tuple(elst)
        if not etup in d.keys():
            d.update({etup:0})

    return d


def betweenness_to_df(gdf,betweenness,betweenness_string):
    '''
    Append betweenness centrality result to the transport network's GeoDataFrame.
    For visualization purpose later.

    Parameters
    ------------
    gdf: GeoDataFrame
        GeoDataFrame (Linestring) of the original transport network
    betweenness: dict
        Dictionary with edge tuple as keys (e.g. (2,3) ) and betweenness value as values
    betweenness_string: str
        String of betweenness dictionary's object name

    Returns
    ------------
    gdf_final: GeoDataFrame
        Updated gdf with additional column of betweenness centrality
    betweenness_df: DataFrame
        Betweenness dictionary transformed into dataframe
    '''

    betweenness_df = pd.DataFrame(list(betweenness.items()), columns=['FromTo_tuple', betweenness_string])

    FromTo_tuple = betweenness_df['FromTo_tuple'].tolist()
    FromTo_tolist = []
    for i in FromTo_tuple:
        odlist = list(i)
        minval = min(odlist)
        maxval = max(odlist)
        val = str(minval) + str(maxval)
        FromTo_tolist.append(val)

    betweenness_df['FromTo'] = FromTo_tolist

    c = []
    for i in range(len(gdf)):
        minval = min([gdf['TNODE_'][i],gdf['FNODE_'][i]])
        maxval = max([gdf['TNODE_'][i],gdf['FNODE_'][i]])
        val = str(minval) + str(maxval)
        c.append(val)
    gdf['FromTo'] = c

    gdf_final = pd.merge(gdf,betweenness_df,on='FromTo',how='outer')

    del gdf_final['FromTo_tuple']

    return gdf_final, betweenness_df


def _shortest_path_record(G, sources, targets, weight):
    '''
    Record links in shortest pats of all od pairs
    '''

    d={}
    for i in range(len(sources)):
        source = sources[i]
        for j in range(len(targets)):
            target = targets[j]
            if source != target :
                sp_dijk_all = nx.dijkstra_path(G, source=source, target=target, weight=weight)
                od_pair = str(source)+str(target)
                d[od_pair] = (sp_dijk_all, source, target)
    return d


def edge_betweenness_subset_od_ema(G, sp_dict, od):
    d={}
    for key, val in sp_dict.items():
        source = val[1]
        target = val[2]
        sp = val[0]
        flow = od[source][target]
        for j in range(len(sp)-1):
            lst = [sp[j],sp[j+1]]
            lst = [min(lst), max(lst)]
            tup = tuple(lst)
            #the codes below take almost one minute
            if tup in d.keys():
                d[tup]+=1*flow
            else:
                d.update({tup:1*flow})
    totalval = (sum(od.sum()))
    for key, val in d.items():
        d[key] = val / totalval

    for u,v in G.edges():
        elst = [u,v]
        elst = [min(elst), max(elst)]
        etup = tuple(elst)
        if not etup in d.keys():
            d.update({etup:0})
    return d


def ema_betweenness(prod_lists, OD_all_dict, G, sp_dict, **factors_dict):
    OD_final_df = od_aggregation(OD_all_dict, **factors_dict)


    betweenness = edge_betweenness_subset_od_ema(G=G, sp_dict=sp_dict, od=OD_final_df)

    new_d = {}
    for key, val in betweenness.items():
        new_key = str(key[0])+str(key[1])
        new_d[new_key] = val

    return new_d


def k_shortest_paths(G, source, target, k=1, weight='weight'):
    #MAY NOT BE USED ANYMORE
    if source == target:
        return ([0], [[source]])

    length, path = nx.single_source_dijkstra(G, source, target, weight=weight)
    if target not in length:
        raise nx.NetworkXNoPath("node %s not reachable from %s" % (source, target))

    lengths = [length[target]]
    paths = [path[target]]
    c = count()
    B = []
    G_original = G.copy()

    for i in range(1, k):
        for j in range(len(paths[-1]) - 1):
            spur_node = paths[-1][j]
            root_path = paths[-1][:j + 1]

            edges_removed = []
            for c_path in paths:
                if len(c_path) > j and root_path == c_path[:j + 1]:
                    u = c_path[j]
                    v = c_path[j + 1]
                    if G.has_edge(u, v):
                        edge_attr = G.edge[u][v]
                        G.remove_edge(u, v)
                        edges_removed.append((u, v, edge_attr))

            for n in range(len(root_path) - 1):
                node = root_path[n]
                # out-edges
                for u, v, edge_attr in G.edges(node, data=True):
                    G.remove_edge(u, v)
                    edges_removed.append((u, v, edge_attr))

                if G.is_directed():
                    # in-edges
                    for u, v, edge_attr in G.in_edges_iter(node, data=True):
                        G.remove_edge(u, v)
                        edges_removed.append((u, v, edge_attr))

            spur_path_length, spur_path = nx.single_source_dijkstra(G, spur_node, target, weight=weight)
            if target in spur_path and spur_path[target]:
                total_path = root_path[:-1] + spur_path[target]
                total_path_length = _get_path_length(G_original, root_path, weight) + spur_path_length[target]
                heappush(B, (total_path_length, next(c), total_path))

            for e in edges_removed:
                u, v, edge_attr = e
                G.add_edge(u, v, **edge_attr)

        if B:
            (l, _, p) = heappop(B)
            lengths.append(l)
            paths.append(p)
        else:
            break

    return (lengths, paths)


def _get_path_length(G, path, weight='weight'):
    #MAY NOT BE USED ANYMORE
    length = 0
    if len(path) > 1:
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]

            length += G.edge[u][v].get(weight, 1)

    return length


def _total_cost_sp(G, sources, targets, weight, od, weighted=True):
    '''
    Input:
        G                : Graph Networkx object
        sources, targets : List of nodes sources IDs and nodes targets IDs (e.g. the centroid nodes)
        weight           : Edge data key corresponding to the edge weight
        od               : OD matrix for calculating weighted total cost
        weighted         : if True, the total cost is weighted by the OD matrix
    Output:
        d                : Dict with centroid pairs tuple as keys (e.g. (2,3) ) and total cost between them as values
    '''
    d={}
    total_cost = 0
    for i in range(len(sources)):
        source = sources[i]
        for j in range(len(targets)):
            target = targets[j]
            if source != target :
                sp_dijk_distance = nx.dijkstra_path_length(G, source=source, target=target, weight=weight)
                flow = od[source][target]
                if weighted:
                    cost = sp_dijk_distance * flow
                else:
                    cost = sp_dijk_distance
                total_cost += cost
                tup=tuple([source,target])
                d.update({tup:cost})

    return total_cost, d


def sp_dict_graph_creation(G, sources, targets, weight):
    '''
    Function to store shortest path of each OD pair into a dictionary

    Parameters
    ------------
    G: Graph
        Transport network Graph Networkx object that will be analyzed
    sources: list
        List of nodes (integer) that will be used as sources. The integer should correspond to
        node id in G Graph
    targets: list
        List of nodes (integer) that will be used as targets. The integer should correspond to
        node id in G Graph
    weight: str
        String which corresponds to attribute of G Graph's edges that will be used as penalty for each
        edge. In most cases this is defined as 'length' of the edge.

    Returns
    ------------
    sp_dict_graph: dict
        Dictionary with OD pair nodes' id tuple as keys (e.g. (2,3) ) and links in shortest path as values
    '''


    sp_dict = _shortest_path_record(G=G, sources = sources, targets = targets, weight=weight)

    edgelist = []
    for edge in list(G.edges()):
        edgelist.append(edge)

    sp_dict_graph = {}
    for key, val in sp_dict.items():
        source = val[1]
        target = val[2]
        tup = tuple([source, target])
        sp_dict_graph.update({tup:[]})
        for j in range(len(val[0])-1):
            test1 = tuple([val[0][j], val[0][j+1]])
            test2 = tuple([val[0][j+1], val[0][j]])
            if test1 in edgelist:
                sp_dict_graph[tup].append(test1)
            if test2 in edgelist:
                sp_dict_graph[tup].append(test2)
    return sp_dict_graph


def interdiction_single_edge(G2, od, weight, sp_dict_graph, sources, targets):    
    interdiction_costs = {}
    interdiction_disconnected = {}
    interdiction_unsatisfieddemand = {}

    total_cost_base, od_cost_dict = _total_cost_sp(G=G2, sources=sources, targets=targets,
                                                       weight='length', od=od)
    
    edges_odmap = collections.defaultdict(list)
    for key, value in sp_dict_graph.items():
        for entry in value:
            edges_odmap[entry].append(key)

    for edge, ods in sorted(list(edges_odmap.items()), key=lambda x:x[0]):
        od_cost = od_cost_dict.copy()
        G = G2.copy()
        G.remove_edge(edge[0], edge[1])

        disconnected = 0 
        unsatisfied_demand = 0

        for source, target in ods:
            demand = od.loc[source, target]

            try: 
                distance = nx.dijkstra_path_length(G, source=source, target=target, 
                                                   weight=weight)
            except nx.NetworkXNoPath:
                disconnected += 1
                unsatisfied_demand += demand
            else:
                costs = distance * demand
                od_cost[(source, target)] = costs

        total_costs = sum(list(od_cost.values()))
        cost_increase = (total_costs - total_cost_base)/total_cost_base
        unsatisfied_demand = unsatisfied_demand/total_cost_base
        if cost_increase < 0:
            cost_increase = 0

        interdiction_costs[edge] = cost_increase
        interdiction_disconnected[edge] = disconnected
        interdiction_unsatisfieddemand[edge] = unsatisfied_demand
        
    # interdiction_costs = {'{}{}'.format(*sorted(k)):v for k,v in interdiction_costs.items()}
    # disconnected = {'{}{}'.format(*sorted(k)):v for k,v in interdiction_disconnected.items()}
    # unsatisfied_demand = {'{}{}'.format(*sorted(k)):v for k,v in interdiction_unsatisfieddemand.items()}
    return interdiction_costs, interdiction_disconnected, interdiction_unsatisfieddemand


def ksp_edge_betweenness_subset_od(G, sources, targets, weight, od, k):
    '''
    MAY NOT BE USED ANYMORE
    Input:
        G                : Graph Networkx object
        sources, targets : List of nodes sources IDs and nodes targets IDs (e.g. the centroid nodes)
        weight           : Edge data key corresponding to the edge weight
    Output:
        d                : Dict with edge tuple as keys (e.g. (2,3) ) and betweenness value as values
    '''
    d={}
    number=0
    for i in range(len(sources)):
        source = sources[i]
        for j in range(len(targets)):
            target = targets[j]
            if source != target :

                #calculate k-shortest path
                ksp = k_shortest_paths(G = G, source = source, target = target, k = k, weight = weight)

                #store the length of the k-shortest paths
                path_length = ksp[0]
                path_length_set = set(path_length)

                #store total flow between od pair
                flow = od[source][target]

                #calculate logit model for route choice
                #firstly calculate the denominator
                sum_exp = 0
                for i in path_length_set:
                    exp_val = np.exp(-0.05*i)
                    sum_exp += exp_val

                #secondly create list which contains probability of each route
                probability = []
                for i in path_length_set:
                    exp_val = np.exp(-0.05*i)
                    prob = exp_val/sum_exp
                    probability.append(prob)

                #now append the flow*probability to each route
                #iterate for each route
                counter = 0
                for path in path_length_set:
                    index = path_length.index(path)
                    sp = ksp[1][index]
                    for j in range(len(sp)-1):
                        lst = [sp[j],sp[j+1]]
                        lst = [min(lst), max(lst)]
                        tup = tuple(lst)
                        if tup in d.keys():
                            d[tup]+=1*flow*probability[counter]
                        else:
                            d.update({tup:1*flow*probability[counter]})
                    counter += 1

    totalval = (sum(od.sum()))
    for key, val in d.items():
        d[key] = val / totalval

    for u,v in G.edges():
        elst = [u,v]
        elst = [min(elst), max(elst)]
        etup = tuple(elst)
        if not etup in d.keys():
            d.update({etup:0})

    return d


def min_edge_cut(G, centroid_nodes):
    '''
    Function to calculate minimum edge cut criticality of all links in a transport network

    Parameters
    ------------
    G: Graph
        Transport network Graph Networkx object that will be analyzed
    centroid_nodes: list
        List of nodes id (integer) that will be used as sources and targets. The integer should correspond to
        node id in G Graph

    Returns
    ------------
    edgecut_dict: dict
        Dictionary with edge tuple as keys (e.g. (2,3) ) and min edge cut criticality value as values
    '''

    # Save the links cut set for each OD pair
    # Links cut set are set of links which if simultaneously removed will disconnect a specific OD pair
    testcut = []
    for i in range(len(centroid_nodes)):
        for j in range(len(centroid_nodes)):
            if i != j:
                testcut2 = nx.minimum_edge_cut(G=G, s=centroid_nodes[i], t=centroid_nodes[j])
                testcut += testcut2

    # Create dictionary which stores the occurence of each edge in all cut sets
    # Save the edge IDs as dictionary key
    edgecut_dict = {}
    for edge in list(G.edges()):
        edgecut_dict.update({edge: 0})
    #increment the value of the dictionary's key if an edge belongs in a cut set
    for i in testcut:
        try:
            edgecut_dict[i] += 1
        except:
            new_tup = tuple([i[1], i[0]])
            edgecut_dict[new_tup] += 1

    return edgecut_dict

# Metrics M1: Change in unewighted daily accessibility and
# change in number of nodes accessible within daily reach


def _daily_accessibility(centroid, G, theta, weight='length', beta=0.5):
    '''
    Helper function for function interdiction_m1

    return:
    a              : daily accessibility index
    len(sp_length) : number of nodes accessible within daily travel threshold (theta)
    '''

    total_sp_length = 0

    #calculate shortest path length to all other centroids
#     for target in G.nodes(): #why need this?
    sp_length = nx.single_source_dijkstra_path_length(G=G, source=centroid, cutoff=theta, weight=weight)
    count_node = 0
    for item in sp_length:
        try:
            total_sp_length += 1 / (item[1]**beta)
        except:
            pass
        count_node += 1

    #calculate the accessibility
    try:
        a = total_sp_length
    except:
        a = 0

    return a, count_node


def _dict_daily_accessibility(centroids, G, theta, weight='length', beta=0.5):
    '''
    Helper function for function interdiction_m1

    return:
    a_dict     : dictionary of daily accessibility, keyed by centroids id
    a_n_dict   : dictionary of number of nodes accessible within daily travel threshold, keyed by centroids id
    '''

    a_dict = {}
    a_n_dict = {}
    for centroid in centroids:
        a, a_n = _daily_accessibility(centroid=centroid, G=G, theta=theta, weight=weight, beta=beta)
        a_dict.update({centroid:a})
        a_n_dict.update({centroid:a_n})

    return a_dict, a_n_dict


def _sum_daily_accessibility(a_dict, a_n_dict):
    '''
    Helper function for function interdiction_m1
    '''
    sum_a = 0
    for key, val in a_dict.items():
        sum_a += val

    sum_a_n = 0
    for key, val in a_n_dict.items():
        sum_a_n += val

    return sum_a, sum_a_n

def _all_daily_sp_record(G, sources, cutoff, weight):
    '''
    Helper function for function interdiction_m1

    return:
    all_daily_sp_list : list of all sources' daily shortest paths (shortest paths to node which
                        are accessible on daily basis)
    all_daily_sp_dict : dictionary of all shortest paths for each source, keyed by source id
    '''
    edgelist = []
    for edge in list(G.edges()):
        edgelist.append(edge)

    all_daily_sp_list = []
    all_daily_sp_dict = {}
    for source in sources:
        sp_dict = nx.single_source_dijkstra_path(G=G, source=source, cutoff=cutoff, weight=weight)
        source_sp_list = []
        for key, val in sp_dict.items():
            for n in np.arange(0, len(val)-1, 1):
                start = val[n]
                end = val[n+1]
                if tuple([start, end]) in edgelist:
                    all_daily_sp_list.append(tuple([start, end]))
                    source_sp_list.append(tuple([start, end]))
                elif tuple([end, start]) in edgelist:
                    all_daily_sp_list.append(tuple([end, start]))
                    source_sp_list.append(tuple([end, start]))
                else:
                    print('edge not found', start, end)

            source_sp_list = list(set(source_sp_list))
            all_daily_sp_dict.update({source:source_sp_list})
            all_daily_sp_list = list(set(all_daily_sp_list))

    return all_daily_sp_list, all_daily_sp_dict

def interdiction_m1(G2, centroids, theta, weight, beta=0.5):
    '''
    Function to calculate metric M1_01 (change in unweighted daily accessibility)
    and metric M1_02 (change in number of nodes accessible within daily reach)

    Parameters
    ------------
    G2: Graph
        Transport network Graph Networkx object that will be analyzed
    centroids: list
        List of nodes id (integer) that will be used as sources and targets. The integer should correspond to
        node id in G Graph
    theta: float
        Threshold of daily travel distance (in kilometer)
    weight: str
        String which corresponds to attribute of G2 Graph's edges that will be used as penalty for each
        edge. In most cases this is defined as 'length' of the edge
    beta: float
        Distance sensitivity parameter for accessibility calculation

    Returns
    ------------
    m1_01_dict: dict
        Dictionary with edge tuple as keys (e.g. (2,3) ) and change in unweighted daily accessibility as values
    m1_02_dict: dict
        Dictionary with edge tuple as keys (e.g. (2,3) ) and change in number of nodes accessible within daily
        reach as values
    '''

    ff=0
    m1_01_dict = {}
    m1_02_dict = {}

    #record the daily accessibility of each centroid
    a_dict_base, a_n_dict_base = _dict_daily_accessibility(centroids=centroids, G=G2, theta=theta, weight=weight, beta=beta)

    #record the business as usual total accessibility
    sum_a_base, sum_a_n_base = _sum_daily_accessibility(a_dict_base, a_n_dict_base)

    #record the daily shortest paths list for each centroid
    all_daily_sp_list, all_daily_sp_dict = _all_daily_sp_record(G=G2, sources=centroids, cutoff=theta, weight='length')

    #iterate for all edges in daily shortest path list
    print('start')
    for edge in all_daily_sp_list:
        ff += 1
        if ff%50 == 0:
            print(str(ff)+' edges have been interdicted')
        u = edge[0]
        v = edge[1]
        tup = tuple([u,v])

        #make a copy of the daily accessibility dictionary
        a_dict_base2 = a_dict_base.copy()
        a_n_dict_base2 = a_n_dict_base.copy()

        #make a copy of the original graph
        G = G2.copy()

        #remove that edge
        G.remove_edge(u,v)

        #iterate over all centroids
        for key, val in all_daily_sp_dict.items():
            #if the removed edge is part of that centroid's daily shortest path
            #recalculate the daily accessibility of that centroid
            #and update the daily accessibility dictionary
            if tup in val:
                a_new, a_n_new = _daily_accessibility(centroid=key, G=G, theta=theta, weight=weight, beta=beta)
                a_dict_base2.update({key:a_new})
                a_n_dict_base2.update({key:a_n_new})

        sum_a_new, sum_a_n_new = _sum_daily_accessibility(a_dict_base2, a_n_dict_base2)
        m1_01 = sum_a_base / sum_a_new
        m1_01_dict.update({edge:m1_01})

        m1_02 = sum_a_n_base / sum_a_n_new
        m1_02_dict.update({edge:m1_02})


    edgelist = []
    for edge in list(G2.edges()):
        edgelist.append(edge)

    #if an edge does not have value yet
    #assign 0 to it
    for edge in edgelist:
        if not edge in m1_01_dict.keys():
            m1_01_dict.update({edge:0})
        if not edge in m1_02_dict.keys():
            m1_02_dict.update({edge:0})

    return m1_01_dict, m1_02_dict

# Metrics M2: Change in unweighted total travel cost and
# change in network average efficiency

def _total_cost_sp_inversed(G, sources, targets, weight):
    '''
    Helper function for function interdiction_m2

    Input:
        G                : Graph Networkx object
        sources, targets : List of nodes sources IDs and nodes targets IDs (e.g. the centroid nodes)
        weight           : Edge data key corresponding to the edge weight
    Output:
        total_cost       : Sum of inversed total cost
        d                : Dict with centroid pairs tuple as keys (e.g. (2,3) ) and inversed unweighted total
                           cost as values
    '''
    d={}
    total_cost = 0
    for i in range(len(sources)):
        source = sources[i]
        for j in range(len(targets)):
            target = targets[j]
            if source != target :
                sp_dijk_distance = nx.dijkstra_path_length(G, source=source, target=target, weight=weight)
                cost = 1/sp_dijk_distance
                total_cost += cost
                tup=tuple([source,target])
                d.update({tup:cost})

    return total_cost, d

def _network_efficiency_calc(G, total_cost_inversed):
    '''
    Helper function for function interdiction_m2

    return:
    eff: Network efficiency
    '''
    node_calc = 2 / (G.number_of_nodes() * (G.number_of_nodes() - 1))
    eff = node_calc * total_cost_inversed
    return eff

def interdiction_m2(G2, od, weight, sources, targets):
    '''
    Function to calculate metric m2_01 (change in total unweighted travel cost) and
    m2_02 (change in network average efficiency)

    Parameters
    ------------
    G2: Graph
        Transport network Graph Networkx object that will be analyzed
    sources, targets: list
        List of nodes (integer) that will be used as centroids. The integer should correspond to
        node id in G Graph
    weight: str
        String which corresponds to attribute of G2 Graph's edges that will be used as penalty for each
        edge. In most cases this is defined as 'length' of the edge.
    od: DataFrame
        OD matrix dataframe

    Returns
    ------------
    m2_01_dict: dict
        Dictionary with edge tuple as keys (e.g. (2,3) ) and change in total unweighted travel cost as values
    m2_02_dict: dict
        Dictionary with edge tuple as keys (e.g. (2,3) ) and change in network average efficiency as values
    '''

    ff=0
    m2_01_dict = {}
    m2_02_dict = {}

    sp_dict_graph = sp_dict_graph_creation(G=G2, sources=sources,
                                           targets=targets, weight=weight)

    #record unweighted total cost, and shortest path cost for each OD pair
    #for calculating metrics M2_01
    total_cost_base, od_cost_dict = _total_cost_sp(G=G2, sources=sources, targets=targets,
                                                  weight=weight, od=od, weighted=False)

    #record inversed unweighted total cost, and inversed shortest path cost for each OD pair
    #for calculating metrics M2_02
    total_cost_sp_inversed, od_cost_inversed_dict =  _total_cost_sp_inversed(G=G2, sources=sources,
                                                                             targets=targets, weight=weight)

    #record the business as usual network efficiency for calculating metrics M2_02
    efficiency_base = _network_efficiency_calc(G=G2, total_cost_inversed=total_cost_sp_inversed)

    #record all shortest paths of all OD pairs
    path_in_sp_list = []
    for i in sp_dict_graph.items():
        path_in_sp_list += i[1]
    path_in_sp_list = list(set(path_in_sp_list))

    #iterate for each shortest path
    print('start')
    for i in path_in_sp_list:
        ff += 1
        if ff%50 == 0:
            print(str(ff)+' edges have been interdicted')
        u = i[0]
        v = i[1]
        tup = tuple([u,v])

        #make a copy of the unweighted (inversed) cost dictionary
        od_cost_dict2 = od_cost_dict.copy()
        od_cost_inversed_dict2 = od_cost_inversed_dict.copy()

        #make a copy of the original graph
        G = G2.copy()

        #remove that edge
        G.remove_edge(u,v)

        #iterate over all OD pairs
        #very time consuming. gimana kalau untuk setiap edge, di store dia melalui shortest path od pair mana aja?
        #jadi gak perlu iterate 4000 kali untuk setiap edge
        for key, val in sp_dict_graph.items():
            #if the removed edge is part of that OD pair's shortest path
            #recalculate the unweighted (inversed) cost
            #and update the corresponding dictionaries
            if tup in val:
                try:
                    print(tup, key)
                    sp_dijk_distance = nx.dijkstra_path_length(G, source=key[0], target=key[1], weight=weight)
                    cost = sp_dijk_distance
                    od_cost_dict2[key] = cost
                    od_cost_inversed_dict2[key] = 1 / cost
                except:
                    sp_dijk_distance = 9999

        #update (inversed) total cost
        total_cost_new = sum(od_cost_dict2.values())
        total_cost_inversed_new = sum(od_cost_inversed_dict2.values())

        #update efficiency
        efficiency_new = _network_efficiency_calc(G, total_cost_inversed_new)

        #calculate metrics
        m2_01 = total_cost_new/total_cost_base
        if m2_01 < 0:
            m2_01 = 0
        m2_01_dict.update({tup:m2_01})

        m2_02 = (efficiency_base - efficiency_new)/efficiency_base
        m2_02_dict.update({tup:m2_02})


    edgelist = []
    for edge in list(G2.edges()):
        edgelist.append(edge)

    #if an edge does not have value yet
    #assign 0 to it
    for edge in edgelist:
        if not edge in m2_01_dict.keys():
            m2_01_dict.update({edge:0})
        if not edge in m2_02_dict.keys():
            m2_02_dict.update({edge:0})

    return m2_01_dict, m2_02_dict

# Metric M6: Change in weighted daily accessibility

def _weighted_accessibility(G, centroid, targets, flow, weight, beta=0.5):
    a = 0
    acc_dict = {}
    if type(targets)==list:
        for target in targets:
            if target != centroid:
                dist = nx.dijkstra_path_length(G=G, source=centroid, target=target, weight=weight)
                a_val = (flow.ix[centroid] * flow.ix[target])/(dist**beta)
                a = a_val
                acc = a.flow
                acc_dict.update({target:acc})
    else:
        dist = nx.dijkstra_path_length(G=G, source=centroid, target=targets, weight=weight)
        a_val = (flow.ix[centroid] * flow.ix[targets])/(dist**beta)
        a = a_val
        acc_dict = a.flow
            
#     print(acc_dict) 
    return acc_dict

def _sum_weighted_accessibility(G, centroids, flow, weight, beta=0.5):
    a_sum_dict = {}
    a_master_dict = {}
    for centroid in centroids:
        a_dict = _weighted_accessibility(G=G, centroid=centroid, targets=centroids,
                                       flow=flow, weight=weight, beta=beta)
        a_val = sum(a_dict.values())
        a_sum_dict.update({centroid:a_val})
        a_master_dict.update({centroid:a_dict})
    a_sum = sum(a_sum_dict.values())
    
    return a_sum, a_master_dict

def interdiction_m6(G2, weight, centroids, od, beta=0.5):
    '''
    Function to calculate metric m6_01 (change in weighted accessibility)

    Parameters
    ------------
    G2: Graph
        Transport network Graph Networkx object that will be analyzed
    centroids: list
        List of nodes (integer) that will be used as centroids. The integer should correspond to
        node id in G Graph
    weight: str
        String which corresponds to attribute of G2 Graph's edges that will be used as penalty for each
        edge. In most cases this is defined as 'length' of the edge.
    od: DataFrame
        OD matrix dataframe
    beta: float
        Distance sensitivity parameter for accessibility calculation

    Returns
    ------------
    m6_01_dict: dict
        Dictionary with edge tuple as keys (e.g. (2,3) ) and change in weighted accessibility as values
    '''

    ff=0
    m6_01_dict = {}

    #record total traffic going in and going out of each district
    flow = pd.DataFrame(od.sum(axis=0) + od.sum(axis=1))

    #record shortest path of each OD pair
    sp_dict_graph = sp_dict_graph_creation(G=G2, sources=centroids, targets=centroids, weight=weight)


    #record total weighted accessibility, and weighted accessibility for each OD pair
    a_sum_base, a_sum_dict = _sum_weighted_accessibility(G=G2, centroids=centroids,
                                                         flow=flow, weight=weight, beta=beta)

    #record all shortest paths of all OD pairs
    path_in_sp_list = []
    for i in sp_dict_graph.items():
        path_in_sp_list += i[1]
    path_in_sp_list = list(set(path_in_sp_list))

    #iterate for each shortest path
    # print('start')
    for i in path_in_sp_list:
        ff += 1
        # if ff%100 == 0:
        #     print(str(ff)+' edges have been interdicted')
        u = i[0]
        v = i[1]
        tup = tuple([u,v])

        #make a copy of the weighted accessibility dictionary
        a_sum_dict2 = a_sum_dict.copy()

        #make a copy of the original graph
        G = G2.copy()

        #remove that edge
        G.remove_edge(u,v)

        # create empty list to store centroid that has been updated
        updated_centroid = []

        #iterate over all OD pairs
        #very time consuming. gimana kalau untuk setiap edge, di store dia melalui shortest path od pair mana aja?
        #jadi gak perlu iterate 4000 kali untuk setiap edge
        for key, val in sp_dict_graph.items():
            #if the removed edge is part of that OD pair's shortest path
            #recalculate the weighted accessibility for the source
            #and update the corresponding dictionaries
            if not key[0] in updated_centroid:
                if tup in val:
                    updated_centroid.append(key[0])
                    try:
                        a_new = _weighted_accessibility(G=G, centroid=key[0], targets=centroids,
                                                        flow=flow, weight=weight, beta=beta)
                        a_sum_dict2.update({key[0]:a_new})
                    except:
                        sp_dijk_distance = 9999

        #update weighted accessibility
        a_sum_new = sum(a_sum_dict2.values())

        #calculate metrics
        m6_01 = a_sum_base/a_sum_new
        if m6_01 < 0:
            m6_01 = 0
        m6_01_dict.update({tup:m6_01})


    edgelist = []
    for edge in list(G2.edges()):
        edgelist.append(edge)

    #if an edge does not have value yet
    #assign 0 to it
    for edge in edgelist:
        if not edge in m6_01_dict.keys():
            m6_01_dict.update({edge:0})

    return m6_01_dict

# Metrics M7: Change in expected user exposure and
# change in worst case user exposure

def _shortest_path_cost(G, centroids, weight):
    '''
    Helper function for function interdiction_user_exposure

    return:
    sp_cost_dict: dictionary with od pairs centroid id tuple as keys (e.g (2,3)) and shortest path cost between
                  them as values
    '''

    sp_cost_dict = {}
    for centroid in centroids:
        for target in centroids:
            if centroid != target:
                cost = nx.dijkstra_path_length(G=G, source=centroid, target=target, weight=weight)
                sp_cost_dict.update({tuple([centroid, target]):cost})

    return sp_cost_dict

def _user_exposure(G_new, centroid, target, weight, od, sp_cost_dict):
    '''
    Helper function for function interdiction_user_exposure

    return:
    exposure: exposure of a centroid due to link disruption
    '''

    old_cost = sp_cost_dict[tuple([centroid, target])]
    try:
        new_cost = nx.dijkstra_path_length(G=G_new, source=centroid, target=target, weight=weight)
    except:
        new_cost = old_cost

    flow = od[centroid][target]
    exposure = flow * (new_cost - old_cost)

    return exposure

# Metrics M7_02 and M7_03

def interdiction_user_exposure(G2, centroids, weight, od):
    '''
    Function to calculate metric m7_02 (Change in expected user exposure) and
    m7_03 (change in worst case user exposure)

    Parameters
    ------------
    G2: Graph
        Transport network Graph Networkx object that will be analyzed
    centroids: list
        List of nodes (integer) that will be used as centroids. The integer should correspond to
        node id in G Graph
    weight: str
        String which corresponds to attribute of G2 Graph's edges that will be used as penalty for each
        edge. In most cases this is defined as 'length' of the edge.
    od: DataFrame
        OD matrix dataframe

    Returns
    ------------
    m7_02_dict: dict
        Dictionary with edge tuple as keys (e.g. (2,3) ) and Change in expected user exposure as values
    m7_03_dict: dict
        Dictionary with edge tuple as keys (e.g. (2,3) ) and change in worst case user exposure as values
    '''

    ff = 0
    m7_02_dict = {}
    m7_03_dict = {}

    # record business as usual shortest path costs for all OD pairs
    sp_cost_dict = _shortest_path_cost(G=G2, centroids=centroids, weight=weight)

    # records business as usual shortest path edges for all OD pairs
    sp_dict_graph = sp_dict_graph_creation(G=G2, sources=centroids, targets=centroids, weight=weight)

    #record all shortest paths of all OD pairs
    path_in_sp_list = []
    for i in sp_dict_graph.items():
        path_in_sp_list += i[1]
    path_in_sp_list = list(set(path_in_sp_list))


    #iterate for each edges in shortest paths
    print('start')
    for i in path_in_sp_list:
        ff += 1
        if ff%50 == 0:
            print(str(ff)+' edges have been interdicted')

        u = i[0]
        v = i[1]

        #create dictionary for user exposure of each centroids
        user_exposure_dict = {}
        for centroid in centroids:
            user_exposure_dict.update({centroid:[]})

        #make a copy of the original graph
        G = G2.copy()

        #remove that edge
        G.remove_edge(u,v)

        sum_exposure = 0

        #iterate over all OD pairs
        for key, val in sp_dict_graph.items():
            if i in val:
                exposure = _user_exposure(G_new=G, centroid=key[0], target=key[1],
                                          weight=weight, od=od, sp_cost_dict=sp_cost_dict)
                user_exposure_dict[key[0]].append(exposure)

        #calculate expected and worst user exposure
        expected_ue_dict = {}
        worst_ue_dict = {}
        for key, val in user_exposure_dict.items():
            if len(val) > 0:
                average_val = average(val)
                worst_val = max(val)
            else:
                average_val = 0
                worst_val = 0
            expected_ue_dict.update({key:average_val})
            worst_ue_dict.update({key:worst_val})

        #calculate metrics
        m7_02 = sum(expected_ue_dict.values())/len(centroids)
        m7_03 = sum(worst_ue_dict.values())/len(centroids)
        m7_02_dict.update({i:m7_02})
        m7_03_dict.update({i:m7_03})

    edgelist = []
    for edge in list(G2.edges()):
        edgelist.append(edge)

    #if an edge does not have value yet
    #assign 0 to it
    for edge in edgelist:
        if not edge in m7_02_dict.keys():
            m7_02_dict.update({edge:0})
        if not edge in m7_03_dict.keys():
            m7_03_dict.update({edge:0})

    return m7_02_dict, m7_03_dict

def _average_shortest_path_length2(G, weight='length'):
    try:
        dist = nx.average_shortest_path_length(G=G, weight=weight)
    except:
        cc = 0
        if nx.number_connected_components(G) > 1:
            for subgraph in nx.connected_component_subgraphs(G):
                if len(subgraph) > cc:
                    cc = len(subgraph)
                    graph = subgraph
        dist = nx.average_shortest_path_length(G=graph, weight=weight)
    return dist

# Metric M3_02

def interdiction_m3_02(row, div_graph_dict, div_init_avrgdist_dict, weight='length'):
    
    #get the division
    division = row['division']
    
    try:
        G = div_graph_dict[division].copy()
        init_average_sp_distance = div_init_avrgdist_dict[division]
        #remove the path
        G.remove_edge(row['FNODE_'], row['TNODE_'])

        #recalculate average shortest path distance
        new_average_sp_distance = _average_shortest_path_length2(G=G, weight=weight)

        m3_02 = new_average_sp_distance / init_average_sp_distance
        
    except:
        m3_02 = 0
    
    return m3_02
	
def _find_distinct_sp(G, source, target, weight, cutoff=3):
    
    #record all edges
    edgelist = []
    for edge in G.edges():
        edgelist.append(edge)
    
    #create empty shortest path list
    sp_list = []
    
    #record the number of shortest path
    n_sp = 0
    
    #record maximum allowed distance, which is 'cutoff' times the initial shortest path distance
    init_len = nx.dijkstra_path_length(G=G, source=source, target=target, weight=weight)
    max_len = init_len * cutoff
    current_len = 0
    
    #copy the graph
    G2 = G.copy()
    
    num = 0
    
    # iterate as long as the new shortest path distance is smaller than the maximum 
    while current_len < max_len:
        num += 1
        if num%10 == 0:
            print('same od pair has been iterated ' + str(num) + ' times')
            print('it is od pair ' + str([source, target]))
        
        try:
            #compute the current shortest path's distance
            pathlength = nx.single_source_dijkstra(G=G2, source=source, target=target, weight=weight)
            new_len = pathlength[0][target]
            current_len = new_len
            
            #if it is still lower than the threshold, do the algorithm
            if new_len < max_len:
                n_sp += 1
                
                #record the new shortest path
                new_sp = pathlength[1][target]
                tup = []
                for i in arange(0, len(new_sp)-1, 1):
                    new_path = tuple([new_sp[i], new_sp[i+1]])
                    if new_path in edgelist:
                        sp_list.append(new_path)
                        #TODO: make sure that the links where node source/target resides are not removed
                        tup.append(new_path)
                    else:
                        new_path = tuple([new_sp[i+1], new_sp[i]])
                        sp_list.append(new_path)
                        tup.append(new_path)
                
                #remove all paths in the new shortest path
                G2.remove_edges_from(tup)
#                 for path in tup:
#                     G2.remove_edge(path[0], path[1])    
                
        #if there is no more path available between the source and the target, break the loop        
        except:
            break
            
    sp_list = list(set(sp_list))
    
    return sp_list, n_sp
	
def _distinct_path_all_pairs(G, centroids, weight, cutoff=3):
    ksp = {}
    centroid_nodes2 = copy.deepcopy(centroids)
    for node in centroids:
        centroid_nodes2.remove(node)
        for node2 in centroid_nodes2:
            sp_list, n_sp = _find_distinct_sp(G=G, source=node, target=node2, weight=weight, cutoff=cutoff)
            ksp.update({(node, node2):[sp_list, n_sp]})
            if len(ksp)%50==0:
                print(str(len(ksp)) + ' od pairs has been calculated')
                
    return ksp

# Metric M5_01

def metric_m5_01(gdf, line, cutoff):
    
    gdf2 = gdf.copy()
    
    area = line.buffer(cutoff)
    
    gdf2 = gdf2.loc[gdf2['geometry'].intersects(area)]
    
    gdf2['endpoint1'] = gdf2.geometry.apply(lambda geom: Point(geom.coords[0]))
    gdf2['endpoint2'] = gdf2.geometry.apply(lambda geom: Point(geom.coords[-1]))
    
    gdf2['geometry'] = gdf2['endpoint1']
    gdf2 = gdf2.loc[gdf2['geometry'].within(area)]
    
    gdf2['geometry'] = gdf2['endpoint2']
    gdf2 = gdf2.loc[gdf2['geometry'].within(area)]
    
    redundancy = len(gdf2) - 3
    
    if redundancy < 0:
        redundancy = 0
    
    return redundancy

# Correlation coefficient analysis for metrics comparison

def correlate_metrics_pearson(df, m_a, m_b):
    df2 = df[df[m_a] + df[m_b] != 0]
    df2 = df.copy()
    r, p = pearsonr(df2[m_a], df2[m_b])

    return r, p, len(df2)

def correlate_metrics_spearman(df, m_a, m_b):

    df2 = df[df[m_a] + df[m_b] != 0]
    df2 = df.copy()
    r, p = spearmanr(df2[m_a], df2[m_b])

    return r, p, len(df2)

def correlate_metrics_ks(df, m_a, m_b):

    df2 = df[df[m_a] + df[m_b] != 0]
    df2 = df.copy()
    D, p = ks_2samp(df2[m_a], df2[m_b])

    return D, p

# Functions for metrics robustness analysis

def rank_robustness(df, all_metrics):
    #create dataframe per replication
    all_df = []
    for i, num in enumerate(list(set(list(df['rep'])))):
        exec("rep_{}_df = df.loc[df['rep']==i]".format(i))
        exec("all_df.append(rep_{}_df)".format(i))

    #calculate Spearman rank correlation coefficient between each replication
    spearman_names = []
    spearman_dfs = []
    for metric in all_metrics:
        c = metric
        exec("{}_spearman_df = _spearman_ks_all_rep(all_df, rep_0_df, metric)".format(metric))
        exec("spearman_dfs.append({}_spearman_df)".format(metric))
        spearman_names.append(metric)
        print("{}_spearman_df has been created".format(metric))

    spearmans_dict = dict(zip(spearman_names, spearman_dfs))
    return spearmans_dict

def dist_robustness(df, all_metrics):
    #create dataframe per replication
    all_df = []
    for i, num in enumerate(list(set(list(df['rep'])))):
        exec("rep_{}_df = df.loc[df['rep']==i]".format(i))
        exec("all_df.append(rep_{}_df)".format(i))

    #calculate Kolmogorov-Smirnov distance between each replication
    ks_names = []
    ks_dfs = []
    for metric in all_metrics:
        c = metric
        exec("{}_ks_df = _spearman_ks_all_rep(all_df, rep_0_df, metric, type='ks')".format(metric))
        exec("ks_dfs.append({}_ks_df)".format(metric))
        ks_names.append(metric)
        print("{}_ks_df has been created".format(metric))

    ks_dict = dict(zip(ks_names, ks_dfs))
    return ks_dict

def _spearman_ks_all_rep(all_df, rep_0_df, metric, type='Spearman'):
    # create dataframe of spearman rank correlation coefficient
    n = 100
    top_link = []
    rep_0_new_df = rep_0_df.loc[rep_0_df[metric]!=0]
    top_link.extend(list(rep_0_new_df.sort_values(metric, ascending=False).osmid[:n]))

    r_df = pd.DataFrame(np.nan, index=list(np.arange(0,len(all_df),1)), columns=list(np.arange(0,len(all_df),1)))

    all_rep = list(r_df.columns)
    all_rep2 = copy.deepcopy(all_rep)

    for i in all_rep:
        current_df1 = all_df[i][[metric,'osmid']]
        current_df1 = current_df1.loc[current_df1['osmid'].isin(top_link)]
        current_df1 = current_df1.sort_values('osmid')
        for j in all_rep2:
            current_df2 = all_df[j][[metric,'osmid']]
            current_df2 = current_df2.loc[current_df2['osmid'].isin(top_link)]
            current_df2 = current_df2.sort_values('osmid')
            if type == 'Spearman':
                r, p = spearmanr(current_df1[metric], current_df2[metric])
                r_df.set_value(i, j, r)
            else:
                r, p = ks_2samp(current_df1[metric], current_df2[metric])
                r_df.set_value(i, j, r)
        all_rep2.remove(i)

    r_df = r_df.transpose()
    r_df.fillna(0)

    return r_df

def value_sensitivity(df, all_metrics):
    #TODO: still use quick fix here
    df2 = df.copy()
    df2['m03_02'] = df2['m03_02'].apply(lambda val: 0 if val <= 1 else val-1)
    df2['m01_02'] = df2['m01_02'].apply(lambda val: 0 if val <= 1 else val-1)
    df2['m01_01'] = df2['m01_01'].apply(lambda val: 0 if val <= 1 else val-1)
    df2['m02_01'] = df2['m02_01'].apply(lambda val: 0 if val <= 1 else val-1)
    df2['m06_01'] = df2['m06_01'].apply(lambda val: 0 if val <= 1 else val-1)

    all_df = []
    for i, num in enumerate(list(set(list(df2['rep'])))):
        exec("rep_{}_df = df2.loc[df2['rep']==i]".format(i))
        exec("all_df.append(rep_{}_df)".format(i))

    mad_all_list = []
    std_all_list = []
    for metric in all_metrics:
        mad, stdev = _mad_std_all_rep(all_df, rep_0_df, metric)
        mad_all_list.append(mad)
        std_all_list.append(stdev)

    allvalue_df =  pd.DataFrame({'metric': all_metrics, 'mad': mad_all_list, 'std': std_all_list})
    allvalue_df2 = allvalue_df.sort_values('metric')
    #remove m04_02 since there is supposed to be no robustness analysis for that metric
    allvalue_df2 = allvalue_df2.loc[allvalue_df2['metric']!='m04_02']

    return allvalue_df2


def _mad_std_all_rep(all_df, rep_0_df, metric):
    n = 100
    top_link = []
    rep_0_new_df = rep_0_df.loc[rep_0_df[metric]!=0]
    top_link.extend(list(rep_0_new_df.sort_values(metric, ascending=False).osmid[:n]))

    mad_list = []
    std_list = []
    for link in top_link:
        crit_list = []
        init_val = all_df[0].loc[all_df[0]['osmid']==link][metric].iloc[0]
        for dataset in all_df:
            crit_score = dataset.loc[dataset['osmid']==link][metric].iloc[0]
            crit_list.append(crit_score)
        crit_list = [x / init_val for x in crit_list]
        mad = median_absolute_deviation(crit_list)
        data_std = std(crit_list)
        mad_list.append(mad)
        std_list.append(data_std)

    mad_list = [x if x >= 0 else 0 for x in mad_list]
    std_list = [x if x >= 0 else 0 for x in std_list]

    return mean(mad_list), mean(std_list)

def node_assignment(flow, nodes_gdf):
    '''
    Put results of flow assignment to nodes
    
    Parameters
    ------------
    flow: dict
        Flow dictionary obtained from assignment function (e.g. from aon_assignment or probit_assignment)
    nodes_gdf: GeoDataFrame
        Geodataframe of all nodes in the original network

    Returns
    ------------
    nodes_gdf: GeoDataFrame
        Geodataframe of all nodes in the simplified network, with 'flow' information added
    '''

    #take all points in the simplified graph
    #take all points in the simplified graph
    all_points = set()
    for tup in flow.keys():
        for entry in tup:
            all_points.add(entry)

    #new gdf of points that exist in the simplified graph
    nodes_gdf2 = nodes_gdf.copy()
    nodes_gdf2.loc[:, 'bool'] = nodes_gdf2.loc[:, 'Node'].apply(lambda n: True if n in all_points else False)
    nodes_gdf2 = nodes_gdf2[nodes_gdf2.loc[:,'bool']==True]

    newflow = {}
    for keys, val in flow.items():
        for key in keys:
            try:
                newflow[key] += val
            except:
                newflow[key] = val
    
    def lab(x):
        return newflow[x]
    
    nodes_gdf2.loc[:, 'flow'] = nodes_gdf2.loc[:, 'Node'].apply(lab)
    
    return nodes_gdf2


def node_betweenness_centrality(flow, nodes_gdf, od):
    '''
    Calculate weighted betweenness centrality on nodes level
    
    Parameters
    ------------
    flow: dict
        Flow dictionary obtained from assignment function (e.g. from aon_assignment or probit_assignment)
    nodes_gdf: GeoDataFrame
        Geodataframe of all nodes in the original network
    od: DataFrame
        OD matrix dataframe

    Returns
    ------------
    nodes_gdf: GeoDataFrame
        Geodataframe of all nodes in the simplified network, with 'centrality' and 'flow' information added
    '''    
    #assign flow onto nodes
    nodes_gdf2 = node_assignment(flow, nodes_gdf)
    
    #record the total flow in the network
    totalval = (sum(od.sum())) * 2
    
    #normalize the flow
    nodes_gdf2['centrality'] = nodes_gdf2['flow']/totalval
    
    return nodes_gdf2