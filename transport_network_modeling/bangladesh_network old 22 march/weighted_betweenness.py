from __future__ import division
from matplotlib import pyplot as plt
import matplotlib.colors as colors
from matplotlib.pylab import *

from heapq import heappush, heappop
from itertools import count

import os
import pandas as pd
import numpy as np

import networkx as nx
import geopandas as gp

import ema_workbench

from od_prep import od_aggregation

def edge_betweenness_subset_od(G, sources, targets, weight, od):
    '''
    Input:
        G                : Graph Networkx object
        sources, targets : List of nodes sources IDs and nodes targets IDs (e.g. the centroid nodes)
        weight           : Edge data key corresponding to the edge weight
    Output:
        d                : Dict with edge tuple as keys (e.g. (2,3) ) and betweenness value as values
    '''
    d={}
    for i in range(len(sources)):
        source = sources[i]
        for j in range(len(targets)):
            target = targets[j]
            if source != target :
                sp_dijk_all = nx.dijkstra_path(G, source=source, target=target, weight=weight)
                flow = od[source][target]
                for j in range(len(sp_dijk_all)-1):
                    lst = [sp_dijk_all[j],sp_dijk_all[j+1]]
                    lst = [min(lst), max(lst)]
                    tup = tuple(lst)
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

def betweenness_to_df(gdf,betweenness,betweenness_string):
    '''
    Input:
        gdf                : geodataframe of the original network (should contain LineString information)
        betweenness        : dictionary with edge tuple as keys (e.g. (2,3) ) and betweenness value as values
        betweenness_string : string of betweenness dictionary's object name
    Output:
        gdf_final          : updated gdf with additional column of betweenness centrality
        betweenness_df     : betweenness dictionary transformed into dataframe
    '''
    betweenness_df = pd.DataFrame(betweenness.items(), columns=['FromTo_tuple', betweenness_string])

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

def shortest_path_record(G, sources, targets, weight):
    '''
    Input:
        G                : Graph Networkx object
        sources, targets : List of nodes sources IDs and nodes targets IDs (e.g. the centroid nodes)
        weight           : Edge data key corresponding to the edge weight
    Output:
        d                : Dict with edge tuple as keys (e.g. (2,3) ) and betweenness value as values
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
    for key, val in sp_dict.iteritems():
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
    for key, val in betweenness.iteritems():
        new_key = str(key[0])+str(key[1])
        new_d[new_key] = val

    return new_d

def k_shortest_paths(G, source, target, k=1, weight='weight'):
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
                total_path_length = get_path_length(G_original, root_path, weight) + spur_path_length[target]
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

def get_path_length(G, path, weight='weight'):
    length = 0
    if len(path) > 1:
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]

            length += G.edge[u][v].get(weight, 1)

    return length

def ksp_edge_betweenness_subset_od(G, sources, targets, weight, od, k):
    '''
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