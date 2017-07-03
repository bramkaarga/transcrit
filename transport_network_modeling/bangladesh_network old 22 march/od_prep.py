from __future__ import division
import os
import pandas as pd
import numpy as np

import networkx as nx
from networkx.algorithms.centrality import betweenness as bt
import geopandas as gp

from math import radians, cos, sin, asin, sqrt
from shapely.geometry import LineString, Point

def prepare_centroids_list(G2_new_tograph):
    '''
    Input:
        G2_new_tograph : Graph Networkx object
    Output:
        centroid_nodes : List of all centroid nodes ID
    '''
    #create list of nodes of centroids
    G = G2_new_tograph.copy()
    SG=G.subgraph( [n[0] for n in G.node.items() if n[1]['IsCentroid'] == 1 ] )
    SG.nodes(data=True)

    centroid_nodes = list(SG.nodes())
    return centroid_nodes

#extract the longitude and latitude from geometry of the shapefile
def getXY(pt):
    return (pt.x, pt.y)

#function to calculate linear distance between two points given lon and lat of each point
#http://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    km = 6367 * c
    return km

#calculate distance centroid
def calc_distance_centroid(centroid_gdf):
    #calculate longitude and latitude of points based on shapefile's geometry attribute
    lon,lat = [list(t) for t in zip(*map(getXY, centroid_gdf['geometry']))]

    #make an arbitrary dataframe to store distance information
    distance = pd.DataFrame({'initiate' : []})
    #calculate the distance between each OD pair
    for i in range(len(lon)):
        d = []
        for j in range(len(lat)):
            d.append(haversine(lon[i], lat[i], lon[j], lat[j]))
        distance[i] = d

    distance.drop(distance.columns[0], axis=1, inplace=True)
    return distance

#generating production and attraction of each district
def gen_prod_attr(district_stats, prod_driver, attr_driver='Population_x'):
    #GENERATING TRIP PRODUCTION
    #assuming one trip consists of 10 tons
    district_stats['trips_production'] = district_stats[prod_driver]
    production = district_stats['trips_production']

    #GENERATING TRIP ATTRACTION
    #first calculate relative attraction of each district
    district_stats[attr_driver] = district_stats[attr_driver].fillna(district_stats[attr_driver].mean())
    relative_attr = district_stats[attr_driver] / district_stats[attr_driver].sum()

    #then distribute the production over the relative attraction
    attraction = relative_attr*production.sum()

    return production, attraction

#calculate OD matrix for a given production driver
#code obtained from https://github.com/joshchea/python-tdm/blob/master/scripts/CalcDistribution.py
def CalcDoublyConstrained(ProdA, AttrA, F, maxIter = 10):
    '''Calculates doubly constrained trip distribution for a given friction factor matrix
    ProdA = Production array
    AttrA = Attraction array
    F = Friction factor matrix
    maxIter (optional) = maximum iterations, default is 10
    Returns trip table
    '''
    Trips1 = np.zeros((len(ProdA),len(ProdA)))
#     print('Checking production, attraction balancing:')
    sumP = sum(ProdA)
    sumA = sum(AttrA)
    if sumP != sumA:
        AttrA = AttrA*(sumP/sumA)
        AttrT = AttrA.copy()
        ProdT = ProdA.copy()
    else:
        AttrT = AttrA.copy()
        ProdT = ProdA.copy()

    for balIter in range(0, maxIter):
        for i in list(range(0,len(ProdA))):
            Trips1[i,:] = ProdA[i]*AttrA*F[i]/max(0.000001, sum(AttrA * F[i]))

        #Run 2D balancing --->
        ComputedAttractions = Trips1.sum(0)
        ComputedAttractions[ComputedAttractions==0]=1
        AttrA = AttrA*(AttrT/ComputedAttractions)

        ComputedProductions = Trips1.sum(1)
        ComputedProductions[ComputedProductions==0]=1
        ProdA = ProdA*(ProdT/ComputedProductions)

    for i in list(range(0,len(ProdA))):
        c = ProdA[i]*AttrA*F[i]/max(0.000001, sum(AttrA * F[i]))
        Trips1[i,:] = c

    dfc = pd.DataFrame(Trips1)
    Trips1 = dfc.values.tolist()
    return Trips1

def district_stats_to_OD_df(gdf_points, prod_driver, attr_driver='Population_x'):
    '''
    Input:
        gdf_points        : geodataframe from shapefile (Points) of centroids
        production_driver : string of gdf's column name which will be production driver
    Output:
        OD_matrix         : Dataframe of OD matrix with node id as column and row indices
    '''
    distance = calc_distance_centroid(gdf_points)
    #simple deterrence function in the meantime
    distance = 10000/distance
    for i in list(distance.columns):
        for j in list(distance.index.values):
            if distance[i][j] > 9999999:
                distance[i][j] = 0

    #calcualte production and attraction based on the production driver
    #attraction is automatically based on population
    production, attraction = gen_prod_attr(gdf_points, prod_driver, attr_driver)

    #calculate OD_Matrix
    Trips1 = CalcDoublyConstrained(production, attraction, distance)

    nodelist = list(gdf_points['Node'])

    #rename the index and column into nodelist (based on the gdf_points)
    OD_matrix = pd.DataFrame(Trips1, index=nodelist, columns=nodelist)
    for i, row in OD_matrix.iterrows():
        for j,val in row.iteritems():
            if OD_matrix[i][j] < 0.1:
                OD_matrix[i][j] = 0
    return OD_matrix

def all_ods_creation(gdf_points, prod_lists, attr_driver):

    od_dict={}
    for prod in prod_lists:
        od = district_stats_to_OD_df(gdf_points,  prod_driver=prod, attr_driver=attr_driver)
        od_dict["od_{0}".format(prod)]=od

    return od_dict

def all_ods_creation_ema(gdf_points, prod_lists,attr_driver):

    od_dict={}
    for prod in prod_lists:
        od = district_stats_to_OD_df(gdf_points,  prod_driver=prod, attr_driver=attr_driver)
        od_dict["od_{0}".format(prod)]=(od,prod)

    return od_dict

def merge_two_dicts(x, y):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z

def factors_dict_creation(prod_lists):
    #create arbitrary dictionary
    factors_dict={}

    #create scaling factors (for EMA run later)
    factors_scale= [1] * len(prod_lists)

    #enumerate all items in production lists
    for i,prod in enumerate(prod_lists):
        #create new item in dictionary with factor_00, factor_01, etc as keys
        #and production name (e.g. Textile_exp_ton) as values
        factors_dict[prod]=factors_scale[i]

    return factors_dict

def od_aggregation(OD_all_dict, **factors_dict):
    #create empty dictionary
    OD_final_dict={}

    #iterate over all items in original OD
    for key1,val1 in OD_all_dict.iteritems():

        #matching the production value of the OD dict and the factors_dict
        for key2,val2 in factors_dict.iteritems():
            #if it is a match
            if val1[1] == key2:
                #scale the OD flows of that particular product
                OD_final_dict["od_{0}".format(val1[1])]=val1[0]*val2

    #creation of final OD dataframe
    OD_final_df = OD_final_dict[OD_final_dict.keys()[0]]
    for i in range(len(OD_final_dict)-1):
        OD_final_df = OD_final_df +  OD_final_dict[OD_final_dict.keys()[i+1]]

    return OD_final_df

def od_preparation(prod_lists, OD_all_dict, **factors_dict):
    OD_final_df = od_aggregation(OD_all_dict, **factors_dict)
    return OD_final_df

