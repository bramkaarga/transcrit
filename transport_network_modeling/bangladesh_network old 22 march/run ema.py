from matplotlib import pyplot as plt
import matplotlib.colors as colors
from matplotlib.pylab import *
import networkx as nx
import pandas as pd
import geopandas as gp
import copy
from rasterstats import zonal_stats
from __future__ import division

import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

#Modules developed by TU Delft team for this project
from bangladesh_network_lib import network_prep as net_p
from bangladesh_network_lib import network_visualization as net_v
from bangladesh_network_lib import od_prep as od_p
from bangladesh_network_lib import weighted_betweenness as betw_w

from matplotlib import pyplot as plt
import matplotlib.colors as colors
from matplotlib.pylab import *
import networkx as nx
import pandas as pd
from __future__ import division

import ema_workbench

import os
filepath = os.getcwd()
network = filepath+'\\Road_BGD_Class1_2_f_connected_v4.shp'
network = filepath+'\\rmms_v11_7_waterway_noZ2.shp'
centroid = filepath+'\\BGD_Districtdata_Citycentroid_level2_v02.shp'


gdf_points, gdf_node_pos, gdf = net_p.prepare_centroids_network(centroid, network)

gdf['penalty'] = gdf.apply(lambda row: 15 if (row['cross']==1 and row['mode']=='road') else 0, axis=1)
gdf['length'] = gdf['length'] + gdf['penalty']

#simplify the graph
G2_new = net_p.gdf_to_simplified_multidigraph(gdf_node_pos, gdf, undirected=True)

#change to simple Graph object type
G2_new_tograph = net_p.multigraph_to_graph(G2_new)

#take the largest components
for g in nx.connected_component_subgraphs(G2_new_tograph):
    if len(list(g.edges())) > 100:
        G3 = g

G2_new_tograph = G3.copy()
#change the simplified transport network back to GeoDataFrame
gdf2 = net_p.graph_to_df(G2_new_tograph)

filepath = os.getcwd()
adm_csv = filepath+'\\District_level_data_v7.csv'
#adm_csv = filepath+'\\District_level_data_v5.csv'
adm_shp = filepath+'\\BGD_adm2.shp'
district_gdf2 = net_p.prepare_adm_background(adm_csv, adm_shp, ['Code', 'Population', 'Population_M', 'Total_export',
                                                               'Jute_mill', 'Flour_mill', 'Tot_Garment_Factory', 'Household',
                                                               'Land_throughput', 'SteelBricks_exp_ton', 'Food_exp_ton',
                                                               'Jutextile_exp_ton', 'Garment_exp_ton', 'Textile_loc_ton',
                                                               'Wheat_loc_ton', 'RawJute_loc_ton', 'Foods_loc_ton',
                                                               'Nonfoods_loc_ton'])

#embed district data to gdf_points

#read district data
district_data = pd.read_csv(adm_csv)

#rename to HASC_2
district_data.rename(columns={'Code':'HASC_2'}, inplace=True)

#merge them
gdf_points = pd.merge(gdf_points,district_data,on='HASC_2',how='outer')

def det_func_basic(distance):
    #distance is a n x n DataFrame of euclidean distance between all centroids
    distance = distance*distance
    distance = 100000/distance
    for i in list(distance.columns):
        for j in list(distance.index.values):
            if distance[i][j] > 9999999:
                distance[i][j] = 0
    return distance

def det_func_prob(distance, beta=0.05):
    #distance is a n x n DataFrame of euclidean distance between all centroids
    haha = pd.DataFrame()
    for i, row in distance.iterrows():
        row2 = row * (-beta)
        exp_row = np.exp(row2)
        sum_exp = exp_row.sum()
        exp_row = 100000 * (exp_row / sum_exp)
        haha = haha.append(exp_row, ignore_index = True)

    haha.values[[np.arange(len(haha))]*2] = 0
    return haha

def det_func_sp(distance, G=G2_new_tograph, gdf_points=gdf_points, weight='length'):
    centroid_list = list(gdf_points['Node'])
    c = []
    for i in arange(len(centroid_list)):
        row_list = []
        for j in arange(len(centroid_list)):
            if i != j:
                dist = nx.dijkstra_path_length(G=G, source=centroid_list[i], target=centroid_list[j], weight=weight)
                row_list.append(dist)
            else:
                row_list.append(0)
        c.append(row_list)


    arr = np.arange(len(centroid_list))
    det_df = pd.DataFrame(c, index=arr, columns=arr)

    return det_df

det_func = {1 : det_func_basic, 2 : det_func_prob, 3 : det_func_sp}

theta = 50 #for m1
beta = 0.5 #for m1, m6
weight='length'
cutoff = 0.05 #for m5_01
m10_buffer = 0.005 #for m10
penalty = 1.2 #for m8_02

centroid_nodes = od_p.prepare_centroids_list(G2_new_tograph)

#export OD
prod_lists1 = ['SteelBricks_exp_ton', 'Food_exp_ton','Jutextile_exp_ton', 'Garment_exp_ton']
attr_driver = 'Total_export'
OD_export_dict = od_p.all_ods_creation(gdf_points = gdf_points, prod_lists = prod_lists1,
                                       attr_driver = attr_driver, dist_deterrence = det_func[1])

prod_lists2 = [ 'Foods_loc_ton', 'Nonfoods_loc_ton']
attr_driver='Population_x'
OD_local_dict1 = od_p.all_ods_creation(gdf_points = gdf_points, prod_lists = prod_lists2,
                                       attr_driver = attr_driver, dist_deterrence = det_func[1])

prod_lists3 = ['RawJute_loc_ton']
attr_driver='Jute_mill'
OD_local_dict2 = od_p.all_ods_creation(gdf_points = gdf_points, prod_lists = prod_lists3,
                                       attr_driver = attr_driver, dist_deterrence = det_func[1])

prod_lists4 = ['Wheat_loc_ton']
attr_driver='Flour_mill'
OD_local_dict3 = od_p.all_ods_creation(gdf_points = gdf_points, prod_lists = prod_lists4,
                                       attr_driver = attr_driver, dist_deterrence = det_func[1])

prod_lists5 = ['Textile_loc_ton']
attr_driver='Tot_Garment_Factory'
OD_local_dict4 = od_p.all_ods_creation(gdf_points = gdf_points, prod_lists = prod_lists5,
                                       attr_driver = attr_driver, dist_deterrence = det_func[1])


#Combine all ODs
OD_local1 = OD_local_dict1[OD_local_dict1.keys()[0]]
for i in range(len(OD_local_dict1)-1):
    OD_local1 = OD_local1 +  OD_local_dict1[OD_local_dict1.keys()[i+1]]

OD_local2 = OD_local_dict2[OD_local_dict2.keys()[0]]

OD_local3 = OD_local_dict3[OD_local_dict3.keys()[0]]

OD_local4 = OD_local_dict4[OD_local_dict4.keys()[0]]

OD_export = OD_export_dict[OD_export_dict.keys()[0]]
for i in range(len(OD_export_dict)-1):
    OD_export = OD_export +  OD_export_dict[OD_export_dict.keys()[i+1]]

OD_all = OD_local1 + OD_local2 + OD_local3 + OD_local4 + OD_export

#create unweighted OD
centroid_district_listed = list(OD_all.columns)

OD_unweighted = pd.DataFrame(1, index=centroid_district_listed, columns=centroid_district_listed)

for i,row in OD_unweighted.iterrows():
    OD_unweighted.loc[i][i] = 0

division_gdf = gp.read_file('C:\\Users\\User\\Desktop\\BramkaFile\\betweenness_ema_thesis\\BGD_adm1.shp')
division_gdf.crs = {'proj': 'longlat', 'ellps': 'WGS84', 'datum': 'WGS84'}
division_gdf['HASC_1'] = division_gdf['HASC_1'].apply(lambda code: code.replace('.', '_'))
division_list = list(set(list(division_gdf['HASC_1'])))
division_gdf['geometry'] = division_gdf.geometry.apply(lambda x: x.buffer(0.01))
def determine_division(geom, division):
    for i, row in division.iterrows():
        if geom.intersects(row['geometry']):
            return row['HASC_1']
gdf2['division'] = gdf2.geometry.apply(lambda geom: determine_division(geom=geom, division=division_gdf))
div_gdf_list = []
for div in division_list:
    exec("{}_gdf = gdf2.loc[gdf2['geometry'].intersects(division_gdf.loc[division_gdf['HASC_1']==div]['geometry'].iloc[0])]".format(div))
    exec("div_gdf_list.append({}_gdf)".format(div))

import pickle

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

ksp = load_obj('ksp_for_m402')

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
    for key, val in sp_length.iteritems():
        try:
            total_sp_length += 1 / (val**beta)
        except:
            pass
        count_node += 1

    #calculate the accessibility
    try:
        a = total_sp_length
    except:
        a = 0

    return a, count_node

def _sum_daily_accessibility(a_dict, a_n_dict):
    '''
    Helper function for function interdiction_m1
    '''
    sum_a = sum(a_dict.values())

    sum_a_n = sum(a_n_dict.values())

    return sum_a, sum_a_n

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

#EMA building block
sp_dict_graph = betw_w.sp_dict_graph_creation(G=G2_new_tograph, sources=centroid_nodes,
                                              targets=centroid_nodes, weight='length')
a_dict_base, a_n_dict_base = _dict_daily_accessibility(centroids=centroid_nodes, G=G2_new_tograph, theta=theta,
                                                                 weight='length', beta=beta)
sum_a_base, sum_a_n_base = _sum_daily_accessibility(a_dict_base, a_n_dict_base)
all_daily_sp_list, all_daily_sp_dict = betw_w._all_daily_sp_record(G=G2_new_tograph, sources=centroid_nodes,
                                                            cutoff=theta, weight='length')

#EMA building block
div_graph_list = []
for div in division_list:
    exec("{}_graph = nx.from_pandas_dataframe(df={}_gdf, source='FNODE_', target='TNODE_',edge_attr='length')".format(div,div))
    exec("div_graph_list.append({}_graph)".format(div))
div_graph_dict = dict(zip(division_list, div_graph_list))
div_init_avrgdist_dict = {}
for key, val in div_graph_dict.iteritems():
    cc = 0
    if nx.number_connected_components(val) > 1:
        for subgraph in nx.connected_component_subgraphs(val):
            if len(subgraph) > cc:
                cc = len(subgraph)
                graph = subgraph
    else:
        graph = val
    average_sp_length = nx.average_shortest_path_length(graph, weight='length')
    div_init_avrgdist_dict.update({key:average_sp_length})

#EMA building block
total_cost_base, od_cost_dict = betw_w._total_cost_sp(G=G2_new_tograph, sources=centroid_nodes, targets=centroid_nodes,
                                              weight=weight, od=OD_all, weighted=False)
total_cost_base_w, od_cost_dict_w = betw_w._total_cost_sp(G=G2_new_tograph, sources=centroid_nodes, targets=centroid_nodes,
                                              weight=weight, od=OD_all, weighted=True)
total_cost_sp_inversed, od_cost_inversed_dict =  betw_w._total_cost_sp_inversed(G=G2_new_tograph, sources=centroid_nodes,
                                                                         targets=centroid_nodes, weight=weight)
efficiency_base = betw_w._network_efficiency_calc(G=G2_new_tograph, total_cost_inversed=total_cost_sp_inversed)

#EMA building block
flow_fromto_df = pd.DataFrame(OD_all.sum(axis=0)+OD_all.sum(axis=1))
flow_fromto_df.columns = ['flow']
a_sum_base, a_master_dict =  betw_w._sum_weighted_accessibility(G=G2_new_tograph, centroids=centroid_nodes,
                                                                flow=flow_fromto_df, weight='length', beta=beta)

#EMA building block
sp_cost_dict = betw_w._shortest_path_cost(G=G2_new_tograph, centroids=centroid_nodes, weight='length')

#EMA building block
flood_file = 'C:\\Users\\User\\Desktop\\BramkaFile\\betweenness_ema_thesis\\fluvial_defended_1in75_tile_1_-9999to0.tif'
gdf3 = gdf2.copy()
gdf3['geometry'] = gdf3.geometry.apply(lambda geom: geom.buffer(m10_buffer))
zonal_stats_dict = zonal_stats(gdf3, flood_file, stats='mean', all_touched=False)

def no_interdiction(G, centroids, od, od_unw, weight, gdf=gdf2, div_graph_dict=div_graph_dict,
                    div_init_avrgdist_dict=div_init_avrgdist_dict, cutoff=cutoff, penalty=penalty,
                    zonal_stats_dict=zonal_stats_dict):

    edgelist = []
    for edge in list(G.edges()):
        edgelist.append(edge)

    #for m3_01
    flow_unweighted = betw_w.aon_assignment(G=G, sources=centroids,
                                            targets=centroids, weight=weight, od=od_unw)
    m3_01_dict = betw_w.edge_betweenness_centrality(flow_unweighted, od_unw)

    #for m3_02
    gdf['m3_02'] = gdf.apply(lambda row: betw_w.interdiction_m3_02(row=row, div_graph_dict=div_graph_dict,
                                                                   div_init_avrgdist_dict=div_init_avrgdist_dict), axis=1)
    gdf['FromToTuple'] = gdf.apply(lambda row: tuple([row['FNODE_'], row['TNODE_']]), axis=1)
    for i in list(gdf['FromToTuple']):
        if not i in m3_01_dict.keys():
            idx = gdf.loc[gdf['FromToTuple']==i].index[0]
            new_tup = tuple([i[1],i[0]])
            gdf['FromToTuple'][idx] = new_tup
    el = list(gdf['FromToTuple'])
    m3_02 = list(gdf['m3_02'])
    m3_02_dict = dict(zip(el, m3_02))

    #for m5_01
    gdf['m5_01'] = gdf.geometry.apply(lambda line: betw_w.m5_01(gdf=gdf, line=line, cutoff=cutoff))
    m5_01 = list(gdf['m5_01'])
    m5_01_dict = dict(zip(el, m5_01))

    #for m8_02
    flow_probit_5 = betw_w.probit_assignment(G=G, sources=centroids, targets=centroids, weight=weight,
                                             od=od, N=4, penalty=penalty)
    m8_02_dict = betw_w.edge_betweenness_centrality(flow_probit_5, od)

    #for m10
    gdf['m10'] = [d.values()[0] for d in zonal_stats_dict]
    gdf['m10'] = gdf.m10.apply(lambda val: 0 if val > 0.8 else val)
    m10 = list(gdf['m10'])
    m10_dict = dict(zip(el, m10))

    #if an edge does not have value yet
    #assign 0 to it
    all_dicts = [m3_01_dict, m3_02_dict, m5_01_dict, m8_02_dict, m10_dict]
    for edge in edgelist:
        for metric in all_dicts:
            if not edge in metric:
                metric.update({edge:0})

    return m3_01_dict, m3_02_dict, m5_01_dict, m8_02_dict, m10_dict

def interdiction(G, centroids, od, weight, theta=theta, beta=beta,
                 a_dict_base=a_dict_base, a_n_dict_base=a_n_dict_base, sum_a_base=sum_a_base, sum_a_n_base=sum_a_n_base,
                 all_daily_sp_list=all_daily_sp_list, all_daily_sp_dict=all_daily_sp_dict, od_cost_dict=od_cost_dict,
                 od_cost_inversed_dict=od_cost_inversed_dict, total_cost_base=total_cost_base,
                 total_cost_sp_inversed=total_cost_sp_inversed, efficiency_base=efficiency_base, ksp=ksp,
                 a_sum_base=a_sum_base, a_master_dict=a_master_dict, flow_df=flow_fromto_df,
                 total_cost_base_w=total_cost_base_w, od_cost_dict_w=od_cost_dict_w, sp_cost_dict=sp_cost_dict):

    ff=0
    m1_01_dict = {}
    m1_02_dict = {}
    m2_01_dict = {}
    m2_02_dict = {}
    m4_02_dict = {}
    m6_01_dict = {}
    m7_01_dict = {}
    m7_02_dict = {}
    m7_03_dict = {}
    m9_01_dict = {}

    #record all paths in all shortest paths, also record initial total distinct shortest paths
    init_n_paths = 0
    for key, val in ksp.iteritems():
        init_n_paths += val[1]

    edgelist = []
    for edge in list(G.edges()):
        edgelist.append(edge)

    for edge in edgelist:

        ff += 1
#         if ff%100 == 0:
#             print(str(ff) + ' edges have been interdicted')

        u = edge[0]
        v = edge[1]
        tup = tuple([u,v])

        #make a copy of the original graph
        G1 = G.copy()

        #remove that edge
        G1.remove_edge(u,v)

        #COPYING DICTIONARY
        #for m1
        a_dict_base2 = a_dict_base.copy()
        a_n_dict_base2 = a_n_dict_base.copy()
        #for m2
        od_cost_dict2 = od_cost_dict.copy()
        od_cost_inversed_dict2 = od_cost_inversed_dict.copy()
        #for m4_02
        new_n_paths = init_n_paths
        #for m6_01
        a_master_dict2 = a_master_dict.copy()
        #for m7_01
        od_cost_dict3 = od_cost_dict_w.copy()
        #for m7_02 and m7_03
        user_exposure_dict = {}
        for centroid in centroids:
            user_exposure_dict.update({centroid:[]})
        sum_exposure = 0
        #for m9_01
        unsatisfied_demand = 0

        #M1
        for key, val in all_daily_sp_dict.iteritems():
            if tup in val:
                a_new, a_n_new = _daily_accessibility(centroid=key, G=G1, theta=theta, weight=weight, beta=beta)
                a_dict_base2.update({key:a_new})
                a_n_dict_base2.update({key:a_n_new})

        #M1 VALUE UPDATE
        sum_a_new, sum_a_n_new = _sum_daily_accessibility(a_dict_base2, a_n_dict_base2)
        try:
            m1_01 = sum_a_base / sum_a_new
            m1_01_dict.update({edge:m1_01})
            m1_02 = sum_a_n_base / sum_a_n_new
            m1_02_dict.update({edge:m1_02})
        except:
            print(sum_a_new, sum_a_n_new, sum_a_base, sum_a_n_base)


        #M2, M6_01, M7_01, M7_02, M7_03, M9_01 ITERATION
        updated_centroid = [] #for m6_01
        for key, val in sp_dict_graph.iteritems():
            if tup in val:
                #for m2, m7_01, m9_01
                try:
                    sp_dijk_distance = nx.dijkstra_path_length(G1, source=key[0], target=key[1], weight=weight)
                    cost = sp_dijk_distance
                    od_cost_dict2[key] = cost
                    od_cost_inversed_dict2[key] = 1 / cost
                    flow = od[key[0]][key[1]]
                    cost2 = sp_dijk_distance * flow
                    od_cost_dict3[key] = cost2
                except:
                    sp_dijk_distance = 9999
                    flow = od[key[0]][key[1]]
                    unsatisfied_demand += flow
                #for m6_01
                try:
                    a_source_old_dict = a_master_dict2[key[0]]
                    a_source_new_dict = a_source_old_dict.copy()
                    a_source_target = betw_w._weighted_accessibility(G=G1, centroid=key[0], targets=key[1],
                                                              flow=flow_df, weight=weight, beta=beta)
                    a_source_new_dict.update({key[1]:a_source_target})
                    a_master_dict2.update({key[0]:a_source_new_dict})
                except:
                    sp_dijk_distance = 9999
                #for m7_02 and m7_03
                exposure = betw_w._user_exposure(G_new=G1, centroid=key[0], target=key[1],
                                          weight=weight, od=od, sp_cost_dict=sp_cost_dict)
                user_exposure_dict[key[0]].append(exposure)

        #M2 VALUE UPDATE
        total_cost_new = sum(od_cost_dict2.values())
        total_cost_inversed_new = sum(od_cost_inversed_dict2.values())
        efficiency_new = betw_w._network_efficiency_calc(G1, total_cost_inversed_new)
        m2_01 = total_cost_new/total_cost_base
        if m2_01 < 0:
            m2_01 = 0
        m2_01_dict.update({edge:m2_01})
        m2_02 = (efficiency_base - efficiency_new)/efficiency_base
        m2_02_dict.update({edge:m2_02})

        #M6_01 VALUE UPDATE
        a_sum_new = sum([sum(x.values()) for x in a_master_dict2.values()])
        m6_01 = a_sum_base/a_sum_new
        if m6_01 < 0:
            m6_01 = 0
        m6_01_dict.update({edge:m6_01})

        #M7_01 AND M9_01 VALUE UPDATE
        total_cost_new = sum(od_cost_dict3.values())
        cost_increase = (total_cost_new - total_cost_base_w)/total_cost_base_w
        unsatisfied_demand = unsatisfied_demand/total_cost_base_w
        if cost_increase < 0:
            cost_increase = 0
        m7_01_dict.update({edge:cost_increase})
        m9_01_dict.update({edge:unsatisfied_demand})

        #M7_02 AND M7_03 VALUE UPDATE
        expected_ue_dict = {}
        worst_ue_dict = {}
        for key, val in user_exposure_dict.iteritems():
            if len(val) > 0:
                average_val = average(val)
                worst_val = max(val)
            else:
                average_val = 0
                worst_val = 0
            expected_ue_dict.update({key:average_val})
            worst_ue_dict.update({key:worst_val})
        m7_02 = sum(expected_ue_dict.values())/len(centroids)
        m7_03 = sum(worst_ue_dict.values())/len(centroids)
        m7_02_dict.update({edge:m7_02})
        m7_03_dict.update({edge:m7_03})

        #M4_02 ITERATION
        for key, val in ksp.iteritems():
            if tup in val[0]:
                new_n_paths -= 1

        #M4_02 VALUE UPDATE
        m4_02 = 1 - (new_n_paths / init_n_paths)
        m4_02_dict.update({edge:m4_02})

    #if an edge does not have value yet
    #assign 0 to it
    all_dicts = [m1_01_dict, m1_02_dict, m2_01_dict, m2_02_dict, m4_02_dict, m6_01_dict, m7_01_dict, m7_02_dict, m7_03_dict, m9_01_dict]
    for edge in edgelist:
        for metric in all_dicts:
            if not edge in metric:
                metric.update({edge:0})

    return m1_01_dict, m1_02_dict, m2_01_dict, m2_02_dict, m4_02_dict, m6_01_dict, m7_01_dict, m7_02_dict, m7_03_dict, m9_01_dict

edgelist = []
edge_dict = {}
for edge in G2_new_tograph.edges():
    key = str(edge[0]) + str(edge[1])
    edgelist.append(edge)
    edge_dict.update({key:[]})

def ema_criticality(det_idx=1, od_exp=1, od_loc1=1, od_loc2=1, od_loc3=1, od_loc4=1, theta=50, beta=0.5, weight='length',
                   cutoff=0.05, m10_buffer=0.005, penalty=1.2):
    # print('here1')
    #replicate the graph and gdf to avoid working on the real file
    G3 = G2_new_tograph.copy()
    gdf3 = gdf2.copy()

#     print('here2')
    #set the edge attribute
    edge_length_change = {}
    for u,v,data in G3.edges(data=True):
        length = data['length'] * np.random.uniform(0.75, 1.5)
        edge_length_change.update({tuple([u,v]):length})

    nx.set_edge_attributes(G3, 'length', edge_length_change)

#     print('here3')
    ##### CREATE OD MATRIX FIRST #####
    centroid_nodes = od_p.prepare_centroids_list(G2_new_tograph)

    #export OD
    prod_lists1 = ['SteelBricks_exp_ton', 'Food_exp_ton','Jutextile_exp_ton', 'Garment_exp_ton']
    attr_driver='Total_export'
    OD_export_dict = od_p.all_ods_creation_ema(gdf_points = gdf_points, prod_lists = prod_lists1,
                                           attr_driver = attr_driver, dist_deterrence = det_func[det_idx])
    for key, val in OD_export_dict.iteritems():
        OD_export_dict[key] = (OD_export_dict[key][0] * od_exp, OD_export_dict[key][1])

#     print('here4')
    prod_lists2 = [ 'Foods_loc_ton', 'Nonfoods_loc_ton']
    attr_driver='Population_x'
    OD_local_dict1 = od_p.all_ods_creation_ema(gdf_points = gdf_points, prod_lists = prod_lists2,
                                           attr_driver = attr_driver, dist_deterrence = det_func[det_idx])
    for key, val in OD_local_dict1.iteritems():
        OD_local_dict1[key] = (OD_local_dict1[key][0] * od_loc1, OD_local_dict1[key][1])

    prod_lists3 = ['RawJute_loc_ton']
    attr_driver='Jute_mill'
    OD_local_dict2 = od_p.all_ods_creation_ema(gdf_points = gdf_points, prod_lists = prod_lists3,
                                           attr_driver = attr_driver, dist_deterrence = det_func[det_idx])
    for key, val in OD_local_dict2.iteritems():
        OD_local_dict2[key] = (OD_local_dict2[key][0] * od_loc2, OD_local_dict2[key][1])

    prod_lists4 = ['Wheat_loc_ton']
    attr_driver='Flour_mill'
    OD_local_dict3 = od_p.all_ods_creation_ema(gdf_points = gdf_points, prod_lists = prod_lists4,
                                           attr_driver = attr_driver, dist_deterrence = det_func[det_idx])
    for key, val in OD_local_dict3.iteritems():
        OD_local_dict3[key] = (OD_local_dict3[key][0] * od_loc3, OD_local_dict3[key][1])

    prod_lists5 = ['Textile_loc_ton']
    attr_driver='Tot_Garment_Factory'
    OD_local_dict4 = od_p.all_ods_creation_ema(gdf_points = gdf_points, prod_lists = prod_lists5,
                                           attr_driver = attr_driver, dist_deterrence = det_func[det_idx])
    for key, val in OD_local_dict4.iteritems():
        OD_local_dict4[key] = (OD_local_dict4[key][0] * od_loc4, OD_local_dict4[key][1])

#     print('here5')
    OD_all_dict = od_p._merge_two_dicts(OD_export_dict, OD_local_dict1)
    OD_all_dict = od_p._merge_two_dicts(OD_all_dict, OD_local_dict2)
    OD_all_dict = od_p._merge_two_dicts(OD_all_dict, OD_local_dict3)
    OD_all_dict = od_p._merge_two_dicts(OD_all_dict, OD_local_dict4)

    prod_lists = prod_lists1+prod_lists2+prod_lists3+prod_lists4+prod_lists5

    factors_dict = od_p.factors_dict_creation(prod_lists)

    #combine all od
    OD_all = od_p.od_aggregation(OD_all_dict, **factors_dict)

    # print('here6')
    ##### THEN PREPARE ALL THE SUPPORTING THINGS #####

    #EMA building block
    sp_dict_graph = betw_w.sp_dict_graph_creation(G=G3, sources=centroid_nodes,
                                                  targets=centroid_nodes, weight='length')
    a_dict_base, a_n_dict_base = _dict_daily_accessibility(centroids=centroid_nodes, G=G3, theta=theta,
                                                                     weight='length', beta=beta)
    sum_a_base, sum_a_n_base = _sum_daily_accessibility(a_dict_base, a_n_dict_base)
    all_daily_sp_list, all_daily_sp_dict = betw_w._all_daily_sp_record(G=G3, sources=centroid_nodes,
                                                                cutoff=theta, weight='length')

#     print('here7')
    #EMA building block
    div_init_avrgdist_dict = {}
    for key, val in div_graph_dict.iteritems():
        cc = 0
        if nx.number_connected_components(val) > 1:
            for subgraph in nx.connected_component_subgraphs(val):
                if len(subgraph) > cc:
                    cc = len(subgraph)
                    graph = subgraph
        else:
            graph = val
        average_sp_length = nx.average_shortest_path_length(graph, weight='length')
        div_init_avrgdist_dict.update({key:average_sp_length})

#     print('here8')
    #EMA building block
    total_cost_base, od_cost_dict = betw_w._total_cost_sp(G=G3, sources=centroid_nodes, targets=centroid_nodes,
                                                  weight=weight, od=OD_all, weighted=False)
    total_cost_base_w, od_cost_dict_w = betw_w._total_cost_sp(G=G3, sources=centroid_nodes, targets=centroid_nodes,
                                                  weight=weight, od=OD_all, weighted=True)
    total_cost_sp_inversed, od_cost_inversed_dict =  betw_w._total_cost_sp_inversed(G=G3, sources=centroid_nodes,
                                                                             targets=centroid_nodes, weight=weight)
    efficiency_base = betw_w._network_efficiency_calc(G=G3, total_cost_inversed=total_cost_sp_inversed)

#     print('here9')
    #EMA building block
    flow_fromto_df = pd.DataFrame(OD_all.sum(axis=0)+OD_all.sum(axis=1))
    flow_fromto_df.columns = ['flow']
    a_sum_base, a_master_dict =  betw_w._sum_weighted_accessibility(G=G3, centroids=centroid_nodes,
                                                                    flow=flow_fromto_df, weight='length', beta=beta)

    #EMA building block
    sp_cost_dict = betw_w._shortest_path_cost(G=G3, centroids=centroid_nodes, weight='length')

#     print('here10')
    #EMA building block
    flood_file = 'C:\\Users\\User\\Desktop\\BramkaFile\\betweenness_ema_thesis\\fluvial_defended_1in75_tile_1_-9999to0.tif'
    gdf4 = gdf3.copy()
    gdf4['geometry'] = gdf4.geometry.apply(lambda geom: geom.buffer(m10_buffer))
    zonal_stats_dict = zonal_stats(gdf4, flood_file, stats='mean', all_touched=False)

    # print('here11')
    #### THEN CALCULATE THE METRICS ####
    m1_01_dict, m1_02_dict, m2_01_dict, m2_02_dict, m4_02_dict, m6_01_dict, m7_01_dict, m7_02_dict, m7_03_dict, m9_01_dict = interdiction(G=G3, centroids=centroid_nodes, od=OD_all,
                                                                                                                                          weight='length', theta=theta, beta=beta,
                                                                                                                                          a_dict_base=a_dict_base, a_n_dict_base=a_n_dict_base,
                                                                                                                                          sum_a_base=sum_a_base, sum_a_n_base=sum_a_n_base,
                                                                                                                                          all_daily_sp_list=all_daily_sp_list,
                                                                                                                                          all_daily_sp_dict=all_daily_sp_dict,
                                                                                                                                          od_cost_dict=od_cost_dict,
                                                                                                                                          od_cost_inversed_dict=od_cost_inversed_dict,
                                                                                                                                          total_cost_base=total_cost_base,
                                                                                                                                          total_cost_sp_inversed=total_cost_sp_inversed,
                                                                                                                                          efficiency_base=efficiency_base, ksp=ksp,
                                                                                                                                          a_sum_base=a_sum_base, a_master_dict=a_master_dict,
                                                                                                                                          flow_df=flow_fromto_df, total_cost_base_w=total_cost_base_w,
                                                                                                                                          od_cost_dict_w=od_cost_dict_w, sp_cost_dict=sp_cost_dict)

    m3_01_dict, m3_02_dict, m5_01_dict, m8_02_dict, m10_dict = no_interdiction(G=G3, centroids=centroid_nodes,
                                                                               od=OD_all, od_unw=OD_unweighted, weight='length',
                                                                               gdf=gdf3, div_graph_dict=div_graph_dict, div_init_avrgdist_dict=div_init_avrgdist_dict,
                                                                               cutoff=cutoff, penalty=penalty, zonal_stats_dict=zonal_stats_dict)
    # print('here12')


    ##### COMBINE ALL DICTIONARIES ####
    all_dicts = [m3_01_dict, m3_02_dict, m5_01_dict, m8_02_dict, m10_dict, m1_01_dict, m1_02_dict, m2_01_dict, m2_02_dict,
                 m4_02_dict, m6_01_dict, m7_01_dict, m7_02_dict, m7_03_dict, m9_01_dict]
    d = {}
    for k in m3_01_dict.iterkeys():
        d[k] = tuple(d[k] for d in all_dicts)

    edge_dict = {}
    for key,val in d.iteritems():
        keyn = str(key[0]) + str(key[1])
        edge_dict.update({keyn:val})

    return edge_dict

from ema_workbench import (Model, RealParameter, ScalarOutcome, Constant, IntegerParameter)

#instantiate the model
criticality_model = Model('criticality', function = ema_criticality)

#specify uncertainties
criticality_model.uncertainties = [IntegerParameter('det_idx', 1, 3),
                            RealParameter('od_exp', 0.75, 1.5),
                            RealParameter('od_loc1', 0.75, 1.5),
                            RealParameter('od_loc2', 0.75, 1.5),
                            RealParameter('od_loc3', 0.75, 1.5),
                            RealParameter('od_loc4', 0.75, 1.5),
                            RealParameter('theta', 30, 70),
                            RealParameter('beta', 0.25, 0.75),
                            RealParameter('cutoff', 0.025, 0.075),
                            RealParameter('m10_buffer', 0.0025, 0.0075),
                            RealParameter('penalty', 1, 1.5)]

#specify outcomes
criticality_model.outcomes = [ScalarOutcome(key) for key,val in edge_dict.iteritems()]

from ema_workbench import Policy, perform_experiments
from ema_workbench import ema_logging

ema_logging.log_to_stderr(ema_logging.INFO)

n_scenarios = 3
results = perform_experiments(criticality_model, n_scenarios)

fh =  r'./data/{} experiments_22may_v01.tar.gz'.format(nr_experiments)
save_results(results, fh)