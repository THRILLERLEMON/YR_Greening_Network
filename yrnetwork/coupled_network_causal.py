# 连边构建策略：inner 逐条输入；outer逐条输入；self整体输入

from setting import *
from useful_class import *
import os
import time
import multiprocessing
# import igraph
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import tigramite.data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
from cartopy.mpl.patch import geos_to_path
import cartopy.feature as cfeature
from sklearn import preprocessing
from geopy.distance import geodesic, lonlat

# Input Data Var
VARS_LIST = ['LAI', 'soiltemperaturelevel1', 'vpd', 'vs', 'temperature']
# Load data
# Data like
# GA_ID  Time1       Time2       Time3       Time4 ...
# [int]  [double]    [double]    [double]    [double]
# 00000  0.001       0.002       0.67        1.34
# 00001  0.003       0.022       0.69        2.34
# ...    ...         ...         ...         ...
# This data must have the same index with GA_Centroid

if not os.path.exists(BaseConfig.OUT_PATH + 'Coupled_Network'):
    os.mkdir(BaseConfig.OUT_PATH + 'Coupled_Network')

# The time scale of vars
VARS_TIME_SCALE_DICT = {
    'LAI': 'monthly_yearly',
    'soiltemperaturelevel1': 'monthly_yearly',
    'vpd': 'yearly',
    'vs': 'yearly',
    'temperature': 'monthly_yearly'
}

VAR_COLOR_DICT = {
    'LAI': '#3EC700',
    'soiltemperaturelevel1': '#A13500',
    'vpd': '#008080',
    'vs': '#002D70',
    'temperature': '#F9003A'
}


def build_edges_to_csv():
    """
    build edges from csv data
    """
    # Load GeoAgent centroid info
    centroid_data = pd.read_csv(BaseConfig.GEO_AGENT_PATH + 'GA_Centroid.csv')
    # set multiprocessing to get links
    new_pool = multiprocessing.Pool()
    jobs = []
    var_index = 0
    self_links_p = new_pool.apply_async(get_self_links, args=())
    jobs.append(self_links_p)
    print('get_self_links')
    for var in VARS_LIST:
        sec_var_index = 0
        for sec_var in VARS_LIST:
            if var == sec_var:
                p = new_pool.apply_async(get_inner_links, args=(var, ))
                jobs.append(p)
                print('get_inner_links', var)
            else:
                if sec_var_index > var_index:
                    p = new_pool.apply_async(get_outer_links,
                                             args=(var, sec_var))
                    jobs.append(p)
                    print('get_outer_links', var, sec_var)
            sec_var_index = sec_var_index + 1
        var_index = var_index + 1
    new_pool.close()
    new_pool.join()
    # get links from jobs
    kinds_links = []
    for job in jobs:
        kinds_links.append(job.get())
    print('already complete all links over')
    # save all the Links to a csv
    all_links = pd.concat(kinds_links, ignore_index=True)
    print('already concat pd')
    # Add distance
    # all_links_dis = get_geo_distance(all_links, centroid_data)
    all_links_dis = all_links
    all_links_dis.to_csv(BaseConfig.OUT_PATH + 'Coupled_Network\\AllLinks.csv')
    print('already add distance and output')
    print('GOOD!')


def filter_links(p_links):
    """
    filter links by some rules
    @param p_links:links
    @return:
    """
    strength = p_links.loc[:, 'Strength']
    # strength_threshold = strength.describe(percentiles=[0.5]).loc['50%']
    strength_threshold = 0.5
    # select have oriented links
    filtered_links = p_links[(p_links['Unoriented'] == 0) & (
        (p_links['Strength'] > strength_threshold)
        | (p_links['Strength'] < -strength_threshold))].copy()
    return filtered_links


# 弃用，使用igraph的构建方法
# def build_coupled_network_ig():
#     """
#     build coupled network by the edges df from csv by igraph
#     """
#     all_edges_df = pd.read_csv(BaseConfig.OUT_PATH +
#                                'Coupled_Network\\AllLinks.csv')
#     print(all_edges_df)
#     # build a net network
#     coupled_network = igraph.Graph(directed=True)
#     # add every vertex to the net
#     var_sou = all_edges_df['VarSou'].map(str)
#     var_tar = all_edges_df['VarTar'].map(str)
#     id_sou = all_edges_df['Source'].map(str)
#     id_tar = all_edges_df['Target'].map(str)
#     all_edges_df['Source_label'] = id_sou + '_' + var_sou
#     all_edges_df['Target_label'] = id_tar + '_' + var_tar
#     all_ver_list = list(all_edges_df['Source_label']) + list(
#         all_edges_df['Target_label'])
#     # set the unique of the vertexs
#     ver_list_unique = list(set(all_ver_list))
#     for v_id_var in ver_list_unique:
#         coupled_network.add_vertex(
#             v_id_var,
#             var_name=v_id_var.split('_')[1],
#             ga_id=v_id_var.split('_')[0],
#             label=v_id_var.split('_')[0],
#             size=30,
#             color=VAR_COLOR_DICT[v_id_var.split('_')[1]],
#             label_size=15)
#     # set all edges
#     tuples_es = [
#         tuple(x) for x in all_edges_df[['Source_label', 'Target_label']].values
#     ]
#     coupled_network.add_edges(tuples_es)
#     coupled_network.es['VarSou'] = list(all_edges_df['VarSou'])
#     coupled_network.es['VarTar'] = list(all_edges_df['VarTar'])
#     coupled_network.es['width'] = list(abs(all_edges_df['Strength'] * 1))
#     igraph.plot(coupled_network,
#                 BaseConfig.OUT_PATH +
#                 'Coupled_Network//Coupled_Network_ig.pdf',
#                 bbox=(1200, 1200),
#                 layout=coupled_network.layout('large'),
#                 margin=200)


def build_coupled_network():
    """
    build coupled network by the edges df from csv
    """
    all_edges_df = pd.read_csv(BaseConfig.OUT_PATH +
                               'Coupled_Network\\AllLinks.csv')
    filtered_edges_df = filter_links(all_edges_df)
    filtered_edges_df.to_csv(BaseConfig.OUT_PATH +
                             'Coupled_Network\\AllLinksFiltered.csv')
    # build a net network
    coupled_network = nx.MultiDiGraph()
    # add every vertex to the net
    var_sou = filtered_edges_df['VarSou'].map(str)
    var_tar = filtered_edges_df['VarTar'].map(str)
    id_sou = filtered_edges_df['Source'].map(str)
    id_tar = filtered_edges_df['Target'].map(str)
    filtered_edges_df['Source_label'] = id_sou + '_' + var_sou
    filtered_edges_df['Target_label'] = id_tar + '_' + var_tar
    all_ver_list = list(filtered_edges_df['Source_label']) + list(
        filtered_edges_df['Target_label'])
    # set the unique of the vertexs
    ver_list_unique = list(set(all_ver_list))
    for v_id_var in ver_list_unique:
        coupled_network.add_node(v_id_var,
                                 var_name=v_id_var.split('_')[1],
                                 ga_id=v_id_var.split('_')[0],
                                 label=v_id_var.split('_')[0],
                                 size=30,
                                 color=VAR_COLOR_DICT[v_id_var.split('_')[1]],
                                 label_size=15)
    for lIndex, lRow in filtered_edges_df.iterrows():
        thisSou = lRow["Source_label"]
        thisTar = lRow["Target_label"]
        coupled_network.add_edge(thisSou, thisTar, weight=lRow['Strength'])
        # for lf in all_edges_df.columns.values:
        #     coupled_network.edges[thisSou, thisTar][lf] = lRow[lf]
    draw_net_on_map(coupled_network, 'coupled_network')

    # draw all inner and outer net
    var_index = 0
    for var in VARS_LIST:
        sec_var_index = 0
        for sec_var in VARS_LIST:
            if var == sec_var:
                sub_net = get_sub_inner_net(coupled_network, var)
                draw_net_on_map(sub_net, var + '_net')
            else:
                if sec_var_index > var_index:
                    sub_net = get_sub_outer_net(coupled_network, var, sec_var)
                    draw_net_on_map(sub_net, var + '_and_' + sec_var + '_net')
            sec_var_index = sec_var_index + 1
        var_index = var_index + 1

    # export_draw_vars_network(coupled_network)
    # export_draw_agents_network(coupled_network)
    # from networkx.readwrite import json_graph
    # import json
    # data1 = json_graph.node_link_data(coupled_network)
    # filename = BaseConfig.OUT_PATH + 'Coupled_Network\\coupled_network.json'
    # with open(filename, 'w') as file_obj:
    #     json.dump(data1, file_obj)
    # nx.write_gexf(
    #     coupled_network,
    #     BaseConfig.OUT_PATH + 'Coupled_Network\\coupled_network.gexf')


def get_geo_distance(p_links, p_centroid_df):
    """
    get distance on earth
    @param p_links: links
    @param p_centroid_df: centroid info
    @return:
    """
    p_links['Distance'] = None
    for index, row in p_links.iterrows():
        thisSou = row["Source"]
        thisTar = row["Target"]
        souPoi = p_centroid_df[p_centroid_df["GA_ID"] == thisSou].copy()
        tarPoi = p_centroid_df[p_centroid_df["GA_ID"] == thisTar].copy()
        dist = geodesic(
            (souPoi.iloc[0]['latitude'], souPoi.iloc[0]['longitude']),
            (tarPoi.iloc[0]['latitude'], tarPoi.iloc[0]['longitude']))
        p_links.loc[index, 'Distance'] = dist.km
    return p_links


# 统一输入获得inner_links
# def get_inner_links(p_var_name):
#     """
#     get edgeLinks in same var Layer
#     @param p_var_name: var name string
#     @return:
#     """
#     data = None
#     if VARS_TIME_SCALE_DICT[p_var_name] == 'yearly':
#         data = pd.read_csv(BaseConfig.COUPLED_NET_DATA_PATH +
#                            BaseConfig.COUPLED_NET_DATA_HEAD + p_var_name +
#                            '_yearly' + BaseConfig.COUPLED_NET_DATA_TAIL)
#     else:
#         data = pd.read_csv(BaseConfig.COUPLED_NET_DATA_PATH +
#                            BaseConfig.COUPLED_NET_DATA_HEAD + p_var_name +
#                            '_monthly' + BaseConfig.COUPLED_NET_DATA_TAIL)
#     data.fillna(value=BaseConfig.BACKGROUND_VALUE, inplace=True)
#     data_values = data.values
#     id_data = data_values[..., 0].astype(np.int32)
#     var_names = list(map(str, id_data))
#     data_values = np.delete(data_values, 0, axis=1)
#     inner_links = build_link_pcmci_noself(data_values.T, var_names, p_var_name,
#                                           p_var_name)
#     return inner_links


# 逐条输入获得inner_links
def get_inner_links(p_var_name):
    """
    get edgeLinks in same var Layer
    @param p_var_name: var name string
    @return:
    """
    data = None
    if VARS_TIME_SCALE_DICT[p_var_name] == 'yearly':
        data = pd.read_csv(BaseConfig.COUPLED_NET_DATA_PATH +
                           BaseConfig.COUPLED_NET_DATA_HEAD + p_var_name +
                           '_yearly' + BaseConfig.COUPLED_NET_DATA_TAIL)
    else:
        data = pd.read_csv(BaseConfig.COUPLED_NET_DATA_PATH +
                           BaseConfig.COUPLED_NET_DATA_HEAD + p_var_name +
                           '_monthly' + BaseConfig.COUPLED_NET_DATA_TAIL)
    data.fillna(value=BaseConfig.BACKGROUND_VALUE, inplace=True)
    data_values = data.values
    id_data = data_values[..., 0].astype(np.int32)
    data_values = np.delete(data_values, 0, axis=1)
    [agent_num, times_num] = data_values.shape
    one_links = []
    for i_sou in np.arange(0, agent_num):
        this_sou = data_values[i_sou, ...]
        for i_tar in np.arange(0, agent_num - 1):
            data_2_row = np.array([this_sou, data_values[i_tar, ...]])
            one_link = build_link_pcmci_noself(
                data_2_row.T, [id_data[i_sou], id_data[i_tar]], p_var_name,
                p_var_name)
            one_links.append(one_link)
    inner_links = pd.concat(one_links, ignore_index=True)
    return inner_links


def get_outer_links(p_var_sou, p_var_tar):
    """
    get edgeLinks in 2 diferent var Layer
    @param p_var_sou: source var name
    @param p_var_tar: target var name
    @return:
    """
    data_sou = None
    data_tar = None
    # monthly data
    if VARS_TIME_SCALE_DICT[p_var_sou] == VARS_TIME_SCALE_DICT[
            p_var_tar] == 'monthly_yearly':
        data_sou = pd.read_csv(BaseConfig.COUPLED_NET_DATA_PATH +
                               BaseConfig.COUPLED_NET_DATA_HEAD + p_var_sou +
                               '_monthly' + BaseConfig.COUPLED_NET_DATA_TAIL)
        data_sou.fillna(value=BaseConfig.BACKGROUND_VALUE, inplace=True)
        data_tar = pd.read_csv(BaseConfig.COUPLED_NET_DATA_PATH +
                               BaseConfig.COUPLED_NET_DATA_HEAD + p_var_tar +
                               '_monthly' + BaseConfig.COUPLED_NET_DATA_TAIL)
        data_tar.fillna(value=BaseConfig.BACKGROUND_VALUE, inplace=True)
        sou_times = list(data_sou.columns.values)
        tar_times = list(data_tar.columns.values)
        # out put the same columns
        same_times = []
        for i_sou_time in sou_times:
            for i_tar_time in tar_times:
                if i_sou_time == i_tar_time:
                    same_times.append(i_sou_time)
        # update the data to same scale and length
        data_sou = data_sou[same_times]
        data_tar = data_tar[same_times]
    # yearly data
    else:
        data_sou = pd.read_csv(BaseConfig.COUPLED_NET_DATA_PATH +
                               BaseConfig.COUPLED_NET_DATA_HEAD + p_var_sou +
                               '_yearly' + BaseConfig.COUPLED_NET_DATA_TAIL)
        data_sou.fillna(value=BaseConfig.BACKGROUND_VALUE)
        data_tar = pd.read_csv(BaseConfig.COUPLED_NET_DATA_PATH +
                               BaseConfig.COUPLED_NET_DATA_HEAD + p_var_tar +
                               '_yearly' + BaseConfig.COUPLED_NET_DATA_TAIL)
        data_tar.fillna(value=BaseConfig.BACKGROUND_VALUE, inplace=True)
        sou_times = list(data_sou.columns.values)
        tar_times = list(data_tar.columns.values)
        # out put the same columns
        same_times = []
        for i_sou_time in sou_times:
            for i_tar_time in tar_times:
                if i_sou_time == i_tar_time:
                    same_times.append(i_sou_time)
        # update the data to same scale and length
        data_sou = data_sou[same_times]
        data_tar = data_tar[same_times]

    data_sou_values = data_sou.values
    data_tar_values = data_tar.values
    id_sou = data_sou_values[..., 0].astype(np.int32)
    id_tar = data_tar_values[..., 0].astype(np.int32)
    sou_values = np.delete(data_sou_values, 0, axis=1)
    tar_values = np.delete(data_tar_values, 0, axis=1)
    sou_values_nor = sou_values
    tar_values_nor = tar_values
    # sou_values_nor = z_score_normalization(sou_values)
    # tar_values_nor = z_score_normalization(tar_values)
    [agent_num, times_num] = sou_values.shape
    one_links = []
    for i_sou in np.arange(0, agent_num):
        this_sou = sou_values_nor[i_sou, ...]
        for i_tar in np.arange(0, agent_num - 1):
            data_2_row = np.array([this_sou, tar_values_nor[i_tar, ...]])
            one_link = build_link_pcmci_noself(data_2_row.T,
                                               [id_sou[i_sou], id_tar[i_tar]],
                                               p_var_sou, p_var_tar)
            one_links.append(one_link)
    outer_links = pd.concat(one_links, ignore_index=True)
    return outer_links


def get_self_links():
    """
    get self links for every GeoAgent
    """
    # out put the same columns
    # make a fake list include all times
    fake_months = []
    for iyear in np.arange(1955, 2030):
        for imonth in np.arange(1, 13):
            fake_months.append(str(iyear) + str(imonth).zfill(2))
    month_times_intersection = list(fake_months)
    year_times_intersection = list(map(str, np.arange(1955, 2030)))
    for var in VARS_LIST:
        # monthly data
        if VARS_TIME_SCALE_DICT[var] != 'yearly':
            var_data = pd.read_csv(BaseConfig.COUPLED_NET_DATA_PATH +
                                   BaseConfig.COUPLED_NET_DATA_HEAD + var +
                                   '_monthly' +
                                   BaseConfig.COUPLED_NET_DATA_TAIL,
                                   index_col='GA_ID')
            month_times_intersection = set(
                month_times_intersection).intersection(
                    set(list(var_data.columns.values)))
        # yearly data
        var_data = pd.read_csv(BaseConfig.COUPLED_NET_DATA_PATH +
                               BaseConfig.COUPLED_NET_DATA_HEAD + var +
                               '_yearly' + BaseConfig.COUPLED_NET_DATA_TAIL,
                               index_col='GA_ID')
        year_times_intersection = set(year_times_intersection).intersection(
            set(list(var_data.columns.values)))
    # sord them
    month_times_intersection = sorted(list(month_times_intersection))
    year_times_intersection = sorted(list(year_times_intersection))

    centroid_data = pd.read_csv(BaseConfig.GEO_AGENT_PATH + 'GA_Centroid.csv')
    agents_links = []
    for agent in list(centroid_data['GA_ID']):
        monthly_vars = []
        yearly_vars = []
        monthly_vars_data = []
        yearly_vars_data = []
        for var in VARS_LIST:
            # monthly data
            if VARS_TIME_SCALE_DICT[var] == 'monthly_yearly':
                var_data_monthly = pd.read_csv(
                    BaseConfig.COUPLED_NET_DATA_PATH +
                    BaseConfig.COUPLED_NET_DATA_HEAD + var + '_monthly' +
                    BaseConfig.COUPLED_NET_DATA_TAIL,
                    index_col='GA_ID')
                var_data_monthly.fillna(value=BaseConfig.BACKGROUND_VALUE,
                                        inplace=True)
                var_data_monthly = var_data_monthly[list(
                    month_times_intersection)]
                monthly_vars_data.append(var_data_monthly.loc[agent])
                monthly_vars.append(var)
            # yearly data
            var_data = pd.read_csv(BaseConfig.COUPLED_NET_DATA_PATH +
                                   BaseConfig.COUPLED_NET_DATA_HEAD + var +
                                   '_yearly' +
                                   BaseConfig.COUPLED_NET_DATA_TAIL,
                                   index_col='GA_ID')
            var_data.fillna(value=BaseConfig.BACKGROUND_VALUE, inplace=True)
            var_data = var_data[list(year_times_intersection)]
            yearly_vars_data.append(var_data.loc[agent])
            yearly_vars.append(var)
        monthly_data = np.array(monthly_vars_data)
        # run month data drived links
        month_data_links = build_link_pcmci_noself(monthly_data.T,
                                                   monthly_vars, '--', '--')
        yearly_data = np.array(yearly_vars_data)
        # run year data drived links
        year_data_links = build_link_pcmci_noself(yearly_data.T, yearly_vars,
                                                  '--', '--')
        # move the yearly var links to the month data network
        only_yearly_vars = list(set(yearly_vars).difference(set(monthly_vars)))
        links_of_yearl_var = year_data_links.loc[
            (year_data_links['Source'].isin(only_yearly_vars))
            | (year_data_links['Target'].isin(only_yearly_vars))]
        agent_links = pd.concat([month_data_links, links_of_yearl_var],
                                ignore_index=True)
        # change the format of agent_links
        agent_links['VarSou'] = agent_links['Source']
        agent_links['VarTar'] = agent_links['Target']
        agent_links['Source'] = agent
        agent_links['Target'] = agent
        agents_links.append(agent_links)
    self_links = pd.concat(agents_links, ignore_index=True)
    return self_links


def build_link_pcmci_noself(p_data_values, p_agent_names, p_var_sou,
                            p_var_tar):
    """
    build links by n column data
    """
    [times_num, agent_num] = p_data_values.shape
    # set the data for PCMCI
    data_frame = pp.DataFrame(p_data_values,
                              var_names=p_agent_names,
                              missing_flag=BaseConfig.BACKGROUND_VALUE)
    # new PCMCI
    pcmci = PCMCI(dataframe=data_frame, cond_ind_test=ParCorr())
    # run PCMCI
    alpha_level = 0.01
    results_pcmci = pcmci.run_pcmciplus(tau_min=0,
                                        tau_max=2,
                                        pc_alpha=alpha_level)
    # get the result
    graph_pcmci = results_pcmci['graph']
    q_matrix = results_pcmci['q_matrix']
    p_matrix = results_pcmci['p_matrix']
    val_matrix = results_pcmci['val_matrix']
    conf_matrix = results_pcmci['conf_matrix']
    ambiguous_triples = results_pcmci['ambiguous_triples']
    # filter these links
    links_df = pd.DataFrame(columns=('VarSou', 'VarTar', 'Source', 'Target',
                                     'TimeLag', 'Strength', 'Unoriented'))
    if graph_pcmci is not None:
        sig_links = (graph_pcmci != "") * (graph_pcmci != "<--")
    elif q_matrix is not None:
        sig_links = (q_matrix <= alpha_level)
    else:
        sig_links = (p_matrix <= alpha_level)
    for j in range(agent_num):
        links = {(p[0], -p[1]): np.abs(val_matrix[p[0], j, abs(p[1])])
                 for p in zip(*np.where(sig_links[:, j, :]))}
        # Sort by value
        sorted_links = sorted(links, key=links.get, reverse=True)
        for p in sorted_links:
            VarSou = p_var_sou
            VarTar = p_var_tar
            Source = p_agent_names[j]
            Target = p_agent_names[p[0]]
            TimeLag = p[1]
            Strength = val_matrix[p[0], j, abs(p[1])]
            Unoriented = None
            if graph_pcmci is not None:
                if p[1] == 0 and graph_pcmci[j, p[0], 0] == "o-o":
                    Unoriented = 1
                    # "unoriented link"
                elif graph_pcmci[p[0], j, abs(p[1])] == "x-x":
                    Unoriented = 1
                    # "unclear orientation due to conflict"
                else:
                    Unoriented = 0
            links_df = links_df.append(pd.DataFrame({
                'VarSou': [VarSou],
                'VarTar': [VarTar],
                'Source': [Source],
                'Target': [Target],
                'TimeLag': [TimeLag],
                'Strength': [Strength],
                'Unoriented': [Unoriented]
            }),
                                       ignore_index=True)
    # remove the self correlation edges
    links_df = links_df.loc[links_df['Source'] != links_df['Target']]
    return links_df


def get_sub_inner_net(p_father_net, p_var_name):
    """
    get subgraph by the var name
    """
    # get nodes
    selected_ns = [
        n for n, d in p_father_net.nodes(data=True)
        if d['var_name'] == p_var_name
    ]
    sub_net = p_father_net.subgraph(selected_ns)
    return sub_net


def get_sub_outer_net(p_father_net, p_var_name_1, p_var_name_2):
    """
    get subgraph by the 2 var name
    """
    # get nodes
    selected_ns = [
        n for n, d in p_father_net.nodes(data=True)
        if d['var_name'] == p_var_name_1 or d['var_name'] == p_var_name_2
    ]
    sub_net = p_father_net.subgraph(selected_ns)
    return sub_net


def remove_inner_net(p_father_net):
    """
    remove the inner net
    """
    del_es = []
    for e in p_father_net.es:
        if e['VarSou'] == e['VarTar']:
            del_es.append(e)
    p_father_net.delete_edges(del_es)
    del_vs = []
    for v in p_father_net.vs:
        if p_father_net.degree(v) == 0:
            del_vs.append(v)
    p_father_net.delete_vertices(del_vs)
    return p_father_net


# 弃用，画出来的图边太多，无可读性
# def export_draw_vars_network(p_coupled_network):
#     """
#     export vars network by quotient_graph
#     """
#     # group of nodes
#     partitions = []
#     for var in VARS_LIST:
#         partitions.append([
#             n for n, d in p_coupled_network.nodes(data=True)
#             if d['var_name'] == var
#         ])
#     block_net = nx.quotient_graph(p_coupled_network, partitions, relabel=False)

#     for n, d in block_net.nodes(data=True):
#         var_label = list(n)[0].split('_')[1]
#         block_net.nodes[n]['label'] = var_label

#     fig = plt.figure(figsize=(10, 10), dpi=300)
#     ax = plt.gca()
#     pos = nx.circular_layout(block_net)
#     for e in block_net.edges:
#         ax.annotate("",
#                     xy=pos[e[0]],
#                     xycoords='data',
#                     xytext=pos[e[1]],
#                     textcoords='data',
#                     arrowprops=dict(arrowstyle="->",
#                                     color=VAR_COLOR_DICT[list(
#                                         e[0])[0].split('_')[1]],
#                                     shrinkA=3,
#                                     shrinkB=3,
#                                     patchA=None,
#                                     patchB=None,
#                                     connectionstyle="arc3,rad=rrr".replace(
#                                         'rrr', str(0.005 * e[2]))))
#     nx.draw_networkx_nodes(block_net,
#                            pos,
#                            node_size=[
#                                block_net.nodes[n]['nedges'] * 30
#                                for n, d in block_net.nodes(data=True)
#                            ],
#                            node_color=[
#                                VAR_COLOR_DICT[list(n)[0].split('_')[1]]
#                                for n, d in block_net.nodes(data=True)
#                            ],
#                            label=[
#                                block_net.nodes[n]['label']
#                                for n, d in block_net.nodes(data=True)
#                            ])
#     nx.draw_networkx_labels(block_net,
#                             pos,
#                             labels={
#                                 n: block_net.nodes[n]['label']
#                                 for n, d in block_net.nodes(data=True)
#                             },
#                             font_size=14,
#                             font_color='#0007DA')
#     plt.savefig(BaseConfig.OUT_PATH + 'Coupled_Network//vars_network.pdf')


def export_draw_vars_network(p_coupled_network):
    """
    export vars network by quotient_graph
    """
    # group of nodes
    partitions = []
    for var in VARS_LIST:
        partitions.append([
            n for n, d in p_coupled_network.nodes(data=True)
            if d['var_name'] == var
        ])
    block_net = nx.quotient_graph(p_coupled_network, partitions, relabel=False)

    for n, d in block_net.nodes(data=True):
        var_label = list(n)[0].split('_')[1]
        block_net.nodes[n]['label'] = var_label

    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = plt.gca()
    pos = nx.circular_layout(block_net)
    for e in block_net.edges:
        ax.annotate("",
                    xy=pos[e[0]],
                    xycoords='data',
                    xytext=pos[e[1]],
                    textcoords='data',
                    arrowprops=dict(arrowstyle="->",
                                    color=VAR_COLOR_DICT[list(
                                        e[0])[0].split('_')[1]],
                                    shrinkA=3,
                                    shrinkB=3,
                                    patchA=None,
                                    patchB=None,
                                    connectionstyle="arc3,rad=rrr".replace(
                                        'rrr', str(0.005 * e[2]))))
    nx.draw_networkx_nodes(block_net,
                           pos,
                           node_size=[
                               block_net.nodes[n]['nedges'] * 30
                               for n, d in block_net.nodes(data=True)
                           ],
                           node_color=[
                               VAR_COLOR_DICT[list(n)[0].split('_')[1]]
                               for n, d in block_net.nodes(data=True)
                           ],
                           label=[
                               block_net.nodes[n]['label']
                               for n, d in block_net.nodes(data=True)
                           ])
    nx.draw_networkx_labels(block_net,
                            pos,
                            labels={
                                n: block_net.nodes[n]['label']
                                for n, d in block_net.nodes(data=True)
                            },
                            font_size=14,
                            font_color='#0007DA')
    plt.savefig(BaseConfig.OUT_PATH + 'Coupled_Network//vars_network.pdf')


def export_draw_agents_network(p_coupled_network):
    """
    export agents network by quotient_graph
    """
    all_edges_df = pd.read_csv(BaseConfig.OUT_PATH +
                               'Coupled_Network\\AllLinks.csv')
    id_sou = all_edges_df['Source'].map(str)
    id_tar = all_edges_df['Target'].map(str)
    all_ver_list = list(id_sou) + list(id_tar)
    # set the unique of the agents
    ver_list_unique = list(set(all_ver_list))
    # group of nodes
    partitions = []

    for a_id in ver_list_unique:
        partitions.append([
            n for n, d in p_coupled_network.nodes(data=True)
            if d['ga_id'] == a_id
        ])
    block_net = nx.quotient_graph(p_coupled_network, partitions, relabel=False)
    name_mapping = {}
    for b_node in block_net.nodes:
        name_mapping[b_node] = list(b_node)[0].split('_')[0]
        block_net.nodes[b_node]['ga_id'] = list(b_node)[0].split('_')[0]
        block_net.nodes[b_node]['color'] = 'r'
        block_net.nodes[b_node]['var_name'] = list(b_node)[0].split('_')[1]
    for e in block_net.edges:
        block_net.edges[e]['weight'] = 1
    nx.relabel_nodes(block_net, name_mapping)
    draw_net_on_map(block_net, 'Agents_Net')


def draw_net_on_map(p_network, p_net_name):
    """
    draw network on map on YR
    """
    fig = plt.figure(figsize=(20, 14), dpi=500)
    targetPro = ccrs.PlateCarree()
    ax = fig.add_subplot(1, 1, 1, projection=targetPro)
    # ax.set_global()
    # ax.stock_img()
    ax.add_feature(cfeature.OCEAN.with_scale('50m'))
    ax.add_feature(cfeature.LAND.with_scale('50m'))
    ax.add_feature(cfeature.RIVERS.with_scale('50m'))
    ax.add_feature(cfeature.LAKES.with_scale('50m'))

    ax.set_extent([95, 120, 31.5, 42])

    geoAgentShpPath = BaseConfig.GEO_AGENT_PATH + 'GA_WGS84.shp'

    geoAgentShp = ShapelyFeature(
        Reader(geoAgentShpPath).geometries(), ccrs.PlateCarree())
    ax.add_feature(geoAgentShp,
                   linewidth=0.5,
                   facecolor='None',
                   edgecolor='#4D5459',
                   alpha=0.8)

    centroid_data = pd.read_csv(BaseConfig.GEO_AGENT_PATH + 'GA_Centroid.csv',
                                index_col='GA_ID')
    pos = {}
    for n, d in p_network.nodes(data=True):
        # transform lon & lat
        lon = centroid_data.loc[int(d['ga_id'])]['longitude']
        lat = centroid_data.loc[int(d['ga_id'])]['latitude']
        mx, my = targetPro.transform_point(lon, lat, ccrs.PlateCarree())
        pos[n] = (mx, my)

    nx.draw_networkx_edges(
        p_network,
        pos=pos,
        width=[float(d['weight']) for (u, v, d) in p_network.edges(data=True)],
        alpha=0.5,
        # edge_color='k',
        edge_color=[
            VAR_COLOR_DICT[p_network.nodes[u]['var_name']]
            for (u, v, d) in p_network.edges(data=True)
        ],
        #    edge_color=[
        #        float(d['weight'])
        #        for (u, v, d) in p_network.edges(data=True)
        #    ],
        #    edge_cmap=plt.cm.Purples,
        arrows=True,
        arrowstyle='simple',
        arrowsize=2,
        connectionstyle="arc3,rad=0.05")
    nx.draw_networkx_nodes(
        p_network,
        pos=pos,
        node_size=[p_network.degree(n) * 5 for n in p_network],
        node_color=[d['color'] for n, d in p_network.nodes(data=True)])
    plt.savefig(BaseConfig.OUT_PATH + 'Coupled_Network//' + p_net_name +
                '.pdf')


# Run Code
if __name__ == "__main__":
    print(time.strftime('%H:%M:%S', time.localtime(time.time())))
    # build_edges_to_csv()
    print(time.strftime('%H:%M:%S', time.localtime(time.time())))
    build_coupled_network()
    print(time.strftime('%H:%M:%S', time.localtime(time.time())))
    print('Done! Thriller!')
