from networkx.algorithms.shortest_paths.generic import average_shortest_path_length
from setting import *
from useful_class import *
import os
import time
import multiprocessing
import itertools
# import igraph
import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms import community
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Ellipse
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patheffects as PathEffects
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
VARS_TARGET = ['LAI']
VARS_LIST_NATURE = [
    'Runoff', 'ETVegetation', 'Pressure', 'AirTemperature', 'SoilTemperature',
    'SoilWater', 'VS', 'Precipitation', 'VPD'
]
# VARS_LIST_LUCC = ['LUCD']
VARS_LIST_LUCC = [
    'Forests', 'Shurblands', 'Grasslands', 'Croplands', 'OrchardandTerrace',
    'UrbanandBuiltup', 'WaterBodies', 'DesertandLowvegetatedLands'
]

VARS_LIST = VARS_TARGET + VARS_LIST_NATURE + VARS_LIST_LUCC

# Load data
# Data like
# GA_ID  Time1       Time2       Time3       Time4 ...
# [int]  [double]    [double]    [double]    [double]
# 00000  0.001       0.002       0.67        1.34
# 00001  0.003       0.022       0.69        2.34
# ...    ...         ...         ...         ...
# This data must have the same index with GA_Centroid

if not os.path.exists(BaseConfig.OUT_PATH + 'InnerNetworkCSV'):
    os.mkdir(BaseConfig.OUT_PATH + 'InnerNetworkCSV')
if not os.path.exists(BaseConfig.OUT_PATH + 'SelfNetworkCSV'):
    os.mkdir(BaseConfig.OUT_PATH + 'SelfNetworkCSV')
if not os.path.exists(BaseConfig.OUT_PATH + 'InnerNetCentralityMap'):
    os.mkdir(BaseConfig.OUT_PATH + 'InnerNetCentralityMap')
if not os.path.exists(BaseConfig.OUT_PATH + 'InnerNetCommunitiesMap'):
    os.mkdir(BaseConfig.OUT_PATH + 'InnerNetCommunitiesMap')
if not os.path.exists(BaseConfig.OUT_PATH + 'SelfNetworkFigs'):
    os.mkdir(BaseConfig.OUT_PATH + 'SelfNetworkFigs')

# The time scale of vars
VARS_TIME_SCALE_DICT = {
    'LAI': 'monthly_yearly',
    'AirTemperature': 'monthly_yearly',
    'ETVegetation': 'monthly_yearly',
    'Precipitation': 'monthly_yearly',
    'Pressure': 'monthly_yearly',
    'Runoff': 'monthly_yearly',
    'SoilTemperature': 'monthly_yearly',
    'SoilWater': 'monthly_yearly',
    'VPD': 'monthly_yearly',
    'VS': 'monthly_yearly',
    'Forests': 'yearly',
    'Shurblands': 'yearly',
    'Grasslands': 'yearly',
    'Croplands': 'yearly',
    'OrchardandTerrace': 'yearly',
    'UrbanandBuiltup': 'yearly',
    'WaterBodies': 'yearly',
    'DesertandLowvegetatedLands': 'yearly',
    # 'LUCD': 'yearly'
}

VAR_COLOR_DICT = {
    'LAI': '#3EC700',
    'AirTemperature': '#D62728',
    'ETVegetation': '#9EDAE5',
    'Precipitation': '#17BECF',
    'Pressure': '#A55194',
    'Runoff': '#5254A3',
    'SoilTemperature': '#AD494A',
    'SoilWater': '#9C9EDE',
    'VPD': '#FF9896',
    'VS': '#CCEBC5',
    'Forests': '#2CA02C',
    'Shurblands': '#BCBD22',
    'Grasslands': '#8CA252',
    'Croplands': '#FFD92F',
    'OrchardandTerrace': '#FF7F0E',
    'UrbanandBuiltup': '#8C564B',
    'WaterBodies': '#1F77B4',
    'DesertandLowvegetatedLands': '#969696',
    # 'LUCD': '#FF60ED'
}

VAR_LABEL_DICT = {
    'LAI': 'LAI',
    'AirTemperature': 'Temp',
    'ETVegetation': 'ETv',
    'Precipitation': 'Prcp',
    'Pressure': 'Prse',
    'Runoff': 'Runoff',
    'SoilTemperature': 'SoilTem',
    'SoilWater': 'SoilWater',
    'VPD': 'VPD',
    'VS': 'VS',
    'Forests': 'Forest',
    'Shurblands': 'Shurb',
    'Grasslands': 'Grass',
    'Croplands': 'Crop',
    'OrchardandTerrace': 'OT',
    'UrbanandBuiltup': 'Urban',
    'WaterBodies': 'Water',
    'DesertandLowvegetatedLands': 'LowV',
    # 'LUCD': 'LUCD'
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
    self_links_p = new_pool.apply_async(get_self_links, args=())
    jobs.append(self_links_p)
    print('get_self_links')
    for var in VARS_LIST:
        p = new_pool.apply_async(get_inner_links, args=(var, ))
        jobs.append(p)
        print('get_inner_links', var)
    new_pool.close()
    new_pool.join()
    print('GOOD!')


def show_self_nets(p_target_var):
    self_df = pd.read_csv(BaseConfig.OUT_PATH +
                          'SelfNetworkCSV//SelfNetworkAll.csv')
    filtered_df = self_df[(self_df['Unoriented'] == 0)].copy()
    centroid_data = pd.read_csv(BaseConfig.GEO_AGENT_PATH + 'GA_Centroid.csv')
    agents_weightiest_var_sp = {}
    agents_weightiest_var_ap = {}
    agents_weightiest_var_sp_pos = {}
    agents_weightiest_var_ap_pos = {}
    for agent in list(centroid_data['GA_ID']):
        if agent in list(filtered_df['Source'].unique()):
            agent_self_net = build_net_by_csv(
                filtered_df[(self_df['Source'] == agent)
                            & (self_df['Target'] == agent)].copy())
            draw_self_net_for_agent(agent_self_net, agent)
            agents_weightiest_var_sp[
                agent] = calculate_shortest_path_causal_var_selfnet(
                    agent_self_net, agent, p_target_var)
            agents_weightiest_var_ap[
                agent] = calculate_all_path_causal_var_selfnet(
                    agent_self_net, agent, p_target_var)
            agents_weightiest_var_sp_pos[
                agent] = calculate_shortest_path_causal_var_selfnet_pos(
                    agent_self_net, agent, p_target_var)
            agents_weightiest_var_ap_pos[
                agent] = calculate_all_path_causal_var_selfnet_pos(
                    agent_self_net, agent, p_target_var)
    draw_self_info(agents_weightiest_var_sp,
                   'Agents_weightiest_var_sp_to_' + p_target_var)
    draw_self_info(agents_weightiest_var_ap,
                   'Agents_weightiest_var_ap_to_' + p_target_var)
    draw_self_info(agents_weightiest_var_sp_pos,
                   'Agents_weightiest_var_sp_pos_to_' + p_target_var)
    draw_self_info(agents_weightiest_var_ap_pos,
                   'Agents_weightiest_var_ap_pos_to_' + p_target_var)


def draw_self_info(p_agents_weightiest_var_dict, p_fig_name):
    # draw map
    fig = plt.figure(figsize=(20, 15), dpi=500)
    # [左, 下, 宽, 高] 规定的矩形区域 （全部是0~1之间的数，表示比例）
    rect_ax = [0.1, 0.1, 0.8, 0.78]
    rectCB = [0, 0.75, 1, 0.2]

    geoagent_shp_path = BaseConfig.GEO_AGENT_PATH + 'GA_WGS84.shp'
    targetPro = ccrs.PlateCarree()
    # ax = fig.add_subplot(1, 2, 1, projection=targetPro)
    ax = plt.axes(rect_ax, projection=targetPro)
    # ax.set_global()
    # ax.stock_img()
    ax.add_feature(cfeature.OCEAN.with_scale('50m'))
    ax.add_feature(cfeature.LAND.with_scale('50m'))
    ax.add_feature(cfeature.RIVERS.with_scale('50m'))
    ax.add_feature(cfeature.LAKES.with_scale('50m'))
    ax.set_extent([95, 120, 31.5, 42])

    for feature in Reader(geoagent_shp_path).records():
        ga_id = feature.attributes['GA_ID']
        polygon_geo = ShapelyFeature(feature.geometry, ccrs.PlateCarree())
        hatch_str = ''
        if p_agents_weightiest_var_dict[ga_id] == 'Uncertain':
            face_color = 'k'
            edge_color = 'k'
        else:
            face_color = VAR_COLOR_DICT[str(
                p_agents_weightiest_var_dict[ga_id]).split(':')[0]]
            if str(p_agents_weightiest_var_dict[ga_id]).split(':')[1] == '+':
                edge_color = '#454545'
                hatch_str = '..'
            else:
                edge_color = '#454545'
                hatch_str = '///'
        ax.add_feature(polygon_geo,
                       linewidth=0.4,
                       facecolor=face_color,
                       edgecolor=edge_color,
                       alpha=1,
                       hatch=hatch_str)

    axCB = plt.axes(rectCB)
    axCB.spines['top'].set_visible(False)
    axCB.spines['right'].set_visible(False)
    axCB.spines['bottom'].set_visible(False)
    axCB.spines['left'].set_visible(False)
    axCB.set_xticks([])
    axCB.set_xticks([])
    cMap = ListedColormap(VAR_COLOR_DICT.values())
    cNorm = BoundaryNorm(np.arange(1 + cMap.N), cMap.N)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=cNorm, cmap=cMap),
                      ax=axCB,
                      orientation='horizontal',
                      aspect=34)
    cb.set_ticks(np.arange(0.5, 1.5 + cMap.N))
    cb.set_ticklabels(list(VAR_COLOR_DICT.keys()))
    cb.set_ticklabels(
        list({x: VAR_LABEL_DICT[x]
              for x in list(VAR_COLOR_DICT.keys())}.values()))
    for l in cb.ax.xaxis.get_ticklabels():
        l.set_family('Times New Roman')
        l.set_size(12)

    plt.savefig(BaseConfig.OUT_PATH + 'SelfNetworkFigs//' + p_fig_name +
                '.pdf')


def draw_self_net_for_agent(p_self_net, p_agent_name):
    """
    draw self net for a agent
    """
    pos_old = nx.circular_layout(p_self_net)
    pos_new_keys = []
    for var in VARS_LIST:
        if str(p_agent_name) + '_' + var in pos_old.keys():
            pos_new_keys.append(str(p_agent_name) + '_' + var)
    pos = dict(map(lambda x, y: [x, y], pos_new_keys, pos_old.values()))
    fig = plt.figure(figsize=(5, 5), dpi=300)
    # ax = fig.add_subplot(111, frame_on=False)
    ax = plt.gca()
    x_values, y_values = zip(*pos.values())
    x_max = max(x_values)
    x_min = min(x_values)
    x_margin = (x_max - x_min) * 0.25
    plt.xlim(x_min - x_margin, x_max + x_margin)
    y_max = max(y_values)
    y_min = min(y_values)
    y_margin = (y_max - y_min) * 0.25
    plt.ylim(y_min - y_margin, y_max + y_margin)

    for n, d in p_self_net.nodes(data=True):
        var_name = str(n).split('_')[1]
        var_label = VAR_LABEL_DICT[var_name]
        p_self_net.nodes[n]['label'] = var_label
        color_edge = VAR_COLOR_DICT[var_name]
        if var_name in VARS_LIST_LUCC:
            color_edge = '#454545'
        c = Ellipse(
            pos[n],
            width=0.2,
            height=0.2,
            clip_on=False,
            facecolor=VAR_COLOR_DICT[var_name],
            edgecolor=color_edge,
            zorder=0,
        )
        ax.add_patch(c)
        p_self_net.nodes[n]["patch"] = c

    cm = plt.cm.RdBu_r
    cNorm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
    for (u, v, d) in p_self_net.edges(data=True):
        rad = str(0)
        width = 2.5
        zorder_value = -1
        if d['timelag'] == 1:
            rad = str(-0.2)
            width = 2
            zorder_value = -2
        elif d['timelag'] >= 2:
            rad = str(-0.3)
            width = 1.5
            zorder_value = -3
        e_p = FancyArrowPatch(
            pos[u],
            pos[v],
            arrowstyle='->,head_length=0.4, head_width=0.2',
            connectionstyle='arc3,rad=' + rad,
            mutation_scale=10,
            lw=width,
            alpha=0.9,
            linestyle='-',
            color=cm(cNorm(d['weight'])),
            clip_on=False,
            patchA=p_self_net.nodes[u]["patch"],
            patchB=p_self_net.nodes[v]["patch"],
            shrinkA=0,
            shrinkB=0,
            zorder=zorder_value,
        )
        ax.add_artist(e_p)

        # Attach labels of lags
        if d['timelag'] != 0:
            trans = None
            path = e_p.get_path()
            verts = path.to_polygons(trans)[0]
            if len(verts) > 2:
                label_vert = verts[1, :]
                string = str(d['timelag'])
                txt = ax.text(
                    label_vert[0],
                    label_vert[1],
                    string,
                    fontsize=4,
                    verticalalignment="center",
                    horizontalalignment="center",
                    color="w",
                    alpha=0.8,
                    zorder=2,
                )
                txt.set_path_effects(
                    [PathEffects.withStroke(linewidth=0.5, foreground="k")])

    # nx.draw_networkx_edges(
    #     p_self_net,
    #     pos=pos,
    #     width=4,
    #     # width=[float(d['weight']) for (u, v, d) in p_self_net.edges(data=True)],
    #     alpha=0.8,
    #     style='solid',
    #     # edge_color='k',
    #     # edge_color=[
    #     #     VAR_COLOR_DICT[p_self_net.nodes[u]['var_name']]
    #     #     for (u, v, d) in p_self_net.edges(data=True)
    #     # ],
    #     edge_color=[
    #         float(d['weight']) for (u, v, d) in p_self_net.edges(data=True)
    #     ],
    #     edge_cmap=plt.cm.RdBu_r,
    #     edge_vmin=-1,
    #     edge_vmax=1,
    #     arrows=True,
    #     arrowstyle='->',
    #     arrowsize=4,
    #     connectionstyle="arc3,rad=0.2")

    # nx.draw_networkx_nodes(p_self_net,
    #                        pos,
    #                        node_size=600,
    #                        node_color=[
    #                            VAR_COLOR_DICT[str(n).split('_')[1]]
    #                            for n, d in p_self_net.nodes(data=True)
    #                        ],
    #                        label=[
    #                            p_self_net.nodes[n]['label']
    #                            for n, d in p_self_net.nodes(data=True)
    #                        ])

    nx.draw_networkx_labels(p_self_net,
                            pos,
                            labels={
                                n: p_self_net.nodes[n]['label']
                                for n, d in p_self_net.nodes(data=True)
                            },
                            font_size=5,
                            font_color='k')
    plt.savefig(BaseConfig.OUT_PATH + 'SelfNetworkFigs//' + str(p_agent_name) +
                '_self_network.pdf',
                bbox_inches='tight')


def calculate_shortest_path_causal_var_selfnet(p_net, p_agent, p_target_var):
    if not str(p_agent) + '_' + p_target_var in p_net.nodes():
        return 'Uncertain'
    short_paths = nx.single_target_shortest_path(
        p_net,
        str(p_agent) + '_' + p_target_var)
    del short_paths[str(p_agent) + '_' + p_target_var]
    if short_paths == {}:
        return 'Uncertain'
    causal_strengths = {}
    for source_n in short_paths:
        causal_strengths[source_n] = calculate_sum_weight_of_path(
            p_net, short_paths[source_n])
    causal_strengths_abs = dict(
        map(lambda x, y: [x, abs(y)], causal_strengths.keys(),
            causal_strengths.values()))
    causal_strengths_abs_sorted = sorted(causal_strengths_abs.items(),
                                         key=lambda x: x[1],
                                         reverse=True)
    weightiest_var = str(causal_strengths_abs_sorted[0][0]).split('_')[1]
    sign = '+'
    if np.sign(causal_strengths[causal_strengths_abs_sorted[0][0]]) < 0:
        sign = '-'
    return weightiest_var + ':' + str(sign)


def calculate_shortest_path_causal_var_selfnet_pos(p_net, p_agent,
                                                   p_target_var):
    if not str(p_agent) + '_' + p_target_var in p_net.nodes():
        return 'Uncertain'
    short_paths = nx.single_target_shortest_path(
        p_net,
        str(p_agent) + '_' + p_target_var)
    del short_paths[str(p_agent) + '_' + p_target_var]
    if short_paths == {}:
        return 'Uncertain'
    causal_strengths = {}
    for source_n in short_paths:
        causal_strengths[source_n] = calculate_sum_weight_of_path(
            p_net, short_paths[source_n])
    causal_strengths_abs_sorted = sorted(causal_strengths.items(),
                                         key=lambda x: x[1],
                                         reverse=True)
    weightiest_var = str(causal_strengths_abs_sorted[0][0]).split('_')[1]
    sign = '+'
    if np.sign(causal_strengths[causal_strengths_abs_sorted[0][0]]) < 0:
        sign = '-'
    return weightiest_var + ':' + str(sign)


def calculate_all_path_causal_var_selfnet(p_net, p_agent, p_target_var):
    causal_strengths = {}
    for var in VARS_LIST:
        if var == p_target_var:
            continue
        source_n = str(p_agent) + '_' + var
        if not source_n in p_net.nodes():
            continue
        all_paths = nx.all_simple_paths(p_net, source_n,
                                        str(p_agent) + '_' + p_target_var, 5)
        strength_sum = 0
        paths_count = 0
        for a_path in all_paths:
            a_path_strength = calculate_sum_weight_of_path(p_net, a_path)
            strength_sum = strength_sum + a_path_strength
            paths_count = paths_count + 1
        if paths_count == 0:
            continue
        causal_strengths[source_n] = strength_sum / paths_count
    if causal_strengths == {}:
        return 'Uncertain'
    causal_strengths_abs = dict(
        map(lambda x, y: [x, abs(y)], causal_strengths.keys(),
            causal_strengths.values()))
    causal_strengths_abs_sorted = sorted(causal_strengths_abs.items(),
                                         key=lambda x: x[1],
                                         reverse=True)
    weightiest_var = str(causal_strengths_abs_sorted[0][0]).split('_')[1]
    sign = '+'
    if np.sign(causal_strengths[causal_strengths_abs_sorted[0][0]]) < 0:
        sign = '-'
    return weightiest_var + ':' + str(sign)


def calculate_all_path_causal_var_selfnet_pos(p_net, p_agent, p_target_var):
    causal_strengths = {}
    for var in VARS_LIST:
        if var == p_target_var:
            continue
        source_n = str(p_agent) + '_' + var
        if not source_n in p_net.nodes():
            continue
        all_paths = nx.all_simple_paths(p_net, source_n,
                                        str(p_agent) + '_' + p_target_var, 5)
        strength_sum = 0
        paths_count = 0
        for a_path in all_paths:
            a_path_strength = calculate_sum_weight_of_path(p_net, a_path)
            strength_sum = strength_sum + a_path_strength
            paths_count = paths_count + 1
        if paths_count == 0:
            continue
        causal_strengths[source_n] = strength_sum / paths_count
    if causal_strengths == {}:
        return 'Uncertain'
    causal_strengths_abs_sorted = sorted(causal_strengths.items(),
                                         key=lambda x: x[1],
                                         reverse=True)
    weightiest_var = str(causal_strengths_abs_sorted[0][0]).split('_')[1]
    sign = '+'
    if np.sign(causal_strengths[causal_strengths_abs_sorted[0][0]]) < 0:
        sign = '-'
    return weightiest_var + ':' + str(sign)


def calculate_sum_weight_of_path(p_net, p_nodes_list):
    sum = 0
    for i, n_list in enumerate(p_nodes_list):
        if i == len(p_nodes_list) - 1:
            break
        props = p_net[p_nodes_list[i]][p_nodes_list[i + 1]]
        weights = 0
        for prop in props.values():
            weights = weights + prop['weight']
        sum = sum + weights
    return sum / (len(p_nodes_list) - 1)


def show_inner_nets():
    """
    output inner net info
    """
    # # draw inner net centrality info in GIS
    # new_pool = multiprocessing.Pool()
    # for var in VARS_LIST:
    #     p = new_pool.apply_async(draw_inner_centrality_info, args=(var, ))
    #     print('draw_inner_info_in_GIS:', var)
    # new_pool.close()
    # new_pool.join()

    # draw inner net communities info in GIS
    new_pool = multiprocessing.Pool()
    for var in VARS_LIST:
        p = new_pool.apply_async(draw_inner_communities_info, args=(var, ))
        print('draw_inner_info_in_GIS:', var)
    new_pool.close()
    new_pool.join()

    # set multiprocessing to get info
    new_pool = multiprocessing.Pool()
    jobs = []
    for var in VARS_LIST:
        p = new_pool.apply_async(get_inner_info_by_var, args=(var, ))
        jobs.append(p)
        print('get_inner_info_by_var:', var)
    new_pool.close()
    new_pool.join()
    kinds_links = []
    for job in jobs:
        kinds_links.append(job.get())
    nets_info = pd.concat(kinds_links, ignore_index=True)
    nets_info.to_csv(BaseConfig.OUT_PATH + 'All_Inner_Nets_Info.csv')


def get_inner_info_by_var(p_var):
    """
    get inner based info by var 
    """
    net_info = pd.DataFrame()
    var_inner_df = pd.read_csv(BaseConfig.OUT_PATH + 'InnerNetworkCSV//' +
                               p_var + '_filtered.csv')
    var_inner_net = build_net_by_csv(var_inner_df)

    net_info = net_info.append(
        pd.DataFrame(
            {
                'Name':
                'Inner Net of ' + p_var,
                'Number of nodes':
                nx.number_of_nodes(var_inner_net),
                'Number of edges':
                nx.number_of_edges(var_inner_net),
                'Average Node Degree':
                sum(d for n, d in var_inner_net.degree()) /
                float(nx.number_of_nodes(var_inner_net)),
                'Density':
                nx.density(var_inner_net),
                'Transitivity':
                nx.transitivity(nx.DiGraph(var_inner_net)),
                'Average Node Connectivity':
                nx.average_node_connectivity(var_inner_net),
            },
            index=[0]))
    return net_info


def draw_inner_communities_info(p_var):
    """
    draw inner communities info on GIS
    """
    var_inner_df = pd.read_csv(BaseConfig.OUT_PATH + 'InnerNetworkCSV//' +
                               p_var + '_filtered.csv')
    var_inner_net = build_net_by_csv(var_inner_df)
    # community_list=community.girvan_newman(var_inner_net)
    # comp_tuple=tuple(sorted(c) for c in next(community_list))
    # print(comp_tuple)
    # print(len(comp_tuple))
    community_list = community.asyn_fluidc(nx.MultiGraph(var_inner_net), 6)
    com_group_number = {}
    for c_index, com in enumerate(list(community_list)):
        for n in com:
            com_group_number[n] = c_index
    cmap = plt.get_cmap('tab10')

    fig = plt.figure(figsize=(20, 12), dpi=500)
    geoagent_shp_path = BaseConfig.GEO_AGENT_PATH + 'GA_WGS84.shp'
    targetPro = ccrs.PlateCarree()
    ax = fig.add_subplot(1, 1, 1, projection=targetPro)
    # ax.set_global()
    # ax.stock_img()
    ax.add_feature(cfeature.OCEAN.with_scale('50m'))
    ax.add_feature(cfeature.LAND.with_scale('50m'))
    ax.add_feature(cfeature.RIVERS.with_scale('50m'))
    ax.add_feature(cfeature.LAKES.with_scale('50m'))
    ax.set_extent([95, 120, 31.5, 42])

    for feature in Reader(geoagent_shp_path).records():
        ga_id = feature.attributes['GA_ID']
        polygon_geo = ShapelyFeature(feature.geometry, ccrs.PlateCarree())
        try:
            face_color = cmap(com_group_number[str(ga_id) + '_' + p_var])
        except:
            face_color = '#737373'
        ax.add_feature(polygon_geo,
                       linewidth=0.4,
                       facecolor=face_color,
                       edgecolor='#4D5459',
                       alpha=0.8)
    plt.savefig(BaseConfig.OUT_PATH + 'InnerNetCommunitiesMap//' + p_var +
                '_Communities_info_Map.pdf')


def draw_inner_centrality_info(p_var):
    """
    draw inner centrality info on GIS
    """
    var_inner_df = pd.read_csv(BaseConfig.OUT_PATH + 'InnerNetworkCSV//' +
                               p_var + '_filtered.csv')
    var_inner_net = build_net_by_csv(var_inner_df)

    # draw degree map
    fig = plt.figure(figsize=(20, 12), dpi=500)
    geoagent_shp_path = BaseConfig.GEO_AGENT_PATH + 'GA_WGS84.shp'
    targetPro = ccrs.PlateCarree()
    ax = fig.add_subplot(2, 2, 1, projection=targetPro)
    # ax.set_global()
    # ax.stock_img()
    ax.add_feature(cfeature.OCEAN.with_scale('50m'))
    ax.add_feature(cfeature.LAND.with_scale('50m'))
    ax.add_feature(cfeature.RIVERS.with_scale('50m'))
    ax.add_feature(cfeature.LAKES.with_scale('50m'))
    ax.set_extent([95, 120, 31.5, 42])
    cmap = plt.get_cmap('Reds')
    norm = matplotlib.colors.Normalize(
        vmin=min(d for n, d in var_inner_net.degree()),
        vmax=max(d for n, d in var_inner_net.degree()))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=ax, orientation='horizontal')
    for l in cb.ax.xaxis.get_ticklabels():
        l.set_family('Times New Roman')
        l.set_size(14)
    cb.set_label('Degree', fontsize=14, fontfamily='Times New Roman')
    for feature in Reader(geoagent_shp_path).records():
        ga_id = feature.attributes['GA_ID']
        polygon_geo = ShapelyFeature(feature.geometry, ccrs.PlateCarree())
        try:
            face_color = sm.to_rgba(var_inner_net.degree[str(ga_id) + '_' +
                                                         p_var])
        except:
            face_color = '#737373'
        ax.add_feature(polygon_geo,
                       linewidth=0.4,
                       facecolor=face_color,
                       edgecolor='#4D5459',
                       alpha=0.8)

    # draw out_degree/indegree
    ax = fig.add_subplot(2, 2, 2, projection=targetPro)
    # ax.set_global()
    # ax.stock_img()
    ax.add_feature(cfeature.OCEAN.with_scale('50m'))
    ax.add_feature(cfeature.LAND.with_scale('50m'))
    ax.add_feature(cfeature.RIVERS.with_scale('50m'))
    ax.add_feature(cfeature.LAKES.with_scale('50m'))
    ax.set_extent([95, 120, 31.5, 42])
    cmap = plt.get_cmap('Reds')
    o_divide_i_values = []
    for n, d in var_inner_net.out_degree():
        try:
            o_divide_i_values.append(d / var_inner_net.in_degree(n))
        except:
            continue
    norm = matplotlib.colors.Normalize(vmin=min(o_divide_i_values),
                                       vmax=max(o_divide_i_values))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=ax, orientation='horizontal')
    for l in cb.ax.xaxis.get_ticklabels():
        l.set_family('Times New Roman')
        l.set_size(14)
    cb.set_label('out_degree/indegree',
                 fontsize=14,
                 fontfamily='Times New Roman')
    for feature in Reader(geoagent_shp_path).records():
        ga_id = feature.attributes['GA_ID']
        polygon_geo = ShapelyFeature(feature.geometry, ccrs.PlateCarree())
        try:
            face_color = sm.to_rgba(
                (var_inner_net.out_degree[str(ga_id) + '_' + p_var]) /
                (var_inner_net.in_degree[str(ga_id) + '_' + p_var]))
        except:
            face_color = '#737373'
        ax.add_feature(polygon_geo,
                       linewidth=0.4,
                       facecolor=face_color,
                       edgecolor='#4D5459',
                       alpha=0.8)

    # draw betweenness map
    ax = fig.add_subplot(2, 2, 3, projection=targetPro)
    # ax.set_global()
    # ax.stock_img()
    ax.add_feature(cfeature.OCEAN.with_scale('50m'))
    ax.add_feature(cfeature.LAND.with_scale('50m'))
    ax.add_feature(cfeature.RIVERS.with_scale('50m'))
    ax.add_feature(cfeature.LAKES.with_scale('50m'))
    ax.set_extent([95, 120, 31.5, 42])

    cmap = plt.get_cmap('Purples')
    norm = matplotlib.colors.Normalize(
        vmin=min(
            nx.betweenness_centrality(nx.DiGraph(var_inner_net)).values()),
        vmax=max(
            nx.betweenness_centrality(nx.DiGraph(var_inner_net)).values()))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=ax, orientation='horizontal')
    for l in cb.ax.xaxis.get_ticklabels():
        l.set_family('Times New Roman')
        l.set_size(14)
    cb.set_label('Betweenness', fontsize=14, fontfamily='Times New Roman')
    for feature in Reader(geoagent_shp_path).records():
        ga_id = feature.attributes['GA_ID']
        polygon_geo = ShapelyFeature(feature.geometry, ccrs.PlateCarree())
        try:
            face_color = sm.to_rgba(
                nx.betweenness_centrality(
                    nx.DiGraph(var_inner_net))[str(ga_id) + '_' + p_var])
        except:
            face_color = '#737373'
        ax.add_feature(polygon_geo,
                       linewidth=0.4,
                       facecolor=face_color,
                       edgecolor='#4D5459',
                       alpha=0.8)

    # draw Closeness map
    ax = fig.add_subplot(2, 2, 4, projection=targetPro)
    # ax.set_global()
    # ax.stock_img()
    ax.add_feature(cfeature.OCEAN.with_scale('50m'))
    ax.add_feature(cfeature.LAND.with_scale('50m'))
    ax.add_feature(cfeature.RIVERS.with_scale('50m'))
    ax.add_feature(cfeature.LAKES.with_scale('50m'))
    ax.set_extent([95, 120, 31.5, 42])

    cmap = plt.get_cmap('Blues')
    norm = matplotlib.colors.Normalize(
        vmin=min(nx.closeness_centrality(var_inner_net).values()),
        vmax=max(nx.closeness_centrality(var_inner_net).values()))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=ax, orientation='horizontal')
    for l in cb.ax.xaxis.get_ticklabels():
        l.set_family('Times New Roman')
        l.set_size(14)
    cb.set_label('Closeness', fontsize=14, fontfamily='Times New Roman')
    for feature in Reader(geoagent_shp_path).records():
        ga_id = feature.attributes['GA_ID']
        polygon_geo = ShapelyFeature(feature.geometry, ccrs.PlateCarree())
        try:
            face_color = sm.to_rgba(
                nx.closeness_centrality(var_inner_net)[str(ga_id) + '_' +
                                                       p_var])
        except:
            face_color = '#737373'
        ax.add_feature(polygon_geo,
                       linewidth=0.4,
                       facecolor=face_color,
                       edgecolor='#4D5459',
                       alpha=0.8)

    plt.savefig(BaseConfig.OUT_PATH + 'InnerNetCentralityMap//' + p_var +
                '_Centrality_info_Map.pdf')
    plt.close()


def build_net_by_csv(p_edges_df):
    """
    built a net by a pands df
    """
    # build a net network
    network = nx.MultiDiGraph()
    # add every vertex to the net
    var_sou = p_edges_df['VarSou'].map(str)
    var_tar = p_edges_df['VarTar'].map(str)
    id_sou = p_edges_df['Source'].map(str)
    id_tar = p_edges_df['Target'].map(str)
    p_edges_df['Source_label'] = id_sou + '_' + var_sou
    p_edges_df['Target_label'] = id_tar + '_' + var_tar
    all_ver_list = list(p_edges_df['Source_label']) + list(
        p_edges_df['Target_label'])
    # set the unique of the vertexs
    ver_list_unique = list(set(all_ver_list))
    for v_id_var in ver_list_unique:
        network.add_node(v_id_var,
                         var_name=v_id_var.split('_')[1],
                         ga_id=v_id_var.split('_')[0],
                         label=v_id_var.split('_')[0],
                         size=30,
                         color=VAR_COLOR_DICT[v_id_var.split('_')[1]],
                         label_size=15)
    for lIndex, lRow in p_edges_df.iterrows():
        thisSou = lRow["Source_label"]
        thisTar = lRow["Target_label"]
        network.add_edge(thisSou,
                         thisTar,
                         weight=lRow['Strength'],
                         timelag=abs(lRow['TimeLag']))
        # for lf in p_edges_df.columns.values:
        #     inner_network.edges[thisSou, thisTar][lf] = lRow[lf]
    return network


def filter_links(p_links):
    """
    filter links by some rules
    @param p_links:links
    @return:
    """
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
    coupled_network = build_net_by_csv(filtered_edges_df)
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
            try:
                one_link = build_link_pcmci_noself(
                    data_2_row.T, [id_data[i_sou], id_data[i_tar]], p_var_name,
                    p_var_name)
                one_links.append(one_link)
            except:
                continue
    inner_links = pd.concat(one_links, ignore_index=True)
    inner_links.to_csv(BaseConfig.OUT_PATH + 'InnerNetworkCSV//' + p_var_name +
                       '.csv')
    filtered_edges_df = filter_links(inner_links)
    filtered_edges_df.to_csv(BaseConfig.OUT_PATH + 'InnerNetworkCSV//' +
                             p_var_name + '_filtered.csv')
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
            try:
                one_link = build_link_pcmci_noself(
                    data_2_row.T, [id_sou[i_sou], id_tar[i_tar]], p_var_sou,
                    p_var_tar)
                one_links.append(one_link)
            except:
                continue
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
                # filter all zero row
                same_counts_month = var_data_monthly.loc[agent].value_counts()
                if same_counts_month.values.max() < 10:
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
            same_counts = var_data.loc[agent].value_counts()
            if same_counts.values.max() < 10:
                yearly_vars_data.append(var_data.loc[agent])
                yearly_vars.append(var)
        month_data_links = pd.DataFrame()
        year_data_links = pd.DataFrame()
        monthly_data = np.array(monthly_vars_data)
        # run month data drived links
        if monthly_data != []:
            month_data_links = build_link_pcmci_noself(monthly_data.T,
                                                       monthly_vars, '--',
                                                       '--')
        yearly_data = np.array(yearly_vars_data)
        # run year data drived links
        if yearly_data != []:
            year_data_links = build_link_pcmci_noself(yearly_data.T,
                                                      yearly_vars, '--', '--')
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
    self_links.to_csv(BaseConfig.OUT_PATH + 'SelfNetworkCSV//' +
                      'SelfNetworkAll' + '.csv')
    filtered_edges_df = filter_links(self_links)
    filtered_edges_df.to_csv(BaseConfig.OUT_PATH + 'SelfNetworkCSV//' +
                             'SelfNetworkAll_filtered' + '.csv')
    return self_links


def get_self_links_only_year_data():
    """
    get self links for every GeoAgent
    """
    # out put the same columns
    year_times_intersection = list(map(str, np.arange(1955, 2030)))
    for var in VARS_LIST:
        # yearly data
        var_data = pd.read_csv(BaseConfig.COUPLED_NET_DATA_PATH +
                               BaseConfig.COUPLED_NET_DATA_HEAD + var +
                               '_yearly' + BaseConfig.COUPLED_NET_DATA_TAIL,
                               index_col='GA_ID')
        year_times_intersection = set(year_times_intersection).intersection(
            set(list(var_data.columns.values)))
    # sord them
    year_times_intersection = sorted(list(year_times_intersection))

    centroid_data = pd.read_csv(BaseConfig.GEO_AGENT_PATH + 'GA_Centroid.csv')
    agents_links = []
    for agent in list(centroid_data['GA_ID']):
        yearly_vars = []
        yearly_vars_data = []
        for var in VARS_LIST:
            # yearly data
            var_data = pd.read_csv(BaseConfig.COUPLED_NET_DATA_PATH +
                                   BaseConfig.COUPLED_NET_DATA_HEAD + var +
                                   '_yearly' +
                                   BaseConfig.COUPLED_NET_DATA_TAIL,
                                   index_col='GA_ID')
            var_data.fillna(value=BaseConfig.BACKGROUND_VALUE, inplace=True)
            var_data = var_data[list(year_times_intersection)]
            same_counts = var_data.loc[agent].value_counts()
            if same_counts.values.max() < 10:
                yearly_vars_data.append(var_data.loc[agent])
                yearly_vars.append(var)

        yearly_data = np.array(yearly_vars_data)
        # run year data drived links
        year_data_links = build_link_pcmci_noself(yearly_data.T, yearly_vars,
                                                  '--', '--')
        agent_links = year_data_links
        # change the format of agent_links
        agent_links['VarSou'] = agent_links['Source']
        agent_links['VarTar'] = agent_links['Target']
        agent_links['Source'] = agent
        agent_links['Target'] = agent
        agents_links.append(agent_links)
    self_links = pd.concat(agents_links, ignore_index=True)
    self_links.to_csv(BaseConfig.OUT_PATH + 'SelfNetworkCSV//' +
                      'SelfNetworkAll' + '.csv')
    filtered_edges_df = filter_links(self_links)
    filtered_edges_df.to_csv(BaseConfig.OUT_PATH + 'SelfNetworkCSV//' +
                             'SelfNetworkAll_filtered' + '.csv')
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
    pcmci = PCMCI(dataframe=data_frame,
                  cond_ind_test=ParCorr(significance='analytic'))
    # run PCMCI
    alpha_level = None
    results_pcmci = pcmci.run_pcmciplus(tau_min=0,
                                        tau_max=3,
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
    # q_matrix is the corrected p_matrix
    # elif q_matrix is not None:
    #     sig_links = (q_matrix <= alpha_level)
    # else:
    #     sig_links = (p_matrix <= alpha_level)
    for j in range(agent_num):
        links = {(p[0], -p[1]): np.abs(val_matrix[p[0], j, abs(p[1])])
                 for p in zip(*np.where(sig_links[:, j, :]))}
        # Sort by value
        sorted_links = sorted(links, key=links.get, reverse=True)
        for p in sorted_links:
            VarSou = p_var_sou
            VarTar = p_var_tar
            Source = p_agent_names[p[0]]
            Target = p_agent_names[j]
            TimeLag = p[1]
            Strength = val_matrix[p[0], j, abs(p[1])]
            Unoriented = None
            if graph_pcmci is not None:
                if graph_pcmci[j, p[0], 0] == "o-o":
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
    # get_self_links()
    # print(time.strftime('%H:%M:%S', time.localtime(time.time())))
    # show_inner_nets()
    print(time.strftime('%H:%M:%S', time.localtime(time.time())))
    show_self_nets('LAI')
    print(time.strftime('%H:%M:%S', time.localtime(time.time())))
    print('Done! Thriller!')
