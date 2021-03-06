# from .setting import *
# from .useful_class import *
# from .useful_class import *
from matplotlib.pyplot import xlim
from setting import *
from useful_class import *
from useful_class import *
import numpy as np
import pandas as pd
import igraph
import pyunicorn.core.network as network
import matplotlib as mpl
mpl.use('cairo')
import matplotlib.pyplot as plt
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'NSimSun'

# The code and label dictionary
LABEL_DIC = {
    101: '落叶阔叶林',
    102: '常绿针叶林',
    103: '落叶常绿混交林',
    201: '灌丛',
    301: '低覆盖草地',
    302: '中覆盖草地',
    303: '高覆盖草地',
    401: '谷物耕地',
    402: '果园和梯田',
    501: '城市及建设用地',
    601: '地表水',
    602: '湿地',
    603: '积雪和冰',
    701: '沙漠和裸土',
    702: '低植被覆盖地表',
}
# The label and color dictionary
COLOR_DIC = {
    '落叶阔叶林': '#1B7201',
    '常绿针叶林': '#064A01',
    '落叶常绿混交林': '#01B61F',
    '灌丛': '#50FF00',
    '低覆盖草地': '#A4D081',
    '中覆盖草地': '#88B75F',
    '高覆盖草地': '#6E9E45',
    '谷物耕地': '#F2F100',
    '果园和梯田': '#F29B00',
    '城市及建设用地': '#F00001',
    '地表水': '#0058F0',
    '湿地': '#00E0F0',
    '积雪和冰': '#BDF4F8',
    '沙漠和裸土': '#6A6A6A',
    '低植被覆盖地表': '#CCCCCC'
}
LABEL_DIC_EN = {
    '落叶阔叶林': 'DBF',
    '常绿针叶林': 'ENF',
    '落叶常绿混交林': 'MF',
    '灌丛': 'Shrub',
    '低覆盖草地': 'LCG',
    '中覆盖草地': 'MCG',
    '高覆盖草地': 'HCG',
    '谷物耕地': 'Crop',
    '果园和梯田': 'OT',
    '城市及建设用地': 'UB',
    '地表水': 'Water',
    '湿地': 'Wet',
    '积雪和冰': 'Snow',
    '沙漠和裸土': 'DB',
    '低植被覆盖地表': 'LV',
}
# LABEL_DIC = {
#     101: 'DBF',
#     102: 'ENF',
#     103: 'MF',
#     201: 'Shrub',
#     301: 'LCG',
#     302: 'MCG',
#     303: 'HCG',
#     401: 'Crop',
#     402: 'OT',
#     501: 'UB',
#     601: 'Water',
#     602: 'Wet',
#     603: 'Snow',
#     701: 'DB',
#     702: 'LV',
# }
# # The label and color dictionary
# COLOR_DIC = {
#     'DBF': '#1B7201',
#     'ENF': '#064A01',
#     'MF': '#01B61F',
#     'Shrub': '#50FF00',
#     'LCG': '#A4D081',
#     'MCG': '#88B75F',
#     'HCG': '#6E9E45',
#     'Crop': '#F2F100',
#     'OT': '#F29B00',
#     'UB': '#F00001',
#     'Water': '#0058F0',
#     'Wet': '#00E0F0',
#     'Snow': '#BDF4F8',
#     'DB': '#6A6A6A',
#     'LV': '#CCCCCC'
# }

# this info is from the clustered result
CLUSTERED_COLOR = {
    '落叶阔叶林': '#028760',
    '常绿针叶林': '#165e83',
    '落叶常绿混交林': '#028760',
    '灌丛': '#028760',
    '低覆盖草地': '#f39800',
    '中覆盖草地': '#f39800',
    '高覆盖草地': '#7ebea5',
    '谷物耕地': '#e95464',
    '果园和梯田': '#e95464',
    '城市及建设用地': '#e95464',
    '地表水': '#2ca9e1',
    '湿地': '#2ca9e1',
    '积雪和冰': '#2ca9e1',
    '沙漠和裸土': '#2ca9e1',
    '低植被覆盖地表': '#c8c2be'
}


def find_threshold(p_timeSta, p_timeEnd, p_threshold_per):
    '''
    Get the suitable threshold according to ChangeArea
    return a threshold according to the input percent number
    '''
    # get all links in a DataFrame
    series_df = pd.DataFrame()
    for year in np.arange(p_timeSta, p_timeEnd + 1):
        this_year = pd.read_csv(BaseConfig.LUC_NET_DATA_PATH +
                                BaseConfig.LUC_NET_DATA_HEAD + str(year) +
                                BaseConfig.LUC_NET_DATA_TAIL)
        series_df = series_df.append(this_year)
    threshold_min = np.percentile(series_df['ChangeArea'].values,
                                  p_threshold_per)
    threshold_max = np.percentile(series_df['ChangeArea'].values, 100)
    series_df = series_df[(series_df['ChangeArea'] > threshold_min)
                          & (series_df['ChangeArea'] < threshold_max)]
    # see the histogram
    fig, axs = plt.subplots(figsize=(7, 5), dpi=400)
    plt.hexbin(series_df['ChangeArea'],
               series_df['ChangeLA'],
               bins=625,
               cmap='Blues')
    plt.xlabel('ChangeArea')
    plt.ylabel('ChangeLA')
    cb = plt.colorbar()
    cb.set_label('counts in bin')
    plt.savefig(BaseConfig.OUT_PATH + 'LUCC_Net//HistofAllLinks.png')
    return threshold_min


def build_luc_net(p_timeSta, p_timeEnd, p_topic_var, p_threshold_area,
                  p_color_var):
    '''
    Build net for change area in a huge multiple net and output information every class
    return this network
    '''
    # build a net network
    lucc_net = igraph.Graph(directed=True)
    # add every vertex to the net
    for v_class in list(LABEL_DIC.values()):
        lucc_net.add_vertex(v_class,
                            name_en=LABEL_DIC_EN[v_class],
                            color=CLUSTERED_COLOR[v_class],
                            size=65,
                            label=v_class,
                            label_size=30,
                            frame_width=0)
    # get all links in a DataFrame
    series_df = pd.DataFrame()
    for year in np.arange(p_timeSta, p_timeEnd + 1):
        this_year = pd.read_csv(BaseConfig.LUC_NET_DATA_PATH +
                                BaseConfig.LUC_NET_DATA_HEAD + str(year) +
                                BaseConfig.LUC_NET_DATA_TAIL)
        source_label = this_year['Source'].map(LABEL_DIC)
        target_label = this_year['Target'].map(LABEL_DIC)
        this_year['Source_label'] = source_label
        this_year['Target_label'] = target_label
        this_year['Change_type'] = source_label + ' to ' + target_label
        this_year = this_year[[
            'Source', 'Target', 'Source_label', 'Target_label', 'Year',
            'Change_type', 'ChangeArea', 'ChangeLA'
        ]]
        series_df = series_df.append(this_year)
    # filter this DataFrame (all edges)
    series_df = series_df[(series_df['ChangeArea'] > p_threshold_area)]
    # set the edges for the net
    tuples = [
        tuple(x) for x in series_df[['Source_label', 'Target_label']].values
    ]
    lucc_net.add_edges(tuples)
    # set the width with a scale and set the max width
    # there we use the absolute value to drwa the edge
    widths = abs(series_df[p_topic_var] * 0.000000001)
    max_width = np.percentile(widths, 95)
    lucc_net.es['width'] = list(
        widths.replace(widths[widths > max_width], max_width))
    # set the year property
    lucc_net.es['year'] = list(series_df['Year'])
    # if the colr var is Year
    if p_color_var == 'Year':
        # get the color bar from matplotlib
        cmap = plt.get_cmap('viridis')
        norm = mpl.colors.Normalize(vmin=p_timeSta, vmax=p_timeEnd)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        lucc_net.es['color'] = list(series_df['Year'].map(sm.to_rgba))
        # output this colorbar legend
        fig_cb = plt.figure(figsize=(10, 6), dpi=400)
        ax_cb = fig_cb.add_subplot(111)
        axcolorbar = ax_cb.scatter(series_df['Year'],
                                   series_df['Year'],
                                   c=series_df['Year'],
                                   norm=norm,
                                   cmap=cmap)
        color_bar = fig_cb.colorbar(axcolorbar,
                                    ax=ax_cb,
                                    orientation='vertical')
        color_bar.set_ticks([1987, 1990, 1995, 2000, 2005, 2010, 2015, 2018])
        color_bar.set_ticklabels(
            [1987, 1990, 1995, 2000, 2005, 2010, 2015, 2018])
        for l in color_bar.ax.xaxis.get_ticklabels():
            l.set_family('Times New Roman')
            l.set_size(14)
        color_bar.set_label('地类转移发生年', fontsize=14)
        plt.savefig(BaseConfig.OUT_PATH + 'LUCC_Net//' + p_topic_var +
                    '_colorbar_legend.pdf',
                    bbox_inches='tight',
                    transparent=True)

    # when the width var has negative value, we should show the +- replace the Year
    if p_color_var == 'Sign':
        SIGN_COLOR_DIC = {-1: '#FF69B4', 1: '#4169E1', 0: '#00FFFFFF'}
        lucc_net.es['color'] = list(series_df[p_topic_var].map(
            np.sign).map(SIGN_COLOR_DIC))
    # set the curved of edge
    # lucc_net.es['curved'] = 0.15
    # also can use this code to get different curved edge
    # lucc_net.es['curved'] = list(series_df['Year']*0.0002)
    # set the arrow size
    lucc_net.es['arrow_size'] = 0.5

    # get the property of this network
    lucc_class = lucc_net.vs['label']
    lucc_degree = lucc_net.degree()
    lucc_indeg = lucc_net.indegree()
    lucc_oudeg = lucc_net.outdegree()
    lucc_out_div_in = np.divide(np.array(lucc_oudeg), np.array(lucc_indeg))

    lucc_betness = lucc_net.betweenness()
    lucc_cloness = lucc_net.closeness()
    lucc_diversity = lucc_net.diversity(weights='width')

    # draw this network
    # set the vertex size according its degree
    lucc_net.vs['size'] = normalization(lucc_degree) * 100
    igraph.plot(lucc_net,
                BaseConfig.OUT_PATH + 'LUCC_Net//' + p_topic_var +
                '_multiple_network.pdf',
                bbox=(1200, 1200),
                margin=200)

    # output information in a csv file
    out_net_info = pd.DataFrame({
        'point': lucc_class,
        'degree': lucc_degree,
        'betweenness': lucc_betness,
        'closeness': lucc_cloness,
        'diversity': lucc_diversity,
        'indegree': lucc_indeg,
        'outdegree': lucc_oudeg,
        'outdegree_div_indegree': lucc_out_div_in
    })
    out_net_info.to_csv(BaseConfig.OUT_PATH + 'LUCC_Net//' + p_topic_var +
                        '_multiple_network_vertex_info.csv')

    # ***** This part is about how to use igraph in Matplotlib *****
    # import matplotlib as mpl
    # mpl.use('cairo')
    # import matplotlib.pyplot as plt
    # fig=plt.figure()
    # axes=fig.add_subplot(111)
    # graph_artist=GraphArtist(lucc_net,(10, 10, 150, 150),layout=layout)
    # graph_artist.set_zorder(float('inf'))
    # axes.artists.append(graph_artist)
    # plt.axis("off")
    # fig.suptitle('sdf')
    # fig.savefig(BaseConfig.OUT_PATH + 'luccnet_mpl.pdf')
    return lucc_net


def build_luc_net_no_mul(p_timeSta, p_timeEnd, p_topic_var, p_threshold_area,
                         p_color_var):
    '''
    Build net for change area in a continuous and im-multiple net and output information of every year net
    return this network
    '''
    # build a net network
    lucc_net = igraph.Graph(directed=True)
    # add every vertex to the net
    for v_class in list(LABEL_DIC.values()):
        lucc_net.add_vertex(v_class + str(1986),
                            label=v_class,
                            color=COLOR_DIC[v_class],
                            year=1986,
                            size=40)
        for o_year in np.arange(p_timeSta, p_timeEnd + 1):
            lucc_net.add_vertex(v_class + str(o_year),
                                label=v_class,
                                color=COLOR_DIC[v_class],
                                year=o_year,
                                size=40)
    # new some var to put information
    sub_net_mean_shortest = []
    sub_net_density = []
    sub_net_transitivity_global = []
    # get all links in a DataFrame
    series_df = pd.DataFrame()
    for year in np.arange(p_timeSta, p_timeEnd + 1):
        this_year = pd.read_csv(BaseConfig.LUC_NET_DATA_PATH +
                                BaseConfig.LUC_NET_DATA_HEAD + str(year) +
                                BaseConfig.LUC_NET_DATA_TAIL)
        source_label = this_year['Source'].map(LABEL_DIC)
        target_label = this_year['Target'].map(LABEL_DIC)
        this_year['Source_label_noY'] = source_label
        this_year['Target_label_noY'] = target_label
        this_year['Source_label'] = source_label + (this_year['Year'] -
                                                    1).map(str)
        this_year['Target_label'] = target_label + this_year['Year'].map(str)
        this_year['Change_type'] = source_label + ' to ' + target_label
        this_year = this_year[[
            'Source', 'Target', 'Source_label_noY', 'Target_label_noY',
            'Source_label', 'Target_label', 'Year', 'Change_type',
            'ChangeArea', 'ChangeLA'
        ]]
        # filter this year's edges
        this_year = this_year[(this_year['ChangeArea'] > p_threshold_area)]
        # build sub net in every continuous two year
        sub_lucc_net = igraph.Graph(directed=True)
        for v_class in list(LABEL_DIC.values()):
            sub_lucc_net.add_vertex(v_class)
        sub_tuples = [
            tuple(x)
            for x in this_year[['Source_label_noY', 'Target_label_noY']].values
        ]
        sub_lucc_net.add_edges(sub_tuples)
        mean_shortest = np.mean(sub_lucc_net.shortest_paths())
        sub_net_mean_shortest.append(mean_shortest)
        density = sub_lucc_net.density()
        sub_net_density.append(density)
        transitivity_global = sub_lucc_net.transitivity_undirected()
        sub_net_transitivity_global.append(transitivity_global)
        # add to the DataFrame for the huge net
        series_df = series_df.append(this_year)
    # output these sub net information in csv file
    out_sub_net_info = pd.DataFrame({
        'target_year':
        np.arange(p_timeSta, p_timeEnd + 1),
        'mean_shortest_path':
        sub_net_mean_shortest,
        'density':
        sub_net_density,
        'transitivity_undirected':
        sub_net_transitivity_global
    })
    out_sub_net_info.to_csv(BaseConfig.OUT_PATH + 'LUCC_Net//' + p_topic_var +
                            '_sub_net_every2year_info.csv')
    # set the edges
    tuples = [
        tuple(x) for x in series_df[['Source_label', 'Target_label']].values
    ]
    lucc_net.add_edges(tuples)
    # set the width with a scale and set the max width
    # there we use the absolute value to drwa the edge
    widths = abs(series_df[p_topic_var] * 0.000000001)
    max_width = np.percentile(widths, 95)
    lucc_net.es['width'] = list(
        widths.replace(widths[widths > max_width], max_width))
    # set the year property
    lucc_net.es['year'] = list(series_df['Year'])
    # if the colr var is Year
    if p_color_var == 'Year':
        cmap = plt.get_cmap('viridis')
        norm = mpl.colors.Normalize(vmin=p_timeSta, vmax=p_timeEnd)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        lucc_net.es['color'] = list(series_df['Year'].map(sm.to_rgba))
    # when the width var has negative value, we should show the +- replace the Year
    if p_color_var == 'Sign':
        SIGN_COLOR_DIC = {-1: '#FF69B480', 1: '#4169E180', 0: '#00FFFFFF'}
        lucc_net.es['color'] = list(series_df[p_topic_var].map(
            np.sign).map(SIGN_COLOR_DIC))
    # set the curved of edge
    lucc_net.es['curved'] = 0.1
    # also can use this code to get different curved edge
    # lucc_net.es['curved'] = list(series_df['Year']*0.00025)
    # set the arrow size
    lucc_net.es['arrow_size'] = 0.5

    # show the network
    igraph.plot(lucc_net,
                BaseConfig.OUT_PATH + 'LUCC_Net//' + p_topic_var +
                '_im_multiple_network.pdf',
                bbox=(2500, 1000),
                margin=100,
                layout=lucc_net.layout_grid(width=(p_timeEnd - p_timeSta + 2),
                                            dim=2))

    # get the property of this network's vertex
    lucc_class = lucc_net.vs['label']
    lucc_degree = lucc_net.degree()
    lucc_ind = lucc_net.indegree()
    lucc_oud = lucc_net.outdegree()
    lucc_betness = lucc_net.betweenness()
    lucc_yea = lucc_net.vs['year']
    # output vertex information in a csv file
    out_net_info = pd.DataFrame({
        'class': lucc_class,
        'year': lucc_yea,
        'degree': lucc_degree,
        'indegree': lucc_ind,
        'outdegree': lucc_oud,
        'betweenness': lucc_betness
    })
    out_net_info[
        'out_in'] = out_net_info['outdegree'] / out_net_info['indegree']
    out_net_info.to_csv(BaseConfig.OUT_PATH + 'LUCC_Net//' + p_topic_var +
                        '_im_multiple_network_vertex_info.csv')

    # get some property's mean value
    out_net_info_temp = out_net_info.replace(np.inf, np.nan).replace(0, np.nan)
    class_mean_degree = out_net_info_temp.groupby('class')['degree'].mean()
    class_mean_out_in = out_net_info_temp.groupby('class')['out_in'].mean()
    out_class_info = pd.DataFrame({
        'class_mean_degree': class_mean_degree,
        'class_mean_out_in': class_mean_out_in
    })
    out_class_info.to_csv(BaseConfig.OUT_PATH + 'LUCC_Net//' + p_topic_var +
                          '_im_multiple_network_class_info.csv')
    # plot this df as a fig
    fig = plt.figure(figsize=(10, 6), dpi=400)
    ax = fig.add_subplot(111)
    ax.plot(np.arange(p_timeSta - 1, p_timeEnd) + 0.5,
            out_sub_net_info['mean_shortest_path'],
            label='平均最短路径长度',
            color='k',
            linewidth=1.5,
            marker='o',
            ms=5,
            linestyle='-')
    ax.plot(np.arange(p_timeSta - 1, p_timeEnd) + 0.5,
            out_sub_net_info['density'],
            label='网络密度',
            color='k',
            linewidth=1.5,
            marker='^',
            ms=5,
            linestyle='-')
    ax.plot(np.arange(p_timeSta - 1, p_timeEnd) + 0.5,
            out_sub_net_info['transitivity_undirected'],
            label='网络传递性',
            color='k',
            linewidth=1.5,
            marker='s',
            ms=5,
            linestyle='-')

    ax.legend(loc='upper left', ncol=3, prop={
        'size': 14,
    }, frameon=False)
    ax.set_xlim(1985.5, 2018.5)
    # ax.set_ylim(-0.8, 1)
    ax.set_xticks(np.arange(p_timeSta - 1, p_timeEnd + 1))
    label_X = [
        '1986', '', '', '', '1990', '', '', '', '', '1995', '', '', '', '',
        '2000', '', '', '', '', '2005', '', '', '', '', '2010', '', '', '', '',
        '2015', '', '', '2018'
    ]
    ax.set_xticklabels(label_X, fontsize=14, fontfamily='Times New Roman')
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    ax.set_xlabel('年份', fontsize=14)
    ax.set_ylabel('值', fontsize=14)
    plt.savefig(BaseConfig.OUT_PATH + 'LUCC_Net//' + p_topic_var +
                '_fig_every_sub_net_every_year.png',
                bbox_inches='tight')
    return lucc_net


def cluster_net(net, p_topic_var):
    """
    cluster a net using some community clustering methods according to edge betweenness
    """
    # This is a method of community clustering method
    clustered_net = net.community_edge_betweenness(clusters=5,
                                                   directed=True,
                                                   weights='width')
    igraph.plot(clustered_net,
                BaseConfig.OUT_PATH + 'LUCC_Net//' + p_topic_var +
                'clustered_Net_Group.pdf',
                bbox=(1000, 1000),
                vertex_label=net.vs['name_en'],
                margin=100)
    igraph.plot(clustered_net.as_clustering(),
                BaseConfig.OUT_PATH + 'LUCC_Net//' + p_topic_var +
                'clustered_Net.pdf',
                bbox=(1000, 1000),
                margin=100)


if __name__ == "__main__":
    # There are two types topic var in the lucc_net
    # 'ChangeArea' and 'ChangeLA'

    # (1) Get the suitable threshold according to ChangeArea
    threshold_area = find_threshold(1987, 2018, 20)
    # (2) Build net for change area in a huge multiple net and output information every class
    chang_area_net = build_luc_net(1987, 2018, 'ChangeArea', threshold_area,
                                   'Year')
    # (3) Cluster net and output result
    cluster_net(chang_area_net, 'ChangeArea')
    # (4) Build net for change area in a continuous and im-multiple net and output information of every year net
    chang_area_net_no_mul = build_luc_net_no_mul(1987, 2018, 'ChangeArea',
                                                 threshold_area, 'Year')

    # The same as above
    chang_la_net = build_luc_net(1987, 2018, 'ChangeLA', threshold_area,
                                 'Sign')
    cluster_net(chang_la_net, 'ChangeLA')
    chang_la_net_no_mul = build_luc_net_no_mul(1987, 2018, 'ChangeLA',
                                               threshold_area, 'Sign')

# ***** This part is for pyunicorn *****
# lucc_net = network.Network(edge_list=series_df[['Source','Target']].values, directed=True)
# lucc_net.set_link_attribute('Change_Area',series_df['ChangeArea'].values)
# out_deg=lucc_net.betweenness()
# print(lucc_net)

# ***** This part is for big class *****
# LABEL_DIC = {
#     1: 'Forests',
#     2: 'Shrublands',
#     3: 'Grasslands',
#     4: 'Crop',
#     5: 'OT',
#     6: 'UB',
#     7: 'Water Bodies',
#     8: 'Desert and Low-vegetated lands',
# }
# COLOR_DIC = {
#     'Forests': '#1B7201',
#     'Shrublands': '#50FF00',
#     'Grasslands': '#A4D081',
#     'Crop': '#F2F100',
#     'OT': '#F29B00',
#     'UB': '#F00001',
#     'Water Bodies': '#0058F0',
#     'Desert and Low-vegetated lands': '#9F9F9F'
# }
# plt.savefig(BaseConfig.OUT_PATH + 'LUCC_Net_BigClass//hist.png')