from operator import truth
from setting import BaseConfig
from useful_class import GraphArtist
from useful_class import normalization
import numpy as np
import pandas as pd
import igraph
import pyunicorn.core.network as network
import matplotlib as mpl
mpl.use('cairo')
import matplotlib.pyplot as plt

LABEL_DIC = {
    101: 'DBF',
    102: 'ENF',
    103: 'MF',
    201: 'Shrub',
    301: 'LCG',
    302: 'MCG',
    303: 'HCG',
    401: 'Crop',
    402: 'OT',
    501: 'UB',
    601: 'Water',
    602: 'Wet',
    603: 'Snow',
    701: 'DB',
    702: 'LV',
}
COLOR_DIC = {
    'DBF':'#1B7201',
    'ENF':'#064A01',
    'MF':'#01B61F',
    'Shrub':'#50FF00',
    'LCG':'#A4D081',
    'MCG':'#88B75F',
    'HCG':'#6E9E45',
    'Crop':'#F2F100',
    'OT':'#F29B00',
    'UB':'#F00001',
    'Water':'#0058F0',
    'Wet':'#00E0F0',
    'Snow':'#BDF4F8',
    'DB':'#6A6A6A',
    'LV':'#CCCCCC'
}

def build_luc_net(p_timeSta, p_timeEnd):
    # SeriesData
    series_df = pd.DataFrame()
    lucc_net=igraph.Graph(directed=True)
    for o_class in list(LABEL_DIC.values()):
        lucc_net.add_vertex(o_class,color = COLOR_DIC[o_class],size=65,label=o_class,label_size=30)

    for year in np.arange(p_timeSta, p_timeEnd+1):
        this_year = pd.read_csv(BaseConfig.LUC_NET_DATA_PATH +
                                BaseConfig.LUC_NET_DATA_HEAD + str(year) + BaseConfig.LUC_NET_DATA_TAIL)
        source_label = this_year['Source'].map(LABEL_DIC)
        target_label = this_year['Target'].map(LABEL_DIC)
        this_year['Source_label'] = source_label
        this_year['Target_label'] = target_label
        this_year['Change_type'] = source_label + ' to ' + target_label
        this_year = this_year[['Source', 'Target', 'Source_label','Target_label','Year','Change_type','ChangeArea','ChangeLA']]
        series_df = series_df.append(this_year)
    # Filter this df (all edges)
    per_tile_20=np.percentile(series_df['ChangeArea'].values,30)
    series_df=series_df[(series_df["ChangeArea"]>per_tile_20)]
    # Set the edges
    tuples=[tuple(x) for x in series_df[['Source_label','Target_label']].values]
    lucc_net.add_edges(tuples)
    lucc_net.es['width'] = list(series_df['ChangeArea']*0.000000001)
    lucc_net.es['year'] = list(series_df['Year'])
    cmap = plt.get_cmap('viridis')
    norm = mpl.colors.Normalize(vmin=p_timeSta, vmax=p_timeEnd)
    sm = plt.cm.ScalarMappable(cmap=cmap,norm=norm)
    lucc_net.es['color'] = list(series_df['Year'].map(sm.to_rgba))
    lucc_net.es['curved'] = 0.1
    # lucc_net.es['curved'] = list(series_df['Year']*0.00025)
    lucc_net.es['arrow_size'] = 0.5


    lucc_cla=lucc_net.vs['label']
    lucc_deg=lucc_net.degree()
    lucc_bet=lucc_net.betweenness()
    lucc_net.vs['size']=normalization(lucc_deg)*100

    igraph.plot(lucc_net,BaseConfig.OUT_PATH + 'luccnet.pdf', bbox=(1000, 1000), margin=100,layout=lucc_net.layout('circle'))
    out_info_df = pd.DataFrame({'point':lucc_cla,'degree':lucc_deg,'betweenness':lucc_bet})
    out_info_df.to_csv(BaseConfig.OUT_PATH + 'PointInfo.csv')
    
    # In Matplotlib
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

def build_luc_net_no_mul(p_timeSta, p_timeEnd):
    # SeriesData
    series_df = pd.DataFrame()
    lucc_net=igraph.Graph(directed=True)

    for o_class in list(LABEL_DIC.values()):
        lucc_net.add_vertex(o_class+str(1986), label=o_class,color = COLOR_DIC[o_class],year=1986,size=40)
        for o_year in np.arange(p_timeSta, p_timeEnd+1):
            lucc_net.add_vertex(o_class+str(o_year), label=o_class,color = COLOR_DIC[o_class],year=o_year,size=40)
    for year in np.arange(p_timeSta, p_timeEnd+1):
        this_year = pd.read_csv(BaseConfig.LUC_NET_DATA_PATH +
                                BaseConfig.LUC_NET_DATA_HEAD + str(year) + BaseConfig.LUC_NET_DATA_TAIL)
        source_label = this_year['Source'].map(LABEL_DIC)
        target_label = this_year['Target'].map(LABEL_DIC)
        this_year['Source_label'] = source_label+(this_year['Year']-1).map(str)
        this_year['Target_label'] = target_label+this_year['Year'].map(str)
        this_year['Change_type'] = source_label + ' to ' + target_label
        this_year = this_year[['Source', 'Target', 'Source_label','Target_label','Year','Change_type','ChangeArea','ChangeLA']]
        series_df = series_df.append(this_year)
    # Filter this df (all edges)
    per_tile_20=np.percentile(series_df['ChangeArea'].values,50)
    series_df=series_df[(series_df["ChangeArea"]>per_tile_20)]
    # Set the edges
    tuples=[tuple(x) for x in series_df[['Source_label','Target_label']].values]
    lucc_net.add_edges(tuples)
    lucc_net.es['width'] = list(series_df['ChangeArea']*0.000000001)
    lucc_net.es['year'] = list(series_df['Year'])
    cmap = mpl.cm.viridis
    # cmap = plt.get_cmap('spring')
    norm = mpl.colors.Normalize(vmin=p_timeSta, vmax=p_timeEnd)
    sm = plt.cm.ScalarMappable(cmap=cmap,norm=norm)
    lucc_net.es['color'] = list(series_df['Year'].map(sm.to_rgba))
    lucc_net.es['curved'] = 0.1
    lucc_net.es['arrow_size'] = 0.5

    lucc_cla=lucc_net.vs['label']
    lucc_deg=lucc_net.degree()
    lucc_ind=lucc_net.indegree()
    lucc_oud=lucc_net.outdegree()
    lucc_bet=lucc_net.betweenness()
    lucc_yea=lucc_net.vs['year']
    # lucc_net.vs['size']=normalization(lucc_deg)*100

    igraph.plot(lucc_net,BaseConfig.OUT_PATH + 'luccnet_no_mul.pdf', bbox=(2500, 1000), margin=100,layout=lucc_net.layout_grid(width=(p_timeEnd-p_timeSta+2),dim=2))
    out_info_df = pd.DataFrame({'class':lucc_cla,
                                'year':lucc_yea,
                                'degree':lucc_deg,
                                'indegree':lucc_ind,
                                'outdegree':lucc_oud,
                                'betweenness':lucc_bet})
    out_info_df.to_csv(BaseConfig.OUT_PATH + 'PointInfo_no_mul.csv')
    return lucc_net


if __name__ == "__main__":
    build_luc_net_no_mul(1987,2009)
    build_luc_net(1987,2009)

# lucc_net = network.Network(edge_list=series_df[['Source','Target']].values, directed=True)
# lucc_net.set_link_attribute('Change_Area',series_df['ChangeArea'].values)
# out_deg=lucc_net.betweenness()
# print(lucc_net)