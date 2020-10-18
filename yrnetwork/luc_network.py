import setting
import GraphArtist
import numpy as np
import pandas as pd
import igraph
import pyunicorn.core.network as network
import matplotlib as mpl
mpl.use('cairo')
import matplotlib.pyplot as plt



def build_luc_net(p_timeSta, p_timeEnd):
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
        'DB':'#000000',
        'LV':'#9F9F9F'
    }


    # SeriesData
    series_df = pd.DataFrame()
    lucc_net=igraph.Graph(directed=True)
    for o_class in list(LABEL_DIC.values()):
        lucc_net.add_vertex(o_class,color = COLOR_DIC[o_class],size=50,label=o_class)

    for year in np.arange(p_timeSta, p_timeEnd+1):
        this_year = pd.read_csv(setting.BaseConfig.LUC_NET_DATA_PATH +
                                setting.BaseConfig.LUC_NET_DATA_HEAD + str(year) + setting.BaseConfig.LUC_NET_DATA_TAIL)
        source_label = this_year['Source'].map(LABEL_DIC)
        target_label = this_year['Target'].map(LABEL_DIC)
        this_year['Source_label'] = source_label
        this_year['Target_label'] = target_label
        this_year['Change_type'] = source_label + ' to ' + target_label
        this_year = this_year[['Source', 'Target', 'Source_label','Target_label','Year','Change_type','ChangeArea','ChangeLA']]
        series_df = series_df.append(this_year)
    tuples=[tuple(x) for x in series_df[['Source_label','Target_label']].values]
    lucc_net.add_edges(tuples)
    lucc_net.es['width'] = list(series_df['ChangeArea']*0.000000001)
    lucc_net.es['year'] = list(series_df['Year'])
    cmap = plt.get_cmap('cool')
    norm = mpl.colors.Normalize(vmin=p_timeSta, vmax=p_timeEnd)
    sm = plt.cm.ScalarMappable(cmap=cmap,norm=norm)
    lucc_net.es['color'] = list(series_df['Year'].map(sm.to_rgba))
    lucc_net.es['arrow_size'] = 0.8

    igraph.plot(lucc_net, bbox=(1000, 1000), margin=100,layout=lucc_net.layout('circular'))
    
    # In Matplotlib
    # import matplotlib as mpl
    # mpl.use('cairo')
    # import matplotlib.pyplot as plt
    # fig=plt.figure()
    # axes=fig.add_subplot(111)
    # graph_artist=GraphArtist.GraphArtist(lucc_net,(10, 10, 150, 150),layout=layout)
    # graph_artist.set_zorder(float('inf'))
    # axes.artists.append(graph_artist)
    # plt.axis("off")
    # fig.suptitle('sdf')
    # fig.savefig(setting.BaseConfig.OUT_PATH + 'luccnet.pdf')

    return lucc_net

def build_luc_net_no_mul(p_timeSta, p_timeEnd):
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
        'DBF':'#7109AA',
        'ENF':'#5F2580',
        'MF':'#9F3ED5',
        'Shrub':'#E667AF',
        'LCG':'#E6399B',
        'MCG':'#412C84',
        'HCG':'#6A48D7',
        'Crop':'#B9F73E',
        'OT':'#FFFF73',
        'UB':'#FFFF00',
        'Water':'#CD0074',
        'Wet':'#679B00',
        'Snow':'#FFDE40',
        'DB':'#AD66D5',
        'LV':'#7109AA'
    }

    # SeriesData
    series_df = pd.DataFrame()
    lucc_net=igraph.Graph(directed=True)
    for o_class in list(LABEL_DIC.values()):
        for o_year in np.arange(p_timeSta, p_timeEnd+1):
            lucc_net.add_vertex(o_class+str(o_year), label=o_class,change_year=o_year,color = COLOR_DIC[o_class])


    for year in np.arange(p_timeSta, p_timeEnd+1):
        this_year = pd.read_csv(setting.BaseConfig.LUC_NET_DATA_PATH +
                                setting.BaseConfig.LUC_NET_DATA_HEAD + str(year) + setting.BaseConfig.LUC_NET_DATA_TAIL)
        source_label = this_year['Source'].map(LABEL_DIC)
        target_label = this_year['Target'].map(LABEL_DIC)
        this_year['Source_label'] = source_label+this_year['Year'].map(str)
        this_year['Target_label'] = target_label+this_year['Year'].map(str)
        this_year['Change_type'] = source_label + ' to ' + target_label
        this_year = this_year[['Source', 'Target', 'Source_label','Target_label','Change_type','ChangeArea','ChangeLA']]
        series_df = series_df.append(this_year)
    tuples=[tuple(x) for x in series_df[['Source_label','Target_label']].values]
    lucc_net.add_edges(tuples)
    lucc_net.es['width'] = list(series_df['ChangeArea']*0.000000001)
    lucc_net.es['color'] = [COLOR_DIC[e.source_vertex['label']] for e in lucc_net.es]

    # lucc_net = network.Network(edge_list=series_df[['Source','Target']].values, directed=True)
    # lucc_net.set_link_attribute('Change_Area',series_df['ChangeArea'].values)
    # out_deg=lucc_net.betweenness()
    # print(lucc_net)
    layout = lucc_net.layout('kk')
    igraph.plot(lucc_net, layout=layout)
    return lucc_net


if __name__ == "__main__":
    build_luc_net(1987,1990)
