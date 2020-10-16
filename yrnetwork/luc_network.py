import setting
import numpy as np
import pandas as pd
import igraph
import pyunicorn.core.network as network


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

    # SeriesData
    series_df = pd.DataFrame()
    lucc_net=igraph.Graph(directed=True)

    for year in np.arange(p_timeSta, p_timeEnd+1):
        this_year = pd.read_csv(setting.BaseConfig.LUC_NET_DATA_PATH +
                                setting.BaseConfig.LUC_NET_DATA_HEAD + str(year) + setting.BaseConfig.LUC_NET_DATA_TAIL)
        source_label = this_year['Source'].map(LABEL_DIC)
        target_label = this_year['Target'].map(LABEL_DIC)
        this_year['Source_label'] = source_label
        this_year['TargetLabel'] = target_label
        this_year['ChangeType'] = this_year['Source_label'] + ' to ' + this_year['TargetLabel']
        this_year = this_year[['Source', 'Target', 'Source_label','TargetLabel','ChangeType','ChangeArea']]


        lucc_net.add_vertices(list(LABEL_DIC.values()))
        tuples=[tuple(x) for x in this_year[['Source_label','TargetLabel']].values]
        print(tuples)
        lucc_net.add_edges(tuples)


        # series_df = series_df.append(this_year)
    # lucc_net = network.Network(edge_list=series_df[['Source','Target']].values, directed=True)
    # lucc_net.set_link_attribute('Change_Area',series_df['ChangeArea'].values)
    # out_deg=lucc_net.betweenness()
    
    print(lucc_net)
    layout = lucc_net.layout()
    igraph.plot(lucc_net)
    return lucc_net


if __name__ == "__main__":
    build_luc_net(1987,1988)
