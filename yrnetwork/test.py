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
from tigramite import plotting as tp

# def build_link_pcmci_noself(p_data_values, p_agent_names, p_var_sou,
# p_var_tar):
dataaa = pd.read_csv('E:\\MY PROGRAM\\YR_Greening_Network\\data\\selfNetSample.csv')
p_data_values = dataaa.values
p_agent_names = [
    'LAI', 'Runoff', 'ETVegetation', 'Pressure', 'AirTemperature',
    'SoilTemperature', 'SoilWater', 'VS', 'Precipitation'
]
p_var_sou = '--'
p_var_tar = '--'

[times_num, agent_num] = p_data_values.shape
# set the data for PCMCI
data_frame = pp.DataFrame(p_data_values,
                          var_names=p_agent_names,
                          missing_flag=BaseConfig.BACKGROUND_VALUE)
# new PCMCI
pcmci = PCMCI(dataframe=data_frame,
              cond_ind_test=ParCorr(significance='analytic'))

# correlations = pcmci.run_bivci(tau_max=10, val_only=True)['val_matrix']
# lag_func_matrix = tp.plot_lagfuncs(val_matrix=correlations,
#                                    setup_args={'var_names':p_agent_names, 'figsize':(15, 10),
#                                     'x_base':5, 'y_base':.5})

# run PCMCI
alpha_level = None
results_pcmci = pcmci.run_pcmciplus(tau_min=0, tau_max=2, pc_alpha=alpha_level)
# get the result
graph_pcmci = results_pcmci['graph']
q_matrix = results_pcmci['q_matrix']
p_matrix = results_pcmci['p_matrix']
val_matrix = results_pcmci['val_matrix']
conf_matrix = results_pcmci['conf_matrix']
ambiguous_triples = results_pcmci['ambiguous_triples']

# Only draw link in one direction among contemp
# Remove lower triangle
# print(graph_pcmci)
# print(graph_pcmci.shape)

# link_matrix_upper = np.copy(graph_pcmci)
# link_matrix_upper[:, :, 0] = np.triu(link_matrix_upper[:, :, 0])
# graph_pcmci = link_matrix_upper

# print(graph_pcmci)
# print('------------------')
# print(link_matrix_upper)

# # net = _get_absmax(link_matrix != "")
# net = np.any(link_matrix_upper != "", axis=2)
# print(net)
# G = nx.DiGraph(net)

pcmci.print_significant_links(p_matrix=results_pcmci['p_matrix'],
                              q_matrix=q_matrix,
                              val_matrix=results_pcmci['val_matrix'],
                              alpha_level=0.01)

# filter these links
links_df = pd.DataFrame(columns=('VarSou', 'VarTar', 'Source', 'Target',
                                 'TimeLag', 'Strength', 'Unoriented'))

if graph_pcmci is not None:
    sig_links = (graph_pcmci != "") * (graph_pcmci != "<--")
    # sig_links = (graph_pcmci != "")
# q_matrix is the corrected p_matrix
# elif q_matrix is not None:
#     sig_links = (q_matrix <= 0.01)
# else:
# sig_links = (p_matrix <= 0.01)
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
        Unoriented = 0
        if graph_pcmci is not None:
            if graph_pcmci[j, p[0], 0] == "o-o":
                Unoriented = 1
                # "unoriented link"
            elif graph_pcmci[p[0], j, abs(p[1])] == "x-x":
                Unoriented = 1
                # "unclear orientation due to conflict"
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
print(links_df.sort_values(by=['Source', 'Target']))

# print(graph_pcmci)

tp.plot_graph(
    val_matrix=results_pcmci['val_matrix'],
    link_matrix=results_pcmci['graph'],
    var_names=p_agent_names,
    link_colorbar_label='cross-MCI (edges)',
    node_colorbar_label='auto-MCI (nodes)',
)
plt.show()