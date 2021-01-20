
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