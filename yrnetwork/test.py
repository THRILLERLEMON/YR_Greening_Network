import numpy as np
import pandas as pd
from setting import *
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr
import tigramite.data_processing as pp

def test1():
    # Data must be array of shape (time, variables)
    data_source = pd.read_csv(BaseConfig.COUPLED_NET_DATA_PATH +
                            BaseConfig.COUPLED_NET_DATA_HEAD + 'temperature' +
                            '_monthly_for_causal' + BaseConfig.COUPLED_NET_DATA_TAIL,index_col='GA_ID')
    data_souT = pd.DataFrame(data_source.values.T,
                            index=data_source.columns,
                            columns=data_source.index)
    data_V = data_souT.values
    T, N = data_V.shape
    var_number = np.arange(1, N + 1)
    var_names = list(map(str, var_number))
    # var_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'G']
    dataframe = pp.DataFrame(data_V, var_names=var_names)
    cond_ind_test = ParCorr()
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test)
    results = pcmci.run_pcmciplus(tau_min=0, tau_max=2, pc_alpha=0.01)
    print(results)

    graph = results['graph']
    q_matrix = results['q_matrix']
    p_matrix = results['p_matrix']
    val_matrix = results['val_matrix']
    conf_matrix = results['conf_matrix']
    ambiguous_triples = results['ambiguous_triples']

    alpha_level = 0.05

    if graph is not None:
        sig_links = (graph != "") * (graph != "<--")
    elif q_matrix is not None:
        sig_links = (q_matrix <= alpha_level)
    else:
        sig_links = (p_matrix <= alpha_level)

    inner_links = pd.DataFrame()

    for j in range(N):
        links = {(p[0], -p[1]): np.abs(val_matrix[p[0], j, abs(p[1])])
                for p in zip(*np.where(sig_links[:, j, :]))}
        # Sort by value
        sorted_links = sorted(links, key=links.get, reverse=True)
        for p in sorted_links:
            VarSou = 'p_var_name'
            VarTar = 'p_var_name'
            Source = var_names[j]
            Target = var_names[p[0]]
            lag = p[1]
            weight = val_matrix[p[0], j, abs(p[1])]
            unoriented = None
            if graph is not None:
                if p[1] == 0 and graph[j, p[0], 0] == "o-o":
                    unoriented = 1
                    # "unoriented link"
                elif graph[p[0], j, abs(p[1])] == "x-x":
                    unoriented = 1
                    # "unclear orientation due to conflict"
                else:
                    unoriented = 0
            inner_links = inner_links.append(pd.DataFrame({
                'VarSou': [VarSou],
                'VarTar': [VarTar],
                'Source': [Source],
                'Target': [Target],
                'lag': [int(lag)],
                'weight': [weight],
                'unoriented': [int(unoriented)]
            }),
                                            ignore_index=True)
    print(inner_links)


def test2():
    data_source = pd.read_csv(BaseConfig.COUPLED_NET_DATA_PATH +
                            BaseConfig.COUPLED_NET_DATA_HEAD + 'temperature' +
                            '_monthly_for_causal' + BaseConfig.COUPLED_NET_DATA_TAIL,index_col='GA_ID')
    data_source=data_source.head(2)
    data_souT = pd.DataFrame(data_source.values.T,
                            index=data_source.columns,
                            columns=data_source.index)

    print(data_souT)
    data_data=data_souT.values
    T, N = data_data.shape
    var_number = np.arange(1, N + 1)
    var_names = list(map(str, var_number))
    dataframe = pp.DataFrame(data_data, var_names=var_names)
    cond_ind_test = ParCorr()
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test)
    results = pcmci.run_pcmciplus(tau_min=0, tau_max=2, pc_alpha=0.01)
    print(results)

    graph = results['graph']
    q_matrix = results['q_matrix']
    p_matrix = results['p_matrix']
    val_matrix = results['val_matrix']
    conf_matrix = results['conf_matrix']
    ambiguous_triples = results['ambiguous_triples']

    alpha_level = 0.05

    if graph is not None:
        sig_links = (graph != "") * (graph != "<--")
    elif q_matrix is not None:
        sig_links = (q_matrix <= alpha_level)
    else:
        sig_links = (p_matrix <= alpha_level)

    inner_links = pd.DataFrame()

    for j in range(N):
        links = {(p[0], -p[1]): np.abs(val_matrix[p[0], j, abs(p[1])])
                for p in zip(*np.where(sig_links[:, j, :]))}
        # Sort by value
        sorted_links = sorted(links, key=links.get, reverse=True)
        for p in sorted_links:
            VarSou = 'p_var_name'
            VarTar = 'p_var_name'
            Source = var_names[j]
            Target = var_names[p[0]]
            lag = p[1]
            weight = val_matrix[p[0], j, abs(p[1])]
            unoriented = None
            if graph is not None:
                if p[1] == 0 and graph[j, p[0], 0] == "o-o":
                    unoriented = 1
                    # "unoriented link"
                elif graph[p[0], j, abs(p[1])] == "x-x":
                    unoriented = 1
                    # "unclear orientation due to conflict"
                else:
                    unoriented = 0
            inner_links = inner_links.append(pd.DataFrame({
                'VarSou': [VarSou],
                'VarTar': [VarTar],
                'Source': [Source],
                'Target': [Target],
                'lag': [int(lag)],
                'weight': [weight],
                'unoriented': [int(unoriented)]
            }),
                                            ignore_index=True)
    print(inner_links)







test2()







# for j in range(N):
#     links = {(p[0], -p[1]): np.abs(val_matrix[p[0], j, abs(p[1])])
#                 for p in zip(*np.where(sig_links[:, j, :]))}
#     # Sort by value
#     sorted_links = sorted(links, key=links.get, reverse=True)
#     n_links = len(links)
#     string = ("\n    Variable %s has %d "
#                 "link(s):" % (var_names[j], n_links))
#     for p in sorted_links:
#         string += ("\n        (%s % d): pval = %.5f" %
#                     (var_names[p[0]], p[1],
#                     p_matrix[p[0], j, abs(p[1])]))
#         if q_matrix is not None:
#             string += " | qval = %.5f" % (
#                 q_matrix[p[0], j, abs(p[1])])
#         string += " | val = % .3f" % (
#             val_matrix[p[0], j, abs(p[1])])
#         if conf_matrix is not None:
#             string += " | conf = (%.3f, %.3f)" % (
#                 conf_matrix[p[0], j, abs(p[1])][0],
#                 conf_matrix[p[0], j, abs(p[1])][1])
#         if graph is not None:
#             if p[1] == 0 and graph[j, p[0], 0] == "o-o":
#                 string += " | unoriented link"
#             if graph[p[0], j, abs(p[1])] == "x-x":
#                 string += " | unclear orientation due to conflict"
#     print(string)

# link_marker = {True:"o-o", False:"-->"}

# if ambiguous_triples is not None and len(ambiguous_triples) > 0:
#     print("\n## Ambiguous triples:\n")
#     for triple in ambiguous_triples:
#         (i, tau), k, j = triple
#         print("    (%s % d) %s %s o-o %s" % (
#             var_names[i], tau, link_marker[tau==0],
#             var_names[k],
#             var_names[j]))
