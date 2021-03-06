from setting import *
import math
import time
import igraph
import numpy as np
import pandas as pd
import scipy.stats as st
import multiprocessing
from sklearn import preprocessing
from geopy.distance import geodesic
from scipy import ndimage
from scipy.integrate import dblquad
from scipy.stats import gaussian_kde

# Input Data Var
VARS_LIST = ['precipitation', 'pressure', 'temperature']
# Load data
# Data like
# GA_ID  Time1       Time2       Time3       Time4 ...
# [int]  [double]    [double]    [double]    [double]
# 00000  0.001       0.002       0.67        1.34
# 00001  0.003       0.022       0.69        2.34
# ...    ...         ...         ...         ...
# This data must have the same index with GA_Centroid

# The time scale of vars
VARS_TIME_SCALE_DICT = {
    'precipitation': 'monthly_yearly',
    'pressure': 'yearly',
    'temperature': 'monthly_yearly'
}

VAR_COLOR_DICT = {
    'precipitation': '#0023BC',
    'pressure': '#00BC23',
    'temperature': '#660000'
}


def build_edges():
    """
    build edges from csv data
    """
    # Load GeoAgent centroid info
    centroid_data = pd.read_csv(BaseConfig.GEO_AGENT_PATH + 'GA_Centroid.csv')

    # set multiprocessing to get links
    new_pool = multiprocessing.Pool()
    jobs = []
    var_index = 0
    for var in VARS_LIST:
        sec_var_index = 0
        for sec_var in VARS_LIST:
            if var == sec_var:
                p = new_pool.apply_async(get_inner_links, args=(var, ))
                jobs.append(p)
                print('get_inner_links', var)
            else:
                if sec_var_index < var_index:
                    continue
                p = new_pool.apply_async(get_mult_links, args=(var, sec_var))
                jobs.append(p)
                print('get_mult_links', var, sec_var)
            sec_var_index = sec_var_index + 1
        var_index = var_index + 1
    new_pool.close()
    new_pool.join()
    # get links from jobs
    kinds_links = []
    for job in jobs:
        kinds_links.append(job.get())
    print(kinds_links)
    print('already complete all links over')
    # save all the Links to a csv
    all_links = pd.concat(kinds_links, ignore_index=True)
    print('already concat pd')
    # Add distance
    all_links_dis = get_geo_distance(all_links, centroid_data)
    all_links_dis.to_csv(BaseConfig.OUT_PATH + 'Coupled_Network\\AllLinks.csv')
    print('already add distance and output')
    print('GOOD!')


def build_coupled_network():
    """
    build coupled network by the edges df from csv
    """
    all_edges_df = pd.read_csv(BaseConfig.OUT_PATH +
                               'Coupled_Network\\AllLinks.csv')
    all_edges_filtered = filter_edges(all_edges_df)
    all_edges_filtered.to_csv(BaseConfig.OUT_PATH +
                              'Coupled_Network\\AllLinks_filtered.csv')
    print(all_edges_filtered)
    # build a net network
    coupled_network = igraph.Graph(directed=True)
    # add every vertex to the net
    var_sou = all_edges_filtered['VarSou'].map(str)
    var_tar = all_edges_filtered['VarTar'].map(str)
    id_sou = all_edges_filtered['Source'].map(str)
    id_tar = all_edges_filtered['Target'].map(str)
    all_edges_filtered['Source_label'] = id_sou + '_' + var_sou
    all_edges_filtered['Target_label'] = id_tar + '_' + var_tar
    all_ver_list = list(all_edges_filtered['Source_label']) + list(
        all_edges_filtered['Target_label'])
    # set the unique of the vertexs
    ver_list_unique = list(set(all_ver_list))
    for v_id_var in ver_list_unique:
        coupled_network.add_vertex(
            v_id_var,
            var_name=v_id_var.split('_')[1],
            ga_id=v_id_var.split('_')[0],
            label=v_id_var.split('_')[0],
            size=30,
            color=VAR_COLOR_DICT[v_id_var.split('_')[1]],
            label_size=15)
    # set all edges
    tuples_es = [
        tuple(x)
        for x in all_edges_filtered[['Source_label', 'Target_label']].values
    ]
    coupled_network.add_edges(tuples_es)
    coupled_network.es['VarSou'] = list(all_edges_filtered['VarSou'])
    coupled_network.es['VarTar'] = list(all_edges_filtered['VarTar'])
    coupled_network.es['width'] = list(
        abs(all_edges_filtered['Correlation_W'] * 1))
    igraph.plot(coupled_network,
                BaseConfig.OUT_PATH + 'Coupled_Network//Coupled_Network.pdf',
                bbox=(1200, 1200),
                layout=coupled_network.layout('large'),
                margin=200)
    coupled_net_noinner = remove_inner_net(coupled_network)
    igraph.plot(coupled_net_noinner,
                BaseConfig.OUT_PATH +
                'Coupled_Network//Coupled_Network_noInner.pdf',
                bbox=(1200, 1200),
                margin=200)


# ******SubFunction******
def z_score_normaliz(p_data):
    """
    normalize data by Z-Score
    @param p_data:
    @return: nomalized data
    """
    z_score = preprocessing.StandardScaler()
    data_zs = z_score.fit_transform(p_data)
    return data_zs


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
    data_values = data.values
    id_data = data_values[..., 0].astype(np.int32)
    data_values = np.delete(data_values, 0, axis=1)
    [agent_num, data_num] = data_values.shape
    data_values_nor = z_score_normaliz(data_values)
    inner_links = pd.DataFrame(columns=('VarSou', 'VarTar', 'Source', 'Target',
                                        'Pear_r', 'Pear_p', 'Correlation_C',
                                        'Correlation_W', 'MutualInfo'))
    for i_agent in np.arange(0, agent_num):
        this_agent = data_values_nor[i_agent, ...]
        other_agent = np.delete(data_values_nor, i_agent, axis=0)
        other_id = np.delete(id_data, i_agent, axis=0)
        for i_other in np.arange(0, agent_num - 1):
            [pear_r, pear_p, correlation_c, correlation_w,
             mutual_info] = build_link(this_agent, other_agent[i_other, ...],
                                       data_num)
            inner_links = inner_links.append(pd.DataFrame({
                'VarSou': [p_var_name],
                'VarTar': [p_var_name],
                'Source': [id_data[i_agent]],
                'Target': [other_id[i_other]],
                'Pear_r': [pear_r],
                'Pear_p': [pear_p],
                'Correlation_C': [correlation_c],
                'Correlation_W': [correlation_w],
                'MutualInfo': [mutual_info],
            }),
                                             ignore_index=True)
    print('Over a Inner Links')
    return inner_links


def get_mult_links(p_var_sou, p_var_tar):
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
        data_tar = pd.read_csv(BaseConfig.COUPLED_NET_DATA_PATH +
                               BaseConfig.COUPLED_NET_DATA_HEAD + p_var_tar +
                               '_monthly' + BaseConfig.COUPLED_NET_DATA_TAIL)
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
        data_tar = pd.read_csv(BaseConfig.COUPLED_NET_DATA_PATH +
                               BaseConfig.COUPLED_NET_DATA_HEAD + p_var_tar +
                               '_yearly' + BaseConfig.COUPLED_NET_DATA_TAIL)
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
    sou_values_nor = z_score_normaliz(sou_values)
    tar_values_nor = z_score_normaliz(tar_values)
    [agent_num, data_num] = sou_values.shape
    mult_links = pd.DataFrame(columns=('VarSou', 'VarTar', 'Source', 'Target',
                                       'Pear_r', 'Pear_p', 'Correlation_C',
                                       'Correlation_W', 'MutualInfo'))
    for i_sou in np.arange(0, agent_num):
        this_sou = sou_values_nor[i_sou, ...]
        for i_tar in np.arange(0, agent_num - 1):
            [pear_r, pear_p, correlation_c, correlation_w,
             mutual_info] = build_link(this_sou, tar_values_nor[i_tar, ...],
                                       data_num)
            mult_links = mult_links.append(pd.DataFrame({
                'VarSou': [p_var_sou],
                'VarTar': [p_var_tar],
                'Source': [id_sou[i_sou]],
                'Target': [id_tar[i_tar]],
                'Pear_r': [pear_r],
                'Pear_p': [pear_p],
                'Correlation_C': [correlation_c],
                'Correlation_W': [correlation_w],
                'MutualInfo': [mutual_info],
            }),
                                           ignore_index=True)
    print('Over a Mult Links')
    return mult_links


def build_link(p_data_i, p_data_j, p_data_num):
    """
    build links by 2 column data
    @param p_data_i: p_data_i
    @param p_data_j: p_data_j
    @param p_data_num: p_data_num
    @return:
    """
    # ******Pearsonr
    Rpear, Ppear = st.pearsonr(p_data_i, p_data_j)

    # ******cross_cor
    maxTimeLag = int(np.round(p_data_num * 0.7))
    std_i = p_data_i.std()
    mean_i = p_data_i.mean()
    C_taus = []
    for iTimeLag in np.arange(1, maxTimeLag):
        Tj_tau = p_data_j[np.hstack(
            [range(iTimeLag, p_data_num),
             range(0, iTimeLag)])]
        std_tau = Tj_tau.std()
        mean_tau = Tj_tau.mean()
        Ti_m_Ttau = p_data_i * Tj_tau
        mean_Ti_m_Ttau = Ti_m_Ttau.mean()
        Cij_tau = (mean_Ti_m_Ttau - mean_i * mean_tau) / (std_i * std_tau)
        C_taus = np.hstack([C_taus, Cij_tau])
    Cij = C_taus.max()
    # minCs=C_taus.min()
    # maxCs = C_taus.max()
    # if abs(minCs)<abs(maxCs):
    #     Cij = maxCs
    # else:
    #     Cij = minCs
    mean_C_taus = C_taus.mean()
    std_C_taus = C_taus.std()
    Wij = (abs(Cij) - mean_C_taus) / std_C_taus

    # ******MutualInformation
    MI = mutual_information_2d(p_data_i, p_data_j, 64, 1, True)

    # # Constants
    # MIN_DOUBLE = 4.9406564584124654e-324
    #                     # The minimum size of a Float64; used here to prevent the
    #                     #  logarithmic function from hitting its undefined region
    #                     #  at its asymptote of 0.
    # INF = float('inf')  # The floating-point representation for "infinity"
    # # Kernel estimation
    # gkde_i = gaussian_kde(Ti)
    # gkde_j = gaussian_kde(Tj)
    # gkde_ij = gaussian_kde([Ti, Tj])
    # print(gkde_i(Ti.min()))
    # print(gkde_i(Ti.max()))
    # print(gkde_j(Tj.min()))
    # print(gkde_j(Tj.max()))
    # print(gkde_ij([Ti.min(),Tj.min()]))
    # mutual_info = lambda a, b: gkde_ij([a, b]) * math.log((gkde_ij([a, b]) / (gkde_i(a) * gkde_j(b))) + MIN_DOUBLE)
    # print('cal one')
    # for target_list in expression_list:
    #     pass
    # (minfo_xy, err_xy) = dblquad(mutual_info, Ti.min(), Ti.max(), lambda a: Tj.min(), lambda a: Tj.max())
    # ## https://stackoverflow.com/questions/8363085/continuous-mutual-information-in-python/8363237

    return Rpear, Ppear, Cij, Wij, MI


def mutual_information_2d(x, y, binNum=64, sigma=1, normalized=False):
    # cite https://gist.github.com/GaelVaroquaux/ead9898bd3c973c40429
    """
    Computes (normalized) mutual information between two 1D variate from a
    joint histogram.
    Parameters
    ----------
    x : 1D array
        first variable
    y : 1D array
        second variable
    sigma: float
        sigma for Gaussian smoothing of the joint histogram
    Returns
    -------
    nmi: float
        the computed similariy measure
    """

    EPS = np.finfo(float).eps

    bins = (binNum, binNum)

    jh = np.histogram2d(x, y, bins=bins)[0]

    # smooth the jh with a gaussian filter of given sigma
    ndimage.gaussian_filter(jh, sigma=sigma, mode='constant', output=jh)
    # compute marginal histograms
    jh = jh + EPS
    sh = np.sum(jh)
    jh = jh / sh
    s1 = np.sum(jh, axis=0).reshape((-1, jh.shape[0]))
    s2 = np.sum(jh, axis=1).reshape((jh.shape[1], -1))

    # Normalised Mutual Information of:
    # Studholme,  jhill & jhawkes (1998).
    # "A normalized entropy measure of 3-D medical image alignment".
    # in Proc. Medical Imaging 1998, vol. 3338, San Diego, CA, pp. 132-143.
    if normalized:
        mi = ((np.sum(s1 * np.log2(s1)) + np.sum(s2 * np.log2(s2))) /
              np.sum(jh * np.log2(jh))) - 1
    else:
        mi = (np.sum(jh * np.log2(jh)) - np.sum(s1 * np.log2(s1)) -
              np.sum(s2 * np.log2(s2)))
    return mi


def filter_edges(p_edges):
    """
    filter edges by some rules
    @param p_edges: edges
    @return:
    """
    pear_r = p_edges.loc[:, "Pear_r"]
    pear_p = p_edges.loc[:, "Pear_p"]
    correlation_c = p_edges.loc[:, "Correlation_C"]
    correlation_w = p_edges.loc[:, "Correlation_W"]
    mutual_info = p_edges.loc[:, "MutualInfo"]
    Cdes = correlation_c.describe(percentiles=[0.5]).loc['50%']
    Wdes = correlation_w.describe(percentiles=[0.5]).loc['50%']
    Mdes = mutual_info.describe(percentiles=[0.5]).loc['50%']
    # Filter the Links
    # 1 Pear_p<1e-10
    # 2 Correlation_C>Cdes50
    # 3 Correlation_W>Wdes50
    # 4 MutualInfo>Mdes50
    filtered_edges = p_edges[(p_edges["Pear_p"] < 1e-10)
                             & (p_edges["Correlation_C"] > Cdes)
                             & (p_edges["Correlation_W"] > Wdes)
                             & (p_edges["MutualInfo"] > Mdes)].copy()
    return filtered_edges


def get_subgraph(p_father_net, p_var_name):
    """
    get subgraph by the var name
    """
    # get vs
    selected_vs = p_father_net.vs.select(var_name=p_var_name)
    subgraph = p_father_net.induced_subgraph(selected_vs)
    return subgraph


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


# Run Code
if __name__ == "__main__":
    print(time.strftime('%H:%M:%S', time.localtime(time.time())))
    # build_edges()
    print(time.strftime('%H:%M:%S', time.localtime(time.time())))
    build_coupled_network()
    print(time.strftime('%H:%M:%S', time.localtime(time.time())))
