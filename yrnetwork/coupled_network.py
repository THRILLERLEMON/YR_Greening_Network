#TODO THRILLER:多要素构建网络
#TODO THRILLER:需要实现不同时间段不同时间尺度的关系识别


# fake code
def build_coupled_net(p_A_var, p_B_var, p_time_scale, p_time_start,
                      p_time_end):
    """
    build a coupled net for A and B
    p_time_scale='yearly'
    p_time_start,p_time_end like 1986 and 2018
    p_time_scale='yearly'
    p_time_start,p_time_end like 198601 and 201812
    """
    pass


from setting import *
import math
import time
import numpy as np
import pandas as pd
import scipy.stats as st
import multiprocessing
from geopy.distance import geodesic
from scipy import ndimage
from scipy.integrate import dblquad
from scipy.stats import gaussian_kde

# Input Data
VARS_LIST = ['precipitation', 'pressure', 'temperature']
# Load data
# Data like
# label  Time1       Time2       Time3       Time4 ...
# [int]  [double]    [double]    [double]    [double]
# 00000  0.001       0.002       0.67        1.34
# 00001  0.003       0.022       0.69        2.34
# ...    ...         ...         ...         ...
# This data must have the same index with pointInfo


def main():
    # ******Main******
    print(time.strftime('%H:%M:%S', time.localtime(time.time())))
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
    all_links.to_csv(BaseConfig.OUT_PATH + 'Coupled_Network\\AllLinks.csv')
    print('already concat pd')
    # Filter links and add distance
    all_links_filtered = filter_links(all_links)
    all_links_filtered_dis = get_geo_distance(all_links_filtered,
                                              centroid_data)
    all_links_filtered_dis.to_csv(BaseConfig.OUT_PATH +
                                  'Coupled_Network\\AllLinksFiltered.csv')
    print('already filter links & add distance')

    print(time.strftime('%H:%M:%S', time.localtime(time.time())))
    print('GOOD!')


# ******SubFunction******
def normalize(p_data):
    """
    normalize data to -1~1
    @param p_data:
    @return: nomalized data
    """
    m = np.mean(p_data)
    mx = max(p_data)
    mn = min(p_data)
    return (p_data - m) / (mx - mn)


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
    data = pd.read_csv(BaseConfig.COUPLED_NET_DATA_PATH +
                       BaseConfig.COUPLED_NET_DATA_HEAD + p_var_name +
                       BaseConfig.COUPLED_NET_DATA_TAIL)
    data_values = data.values
    id_data = data_values[..., 0].astype(np.int32)
    data_values = np.delete(data_values, 0, axis=1)
    [agent_num, data_num] = data_values.shape
    data_values_nor = normalize(data_values)
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
    data_sou = pd.read_csv(VARS_LIST[p_var_sou])
    data_tar = pd.read_csv(VARS_LIST[p_var_tar])
    data_sou_values = data_sou.values
    data_tar_values = data_tar.values
    id_sou = data_sou_values[..., 0].astype(np.int32)
    id_tar = data_tar_values[..., 0].astype(np.int32)
    sou_values = np.delete(data_sou_values, 0, axis=1)
    tar_values = np.delete(data_tar_values, 0, axis=1)
    sou_values_nor = normalize(sou_values)
    tar_values_nor = normalize(tar_values)
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


def filter_links(p_links):
    """
    filter links by some rules
    @param Links: links
    @return:
    """
    pear_r = p_links.loc[:, "Pear_r"]
    pear_p = p_links.loc[:, "Pear_p"]
    correlation_c = p_links.loc[:, "Correlation_C"]
    correlation_w = p_links.loc[:, "Correlation_W"]
    mutual_info = p_links.loc[:, "MutualInfo"]
    Cdes = correlation_c.describe(percentiles=[0.5]).loc['50%']
    Wdes = correlation_w.describe(percentiles=[0.5]).loc['50%']
    Mdes = mutual_info.describe(percentiles=[0.5]).loc['50%']
    # Filter the Links
    # 1 Pear_p<1e-10
    # 2 Correlation_C>Cdes50
    # 3 Correlation_W>Wdes50
    # 4 MutualInfo>Mdes50
    filtered_links = p_links[(p_links["Pear_p"] < 1e-10)
                             & (p_links["Correlation_C"] > Cdes)
                             & (p_links["Correlation_W"] > Wdes)
                             & (p_links["MutualInfo"] > Mdes)].copy()
    return filtered_links


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


# Run main
if __name__ == "__main__":
    main()
