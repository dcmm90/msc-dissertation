import numpy as np
import pandas as pd
from multiprocessing import cpu_count, Pool

def feature_sel_t_test(betas, info, size):
    c_info = info.loc[betas.index]
    c1 = c_info[c_info['braak_bin'] == 0].index
    c2 = c_info[c_info['braak_bin'] == 1].index
    m1 = betas.loc[c1].apply(np.mean,0)
    m2 = betas.loc[c2].apply(np.mean,0)
    numerator = np.absolute(m1 - m2)
    std1 = betas.loc[c1].apply(np.std,0)
    std2 = betas.loc[c2].apply(np.std,0)
    denominator = np.sqrt((std1/c1.shape[0])+(std2/c2.shape[0]))
    t_stat = numerator/denominator
    ind = np.argsort(t_stat)[-size:]
    ec_feat = betas.iloc[:,ind]
    return list(ec_feat), ec_feat

def feature_sel_t_test_parallel(betas, info, size):
    c_info = info.loc[betas.index]
    c1 = c_info[c_info['braak_bin'] == 0].index
    c2 = c_info[c_info['braak_bin'] == 1].index
    betas_c1 = betas.loc[c1]
    betas_c2 = betas.loc[c2]
    m1 = parallelize(betas_c1, mean_par)
    m2 = parallelize(betas_c2, mean_par)
    numerator = np.absolute(m1 - m2)
    std1 = parallelize(betas_c1, std_par)
    std2 = parallelize(betas_c2, std_par)
    denominator = np.sqrt((std1/c1.shape[0])+(std2/c2.shape[0]))
    t_stat = numerator/denominator
    #ind = np.argsort(t_stat)[-size:]
    #ec_feat = betas.iloc[:,ind]
    ind = np.argsort(t_stat)[::-1]
    ec_feat = betas.iloc[:,ind]
    return list(ec_feat)


def feature_fisher_score(betas, info, size):
    c_info = info.loc[betas.index]
    c1 = c_info[c_info['braak_bin'] == 0].index
    c2 = c_info[c_info['braak_bin'] == 1].index

    ec_mean = ec.apply(np.mean,0)
    numerator = (c1.shape[0]*np.power(ec.loc[c1].apply(np.mean,0) - ec_mean,2) +
                c2.shape[0]*np.power(ec.loc[c2].apply(np.mean,0) - ec_mean,2))
    denominator = ((ec.loc[c1].apply(np.var,0) * c1.shape[0]) +
                   (ec.loc[c2].apply(np.var,0)*c2.shape[0]))
    fisher_score = numerator/denominator
    ind = np.argsort(fisher_score)[-size:]
    ec_feat = betas.iloc[:,ind]
    return list(ec_feat), ec_feat


def parallelize(data, func):
    cores = cpu_count() #Number of CPU cores on your system
    print('num of cores: %d' %cores)
    partitions = cores #Define as many partitions as you want
    data_split = np.array_split(data, partitions, axis=1)
    pool = Pool(cores)
    data = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return data

def mean_par(data):
    return data.apply(np.mean,0)

def std_par(data):
    return data.apply(np.std,0)

def var_par(data):
    return data.apply(np.var,0)
