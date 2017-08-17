import numpy as np
import pandas as pd
from multiprocessing import cpu_count, Pool
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.feature_selection import RFE


def feature_sel_rfe(betas, info, size):
    c_info = info.loc[betas.index]
    y = c_info.braak_bin
    X = betas
    svc = SVC(kernel="linear", C=1)
    rfe = RFE(estimator=svc, n_features_to_select=size, step=0.1)
    rfe.fit(X, y)
    ranking = rfe.ranking_
    selection = X.iloc[:,np.where(ranking == 1)[0]]
    print(selection.shape)
    return list(selection)


def feature_sel_rfe_group(betas, cat, size):
    y = cat
    X = betas
    svc = SVC(kernel="linear", C=1)
    rfe = RFE(estimator=svc, n_features_to_select=size, step=0.1)
    rfe.fit(X, y)
    ranking = rfe.ranking_
    selection = X.iloc[:,np.where(ranking == 1)[0]]
    print(selection.shape)
    return list(selection)


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
    return list(ec_feat)[0:size]

def feature_sel_t_test_group(betas, cat, size):
    c1 = cat[cat == 0].index
    c2 = cat[cat == 1].index
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

    betas_mean = betas.apply(np.mean,0)
    numerator = (c1.shape[0]*np.power(ec.loc[c1].apply(np.mean,0) - betas_mean,2) +
                c2.shape[0]*np.power(ec.loc[c2].apply(np.mean,0) - betas_mean,2))
    denominator = ((ec.loc[c1].apply(np.var,0) * c1.shape[0]) +
                   (ec.loc[c2].apply(np.var,0) * c2.shape[0]))
    fisher_score = numerator/denominator
    ind = np.argsort(fisher_score)[::-1]
    ec_feat = betas.iloc[:,ind]
    return list(ec_feat)

def feature_fisher_score_parallel(betas, info, size):
    c_info = info.loc[betas.index]
    c1 = c_info[c_info['braak_bin'] == 0].index
    c2 = c_info[c_info['braak_bin'] == 1].index
    betas_c1 = betas.loc[c1]
    betas_c2 = betas.loc[c2]
    m = parallelize(betas, mean_par)
    m1 = parallelize(betas_c1, mean_par)
    m2 = parallelize(betas_c2, mean_par)
    var1 = parallelize(betas_c1, var_par)
    var2 = parallelize(betas_c2, var_par)

    numerator = c1.shape[0]*np.power(m1 - m,2) + c2.shape[0]*np.power(m2 - m,2)
    denominator = ((var1 * c1.shape[0]) + (var2 * c2.shape[0]))
    fisher_score = numerator/denominator
    ind = np.argsort(fisher_score)[::-1]
    ec_feat = betas.iloc[:,ind]
    return list(ec_feat)[0:size]

def feature_fisher_score_group(betas, cat, size):
    c1 = cat[cat == 0].index
    c2 = cat[cat == 1].index
    betas_c1 = betas.loc[c1]
    betas_c2 = betas.loc[c2]
    m = parallelize(betas, mean_par)
    m1 = parallelize(betas_c1, mean_par)
    m2 = parallelize(betas_c2, mean_par)
    var1 = parallelize(betas_c1, var_par)
    var2 = parallelize(betas_c2, var_par)

    numerator = c1.shape[0]*np.power(m1 - m,2) + c2.shape[0]*np.power(m2 - m,2)
    denominator = ((var1 * c1.shape[0]) + (var2 * c2.shape[0]))
    fisher_score = numerator/denominator
    ind = np.argsort(fisher_score)[::-1]
    ec_feat = betas.iloc[:,ind]
    return list(ec_feat)[0:size]


def parallelize(data, func):
    cores = cpu_count() #Number of CPU cores on your system
    #print('num of cores: %d' %cores)
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
