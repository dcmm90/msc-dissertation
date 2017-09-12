# ----------------------------------------------------
# Dissertation MSc CSML
# Author: Diana Carolina Montanes Mondragon
# ----------------------------------------------------
# file_name: feature_selection.py
# description: This file contains the functions for
#              classifying using SVM
# ----------------------------------------------------

# ------------------- imports -------------------------
import numpy as np
import pandas as pd
from multiprocessing import cpu_count, Pool
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
# ----------------------------------------------------


# ------------------- Function -------------------------
# feature_sel_rfe(betas, info, size)
# This function performs the RFE
# inputs: betas - DNA methylation data
#         info - categories of data
#         size - number of features selected
# returns: list(selection) - ids of features selected
# ----------------------------------------------------
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


# ------------------- Function -------------------------
# feature_sel_rfe(betas, info, size)
# This function uses t_test for feature selection
# inputs: betas - DNA methylation data
#         info - categories of data
#         size - number of features selected
# returns: (list(ec_feat))
#          list(ec_feat) - ids of features selected
# ----------------------------------------------------
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
    return list(ec_feat)


# ------------------- Function -------------------------
# feature_sel_t_test_parallel(betas, info, size)
# This function uses t_test for feature selection.
# Optimized to work in parallel
# inputs: betas - DNA methylation data
#         info - categories of data
#         size - number of features selected
# returns: list(ec_feat)[0:size]
#          list(ec_feat) - ids of features selected
# ----------------------------------------------------
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
    ind = np.argsort(t_stat)[::-1]
    ec_feat = betas.iloc[:,ind]
    return list(ec_feat)[0:size]


# ------------------- Function -------------------------
# feature_sel_rfe(betas, info, size)
# This function uses fisher score for feature selection
# inputs: betas - DNA methylation data
#         info - categories of data
#         size - number of features selected
# returns: (list(ec_feat))
#          list(ec_feat) - ids of features selected
# ----------------------------------------------------
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


# ------------------- Function -------------------------
# feature_sel_t_test_parallel(betas, info, size)
# This function uses fisher score for feature selection.
# Optimized to work in parallel
# inputs: betas - DNA methylation data
#         info - categories of data
#         size - number of features selected
# returns: list(ec_feat)[0:size]
#          list(ec_feat) - ids of features selected
# ----------------------------------------------------
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


# ------------------- Function -------------------------
# parallelize(data, func)
# This function parallelises the calculations performed to
# matrices of data
# inputs: data - the matrix of data (entry per row)
#         func - function to be applied to data
# returns: data - the matrix of data
# ----------------------------------------------------
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


# ------------------- Function -------------------------
# mean_par(data)
# Calculates mean of the columns (features)
# inputs: data - the matrix of data (entry per row)
# returns: data.apply(np.mean,0) - the mean of data
# ----------------------------------------------------
def mean_par(data):
    return data.apply(np.mean, 0)


# ------------------- Function -------------------------
# std_par(data)
# Calculates standard deviation of the columns (features)
# inputs: data - the matrix of data (entry per row)
# returns: data.apply(np.std,0) - the std of data
# ----------------------------------------------------
def std_par(data):
    return data.apply(np.std, 0)


# ------------------- Function -------------------------
# var_par(data)
# Calculates variance of the columns (features)
# inputs: data - the matrix of data (entry per row)
# returns: data.apply(np.var,0) - the var of data
# ----------------------------------------------------
def var_par(data):
    return data.apply(np.var, 0)
