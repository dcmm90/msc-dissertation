from __future__ import division
import time
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import pickle
import classification as cl
import feature_selection as fs
import os.path
from zipfile import ZipFile
import sys, os
from os.path import join, dirname, abspath
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from multiprocessing import cpu_count, Pool
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.feature_selection import RFE




def load_data():
    beta_file = os.path.realpath('../GSE59685_betas2.csv.zip')
    zipfile = ZipFile(beta_file)
    zipfile.getinfo('GSE59685_betas2.csv').file_size += (2 ** 64) - 1
    betaqn = pd.read_csv(zipfile.open('GSE59685_betas2.csv'),skiprows=(1,2), index_col=0,sep=',')
    betaqn = betaqn.T

    info = pd.read_csv('info.csv.zip',index_col=1, compression='zip',sep=',')
    info = info.drop('Unnamed: 0', 1)

    info.loc[(info.braak_stage=='5') | (info.braak_stage=='6'),'braak_bin'] = 1
    cond = ((info.braak_stage=='0') | (info.braak_stage=='1') | (info.braak_stage=='2') |
            (info.braak_stage=='3') | (info.braak_stage=='4'))
    info.loc[cond ,'braak_bin'] = 0
    info.loc[info.source_tissue == 'entorhinal cortex', 'tissue'] = 'EC'
    info.loc[info.source_tissue == 'whole blood', 'tissue'] = 'WB'
    info.loc[info.source_tissue == 'frontal cortex', 'tissue'] = 'FC'
    info.loc[info.source_tissue == 'superior temporal gyrus', 'tissue'] = 'STG'
    info.loc[info.source_tissue == 'cerebellum', 'tissue'] = 'CER'
    return (betaqn, info)


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

def parallelize(data, func):
    cores = cpu_count() #Number of CPU cores on your system
    #print('num of cores: %d' %cores)
    partitions = cores#Define as many partitions as you want
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


def main():
    tissue='EC'
    betaqn, info = load_data()
    ec = betaqn.loc[info[(info.tissue == tissue) & (info.braak_stage != 'Exclude')].index]
    #[100000, 50000, 1000, 500, 250, 100, 75, 50]
    features_num = [20, 50, 75, 100, 250, 500, 1000, 5000, 100000]
    feat_sel = 't_test'
    num = 20
    print('num of cores: %d' %cpu_count())
    start_time = time.time()
    if feat_sel == 't_test':
        features_all = feature_sel_t_test_parallel(ec, info, num)
    elif feat_sel == 'fisher':
        features_all = feature_fisher_score_parallel(train_full, info, num)
    print("--- %s seconds for feature selection ---" % (time.time() - start_time))



if __name__ == '__main__':
	main()
