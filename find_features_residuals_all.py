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
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2



# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


def load_data():
    betaqn = pickle.load( open( '../tissues/resi_norm_ALL.p', "rb" ) )
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


def main():
    tissues=['EC','CER','FC','STG','WB']
    open_file = os.path.realpath('../data_str/')
    ec, info = load_data()
    print('cargo datos')
    features_sel = ['t_test','fisher','rfe']
    num= 15
    for tissue in tissues:
        for feat_sel in features_sel:
            train_full = ec.loc[info[(info.tissue == tissue) & (info.braak_stage != 'Exclude')].index]
            features_file = open_file + "/features_ALL_CV_%s_%s.p" % (tissue, feat_sel)
            print(train_full.shape)
            start_time = time.time()
            if feat_sel == 't_test':
                features_all = fs.feature_sel_t_test_parallel(train_full, info, num)
            elif feat_sel == 'fisher':
                features_all = fs.feature_fisher_score_parallel(train_full, info, num)
            elif feat_sel == 'rfe':
                features_all = fs.feature_sel_rfe(train_full, info, num)
            print("--- %s seconds for feature selection ---" % (time.time() - start_time))
            pickle.dump(features_all, open(features_file, "wb"))




if __name__ == '__main__':
	main()
