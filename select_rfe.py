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

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


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


def main():
    tissues=['STG']
    #tissues=['WB', 'FC', 'STG']
    betaqn, info = load_data()
    feat_sel = 'rfe'
    for tissue in tissues:
        save_file = os.path.realpath('../data_str/')
        iters_big = 10
        iters_small = 30
        big_small = 200

        ec = betaqn.loc[info[(info.tissue == tissue) & (info.braak_stage != 'Exclude')].index]
        svm_accuracy = {}
        samples = ec.shape[0]

        features_num = [75, 100, 250, 500, 1000, 50000, 100000]
        for num in features_num:
            features_file = save_file + "/features_%s_%s_%d.p" % (tissue, feat_sel, num)
            my_file = Path(features_file)
            if my_file.is_file():
                features_per_i = pickle.load( open( features_file, "rb" ) )
            else:
                features_per_i = {}
                for i in range(samples):
                    print('iteracion %d para feature sel de %s' %(i,tissue))
                    start_time = time.time()
                    train_full = ec.loc[ec.index != ec.index[i]]
                    features_per_i[i] = fs.feature_sel_rfe(train_full, info, num)
                    print("--- %s seconds for feature selection ---" % (time.time() - start_time))
                pickle.dump(features_per_i, open(features_file, "wb"))

if __name__ == '__main__':
	main()
