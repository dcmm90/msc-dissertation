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
    tissues=['EC', 'CER', 'WB', 'FC', 'STG']
    iters_big = 10
    iters_small = 30
    big_small = 200
    #tissues=['WB', 'FC', 'STG']
    betaqn, info = load_data()
    for tissue in tissues:
        feat_sel = 'fisher'
        open_file = os.path.realpath('../DATA/%s' %feat_sel)
        save_file = os.path.realpath('../data_str/')

        ec = betaqn.loc[info[(info.tissue == tissue) & (info.braak_stage != 'Exclude')].index]
        svm_accuracy = {}
        samples = ec.shape[0]

        features_num = [100000, 50000, 1000, 500, 250, 100, 75, 50, 20]
        #features_num = [200000, 100000, 50000, 1000, 500, 250, 100, 75, 50, 20, 10]
        #features_num = [500, 250, 100, 75, 50, 20, 10]

        features_file = open_file + "/features_%s_%s.p" % (tissue, feat_sel)
        my_file = Path(features_file)
        if my_file.is_file():
            features_per_i = pickle.load( open( features_file, "rb" ) )
        else:
            features_per_i = {}
            for i in range(samples):
                print('iteracion %d para feature sel' %i)
                start_time = time.time()
                train_full = ec.loc[ec.index != ec.index[i]]
                if feat_sel == 't_test':
                    features_per_i[i] = fs.feature_sel_t_test_parallel(train_full, info, features_num[0])
                elif feat_sel == 'fisher':
                    features_per_i[i] = fs.feature_fisher_score_parallel(train_full, info, features_num[0])
                print("--- %s seconds for feature selection ---" % (time.time() - start_time))
            pickle.dump(features_per_i, open(features_file, "wb"))

        for num in features_num:
            print(num)
            features_sel = dict.fromkeys(list(ec),0)
            y_true = np.zeros(samples)
            y_pred_lr = np.zeros(samples)
            for i in range(samples):
                print('iteracion %d para %d features' %(i,num))
                train_full = ec.loc[ec.index != ec.index[i]]
                start_time = time.time()
                train = train_full[features_per_i[i][0:num]]
                print("--- %s seconds for feature selection ---" % (time.time() - start_time))
                print('features selected')
                test = ec.loc[ec.index == ec.index[i]]
                test = test[features_per_i[i][0:num]]
                y_train = info['braak_bin'].loc[train.index]
                y_true[i] = info['braak_bin'].loc[test.index]
                y_pred_lr[i] = cl.log_reg(train, y_train, test)
            predictions = pd.DataFrame(
            {'y_true': y_true,
             'y_lr': y_pred_lr,
            })
            pickle.dump(predictions, open(save_file + "/pred_lr_%s_%s_%d.p" %(tissue, feat_sel, num), "wb"))
            #pickle.dump(features_sel, open(save_file + "/feat_%s_%s_%d.p" %(tissue, feat_sel, num), "wb"))
            #features_sel_total = {key: value + [features_sel[key]] for key, value in features_sel_total.items()}
            svm_accuracy[num] = [np.where((predictions['y_true']==predictions['y_lr'])==True)[0].shape[0]/samples]
        pickle.dump(svm_accuracy, open(save_file + "/accuracy_lr_%s_%s.p" % (tissue, feat_sel), "wb"))



if __name__ == '__main__':
	main()