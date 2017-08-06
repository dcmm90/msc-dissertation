# ------------
# Diana Carolina Montañes M.
# predict_blood: train in all tissues(3) and
# predict on blood sample


from __future__ import division
import time
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
import pickle
import classification as cl
import feature_selection as fs
import os.path
from zipfile import ZipFile
import sys, os
from os.path import join, dirname, abspath
from pathlib import Path


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

def test_blood(betaqn, info):
    save_file = os.path.realpath('../data_str/')
    feat_sel = 'rfe'
    test_all = betaqn.loc[info[(info.tissue == 'WB') & (info.braak_stage != 'Exclude')].index]
    ec = betaqn.loc[info[((info.tissue == 'EC')|(info.tissue == 'FC')|(info.tissue == 'STG')) & (info.braak_stage != 'Exclude')].index]
    svm_accuracy = {}
    samples = ec.shape[0]
    features_num = [50000, 1000, 500, 250, 100, 75, 50, 20]
    for num in features_num:
        print(num)
        features_per_i = {}
        y_true = np.zeros(samples)
        y_pred_rbf = np.zeros(samples)
        c_val_rbf = np.zeros(samples)
        gamma_val_rbf = np.zeros(samples)
        y_pred_lin = np.zeros(samples)
        c_val_lin = np.zeros(samples)
        y_pred_pol = np.zeros(samples)
        c_val_pol = np.zeros(samples)
        gamma_val_pol = np.zeros(samples)
        X_train = ec
        y_train = info['braak_bin'].loc[X_train.index]
        X_test = test_all
        y_test = info['braak_bin'].loc[X_test.index]
        if feat_sel == 't_test':
            features_per_i[i] = fs.feature_sel_t_test_group(X_train, y_train, num)
        elif feat_sel == 'fisher':
            features_per_i[i] = fs.feature_fisher_score_group(X_train, y_train, num)
        elif feat_sel == 'rfe':
            features_per_i[i] = fs.feature_sel_rfe_group(X_train, y_train, num)
        start_time = time.time()
        train = X_train[features_per_i[i]]
        print("--- %s seconds for feature selection ---" % (time.time() - start_time))
        print('features selected')
        test = X_test[features_per_i[i]]
        (y_pred_pol[i], c_val_pol[i], gamma_val_pol[i]) = cl.SVM_classify_poly(train, y_train, test)
        (y_pred_rbf[i], c_val_rbf[i], gamma_val_rbf[i]) = cl.SVM_classify_rbf(train, y_train, test)
        (y_pred_lin[i], c_val_lin[i]) = cl.SVM_classify_lin(train, y_train, test)
        parameters = pd.DataFrame(
        {'C_rbf': c_val_rbf,
         'gamma_rbf': gamma_val_rbf,
         'C_poly': c_val_pol,
         'gamma_poly': gamma_val_pol,
         'C_lin': c_val_lin
        })
        pickle.dump(parameters, open(save_file + "/params_%s_%s_%d.p" %(tissue, feat_sel, num), "wb"))
        predictions = pd.DataFrame(
        {'y_true': y_test,
         'y_rbf': y_pred_rbf,
         'y_poly': y_pred_pol,
         'y_lin': y_pred_lin,
        })
        pickle.dump(features_per_i, open(save_file + "/features_%s_%s_%d.p" % (tissue, feat_sel, num), "wb"))
        pickle.dump(predictions, open(save_file + "/pred_%s_%s_%d.p" %(tissue, feat_sel, num), "wb"))
        svm_accuracy[num] = [np.where((predictions['y_true']==predictions['y_rbf'])==True)[0].shape[0]/samples,
                            np.where((predictions['y_true']==predictions['y_poly'])==True)[0].shape[0]/samples,
                            np.where((predictions['y_true']==predictions['y_lin'])==True)[0].shape[0]/samples]
    pickle.dump(svm_accuracy, open(save_file + "/accuracy_%s_%s.p" % (tissue, feat_sel), "wb"))



def main():
    betaqn, info = load_data()
    test_blood(betaqn, info)


if __name__ == '__main__':
	main()
