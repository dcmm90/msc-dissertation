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

def get_intervals(cv_splits, i, zeros, ones):
    div_zeros = int(np.floor(len(zeros)/cv_splits))
    div_ones = int(np.floor(len(ones)/cv_splits))
    if (i<(cv_splits-1)):
        mini_zero = div_zeros*i
        maxi_zero = (div_zeros*i) + div_zeros
        mini_one = div_ones*i
        maxi_one = (div_ones*i) + div_ones
    else:
        mini_zero = div_zeros*i
        maxi_zero = len(zeros)
        mini_one = div_ones*i
        maxi_one = len(ones)
    index_zeros = list(zeros[mini_zero: maxi_zero])
    index_ones = list(ones[mini_one: maxi_one])
    test = np.array(index_zeros + index_ones )
    train = np.array(list(set(list(ones)+list(zeros)) - set(test)))
    return test,train


def main():
    tissues=['EC', 'CER', 'WB', 'FC', 'STG']
    betaqn, info = load_data()
    #[100000, 50000, 1000, 500, 250, 100, 75, 50]
    features_num = [20, 50, 75, 100, 250, 500, 1000]
    for tissue in tissues:
        feat_sel = 't_test'
        open_file = os.path.realpath('../data_str/')
        ec = betaqn.loc[info[(info.tissue == tissue) & (info.braak_stage != 'Exclude')].index]
        cat = info['braak_bin'].loc[ec.index]
        svm_accuracy = {}
        samples = ec.shape[0]
        zeros = np.where(cat == 0)[0]
        ones = np.where(cat == 1)[0]
        cv_splits = 10
        div_zeros = np.ceil(len(zeros)/cv_splits)
        div_ones = np.ceil(len(ones)/cv_splits)

        for num in features_num:
            c_val_rbf = np.zeros(cv_splits)
            gamma_val_rbf = np.zeros(cv_splits)
            c_val_lin = np.zeros(cv_splits)
            c_val_pol = np.zeros(cv_splits)
            gamma_val_pol = np.zeros(cv_splits)
            svm_accuracy = {}
            zeros = np.random.permutation(zeros)
            ones = np.random.permutation(ones)
            for i in range(cv_splits):
                print('split: %d - num_features: %d' %(i,num))
                test_index, train_index = get_intervals(cv_splits, i, zeros, ones)
                train_full = ec.iloc[train_index]
                y_train = cat[train_index]
                test_full = ec.iloc[test_index]
                samples = test_full.shape[0]
                start_time = time.time()
                features_file = open_file + "/features_CV_%s_%s_%d_%d.p" % (tissue, feat_sel, num, i)
                if feat_sel == 't_test':
                    features_all = fs.feature_sel_t_test_parallel(train_full, info, num)
                elif feat_sel == 'fisher':
                    features_all = fs.feature_fisher_score_parallel(train_full, info, num)
                elif feat_sel == 'rfe':
                    features_all = fs.feature_sel_rfe(train_full, info, num)
                print("--- %s seconds for feature selection ---" % (time.time() - start_time))
                pickle.dump(features_all, open(features_file, "wb"))

                train = train_full[features_all[0:num]]
                test = test_full[features_all[0:num]]
                y_true = cat[test_index]
                start_time = time.time()
                (y_pred_rbf, c_val_rbf[i], gamma_val_rbf[i]) = cl.SVM_classify_rbf_all(train, y_train, test, y_true,balance = 1)
                (y_pred_pol,  c_val_pol[i], gamma_val_pol[i]) = cl.SVM_classify_poly_all(train, y_train, test,y_true, balance = 1)
                (y_pred_lin, c_val_lin[i]) = cl.SVM_classify_lin_all(train, y_train, test, y_true, balance = 1)
                print("--- %s seconds for classification ---" % (time.time() - start_time))


                predictions = pd.DataFrame(
                {'y_true': y_true,
                 'y_rbf': y_pred_rbf,
                 'y_poly': y_pred_pol,
                 'y_lin': y_pred_lin,
                })
                pickle.dump(predictions, open(open_file + "/pred_CV_%s_%s_%d_%d.p" %(tissue, feat_sel, num, i), "wb"))
                svm_accuracy[i] = [np.where((predictions['y_true']==predictions['y_rbf'])==True)[0].shape[0]/samples,
                                    np.where((predictions['y_true']==predictions['y_poly'])==True)[0].shape[0]/samples,
                                    np.where((predictions['y_true']==predictions['y_lin'])==True)[0].shape[0]/samples]
                print(svm_accuracy[i])
            pickle.dump(svm_accuracy, open(open_file + "/accuracy_CV_%s_%s_%d.p" % (tissue, feat_sel,num), "wb"))
            parameters = pd.DataFrame(
            {'C_rbf': c_val_rbf,
             'gamma_rbf': gamma_val_rbf,
             'C_poly': c_val_pol,
             'gamma_poly': gamma_val_pol,
             'C_lin': c_val_lin
            })
            pickle.dump(parameters, open(open_file + "/params_CV_%s_%s_%d.p" %(tissue, feat_sel, num), "wb"))





if __name__ == '__main__':
	main()
