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



def load_data(tissue):
    betaqn = pickle.load( open( '../tissues/resi_norm_%s.p'%(tissue), "rb" ) )
    info = pd.read_csv('../tissues/info_%s.csv.zip'%(tissue),index_col=0, compression='zip',sep=',')

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
    tissue='EC'
    betaqn, info = load_data(tissue)
    #[100000, 50000, 1000, 500, 250, 100, 75, 50]
    num = 100
    CV = [3, 5, 7, 10, 15, 20, 25]
    features_sel = ['t_test','fisher','rfe']
    for feat_sel in features_sel:
        open_file = os.path.realpath('../data_str/')
        ec = betaqn
        cat = info['braak_bin'].loc[ec.index]
        svm_accuracy = {}
        samples = ec.shape[0]
        zeros = np.where(cat == 0)[0]
        ones = np.where(cat == 1)[0]

        for cv in CV:
            cv_splits = cv
            div_zeros = np.ceil(len(zeros)/cv_splits)
            div_ones = np.ceil(len(ones)/cv_splits)
            c_val_rbf = np.zeros(cv_splits)
            gamma_val_rbf = np.zeros(cv_splits)
            c_val_lin = np.zeros(cv_splits)
            best_score_rbf = np.zeros(cv_splits)
            best_score_lin = np.zeros(cv_splits)
            svm_accuracy = {}
            svm_accuracy_tr = {}
            zeros = np.random.permutation(zeros)
            ones = np.random.permutation(ones)
            for i in range(cv_splits):
                print('split: %d - cv: %d' %(i,cv))
                test_index, train_index = get_intervals(cv_splits, i, zeros, ones)
                train_full = ec.iloc[train_index]
                y_train = cat[train_index]
                test_full = ec.iloc[test_index]
                samples = test_full.shape[0]
                samples_tr = train_full.shape[0]
                start_time = time.time()
                features_file = open_file + "/features_diffCV_%s_%s_%d_%d.p" % (tissue, feat_sel, cv, i)
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
                (y_pred_rbf, y_tr_rbf, c_val_rbf[i], gamma_val_rbf[i],best_score_rbf[i]) = cl.SVM_classify_rbf_all(train, y_train,test,y_true,
                C_range = np.logspace(-6, 4, 20),gamma_range = np.logspace(-8, 2, 20))
                (y_pred_lin, y_tr_lin, c_val_lin[i], best_score_lin[i]) = cl.SVM_classify_lin_all(train, y_train, test, y_true,
                C_range = np.logspace(-6, 2, 20))
                print("--- %s seconds for classification ---" % (time.time() - start_time))

                pred_train = pd.DataFrame(
                {'y_train': y_train,
                 'y_tr_rbf': y_tr_rbf,
                 'y_tr_lin': y_tr_lin,
                })
                pickle.dump(pred_train, open(open_file + "/pred_tr_diffCV_%s_%s_%d_%d.p" %(tissue, feat_sel, cv, i), "wb"))
                svm_accuracy_tr[i] = [np.where((pred_train['y_train']==pred_train['y_tr_rbf'])==True)[0].shape[0]/samples_tr,
                                    np.where((pred_train['y_train']==pred_train['y_tr_lin'])==True)[0].shape[0]/samples_tr]
                print(svm_accuracy_tr[i])
                predictions = pd.DataFrame(
                {'y_true': y_true,
                 'y_rbf': y_pred_rbf,
                 'y_lin': y_pred_lin,
                })
                pickle.dump(predictions, open(open_file + "/pred_diffCV_%s_%s_%d_%d.p" %(tissue, feat_sel, cv, i), "wb"))
                svm_accuracy[i] = [np.where((predictions['y_true']==predictions['y_rbf'])==True)[0].shape[0]/samples,
                                    np.where((predictions['y_true']==predictions['y_lin'])==True)[0].shape[0]/samples]

                print(svm_accuracy[i])
            pickle.dump(svm_accuracy_tr, open(open_file + "/accuracy_tr_diffCV_%s_%s_%d.p" % (tissue, feat_sel,cv), "wb"))
            pickle.dump(svm_accuracy, open(open_file + "/accuracy_diffCV_%s_%s_%d.p" % (tissue, feat_sel,cv), "wb"))
            parameters = pd.DataFrame(
            {'C_rbf': c_val_rbf,
             'gamma_rbf': gamma_val_rbf,
             'C_lin': c_val_lin,
             'best_rbf': best_score_rbf,
             'best_lin': best_score_lin,
            })
            pickle.dump(parameters, open(open_file + "/params_diffCV_%s_%s_%d.p" %(tissue, feat_sel, cv), "wb"))




if __name__ == '__main__':
	main()
