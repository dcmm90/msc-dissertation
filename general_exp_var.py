# ----------------------------------------------------
# Dissertation MSc CSML
# Author: Diana Carolina Montanes Mondragon
# ----------------------------------------------------
# file_name: general_exp_var.py
# description: This file contains the main function for
#               Experiment 2
# ----------------------------------------------------

# ------------------- imports -------------------------
from __future__ import division
import time
import pandas as pd
import numpy as np
import pickle
import classification as cl
import feature_selection as fs
import utils_msc as ut
import os
from sklearn.decomposition import PCA
from sklearn import preprocessing
# ----------------------------------------------------

# ------------------- Constant -------------------------
OPEN_FILE = os.path.realpath('../data_str/')
FEATURES_SEL = ['t_test', 'fisher', 'rfe']
FEATURES_NUM = [5, 10, 15, 20, 50, 75, 100, 250, 500, 1000, 5000]
TISSUES = ['EC', 'CER', 'FC', 'STG', 'WB']
# ----------------------------------------------------


# ------------------- Function -------------------------
# load_data(tissue)
# This function load the data
# inputs: tissue - tissue to load data from
# returns: (betaqn, info)
#          betaqn - DNA methylation from the tissue
#          info - info from DNA methylation data
# ----------------------------------------------------
def load_data(tissue):
    betaqn = pickle.load( open( '../tissues/resi_norm_%s.p'%(tissue), "rb" ) )
    info = pd.read_csv('../tissues/info_%s.csv.zip'%(tissue),index_col=0, compression='zip',sep=',')
    return (betaqn, info)


# ------------------- Function -------------------------
# general_exp_var()
# Nested cross validation after reducing to 100.000 features
# of largest variance.
# ----------------------------------------------------
def general_exp_var():

    for tissue in TISSUES:

        for feat_sel in FEATURES_SEL:
            beta, info = load_data(tissue)
            vari = beta.var()
            ind = np.argsort(vari)[-50000:]
            ec = beta.iloc[:, ind]
            cat = info['braak_bin'].loc[ec.index]
            nzeros = np.where(cat == 0)[0]
            nones = np.where(cat == 1)[0]
            cv_splits = 5

            for num in FEATURES_NUM:
                c_val_rbf = np.zeros(cv_splits)
                gamma_val_rbf = np.zeros(cv_splits)
                c_val_lin = np.zeros(cv_splits)
                best_score_rbf = np.zeros(cv_splits)
                best_score_lin = np.zeros(cv_splits)
                svm_accuracy = {}
                svm_accuracy_tr = {}
                zeros = np.random.permutation(nzeros)
                ones = np.random.permutation(nones)
                for i in range(cv_splits):
                    print('split: %d - num_features: %d - tissue:%s- feat_sel:%s' % (i, num, tissue, feat_sel))
                    test_index, train_index = ut.get_intervals(cv_splits, i, zeros, ones)
                    print(test_index)
                    train_full = ec.iloc[train_index]
                    y_train = cat[train_index]
                    test_full = ec.iloc[test_index]
                    samples = test_full.shape[0]
                    samples_tr = train_full.shape[0]
                    start_time = time.time()
                    features_file = OPEN_FILE + "/features_exp_CV_%s_%s_%d_%d.p" % (tissue, feat_sel, num, i)
                    print(train_full.shape)
                    if feat_sel == 't_test':
                        features_all = fs.feature_sel_t_test_parallel(train_full, info, num)
                    elif feat_sel == 'fisher':
                        features_all = fs.feature_fisher_score_parallel(train_full, info, num)
                    elif feat_sel == 'rfe':
                        features_all = fs.feature_sel_rfe(train_full, info, num)

                    print("--- %s seconds for feature selection ---" % (time.time() - start_time))
                    pickle.dump(features_all, open(features_file, "wb"))

                    if feat_sel == 'PCA':
                        # SCALING
                        scale = preprocessing.StandardScaler().fit(train_full)
                        train_sc = scale.transform(train_full)
                        test_sc = scale.transform(test_full)
                        # PCA
                        pca = PCA(n_components=num)
                        pca.fit(train_sc)
                        train = pca.transform(train_sc)
                        test = pca.transform(test_sc)
                    else:
                        train = train_full[features_all[0:num]]
                        print(train.shape)
                        test = test_full[features_all[0:num]]

                    y_true = cat[test_index]
                    start_time = time.time()
                    (y_pred_rbf, y_tr_rbf, c_val_rbf[i], gamma_val_rbf[i], best_score_rbf[i]) = cl.SVM_classify_rbf_all(
                        train, y_train, test, y_true,C_range=np.logspace(-4, 4, 20), gamma_range=np.logspace(-7, 2, 20))
                    (y_pred_lin, y_tr_lin, c_val_lin[i], best_score_lin[i]) = cl.SVM_classify_lin_all(train, y_train,
                        test, y_true, C_range=np.logspace(-4, 3, 20))

                    print("--- %s seconds for classification ---" % (time.time() - start_time))
                    pred_train = pd.DataFrame(
                        {'y_train': y_train,
                         'y_tr_rbf': y_tr_rbf,
                         'y_tr_lin': y_tr_lin,
                         })
                    pickle.dump(pred_train,
                                open(OPEN_FILE + "/pred_exp_tr_CV_%s_%s_%d_%d.p" % (tissue, feat_sel, num, i), "wb"))
                    svm_accuracy_tr[i] = [
                        np.where((pred_train['y_train'] == pred_train['y_tr_rbf']) == True)[0].shape[0] / samples_tr,
                        np.where((pred_train['y_train'] == pred_train['y_tr_lin']) == True)[0].shape[0] / samples_tr]
                    print(svm_accuracy_tr[i])
                    predictions = pd.DataFrame(
                        {'y_true': y_true,
                         'y_rbf': y_pred_rbf,
                         'y_lin': y_pred_lin,
                         })
                    pickle.dump(predictions,
                                open(OPEN_FILE + "/pred_exp_CV_%s_%s_%d_%d.p" % (tissue, feat_sel, num, i), "wb"))
                    svm_accuracy[i] = [
                        np.where((predictions['y_true'] == predictions['y_rbf']) == True)[0].shape[0] / samples,
                        np.where((predictions['y_true'] == predictions['y_lin']) == True)[0].shape[0] / samples]

                    print(svm_accuracy[i])

                pickle.dump(svm_accuracy_tr,
                            open(OPEN_FILE + "/accuracy_exp_tr_CV_%s_%s_%d.p" % (tissue, feat_sel, num), "wb"))
                pickle.dump(svm_accuracy,
                            open(OPEN_FILE + "/accuracy_exp_CV_%s_%s_%d.p" % (tissue, feat_sel, num), "wb"))
                parameters = pd.DataFrame(
                    {'C_rbf': c_val_rbf,
                     'gamma_rbf': gamma_val_rbf,
                     'C_lin': c_val_lin,
                     'best_rbf': best_score_rbf,
                     'best_lin': best_score_lin,
                     })
                pickle.dump(parameters, open(OPEN_FILE + "/params_exp_CV_%s_%s_%d.p" % (tissue, feat_sel, num), "wb"))

# ------------------- Function -------------------------
# main()
# ----------------------------------------------------
def main():
    general_exp_var()

if __name__ == '__main__':
	main()
