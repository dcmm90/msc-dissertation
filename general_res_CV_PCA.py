# ----------------------------------------------------
# Dissertation MSc CSML
# Author: Diana Carolina Montanes Mondragon
# ----------------------------------------------------
# file_name: general_res_CV_PCA.py
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
import utils_msc as ut
import os
from sklearn.decomposition import PCA
from sklearn import preprocessing
# ----------------------------------------------------

# ------------------- Constant -------------------------
open_file = os.path.realpath('../data_str/')
tissues = ['EC', 'FC', 'STG', 'CER', 'WB']
features_num = [5, 10, 15, 20, 50, 75, 100, 250, 500, 1000, 5000, 10000]
feat_sel = 'PCA'
# ----------------------------------------------------


# ------------------- Function -------------------------
# general_PCA_CV(tissues)
# This function performs the nested cross-validation for
# data using PCA as dimension reduction.
# inputs: tissues - the tissues
# ----------------------------------------------------
def general_PCA_CV():
    for tissue in tissues:
        ec, info = load_data(tissue)
        cat = info['braak_bin'].loc[ec.index]
        zeros = np.where(cat == 0)[0]
        ones = np.where(cat == 1)[0]
        cv_splits = 5
        for num in features_num:
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
                print('split: %d - num_features: %d - tissue:%s- feat_sel:%s' % (i, num, tissue, feat_sel))
                test_index, train_index = ut.get_intervals(cv_splits, i, zeros, ones)
                train_full = ec.iloc[train_index]
                y_train = cat[train_index]
                test_full = ec.iloc[test_index]
                samples = test_full.shape[0]
                samples_tr = train_full.shape[0]
                # SCALING
                scale = preprocessing.StandardScaler().fit(train_full)
                train_sc = scale.transform(train_full)
                test_sc = scale.transform(test_full)
                # PCA
                pca = PCA(n_components=num)
                pca.fit(train_sc)
                train = pca.transform(train_sc)
                test = pca.transform(test_sc)
                y_true = cat[test_index]
                start_time = time.time()
                (y_pred_rbf, y_tr_rbf, c_val_rbf[i], gamma_val_rbf[i], best_score_rbf[i]) = cl.SVM_classify_rbf_all(
                    train, y_train, test, y_true, C_range=np.logspace(-4, 4, 20), gamma_range=np.logspace(-7, 2, 20))
                (y_pred_lin, y_tr_lin, c_val_lin[i], best_score_lin[i]) = cl.SVM_classify_lin_all(train, y_train, test,
                    y_true, C_range=np.logspace(-3, 1, 6))

                print("--- %s seconds for classification ---" % (time.time() - start_time))
                pred_train = pd.DataFrame(
                    {'y_train': y_train,
                     'y_tr_rbf': y_tr_rbf,
                     'y_tr_lin': y_tr_lin,
                     })
                pickle.dump(pred_train,
                            open(open_file + "/pred_tr_CV_%s_%s_%d_%d.p" % (tissue, feat_sel, num, i), "wb"))
                svm_accuracy_tr[i] = [
                    np.where((pred_train['y_train'] == pred_train['y_tr_rbf']) == True)[0].shape[0] / samples_tr,
                    np.where((pred_train['y_train'] == pred_train['y_tr_lin']) == True)[0].shape[0] / samples_tr]
                print(svm_accuracy_tr[i])

                predictions = pd.DataFrame(
                    {'y_true': y_true,
                     'y_rbf': y_pred_rbf,
                     'y_lin': y_pred_lin,
                     })
                pickle.dump(predictions, open(open_file + "/pred_CV_%s_%s_%d_%d.p" % (tissue, feat_sel, num, i), "wb"))
                svm_accuracy[i] = [
                    np.where((predictions['y_true'] == predictions['y_rbf']) == True)[0].shape[0] / samples,
                    np.where((predictions['y_true'] == predictions['y_lin']) == True)[0].shape[0] / samples]
                print(svm_accuracy[i])
            pickle.dump(svm_accuracy_tr, open(open_file + "/accuracy_tr_CV_%s_%s_%d.p" % (tissue, feat_sel, num), "wb"))
            pickle.dump(svm_accuracy, open(open_file + "/accuracy_CV_%s_%s_%d.p" % (tissue, feat_sel, num), "wb"))
            parameters = pd.DataFrame(
                {'C_rbf': c_val_rbf,
                 'gamma_rbf': gamma_val_rbf,
                 'C_lin': c_val_lin,
                 'best_rbf': best_score_rbf,
                 'best_lin': best_score_lin,
                 })
            pickle.dump(parameters, open(open_file + "/params_CV_%s_%s_%d.p" % (tissue, feat_sel, num), "wb"))


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
    return betaqn, info

# ------------------- Function -------------------------
# main()
# ----------------------------------------------------
def main():
    general_PCA_CV()


if __name__ == '__main__':
	main()
