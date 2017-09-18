# ----------------------------------------------------
# Dissertation MSc CSML
# Author: Diana Carolina Montanes Mondragon
# ----------------------------------------------------
# file_name: general_all.py
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
FEATURES_SEL = ['t_test','fisher','rfe','PCA']
FEATURES_NUM = [5, 10, 15, 20, 50, 75, 100]
TISSUE = 'all'
# ----------------------------------------------------


# ------------------- Function -------------------------
# general_all()
# This function performs the nested cross-validation for
# all the tissues together
# ----------------------------------------------------
def general_all():
    ec, info = load_data()
    for feat_sel in FEATURES_SEL:
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
                print('gen_all -split: %d - num_features: %d - feat_sel:%s' % (i, num, feat_sel))
                test_index, train_index = ut.get_intervals(cv_splits, i, zeros, ones)
                print('tamaño de test: %s' % len(test_index))
                train_full = ec.iloc[train_index]
                y_train = cat[train_index]
                test_full = ec.iloc[test_index]
                samples = test_full.shape[0]
                samples_tr = train_full.shape[0]
                start_time = time.time()
                features_file = OPEN_FILE + "/features_CV_%s_%s_%d_%d.p" % (TISSUE, feat_sel, num, i)
                print(train_full.shape)
                if feat_sel == 't_test':
                    features_all = fs.feature_sel_t_test_parallel(train_full, info, num)
                elif feat_sel == 'fisher':
                    features_all = fs.feature_fisher_score_parallel(train_full, info, num)
                elif feat_sel == 'rfe':
                    features_all = fs.feature_sel_rfe(train_full, info, num)
                # elif feat_sel == 'chi2':

                print("--- %s seconds for feature selection ---" % (time.time() - start_time))

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
                    pickle.dump(features_all, open(features_file, "wb"))
                    train = train_full[features_all[0:num]]
                    print(train.shape)
                    test = test_full[features_all[0:num]]

                y_true = cat[test_index]
                start_time = time.time()
                (y_pred_rbf, y_tr_rbf, c_val_rbf[i], gamma_val_rbf[i], best_score_rbf[i]) = cl.SVM_classify_rbf_all(
                    train, y_train, test, y_true, C_range=np.logspace(-3, 5, 10), gamma_range=np.logspace(-6, 3, 10))
                (y_pred_lin, y_tr_lin, c_val_lin[i], best_score_lin[i]) = cl.SVM_classify_lin_all(train, y_train, test,
                     y_true, C_range=np.logspace(-2, 0, 6))
                print("--- %s seconds for classification ---" % (time.time() - start_time))
                pred_train = pd.DataFrame(
                    {'y_train': y_train,
                     'y_tr_rbf': y_tr_rbf,
                     'y_tr_lin': y_tr_lin,
                     })
                pickle.dump(pred_train,
                            open(OPEN_FILE + "/pred_tr_CV_%s_%s_%d_%d.p" % (TISSUE, feat_sel, num, i), "wb"))
                svm_accuracy_tr[i] = [
                    np.where((pred_train['y_train'] == pred_train['y_tr_rbf']) == True)[0].shape[0] / samples_tr,
                    np.where((pred_train['y_train'] == pred_train['y_tr_lin']) == True)[0].shape[0] / samples_tr]
                print(svm_accuracy_tr[i])
                predictions = pd.DataFrame(
                    {'y_true': y_true,
                     'y_rbf': y_pred_rbf,
                     'y_lin': y_pred_lin,
                     })
                pickle.dump(predictions, open(OPEN_FILE + "/pred_CV_%s_%s_%d_%d.p" % (TISSUE, feat_sel, num, i), "wb"))
                svm_accuracy[i] = [
                    np.where((predictions['y_true'] == predictions['y_rbf']) == True)[0].shape[0] / samples,
                    np.where((predictions['y_true'] == predictions['y_lin']) == True)[0].shape[0] / samples]

                print(svm_accuracy[i])

            pickle.dump(svm_accuracy_tr, open(OPEN_FILE + "/accuracy_tr_CV_%s_%s_%d.p" % (TISSUE, feat_sel, num), "wb"))
            pickle.dump(svm_accuracy, open(OPEN_FILE + "/accuracy_CV_%s_%s_%d.p" % (TISSUE, feat_sel, num), "wb"))
            parameters = pd.DataFrame(
                {'C_rbf': c_val_rbf,
                 'gamma_rbf': gamma_val_rbf,
                 'C_lin': c_val_lin,
                 'best_rbf': best_score_rbf,
                 'best_lin': best_score_lin,
                 })
            pickle.dump(parameters, open(OPEN_FILE + "/params_CV_%s_%s_%d.p" % (TISSUE, feat_sel, num), "wb"))


# ------------------- Function -------------------------
# load_data()
# This function load the data
# returns: (betaqn, info)
#          betaqn - DNA methylation from the TISSUE
#          info - info from DNA methylation data
# ----------------------------------------------------
def load_data():
    EC_data = pickle.load(open('../tissues/resi_norm_EC.p', "rb"))
    FC_data = pickle.load(open('../tissues/resi_norm_FC.p', "rb"))
    STG_data = pickle.load(open('../tissues/resi_norm_STG.p', "rb"))
    CER_data = pickle.load(open('../tissues/resi_norm_CER.p', "rb"))
    WB_data = pickle.load(open('../tissues/resi_norm_WB.p', "rb"))
    frames = [EC_data, FC_data, STG_data, CER_data, WB_data]
    betaqn = pd.concat(frames)
    info = pd.read_csv('info.csv.zip',index_col=1, compression='zip',sep=',')
    info = info.drop('Unnamed: 0', 1)
    info.loc[(info.braak_stage=='5') | (info.braak_stage=='6'),'braak_bin'] = 1
    cond = ((info.braak_stage=='0') | (info.braak_stage=='1') | (info.braak_stage=='2') |
            (info.braak_stage=='3') | (info.braak_stage=='4'))
    info.loc[cond,'braak_bin'] = 0
    info.loc[info.source_tissue == 'entorhinal cortex', 'tissue'] = 'EC'
    info.loc[info.source_tissue == 'whole blood', 'tissue'] = 'WB'
    info.loc[info.source_tissue == 'frontal cortex', 'tissue'] = 'FC'
    info.loc[info.source_tissue == 'superior temporal gyrus', 'tissue'] = 'STG'
    info.loc[info.source_tissue == 'cerebellum', 'tissue'] = 'CER'
    infoExtra = pd.read_csv('infoExtra.csv.zip', index_col=8, compression='zip', sep=',')
    info.loc[info.index == infoExtra.index, 'subject'] = infoExtra['subjectid']
    return betaqn, info

# ------------------- Function -------------------------
# main()
# ----------------------------------------------------
def main():
    general_all()

if __name__ == '__main__':
	main()

