# ----------------------------------------------------
# Dissertation MSc CSML
# Author: Diana Carolina Montanes Mondragon
# ----------------------------------------------------
# file_name: blood_surr.py
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
import os.path
import os
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
    infoExtra = pd.read_csv('infoExtra.csv.zip',index_col=8, compression='zip',sep=',')
    info.loc[info.index == infoExtra.index, 'subject'] = infoExtra['subjectid']
    return (betaqn, info)


# ------------------- Function -------------------------
# blood_surr()
# This function performs the nested cross-validation for
# using blood as surrogate (experiment 4)
# ----------------------------------------------------
def blood_surr():
    tissues = ['EC', 'STG', 'CER', 'FC']
    for tissue in tissues:
        open_file = os.path.realpath('../data_str/')
        ec, info = load_data(tissue)
        blood = pickle.load(open('../tissues/resi_norm_WB.p', "rb"))
        features_sel = ['t_test', 'fisher', 'rfe']
        features_num = [5, 10, 15, 20, 50, 75, 100, 250, 500]
        for feat_sel in features_sel:

            cat = info['braak_bin'].loc[blood.index]
            nzeros = np.where(cat == 0)[0]
            nones = np.where(cat == 1)[0]
            cv_splits = 5

            for num in features_num:
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

                    train_blood = blood.iloc[train_index]
                    y_train = cat[train_index]
                    test_blood = blood.iloc[test_index]
                    samples = test_blood.shape[0]
                    samples_tr = train_blood.shape[0]
                    # get the index of the samples in the test set- we dont want to train with those subjects
                    rem = test_blood.index
                    unwanted = list(info['subject'].loc[rem])
                    valids = []
                    for ids in info.index:
                        if ((info['tissue'].loc[ids] == tissue) and (info['subject'].loc[ids] not in unwanted)
                            and (info['braak_stage'].loc[ids] != 'Exclude')):
                            valids.append(ids)
                    ec_train = ec.loc[valids]

                    start_time = time.time()
                    features_file = open_file + "/features_blood_CV_%s_%s_%d_%d.p" % (tissue, feat_sel, num, i)
                    print(ec_train.shape)
                    if feat_sel == 't_test':
                        features_all = fs.feature_sel_t_test_parallel(ec_train, info, num)
                    elif feat_sel == 'fisher':
                        features_all = fs.feature_fisher_score_parallel(ec_train, info, num)
                    elif feat_sel == 'rfe':
                        features_all = fs.feature_sel_rfe(ec_train, info, num)
                    # elif feat_sel == 'chi2':
                    print("--- %s seconds for feature selection ---" % (time.time() - start_time))
                    if feat_sel == 't_test' or feat_sel == 'fisher' or feat_sel == 'rfe':
                        pickle.dump(features_all, open(features_file, "wb"))
                    train = train_blood[features_all]
                    print(train.shape)
                    test = test_blood[features_all]

                    y_true = cat[test_index]

                    start_time = time.time()
                    (y_pred_rbf, y_tr_rbf, c_val_rbf[i], gamma_val_rbf[i], best_score_rbf[i]) = cl.SVM_classify_rbf_all(
                        train, y_train, test, y_true, C_range=np.logspace(-4, 2, 10), gamma_range=np.logspace(-6, 2, 10))
                    (y_pred_lin, y_tr_lin, c_val_lin[i], best_score_lin[i]) = cl.SVM_classify_lin_all(train, y_train,
                        test, y_true, C_range=np.logspace(-4, 2, 10))
                    print("--- %s seconds for classification ---" % (time.time() - start_time))
                    pred_train = pd.DataFrame(
                        {'y_train': y_train,
                         'y_tr_rbf': y_tr_rbf,
                         'y_tr_lin': y_tr_lin,
                         })
                    pickle.dump(pred_train,
                                open(open_file + "/pred_blood_tr_CV_%s_%s_%d_%d.p" % (tissue, feat_sel, num, i), "wb"))
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
                                open(open_file + "/pred_blood_CV_%s_%s_%d_%d.p" % (tissue, feat_sel, num, i), "wb"))
                    svm_accuracy[i] = [
                        np.where((predictions['y_true'] == predictions['y_rbf']) == True)[0].shape[0] / samples,
                        np.where((predictions['y_true'] == predictions['y_lin']) == True)[0].shape[0] / samples]

                    print(svm_accuracy[i])

                pickle.dump(svm_accuracy_tr,
                            open(open_file + "/accuracy_blood_tr_CV_%s_%s_%d.p" % (tissue, feat_sel, num), "wb"))
                pickle.dump(svm_accuracy,
                            open(open_file + "/accuracy_blood_CV_%s_%s_%d.p" % (tissue, feat_sel, num), "wb"))
                parameters = pd.DataFrame(
                    {'C_rbf': c_val_rbf,
                     'gamma_rbf': gamma_val_rbf,
                     'C_lin': c_val_lin,
                     'best_rbf': best_score_rbf,
                     'best_lin': best_score_lin,
                     })
                pickle.dump(parameters, open(open_file + "/params_CV_%s_%s_%d.p" % (tissue, feat_sel, num), "wb"))


def main():
    blood_surr()

if __name__ == '__main__':
	main()
