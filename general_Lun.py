# ----------------------------------------------------
# Dissertation MSc CSML
# Author: Diana Carolina Montanes Mondragon
# ----------------------------------------------------
# file_name: general_Lun.py
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

# ------------------- Constant -------------------------
OPEN_FILE = os.path.realpath('../data_str/')
TISSUES = ['EC', 'CER', 'WB', 'FC', 'STG']
FEATURES_SEL = ['t_test', 'fisher', 'rfe', 'leo', 'leo_all', 'jager']
FEATURES_NUM = [5, 10, 15, 20, 50, 75, 100]
CV_SPLITS = 5
# ----------------------------------------------------


# ------------------- Function -------------------------
# general_Lun()
# This function performs the nested cross-validation for
# groups of features using all the selection of data
# ----------------------------------------------------
def general_Lun():
    tissue = 'all_non'
    num = 12
    for feat_sel in FEATURES_SEL:
        for tissue in TISSUES:
            save_file = OPEN_FILE
            betaqn, info = load_data(tissue)
            ec = betaqn
            start_time = time.time()
            features_file = OPEN_FILE + "/features_LEO_CV_%s_%s_%d.p" % (tissue, feat_sel, num)
            if feat_sel == 't_test':
                features_all = fs.feature_sel_t_test_parallel(ec, info, num)
            elif feat_sel == 'fisher':
                features_all = fs.feature_fisher_score_parallel(ec, info, num)
            elif feat_sel == 'rfe':
                features_all = fs.feature_sel_rfe(ec, info, num)
            elif feat_sel == 'leo':
                if tissue == 'EC':
                    features_all = ['cg11823178', 'cg22997194', 'cg06653632', 'cg05066959',
                                    'cg24152732', 'cg14972141', 'cg04029027', 'cg05030077',
                                    'cg04151012', 'cg18522315', 'cg20618448', 'cg24770624']
                elif tissue == 'STG':
                    features_all = ['cg04525464', 'cg06108383', 'cg10752406', 'cg25018458',
                                    'cg13942103', 'cg00767503', 'cg02961798', 'cg05810363',
                                    'cg15849154', 'cg06745695', 'cg15520955', 'cg03601797']
                elif tissue == 'FC':
                    features_all = ['cg04147621', 'cg05726109', 'cg11724984', 'cg23968456',
                                    'cg24671734', 'cg06926306', 'cg07859799', 'cg02997560',
                                    'cg13507269', 'cg19900677', 'cg14071588', 'cg15928398']
                elif tissue == 'CER':
                    features_all = ['cg22570053', 'cg00065957', 'cg21781422', 'cg17715556',
                                    'cg01339004', 'cg18882687', 'cg20767910', 'cg24462001',
                                    'cg07869256', 'cg20698501', 'cg21618635', 'cg17468317']
                else:
                    continue
            elif feat_sel == 'leo_all':
                leo_all = ['cg11823178', 'cg25018458', 'cg05810363', 'cg05066959',
                           'cg18428542', 'cg16665310', 'cg05912299', 'cg03169557',
                           'cg23968456', 'cg02672452', 'cg04147621', 'cg17910899']
            elif feat_sel == 'jager':
                features_all = ['cg11724984', 'cg23968456', 'cg15821544', 'cg16733298',
                                'cg22962123', 'cg13076843', 'cg25594100', 'cg00621289',
                                'cg19803550', 'cg03169557', 'cg05066959', 'cg05810363']
            print("--- %s seconds for feature selection ---" % (time.time() - start_time))
            pickle.dump(features_all, open(features_file, "wb"))

            cat = info['braak_bin'].loc[ec.index]

            samples = ec.shape[0]
            nzeros = np.where(cat == 0)[0]
            nones = np.where(cat == 1)[0]
            svm_accuracy = {}
            svm_accuracy_tr = {}

            c_val_rbf = np.zeros(CV_SPLITS)
            gamma_val_rbf = np.zeros(CV_SPLITS)
            c_val_lin = np.zeros(CV_SPLITS)
            best_score_rbf = np.zeros(CV_SPLITS)
            best_score_lin = np.zeros(CV_SPLITS)

            zeros = np.random.permutation(nzeros)
            ones = np.random.permutation(nones)
            for i in range(CV_SPLITS):
                print('split: %d - num_features: %d - tissue:%s- feat_sel:%s' % (i, num, tissue, feat_sel))
                test_index, train_index = ut.get_intervals(CV_SPLITS, i, zeros, ones)
                train_full = ec.iloc[train_index]
                test_full = ec.iloc[test_index]
                samples = test_full.shape[0]
                samples_tr = train_full.shape[0]

                train = train_full[features_all]
                test = test_full[features_all]
                y_train = info['braak_bin'].loc[train.index]
                y_true = cat[test_index]
                start_time = time.time()
                (y_pred_rbf, y_tr_rbf, c_val_rbf[i], gamma_val_rbf[i], best_score_rbf[i]) = cl.SVM_classify_rbf_all(
                    train, y_train, test, y_true, C_range=np.logspace(-4, 4, 10), gamma_range=np.logspace(-5, 2, 10))
            (y_pred_lin, y_tr_lin, c_val_lin[i], best_score_lin[i]) = cl.SVM_classify_lin_all(train, y_train, test,
                    y_true, C_range=np.logspace(-4, 3, 10))
            print("--- %s seconds for classification ---" % (time.time() - start_time))
            parameters = pd.DataFrame(
                {'C_rbf': c_val_rbf,
                 'gamma_rbf': gamma_val_rbf,
                 'C_lin': c_val_lin,
                 'best_rbf': best_score_rbf,
                 'best_lin': best_score_lin,
                 })
            pickle.dump(parameters, open(save_file + "/params_LEO2_%s_%s_%d.p" % (tissue, feat_sel, i), "wb"))
            predictions = pd.DataFrame(
                {'y_true': y_true,
                 'y_rbf': y_pred_rbf,
                 'y_lin': y_pred_lin,
                 })
            pickle.dump(predictions, open(save_file + "/pred_LEO2_%s_%s_%d.p" % (tissue, feat_sel, i), "wb"))
            pred_train = pd.DataFrame(
                {'y_train': y_train,
                 'y_tr_rbf': y_tr_rbf,
                 'y_tr_lin': y_tr_lin,
                 })
            pickle.dump(pred_train, open(save_file + "/pred_LEO2_tr_%s_%s_%d.p" % (tissue, feat_sel, i), "wb"))
            svm_accuracy[i] = [np.where((predictions['y_true'] == predictions['y_rbf']) == True)[0].shape[0] / samples,
                               np.where((predictions['y_true'] == predictions['y_lin']) == True)[0].shape[0] / samples]
            svm_accuracy_tr[i] = [
                np.where((pred_train['y_train'] == pred_train['y_tr_rbf']) == True)[0].shape[0] / samples_tr,
                np.where((pred_train['y_train'] == pred_train['y_tr_lin']) == True)[0].shape[0] / samples_tr]
            print(svm_accuracy[i])
            print(svm_accuracy_tr[i])
        pickle.dump(svm_accuracy, open(save_file + "/accuracy_LEO2_%s_%s.p" % (tissue, feat_sel), "wb"))
        pickle.dump(svm_accuracy_tr, open(save_file + "/accuracy_LEO2_tr_%s_%s.p" % (tissue, feat_sel), "wb"))


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
    general_Lun()

if __name__ == '__main__':
	main()
