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
    tissues=['CER', 'WB', 'FC', 'STG']
    betaqn, info = load_data()
    cv_splits = 10
    features = ['cg11724984', 'cg23968456', 'cg15821544', 'cg16733298', 'cg22962123',
                'cg13076843', 'cg25594100', 'cg00621289', 'cg19803550', 'cg03169557',
                'cg05066959', 'cg05810363', 'cg22883290', 'cg02308560', 'cg11823178']
    for tissue in tissues:
        save_file = os.path.realpath('../data_str/')

        ec = betaqn.loc[info[(info.tissue == tissue) & (info.braak_stage != 'Exclude')].index]
        cat = info['braak_bin'].loc[ec.index]
        svm_accuracy = {}
        samples = ec.shape[0]
        nzeros = np.where(cat == 0)[0]
        nones = np.where(cat == 1)[0]
        div_zeros = np.ceil(len(nzeros)/cv_splits)
        div_ones = np.ceil(len(nones)/cv_splits)
        svm_accuracy = {}
        svm_accuracy_tr = {}
        samples = ec.shape[0]

        c_val_rbf = np.zeros(cv_splits)
        gamma_val_rbf = np.zeros(cv_splits)
        c_val_lin = np.zeros(cv_splits)
        c_val_pol = np.zeros(cv_splits)
        gamma_val_pol = np.zeros(cv_splits)
        svm_accuracy = {}
        zeros = np.random.permutation(nzeros)
        ones = np.random.permutation(nones)
        for i in range(cv_splits):
            print('split: %d' %(i))
            test_index, train_index = get_intervals(cv_splits, i, zeros, ones)
            train_full = ec.iloc[train_index]
            y_train = cat[train_index]
            test_full = ec.iloc[test_index]
            samples = test_full.shape[0]
            samples_tr = train_full.shape[0]

            train = train_full[features]
            test = test_full[features]
            y_train = info['braak_bin'].loc[train.index]
            y_true = cat[test_index]
            start_time = time.time()
            (y_pred_pol, y_tr_pol, c_val_pol[i], gamma_val_pol[i]) = cl.SVM_classify_poly_all(train, y_train, test, y_true,
            C_range = [0.01,0.05,0.1,0.5,1],gamma_range = [0.5,1,1.5])
            (y_pred_rbf, y_tr_rbf, c_val_rbf[i], gamma_val_rbf[i]) = cl.SVM_classify_rbf_all(train, y_train,test,y_true)
            (y_pred_lin, y_tr_lin, c_val_lin[i]) = cl.SVM_classify_lin_all(train, y_train, test, y_true)
            print("--- %s seconds for classification ---" % (time.time() - start_time))
            parameters = pd.DataFrame(
            {'C_rbf': c_val_rbf,
             'gamma_rbf': gamma_val_rbf,
             'C_poly': c_val_pol,
             'gamma_poly': gamma_val_pol,
             'C_lin': c_val_lin
            })
            pickle.dump(parameters, open(save_file + "/params_LEO_%s_%d.p" %(tissue,i), "wb"))
            predictions = pd.DataFrame(
            {'y_true': y_true,
             'y_rbf': y_pred_rbf,
             'y_poly': y_pred_pol,
             'y_lin': y_pred_lin,
            })
            pickle.dump(predictions, open(save_file + "/pred_LEO_%s_%d.p" %(tissue,i), "wb"))
            pred_train = pd.DataFrame(
            {'y_train': y_train,
             'y_tr_rbf': y_tr_rbf,
             'y_tr_poly': y_tr_pol,
             'y_tr_lin': y_tr_lin,
            })
            pickle.dump(pred_train, open(save_file + "/pred_LEO_tr_%s_%d.p" %(tissue, i), "wb"))
            #pickle.dump(features_sel, open(save_file + "/feat_%s_%s_%d.p" %(tissue, feat_sel, num), "wb"))
            #features_sel_total = {key: value + [features_sel[key]] for key, value in features_sel_total.items()}
            svm_accuracy[i] = [np.where((predictions['y_true']==predictions['y_rbf'])==True)[0].shape[0]/samples,
                                np.where((predictions['y_true']==predictions['y_poly'])==True)[0].shape[0]/samples,
                                np.where((predictions['y_true']==predictions['y_lin'])==True)[0].shape[0]/samples]
            svm_accuracy_tr[i] = [np.where((pred_train['y_train']==pred_train['y_tr_rbf'])==True)[0].shape[0]/samples_tr,
                                np.where((pred_train['y_train']==pred_train['y_tr_poly'])==True)[0].shape[0]/samples_tr,
                                np.where((pred_train['y_train']==pred_train['y_tr_lin'])==True)[0].shape[0]/samples_tr]
            print(svm_accuracy_tr[i])
        pickle.dump(svm_accuracy, open(save_file + "/accuracy_LEO_%s.p" % (tissue), "wb"))
        pickle.dump(svm_accuracy_tr, open(save_file + "/accuracy_LEO_tr_%s.p" % (tissue), "wb"))



if __name__ == '__main__':
	main()
