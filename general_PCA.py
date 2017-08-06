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
from sklearn.decomposition import PCA
from sklearn import preprocessing

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
    #tissues=['WB', 'FC', 'STG']
    betaqn, info = load_data()
    for tissue in tissues:
        save_file = os.path.realpath('../data_str/')
        iters_big = 10
        iters_small = 30
        big_small = 200
        feat_sel = 'PCA'

        ec = betaqn.loc[info[(info.tissue == tissue) & (info.braak_stage != 'Exclude')].index]
        svm_accuracy = {}
        samples = ec.shape[0]

        features_num = [100000, 50000, 1000, 500, 250, 100, 75, 50, 20]
        #features_num = [200000, 100000, 50000, 1000, 500, 250, 100, 75, 50, 20, 10]
        #features_num = [500, 250, 100, 75, 50, 20, 10]

        for num in features_num:
            print(num)
            y_true = np.zeros(samples)
            y_pred_rbf = np.zeros(samples)
            c_val_rbf = np.zeros(samples)
            gamma_val_rbf = np.zeros(samples)
            y_pred_lin = np.zeros(samples)
            c_val_lin = np.zeros(samples)
            y_pred_pol = np.zeros(samples)
            c_val_pol = np.zeros(samples)
            gamma_val_pol = np.zeros(samples)
            for i in range(samples):
                print('iteracion %d para %d features' %(i,num))
                train_full = ec.loc[ec.index != ec.index[i]]
                test_full = ec.loc[ec.index == ec.index[i]]
                y_train = info['braak_bin'].loc[train_full.index]
                y_true[i] = info['braak_bin'].loc[test_full.index]
                #SCALING
                scale = preprocessing.StandardScaler().fit(train_full)
                train_sc = scale.transform(train_full)
                test_sc = scale.transform(test_full)
                #PCA
                pca = PCA(n_components=num)
                pca.fit(train_sc)
                train = pca.transform(train_sc)
                test = pca.transform(test_sc)
                if(((i < iters_big) & (num > big_small)) | ((i < iters_small) & (num < big_small))):
                    print('entro primeros')
                    if(num==10):
                        (y_pred_pol[i], c_val_pol[i], gamma_val_pol[i]) = cl.SVM_classify_poly(train, y_train, test,
                        C_range = np.logspace(-1, 10, 10),gamma_range = np.logspace(-3, 3, 8))
                    else:
                        (y_pred_pol[i], c_val_pol[i], gamma_val_pol[i]) = cl.SVM_classify_poly(train, y_train, test)
                    (y_pred_rbf[i], c_val_rbf[i], gamma_val_rbf[i]) = cl.SVM_classify_rbf(train, y_train, test)
                    (y_pred_lin[i], c_val_lin[i]) = cl.SVM_classify_lin(train, y_train, test)
                elif((i >= iters_big) & (num > big_small)):
                    print('entro big iters')
                    (y_pred_rbf[i], c_val_rbf[i], gamma_val_rbf[i]) = cl.SVM_classify_rbf(train, y_train, test,
                    C_range = np.unique(c_val_rbf[0:iters_big]),gamma_range = np.unique(gamma_val_rbf[0:iters_big]))
                    (y_pred_pol[i], c_val_pol[i], gamma_val_pol[i]) = cl.SVM_classify_poly(train, y_train, test,
                    C_range = np.unique(c_val_pol[0:iters_big]),gamma_range = np.unique(gamma_val_pol[0:iters_big]))
                    (y_pred_lin[i], c_val_lin[i]) = cl.SVM_classify_lin(train, y_train, test,C_range = np.unique(c_val_lin[0:iters_big]))
                elif((i >= iters_small) & (num < big_small)):
                    print('entro small iters')
                    (y_pred_rbf[i], c_val_rbf[i], gamma_val_rbf[i]) = cl.SVM_classify_rbf(train, y_train, test,
                    C_range = np.unique(c_val_rbf[0:iters_small]),gamma_range = np.unique(gamma_val_rbf[0:iters_small]))
                    (y_pred_pol[i], c_val_pol[i], gamma_val_pol[i]) = cl.SVM_classify_poly(train, y_train, test,
                    C_range = np.unique(c_val_pol[0:iters_small]),gamma_range = np.unique(gamma_val_pol[0:iters_small]))
                    (y_pred_lin[i], c_val_lin[i]) = cl.SVM_classify_lin(train, y_train, test,C_range = np.unique(c_val_lin[0:iters_small]))
            predictions = pd.DataFrame(
            {'y_true': y_true,
             'y_rbf': y_pred_rbf,
             'y_poly': y_pred_pol,
             'y_lin': y_pred_lin,
            })
            pickle.dump(predictions, open(save_file + "/pred_%s_%s_%d.p" %(tissue, feat_sel, num), "wb"))
            #pickle.dump(features_sel, open(save_file + "/feat_%s_%s_%d.p" %(tissue, feat_sel, num), "wb"))
            #features_sel_total = {key: value + [features_sel[key]] for key, value in features_sel_total.items()}
            svm_accuracy[num] = [np.where((predictions['y_true']==predictions['y_rbf'])==True)[0].shape[0]/samples,
                                np.where((predictions['y_true']==predictions['y_poly'])==True)[0].shape[0]/samples,
                                np.where((predictions['y_true']==predictions['y_lin'])==True)[0].shape[0]/samples]
        pickle.dump(svm_accuracy, open(save_file + "/accuracy_%s_%s.p" % (tissue, feat_sel), "wb"))



if __name__ == '__main__':
	main()