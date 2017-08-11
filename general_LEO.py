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


def main():
    tissues=['CER', 'WB', 'FC', 'STG']
    betaqn, info = load_data()
    for tissue in tissues:
        save_file = os.path.realpath('../data_str/')
        iters_big = 15

        ec = betaqn.loc[info[(info.tissue == tissue) & (info.braak_stage != 'Exclude')].index]
        svm_accuracy = {}
        samples = ec.shape[0]

        features = ['cg11724984', 'cg23968456', 'cg15821544', 'cg16733298', 'cg22962123',
                    'cg13076843', 'cg25594100', 'cg00621289', 'cg19803550', 'cg03169557',
                    'cg05066959', 'cg05810363', 'cg22883290', 'cg02308560', 'cg11823178']

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
            print('iteracion %d para features' %(i))
            train_full = ec.loc[ec.index != ec.index[i]]
            start_time = time.time()
            train = train_full[features]
            print("--- %s seconds for feature selection ---" % (time.time() - start_time))
            print('features selected')
            test = ec.loc[ec.index == ec.index[i]]
            test = test[features]
            y_train = info['braak_bin'].loc[train.index]
            y_true[i] = info['braak_bin'].loc[test.index]
            if(i < iters_big):
                (y_pred_pol[i], c_val_pol[i], gamma_val_pol[i]) = cl.SVM_classify_poly(train, y_train, test,
                C_range = [0.01,0.1,0.5,1],gamma_range = [0.5,1,1.5])
                (y_pred_rbf[i], c_val_rbf[i], gamma_val_rbf[i]) = cl.SVM_classify_rbf(train, y_train, test)
                (y_pred_lin[i], c_val_lin[i]) = cl.SVM_classify_lin(train, y_train, test)
            else:
                print('entro big iters')
                (y_pred_rbf[i], c_val_rbf[i], gamma_val_rbf[i]) = cl.SVM_classify_rbf(train, y_train, test,
                C_range = np.unique(c_val_rbf[0:iters_big]),gamma_range = np.unique(gamma_val_rbf[0:iters_big]))
                (y_pred_pol[i], c_val_pol[i], gamma_val_pol[i]) = cl.SVM_classify_poly(train, y_train, test,
                C_range = np.unique(c_val_pol[0:iters_big]),gamma_range = np.unique(gamma_val_pol[0:iters_big]))
                (y_pred_lin[i], c_val_lin[i]) = cl.SVM_classify_lin(train, y_train, test,C_range = np.unique(c_val_lin[0:iters_big]))
        parameters = pd.DataFrame(
        {'C_rbf': c_val_rbf,
         'gamma_rbf': gamma_val_rbf,
         'C_poly': c_val_pol,
         'gamma_poly': gamma_val_pol,
         'C_lin': c_val_lin
        })
        pickle.dump(parameters, open(save_file + "/params_LEO_%s.p" %(tissue), "wb"))
        predictions = pd.DataFrame(
        {'y_true': y_true,
         'y_rbf': y_pred_rbf,
         'y_poly': y_pred_pol,
         'y_lin': y_pred_lin,
        })
        pickle.dump(predictions, open(save_file + "/pred_LEO_%s.p" %(tissue), "wb"))
        #pickle.dump(features_sel, open(save_file + "/feat_%s_%s_%d.p" %(tissue, feat_sel, num), "wb"))
        #features_sel_total = {key: value + [features_sel[key]] for key, value in features_sel_total.items()}
        svm_accuracy = [np.where((predictions['y_true']==predictions['y_rbf'])==True)[0].shape[0]/samples,
                            np.where((predictions['y_true']==predictions['y_poly'])==True)[0].shape[0]/samples,
                            np.where((predictions['y_true']==predictions['y_lin'])==True)[0].shape[0]/samples]
        pickle.dump(svm_accuracy, open(save_file + "/accuracy_LEO_%s.p" % (tissue), "wb"))



if __name__ == '__main__':
	main()
