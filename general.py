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

from os.path import join, dirname, abspath



def main():

    tissue='EC'
    feat_sel = 't_test'
    beta_file = os.path.realpath('../GSE59685_betas2.csv.zip')
    save_file = os.path.realpath('../data_str/')
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

    sample_barcode = []
    ec = betaqn.loc[info[(info.tissue == tissue) & (info.braak_stage != 'Exclude')].index]
    features_sel_total = dict.fromkeys(list(ec),[0])
    svm_accuracy = {}
    samples = ec.shape[0]
    features_num = [200000,50000,1000,500,100,20,10,5]
    for num in features_num:
        print(num)
        features_sel = dict.fromkeys(list(ec),0)
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
            print(i)
            train_full = ec.loc[ec.index != ec.index[i]]
            sample_barcode.append(ec.index[i])
            start_time = time.time()
            if feat_sel == 't_test':
                features, train = fs.feature_sel_t_test_parallel(train_full, info, num)
            elif feat_sel == 'fisher':
                features, train = fs.feature_fisher_score(train_full, info, num)
            for elem in features:
                features_sel[elem] +=1
            print("--- %s seconds for feature selection ---" % (time.time() - start_time))
            print('features selected')
            test = ec.loc[ec.index == ec.index[i]]
            test = test.loc[:,features]
            y_train = info['braak_bin'].loc[train.index]
            y_true[i] = info['braak_bin'].loc[test.index]
            (y_pred_rbf[i], c_val_rbf[i], gamma_val_rbf[i], y_pred_pol[i],
            c_val_pol[i], gamma_val_pol[i], y_pred_lin[i], c_val_lin[i]) = cl.SVM_classify(train, y_train, test)
        parameters = pd.DataFrame(
        {'C_rbf': c_val_rbf,
         'gamma_rbf': gamma_val_rbf,
         'C_poly': c_val_pol,
         'gamma_poly': gamma_val_pol,
         'C_lin': c_val_lin
        }, index = sample_barcode)
        pickle.dump(parameters, open(save_file + "/params_%s_%s_%d.p" %(tissue, feat_sel, num), "wb"))
        predictions = pd.DataFrame(
        {'y_true': y_true,
         'y_rbf': y_pred_rbf,
         'y_poly': y_pred_pol,
         'y_lin': y_pred_lin,
        }, index = sample_barcode)
        pickle.dump(predictions, open(save_file + "/pred_%s_%s_%d.p" %(tissue, feat_sel, num), "wb"))
        features_sel_total = {key: value + [features_sel[key]] for key, value in features_sel_total.iteritems()}
        svm_accuracy[num] = [np.where((predictions['y_true']==predictions['y_rbf'])==True)[0].shape[0]/samples,
                            np.where((predictions['y_true']==predictions['y_poly'])==True)[0].shape[0]/samples,
                            np.where((predictions['y_true']==predictions['y_lin'])==True)[0].shape[0]/samples]
    pickle.dump(features_sel_total, open(save_file + "/features_%s_%s.p" % (tissue, feat_sel), "wb"))
    pickle.dump(svm_accuracy, open(save_file + "/accuracy_%s_%s.p" % (tissue, feat_sel), "wb"))



if __name__ == '__main__':
	main()
