# ------------
# Diana Carolina Montañes M.
# train_blood: train in blood, predict each tissue


from __future__ import division
import time
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
import pickle
import classification as cl
import feature_selection as fs
import os.path
from zipfile import ZipFile
import sys, os
from os.path import join, dirname, abspath
from pathlib import Path


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

def train_blood_itsfeatures(betaqn, info, feat_sel = 't_test'):
    tissues=['EC', 'CER', 'FC', 'STG']
    features_num = [50000, 1000, 500, 250, 100, 75, 50, 20]
    for tissue in tissues:
        save_file = os.path.realpath('../data_str/')

        ec = betaqn.loc[info[(info.tissue == tissue) & (info.braak_stage != 'Exclude')].index]
        blood = betaqn.loc[info[(info.tissue == 'WB') & (info.braak_stage != 'Exclude')].index]
        svm_accuracy = {}
        samples = ec.shape[0]
        cat = info['braak_bin'].loc[blood.index]

        if feat_sel == 't_test' or feat_sel == 'fisher' :
            features_file = "../DATA/predict_on_blood/features_blood_%s_%s.p" % (tissue, feat_sel)
            my_file = Path(features_file)
            if my_file.is_file():
                features_all = pickle.load( open( features_file, "rb" ) )
            else:
                print('No features file')


        for num in features_num:
            if feat_sel == 'rfe':
                for num in features_num:
                    features_file = "../DATA/predict_on_blood/features_blood_%s_%s_%d.p" % (tissue, feat_sel,num)
                    my_file = Path(features_file)
                    if my_file.is_file():
                        features_all = pickle.load( open( features_file, "rb" ) )
                    else:
                        print('No features file')

            skf = StratifiedKFold(n_splits=10, random_state=11)
            i = 0
            for train_index, test_index in skf.split(blood, cat):
                cur_ec = blood.iloc[train_index]
                cur_cat = cat[train_index]
                y_pred_rbf = np.zeros(samples)
                y_pred_lin = np.zeros(samples)

                train_full = cur_ec
                train = train_full[features_all[0:num]]
                test = ec
                test = test[features_all[0:num]]
                y_train = cur_cat
                y_true = info['braak_bin'].loc[test.index]

                (y_pred_rbf, c_rbf, gamma_rbf) = cl.SVM_classify_rbf_all(train, y_train, test)
                (y_pred_lin, c_lin) = cl.SVM_classify_lin_all(train, y_train, test)

                predictions = pd.DataFrame(
                {'y_true': y_true,
                 'y_rbf': y_pred_rbf,
                 'y_lin': y_pred_lin,
                })
                pickle.dump(predictions, open(save_file + "/pred_train_blood_otherFeat_%s_%s_%d.p" %(tissue, feat_sel, num), "wb"))
                #pickle.dump(features_sel, open(save_file + "/feat_%s_%s_%d.p" %(tissue, feat_sel, num), "wb"))
                #features_sel_total = {key: value + [features_sel[key]] for key, value in features_sel_total.items()}
                svm_accuracy[num] = [np.where((predictions['y_true']==predictions['y_rbf'])==True)[0].shape[0]/samples,
                                    np.where((predictions['y_true']==predictions['y_lin'])==True)[0].shape[0]/samples]
            pickle.dump(svm_accuracy, open(save_file + "/accuracy_train_blood_otherFeat_%s_%s.p" % (tissue, feat_sel), "wb"))

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

            print('iteracion %d features' %(num))
            train_full = blood
            start_time = time.time()
            train = train_full[features_all[0:num]]
            print("--- %s seconds for feature selection ---" % (time.time() - start_time))
            print('features selected')
            test = ec
            test = test[features_all[0:num]]
            y_train = info['braak_bin'].loc[train.index]
            y_true = info['braak_bin'].loc[test.index]

            (y_pred_rbf, c_rbf, gamma_rbf) = cl.SVM_classify_rbf_all(train, y_train, test)
            (y_pred_pol, c_pol, gamma_pol) = cl.SVM_classify_poly_all(train, y_train, test)
            (y_pred_lin, c_lin) = cl.SVM_classify_lin_all(train, y_train, test)

            predictions = pd.DataFrame(
            {'y_true': y_true,
             'y_rbf': y_pred_rbf,
             'y_poly': y_pred_pol,
             'y_lin': y_pred_lin,
            })
            pickle.dump(predictions, open(save_file + "/pred_train_blood_otherFeat_%s_%s_%d.p" %(tissue, feat_sel, num), "wb"))
            #pickle.dump(features_sel, open(save_file + "/feat_%s_%s_%d.p" %(tissue, feat_sel, num), "wb"))
            #features_sel_total = {key: value + [features_sel[key]] for key, value in features_sel_total.items()}
            svm_accuracy[num] = [np.where((predictions['y_true']==predictions['y_rbf'])==True)[0].shape[0]/samples,
                                np.where((predictions['y_true']==predictions['y_poly'])==True)[0].shape[0]/samples,
                                np.where((predictions['y_true']==predictions['y_lin'])==True)[0].shape[0]/samples]
        pickle.dump(svm_accuracy, open(save_file + "/accuracy_train_blood_otherFeat_%s_%s.p" % (tissue, feat_sel), "wb"))



def main():
    betaqn, info = load_data()
    for feat_sel in ['rfe','t_test','fisher']:
        train_blood_itsfeatures(betaqn, info, feat_sel)


if __name__ == '__main__':
	main()