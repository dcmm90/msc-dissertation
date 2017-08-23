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
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA
from sklearn import preprocessing


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


def load_data():
    EC_data = pickle.load( open( '../tissues/resi_norm_EC.p', "rb" ) )
    FC_data = pickle.load( open( '../tissues/resi_norm_FC.p', "rb" ) )
    STG_data = pickle.load( open( '../tissues/resi_norm_STG.p', "rb" ) )
    CER_data = pickle.load( open( '../tissues/resi_norm_CER.p', "rb" ) )
    WB_data = pickle.load( open( '../tissues/resi_norm_WB.p', "rb" ) )
    frames = [EC_data,FC_data,STG_data,CER_data,WB_data]
    betaqn = pd.concat(frames)
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
    open_file = os.path.realpath('../data_str/')
    tissue = 'all'
    ec, info = load_data()
    #'t_test','fisher','rfe'
    features_sel = ['t_test','fisher','rfe','PCA']
    #features_num = [20,50,75,100,250,500,1000,]
    #features_num = [20,50,75,100,250,500,1000,5000,10000,100000]
    features_num = [5,10,15,20,50,75,100,250,500,1000]
    #features_num = [5,10,15,20,50]
    #features_num = [10]
    for feat_sel in features_sel:
        print('cargo datos')
        #ec = betaqn.loc[info[(info.tissue == tissue) & (info.braak_stage != 'Exclude')].index]
        cat = info['braak_bin'].loc[ec.index]
        svm_accuracy = {}
        samples = ec.shape[0]
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
                print('gen_all -split: %d - num_features: %d - feat_sel:%s' %(i,num,feat_sel))
                test_index, train_index = get_intervals(cv_splits, i, zeros, ones)
                print('tamaño de test: %s'%len(test_index))
                train_full = ec.iloc[train_index]
                y_train = cat[train_index]
                test_full = ec.iloc[test_index]
                samples = test_full.shape[0]
                samples_tr = train_full.shape[0]
                start_time = time.time()
                features_file = open_file + "/features_CV_%s_%s_%d_%d.p" % (tissue, feat_sel, num, i)
                print(train_full.shape)
                if feat_sel == 't_test':
                    features_all = fs.feature_sel_t_test_parallel(train_full, info, num)
                elif feat_sel == 'fisher':
                    features_all = fs.feature_fisher_score_parallel(train_full, info, num)
                elif feat_sel == 'rfe':
                    features_all = fs.feature_sel_rfe(train_full, info, num)
                #elif feat_sel == 'chi2':

                print("--- %s seconds for feature selection ---" % (time.time() - start_time))
                pickle.dump(features_all, open(features_file, "wb"))

                if feat_sel == 'PCA':
                    #SCALING
                    scale = preprocessing.StandardScaler().fit(train_full)
                    train_sc = scale.transform(train_full)
                    test_sc = scale.transform(test_full)
                    #PCA
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
                (y_pred_rbf, y_tr_rbf, c_val_rbf[i], gamma_val_rbf[i],best_score_rbf[i]) = cl.SVM_classify_rbf_all(train, y_train,test,y_true,
                C_range = np.logspace(-3, 5, 10),gamma_range = np.logspace(-6, 3, 10))
                (y_pred_lin, y_tr_lin, c_val_lin[i], best_score_lin[i]) = cl.SVM_classify_lin_all(train, y_train, test, y_true,
                C_range = np.logspace(-3, 5, 10))
                print("--- %s seconds for classification ---" % (time.time() - start_time))
                pred_train = pd.DataFrame(
                {'y_train': y_train,
                 'y_tr_rbf': y_tr_rbf,
                 'y_tr_lin': y_tr_lin,
                })
                pickle.dump(pred_train, open(open_file + "/pred_tr_CV_%s_%s_%d_%d.p" %(tissue, feat_sel, num, i), "wb"))
                svm_accuracy_tr[i] = [np.where((pred_train['y_train']==pred_train['y_tr_rbf'])==True)[0].shape[0]/samples_tr,
                                    np.where((pred_train['y_train']==pred_train['y_tr_lin'])==True)[0].shape[0]/samples_tr]
                print(svm_accuracy_tr[i])
                predictions = pd.DataFrame(
                {'y_true': y_true,
                 'y_rbf': y_pred_rbf,
                 'y_lin': y_pred_lin,
                })
                pickle.dump(predictions, open(open_file + "/pred_CV_%s_%s_%d_%d.p" %(tissue, feat_sel, num, i), "wb"))
                svm_accuracy[i] = [np.where((predictions['y_true']==predictions['y_rbf'])==True)[0].shape[0]/samples,
                                    np.where((predictions['y_true']==predictions['y_lin'])==True)[0].shape[0]/samples]

                print(svm_accuracy[i])

            pickle.dump(svm_accuracy_tr, open(open_file + "/accuracy_tr_CV_%s_%s_%d.p" % (tissue, feat_sel,num), "wb"))
            pickle.dump(svm_accuracy, open(open_file + "/accuracy_CV_%s_%s_%d.p" % (tissue, feat_sel,num), "wb"))
            parameters = pd.DataFrame(
            {'C_rbf': c_val_rbf,
             'gamma_rbf': gamma_val_rbf,
             'C_lin': c_val_lin,
             'best_rbf': best_score_rbf,
             'best_lin': best_score_lin,
            })
            pickle.dump(parameters, open(open_file + "/params_CV_%s_%s_%d.p" %(tissue, feat_sel, num), "wb"))


if __name__ == '__main__':
	main()
