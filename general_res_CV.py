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


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


def load_data(tissue):
    #beta_file = os.path.realpath('../tissues/residuals_%s.csv.zip'%(tissue))
    #zipfile = ZipFile(beta_file)
    #zipfile.getinfo('residuals_%s.csv'%(tissue)).file_size += (2 ** 64) - 1
    #betaqn = pd.read_csv(zipfile.open('residuals_%s.csv'%(tissue)),index_col=0,sep=',')
    #betaqn = betaqn.T
    betaqn = pickle.load( open( '../tissues/resi_norm_%s.p'%(tissue), "rb" ) )
    info = pd.read_csv('../tissues/info_%s.csv.zip'%(tissue),index_col=0, compression='zip',sep=',')

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
    tissue='EC'
    open_file = os.path.realpath('../data_str/')
    ec, info = load_data(tissue)
    #'t_test','fisher',
    features_sel = ['rfe']
    #betaqn, info = load_data()
    #[100000, 50000, 1000, 500, 250, 100, 75, 50]
    #[5000,10000,50000,100000,200000,300000,400000]
    #features_num = [20,50,75,100,250,500,1000,]
    #features_num = [20,50,75,100,250,500,1000,5000,10000,100000]
    features_num = [100000]
    for feat_sel in features_sel:
        #feat_sel = 't_test'
        ec, info = load_data(tissue)
        #min_max_scaler = preprocessing.MinMaxScaler()
        #ec = min_max_scaler.fit_transform(betas)
        print('cargo datos')
        #ec = betaqn.loc[info[(info.tissue == tissue) & (info.braak_stage != 'Exclude')].index]
        cat = info['braak_bin'].loc[ec.index]
        svm_accuracy = {}
        samples = ec.shape[0]
        nzeros = np.where(cat == 0)[0]
        nones = np.where(cat == 1)[0]
        cv_splits = 10
        div_zeros = np.ceil(len(nzeros)/cv_splits)
        div_ones = np.ceil(len(nones)/cv_splits)

        for num in features_num:
            c_val_rbf = np.zeros(cv_splits)
            gamma_val_rbf = np.zeros(cv_splits)
            c_val_lin = np.zeros(cv_splits)
            c_val_pol = np.zeros(cv_splits)
            gamma_val_pol = np.zeros(cv_splits)
            svm_accuracy = {}
            svm_accuracy_tr = {}
            zeros = np.random.permutation(nzeros)
            ones = np.random.permutation(nones)
            for i in range(cv_splits):
                print('split: %d - num_features: %d - tissue:%s- feat_sel:%s' %(i,num,tissue,feat_sel))
                test_index, train_index = get_intervals(cv_splits, i, zeros, ones)
                train_full = ec.iloc[train_index]
                y_train = cat[train_index]
                test_full = ec.iloc[test_index]
                samples = test_full.shape[0]
                samples_tr = train_full.shape[0]
                start_time = time.time()
                features_file = open_file + "/features_CV_%s_%s_%d_%d.p" % (tissue, feat_sel, num, i)
                if feat_sel == 't_test':
                    features_all = fs.feature_sel_t_test_parallel(train_full, info, num)
                elif feat_sel == 'fisher':
                    features_all = fs.feature_fisher_score_parallel(train_full, info, num)
                elif feat_sel == 'rfe':
                    features_all = fs.feature_sel_rfe(train_full, info, num)
                #elif feat_sel == 'chi2':

                print("--- %s seconds for feature selection ---" % (time.time() - start_time))
                pickle.dump(features_all, open(features_file, "wb"))

                train = train_full[features_all[0:num]]
                print(train.shape)
                test = test_full[features_all[0:num]]
                y_true = cat[test_index]
                #SCALING
                scale = preprocessing.StandardScaler().fit(train)
                train = scale.transform(train)
                test = scale.transform(test)
                start_time = time.time()
                (y_pred_rbf, y_tr_rbf, c_val_rbf[i], gamma_val_rbf[i]) = cl.SVM_classify_rbf_all(train, y_train, test, y_true,
                C_range = np.logspace(-5,2,20),gamma_range = np.logspace(-6,2,15))
                (y_pred_pol, y_tr_pol, c_val_pol[i], gamma_val_pol[i]) = cl.SVM_classify_poly_all(train, y_train, test,y_true,
                C_range = np.logspace(-5,2,20),gamma_range = np.logspace(-6,2,15))
                (y_pred_lin, y_tr_lin, c_val_lin[i]) = cl.SVM_classify_lin_all(train, y_train, test, y_true,C_range = np.logspace(-5,2,20))
                print("--- %s seconds for classification ---" % (time.time() - start_time))
                pred_train = pd.DataFrame(
                {'y_train': y_train,
                 'y_tr_rbf': y_tr_rbf,
                 'y_tr_poly': y_tr_pol,
                 'y_tr_lin': y_tr_lin,
                })
                pickle.dump(pred_train, open(open_file + "/pred_tr_CV_%s_%s_%d_%d.p" %(tissue, feat_sel, num, i), "wb"))
                svm_accuracy_tr[i] = [np.where((pred_train['y_train']==pred_train['y_tr_rbf'])==True)[0].shape[0]/samples_tr,
                                    np.where((pred_train['y_train']==pred_train['y_tr_poly'])==True)[0].shape[0]/samples_tr,
                                    np.where((pred_train['y_train']==pred_train['y_tr_lin'])==True)[0].shape[0]/samples_tr]
                print(svm_accuracy_tr[i])
                predictions = pd.DataFrame(
                {'y_true': y_true,
                 'y_rbf': y_pred_rbf,
                 'y_poly': y_pred_pol,
                 'y_lin': y_pred_lin,
                })
                pickle.dump(predictions, open(open_file + "/pred_CV_%s_%s_%d_%d.p" %(tissue, feat_sel, num, i), "wb"))
                svm_accuracy[i] = [np.where((predictions['y_true']==predictions['y_rbf'])==True)[0].shape[0]/samples,
                                    np.where((predictions['y_true']==predictions['y_poly'])==True)[0].shape[0]/samples,
                                    np.where((predictions['y_true']==predictions['y_lin'])==True)[0].shape[0]/samples]

                print(svm_accuracy[i])
            pickle.dump(svm_accuracy_tr, open(open_file + "/accuracy_tr_CV_%s_%s_%d.p" % (tissue, feat_sel,num), "wb"))
            pickle.dump(svm_accuracy, open(open_file + "/accuracy_CV_%s_%s_%d.p" % (tissue, feat_sel,num), "wb"))
            parameters = pd.DataFrame(
            {'C_rbf': c_val_rbf,
             'gamma_rbf': gamma_val_rbf,
             'C_poly': c_val_pol,
             'gamma_poly': gamma_val_pol,
             'C_lin': c_val_lin
            })
            pickle.dump(parameters, open(open_file + "/params_CV_%s_%s_%d.p" %(tissue, feat_sel, num), "wb"))





if __name__ == '__main__':
	main()
