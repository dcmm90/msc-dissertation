# ----------------------------------------------------
# Dissertation MSc CSML
# Author: Diana Carolina Montanes Mondragon
# ----------------------------------------------------
# file_name: classification.py
# description: This file contains the functions for
#              classifying using SVM
# ----------------------------------------------------

# ------------------- imports -------------------------
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import numpy as np
# ----------------------------------------------------


# ------------------- Constant -------------------------
CV_numbers = 5          # Inner loop CV
# ----------------------------------------------------


# ------------------- Function -------------------------
# SVM_classify_rbf_all(train, y_train, test, y_test,  C_range = np.logspace(-4, 4, 100),
#                      gamma_range = np.logspace(-8, 2, 100), balance = 0)
# This function performs the inner fold CV with RBF kernel, selects the
# best parameters, trains the final SVM and predicts the validation set.
# inputs: train - train data
#         y_train - categories of train data
#         test - test data
#         y_test - categories of test data
#         C_range - range of values for parameter C
#         balance - give balanced weights to unbalanced classes
# returns: (y_rbf,y_rbf_tr,c_rbf, gamma_rbf,bs)
#           y_rbf - prediction of test data
#           y_rbf_tr - prediction of train data
#           c_rbf - parameter C chosen
#           gamma_rbf - parameter gamma chosen
#           bs - best accuracy inner CV
# ----------------------------------------------------
def SVM_classify_rbf_all(train, y_train, test, y_test, C_range = np.logspace(-4, 4, 100),gamma_range = np.logspace(-8, 2, 100), balance = 0):
    print('SVM-rbf')
    param_grid = [{'C': C_range, 'gamma': gamma_range, 'kernel': ['rbf']}]
    svr = svm.SVC()
    if balance == 1:
        svr = svm.SVC(class_weight ='balanced')
    clf = GridSearchCV(svr, param_grid, cv=CV_numbers, verbose=1, n_jobs = -1)
    clf.fit(train, y_train)
    tr = clf.score(train, y_train)
    print('best score')
    bs = clf.best_score_
    print(bs)
    print('train score')
    print(tr)
    ts = clf.score(test,y_test)
    print('test score')
    print(ts)
    y_rbf = clf.predict(test)
    y_rbf_tr = clf.predict(train)
    c_rbf = clf.best_params_['C']
    print('C')
    print(c_rbf)
    #print(C_range)
    gamma_rbf = clf.best_params_['gamma']
    print('gamma')
    print(gamma_rbf)
    #print(gamma_range)
    return (y_rbf,y_rbf_tr,c_rbf, gamma_rbf,bs)


# ------------------- Function -------------------------
# SVM_classify_poly_all(train, y_train, test, y_test,  C_range = np.logspace(-4, 4, 100),
#                      gamma_range = np.logspace(-8, 2, 100), balance = 0)
# This function performs the inner fold CV with Polynomial kernel, selects the
# best parameters, trains the final SVM and predicts the validation set.
# inputs: train - train data
#         y_train - categories of train data
#         test - test data
#         y_test - categories of test data
#         C_range - range of values for parameter C
#         balance - give balanced weights to unbalanced classes
# returns: (y_pol, y_pol_tr, c_pol, gamma_pol,bs)
#           y_pol - prediction of test data
#           y_pol_tr - prediction of train data
#           c_pol - parameter C chosen
#           gamma_pol - parameter gamma chosen
#           bs - best accuracy inner CV
# ----------------------------------------------------
def SVM_classify_poly_all(train, y_train, test, y_test, C_range = np.logspace(-5, 2, 100),gamma_range = np.logspace(-6, 4, 100), balance = 0):
    print('SVM-polynomial')
    param_grid = [{'C': C_range, 'gamma': gamma_range, 'kernel': ['poly'],'degree': [3,4]}]
    svr = svm.SVC()
    if balance == 1:
        svr = svm.SVC(class_weight ='balanced')
    clf = GridSearchCV(svr, param_grid, cv=CV_numbers, verbose=1, n_jobs = -1)
    clf.fit(train, y_train)
    tr = clf.score(train, y_train)
    print('best score')
    bs = clf.best_score_
    print(bs)
    print('train score')
    print(tr)
    ts = clf.score(test,y_test)
    print('test score')
    print(ts)
    y_pol = clf.predict(test)
    y_pol_tr = clf.predict(train)
    c_pol = clf.best_params_['C']
    print('C')
    print(c_pol)
    #print(C_range)
    gamma_pol = clf.best_params_['gamma']
    print('gamma')
    print(gamma_pol)
    #print(gamma_range)
    return (y_pol, y_pol_tr, c_pol, gamma_pol,bs)


# ------------------- Function -------------------------
# SVM_classify_lin_all(train, y_train, test, y_test,
#               C_range = np.logspace(-5, 5, 200), balance = 0)
# This function performs the inner fold CV with Linear kernel, selects the
# best parameters, trains the final SVM and predicts the validation set.
# inputs: train - train data
#         y_train - categories of train data
#         test - test data
#         y_test - categories of test data
#         C_range - range of values for parameter C
#         balance - give balanced weights to unbalanced classes
# returns: (y_lin, y_lin_tr, c_lin,bs)
#           y_lin - prediction of test data
#           y_lin_tr - prediction of train data
#           c_lin - parameter C chosen
#           bs - best accuracy inner CV
# ----------------------------------------------------
def SVM_classify_lin_all(train, y_train, test, y_test, C_range = np.logspace(-5, 5, 200), balance = 0):
    print('SVM-linear')
    param_grid = [{'C': C_range, 'kernel': ['linear']}]
    svr = svm.SVC()
    if balance == 1:
        svr = svm.SVC(class_weight ='balanced')
    clf = GridSearchCV(svr, param_grid, cv=CV_numbers, verbose=1, n_jobs = -1)
    clf.fit(train, y_train)
    tr = clf.score(train, y_train)
    print('best score')
    bs = clf.best_score_
    print(bs)
    print('train score')
    print(tr)
    ts = clf.score(test,y_test)
    print('test score')
    print(ts)
    y_lin = clf.predict(test)
    y_lin_tr = clf.predict(train)
    c_lin = clf.best_params_['C']
    print('C')
    print(c_lin)
    #print(C_range)
    return (y_lin, y_lin_tr, c_lin,bs)

