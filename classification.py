from sklearn import svm
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np


def SVM_classify_rbf(train, y_train, test,C_range = np.logspace(-2, 10, 13),gamma_range = np.logspace(-9, 3, 13)):
    #C_range = np.logspace(-2, 10, 13)
    #gamma_range = np.logspace(-9, 3, 13)
    #C_range = np.logspace(-2, 10, 6)
    #gamma_range = np.logspace(-6, 3, 8)
    #rbf
    print('SVM-rbf')
    param_grid = [{'C': C_range, 'gamma': gamma_range, 'kernel': ['rbf']}]

    svr = svm.SVC(random_state=1234)
    clf = GridSearchCV(svr, param_grid, cv=5, verbose=1, n_jobs = -1)
    clf.fit(train, y_train)
    y_rbf = clf.predict(test)[0]
    c_rbf = clf.best_params_['C']
    gamma_rbf = clf.best_params_['gamma']
    return (y_rbf, c_rbf, gamma_rbf)


def SVM_classify_poly(train, y_train, test,C_range = np.logspace(-2, 10, 13),gamma_range = np.logspace(-9, 3, 13)):
    print('SVM-polynomial')
    param_grid = [{'C': C_range, 'gamma': gamma_range, 'kernel': ['poly']}]
    svr = svm.SVC(random_state=1234)
    clf = GridSearchCV(svr, param_grid, cv=5, verbose=1, n_jobs = -1)
    clf.fit(train, y_train)
    y_pol = clf.predict(test)[0]
    c_pol = clf.best_params_['C']
    gamma_pol = clf.best_params_['gamma']
    return (y_pol, c_pol, gamma_pol)


def SVM_classify_lin(train, y_train, test, C_range = np.logspace(-2, 10, 13)):
    print('SVM-linear')
    param_grid = [{'C': C_range, 'kernel': ['linear']}]
    svr = svm.SVC(random_state=1234)
    clf = GridSearchCV(svr, param_grid, cv=5, verbose=1, n_jobs = -1)
    clf.fit(train, y_train)
    y_lin = clf.predict(test)[0]
    c_lin = clf.best_params_['C']
    return (y_lin, c_lin)
