from sklearn import svm
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV


def log_reg(train, y_train, test):
    lr = LogisticRegressionCV(Cs=10, cv=5, dual=False, solver = 'sag', tol=0.001, max_iter=1000, n_jobs=-1, verbose=1, random_state=1234)
    lr.fit(train, y_train)
    y_lr = lr.predict(test)[0]
    return (y_lr)



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

def SVM_classify_rbf_all(train, y_train, test, y_test, C_range = np.logspace(-10, 11, 35),gamma_range = np.logspace(-11, 3, 35), balance = 0):
    #C_range = np.logspace(-2, 10, 13)
    #gamma_range = np.logspace(-9, 3, 13)
    #C_range = np.logspace(-2, 10, 6)
    #gamma_range = np.logspace(-6, 3, 8)
    #rbf
    print('SVM-rbf')
    param_grid = [{'C': C_range, 'gamma': gamma_range, 'kernel': ['rbf']}]
    svr = svm.SVC()
    if balance == 1:
        svr = svm.SVC(class_weight ='balanced')
    clf = GridSearchCV(svr, param_grid, cv=5, verbose=1, n_jobs = -1)
    clf.fit(train, y_train)
    tr = clf.score(train, y_train)
    print('train score')
    print(tr)
    ts = clf.score(test,y_test)
    print('test score')
    print(ts)
    y_rbf = clf.predict(test)
    c_rbf = clf.best_params_['C']
    gamma_rbf = clf.best_params_['gamma']
    return (y_rbf, c_rbf, gamma_rbf)


def SVM_classify_poly_all(train, y_train, test, y_test, C_range = np.logspace(-5, 10, 25),gamma_range = np.logspace(-9, 3, 20), balance = 0):
    print('SVM-polynomial')
    param_grid = [{'C': C_range, 'gamma': gamma_range, 'kernel': ['poly'],'degree': [2,3,4,5]}]
    svr = svm.SVC()
    if balance == 1:
        svr = svm.SVC(class_weight ='balanced')
    clf = GridSearchCV(svr, param_grid, cv=5, verbose=1, n_jobs = -1)
    clf.fit(train, y_train)
    tr = clf.score(train, y_train)
    print('train score')
    print(tr)
    ts = clf.score(test,y_test)
    print('test score')
    print(ts)
    y_pol = clf.predict(test)
    c_pol = clf.best_params_['C']
    gamma_pol = clf.best_params_['gamma']
    return (y_pol, c_pol, gamma_pol)


def SVM_classify_lin_all(train, y_train, test, y_test, C_range = np.logspace(-10, 11, 30), balance = 0):
    print('SVM-linear')
    param_grid = [{'C': C_range, 'kernel': ['linear']}]
    svr = svm.SVC()
    if balance == 1:
        svr = svm.SVC(class_weight ='balanced')
    clf = GridSearchCV(svr, param_grid, cv=5, verbose=1, n_jobs = -1)
    clf.fit(train, y_train)
    tr = clf.score(train, y_train)
    print('train score')
    print(tr)
    ts = clf.score(test,y_test)
    print('test score')
    print(ts)
    y_lin = clf.predict(test)
    c_lin = clf.best_params_['C']
    return (y_lin, c_lin)
