# ----------------------------------------------------
# Dissertation MSc CSML
# Author: Diana Carolina Montanes Mondragon
# ----------------------------------------------------
# file_name: general_res_CV.py
# description: This file contains the main function for
#               Experiment 2
# ----------------------------------------------------

# ------------------- imports -------------------------
from __future__ import division
import numpy as np
# -----------------------------------------------------

# ------------------- Function -------------------------
# get_intervals(cv_splits, i, zeros, ones)
# This function makes the folds for outer loop CV
# inputs: cv_splits - number of folds
#         i - current fold
#         zeros - ids of entries that belong to category 0
#         ones - ids of entries that belong to category 1
# returns: test, train
#          test - Index of test samples on that fold
#          train - Index of train samples on that fold
# ----------------------------------------------------
def get_intervals(cv_splits, i, zeros, ones):
    div_zeros = int(np.floor(len(zeros)/cv_splits))
    div_ones = int(np.floor(len(ones)/cv_splits))
    if i<(cv_splits-1):
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
    return test, train


# ------------------- Function -------------------------
# get_intervals_all(cv_splits, i, zeros, ones, new_inf, subjects)
# This function makes the folds for outer loop CV.
# Takes into account the restriction for experiment 1
# inputs: cv_splits - number of folds
#         i - current fold
#         zeros - ids of entries that belong to category 0
#         ones - ids of entries that belong to category 1
#         new_inf - information from data including subjects
#         subjects - ids of the subject
# returns: test, train
#          test - Index of test samples on that fold
#          train - Index of train samples on that fold
# ----------------------------------------------------
def get_intervals_all(cv_splits, i, zeros, ones, new_inf, subjects):
    div_zeros = int(np.floor(len(zeros)/cv_splits))
    div_ones = int(np.floor(len(ones)/cv_splits))
    if i<(cv_splits-1):
        mini_zero = div_zeros*i
        maxi_zero = (div_zeros*i) + div_zeros
        mini_one = div_ones*i
        maxi_one = (div_ones*i) + div_ones
    else:
        mini_zero = div_zeros*i
        maxi_zero = len(zeros)
        mini_one = div_ones*i
        maxi_one = len(ones)
    index_zeros_temp = list(zeros[mini_zero: maxi_zero])
    index_ones_temp = list(ones[mini_one: maxi_one])
    index_zeros = []
    index_ones = []
    for i in range(len(new_inf.index)):
        ids = new_inf.index[i]
        if new_inf['subject'].loc[ids] in subjects[index_zeros_temp]:
            index_zeros.append(i)
        elif new_inf['subject'].loc[ids] in subjects[index_ones_temp]:
            index_ones.append(i)
    index_test = np.array(index_zeros + index_ones)
    index_train = np.array(list(set(range(len(new_inf))) - set(index_test)))
    id_test = new_inf.iloc[index_test].index
    id_train = new_inf.iloc[index_train].index
    return id_test,id_train