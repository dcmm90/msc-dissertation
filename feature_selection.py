import numpy as np
import pandas as pd




def feature_sel_t_test(betas, info, size):
    c_info = info.loc[betas.index]
    c1 = c_info[c_info['braak_bin'] == 0].index
    c2 = c_info[c_info['braak_bin'] == 1].index

    numerator = np.absolute(betas.loc[c1].apply(np.mean,0) - betas.loc[c2].apply(np.mean,0))
    denominator = np.sqrt((betas.loc[c1].apply(np.std,0)/c1.shape[0])+(betas.loc[c2].apply(np.std,0)/c2.shape[0]))
    t_stat = numerator/denominator
    ind = np.argsort(t_stat)[-size:]
    ec_feat = betas.iloc[:,ind]
    return list(ec_feat), ec_feat


def feature_fisher_score(betas, info, size):
    c_info = info.loc[betas.index]
    c1 = c_info[c_info['braak_bin'] == 0].index
    c2 = c_info[c_info['braak_bin'] == 1].index

    ec_mean = ec.apply(np.mean,0)
    numerator = (c1.shape[0]*np.power(ec.loc[c1].apply(np.mean,0) - ec_mean,2) +
                c2.shape[0]*np.power(ec.loc[c2].apply(np.mean,0) - ec_mean,2))
    denominator = ((ec.loc[c1].apply(np.var,0) * c1.shape[0]) +
                   (ec.loc[c2].apply(np.var,0)*c2.shape[0]))
    fisher_score = numerator/denominator
    ind = np.argsort(fisher_score)[-size:]
    ec_feat = betas.iloc[:,ind]
    return list(ec_feat), ec_feat
