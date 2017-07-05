import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def PCA_tissue_braak(tissue, info, betas, reduced, standardize = 0):
    if tissue=='all':
        ec = betas.loc[info.barcode[info.braak_stage != 'Exclude']]
    else:
        ec = betas.loc[info.barcode[(info.source_tissue == tissue) & (info.braak_stage != 'Exclude')]]
    labels = info.braak_stage[info.barcode.isin(ec.index)]
    labels = np.array(labels).astype(np.int32)
    if(standardize == 0):
        ec = StandardScaler().fit_transform(ec)
    pca_2 = PCA(2)
    plot_columns = pca_2.fit_transform(ec)
    target_names = ['0','1','2','3','4','5','6']
    with plt.style.context('seaborn-whitegrid'):
        plt.figure(figsize=(8, 6))
        for c, i, target_name in zip(['lightskyblue','dodgerblue','royalblue','mediumorchid','purple','salmon','orangered'],
                                     [0,1,2,3,4,5,6], target_names):
            plt.scatter(x=plot_columns[labels==i,0],
                        y=plot_columns[labels==i,1],
                        c=c, label=target_name, s=80)
        plt.xlabel('PC1 (%.2f %% explained var.)' % (pca_2.explained_variance_ratio_[0] * 100), fontsize=10)
        plt.ylabel('PC2 (%.2f %% explained var.)' % (pca_2.explained_variance_ratio_[1] * 100), fontsize=10)
        lgd = plt.legend(title='Braak-stage', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.title('PCA for tissue: %s' % tissue, fontsize=11)
        std = 'std'
        if(standardize == 1): std = 'nostd'
        plt.savefig('plots/%s_PCA_tissue_braak_%s_%s.png' % (tissue,std,reduced), bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close()


def PCA_tissue_status(tissue, info, betas, reduced, standardize = 0):
    if tissue=='all':
        ec = betas.loc[info.barcode[info.braak_stage != 'Exclude']]
    else:
        ec = betas.loc[info.barcode[(info.source_tissue == tissue) & (info.ad_status != 'Exclude')]]
    labels = info.ad_status[info.barcode.isin(ec.index)]
    labels = np.array(labels)
    if(standardize == 0):
        ec = StandardScaler().fit_transform(ec)
    pca_2 = PCA(2)
    plot_columns = pca_2.fit_transform(ec)
    target_names = ['C','AD']
    with plt.style.context('seaborn-whitegrid'):
        plt.figure(figsize=(8, 6))
        for c, i, target_name in zip(['lightskyblue','orangered'], target_names, target_names):
            plt.scatter(x=plot_columns[labels==i,0],
                        y=plot_columns[labels==i,1],
                        c=c, label=target_name, s=80)
        plt.xlabel('PC1 (%.2f %% explained var.)' % (pca_2.explained_variance_ratio_[0] * 100), fontsize=10)
        plt.ylabel('PC2 (%.2f %% explained var.)' % (pca_2.explained_variance_ratio_[1] * 100), fontsize=10)
        lgd = plt.legend(title='Status', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.title('PCA for tissue: %s' %tissue, fontsize=11)
        std = 'std'
        if(standardize == 1): std = 'nostd'
        plt.savefig('plots/%s_PCA_tissue_status_%s_%s.png' % (tissue,std,reduced), bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close()


def PCA_braak_tissue(stage, info, betas, reduced, standardize = 0):
    ec = betas.loc[info.barcode[info.braak_stage == stage]]
    labels = info.source_tissue[info.barcode.isin(ec.index)]
    labels = np.array(labels)
    if(standardize == 0):
        ec = StandardScaler().fit_transform(ec)
    pca_2 = PCA(2)
    plot_columns = pca_2.fit_transform(ec)
    target_names = ['entorhinal cortex', 'cerebellum', 'frontal cortex', 'superior temporal gyrus','whole blood']
    with plt.style.context('seaborn-whitegrid'):
        plt.figure(figsize=(8, 6))
        for c, i, target_name in zip(['lightskyblue','royalblue','purple','salmon','orangered'], target_names, target_names):
            plt.scatter(x=plot_columns[labels==i,0],
                        y=plot_columns[labels==i,1],
                        c=c, label=target_name, s=80)
        plt.xlabel('PC1 (%.2f %% explained var.)' % (pca_2.explained_variance_ratio_[0] * 100), fontsize=10)
        plt.ylabel('PC2 (%.2f %% explained var.)' % (pca_2.explained_variance_ratio_[1] * 100), fontsize=10)
        lgd = plt.legend(title='Tissue', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.title('PCA for braak-stage: %s' % stage, fontsize=11)
        std = 'std'
        if(standardize == 1): std = 'nostd'
        plt.savefig('plots/%s_PCA_braak_%s_%s.png' % (stage,std,reduced), bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close()

def PCA_braak_tissue_sel(stage, info, betas, reduced, standardize = 0):
    ec = betas.loc[info.barcode[(info.braak_stage == stage) & ((info.source_tissue == 'entorhinal cortex') |
                                                                (info.source_tissue == 'frontal cortex') |
                                                               (info.source_tissue == 'superior temporal gyrus'))]]
    labels = info.source_tissue[info.barcode.isin(ec.index)]
    labels = np.array(labels)
    if(standardize == 0):
        ec = StandardScaler().fit_transform(ec)
    pca_2 = PCA(2)
    plot_columns = pca_2.fit_transform(ec)
    target_names = ['entorhinal cortex', 'frontal cortex', 'superior temporal gyrus']
    with plt.style.context('seaborn-whitegrid'):
        plt.figure(figsize=(8, 6))
        for c, i, target_name in zip(['lightskyblue','royalblue','purple','salmon','orangered'], target_names, target_names):
            plt.scatter(x=plot_columns[labels==i,0],
                        y=plot_columns[labels==i,1],
                        c=c, label=target_name, s=80)
        plt.xlabel('PC1 (%.2f %% explained var.)' % (pca_2.explained_variance_ratio_[0] * 100), fontsize=10)
        plt.ylabel('PC2 (%.2f %% explained var.)' % (pca_2.explained_variance_ratio_[1] * 100), fontsize=10)
        lgd = plt.legend(title='Tissue', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.title('PCA for braak-stage: %s' % stage, fontsize=11)
        std = 'std'
        if(standardize == 1): std = 'nostd'
        plt.savefig('plots/%s_PCA_braak_sel_%s_%s.png' % (stage,std,reduced), bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close()

def PCA_components(tissue, info, betas, reduced, standardize = 0):
    if tissue=='all':
        ec = betas.loc[info.barcode[info.braak_stage != 'Exclude']]
    else:
        ec = betas.loc[info.barcode[(info.source_tissue == tissue) & (info.braak_stage != 'Exclude')]]
    labels = info.braak_bin[info.barcode.isin(ec.index)]
    labels = np.array(labels)
    if(standardize == 0):
        ec = StandardScaler().fit_transform(ec)
    pca_2 = PCA(8)
    plot_columns = pca_2.fit_transform(ec)
    target_names = [0,1]
    for j in range(8):
        with plt.style.context('seaborn-white',{'axes.grid':False}):
            plt.figure(figsize=(10, 3))
            for c, i, target_name in zip(['lightskyblue','orangered'], target_names, target_names):
                plt.scatter(x=plot_columns[labels==i,j],
                            y=labels[labels==i],
                            c=c, label=target_name, s=80)
            plt.xlabel('PC%d (%.2f %% explained var.)' % (j + 1, pca_2.explained_variance_ratio_[j] * 100))
            lgd = plt.legend(title='braak-stage', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            plt.yticks([])
            plt.title('PCA-%d for tissue: %s' %(j+1,tissue))
            std = 'std'
            if(standardize == 1): std = 'nostd'
            plt.tight_layout()
            plt.savefig('plots/%s_PCA_%d_tissue_braak_%s_%s.png' % (tissue,j+1,std,reduced), bbox_extra_artists=(lgd,), bbox_inches='tight')
            plt.close()


def main():
    betaqn = pd.read_csv('GSE59685_betas2.csv',skiprows=(1,2),index_col=0)
    betaqn = betaqn.T

    var = betaqn.var(axis=0)
    ind = np.argsort(var)[-5000:]
    red_beta = betaqn.iloc[:,ind]

    info = pd.read_csv('info.csv')
    info = info.drop('Unnamed: 0', 1)

    info.loc[(info.braak_stage=='5') | (info.braak_stage=='6'),'braak_bin'] = 1
    cond = ((info.braak_stage=='0') | (info.braak_stage=='1') | (info.braak_stage=='2') |
            (info.braak_stage=='3') | (info.braak_stage=='4'))
    info.loc[cond ,'braak_bin'] = 0

    tissue=['all', 'entorhinal cortex', 'cerebellum', 'frontal cortex', 'superior temporal gyrus','whole blood']
    stage = ['0','1','2','3','4','5','6']
    #for stand in [0,1]:
    for stand in [0]:
        for ts in tissue:
            print ts
            #PCA_tissue_braak(ts,info,red_beta,'reduced',stand)
            #PCA_tissue_braak(ts,info,betaqn,'complete',stand)
            #PCA_tissue_status(ts,info,red_beta,'reduced',stand)
            #PCA_tissue_status(ts,info,betaqn,'complete',stand)
            PCA_components(ts,info,betaqn,'complete',stand)
        #for st in stage:
        #    print st
        #    PCA_braak_tissue(st,info,red_beta,'reduced',stand)
        #    PCA_braak_tissue(st,info,betaqn,'complete',stand)
        #    PCA_braak_tissue_sel(st,info,red_beta,'reduced',stand)
        #    PCA_braak_tissue_sel(st,info,betaqn,'complete',stand)



if __name__ == '__main__':
	main()
