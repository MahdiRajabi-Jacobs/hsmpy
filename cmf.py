import sys
import math
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from time import gmtime, strftime
from scipy import stats
import scipy

def CalculateCMF(before,after,bootstrap=1000,plot=True,fit_gamma=False):
    cmf_L = []
    for bs in range(bootstrap):
        b = before.sample(frac=1,replace=True)
        a = after.sample(frac=1,replace=True)
        if b.sum()>0:
            cmf_L.append(a.mean()/b.mean())
    CMFL = pd.Series(cmf_L)
    m = after.mean() / before.mean()
    n = len(before) + len(after) - 2
    mu, sd = stats.norm.fit(CMFL)
    if fit_gamma:
        alpha, loc, beta = stats.gamma.fit(CMFL)
        beta = 1/beta
        mu,sd = alpha/beta,np.sqrt(alpha/beta**2)

    se = sd
    h = se * stats.t.ppf((1.95) / 2., n-1)
    cv = sd/mu
    h0 = 1
    p=np.NaN
    if mu>h0:
        t = (mu-h0)/se
        p = stats.t.sf(t, n-1)
    if mu<h0:
        t = (h0-mu)/se
        p = stats.t.sf(t, n-1)

    if plot:
        if fit_gamma:
            try:
                sns.distplot(CMFL,fit=stats.gamma,label='Bootstrap CMFs')
            except:
                sns.distplot(CMFL)
        else:
            try:
                sns.distplot(CMFL,fit=stats.norm,label='Bootstrap CMFs')
            except:
                sns.distplot(CMFL)
        plt.xlabel('CMF Mean Value')
        plt.ylabel('Density')    

        handles, labels = plt.gca().get_legend_handles_labels()
        Legend_DF = pd.DataFrame(columns=['Handles','Labels'])
        Legend_DF['Handles'] = handles
        Legend_DF['Labels'] = labels
        Legend_DF.loc[Legend_DF.shape[0]] = [mpl.lines.Line2D([0], [0], color='black', linewidth=2),'Fitted Gamma Distribution' if fit_gamma else 'Fitted Normal Distribution'] 
        Legend_DF.loc[Legend_DF.shape[0]] = [mpl.lines.Line2D([0], [0], linewidth=0),'Raw CMF: {:0.2f}'.format(m)] 
        Legend_DF.loc[Legend_DF.shape[0]] = [mpl.lines.Line2D([0], [0], linewidth=0),'Fitted CMF: {:0.2f}'.format(mu)] 
        Legend_DF.loc[Legend_DF.shape[0]] = [mpl.lines.Line2D([0], [0], linewidth=0),'Standard Error: {:0.4f}'.format(se)] 
        Legend_DF.loc[Legend_DF.shape[0]] = [mpl.lines.Line2D([0], [0], linewidth=0),'Coeff. of Variation: {:0.1f}%'.format(cv*100)] 
        Legend_DF.loc[Legend_DF.shape[0]] = [mpl.lines.Line2D([0], [0], linewidth=0),'p-value: {:0.4f}'.format(p)] 
        Legend_DF.loc[Legend_DF.shape[0]] = [mpl.lines.Line2D([0], [0], linewidth=0),'95% Confidence Interval: ({:0.2f}, {:0.2f})'.format(mu-h,mu+h)] 

        plt.legend(Legend_DF.Handles.tolist(),Legend_DF.Labels.tolist())
        plt.grid()
    return(mu,cv,p)