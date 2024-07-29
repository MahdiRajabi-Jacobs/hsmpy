import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

class SegmentSPF(object):
    """
    
    """
    import statsmodels.formula.api as smf 
    import statsmodels as sm
    
    def __init__(self, data, Length='LENGTH',AADT='AADT',crash='N_Obs',Num_Years = 5,alpha_range=(0.01,2),alpha_iter = 200):
        data = data[[Length,AADT,crash]].copy(deep=True)
        data = data.rename(columns={Length:'LENGTH',AADT:'AADT',crash:'N_Obs'})
        
        data['AADT_Log']    = data.AADT.apply(np.log)
        data['Length_Log']  = data.LENGTH.apply(np.log)
        data['CF_PerY']     = data.N_Obs/Num_Years
        data['CF_PerYPerM'] = data.CF_PerY/data.LENGTH
        data['Constant'] = 1
        self.data   = data
        self.n_years= Num_Years
        self.alpha_range = alpha_range
        self.alpha_iter = alpha_iter
        
    @property
    def alpha(self):
        LL_DS = pd.Series()
        for a in np.linspace(self.alpha_range[0], self.alpha_range[1], self.alpha_iter) :
            model = self.smf.glm(formula = "N_Obs ~ AADT_Log", data=self.data, family=self.sm.genmod.families.NegativeBinomial(alpha=a),offset=self.data.Length_Log).fit()
            LL_DS.loc[a] = model.llf
        alpha = LL_DS[LL_DS==LL_DS.max()].index[0]
        #print('alpha: {:0.4f}'.format(alpha))
        self._ll_vs_alpha = LL_DS
        self._alpha = alpha
        return(self._alpha)
    
    @property
    def spf_fun(self):
        """calculates crashes per year per mile based on current SPF"""
        self._spf_fun = lambda row:row.AADT**self.result.params.loc['AADT_Log']*np.exp(self.result.params.loc['Intercept'])
        return(self._spf_fun)
    
    @property
    def overdisperssion(self):
        """overdisperssion in observed crashes"""
        m = np.mean(self.data.N_Obs)
        v = np.var(self.data.N_Obs)
        if v>m:
            self._overdisperssion = (v-m)/(m**2)
        else:
            self._overdisperssion = 0
        return(self._overdisperssion)
    
    @property
    def model(self):
        fr = "CF_PerY ~ AADT_Log"
        fm = self.sm.genmod.families.NegativeBinomial(alpha=self.alpha)
        of = self.data.Length_Log
        self._model = self.smf.glm(formula = fr, data=self.data, family=fm, offset=of)
        return(self._model)
    
    def calibrate_spf(self):
        o = self.data.N_Obs.sum()
        p = self.data.apply(self.spf_fun,axis=1) * self.n_years * self.data.LENGTH
        cf = o/p.sum()
        self.calibration_factor = cf
        self.result.params.loc['Intercept'] = self.result.params.loc['Intercept'] + np.log(self.calibration_factor)
        
    def fit(self):
        result = self.model.fit()
        self.result = result
        self.calibrate_spf()
    
        self.data['N_Pred'] = self.data.apply(self.spf_fun,axis=1) * self.n_years * self.data.LENGTH
        self.data['Pred_CF_PerY'] = self.data.N_Pred/self.n_years
        self.data['Pred_CF_PerYPerM'] = self.data.Pred_CF_PerY/self.data.LENGTH
    
        sel_flds = ['Constant','AADT_Log']
        self.data['Variance'] = self.data.apply(lambda row:np.dot(np.dot(row[sel_flds].to_numpy(),result.cov_params()),row[sel_flds].to_numpy().T),axis=1)
        self.data['Std_Err']  = self.data.Variance.apply(np.sqrt)
    
        
        self.data['Upper'] = self.data.Pred_CF_PerYPerM.apply(np.log) + self.data.Std_Err * 1.96
        self.data['Lower'] = self.data.Pred_CF_PerYPerM.apply(np.log) - self.data.Std_Err * 1.96
        self.data['Upper'] = self.data.Upper.apply(np.exp)
        self.data['Lower'] = self.data.Lower.apply(np.exp)

        self.data['Residuals'] = self.data.CF_PerY - self.data.Pred_CF_PerY
        self.data.sort_values('AADT',inplace=True)
        self.data['CumRes'] = self.data.Residuals.cumsum()
        self.data['CumResSquared'] = self.data.Residuals.pow(2).cumsum()
        self.data['Var'] = self.data.CumResSquared * (1-self.data.CumResSquared/self.data.CumResSquared.max())
        self.data['Lim'] = 2 * self.data.Var.pow(0.5)
        self.data = self.data.drop(columns=['CumResSquared','Var'])
        return(result)
    
    def predict(self):
        Predicted_DF = pd.DataFrame([
                pd.Series(np.linspace(self.data.AADT.min(),self.data.AADT.max(),1000),name='AADT')
            ]).T
        Predicted_DF['LENGTH'] = 1
        Predicted_DF['AADT_Log'] = Predicted_DF.AADT.apply(np.log)
        Predicted_DF['Length_Log'] = Predicted_DF.LENGTH.apply(np.log)

        Predicted_DF['Pred_CF'] = Predicted_DF.apply(self.spf_fun,axis=1)

        Predicted_DF['Constant'] = 1
        sel_flds = ['Constant','AADT_Log']
        Predicted_DF['Variance'] = Predicted_DF.apply(lambda row:np.dot(np.dot(row[sel_flds].to_numpy(),self.result.cov_params()),row[sel_flds].to_numpy().T),axis=1)
        Predicted_DF['Std_Err'] = Predicted_DF.Variance.apply(np.sqrt)
        Predicted_DF['Upper'] = Predicted_DF.Pred_CF.apply(np.log) + Predicted_DF.Std_Err * 1.96
        Predicted_DF['Lower'] = Predicted_DF.Pred_CF.apply(np.log) - Predicted_DF.Std_Err * 1.96
        Predicted_DF['Upper'] = Predicted_DF.Upper.apply(np.exp)
        Predicted_DF['Lower'] = Predicted_DF.Lower.apply(np.exp)
        self.predicted = Predicted_DF
        return(Predicted_DF)
    
    def plot_spf(self):
        coef = self.result.params
        L1 = 'Data Points\nNumber of Segments: {:0,.0f}\nTotal Mileage: {:0,.1f}\nTotal Crashes: {:0,.0f}'.format(self.data.shape[0],self.data.LENGTH.sum(),self.data.N_Obs.sum())
        L2 = 'Fitted SPF & 95% CI\nIntercept: {:0,.3f}\nAADT Coef: {:0.4f}\nOverdisperssion (k): {:0,.3f}'.format(coef.Intercept,coef.AADT_Log,self.alpha)

        fig = plt.figure(figsize=(18,5))

        ax = plt.subplot(131)
        sns.scatterplot(data=self.data, x='LENGTH', y='AADT',label=L1,color='grey',marker='.',ax=ax)
        #sns.kdeplot(data=self.data, x='LENGTH', y='AADT',fill=False,label='Kernel Density Of Data Points')
        ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.2f}'))
        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
        plt.xlabel('Segment Length (Miles)')
        plt.ylabel('AADT')
        plt.title('Domain Of Applicability')
        plt.legend(loc='upper left')
        plt.grid()

        ax = plt.subplot(132)
        sns.scatterplot(data=self.data, x='AADT', y='CF_PerYPerM',label='Data Points',color='grey',marker='.',ax=ax)
        #sns.kdeplot(data=self.data, x='AADT', y='CF_PerYPerM',fill=False,label='Kernel Density Of Data Points')
        plt.plot(self.predicted.AADT,self.predicted.Pred_CF,'red',label=L2)
        ax.fill_between(self.predicted.AADT, self.predicted.Lower, self.predicted.Upper, alpha=0.2,color='red')

        ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
        plt.xlabel('AADT')
        plt.ylabel('Crash Frequency per Mile per Year')
        plt.title('Fitted Model')
        plt.legend(loc='upper left')
        plt.grid()

        ax = plt.subplot(133)
        ax.plot(self.data.AADT.tolist(),self.data.Lim.tolist(),'g',label='Two Standard Deviation Boundaries')
        ax.plot(self.data.AADT.tolist(),(-self.data.Lim).tolist(),'g')
        ax.plot(self.data.AADT.tolist(),self.data.CumRes.tolist(),'b',label='Cumulative Residuals')
        ax.hlines(0,self.data.AADT.min(),self.data.AADT.max(),'black')
        ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
        plt.xlabel('AADT')
        plt.ylabel('Cumulative Residuals (CURE)')
        plt.title('Cumulative Residuals (CURE)')
        plt.legend(loc='upper left')
        plt.grid()
        
        return(fig)

    def summary(self):
        ds = pd.Series({'NumberOfYears':self.n_years,
                    'Segments':self.data.shape[0],'Mileage':self.data.LENGTH.sum(),'Total_Crashes':self.data.N_Obs.sum(),
                    'Min_AADT':self.data.AADT.min(),'Max_AADT':self.data.AADT.max(),
                    
                    'Intercept':self.result.params.Intercept,'AADT_Coef':self.result.params.AADT_Log,
                    'Intercept_pvalue':self.result.pvalues.Intercept,'AADT_pvalue':self.result.pvalues.AADT_Log,
                    'Intercept_se':self.result.bse.Intercept,'AADT_se':self.result.bse.AADT_Log,
                    'Intercept_cv':self.result.bse.Intercept/self.result.params.Intercept,'AADT_cv':self.result.bse.AADT_Log/self.result.params.AADT_Log,
                    
                    'Overdisperssion':self.alpha,
                    'Calibration_Factor':self.calibration_factor,
                    'Obs_Crashes_Mean':self.data.N_Obs.mean(),'Pred_Crashes_Mean':self.data.N_Pred.mean(),
                    'Obs_CF_PerY':self.data.CF_PerY.mean(),'Pred_CF_PerY':self.data.Pred_CF_PerY.mean(),
                    'Obs_CF_PerYPerM':self.data.CF_PerYPerM.mean(),'Pred_CF_PerYPerM':self.data.Pred_CF_PerYPerM.mean()
                    })
        self.spf_summary = ds
        return(ds)
class Calibration_Function(object):
    """
    
    """
    import statsmodels.formula.api as smf 
    import statsmodels as sm
    
    def __init__(self, data, Observed='obs_kabco',Predicted='pred_kabco',alpha_range=(0.01,2),alpha_iter = 200):
        data = data[[Observed,Predicted]].copy(deep=True)
        data = data.rename(columns={Observed:'N_Obs',Predicted:'N_Pred'})
        data['N_Pred_Log'] = data.N_Pred.apply(np.log)
        data['Constant'] = 1
        
        self.data   = data
        self.alpha_range = alpha_range
        self.alpha_iter = alpha_iter
        
    @property
    def alpha(self):
        LL_DS = pd.Series()
        for a in np.linspace(self.alpha_range[0], self.alpha_range[1], self.alpha_iter) :
            model = self.smf.glm(formula = "N_Obs ~ N_Pred_Log", data=self.data, family=self.sm.genmod.families.NegativeBinomial(alpha=a)).fit()
            LL_DS.loc[a] = model.llf
        alpha = LL_DS[LL_DS==LL_DS.max()].index[0]
        #print('alpha: {:0.4f}'.format(alpha))
        self._ll_vs_alpha = LL_DS
        self._alpha = alpha
        return(self._alpha)
    
    @property
    def c_function(self):
        """calculates total crashes based on calibration function"""
        self._c_function = lambda row:row.N_Pred**self.result.params.loc['N_Pred_Log']*np.exp(self.result.params.loc['Intercept'])
        return(self._c_function)
    

    @property
    def overdisperssion(self):
        """overdisperssion in observed crashes"""
        m = np.mean(self.data.N_Obs)
        v = np.var(self.data.N_Obs)
        if v>m:
            self._overdisperssion = (v-m)/(m**2)
        else:
            self._overdisperssion = 0
        return(self._overdisperssion)
    
    @property
    def model(self):
        fr = "N_Obs ~ N_Pred_Log"
        fm = self.sm.genmod.families.NegativeBinomial(alpha=self.alpha)
        self._model = self.smf.glm(formula = fr, data=self.data, family=fm)
        return(self._model)
    
    def calibrate(self):
        result = self.model.fit()
        self.result = result
    
    
        self.data['N_Pred_cfun'] = self.data.apply(self.c_function,axis=1)
        o = self.data.N_Obs.sum()
        p = self.data.N_Pred_cfun.sum()
        cf = o/p
        self.result.params.loc['Intercept'] = self.result.params.loc['Intercept'] + np.log(cf)
        self.data['N_Pred_cfun'] = self.data.apply(self.c_function,axis=1)
        
    
        sel_flds = ['Constant','N_Pred_Log']
        self.data['Variance'] = self.data.apply(lambda row:np.dot(np.dot(row[sel_flds].to_numpy(),result.cov_params()),row[sel_flds].to_numpy().T),axis=1)
        self.data['Std_Err']  = self.data.Variance.apply(np.sqrt)
    
        
        self.data['Upper'] = self.data.N_Pred_cfun.apply(np.log) + self.data.Std_Err * 1.96
        self.data['Lower'] = self.data.N_Pred_cfun.apply(np.log) - self.data.Std_Err * 1.96
        self.data['Upper'] = self.data.Upper.apply(np.exp)
        self.data['Lower'] = self.data.Lower.apply(np.exp)

        self.data['Residuals'] = self.data.N_Obs - self.data.N_Pred_cfun
        self.data.sort_values('N_Pred',inplace=True)
        self.data['CumRes'] = self.data.Residuals.cumsum()
        self.data['CumResSquared'] = self.data.Residuals.pow(2).cumsum()
        self.data['Var'] = self.data.CumResSquared * (1-self.data.CumResSquared/self.data.CumResSquared.max())
        self.data['Lim'] = 2 * self.data.Var.pow(0.5)
        self.data = self.data.drop(columns=['CumResSquared','Var'])
        return(result)
    
    def predict(self):
        Predicted_DF = pd.DataFrame([
                pd.Series(np.linspace(self.data.N_Pred.min(),self.data.N_Pred.max(),1000),name='N_Pred')
            ]).T
        Predicted_DF['N_Pred_Log'] = Predicted_DF.N_Pred.apply(np.log)
        Predicted_DF['N_Pred_cfun'] = Predicted_DF.apply(self.c_function,axis=1)

        Predicted_DF['Constant'] = 1
        sel_flds = ['Constant','N_Pred_Log']
        Predicted_DF['Variance'] = Predicted_DF.apply(lambda row:np.dot(np.dot(row[sel_flds].to_numpy(),self.result.cov_params()),row[sel_flds].to_numpy().T),axis=1)
        Predicted_DF['Std_Err'] = Predicted_DF.Variance.apply(np.sqrt)
        Predicted_DF['Upper'] = Predicted_DF.N_Pred_cfun.apply(np.log) + Predicted_DF.Std_Err * 1.96
        Predicted_DF['Lower'] = Predicted_DF.N_Pred_cfun.apply(np.log) - Predicted_DF.Std_Err * 1.96
        Predicted_DF['Upper'] = Predicted_DF.Upper.apply(np.exp)
        Predicted_DF['Lower'] = Predicted_DF.Lower.apply(np.exp)
        self.predicted = Predicted_DF
        return(Predicted_DF)
    
    def plot_calibration_function(self):
        coef = self.result.params
        L1 = 'Data Points\n - Number of Sites: {:0,.0f}\n - Total Crashes: {:0,.0f}'.format(self.data.shape[0],self.data.N_Obs.sum())
        L2 = 'Fitted Calibration Function & 95% CI\n - A_Coef: {:0,.3f}\n - B_Coef: {:0.4f}\n - Overdisperssion (k): {:0,.3f}'.format(np.exp(coef.Intercept),coef.N_Pred_Log,self.alpha)

        fig = plt.figure(figsize=(18,5))

        plt.subplot(121)
        ax = sns.scatterplot(data=self.data, x='N_Pred', y='N_Obs',label=L1,color='grey',marker='.')
        sns.kdeplot(data=self.data, x='N_Pred', y='N_Obs',fill=False,label='Kernel Density Of Data Points')
        plt.plot(self.predicted.N_Pred,self.predicted.N_Pred_cfun,'red',label=L2)
        ax.fill_between(self.predicted.N_Pred, self.predicted.Lower, self.predicted.Upper, alpha=0.2,color='red')

        ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.2f}'))
        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.2f}'))
        plt.xlabel('Predicted Crashes')
        plt.ylabel('Observed Crashes')
        plt.title('Calibration Function')
        plt.legend(loc='upper left')
        plt.grid()

        ax = plt.subplot(122)
        plt.plot(self.data.N_Pred_cfun.tolist(),self.data.Lim.tolist(),'g',label='Two Standard Deviation Boundaries')
        plt.plot(self.data.N_Pred_cfun.tolist(),(-self.data.Lim).tolist(),'g')
        plt.plot(self.data.N_Pred_cfun.tolist(),self.data.CumRes.tolist(),'b',label='Cumulative Residuals')
        plt.hlines(0,self.data.N_Pred_cfun.min(),self.data.N_Pred_cfun.max(),'black')
        ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.2f}'))
        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.2f}'))
        plt.xlabel('Predicted Crashes')
        plt.ylabel('Cumulative Residuals (CURE)')
        plt.title('Cumulative Residuals (CURE)')
        plt.legend(loc='upper left')
        plt.grid()
        
        return(fig)
        
    def summary(self):
        ds = pd.Series({
                    'Sites':self.data.shape[0],'Total_Crashes':self.data.N_Obs.sum(),
                    
                    'A_Coef':np.exp(self.result.params.Intercept),'B_Coef':self.result.params.N_Pred_Log,
                    'A_Coef_pvalue':self.result.pvalues.Intercept,'B_Coef_pvalue':self.result.pvalues.N_Pred_Log,
                    'A_Coef_se':self.result.bse.Intercept,'B_Coef_se':self.result.bse.N_Pred_Log,
                    'A_Coef_cv':self.result.bse.Intercept/self.result.params.Intercept,'B_Coef_cv':self.result.bse.N_Pred_Log/self.result.params.N_Pred_Log,
                    
                    'Overdisperssion':self.alpha,
                    
                    'Obs_Crashes_Mean':self.data.N_Obs.mean(),'Pred_Crashes_Mean':self.data.N_Pred.mean(),
                    })
        self.cf_summary = ds
        return(ds)
class Calibration_Factor(object):
    """
    
    """
    
    def __init__(self, data, Observed='obs_kabco',Predicted='pred_kabco',bootstrap=5000):
        data = data[[Observed,Predicted]].copy(deep=True)
        data = data.rename(columns={Observed:'N_Obs',Predicted:'N_Pred'})
        
        self.data   = data
        self.bootstrap = bootstrap
        
    @property
    def overdisperssion(self):
        """overdisperssion in observed crashes"""
        m = np.mean(self.data.N_Obs)
        v = np.var(self.data.N_Obs)
        if v>m:
            self._overdisperssion = (v-m)/(m**2)
        else:
            self._overdisperssion = 0
        return(self._overdisperssion)
    
    
    def bootstrap_function(self,n_obs,n_pred):
        calibration_factor_list = []
        idx = pd.Series(n_obs.index)
        for i in range(self.bootstrap):
            idx_bs = idx.sample(len(idx),replace=True)
            calibration_factor_list.append(n_obs.loc[idx_bs].sum()/n_pred.loc[idx_bs].sum())
        return(calibration_factor_list)
    
    def calibrate(self):
        o = self.data.N_Obs
        p = self.data.N_Pred
        cf = o.sum()/p.sum()
        self.calibration_factor = cf
        self.calibration_factor_list = self.bootstrap_function(o,p)
        self.calibration_factor_se  = np.std(self.calibration_factor_list)
        self.calibration_factor_cv  = self.calibration_factor_se/cf
        
        self.data['N_Pred_cfac'] = self.data.N_Pred * cf
        self.data['Residuals'] = self.data.N_Obs - self.data.N_Pred_cfac
        self.data.sort_values('N_Pred',inplace=True)
        self.data['CumRes'] = self.data.Residuals.cumsum()
        self.data['CumResSquared'] = self.data.Residuals.pow(2).cumsum()
        self.data['Var'] = self.data.CumResSquared * (1-self.data.CumResSquared/self.data.CumResSquared.max())
        self.data['Lim'] = 2 * self.data.Var.pow(0.5)
        self.data = self.data.drop(columns=['CumResSquared','Var'])
        
        return(cf)
        
    def predict(self):
        Predicted_DF = pd.DataFrame([
                pd.Series(np.linspace(self.data.N_Pred.min(),self.data.N_Pred.max(),1000),name='N_Pred')
            ]).T
        
        Predicted_DF['N_Pred_CFac'] = Predicted_DF.N_Pred * self.calibration_factor

        Predicted_DF['Std_Err'] = Predicted_DF.N_Pred_CFac * self.calibration_factor_se
        Predicted_DF['Upper'] = Predicted_DF.N_Pred_CFac + Predicted_DF.Std_Err * 1.96
        Predicted_DF['Lower'] = Predicted_DF.N_Pred_CFac - Predicted_DF.Std_Err * 1.96
        self.predicted = Predicted_DF
        return(Predicted_DF)
    
    def plot_calibration_factor(self):
        L1 = 'Data Points\n - Number of Sites: {:0,.0f}\n - Total Crashes: {:0,.0f}'.format(self.data.shape[0],self.data.N_Obs.sum())
        L2 = 'Calibrated Crashes & 95% CI\n - Calibration Factor: {:0.4f}\n - Coefficient of Variation (CV): {:0,.2%}'.format(self.calibration_factor,self.calibration_factor_cv)

        fig = plt.figure(figsize=(18,5))

        plt.subplot(121)
        ax = sns.scatterplot(data=self.data, x='N_Pred', y='N_Obs',label=L1,color='grey',marker='.')
        sns.kdeplot(data=self.data, x='N_Pred', y='N_Obs',fill=False,label='Kernel Density Of Data Points')
        plt.plot(self.predicted.N_Pred,self.predicted.N_Pred_CFac,'red',label=L2)
        ax.fill_between(self.predicted.N_Pred, self.predicted.Lower, self.predicted.Upper, alpha=0.2,color='red')

        ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.2f}'))
        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.2f}'))
        plt.xlabel('Predicted Crashes')
        plt.ylabel('Observed Crashes')
        plt.title('Calibration Factor')
        plt.legend(loc='upper left')
        plt.grid()

        ax = plt.subplot(122)
        plt.plot(self.data.N_Pred_cfac.tolist(),self.data.Lim.tolist(),'g',label='Two Standard Deviation Boundaries')
        plt.plot(self.data.N_Pred_cfac.tolist(),(-self.data.Lim).tolist(),'g')
        plt.plot(self.data.N_Pred_cfac.tolist(),self.data.CumRes.tolist(),'b',label='Cumulative Residuals')
        plt.hlines(0,self.data.N_Pred_cfac.min(),self.data.N_Pred_cfac.max(),'black')
        ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.2f}'))
        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.2f}'))
        plt.xlabel('Predicted Crashes')
        plt.ylabel('Cumulative Residuals (CURE)')
        plt.title('Cumulative Residuals (CURE)')
        plt.legend(loc='upper left')
        plt.grid()
        
        return(fig)
    def summary(self):
        ds = pd.Series({
                    'Sites':self.data.shape[0],'Total_Crashes':self.data.N_Obs.sum(),
                    
                    'Calibration_Factor':self.calibration_factor,'Coefficient_Of_Variation':self.calibration_factor_cv,
                    
                    'Obs_Crashes_Mean':self.data.N_Obs.mean(),'Pred_Crashes_Mean':self.data.N_Pred.mean(),
                    })
        self.cf_summary = ds
        return(ds)
class IntersectionSPF(object):
    """
    crash : is the total observed crashes over the Num_Years. Do not divide number of crashes by number of years.
    the calculated spf will return the crashes per year
    CURE plots are against AADT Major
    """
    import statsmodels.formula.api as smf 
    import statsmodels as sm
    
    def __init__(self, data,AADT_Major='AADT_Major',AADT_Minor='AADT_Minor',crash='N_Obs',Num_Years = 5,alpha_range=(0.01,2),alpha_iter = 200):
        data = data[[AADT_Major,AADT_Minor,crash]].copy(deep=True)
        data = data.rename(columns={AADT_Major:'AADT_Major',AADT_Minor:'AADT_Minor',crash:'N_Obs'})
        
        data['AADT_Major_Log']    = data.AADT_Major.apply(np.log)
        data['AADT_Minor_Log']    = data.AADT_Minor.apply(np.log)
        data['CF_PerY']     = data.N_Obs/Num_Years
        data['Constant'] = 1
        self.data   = data
        self.n_years= Num_Years
        self.alpha_range = alpha_range
        self.alpha_iter = alpha_iter
        
    @property
    def alpha(self):
        LL_DS = pd.Series()
        for a in np.linspace(self.alpha_range[0], self.alpha_range[1], self.alpha_iter) :
            model = self.smf.glm(formula = "N_Obs ~ AADT_Major_Log + AADT_Minor_Log", data=self.data, family=self.sm.genmod.families.NegativeBinomial(alpha=a)).fit()
            LL_DS.loc[a] = model.llf
        alpha = LL_DS[LL_DS==LL_DS.max()].index[0]
        #print('alpha: {:0.4f}'.format(alpha))
        self._ll_vs_alpha = LL_DS
        self._alpha = alpha
        return(self._alpha)
    
    @property
    def spf_fun(self):
        """calculates crashes per year per mile based on current SPF"""
        self._spf_fun = lambda row:row.AADT_Major**self.result.params.loc['AADT_Major_Log']*row.AADT_Minor**self.result.params.loc['AADT_Minor_Log']*np.exp(self.result.params.loc['Intercept'])
        return(self._spf_fun)
    
    @property
    def overdisperssion(self):
        """overdisperssion in observed crashes"""
        m = np.mean(self.data.N_Obs)
        v = np.var(self.data.N_Obs)
        if v>m:
            self._overdisperssion = (v-m)/(m**2)
        else:
            self._overdisperssion = 0
        return(self._overdisperssion)
    
    @property
    def model(self):
        fr = "CF_PerY ~ AADT_Major_Log + AADT_Minor_Log"
        fm = self.sm.genmod.families.NegativeBinomial(alpha=self.alpha)
        self._model = self.smf.glm(formula = fr, data=self.data, family=fm)
        return(self._model)
    
    def calibrate_spf(self):
        o = self.data.N_Obs.sum()
        p = self.data.apply(self.spf_fun,axis=1) * self.n_years
        cf = o/p.sum()
        self.calibration_factor = cf
        self.result.params.loc['Intercept'] = self.result.params.loc['Intercept'] + np.log(self.calibration_factor)
        
    def fit(self):
        result = self.model.fit()
        self.result = result
        self.calibrate_spf()
    
        self.data['N_Pred'] = self.data.apply(self.spf_fun,axis=1) * self.n_years
        self.data['Pred_CF_PerY'] = self.data.N_Pred/self.n_years
    
        sel_flds = ['Constant','AADT_Major_Log','AADT_Minor_Log']
        self.data['Variance'] = self.data.apply(lambda row:np.dot(np.dot(row[sel_flds].to_numpy(),result.cov_params()),row[sel_flds].to_numpy().T),axis=1)
        self.data['Std_Err']  = self.data.Variance.apply(np.sqrt)
    
        
        self.data['Upper'] = self.data.Pred_CF_PerY.apply(np.log) + self.data.Std_Err * 1.96
        self.data['Lower'] = self.data.Pred_CF_PerY.apply(np.log) - self.data.Std_Err * 1.96
        self.data['Upper'] = self.data.Upper.apply(np.exp)
        self.data['Lower'] = self.data.Lower.apply(np.exp)

        self.data['Residuals'] = self.data.CF_PerY - self.data.Pred_CF_PerY
        self.data.sort_values('AADT_Major',inplace=True)
        self.data['CumRes'] = self.data.Residuals.cumsum()
        self.data['CumResSquared'] = self.data.Residuals.pow(2).cumsum()
        self.data['Var'] = self.data.CumResSquared * (1-self.data.CumResSquared/self.data.CumResSquared.max())
        self.data['Lim'] = 2 * self.data.Var.pow(0.5)
        self.data = self.data.drop(columns=['CumResSquared','Var'])
        return(result)
    
    def predict(self):
        Predicted_DF = pd.DataFrame([
                pd.Series(np.linspace(self.data.AADT_Major.min(),self.data.AADT_Major.max(),1000),name='AADT_Major')
            ]).T
        Predicted_DF['Constant'] = 1
        Predicted_DF['AADT_Minor'] = self.data.AADT_Minor.median()
        Predicted_DF['AADT_Major_Log'] = Predicted_DF.AADT_Major.apply(np.log)
        Predicted_DF['AADT_Minor_Log'] = Predicted_DF.AADT_Minor.apply(np.log)

        Predicted_DF['Pred_CF'] = Predicted_DF.apply(self.spf_fun,axis=1)

        sel_flds = ['Constant','AADT_Major_Log','AADT_Minor_Log']
        Predicted_DF['Variance'] = Predicted_DF.apply(lambda row:np.dot(np.dot(row[sel_flds].to_numpy(),self.result.cov_params()),row[sel_flds].to_numpy().T),axis=1)
        Predicted_DF['Std_Err'] = Predicted_DF.Variance.apply(np.sqrt)
        Predicted_DF['Upper'] = Predicted_DF.Pred_CF.apply(np.log) + Predicted_DF.Std_Err * 1.96
        Predicted_DF['Lower'] = Predicted_DF.Pred_CF.apply(np.log) - Predicted_DF.Std_Err * 1.96
        Predicted_DF['Upper'] = Predicted_DF.Upper.apply(np.exp)
        Predicted_DF['Lower'] = Predicted_DF.Lower.apply(np.exp)
        self.predicted = Predicted_DF
        return(Predicted_DF)
    
    def plot_spf(self):
        coef = self.result.params
        L1 = 'Data Points\nNumber of Intersections: {:0,.0f}\nTotal Crashes: {:0,.0f}'.format(self.data.shape[0],self.data.N_Obs.sum())
        L2 = 'Fitted SPF & 95% CI\nIntercept: {:0,.3f}\nAADT Major Coef: {:0.4f}\nAADT Minor Coef: {:0.4f}\nOverdisperssion (k): {:0,.3f}'.format(coef.Intercept,coef.AADT_Major_Log,coef.AADT_Minor_Log,self.alpha)

        fig = plt.figure(figsize=(18,5))

        ax = plt.subplot(131)
        sns.scatterplot(data=self.data, x='AADT_Major', y='AADT_Minor',label=L1,color='grey',marker='.',ax=ax)
        #sns.kdeplot(data=self.data, x='AADT_Major', y='AADT_Minor',label='Kernel Density Of Data Points',levels=10,thresh=0.2)
        ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
        plt.xlabel('AADT Major')
        plt.ylabel('AADT Minor')
        plt.title('Domain Of Applicability')
        plt.legend(loc='upper left')
        plt.grid()

        ax = plt.subplot(132)
        sns.scatterplot(data=self.data, x='AADT_Major', y='CF_PerY',label='Data Points',color='grey',marker='.',ax=ax)
        #sns.kdeplot(data=self.data, x='AADT_Major', y='CF_PerY',fill=False,label='Kernel Density Of Data Points',ax=ax)
        plt.plot(self.predicted.AADT_Major,self.predicted.Pred_CF,'red',label=L2)
        ax.fill_between(self.predicted.AADT_Major, self.predicted.Lower, self.predicted.Upper, alpha=0.2,color='red')

        ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.2f}'))
        plt.xlabel('AADT Major')
        plt.ylabel('Crash Frequency per Year (AADT Minor is assumed: {:,.0f})'.format(self.predicted.AADT_Minor.iloc[0]))
        plt.title('Fitted Model')
        plt.legend(loc='upper left')
        plt.grid()

        ax = plt.subplot(133)
        plt.plot(self.data.AADT_Major.tolist(),self.data.Lim.tolist(),'g',label='Two Standard Deviation Boundaries')
        plt.plot(self.data.AADT_Major.tolist(),(-self.data.Lim).tolist(),'g')
        plt.plot(self.data.AADT_Major.tolist(),self.data.CumRes.tolist(),'b',label='Cumulative Residuals')
        plt.hlines(0,self.data.AADT_Major.min(),self.data.AADT_Major.max(),'black')
        ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
        plt.xlabel('AADT Major')
        plt.ylabel('Cumulative Residuals (CURE)')
        plt.title('Cumulative Residuals (CURE)')
        plt.legend(loc='upper left')
        plt.grid()
        
        return(fig)

    def summary(self):
        ds = pd.Series({'NumberOfYears':self.n_years,
                    'Intersections':self.data.shape[0],'Total_Crashes':self.data.N_Obs.sum(),
                    'Min_AADT_Major':self.data.AADT_Major.min(),'Max_AADT_Major':self.data.AADT_Major.max(),
                    
                    'Intercept':self.result.params.Intercept,'AADT_Major_Coef':self.result.params.AADT_Major_Log,'AADT_Minor_Coef':self.result.params.AADT_Minor_Log,
                    'Intercept_pvalue':self.result.pvalues.Intercept,'AADT_Major_pvalue':self.result.pvalues.AADT_Major_Log,'AADT_Minor_pvalue':self.result.pvalues.AADT_Minor_Log,
                    'Intercept_se':self.result.bse.Intercept,'AADT_Major_se':self.result.bse.AADT_Major_Log,'AADT_Minor_se':self.result.bse.AADT_Minor_Log,
                    'Intercept_cv':self.result.bse.Intercept/self.result.params.Intercept,'AADT_Major_cv':self.result.bse.AADT_Major_Log/self.result.params.AADT_Major_Log,'AADT_Minor_cv':self.result.bse.AADT_Minor_Log/self.result.params.AADT_Minor_Log,
                    
                    'Overdisperssion':self.alpha,
                    'Calibration_Factor':self.calibration_factor,
                    'Obs_Crashes_Mean':self.data.N_Obs.mean(),'Pred_Crashes_Mean':self.data.N_Pred.mean(),
                    'Obs_CF_PerY':self.data.CF_PerY.mean(),'Pred_CF_PerY':self.data.Pred_CF_PerY.mean(),
                    })
        self.spf_summary = ds
        return(ds)