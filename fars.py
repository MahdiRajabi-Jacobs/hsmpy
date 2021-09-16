import math
from datetime import datetime
import pandas as pd
import arcpy
import os
import hsmpy3.common as common
import sys, csv, json, subprocess
import json
import xmltodict
import numpy as np
import urllib.request
import hsmpy3.network as network
import matplotlib.pyplot as plt
from time import gmtime, strftime
import scipy
import numpy
import matplotlib
import astropy
import pytz
import warnings
from statsmodels.tsa.seasonal import seasonal_decompose

#class domains():
State_Dict = {
    1:"Alabama",
    2:"Alaska",
    4:"Arizona",
    5:"Arkansas",
    6:"California",
    8:"Colorado",
    9:"Connecticut",
    10:"Delaware",
    11:"District of Columbia",
    12:"Florida",
    13:"Georgia",
    15:"Hawaii",
    16:"Idaho",
    17:"Illinois",
    18:"Indiana",
    19:"Iowa",
    20:"Kansas",
    21:"Kentucky",
    22:"Louisiana",
    23:"Maine",
    24:"Maryland",
    25:"Massachusetts",
    26:"Michigan",
    27:"Minnesota",
    28:"Mississippi",
    29:"Missouri",
    30:"Montana",
    31:"Nebraska",
    32:"Nevada",
    33:"New Hampshire",
    34:"New Jersey",
    35:"New Mexico",
    36:"New York",
    37:"North Carolina",
    38:"North Dakota",
    39:"Ohio",
    40:"Oklahoma",
    41:"Oregon",
    42:"Pennsylvania",
    43:"Puerto Rico",
    44:"Rhode Island",
    45:"South Carolina",
    46:"South Dakota",
    47:"Tennessee",
    48:"Texas",
    49:"Utah",
    50:"Vermont",
    51:"Virginia",
    52:"Virgin Islands",
    53:"Washington",
    54:"West Virginia",
    55:"Wisconsin",
    56:"Wyoming",
}
ROAD_FUN_Dict = {
        1: 'Rural-Principal Arterial – Interstate',
        2: 'Rural-Principal Arterial – Other',
        3: 'Rural-Minor Arterial',
        4: 'Rural-Major Collector',
        5: 'Rural-Minor Collector',
        6: 'Rural-Local Road or Street',
        9: 'Rural-Unknown Rural',
        11: 'Urban-Principal Arterial – Interstate',
        12: 'Urban-Principal Arterial – Other (Freeways or Expressways)',
        13: 'Urban-Other Principal Arterial',
        14: 'Urban-Minor Arterial',
        15: 'Urban-Collector',
        16: 'Urban-Local Road or Street',
        19: 'Urban-Unknown Urban',
        99: 'Unknown',
}
FUN_SYS_Dict = {
        1: '01.Interstate',
        2: '02.Principal Arterial – Other Freeways and Expressways',
        3: '03.Principal Arterial – Other ',
        4: '04.Minor Arterial',
        5: '05.Major Collector',
        6: '06.Minor Collector',
        7: '07.Local',
        96: '08.Trafficway Not in State Inventory',
        98: '09.Not Reported',
        99: '10.Unknown'
}
RUR_URB_Dict = {
    1: '1.Rural',
    2: '2.Urban',
    6: '3.Trafficway Not in State Inventory',
    8: '4.Not Reported',
    9: '5.Unknown'
}
FARS_Ownership = {
1:'State Highway Agency',
2:'County Highway Agency',
3:'Town or Township Highway Agency',
4:'City or Municipal Highway Agency',
11:'State Park, Forest or Reservation Agency',
12:'Local Park, Forest or Reservation Agency',
21:'Other State Agency',
25:'Other Local Agency',
26:'Private (other than Railroad)',
27:'Railroad',
31:'State Toll Road',
32:'Local Toll Authority',
40:'Other Public Instrumentality (i.e., Airport)',
50:'Indian Tribe Nation',
60:'Other Federal Agency',
62:'Bureau of Indian Affairs',
63:'Bureau of Fish and Wildlife',
64:'U.S. Forest Service',
66:'National Park Service',
67:'Tennessee Valley Authority',
68:'Bureau of Land Management',
69:'Bureau of Reclamation',
70:'Corps of Engineers',
72:'Air Force',
74:'Navy/Marines',
80:'Army',
96:'Trafficway Not in State Inventory',
98:'Not Reported',
99:'Unknown'}



# Figures
def EMA(Data,window):
        Data =Data.fillna(0)
        ema = pd.ewma(Data,span = window).loc[list(Data.index)[window-1:]]
        #z = np.polyfit(ema.index, ema, 4)
        #f = np.poly1d(z)
        x = np.linspace(list(ema.index)[0], list(ema.index)[-1],100)
        try:
            y = spline(list(ema.index),ema,x)
            return(pd.Series(index=x,data=y))
        except:
            print(ema)
            return(pd.Series(index=x,data=0))
def Get_NM_EMA_Diff(cr,slow,fast):
    ema1 = pd.ewma(cr,span = slow)
    ema2 = pd.ewma(cr,span = fast)
    diff = ema2-ema1
    diff_norm = diff/ema1 * 100
    diff_norm = diff_norm.loc[list(cr.index)[slow-1:]]
    return(diff_norm) 
def EMA_Trend_Analysis(S,slow,fast,title,png_out,width=10,height=5):    
    warnings.filterwarnings('ignore')
    gs = matplotlib.gridspec.GridSpec(4, 1)
    #tPlot, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False, gridspec_kw={'height_ratios':[3,1]})
    plt.figure(figsize=(width, height), dpi=300, facecolor='w', edgecolor='k')
    plt.subplot(gs[0:3, :])
    plt.plot(S.index,S,'-.o')
    for k,win in enumerate([slow,fast]):
        plt.plot(EMA(S,win),{0:'-',1:'--',2:'-.'}[k],label=' - EMA {} Years'.format(win))
    #plt.ylim(0,max(df.Fatalities)+20)
    plt.legend(loc='upper right',fancybox=True,framealpha=0.5, prop={'size': 7},ncol=1)
    plt.xticks(df.index,[])
    plt.xlim(min(df.index)-1,max(df.index)+1)
    plt.title(title)
    plt.grid()
    plt.subplot(gs[3, :])
    diff = Get_NM_EMA_Diff(S,slow,fast)
    my_cmap = matplotlib.cm.get_cmap('RdYlGn_r')
    my_norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
    barwidth = 0.5
    plt.bar(diff.index,diff,align='center',color = my_cmap(np.sign(diff)),label='Normalized EMA{}-EMA{} Indicator'.format(fast,slow),width=barwidth)
    x = np.linspace(list(diff.index)[0], list(diff.index)[-1],100)
    y = spline(list(diff.index),diff,x)
    plt.plot(x,y)
    yt = list(plt.gca().get_yticks())
    #yt.reverse()
    #plt.gca().set_yticklabels(['{:,.2%}'.format(x) for x in yt])
    plt.gca().yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
    plt.xticks(df.index,df.index.astype(int),rotation=90)
    plt.xlim(min(df.index)-1,max(df.index)+1)
    plt.legend(loc='upper left',fancybox=True,framealpha=0.5, prop={'size': 7},ncol=1)
    plt.grid()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(png_out,transparent=True,dpi=1200)
    plt.close()
def SeasonalDecomposition(FARS_DF,png_out,width=20,height=15,model='multiplicative',title=''):
    K_Crash_DF = FARS_DF
    K_Crash_DF['K'] = K_Crash_DF.FATALS
    plt.figure(figsize=(width,height))
    Model = 'multiplicative'
    #Model = 'additive'
    N = [3,12]
    K_Crash_DF['Month'] = [d.strftime('%Y-%m') for d in K_Crash_DF.DATE]
    df = pd.DataFrame(K_Crash_DF.groupby('Month')['K'].aggregate(sum).sort_index())
    df.index = pd.to_datetime(df.index)
    result = seasonal_decompose(df.K, model=Model)

    plt.subplot2grid((6, 1), (0, 0), rowspan=4)
    p1 = plt.plot(df,'-o',label='Fataliteis')
    for i in range(2005,2018):
        plt.vlines(datetime(i,1,1,0,0),df.K.min(),df.K.max(),'red',':')
    for n in N:
        df.K.rolling(window=n,center=True,min_periods=n).mean().plot(label='{} Month(s) Moving Average'.format(n))
    result.trend.plot(label='Trend')
    plt.xticks(list(df.index),[])
    plt.xlabel('')
    plt.ylabel('Number of Fatalities per Month')
    plt.grid()
    plt.title('Decomposition of Number of Fatalities by Month - {}'.format(title))
    plt.legend(loc='upper left',fancybox=True)

    plt.subplot2grid((6, 1), (4, 0))
    result.seasonal.plot(label='Seasonal Effect - {}'.format(Model))
    for i in range(2005,2018):
        plt.vlines(datetime(i,1,1,0,0),result.seasonal.min(),result.seasonal.max(),'red',':')
    plt.hlines(y={'additive':0,'multiplicative':1}[Model],colors='red',xmin=df.index.min(),xmax=df.index.max())

    mdf = pd.DataFrame(result.seasonal.iloc[0:12])
    mdf.columns = ['SeasonalFactor']
    mdf.index = [d.strftime('%B') for d in mdf.index]
    mdf = mdf.T
    for c in mdf.columns:
        plt.plot([],[],label = '{}: {:0.3f}'.format(c,mdf.loc['SeasonalFactor',c]))
    plt.xticks(list(df.index),[])
    plt.xlabel('')
    plt.ylabel('Seasonal Factor')
    plt.grid()
    plt.legend(loc='upper left',fancybox=True, prop={'size': 7},ncol=3)

    plt.subplot2grid((6, 1), (5, 0))
    result.resid.plot(label='Residual')
    for i in range(2005,2018):
        plt.vlines(datetime(i,1,1,0,0),result.resid.min(),result.resid.max(),'red',':')
    plt.hlines(y={'additive':0,'multiplicative':1}[Model],colors='red',xmin=df.index.min(),xmax=df.index.max())
    plt.xticks(df.index)
    plt.ylabel('Residuals')
    plt.legend(loc='upper left',fancybox=True)
    plt.grid()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(png_out,transparent=True,dpi=1200)
    plt.close()