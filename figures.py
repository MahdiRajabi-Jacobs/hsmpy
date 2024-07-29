import sys
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

def PlotPolylines(List,anotate = False):
    size = 25
    plt.figure(figsize=(size,size))
    for l in List:
        x = [i.X for i in l]
        y = [i.Y for i in l]
        plt.plot(x, y,'-')
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off') # labels along the bottom edge are off
    plt.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the bottom edge are off
        right='off',         # ticks along the top edge are off
        labelleft='off') # labels along the bottom edge are off
    if anotate:
        for l in List:    
            x = [i.X for i in l]
            y = [i.Y for i in l]
            for i in range(len(l)):
                plt.annotate(str(i),(x[i],y[i]))    
    for l in List:    
        x = list(l)[0].X
        y = list(l)[0].Y
        m = list(l)[0].M
        plt.annotate(m,(x,y))    
        x = list(l)[-1].X
        y = list(l)[-1].Y
        m = list(l)[-1].M
        plt.annotate(m,(x,y))    

    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

# Plots
def PairTable(DF,Rows,Cols,Percentage=False,Plot=False,figsize=(10,10)):
    All = Rows + Cols
    df = pd.DataFrame(DF.groupby(All).size())
    df = df.rename(columns={0:''})
    #df1.Fatalities = df1.Fatalities/df1.Fatalities.sum()
    df = df.unstack(level=[All.index(c) for c in Cols])
    row_drop = []
    col_drop = []
    try:
        for i,l in enumerate(df.index.levels):
            if len(list(l))==1:
                row_drop.append(i)
    except:
        pass
    try:
        for i,l in enumerate(df.columns.levels):
            if len(list(l))==1:
                col_drop.append(i)
    except:
        pass
    if len(row_drop)>0:
        df.index = df.index.droplevel(row_drop)
    if len(col_drop)>0:
        df.columns = df.columns.droplevel(col_drop)
    if Percentage:
        for i,r in df.iterrows():
            df.loc[i] = df.loc[i]/sum(df.loc[i])
    if Plot:
        plt.imshow(df,cmap=plt.cm.Reds,interpolation='nearest',aspect='equal')
        plt.xticks(range(0,df.shape[1]),df.columns,rotation=90)
        plt.yticks(range(0,df.shape[0]),df.index,rotation=0)
        plt.show()
    return(df)

#Crash Trends
def TrendFigure(DataSeries,RollingSpan,ylab,title,fname):

    F_Cond_B = mpl.font_manager.FontProperties(fname=r"\\chcfpp01\Groups\HTS\Code_Repository\Fonts\Roboto-BoldCondensed_0.ttf")
    F_Cond_B1 = mpl.font_manager.FontProperties(fname=r"\\chcfpp01\Groups\HTS\Code_Repository\Fonts\Roboto-BoldCondensed_0.ttf",size=16)
    S = DataSeries.rolling(RollingSpan).mean()
    S.index = list(range(0,len(S)))
    fs = (14,7)
    ax = DataSeries.plot.bar(stacked=True,figsize=fs,color='#AA2D29',label=DataSeries.name) 
    S.plot(label= '{} Year Rolling Average'.format(RollingSpan),ax=ax,marker='o',color='#003057',markersize=8)    
    plt.ylim(0,DataSeries.max()*1.1)
    plt.grid(True,linestyle='--', color='#7B868C')
    plt.ylabel(ylab,fontproperties=F_Cond_B,size=18)
    plt.xlabel('')
    plt.gca().yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    plt.legend()
    plt.title(title,fontproperties=F_Cond_B,fontsize=16)
    
    #ax = plt.gca()
    ax.set_axisbelow(True)
    for l in ax.yaxis.get_ticklabels():
        l.set_fontproperties(F_Cond_B)
    ax.yaxis.set_tick_params(color='#414042', labelsize=16)
    for l in ax.xaxis.get_ticklabels():
        l.set_fontproperties(F_Cond_B)
    ax.xaxis.set_tick_params(color='#414042', labelsize=16)
    ax.grid(True,linestyle='--', color='#7B868C')
    table2 = ax.table(cellText = [list(DataSeries.apply('{:,.0f}'.format))],
                      rowLabels = [ylab],
                         bbox=[0,-0.07,1,-0.05],
                         rowLoc='right',colLoc='right',cellLoc='center',
                         loc ='bottom')  
    
    for (row, col), cell in table2.get_celld().items():
            cell.set_text_props(fontproperties=F_Cond_B1)
    table2 = ax.table(cellText = [list(S.apply('{:,.0f}'.format).replace({'nan':'-'}))],
                      rowLabels = ['{} Year Rolling Average'.format(RollingSpan)],
                         bbox=[0,-0.12,1,-0.05],
                         fontsize=16,rowLoc='right',colLoc='right',cellLoc='center',
                         loc ='bottom')  
    for (row, col), cell in table2.get_celld().items():
            cell.set_text_props(fontproperties=F_Cond_B1)
    plt.savefig(fname,dpi=1200,transparent=True,bbox_inches = 'tight',pad_inches = 0)
    return(ax)

def TimeTrend_Contour(S):
    """
    S is a pandas series with datetime index and the values are used to create the contour plots
    """

    F_Cond_B = mpl.font_manager.FontProperties(fname=r"C:\Local_Proj\CTDOT\TechMemo\Fonts\Roboto-BoldCondensed_0.ttf")
    import warnings
    warnings.filterwarnings('ignore')
    TimeOrder = [datetime(2000,1,1,d,0).strftime('%I:%M %p') for d in list(range(6,24)) + list(range(0,6))]
    TimeOrder.reverse()
    DayOrder = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    MonthOrder = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    DF  = S.reset_index()
    DF.columns = ['DATE','VALUE']
    DF['Time'] = DF.DATE.dt.hour.apply(lambda x:datetime(2000,1,1,x,0).strftime('%I:%M %p'))
    DF['DayName'] = DF.DATE.dt.day_name()
    DF['Month']  = DF.DATE.dt.month_name().apply(lambda x:x[:3])#[d.strftime('%b') for d in DF.Date]

    df = DF.groupby(['Month','DayName','Time'])['VALUE'].sum().unstack([0,1])
    df = df.reindex(TimeOrder)
    df = df.T.reindex(pd.MultiIndex.from_product([MonthOrder,DayOrder])).T
    df = df.fillna(0).astype(int)

    sdate = datetime(DF.DATE.dt.year.min(), 1, 1)
    edate = datetime(DF.DATE.dt.year.max()+1, 1, 1) - timedelta(days=1)

    idx = pd.to_datetime(pd.Series([sdate + timedelta(days=i) for i in range((edate - sdate).days + 1)],name='DATE'))
    df1 = pd.DataFrame(index=idx)
    df1.reset_index(inplace=True)
    df1['DayName'] = df1.DATE.dt.day_name()
    df1['Month']   = df1.DATE.dt.month_name().apply(lambda x:x[:3])#[d.strftime('%b') for d in DF.Date]
    df1 = df1.groupby(['Month','DayName']).size()
    df1 = df1.loc[pd.MultiIndex.from_product([MonthOrder,DayOrder])]
    df = df.T.div(df1,axis=0).T

    ylabels = [datetime(2000,1,1,d,0).strftime('%I %p') for d in [6,8,10,12,14,16,18,20,22,0,2,4]]
    ylabels.reverse()
    yticks  = [1,3,5,7,9,11,13,15,17,19,21,23]

    fig = plt.figure(figsize=(10, 6))#, facecolor='w', edgecolor='k')
    ax1 = fig.add_subplot(2,1,1)

    df1 = df[MonthOrder[0:6]]
    pl1 = plt.contourf(df1, cmap=plt.cm.Reds,corner_mask=True ,alpha=1,v_min=df.min().min(),v_max = df.max().max())
    xposition = [7*(i1)-0.5 for i1 in range(1,6)]
    for xc in xposition:
        plt.axvline(x=xc, color='k', linestyle='--')
    xl = [{True:f[1][0:2] + '  ' + f[0],False:f[1][0:2]}[f[1][0:2]=='Th'] for f in df1.columns]
    plt.xticks(range(len(list(df1))),xl,rotation=90,fontsize=8,fontproperties=F_Cond_B,color='#414042')
    ax1.xaxis.tick_top()
    #plt.yticks(range(1,len(df1.index)),list(df1.index)[1:],rotation=0,fontsize=6,fontproperties=F_Cond_B)
    plt.yticks(yticks,ylabels,rotation=0,fontsize=10,color='#414042')
    
    for y in [5,17]:
        ax1.axhline(y , linestyle='--', color='#7B868C') # horizontal lines
    for x in [4,11,18,25,32,39]:
        ax1.axvline(x, linestyle='--', color='#7B868C') # vertical lines
    
    #plt.grid()
    df1 = df[MonthOrder[6:13]]

    ax2 = fig.add_subplot(2,1,2)
    pl2 = plt.contourf(df1, cmap=plt.cm.Reds,corner_mask=True ,alpha=1,v_min=df.min().min(),v_max = df.max().max())
    for xc in xposition:
        plt.axvline(x=xc, color='k', linestyle='--')
    xl = [{True:f[0] + '  ' + f[1][0:2],False:f[1][0:2]}[f[1][0:2]=='Th'] for f in df1.columns]

    plt.xticks(range(len(list(df1))),xl,rotation=90,fontsize=8,fontproperties=F_Cond_B,color='#414042')
    #plt.yticks(range(len(df1.index)),df1.index,rotation=0,fontsize=6,fontproperties=F_Cond_B)
    plt.yticks(yticks,ylabels,rotation=0,fontsize=10,color='#414042')

    for y in [5,17]:
        ax2.axhline(y , linestyle='--', color='#7B868C') # horizontal lines
    for x in [4,11,18,25,32,39]:
        ax2.axvline(x, linestyle='--', color='#7B868C') # vertical lines
        
    #plt.grid()
    plt.subplots_adjust(wspace=0, hspace=0.04)
    cb = fig.colorbar(pl1,ax=[ax1,ax2],pad=0.01)
    
    for l in cb.ax.yaxis.get_ticklabels():
        l.set_fontproperties(F_Cond_B)
    cb.ax.yaxis.set_tick_params(color='#414042')
    cb.ax.get_yaxis().labelpad = 15
    cb.ax.set_ylabel('Average Number of KA Injuries per Month-Weekday-Hour (2016-2020)', rotation=270,fontproperties=F_Cond_B,color='#414042')
    params = {"text.color": '#414042',
              "xtick.color": '#414042',
              "ytick.color": '#414042'}
    plt.rcParams.update(params)
    return(fig)