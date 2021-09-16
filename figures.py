import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class MidPointNorm(Normalize):    
    from numpy import ma
    from matplotlib import cbook
    from matplotlib.colors import Normalize

    def __init__(self, midpoint=0, vmin=None, vmax=None, clip=False):
        Normalize.__init__(self,vmin, vmax, clip)
        self.midpoint = midpoint

    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip

        result, is_scalar = self.process_value(value)

        self.autoscale_None(result)
        vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint

        if not (vmin < midpoint < vmax):
            raise ValueError("midpoint must be between maxvalue and minvalue.")       
        elif vmin == vmax:
            result.fill(0) # Or should it be all masked? Or 0.5?
        elif vmin > vmax:
            raise ValueError("maxvalue must be bigger than minvalue")
        else:
            vmin = float(vmin)
            vmax = float(vmax)
            if clip:
                mask = ma.getmask(result)
                result = ma.array(np.clip(result.filled(vmax), vmin, vmax),
                                  mask=mask)

            # ma division is very slow; we can take a shortcut
            resdat = result.data

            #First scale to -1 to 1 range, than to from 0 to 1.
            resdat -= midpoint            
            resdat[resdat>0] /= abs(vmax - midpoint)            
            resdat[resdat<0] /= abs(vmin - midpoint)

            resdat /= 2.
            resdat += 0.5
            result = ma.array(resdat, mask=result.mask, copy=False)                

        if is_scalar:
            result = result[0]            
        return result

    def inverse(self, value):
        if not self.scaled():
            raise ValueError("Not invertible until scaled")
        vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint

        if cbook.iterable(value):
            val = ma.asarray(value)
            val = 2 * (val-0.5)  
            val[val>0]  *= abs(vmax - midpoint)
            val[val<0] *= abs(vmin - midpoint)
            val += midpoint
            return val
        else:
            val = 2 * (val - 0.5)
            if val < 0: 
                return  val*abs(vmin-midpoint) + midpoint
            else:
                return  val*abs(vmax-midpoint) + midpoint

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
