# Developed By Mahdi Rajabi mrajabi@clemson.edu
import os
import sys
#import hsmpy3.common as common
#mport hsmpy3.fields as fields
import datetime
import copy
import arcpy
import subprocess 
import time
import pandas as pd
from time import gmtime, strftime
from datetime import timedelta
import numpy as np
from scipy import stats

def secondary(CrashInput,TimeInt,Distance,Output):
    print("Secondary Crashes")
    SPJ = os.path.splitext(Output)[0] + '_SpatialJoin' + os.path.splitext(Output)[1]
    arcpy.Delete_management(Output)
    arcpy.Delete_management(SPJ)

    print('Count: ' + os.path.basename(CrashInput))
    C = arcpy.GetCount_management(CrashInput)
    arcpy.AddMessage("     - Total Items Found: " + str(C))

    def DateDecompose(Date,Time):
            Date = str(Date)
            D = 0
            M = 0
            Y = 0
            if len(Date) == 7:
                M = (Date[0])
                D = (Date[1:3])
                Y = (Date[3:7])
            if len(Date) == 8:
                M = (Date[0:2])
                D = (Date[2:4])
                Y = (Date[4:8])

            h = 0
            m = 0
            Time = str(Time)
            if len(Time) <= 2:
                h = '0'
                m = (Time)
            if len(Time) == 3:
                h = (Time[0])
                m = (Time[1:3])
            if len(Time) == 4:
                h = (Time[0:2])
                m = (Time[2:4])
            
            flag = False
            try:
                m = int(m)
            except:
                flag = True
                m = 0
            try:
                h = int(h)
            except:
                flag = True
                h = 0
            try:
                M = int(M)
            except:
                flag = True
                M = 1
            try:
                D = int(D)
            except:
                flag = True
                D = 1
            try:
                Y = int(Y)
            except:
                flag = True
                Y = 2007
            if m>59: flag = True;m = 59
            if m<0 : flag = True;m = 0
            if h>23: flag = True;h = 23;m = 59
            if h<0 : flag = True;h = 0
            if M>12: flag = True;M = 12
            if M<0 : flag = True;M = 0
            if D>31: flag = True;D = 31
            if D<0 : flag = True;D = 0
            try:
                out = datetime.datetime(int(Y),int(M),int(D),int(h),int(m))
            except:
                falg = True
                out = datetime.datetime(2007,1,1,0,0)
            #if flag:
            #    arcpy.AddWarning('     - Out of range date: ' + Date + ', ' + Time + ' => ' + str(out))

            return(out)

    print("Search Cursor: " + os.path.basename(CrashInput))
    CDic = {SRow.getValue('ANO'):{'P'   :SRow.getValue('Shape'),
                                  'Time':SRow.getValue(fields.loc.TIM['name']),
                                  'Date':SRow.getValue(fields.loc.DAT['name']),
                                  'RCT' :SRow.getValue(fields.loc.RCT['name']),
                                  'RTN' :SRow.getValue(fields.loc.RTN['name']),
                                  'DIR' :SRow.getValue(fields.loc.DLR['name'])} for SRow in arcpy.SearchCursor(CrashInput)}

    print('Merge: ' + os.path.basename(CrashInput))
    fm  = arcpy.FieldMap()
    fms = arcpy.FieldMappings()
    fm.addInputField(CrashInput,'ANO')
    outF = fm.outputField
    outF.name  = fields.loc.ANO['name']
    outF.alias = fields.loc.ANO['alias']  
    outF.type  = fields.loc.ANO['type']
    fm.outputField = outF
    fms.addFieldMap(fm)
    arcpy.Merge_management(CrashInput,Output,fms)

    print("Spatial Join: " + os.path.basename(Output))
    arcpy.SpatialJoin_analysis(Output, Output, SPJ,"JOIN_ONE_TO_MANY","KEEP_ALL",'',"WITHIN_A_DISTANCE",str(Distance)+" Feet")
    SortedANO = sorted(CDic.keys())
    Pairs   = {ANO:[] for ANO in SortedANO}
    for SRow in arcpy.SearchCursor(SPJ):
        ANO1 = SRow.getValue("ANO")
        ANO2 = SRow.getValue("ANO_1")
        if ANO2>ANO1:
            Pairs[ANO1].append(ANO2)
    arcpy.Delete_management(SPJ)
    
    print("Finding Crash Pairs ...")
    
    FPair   = {ANO:-1 for ANO in SortedANO}
    Primary = []
    Secondary = []
    Prg = 0.1
    for ANO1 in SortedANO:
        if float(SortedANO.index(ANO1))/float(str(C)) >= Prg:
            print(" - " + str(int(Prg*100)) + "% Completed")
            Prg = Prg + 0.1
        for ANO2 in Pairs[ANO1]:
            if CDic[ANO1]['RCT'] == CDic[ANO2]['RCT']:
                if CDic[ANO1]['RTN'] == CDic[ANO2]['RTN']:
                    D1 = DateDecompose(CDic[ANO1]['Date'],CDic[ANO1]['Time'])
                    D2 = DateDecompose(CDic[ANO2]['Date'],CDic[ANO2]['Time'])
                    if abs(D1-D2).total_seconds()/3600<=float(TimeInt):
                        if (D2-D1).total_seconds() > 0 and not (ANO1 in Secondary):
                            FPair[ANO2]=(ANO1)
                            Primary.append(ANO1)
                            Secondary.append(ANO2)
                            #print(str(ANO1)+', ' + str(ANO2)+','+str(abs(D1-D2).total_seconds()/3600))
                        if (D1-D2).total_seconds() > 0 and not (ANO2 in Secondary):
                            FPair[ANO1]=(ANO2)
                            Primary.append(ANO2)
                            Secondary.append(ANO1)
                                
    print("Add Field: " + os.path.basename(Output))
    for field in [fields.crash.PrmANO,fields.crash.Tempor,fields.crash.Spatio,fields.loc.Label]:
        print(' - ' + field['name'])
        arcpy.AddField_management(Output,field['name'],field['type'],field['precision'],field['scale'],field['length'],field['alias'],field['nullable'],field['required'])

    print("Update Cursor: " + os.path.basename(Output))
    i = 0
    UC = arcpy.UpdateCursor(Output)
    for URow in UC:
        ANO = URow.getValue("ANO")
        if ANO in Secondary:
            URow.setValue(fields.crash.PrmANO['name'],FPair[ANO])
            D1 = DateDecompose(CDic[ANO       ]['Date'],CDic[ANO       ]['Time'])
            D2 = DateDecompose(CDic[FPair[ANO]]['Date'],CDic[FPair[ANO]]['Time'])
            t = abs(D1-D2).total_seconds()/60
            d = common.GetDistance(CDic[ANO]['P'],CDic[FPair[ANO]]['P'])
            URow.setValue(fields.crash.Tempor['name'],t)
            URow.setValue(fields.crash.Spatio['name'],d)
            URow.setValue(fields.loc.Label   ['name'],'{:3.0f}{}{:4.0f}{}'.format(t,' Min, ',d,' Feet'))
            UC.updateRow(URow)
        else:
            UC.deleteRow(URow)

    print(" --> Done.")
def FindSecondaryCrashes(CSV_In,ID_Col,Date_Col,RID_Col,MP_Col,CSV_Out):
    global i
    global j
    def GroupbyDifference_hours(DF,GroupbyField,TargetField,Difference):
        global i
        df = DF[[GroupbyField, TargetField]].copy(deep=True)
        df = df.sort_values([GroupbyField, TargetField])
        idx1 = DF.index
        idx2 = df.index
        Arr = idx2.get_indexer(idx1)    
        df.reset_index(drop=True,inplace=True)
        df.index = pd.MultiIndex.from_arrays([df[GroupbyField], df[TargetField]])
        df['Delta'] = df.groupby(GroupbyField)[TargetField].diff().fillna(timedelta(1)).apply(lambda x:x.days*24+x.seconds/3600.0).round(2)
        df['Date_Diff'] = df.Delta.apply(lambda x:(x<=Difference)*1 if not pd.isnull(x) else 0)
        df['Diff_Shifted'] = df.Date_Diff.shift(1).fillna(0).astype(int)
        
        def Block_D(DF):
            global i
            def Block_R(row):
                global i
                if row.Date_Diff==1: 
                    if row.Diff_Shifted==0:
                        i += 1
                        return(i)
                    else:
                        return(i)
                return(0)
            s = DF.apply(Block_R,axis=1)
            s.index = s.index.droplevel(0)
            return(s)
        df['block'] = df.groupby(GroupbyField).apply(Block_D)
        df['block_Shifted'] = df.groupby(GroupbyField).block.shift(-1).fillna(0).astype(int)
        def Block2_D(DF):
            s = DF.apply(lambda row:max(row.block_Shifted,row.block),axis=1)
            s.index = s.index.droplevel(0)
            return(s)
        s = df.groupby(GroupbyField).apply(Block2_D)
        s = s.iloc[Arr]
        s.index = DF.index
        return(s)
    def GroupbyDifference(DF,GroupbyField,TargetField,Difference):
        global j
        df = DF[[GroupbyField, TargetField]].copy(deep=True)
        df = df.sort_values([GroupbyField, TargetField])
        idx1 = DF.index
        idx2 = df.index
        Arr = idx2.get_indexer(idx1)    
        df.reset_index(drop=True,inplace=True)
        df.index = pd.MultiIndex.from_arrays([df[GroupbyField], df[TargetField]])
        df['Delta'] = df.groupby(GroupbyField)[TargetField].diff().fillna(Difference*2)
        df['Date_Diff'] = df.Delta.apply(lambda x:(x<=Difference)*1 if not pd.isnull(x) else 0)
        df['Diff_Shifted'] = df.Date_Diff.shift(1).fillna(0).astype(int)
        def Block_D(DF):
            global j
            def Block_R(row):
                global j
                if row.Date_Diff==1: 
                    if row.Diff_Shifted==0:
                        j += 1
                        return(j)
                    else:
                        return(j)
                return(0)
            s = DF.apply(Block_R,axis=1)
            s.index = s.index.droplevel(0)
            return(s)
        df['block'] = df.groupby(GroupbyField).apply(Block_D)
        df['block_Shifted'] = df.groupby(GroupbyField).block.shift(-1).fillna(0).astype(int)
        def Block2_D(DF):
            s = DF.apply(lambda row:max(row.block_Shifted,row.block),axis=1)
            s.index = s.index.droplevel(0)
            return(s)
        s = df.groupby(GroupbyField).apply(Block2_D)
        s = s.iloc[Arr]
        s.index = DF.index
        return(s)
    print('[{}] read and filter crash data'.format(strftime("%Y-%m-%d %H:%M:%S")))
    Crash_DF = pd.read_csv(CSV_In,low_memory=False)
    Crash_DF[Date_Col] = pd.to_datetime(Crash_DF[Date_Col])
    Crash_DF[MP_Col] = Crash_DF[MP_Col].round(4)
    Crash_DF = Crash_DF[(~pd.isnull(Crash_DF[RID_Col])) & (~pd.isnull(Crash_DF[MP_Col]))]
    print('[{}]  - {}'.format(strftime("%Y-%m-%d %H:%M:%S"),Crash_DF.shape))

    print('[{}] start iteration over {} rows'.format(strftime("%Y-%m-%d %H:%M:%S"),Crash_DF.shape[0]))
    for iteration in range(1,10):
        i = 0
        j = 0
        if iteration==1:
            Crash_DF['Time_Blocks'] = GroupbyDifference_hours(Crash_DF,'INVENTORY','DATE',2).astype(int)
        else:
            Crash_DF['Time_Blocks'] = GroupbyDifference_hours(Crash_DF[Crash_DF.MP_Blocks>0],'MP_Blocks','DATE',2).astype(int)
        Crash_DF['Time_Blocks'] = Crash_DF.Time_Blocks.fillna(0).astype(int)
        print('[{}]  - iteration: {}, time blocks: {}, crashes: {}'.format(strftime("%Y-%m-%d %H:%M:%S"),iteration,i,Crash_DF[Crash_DF.Time_Blocks>0].shape[0]))
            
        Crash_DF['MP_Blocks'] = GroupbyDifference(Crash_DF[Crash_DF.Time_Blocks>0],'Time_Blocks','MP',2)
        Crash_DF['MP_Blocks'] = Crash_DF.MP_Blocks.fillna(0).astype(int)
        print('[{}]  - iteration: {}, milepost blocks: {}, crashes: {}'.format(strftime("%Y-%m-%d %H:%M:%S"),iteration,j,Crash_DF[Crash_DF.MP_Blocks>0].shape[0]))
        if i==j:
            break
            print('[{}] converged, total blocks: {}, crashes: {}'.format(strftime("%Y-%m-%d %H:%M:%S"),iteration,j,Crash_DF[Crash_DF.MP_Blocks>0].shape[0]))

    print('[{}] adding primary/secondary fields'.format(strftime("%Y-%m-%d %H:%M:%S"),iteration,j,Crash_DF[Crash_DF.MP_Blocks>0].shape[0]))
    Crash_DF['CrashChain'] = Crash_DF.Time_Blocks
    Crash_DF = Crash_DF.drop(columns=['Time_Blocks','MP_Blocks'])

    Crash_DF = Crash_DF.sort_values(['CrashChain','DATE'])
    def IsSecondary(DF):
        s = pd.Series(index=DF.index,data='S',name='Sec')
        s.iloc[0]='P'
        return(s)
    S = Crash_DF[Crash_DF.CrashChain>0].groupby('CrashChain').CID.apply(IsSecondary)
    Crash_DF.loc[Crash_DF.CrashChain>0,'PrimSec'] = S

    def PrimaryCID(S):
        s = pd.Series(index=S.index,data=S.iloc[0],name='PrmCID')
        s.iloc[0]=0
        return(s)
    S = Crash_DF[Crash_DF.CrashChain>0].groupby('CrashChain')[ID_Col].apply(PrimaryCID)
    S2 = pd.Series(index=Crash_DF.index,data=0)
    S2.loc[S.index] = S
    Crash_DF['PrimCID'] = S2

    print('[{}] export results'.format(strftime("%Y-%m-%d %H:%M:%S")))
    Crash_DF.to_csv(CSV_Out,index=False)
    print('[{}] done!'.format(strftime("%Y-%m-%d %H:%M:%S")))
class CrashTypeExceedance(object):
    """
    Object class to compute the probability of specific crash types exceeding 
    a threshold proportion.
    
    Based on section 4.4.2.9 of the Highway Safety Manual with modifications.
    """
    
    def __init__(self, type_crashes, all_crashes, floor=2, bootstraps=5000):
        # Validate inputs
        type_crashes = np.asarray(type_crashes)
        all_crashes  = np.asarray(all_crashes)
        if not len(type_crashes) == len(all_crashes):
            raise ValueError()
        elif (type_crashes > all_crashes).sum() > 0:
            raise ValueError("Type crashes may not be larger than total crashes.")
        # Log crash information
        self._type_crashes  = type_crashes
        self._all_crashes   = all_crashes
        self._floor = floor
        self._num_bootstraps = bootstraps
    
    @property
    def type_crashes(self):
        return self._type_crashes
    
    @property
    def all_crashes(self):
        return self._all_crashes
    
    @property
    def floor(self):
        return self._floor
    
    @property
    def num_bootstraps(self):
        return self._num_bootstraps
    
    @property
    def size(self):
        return len(self.all_crashes)
    
    @property
    def proportions(self):
        nonzero = self.are_nonzero
        props = np.zeros(self.all_crashes.shape)
        props[nonzero] = self.type_crashes[nonzero] / self.all_crashes[nonzero]
        return props
    
    @property
    def threshold(self):
        return self.mean_proportion
        
    @property
    def sum_all(self):
        return self.all_crashes.sum()
    
    @property
    def sum_type(self):
        return self.type_crashes.sum()
    
    @property
    def are_valid(self):
        try:
            return self._are_valid
        except AttributeError:        
            self._are_valid = self.type_crashes >= self.floor
            return self._are_valid
        
    @property
    def are_nonzero(self):
        try:
            return self._are_nonzero
        except AttributeError:
            self._are_nonzero = self.all_crashes > 0
            return self._are_nonzero
    
    @property
    def variance(self):
        try:
            return self._variance
        except AttributeError:        
            # Identify valid locations based on floor value
            valid = self.are_valid

            # Compute variance
            self._variance = self.proportions[valid].var()
            return self._variance
    
    @property
    def stdev(self):
        return self.variance ** 0.5
    
    @property
    def mean_proportion(self):
        try:
            return self._mean_proportion
        except AttributeError:
            # Identify valid locations based on floor value
            valid = self.are_valid

            # Compute mean proportion
            self._mean_proportion = self.proportions[valid].mean() # self.type_crashes[valid].sum() / self.all_crashes[valid].sum()
            return self._mean_proportion
    
    @property
    def alpha(self):
        """
        The alpha parameter used in defining the beta cumulative distribution 
        function to test the exceedance of test locations.
        """
        try:
            return self._alpha
        except AttributeError:
            # Compute alpha parameter
            self._alpha = ((self.mean_proportion ** 2) - \
                           (self.mean_proportion ** 3) - \
                           (self.variance * self.mean_proportion)) / self.variance
            return self._alpha
        
    @property
    def beta(self):
        """
        The beta parameter used in defining the beta cumulative distribution 
        function to test the exceedance of test locations. If alpha + beta < 1, 
        a beta value of 1 - alpha will be enforced to ensure valid distribution 
        shape.
        """
        try:
            return self._beta
        except AttributeError:
            # Compute beta parameter
            beta = (self.alpha / self.mean_proportion) - self.alpha
            self._beta = max(beta, 1-self.alpha)
            return self._beta
        
    def probabilities(self, min_type=0):
        """
        Return an array of the probabilities that the mean of each location's proportion of
        crashes of the target type exceeds that of the given population.
        """
        ps = [self.betadist(self.type_crashes[i], self.all_crashes[i], min_type=min_type) \
              for i in range(self.size)]
        return np.asarray(ps)
    
    def betadist(self, type_crashes, all_crashes, min_type=0):
        """
        Test the input type and total crashes against the population mean proportion.
        """
        # Validate input
        if min_type < 0:
            raise TypeError(f"Minimum threshold input ({min_type}) is invalid. Must be a non-negative integer.")
        # If the provided crashes do not meet the minimum parameter, return a probability of zero
        if type_crashes < min_type:
            p = 0
        # If not enough locations are available to analyze, return a probability of zero
        elif self.are_valid.sum() < 2:
            p = 0
        # If variance is equal to zero, return a probability of zero
        elif self.variance == 0:
            p = 0
        # Else, calculate the probability using the beta distribution survival function
        else:
            p = stats.beta.sf(self.mean_proportion, self.alpha + type_crashes, self.beta + all_crashes - type_crashes)
            if np.isnan(p):
                raise ValueError("Invalid result!")
        return p