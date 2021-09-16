#HSMPY3
# Developed By Mahdi Rajabi mrajabi@clemson.edu
import os
import sys
import datetime
import json
# import arcpy
import pandas as pd
import numpy as np
# from scipy import optimize
# import re
from time import gmtime, strftime
# import matplotlib.pyplot as plt
from hsmpy31 import common
from hsmpy31 import gdb


def PolylineToDF(pl):
    import arcpy
    sr = pl.spatialReference
    df = pd.DataFrame(columns=['PartNumber','BMP','EMP','Mileage','Shape'])
    for i,prt in enumerate(pl):
        M = [pnt.M for pnt in prt]
        prt_shp = arcpy.Polyline(prt,sr,True,True)
        l = prt_shp.length/5280.0
        df.loc[i] = [i,M[0],M[-1],l,prt_shp]
    df['FirstP'] = df.Shape.apply(lambda x:x.firstPoint)
    df['lastP'] = df.Shape.apply(lambda x:x.lastPoint)
    df['FirstP_SHFT'] = df.FirstP.shift(-1)
    df['Dist_To_NextPart'] = np.NaN
    def GetDistance(P1,P2):
            X1 = P1.X
            X2 = P2.X
            Y1 = P1.Y
            Y2 = P2.Y
            return(((X1-X2)**2+(Y1-Y2)**2)**0.5)
    df['Dist_To_NextPart'] = df.iloc[:-1].apply(lambda row:GetDistance(row.lastP,row.FirstP_SHFT),axis=1)
    df['Dist_To_NextPart'] = df.Dist_To_NextPart.fillna(0)
    return(df[['PartNumber','BMP','EMP','Mileage','Shape','Dist_To_NextPart']])
def UnsplitParts(df,Tolerance,SpatRef):
    df.index = np.arange(0, len(df) )
    df['FP'] = [s[0] for s in df.Shape]
    df['LP'] = [s[-1] for s in df.Shape]
    df = df.sort_values('BMP')
    df['Spread'] = -1
    df['Continuous'] = 0
    for i,r in df.iterrows():
        if i==df.shape[0]-1:
            continue
        if r.EMP==df.loc[i+1]['BMP']:
            d = hsmpy3.common.Distance(x1=r.LP.X,y1=r.LP.Y,x2=df.loc[i+1]['FP'].X,y2=df.loc[i+1]['FP'].Y)
            df.set_value(i,'Spread',d)
            if d<Tolerance:
                df.set_value(i,'Continuous',1)
                #mp = arcpy.Multipoint(arcpy.Array([r.LP,df.loc[i+1]['FP']])).centeroid
    df['MergeID'] = -1
    m = 1
    for i,r in df.iterrows():
        df.set_value(i,'MergeID',m)
        if r.Continuous==0:
            m += 1
    mdf = pd.DataFrame(columns=['BMP','EMP','Shape'])
    ml = list(set(df.MergeID))
    ml.sort()
    for m in ml:
        rdf = df[df.MergeID==m]
        pntL = []
        rdf.index = range(1,rdf.shape[0]+1)
        for i,r in rdf.iterrows():
            if i==1 and rdf.shape[0]==1:
                pntL.extend([pnt for pnt in r.Shape])
            elif i==1 and rdf.shape[0]>1:
                pntL.extend([pnt for pnt in r.Shape][:-1])
            elif i == rdf.shape[0] and rdf.shape[0]>1:
                pntL.extend([pnt for pnt in r.Shape])
            else:
                pntL.extend([pnt for pnt in r.Shape][:-1])

        pl = arcpy.Polyline(arcpy.Array(pntL),SpatRef,False,True)
        M = [pnt.M for pnt in pntL]
        mdf.loc[m] = [M[0],M[-1],arcpy.Array(pntL)]
    a = arcpy.Array(list(mdf.Shape))
    return(arcpy.Polyline(a,SpatRef,True,True))
def MergeOverlappingParts(df):
    df.index = np.arange(0, len(df) )
    df['FP'] = [s.firstPoint for s in df.Shape]
    df['LP'] = [s.lastPoint for s in df.Shape]
    df = df.sort_values('BMP')
    df['Overlapping'] = -1
    for i,r in df.iterrows():
        if i==df.shape[0]-1:
            continue
        if r.EMP>df.loc[i+1]['BMP']:
            ip = r.Shape.intersect(df.loc[i+1]['Shape'],2)
            df.set_value(i,'Overlapping',ip.length)
    df['MergeID'] = -1
    m = 1
    for i,r in df.iterrows():
        df.set_value(i,'MergeID',m)
        if r.Overlapping<=0:
            m += 1
    mdf = pd.DataFrame(columns=['BMP','EMP','Shape'])
    ml = list(set(df.MergeID))
    ml.sort()
    for m in ml:
        rdf = df[df.MergeID==m]
        plL = list(rdf.Shape)
        upl = plL[0]
        for pl in plL[1:]:
            upl = upl.union(pl)
        #prt = UnionPrts([s[0] for s in rdf.Shape])
        prt = upl[0]
        M = [pnt.M for pnt in prt]
        mdf.loc[m] = [M[0],M[-1],prt]
    return(mdf)
def ParttoDF(prt):
    df = pd.DataFrame(columns=['ID','X','Y','Z','M'])
    for i,pnt in enumerate(prt):
        df.loc[i] = [i,pnt.X,pnt.Y,pnt.Z,pnt.M]
    return(df)
def CreateRoutes(Input,RID,BMP,EMP,SpatRef,Tolerance,Output):
    from hsmpy31 import common
    from hsmpy31 import gdb
    import arcpy
    print('[{}] {} to pandas:'.format(strftime("%Y-%m-%d %H:%M:%S"),os.path.basename(Input)))
    df  = gdb.FCtoDF_cursor(Input,readGeometry=True,selectedFields=[RID,BMP,EMP])
    df = df.rename(columns = {RID:'RID',BMP:'BMP',EMP:'EMP'})
    df['EMP'] = df.EMP.astype(float)
    df['BMP'] = df.BMP.astype(float)
    df['RID'] = df.RID.astype(str)
    print('[{}] - {}'.format(strftime("%Y-%m-%d %H:%M:%S"),df.shape[0]))

    print('[{}] droping duplicates:'.format(strftime("%Y-%m-%d %H:%M:%S")))
    s1 = df.shape[0]
    df = df.drop_duplicates(subset=['RID','BMP','EMP'])
    print('[{}] - {}'.format(strftime("%Y-%m-%d %H:%M:%S"),s1-df.shape[0]))

    print('[{}] droping zero length:'.format(strftime("%Y-%m-%d %H:%M:%S")))
    shape = arcpy.Describe(Input).shapeFieldName
    df['ShapeLength'] = [s.length for s in df[shape]]
    s1 = df.shape[0]
    df = df[df.ShapeLength>0]
    print('[{}] - {}'.format(strftime("%Y-%m-%d %H:%M:%S"),s1-df.shape[0]))

    print('[{}] droping overlaps:'.format(strftime("%Y-%m-%d %H:%M:%S")))
    s1 = df.shape[0]
    df['Diff'] = df.EMP - df.BMP
    df['FlagForRemove'] = 0
    df = df.sort_values(by=['RID','Diff'],ascending=[True,False])
    vdf = pd.DataFrame(df.RID.value_counts())
    vdf = vdf[vdf.RID>1]
    ridL = list(vdf.index)
    for rid in ridL:
        rdf = df[df.RID==rid]
        for j,k in rdf.iterrows():
            zdf = rdf[(rdf.BMP<=k.BMP) & (rdf.EMP>=k.EMP)]
            if zdf.shape[0]>1:
                df.set_value(j,'FlagForRemove',1)
    df = df[df.FlagForRemove==0]
    df = df[['RID','BMP','EMP','Shape']]
    df = df.sort_values(['RID','BMP'])
    print('[{}] - {}'.format(strftime("%Y-%m-%d %H:%M:%S"),s1-df.shape[0]))

    print('[{}] create feature class:'.format(strftime("%Y-%m-%d %H:%M:%S")))
    IC = common.CreateOutPath(os.path.dirname(Output) + '\\CR_ic',strftime("%Y%m%d_%H%M%S"),'')
    arcpy.management.CreateFeatureclass(out_name = os.path.basename(IC),
                                        out_path = os.path.dirname(IC),
                                        spatial_reference=SpatRef,
                                        geometry_type='Polyline',
                                        has_m='ENABLED',
                                        has_z='DISABLED')
    arcpy.AddField_management(IC,'RID','Text')
    arcpy.AddField_management(IC,'BMP','Double')
    arcpy.AddField_management(IC,'EMP','Double')
    ic = arcpy.InsertCursor(IC)
    for i,r in df.iterrows():
                Pl =  r.Shape
                row = ic.newRow()
                row.setValue('RID',int(i))
                row.setValue('BMP',float(r.BMP))
                row.setValue('EMP',float(r.EMP))
                row.shape = Pl
                ic.insertRow(row)
    del ic
    del row
    print('[{}] - {}'.format(strftime("%Y-%m-%d %H:%M:%S"),str(arcpy.management.GetCount(IC))))

    print('[{}] create routes:'.format(strftime("%Y-%m-%d %H:%M:%S")))
    R = hsmpy3.common.CreateOutPath(os.path.dirname(Output) + '\\CR_route',strftime("%Y%m%d_%H%M%S"),'')
    arcpy.lr.CreateRoutes(
        in_line_features    = IC,
        route_id_field      = 'RID',
        out_feature_class   = R,
        measure_source      = "TWO_FIELDS", 
        from_measure_field  = 'BMP', 
        to_measure_field    = 'EMP', 
        measure_factor      = "1", 
        measure_offset      = "0", 
        build_index         = "INDEX"
    )
    print('[{}] - {}'.format(strftime("%Y-%m-%d %H:%M:%S"),int(str(arcpy.management.GetCount(R)))))

    print('[{}] generalize'.format(strftime("%Y-%m-%d %H:%M:%S")))
    arcpy.Generalize_edit (in_features =R, tolerance = "0.001 Feet")

    print('[{}] multi to single part:'.format(strftime("%Y-%m-%d %H:%M:%S")))
    #RSP = hsmpy3.common.CreateOutPath(os.path.dirname(Output) + '\\CR_route_mp2sp',strftime("%Y%m%d_%H%M%S"),'')
    #arcpy.management.MultipartToSinglepart(in_features=R,out_feature_class=RSP)
    RSP = R
    print('[{}] - {}'.format(strftime("%Y-%m-%d %H:%M:%S"),str(arcpy.management.GetCount(RSP))))

    print('[{}] fc to pandas:'.format(strftime("%Y-%m-%d %H:%M:%S")))
    rdf = hsmpy3.gdb.FCtoDF_cursor(RSP,readGeometry=True,selectedFields=['RID'])
    rdf['RShape'] = rdf[arcpy.Describe(RSP).shapeFieldName]
    rdf['RID'] = rdf.RID.astype(int)
    rdf = rdf.sort_values('RID')
    rdf.index = list(rdf.RID)
    rdf = rdf[['RShape']]
    df = pd.concat([rdf,df[['RID']]],axis=1)
    df['BMP'] = [pl.firstPoint.M for pl in list(df.RShape)]
    df['EMP'] = [pl.lastPoint.M for pl in list(df.RShape)]
    df = df.sort_values(['RID','BMP','EMP'])
    df.columns = ['Shape','RID','BMP','EMP']
    print('[{}] - {}'.format(strftime("%Y-%m-%d %H:%M:%S"),rdf.shape[0]))

    print('[{}] unsplit & merge overlaps:'.format(strftime("%Y-%m-%d %H:%M:%S")))
    vdf = pd.DataFrame(df.RID.value_counts())
    RIDs = list(vdf[vdf.RID==1].index)
    RIDm = list(vdf[vdf.RID>1].index)
    fdf = pd.DataFrame(columns=['Shape'])
    sdf = df[df.RID.isin(RIDs)].copy(deep=True)
    sdf.index = sdf['RID']
    fdf['Shape'] = sdf.loc[RIDs,'Shape']
    fdf.index = RIDs
    for rid in RIDm:
        rdf = df[df.RID==rid].copy(deep=True)
        rdf = MergeOverlappingParts(rdf) 
        pl = UnsplitParts(rdf,Tolerance,SpatRef)
        fdf.set_value(rid,'Shape',pl) 
    fdf = fdf.sort_index()
    fdf['BMP'] = [s.firstPoint.M for s in fdf.Shape]
    fdf['EMP'] = [s.lastPoint.M for s in fdf.Shape]
    fdf['partCount'] = [s.partCount for s in fdf.Shape]
    fdf['ShapeLength'] = [s.length for s in fdf.Shape]
    print('[{}] - {}'.format(strftime("%Y-%m-%d %H:%M:%S"),fdf.shape[0]))

    print('[{}] create feature class:'.format(strftime("%Y-%m-%d %H:%M:%S")))
    try:
        arcpy.management.Delete(Output)
    except:pass
    arcpy.management.CreateFeatureclass(out_name = os.path.basename(Output),
                                        out_path = os.path.dirname(Output),
                                        spatial_reference=SpatRef,
                                        geometry_type='Polyline',
                                        has_m='ENABLED',
                                        has_z='DISABLED')
    arcpy.AddField_management(Output,RID,'Text')
    arcpy.AddField_management(Output,BMP,'Double')
    arcpy.AddField_management(Output,EMP,'Double')
    arcpy.AddField_management(Output,'partCount','Long')
    ic = arcpy.InsertCursor(Output)
    for i,r in fdf.iterrows():
                Pl =  r.Shape
                row = ic.newRow()
                row.setValue(RID,i)
                row.setValue(BMP,float(r.BMP))
                row.setValue(EMP,float(r.EMP))
                row.setValue('partCount',int(r.partCount))
                row.shape = Pl
                ic.insertRow(row)
    del ic
    del row
    print('[{}] - {}'.format(strftime("%Y-%m-%d %H:%M:%S"),str(arcpy.management.GetCount(Output))))
    arcpy.management.Delete(R)
    arcpy.management.Delete(IC)

    return(Output)
def CompassFromShapefile(RoutesFC,RID_Field,CSV_Out):
    from hsmpy31 import common
    import arcpy
    import math
    def degToCompass(num):
        if not pd.isnull(num):
            val=int((num/22.5)+.5)
            arr=["W","SW" ,"SW","SW" ,"S","SE" , "SE", "SE" ,"E","NE" ,"NE","NE" ,"N","NW" ,"NW","NW" ]
            return(arr[(val % 16)])
    def FindBMPnEMP(DF):
        df= pd.DataFrame({'BMP':DF.M.min(),'EMP':DF.M.max(),'Compass':DF.Compass.iloc[0]},index=[0])
        return(df)
    def ConvertPLtoCompass(row):
        pl = row.Shape
        pl_df = PolylineToDF(pl)
        L = []
        for i,r in pl_df.iterrows():
            prt_df = ParttoDF(r.Shape)
            prt_df['dx'] = prt_df.X.diff()
            prt_df['dy'] = prt_df.Y.diff()
            prt_df['Angle'] = prt_df.apply(lambda row:math.atan2(row.dy,row.dx)/math.pi*180+180,axis=1)
            prt_df['Compass'] = prt_df.Angle.apply(degToCompass)
            prt_df.loc[0,'Compass'] = prt_df.loc[1,'Compass']
            prt_df['Shifted'] = prt_df.Compass.shift(1).fillna(prt_df.Compass.iloc[0])
            prt_df['Blocks'] = prt_df.apply(lambda row:1 if row.Compass!= row.Shifted else 0,axis=1).cumsum()
            df = prt_df.groupby('Blocks').apply(FindBMPnEMP)
            df.index = df.index.droplevel(1)
            df = df[['BMP','EMP','Compass']]
            if df.shape[0]>1:
                df.BMP.iloc[1:] = df.EMP.shift(+1)
            L.append(df)
        Compass_DF = pd.concat(L)
        return(Compass_DF)
    A = []
    print('[{}] read route data'.format(strftime("%Y-%m-%d %H:%M:%S")))
    R_DF = gdb.FCtoDF_cursor(RoutesFC,selectedFields=[RID_Field],readGeometry=True)
    print('[{}]  - {}'.format(strftime("%Y-%m-%d %H:%M:%S"),R_DF.shape[0]))

    print('[{}] iterating through routes ...'.format(strftime("%Y-%m-%d %H:%M:%S")))
    for i,r in R_DF.iterrows():
        df = ConvertPLtoCompass(r)
        df['INVENTORY'] = r[RID_Field]
        A.append(df)
    df = pd.concat(A)
    print('[{}]  - {}'.format(strftime("%Y-%m-%d %H:%M:%S"),df.shape[0]))
    print('[{}] exporting the results'.format(strftime("%Y-%m-%d %H:%M:%S")))
    df = df[['INVENTORY','BMP','EMP','Compass']]
    df.reset_index(drop=True,inplace=True)
    df.to_csv(CSV_Out,index=False)
    print('[{}] done!'.format(strftime("%Y-%m-%d %H:%M:%S")))
def DissolveDF(DF,RID,BMP,EMP,STIM,ETIM,DissFields):
    print('[{}] dissolving by milepost: {}'.format(strftime("%Y-%m-%d %H:%M:%S"),DissFields))
    print('[{}] sorting and indexing:'.format(strftime("%Y-%m-%d %H:%M:%S")))
    F1 = [RID,STIM,ETIM,BMP,EMP]
    F2 = DissFields
    df = DF[F1+F2]
    df.index = range(0,df.shape[0])
    df = df.sort_values(by=[RID,STIM])
    df = df.fillna(-1)
    print('[{}]  - {}:'.format(strftime("%Y-%m-%d %H:%M:%S"),df.shape))
    print('[{}] grouping intervals:'.format(strftime("%Y-%m-%d %H:%M:%S")))
    idx = df.groupby([RID,STIM,ETIM] + F2)[BMP].shift(-1) != df[EMP]
    df['EMP2'] = df.loc[idx, EMP]
    df['EMP2'] = df.groupby([RID,STIM,ETIM] + F2)['EMP2'].fillna(method='backfill')
    df['EMP2'] = df['EMP2'].fillna(df[EMP]) 
    print('[{}] aggregating groups:'.format(strftime("%Y-%m-%d %H:%M:%S")))
    sdf = df.groupby([RID,STIM,ETIM] + F2 + ['EMP2'], as_index=False).agg({BMP: 'first', EMP: 'last'}).drop(['EMP2'], axis=1)
    print('[{}]  - {}:'.format(strftime("%Y-%m-%d %H:%M:%S"),sdf.shape))

    print('[{}] dissolving by time:'.format(strftime("%Y-%m-%d %H:%M:%S")))
    print('[{}] sorting and indexing:'.format(strftime("%Y-%m-%d %H:%M:%S")))
    df = sdf
    df.index = range(0,df.shape[0])
    df = df.sort_values(by=[RID,BMP])
    print('[{}] grouping intervals:'.format(strftime("%Y-%m-%d %H:%M:%S")))
    idx = df.groupby([RID,BMP,EMP] + F2)[STIM].shift(-1) != df[ETIM]
    df['EMP2'] = df.loc[idx, ETIM]
    df['EMP2'] = df.groupby([RID,BMP,EMP] + F2)['EMP2'].fillna(method='backfill')
    df['EMP2'] = df['EMP2'].fillna(df[ETIM]) 
    print('[{}] aggregating groups:'.format(strftime("%Y-%m-%d %H:%M:%S")))
    sdf = df.groupby([RID,BMP,EMP] + F2 + ['EMP2'], as_index=False).agg({STIM: 'first', ETIM: 'last'}).drop(['EMP2'], axis=1)
    print('[{}]  - {}:'.format(strftime("%Y-%m-%d %H:%M:%S"),sdf.shape))
    print('[{}] sorting the output:'.format(strftime("%Y-%m-%d %H:%M:%S")))
    sdf = sdf.replace(-1,np.NaN)
    sdf = sdf[F1+F2].sort_values(F1)
    print('[{}] done!'.format(strftime("%Y-%m-%d %H:%M:%S")))
    return(sdf)
def DissolveDF_MP(DF,DissFields,RID='RID',YEAR='YEAR',BMP='BMP',EMP='EMP'):
    print('[{}] dissolving by milepost: {}'.format(strftime("%Y-%m-%d %H:%M:%S"),DissFields))
    print('[{}] sorting and indexing:'.format(strftime("%Y-%m-%d %H:%M:%S")))
    F1 = [RID,YEAR,BMP,EMP]
    F2 = DissFields
    df = DF[F1+F2]
    df.index = range(0,df.shape[0])
    df = df.sort_values(by=[RID,YEAR])
    try:
        df = df.fillna(-1)
    except:
        pass
    print('[{}]  - {}:'.format(strftime("%Y-%m-%d %H:%M:%S"),df.shape))
    print('[{}] grouping intervals:'.format(strftime("%Y-%m-%d %H:%M:%S")))
    idx = df.groupby([RID,YEAR] + F2)[BMP].shift(-1) != df[EMP]
    df['EMP2'] = df.loc[idx, EMP]
    df['EMP2'] = df.groupby([RID,YEAR] + F2)['EMP2'].fillna(method='backfill')
    df['EMP2'] = df['EMP2'].fillna(df[EMP]) 
    print('[{}] aggregating groups:'.format(strftime("%Y-%m-%d %H:%M:%S")))
    sdf = df.groupby([RID,YEAR] + F2 + ['EMP2'], as_index=False).agg({BMP: 'first', EMP: 'last'}).drop(['EMP2'], axis=1)
    print('[{}]  - {}:'.format(strftime("%Y-%m-%d %H:%M:%S"),sdf.shape))

    idx2 = pd.Index(sdf[RID])
    idx1 = pd.Index(sdf[YEAR])
    idx3 = pd.IntervalIndex.from_tuples([(stim,etim) for stim,etim in zip(sdf[BMP],sdf[EMP])],closed='left')
    sdf.index = pd.MultiIndex.from_arrays([idx2,idx1,idx3],names=['RID','Time','MP'])
    sdf= sdf.sort_index()
    print('[{}] done!'.format(strftime("%Y-%m-%d %H:%M:%S")))
    return(sdf)



def OverlapLinearEvents(Source_DF,Target_DF,AttList):
    # importing attributes from two line event tables without resegmentation
    #print('[{}] add IRIS attributes to intersection approaches'.format(strftime("%Y-%m-%d %H:%M:%S")))
    def FindOverlaps(app_bmp,app_emp,sids):
        app_len = app_emp-app_bmp
        def getOverlap(a, b):
            return(max(0, min(a[1], b[1]) - max(a[0], b[0])))
        l = []
        for sid in sids:
            bmp,emp = [float(i) for i in sid.split('_')[-2:]]
            l.append(getOverlap([app_bmp,app_emp],[bmp,emp])*5280.0)
        return(l)

    Source_DF['SID'] = Source_DF.RID + '_' + Source_DF.YEAR.astype(int).astype(str) + '_' + Source_DF.BMP.round(3).astype(str) + '_' + Source_DF.EMP.round(3).astype(str)
    Source_DF.drop_duplicates('SID',inplace=True)
    Target_DF.index = pd.MultiIndex.from_arrays([Target_DF.RID,Target_DF.YEAR])
    Target_DF['SIDs'] = Source_DF.groupby(['RID','YEAR']).SID.agg(list).apply(lambda d: d if isinstance(d, list) else [])
    Target_DF['SIDs'] = Target_DF.SIDs.apply(lambda d: d if isinstance(d, list) else [])
    Target_DF['Overlaps'] = Target_DF.apply(lambda row:FindOverlaps(row.BMP,row.EMP,row.SIDs),axis=1)
    Target_DF.reset_index(drop=True,inplace=True)
    Target_DF['SID'] = Target_DF[Target_DF.Overlaps.apply(len)>0].apply(lambda row:row.SIDs[row.Overlaps.index(max(row.Overlaps))],axis=1)
    Target_DF.index = Target_DF.SID
    Source_DF.index = Source_DF.SID
    for c in AttList:
        print(c)
        Target_DF[c] = Source_DF[c]
    Target_DF['Coverage'] = (Target_DF.Overlaps.apply(sum)/5280.0/(Target_DF.EMP-Target_DF.BMP))
    Target_DF.drop(columns=['SIDs','Overlaps','SID'],inplace=True)
    Target_DF.reset_index(drop=True,inplace=True)
    return(Target_DF)

def OverlayRouteEvents(DF_List,Field_List,GDB,Type='UNION'):
    import arcpy
    ts = strftime("%Y%m%d_%H%M%S")

    table1 = GDB + '\\DF1_'+ts
    if arcpy.Exists(table1):
        arcpy.Delete_management(table1)
    gdb.DFtoTable_numpy(DF_List[0],table1)

    table2 = GDB + '\\DF2_'+ts
    if arcpy.Exists(table2):
        arcpy.Delete_management(table2)
    gdb.DFtoTable_numpy(DF_List[1],table2)

    outTable = GDB + '\\overlay_'+ts
    if arcpy.Exists(outTable):
        arcpy.Delete_management(outTable)
    arcpy.lr.OverlayRouteEvents(
        table1, 
        ' '.join([Field_List[0].split(' ')[0]] + ['LINE'] + Field_List[0].split(' ')[1:]), 
        table2, 
        ' '.join([Field_List[1].split(' ')[0]] + ['LINE'] + Field_List[1].split(' ')[1:]), 
        Type, 
        outTable, 
        "RID Line BMP EMP", "NO_ZERO", "FIELDS", "INDEX")
    arcpy.Delete_management(table1)
    arcpy.Delete_management(table2)

    for df,fl in zip(DF_List[2:],Field_List[2:]):
        ts = strftime("%Y%m%d_%H%M%S")
        table_i = GDB + '\\DF_i_'+ts
        if arcpy.Exists(table_i):
            arcpy.Delete_management(table_i)
        gdb.DFtoTable_numpy(df,table_i)

        outTable_i = GDB + '\\overlay_i_'+ts
        if arcpy.Exists(outTable_i):
            arcpy.Delete_management(outTable_i)
        arcpy.lr.OverlayRouteEvents(
            outTable, 
            "RID Line BMP EMP", 
            table_i, 
            ' '.join([fl.split(' ')[0]] + ['LINE'] + fl.split(' ')[1:]), 
            Type, 
            outTable_i, 
            "RID Line BMP EMP", "NO_ZERO", "FIELDS", "INDEX")

        arcpy.Delete_management(table_i)
        arcpy.Delete_management(outTable)
        outTable = outTable_i
    Out_DF = gdb.FCtoDF_numpy(outTable)
    arcpy.Delete_management(outTable)
    return(Out_DF)

def GetCIDs(roads,crashes):
    p = r'\\Chcfpp01\Groups\HTS\Code_Repository\Python\Libraries'
    if not p in sys.path:
        sys.path.append(p)
    from htspy.linref.events import NetworkEventsCollection
    nec = NetworkEventsCollection(crashes, rid_col='RID', year_col='YEAR', beg_col='MP',end_col='MP')
    def GetCID(r):
        try:
            df = nec.intersecting(rid=r.RID, year=r.YEAR, beg=r.BMP, end=r.EMP)
            return(df.CID.values)
        except:
            return([])
    return(roads.apply(GetCID,axis=1))
def GetRoadAtt(roads,crashes,fields):
    import pandas as pd
    from htspy.linref.events import NetworkEventsCollection
    nec = NetworkEventsCollection(roads, rid_col='RID', year_col='YEAR', beg_col='BMP',end_col='EMP')
    def GetCID(r):
        try:
            df = nec.intersecting(rid=r.RID, year=r.YEAR, beg=r.MP, end=r.MP+.001)
            return(pd.Series({c:df[c].values[0] for c in fields}))
        except:
            return(pd.Series({c:np.NaN for c in fields}))
    return(crashes.apply(GetCID,axis=1))

def SimplifyByCurves(Input_FC,max_offset=12,min_radius=80,max_radius=10000,min_arc_angle=1,fit_to_segment=False,max_arc_angle_step=50,anchor_points=''):

    import arcpy
    import math
    def define_circle(p1, p2, p3):
            """
            Returns the center and radius of the circle passing the given 3 points.
            In case the 3 points form a line, returns (None, infinity).
            """
            temp = p2[0] * p2[0] + p2[1] * p2[1]
            bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
            cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
            det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])

            if abs(det) < 1.0e-6:
                return (None, np.inf)

            # Center of circle
            cx = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
            cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det

            radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
            radius = radius * FindDirection(p1,p2,p3)
            return ((cx, cy), radius)
    def polar(x, y):
                """returns r, theta(degrees)
                """
                #r = (x ** 2 + y ** 2) ** .5
                if y == 0:
                    theta = 180 if x < 0 else 0
                elif x == 0:
                    theta = 90 if y > 0 else 270
                else:
                    theta = math.degrees(math.atan(float(y) / x))
                    if x<0 and y<0:
                        theta = theta+180.0
                    if y<0 and x>0:
                        theta = 360.0 + theta
                    if x<0 and y>0:
                        theta = 180.0 + theta
                return theta 
    def FindDirection(p1,p2,p3):
            theta1 = polar(p2[0]-p1[0],p2[1]-p1[1])
            theta2 = polar(p3[0]-p2[0],p3[1]-p2[1])
            theta = theta2-theta1
            if theta>180:
                theta = theta-360
            return(1 if theta>0 else -1)
    def SplitatCurves2(IRIS_DF):
        Master_List = []
        INV_LIST = list(IRIS_DF.index)
        pl = IRIS_DF.Shape.iloc[0]
        try:
            pl[0][0].Z
            m_index = 3
        except:
            m_index = 2
        for i in INV_LIST:
            pl = IRIS_DF.Shape.loc[i]
            js = json.loads(pl.JSON)
            inv = IRIS_DF.RID.loc[i]
            if 'curvePaths'in js:
                parts = js['curvePaths']
            else:
                parts = js['paths']
            L = []
            for pi, part in enumerate(parts):
                for seg in part:
                    if type(seg)==list:
                        L.append(pd.Series({'Part':pi,'X':seg[0],'Y':seg[1],'M':seg[m_index],'Type':'tangent'}))
                    if type(seg)==dict:
                        if 'c' in seg:
                            p = seg['c'][0]
                            mp = seg['c'][1]
                            L.append(pd.Series({'Part':pi,'X':p[0],'Y':p[1],'M':p[m_index],'Type':'curve','Mid_X':mp[0],'Mid_Y':mp[1]}))
                        if 'a' in seg:
                            p = seg['a'][0]
                            cp = seg['a'][1]
                            L.append(pd.Series({'Part':pi,'X':p[0],'Y':p[1],'M':p[m_index],'Type':'curve','Cen_X':mp[0],'Cen_Y':mp[1]}))
            df = pd.DataFrame(L)
            cdf = pd.DataFrame()
            cdf['BMP'] = df.M.iloc[:-1]
            cdf['EMP'] = df.M.shift(-1)
            cdf['Type'] = df.Type.shift(-1)
            cdf['B_X'] = df.X.iloc[:-1]
            cdf['B_Y'] = df.Y.iloc[:-1]
            cdf['E_X'] = df.X.shift(-1)
            cdf['E_Y'] = df.Y.shift(-1)
            if 'curvePaths'in js and 'Mid_X' in df.columns:
                cdf['M_X'] = df.Mid_X.shift(-1)
                cdf['M_Y'] = df.Mid_Y.shift(-1)
                cdf['Circle'] = cdf[cdf.Type=='curve'].apply(lambda S:define_circle([S.B_X,S.B_Y],[S.M_X,S.M_Y],[S.E_X,S.E_Y]),axis=1)
                cdf['Radius'] = cdf[cdf.Type=='curve']['Circle'].apply(lambda x:x[1])
                cdf['C_X'] = cdf[cdf.Type=='curve']['Circle'].apply(lambda x:x[0][0])
                cdf['C_Y'] = cdf[cdf.Type=='curve']['Circle'].apply(lambda x:x[0][1])
                cdf.drop(columns ='Circle',inplace=True)
            cdf["RID"] = inv

            Master_List.append(cdf)
        CDF = pd.concat(Master_List) 
        
        return(CDF)    

    from hsmpy31 import gdb
    t_print = lambda x:print('\x1b[31m[{}]\x1b[39m {}'.format(strftime("%Y-%m-%d %H:%M:%S"),x))
    t_print('creating simplified version:')
    Params = [max_offset,min_radius,max_radius,min_arc_angle]
    R_FC = Input_FC

    R1_S = R_FC + strftime("%Y%m%d%H%M%S")
    R1 = R1_S + '_Simplified' 
    if arcpy.Exists(R1):
        arcpy.management.Delete(R1)
    arcpy.management.CopyFeatures(R_FC,R1)
    arcpy.edit.SimplifyByStraightLinesAndCircularArcs(
        in_features=R1, 
        max_offset="{} Feet".format(Params[0]), 
        fitting_type={True:"FIT_TO_SEGMENTS",False:'FIT_TO_VERTICES'}[fit_to_segment], 
        circular_arcs="CREATE", 
        max_arc_angle_step=max_arc_angle_step, 
        min_vertex_count=4, 
        min_radius="{} Feet".format(Params[1]), 
        max_radius='{} Feet'.format(Params[2]),  
        min_arc_angle=Params[3], 
        closed_ends="PRESERVE",
        anchor_points = anchor_points
    )
    DF1 = gdb.FCtoDF_numpy(R1,readGeometry=True)
    t_print(DF1.shape)

    t_print('split at curves:')
    CDF = SplitatCurves2(DF1)
    CDF['Length'] = CDF.EMP - CDF.BMP
    m = CDF.Type=='curve'
    CDF.loc[m,'Curve_Length'] = (CDF.loc[m,'EMP'] - CDF.loc[m,'BMP'])*5280.0
    s = ['RID', 'BMP', 'EMP', 'Type', 'Radius','Curve_Length', 'C_X', 'C_Y']
    for c in [i for i in s if not i in CDF.columns]:
        CDF[c] = np.NaN
    CDF = CDF[s]
    CDF = CDF.sort_values(['RID','BMP'])
    t_print('done!')
    return(CDF)
def SimplifyByCurves_DF(Input_DF,GDB,max_offset=12,min_radius=80,max_radius=10000,min_arc_angle=1,fit_to_segment=False,max_arc_angle_step=50,anchor_points=''):

    import arcpy
    import math
    def define_circle(p1, p2, p3):
            """
            Returns the center and radius of the circle passing the given 3 points.
            In case the 3 points form a line, returns (None, infinity).
            """
            temp = p2[0] * p2[0] + p2[1] * p2[1]
            bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
            cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
            det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])

            if abs(det) < 1.0e-6:
                return (None, np.inf)

            # Center of circle
            cx = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
            cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det

            radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
            radius = radius * FindDirection(p1,p2,p3)
            return ((cx, cy), radius)
    def polar(x, y):
                """returns r, theta(degrees)
                """
                #r = (x ** 2 + y ** 2) ** .5
                if y == 0:
                    theta = 180 if x < 0 else 0
                elif x == 0:
                    theta = 90 if y > 0 else 270
                else:
                    theta = math.degrees(math.atan(float(y) / x))
                    if x<0 and y<0:
                        theta = theta+180.0
                    if y<0 and x>0:
                        theta = 360.0 + theta
                    if x<0 and y>0:
                        theta = 180.0 + theta
                return theta 
    def FindDirection(p1,p2,p3):
            theta1 = polar(p2[0]-p1[0],p2[1]-p1[1])
            theta2 = polar(p3[0]-p2[0],p3[1]-p2[1])
            theta = theta2-theta1
            if theta>180:
                theta = theta-360
            return(1 if theta>0 else -1)
    def SplitatCurves2(IRIS_DF):
        Master_List = []
        INV_LIST = list(IRIS_DF.index)
        pl = IRIS_DF.Shape.iloc[0]
        try:
            pl[0][0].Z
            m_index = 3
        except:
            m_index = 2
        for i in INV_LIST:
            pl = IRIS_DF.Shape.loc[i]
            js = json.loads(pl.JSON)
            inv = IRIS_DF.RID.loc[i]
            if 'curvePaths'in js:
                parts = js['curvePaths']
            else:
                parts = js['paths']
            L = []
            for pi, part in enumerate(parts):
                for seg in part:
                    if type(seg)==list:
                        L.append(pd.Series({'Part':pi,'X':seg[0],'Y':seg[1],'M':seg[m_index],'Type':'tangent'}))
                    if type(seg)==dict:
                        if 'c' in seg:
                            p = seg['c'][0]
                            mp = seg['c'][1]
                            L.append(pd.Series({'Part':pi,'X':p[0],'Y':p[1],'M':p[m_index],'Type':'curve','Mid_X':mp[0],'Mid_Y':mp[1]}))
                        if 'a' in seg:
                            p = seg['a'][0]
                            cp = seg['a'][1]
                            L.append(pd.Series({'Part':pi,'X':p[0],'Y':p[1],'M':p[m_index],'Type':'curve','Cen_X':mp[0],'Cen_Y':mp[1]}))
            df = pd.DataFrame(L)
            cdf = pd.DataFrame()
            cdf['BMP'] = df.M.iloc[:-1]
            cdf['EMP'] = df.M.shift(-1)
            cdf['Type'] = df.Type.shift(-1)
            cdf['B_X'] = df.X.iloc[:-1]
            cdf['B_Y'] = df.Y.iloc[:-1]
            cdf['E_X'] = df.X.shift(-1)
            cdf['E_Y'] = df.Y.shift(-1)
            if 'curvePaths'in js and 'Mid_X' in df.columns:
                cdf['M_X'] = df.Mid_X.shift(-1)
                cdf['M_Y'] = df.Mid_Y.shift(-1)
                cdf['Circle'] = cdf[cdf.Type=='curve'].apply(lambda S:define_circle([S.B_X,S.B_Y],[S.M_X,S.M_Y],[S.E_X,S.E_Y]),axis=1)
                cdf['Radius'] = cdf[cdf.Type=='curve']['Circle'].apply(lambda x:x[1])
                cdf['C_X'] = cdf[cdf.Type=='curve']['Circle'].apply(lambda x:x[0][0])
                cdf['C_Y'] = cdf[cdf.Type=='curve']['Circle'].apply(lambda x:x[0][1])
                cdf.drop(columns ='Circle',inplace=True)
            cdf["RID"] = inv

            Master_List.append(cdf)
        CDF = pd.concat(Master_List) 
        
        return(CDF)    

    from hsmpy31 import gdb
    t_print = lambda x:print('\x1b[31m[{}]\x1b[39m {}'.format(strftime("%Y-%m-%d %H:%M:%S"),x))
    t_print('creating simplified version:')
    Params = [max_offset,min_radius,max_radius,min_arc_angle]


    ts = strftime("%Y%m%d%H%M%S")
    R1 = GDB + '\\t_curve_DF_' + ts
    if arcpy.Exists(R1):
        arcpy.management.Delete(R1)
    gdb.DFtoFCorTable_cursor(Input_DF,R1,shape='Shape')

    arcpy.edit.SimplifyByStraightLinesAndCircularArcs(
        in_features=R1, 
        max_offset="{} Feet".format(Params[0]), 
        fitting_type={True:"FIT_TO_SEGMENTS",False:'FIT_TO_VERTICES'}[fit_to_segment], 
        circular_arcs="CREATE", 
        max_arc_angle_step=max_arc_angle_step, 
        min_vertex_count=4, 
        min_radius="{} Feet".format(Params[1]), 
        max_radius='{} Feet'.format(Params[2]),  
        min_arc_angle=Params[3], 
        closed_ends="PRESERVE",
        anchor_points = anchor_points
    )
    DF1 = gdb.FCtoDF_numpy(R1,readGeometry=True)
    t_print(DF1.shape)

    t_print('split at curves:')
    CDF = SplitatCurves2(DF1)
    CDF['Length'] = CDF.EMP - CDF.BMP
    m = CDF.Type=='curve'
    CDF.loc[m,'Curve_Length'] = (CDF.loc[m,'EMP'] - CDF.loc[m,'BMP'])*5280.0
    s = ['RID', 'BMP', 'EMP', 'Type', 'Radius','Curve_Length', 'C_X', 'C_Y']
    for c in [i for i in s if not i in CDF.columns]:
        CDF[c] = np.NaN
    
    CDF = CDF[s]
    CDF = CDF.sort_values(['RID','BMP'])
    t_print('done!')
    return(CDF)
