import arcpy
import sys
import math
import urllib
import zipfile
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ftplib import FTP
from urllib.parse import urlparse
import subprocess
import time
import shutil
from time import gmtime, strftime
#from lxml import html
import requests
from urllib.request import urlretrieve
from matplotlib import cbook
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap

NAD1983IL = arcpy.SpatialReference(102672)
NAD1983SC = arcpy.SpatialReference(102733)
NAD1983NC = arcpy.SpatialReference(2264)
WGS1984   = arcpy.SpatialReference(4326)
# Find the route between two points, local HSIP projects
FRTypes= ['U4F', 'U6F', 'R4F']
RTypes = ['R2U', 'R4U', 'R4D', 'U2U', 'U3T', 'U4D', 'U4U', 'U5T']
ITypes = ['R3ST' ,'R4ST' , 'R4SG', 'RM3ST', 'RM4ST', 'RM4SG', 'U3ST', 'U4ST', 'U3SG', 'U4SG']
FTypes = RTypes + ITypes + FRTypes

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

def Downloadfile(URL,OutputDir,extension,filterForSC=False):
    name = os.path.join(OutputDir, URL.split('/')[-1])
    urlp = urlparse(URL)
    print('[{}] Downloading: {}'.format(strftime("%Y-%m-%d %H:%M:%S"),URL))
    try:
        if urlp.scheme == 'ftp':
            ftp = FTP(urlp.hostname)
            ftp.login()
            ftp.retrbinary('RETR '+urlp.path, open(name, 'wb').write)
        if urlp.scheme == 'http':
            
            urlretrieve(URL, name)
    except IOError as e:
        print("Can't retrieve {} to{}: {}".format(URL, OutputDir, e))
        return
    try:
        z = zipfile.ZipFile(name)
    except zipfile.error as e:
        #print("Bad zipfile (from %r): %s" % (URL, e))
        return
    shapefile = []
    #print('Extracting:')
    for n in z.namelist():
        #print(n)
        if n.split('.')[-1] == extension:
            shapefile.append(n)
    z.extractall(OutputDir)
    output = [os.path.join(OutputDir,shp) for shp in shapefile]
    if filterForSC:
        print('Filtering for STATEFP = 45')
        filter_output = os.path.splitext(output)[0] + '_Filtered' + os.path.splitext(output)[1]
        arcpy.Delete_management(filter_output)
        arcpy.Select_analysis(in_features=output,
                              out_feature_class=filter_output,
                              where_clause=""""STATEFP"='45'""")
        output = filter_output
        #sum = PrintSummary(output)
    #else:
        #sum = PrintSummary(output)
    print('\n'.join(output))
    return(output)
def GetFID(Row):
    FID = ''
    try:
        FID = Row.getValue('FID')
    except:
        try:
            FID = Row.getValue('OBJECTID')
        except:
            print("FID or OBJECTID not Found")
    return FID
def GetVal(Row, Field, Default=0, AddWarning=False):
    try:
        Val = Row.getValue(Field)
        #return(Val)
        if not Val is None:
            return Val
        else:
        #    if AddWarning: print('Failed to read: ' + Field + ', Default value Assigned')
            return Default
    except:
        if AddWarning: print('Failed to read: ' + Field + ', Default value Assigned')
        return Default
def GetIntVal(Row, Field, Default=0, AddWarning=False):
        try:
            Val = int(Row.getValue(Field))
            if not Val is None:
                return Val
            else:
                if AddWarning: print('Failed to read: ' + Field + ', Default value Assigned')
                return Default
        except:
            if AddWarning: print('Failed to read: ' + Field + ', Default value Assigned')
            return Default
def GetFloatVal(Row, Field, Default=0.0, AddWarning=False):
        try:
            Val = float(Row.getValue(Field))
            if not Val is None:
                return Val
            else:
                if AddWarning: print('Failed to read: ' + Field + ', Default value Assigned')
                return Default
        except:
            if AddWarning: print('Failed to read: ' + Field + ', Default value Assigned')
            return Default
def MaximumValue(Layer,FieldName):
    SC = arcpy.SearchCursor(Layer)
    SRow = SC.next()
    if SRow:
        Maximum = GetVal(SRow,FieldName)
    else:
        Maximum = None
    for SRow in SC:
        Val = GetVal(SRow,FieldName)
        if Val > Maximum:
            Maximum = Val
    return Maximum
def GetANO(Row, IntErr=-1, RowErr=99999999):
        if Row:
            ANO = Row.getValue('ANO')
            try:
                ANO = int(ANO)
            except:
                ANO = IntErr
        else:
            ANO = RowErr
        if int(str(ANO)[0:4]) in [2001,2005,2006]:
            ANO = int(str(ANO)[2:])
        return ANO
def ConvertType(Value, Type):
        if   Type in ['TEXT']:
            try:
                fval = str(Value)
            except:
                fval = None
        elif Type in ['SHORT', 'LONG']:
            try:
                fval = int(Value)
            except:
                fval = None
        elif Type in ['DOUBLE']:
            try:
                fval = float(Value)
            except:
                fval = None
        return fval
def SOEExtract(SOE):
        Flag = False
        if not SOE:
            Flag = True
        if type(SOE) != str:
            SOE = str(SOE)
        try:
            a = int(SOE)
        except:
            Flag = True
        SOE1 = 0
        SOE2 = 0
        SOE3 = 0
        SOE4 = 0
        if not Flag:
            n = len(SOE)
            if   n in [1,2]:
                SOE1 = int(SOE)
            elif n == 3:
                c1 = int(SOE[0])
                c23 = int(SOE[1:3])
                SOE1 = c1
                SOE2 = c23
            elif n == 4:
                c12 = int(SOE[0:2])
                c34 = int(SOE[2:4])
                SOE1 = c12
                SOE2 = c34
            elif n == 5:
                c1 = int(SOE[0])
                c23 = int(SOE[1:3])
                c45 = int(SOE[3:5])
                SOE1 = c1
                SOE2 = c23
                SOE3 = c45
            elif n == 6:
                c12 = int(SOE[0:2])
                c34 = int(SOE[2:4])
                c56 = int(SOE[4:6])
                SOE1 = c12
                SOE2 = c34
                SOE3 = c56
            elif n == 7:
                c1 = int(SOE[0])
                c23 = int(SOE[1:3])
                c45 = int(SOE[3:5])
                c67 = int(SOE[5:7])
                SOE1 = c1
                SOE2 = c23
                SOE3 = c45
                SOE4 = c67
            elif n == 8:
                c12 = int(SOE[0:2])
                c34 = int(SOE[2:4])
                c56 = int(SOE[4:6])
                c78 = int(SOE[6:8])
                SOE1 = c12
                SOE2 = c34
                SOE3 = c56
                SOE4 = c78
        return [SOE1,SOE2,SOE3,SOE4]
def GetDistance(P1,P2):
        X1 = P1.firstPoint.X
        X2 = P2.firstPoint.X
        Y1 = P1.firstPoint.Y
        Y2 = P2.firstPoint.Y
        return(((X1-X2)**2+(Y1-Y2)**2)**0.5)
def NaturalLog(A):
        
        if A > 0:
            return math.log1p(A - 1)
        else:
            return 0
def PrintSummary(Input,extension='shp'):
    desc = arcpy.Describe(Input)
    ext = extension
    out = {'ShapeType':'','Rows':'','Columns':''}
    if hasattr(desc,'shapeType'):
        if ext == 'shp':
            print('Type: ' + desc.shapeType)
            FieldObjList = arcpy.ListFields(Input)
            FieldNameList = [Field.name for Field in arcpy.ListFields(Input)]
            FieldNameList.sort()
            TotalSites = int(str(arcpy.GetCount_management(Input)))
            print("Columns: " + str(len(FieldNameList)) + " x Rows: " + str(TotalSites))
            print(FieldNameList)
            out = {'ShapeType':desc.shapeType,'Rows':TotalSites,'Columns':FieldNameList}
        if ext == 'rrd':
            print('Type: ' + desc.format)
            out['Type']=desc.format
    else:
            print('Type: Table')
            FieldObjList = arcpy.ListFields(Input)
            FieldNameList = [Field.name for Field in arcpy.ListFields(Input)]
            FieldNameList.sort()
            TotalSites = int(str(arcpy.GetCount_management(Input)))
            print("Columns: " + str(len(FieldNameList)) + " x Rows: " + str(TotalSites))
            print(FieldNameList)
            out = {'ShapeType':'Table','Rows':TotalSites,'Columns':FieldNameList}
    return(out)
def FieldSummary(Layer,FieldName):
    s1 = pd.Series([row.getValue(FieldName) for row in arcpy.SearchCursor(Layer)])
    plt.bar(np.arange(len(s1.value_counts())),list(s1.value_counts()),align	= 'center')
    plt.xticks(np.arange(len(s1.value_counts())),list(s1.value_counts().index), rotation='vertical')
    plt.xlabel(FieldName)
    plt.title(os.path.basename(Layer))
    plt.show()
    return(s1.value_counts())
def Distance(x1,x2,y1,y2):
    return ((x1-x2)**2+(y1-y2)**2)**0.5
def CreateOutPath(MainFile,appendix,Extension = 'shp'):
    if Extension != '':
        out = os.path.splitext(MainFile)[0] + '_' + appendix + '.' + Extension
    else:
        out = os.path.splitext(MainFile)[0] + '_' + appendix
    try:
        arcpy.Delete_management(out)
    except:
        pass
    return(out)
def CreateOutLayer(Name):
    out = Name
    try:
        arcpy.Delete_management(out)
    except:
        pass
    return(out)
def IRIS_HSM_ShType(SHD_TYP):
    #Converts IRIS shoulder type for hsmpy
    IRIS = {0: 'Not applicable',1: 'Earth',2: 'Sod',3: 'Aggregate',4: 'treated',5: 'Bituminous',6: 'Concrete-untied',7: 'Concrete-tied',8: 'V Gutter',9: 'Curb and Gutter'}
    HSMPY = {'Paved':1,'Gravel':2,'Composite':3,'Turf':4}
    Conv = {'Not applicable':'Turf',
        'Earth':'Turf',
        'Sod':'Turf',
        'Aggregate':'Gravel',
        'treated':'Paved',
        'Bituminous':'Paved',
        'Concrete-untied':'Paved',
        'Concrete-tied':'Paved',
        'V Gutter':'Paved',
        'Curb and Gutter':'Paved'}
    return(HSMPY[Conv[IRIS[SHD_TYP]]])
def AddField(FC,Fields):
    for field in Fields:
        try:
            res = arcpy.AddField_management(FC,field['name'],field['type'],field['precision'],field['scale'],field['length'],field['alias'],field['nullable'],field['required'])
        except:
            try:
                print(res.res.getMessages())
            except:
                pass
            pass
def AddPointFromAddress(Input,AddressField):
    import requests
    APIKey = 'AIzaSyCs80htAI4UAHHuF5m9IclsbMqg1FKxoEQ'
    UC = arcpy.UpdateCursor(Input)
    for r in UC:
        Address = r.getValue(AddressField)
        response = requests.get('https://maps.googleapis.com/maps/api/geocode/json?address=' 
                                + Address.replace(' ','+') +
                               '&key=' + APIKey)
        resp_json_payload = response.json()
        if resp_json_payload['status'] == 'OK':
            
            pnt = arcpy.Point(resp_json_payload['results'][0]['geometry']['location']['lng'],
                          resp_json_payload['results'][0]['geometry']['location']['lat'])
            pntg = arcpy.PointGeometry(pnt,WGS1984).projectAs(NAD1983IL)
            r.setValue(arcpy.Describe(Input).shapeFieldName,pntg)
            UC.updateRow(r)
        else:
            print(resp_json_payload['status'])
        print(','.join([Address,
                        str(resp_json_payload['results'][0]['geometry']['location']['lng']),
                        str(resp_json_payload['results'][0]['geometry']['location']['lat']),
                        str(pntg.firstPoint.X),
                        str(pntg.firstPoint.Y),
                        resp_json_payload['status']]))
    del UC
    del r
def AddSegFromAddresses(AddressList,SegInput,RouteID,Output):
    import requests
    APIKey = 'AIzaSyCs80htAI4UAHHuF5m9IclsbMqg1FKxoEQ'

    PntLayer = CreateOutPath(Output,'pnts','')
    arcpy.CreateFeatureclass_management(
        out_path = os.path.dirname(Output),
        out_name = os.path.basename(PntLayer),
        geometry_type='POINT',
        spatial_reference=NAD1983IL)
    arcpy.AddField_management(PntLayer,'SegID','SHORT')
    arcpy.AddField_management(PntLayer,'Address','TEXT')
    IC = arcpy.InsertCursor(PntLayer)
    i = 0
    for add in AddressList:
        r = IC.newRow()
        r.setValue('SegID',i)
        r.setValue('Address',add[0])
        IC.insertRow(r)
        r = IC.newRow()
        r.setValue('SegID',i)
        r.setValue('Address',add[1])
        IC.insertRow(r)
        i += 1
    del IC
    AddPointFromAddress(PntLayer,'Address')
    
    Buffer = "200 Feet"
    SPJ = CreateOutPath(MainFile=Output,appendix='SPJ',Extension='')
    arcpy.SpatialJoin_analysis(
        target_features = SegInput, 
        join_features = PntLayer, 
        out_feature_class = SPJ, 
        join_operation = "JOIN_ONE_TO_ONE", 
        join_type = "KEEP_COMMON", 
        match_option = "INTERSECT", 
        search_radius = Buffer, 
    )

    UnSplt = CreateOutPath(MainFile=Output,appendix='Unsplt',Extension='')
    arcpy.UnsplitLine_management(
        in_features=SPJ, 
        out_feature_class=UnSplt, 
        dissolve_field="", 
        statistics_fields="")

    SPJ2 = CreateOutPath(MainFile=Output,appendix='SPJ2',Extension='')
    arcpy.SpatialJoin_analysis(
        target_features = UnSplt, 
        join_features = PntLayer, 
        out_feature_class = SPJ2, 
        join_operation = "JOIN_ONE_TO_ONE", 
        join_type = "KEEP_COMMON", 
        match_option = "INTERSECT", 
        search_radius = Buffer, 
    )

    Final_Layer = CreateOutLayer('FinalLayer')
    arcpy.MakeFeatureLayer_management(in_features=SPJ2,out_layer=Final_Layer)
    arcpy.SelectLayerByAttribute_management(in_layer_or_view = Final_Layer,
                                            selection_type = 'NEW_SELECTION',
                                            where_clause = "Join_Count = 2")
    
    EventTable = CreateOutPath(MainFile=Output,appendix='EventTable',Extension='')
    arcpy.LocateFeaturesAlongRoutes_lr(
        in_features                = PntLayer, 
        in_routes                = SPJ, 
        route_id_field            = RouteID, 
        radius_or_tolerance        = Buffer, 
        out_table                = EventTable, 
        out_event_properties    = " ".join([RouteID, "POINT", "MP"]),
        route_locations            = "FIRST", 
        distance_field            = "DISTANCE", 
        in_fields                = "FIELDS", 
        m_direction_offsetting    = "M_DIRECTON"
        )

    SegTable = CreateOutPath(MainFile=Output,appendix='SegTable',Extension='')
    arcpy.CreateTable_management(out_path=os.path.dirname(SegTable),out_name=os.path.basename(SegTable))
    arcpy.AddField_management(SegTable,RouteID,'TEXT')
    arcpy.AddField_management(SegTable,'BEG_STA','DOUBLE')
    arcpy.AddField_management(SegTable,'END_STA','DOUBLE')
    arcpy.AddField_management(SegTable,'Address1','TEXT')
    arcpy.AddField_management(SegTable,'Address2','TEXT')
    #SegIDDict = {r.getValue('SegID'):{'INV':'','BMP':0,'EMP':0,'Add1':'','Add2':''}}
    SegIDDict = {}
    for r in arcpy.SearchCursor(EventTable):
        k = r.getValue('SegID')
        if k in SegIDDict.keys():
            mp = r.getValue('MP')
            add = r.getValue('Address')
            if SegIDDict[k]['BMP']<=mp:
                SegIDDict[k]['EMP'] = mp
                SegIDDict[k]['Add2'] = add
            else:
                SegIDDict[k]['EMP'] = SegIDDict[k]['BMP']
                SegIDDict[k]['BMP'] = mp
                SegIDDict[k]['Add2'] = SegIDDict[k]['Add1']
                SegIDDict[k]['Add1'] = add
        else:
            SegIDDict.update({r.getValue('SegID'):{'INV':r.getValue(RouteID),
                                                   'BMP':r.getValue('MP'),
                                                   'EMP':-1,
                                                   'Add1':r.getValue('Address'),
                                                   'Add2':''}})
            print('End point was not found')
    IC = arcpy.InsertCursor(SegTable)
    for k in SegIDDict.keys():
        r = IC.newRow()
        r.setValue(RouteID,SegIDDict[k]['INV'])
        r.setValue('BEG_STA',SegIDDict[k]['BMP'])
        r.setValue('END_STA',SegIDDict[k]['EMP'])
        r.setValue('Address1',SegIDDict[k]['Add1'])
        r.setValue('Address2',SegIDDict[k]['Add2'])
        IC.insertRow(r)
    del IC

    Overlay_Event_Layer = CreateOutLayer('OverlayEventLayer')
    arcpy.MakeRouteEventLayer_lr(in_routes = SegInput, 
                                 route_id_field = RouteID, 
                                 in_table = SegTable, 
                                 in_event_properties = ' '.join([RouteID,'LINE','BEG_STA','END_STA']), 
                                 out_layer = Overlay_Event_Layer, 
                                 offset_field = "", 
                                 add_error_field = "ERROR_FIELD") 
    
    Sort = CreateOutPath(MainFile=Output,appendix='sort',Extension='')
    arcpy.Sort_management(in_dataset = Overlay_Event_Layer,
                          out_dataset = Sort,
                          sort_field = ';'.join([RouteID,'BEG_STA','END_STA']))
    Final_Layer = CreateOutLayer('FinalLayer')
    
    arcpy.MakeFeatureLayer_management(in_features=Sort,out_layer=Final_Layer)
    arcpy.SelectLayerByAttribute_management(in_layer_or_view = Final_Layer,
                                            selection_type = 'NEW_SELECTION',
                                            where_clause = "Shape_Length > 0")
    arcpy.Delete_management(Output)
    arcpy.CopyFeatures_management(in_features=Final_Layer,out_feature_class=Output)

    arcpy.Delete_management(PntLayer)
    arcpy.Delete_management(SPJ)
    arcpy.Delete_management(SPJ2)
    arcpy.Delete_management(EventTable)
    arcpy.Delete_management(SegTable)
    arcpy.Delete_management(Overlay_Event_Layer)
    arcpy.Delete_management(Sort)
    arcpy.Delete_management(Final_Layer)
    arcpy.Delete_management(UnSplt)
def AttributeTabletoDF(FC,readGeometry = False):
    if readGeometry:
        col = [f.name for f in arcpy.ListFields(FC)]
    else:
        shape = ''
        try:
            shape = arcpy.Describe(FC).shapeFieldName
        except:pass
        col = [f.name for f in arcpy.ListFields(FC) if not f.name == shape]
    df = pd.DataFrame(columns=col)
    for c in col:
        df[c] = [r.getValue(c) for r in arcpy.SearchCursor(FC)]
    df
    return(df)
def AttributeTabletoDF_Old(FC):
    f = os.path.basename(FC)
    ExOut = CreateOutPath(os.path.join(os.getcwd(),f +'_' + strftime("%Y%m%d%H%M%S")) + str(np.random.normal()),'Out','xlsx')

    arcpy.TableToExcel_conversion(FC,ExOut)
    DF = pd.read_excel(ExOut)
    arcpy.Delete_management(ExOut)
    return(DF)
def FieldSummary_temporal(InputDict,Field):
    HSMPY_PATH = r'C:\Users\MR068144\Downloads\Python Scripts'
    
    sys.path.append(HSMPY_PATH)
    
    SubProcess = []
    PyList = []
    keylist = InputDict.keys()
    keylist.sort()
    for key in keylist:
        Output = os.path.join(os.path.dirname(InputDict[keylist[0]]) , 'FST_' + str(key) + '.csv')
        pyFN = os.path.join(os.path.dirname(InputDict[keylist[0]]) , 'FST_' + str(key) + '.py')
        OutFile = open(pyFN, 'w')
        pyfile = """try:
    from time import gmtime, strftime
    print(strftime("%Y-%m-%d %H:%M:%S"))
    import os, sys
    import pandas as pd
    import arcpy
    sys.path.append(r'{}') #1
    import hsmpy
    Input = r"{}"
    Field = "{}"
    key = "{}"
    Output = r"{}"
    print(Input)
    print(Field)
    s = pd.Series([r.getValue(Field) for r in arcpy.SearchCursor(Input)])
    df1 = pd.DataFrame(s.value_counts())
    df1.columns = [key]
    df1['key'] = df1.index
    df1.to_csv(Output)
    print(strftime("%Y-%m-%d %H:%M:%S"))
except Exception as e:
    print e
    raw_input('Press Enter to continue...')
""".format(HSMPY_PATH,InputDict[key],Field,key,Output)
        OutFile.write(pyfile)
        OutFile.close()
        PyList.append(pyFN)
    for py in PyList:
        SubProcess.append(subprocess.Popen(
                [sys.executable, py],
                shell=False,creationflags = subprocess.CREATE_NEW_CONSOLE))  
    w = [p.wait() for p in SubProcess]
    df = pd.DataFrame()
    for key in keylist:
        Output = os.path.join(os.path.dirname(InputDict[keylist[0]]) , 'FST_' + str(key) + '.csv')
        df1 = pd.read_csv(Output)
        df1 = pd.DataFrame(index = df1['key'].values,data=df1[str(key)].values,columns=[key])
        df = pd.concat([df, df1], axis=1)

    for key in keylist:
        Output = os.path.join(os.path.dirname(InputDict[keylist[0]]) , 'FST_' + str(key) + '.csv')
        pyFN = os.path.join(os.path.dirname(InputDict[keylist[0]]) , 'FST_' + str(key) + '.py')
        try:os.remove(Output)
        except:pass
        try:os.remove(pyFN)
        except:pass
    df = df.sort_index()
    #df.plot(title=Field,grid=True,rot=90)
    return(df)
def ListFCinGDBorMDB(DB):
    #print('List of datasets in the {}:\n'.format(DB))
    return([datasets for root, dirs, datasets in arcpy.da.Walk(DB)][0])
def AddIfinFields(fieldp,FCp,fieldnames):
    try:
        FieldList = [f.name for f in arcpy.ListFields(FCp.value)]
        for fieldname in fieldnames:
            if fieldname in FieldList:
                fieldp.value = fieldname
                break
    except: pass
    return fieldp
def ConvertMDBtoGDB(mdb,gdb):
    tables = []
    featurs = []
    FCs = ListFCinGDBorMDB(mdb)
    #print(FCs)
    for fc in FCs:
        ft = arcpy.Describe(mdb+'\\'+fc).dataElementType
        print(ft)
        if ft=='DEFeatureClass':
            featurs.append(mdb + '\\' + fc)
        if ft=='DETable':
            tables.append(mdb + '\\' + fc)
    if os.path.exists(gdb):
        try:
            shutil.rmtree(gdb)
        except Exception as e:
            print(e)
            pass
    arcpy.CreateFileGDB_management(os.path.dirname(gdb),os.path.basename(mdb).split('.')[0])
    if len(featurs)>0:
        arcpy.FeatureClassToGeodatabase_conversion(featurs,gdb)
    if len(tables)>0:
        arcpy.TableToGeodatabase_conversion (tables, gdb)
    outL = featurs
    outL.extend(tables)
    return(outL)
def CON_ConvertMDBtoGDB(WDir,HSMPY_PATH,mdb,gdb):
    sys.path.append(HSMPY_PATH)
    Title = os.path.basename(mdb).split('.')[0]
    pyFN = os.path.join(WDir , 'MDB2GDB_' + Title + '.py')
    OutFile = open(pyFN, 'w')
    pyfile = """from time import gmtime, strftime
print(strftime("%Y-%m-%d %H:%M:%S"))
print("MDB to GDB")
import os, sys
import atexit
#atexit.register(input, 'Press Enter to continue...')
HSMPY_PATH = r'{}'
mdb = r'{}'
gdb = r'{}'
sys.path.append(HSMPY_PATH) 
import hsmpy3
import arcpy
print(gdb)
Out = hsmpy3.common.ConvertMDBtoGDB(mdb,gdb)
print(Out)
print(strftime("%Y-%m-%d %H:%M:%S"))
""".format(HSMPY_PATH,mdb,gdb)
    OutFile.write(pyfile)
    OutFile.close()
    SW_MINIMIZE = 6
    SW_HIDE = 0
    info = subprocess.STARTUPINFO()
    info.dwFlags = subprocess.STARTF_USESHOWWINDOW
    info.wShowWindow = SW_MINIMIZE
    SP = subprocess.Popen(
                [sys.executable, pyFN])
    return(SP)
def WaitIfNecessary(Processes,MaxOpenProcesses,Frequency = 2):
    Poll = [p['Process'].poll() for p in Processes if p['Process'].poll() is None]
    while len(Poll)>=MaxOpenProcesses:
        time.sleep(Frequency)
        Poll = [p['Process'].poll() for p in Processes if p['Process'].poll() is None]
    for p in Processes:
        status = p['Process'].poll()
        p['LastStatus'] = status
        if status is not None and not p['Printed']:
            print('{}: {}'.format(p['HSIP_ID'],status))
            p['Printed'] = True
    return(Processes)


def ConvertPandasDFtoTable(DF,Table,Overwrite={},shape='',spatialReference='',Continue=False): # Incomplete
    def GetFieldTypes(DF,Overwrite={},shape=''):
        df = pd.DataFrame(columns=['Pandas_FieldName','Pandas_FieldType','Esri_FieldName','Esri_FieldType','Esri_FieldLength','Esri_Domain'])
        TypeConverter = {'TEXT':['O','S'],'FLOAT':[],'DOUBLE':['f'],'SHORT':[],'LONG':['i','u'],'DATE':['M','m']}
        try:
            GDBDomains = [d.name for d in arcpy.da.ListDomains(os.path.dirname(Table))]
        except:
            GDBDomains = []
        for c in DF.columns:
            if c!=shape:
                sdt = DF[c].dtype.kind
                cdt = None
                for dt in TypeConverter:
                    if sdt in TypeConverter[dt]:
                        cdt = dt
                max_len = None
                if cdt == 'TEXT':
                    max_len = DF[c].astype(str).apply(len).max()
                if cdt == 'LONG':
                    max_number = DF[c].max()
                    if abs(max_number)<32000:
                        cdt = 'SHORT'
                if cdt == 'DOUBLE':
                    max_number = DF[c].max()
                    if abs(max_number)<1.2E30:
                        cdt = 'FLOAT'
                domain = None
                cc = c.replace(' ','_')
                if cc in GDBDomains:
                    domain = cc
                df.loc[c] = [c,sdt,cc,cdt,max_len,domain]
        for k in Overwrite:
            if k in df.index:
                for c in Overwrite[k]:
                    if c in df.columns:
                        df.loc[k,c]=Overwrite[k][c]
        return(df)
    CurrentRows = 0
    if arcpy.Exists(Table):
        try:
            CurrentRows = int(arcpy.GetCount_management(Table)[0])
        except:
            pass
    if (not Continue) or (Continue and CurrentRows==0):
        print('[{}] create table'.format(strftime("%Y-%m-%d %H:%M:%S")))
        try:
            if arcpy.Exists(Table):
                print('[{}] deleting current table: {}'.format(strftime("%Y-%m-%d %H:%M:%S"),CurrentRows))
                arcpy.Delete_management(Table)
        except:
            print('[{}]  - failed to delete current table'.format(strftime("%Y-%m-%d %H:%M:%S")))
            pass
        AddShape = False
        if shape == '':
            arcpy.CreateTable_management(out_path=os.path.dirname(Table),out_name=os.path.basename(Table))
        else:
            SR = spatialReference
            if SR!='':
                print('[{}] spatial reference passed: {}'.format(strftime("%Y-%m-%d %H:%M:%S"),SR.name))
            srL = DF[shape].loc[~pd.isnull(DF[shape])].apply(lambda pg:pg.spatialReference.factoryCode).unique().tolist()
            if len(srL)==1:
                if srL[0]!=0:
                    SR = arcpy.SpatialReference(srL[0])
                    print('[{}] spatial reference detected: {}'.format(strftime("%Y-%m-%d %H:%M:%S"),SR.name))
            st = DF[shape].loc[~pd.isnull(DF[shape])].apply(lambda x:x.type).unique()
            if len(st)==1:
                #try:
                    if st[0] == 'polyline':
                        AddShape = True
                        import json
                        HasM = 'DISABLED'
                        HasZ = 'DISABLED'
                        for pl in DF[shape].tolist():
                            if 'hasM' in json.loads(pl.JSON):
                                if json.loads(pl.JSON)['hasM']:
                                    HasM = 'ENABLED' 
                            if 'hasZ' in json.loads(pl.JSON):
                                if json.loads(pl.JSON)['hasZ']:
                                    HasZ = 'ENABLED'
                            if HasZ == 'ENABLED' and HasM == 'ENABLED':
                                break
                        print('[{}] geometry detected: POLYLINE {}{}'.format(strftime("%Y-%m-%d %H:%M:%S"),{'ENABLED':'Z','DISABLED':''}[HasZ],{'ENABLED':'M','DISABLED':''}[HasM]))
                        arcpy.CreateFeatureclass_management(out_path=os.path.dirname(Table),out_name=os.path.basename(Table),geometry_type='POLYLINE',spatial_reference=SR,has_z=HasZ,has_m=HasM)
                #except:
                #    pass
                #try:
                    if st[0] == 'point':
                        print('[{}] geometry detected: POINT'.format(strftime("%Y-%m-%d %H:%M:%S")))
                        AddShape = True
                        arcpy.CreateFeatureclass_management(out_path=os.path.dirname(Table),out_name=os.path.basename(Table),geometry_type='POINT',spatial_reference=SR)
                #except:
                #    pass
        print('[{}] adding fields: {}'.format(strftime("%Y-%m-%d %H:%M:%S"),DF.shape[1]))
        df = GetFieldTypes(DF,Overwrite,shape)
        DF[df.loc[df.Esri_FieldType=='TEXT','Pandas_FieldName']] = DF[df.loc[df.Esri_FieldType=='TEXT','Pandas_FieldName']].astype(str)
        for i,r in df.iterrows():
                arcpy.AddField_management(in_table=Table,field_name=r.Esri_FieldName,field_type=r.Esri_FieldType,field_length=r.Esri_FieldLength,field_domain=r.Esri_Domain)
        
        #x = np.array(np.rec.fromrecords(s_df.values))
        #x.dtype.names = tuple(s_df.dtypes.index.tolist())
        #Int_App_FC = Dir_DS.GDB + '\\Seg_' + str(year)
        #arcpy.Delete_management(Int_App_FC)
        #arcpy.da.NumPyArrayToTable(x, Int_App_FC)
        #https://pro.arcgis.com/en/pro-app/arcpy/data-access/extendtable.htm

        print('[{}] inserting rows: {}'.format(strftime("%Y-%m-%d %H:%M:%S"),DF.shape[0]))
        ic = arcpy.InsertCursor(Table)
        for i,r in DF.iterrows():
            row = ic.newRow()
            for j,k in df.iterrows():
                if k.Pandas_FieldName!=shape:
                    v = r[k.Pandas_FieldName]
                    if not pd.isnull(v):
                        try:
                            row.setValue(k.Esri_FieldName,v)
                        except:
                            print('failed: {}, {}'.format(k.Esri_FieldName,v))
            if AddShape and not pd.isnull(r[shape]):
                row.shape = r[shape]
            ic.insertRow(row)
        del ic
        del row
        print('[{}] done! {}'.format(strftime("%Y-%m-%d %H:%M:%S"),arcpy.GetCount_management(Table)))
    else:
        print('[{}] continuing:'.format(strftime("%Y-%m-%d %H:%M:%S")))
        df = GetFieldTypes(DF,Overwrite,shape)
        print('[{}] current rows: {}'.format(strftime("%Y-%m-%d %H:%M:%S"),CurrentRows))
        DF = DF.iloc[CurrentRows:]
        print('[{}] remaining rows: {}'.format(strftime("%Y-%m-%d %H:%M:%S"),DF.shape[0]))
        print('[{}] inserting rows'.format(strftime("%Y-%m-%d %H:%M:%S")))
        ic = arcpy.InsertCursor(Table)
        for i,r in DF.iterrows():
            row = ic.newRow()
            for j,k in df.iterrows():
                if k.Pandas_FieldName!=shape:
                    v = r[k.Pandas_FieldName]
                    if not pd.isnull(v):
                        try:
                            row.setValue(k.Esri_FieldName,v)
                        except:
                            print('failed: {}, {}'.format(k.Esri_FieldName,v))
            if AddShape and not pd.isnull(r[shape]):
                row.shape = r[shape]
            ic.insertRow(row)
        del ic
        del row
        print('[{}] done! {}'.format(strftime("%Y-%m-%d %H:%M:%S"),arcpy.GetCount_management(Table)))
        
        


def GetSpatialReference(FC):
    d = arcpy.Describe(FC)
    s = d.spatialReference
    print(s.factoryCode)
    return(arcpy.SpatialReference(s.factoryCode))
def FCtoDF(FC,readGeometry = False,selectedFields=None):
    if selectedFields is not None:
        col = [f.name for f in arcpy.ListFields(FC) if f.name in selectedFields]
    else:
        shape = ''
        try:
            shape = arcpy.Describe(FC).shapeFieldName
        except:pass
        col = [f.name for f in arcpy.ListFields(FC) if f.name != shape]

    if readGeometry:
        shape = ''
        try:
            shape = arcpy.Describe(FC).shapeFieldName
        except:pass
        if shape != '':
            if not shape in col:
                col.append(shape)
    oid_fieldname = ''
    try:
        oid_fieldname = arcpy.Describe(FC).OIDFieldName
    except:pass
    df = pd.DataFrame(columns=[c for c in col if c != oid_fieldname])
    for c in list(df):
        df[c] = [r.getValue(c) for r in arcpy.SearchCursor(FC)]
    if oid_fieldname != '':
        df.index = [r.getValue(oid_fieldname) for r in arcpy.SearchCursor(FC)]
        df.index.name = oid_fieldname
    df.columns.name = os.path.basename(FC)
    return(df)

#Clearinghouse Scraping:
def CMFClearingHousetoExcel(CMFOutput,MaxCMFID=15000,MAxStudyID=3000,TimeSleep=0.75):
    def CMFtoDF(cmfid):
        cmf_base_url = 'http://www.cmfclearinghouse.org/detail.cfm?facid={}'.format
        cont = requests.get(cmf_base_url(cmfid)).content
        tree = html.fromstring(cont)
        try:
            title = tree.xpath('//*[@id="content"]/h2[2]')[0].text
        except:title = ''
        try:
            desc = tree.xpath('//*[@id="content"]/h3[1]')[0].text[13:]
        except:desc = ''
        try:
            prcond = tree.xpath('//*[@id="content"]/h3[2]/em')[0].text
        except:prcond = ''
        try:
            cat = tree.xpath('//*[@id="content"]/h3[3]')[0].text[10:]
        except:cat = ''
        try:
            studydesc = tree.xpath('//*[@id="content"]/h3[4]')[0].text_content()
        except:studydesc = ''
        try:
            studynum = [c.attrib['href'] for c in tree.xpath('//*[@id="content"]/h3[4]')[0].iterchildren()][0][22:]
        except:studynum = ''
        try:
            star = tree.xpath('//*[@id="details"]')[0][1][1][0][0].attrib['alt']
        except:star = ''

        df = pd.DataFrame(columns=['CMFID','Title','Description','PreCondition','Category','StudyID','StarRating'])
        df.loc[cmfid] = [cmfid,title,desc,prcond,cat,studynum,star]
        return(df)

    cmf_base_url = 'http://www.cmfclearinghouse.org/detail.cfm?facid={}'.format
    CMF_DF = pd.DataFrame()

    for cmfid in range(1,MaxCMFID+1):
        df = CMFtoDF(cmfid)
        try:
            L = pd.read_html(cmf_base_url(cmfid))
        except:
            CMF_DF = pd.concat([CMF_DF,df])
            continue
        try:
            df['CMF']        = L[1].T.loc[1,1]
            df['AdjCMFSE']   = L[1].T.loc[1,2]
            df['UnAdjCMFSE'] = L[1].T.loc[1,3]
        except:pass
        try:
            df['CRF']        = L[2].T.loc[1,1]
            df['AdjCRFSE']   = L[2].T.loc[1,2]
            df['UnAdjCRFSE'] = L[2].T.loc[1,3]
        except:pass
        try:
            df['CrashType']        = L[3].T.loc[1,1]
            df['CrashSeverity']    = L[3].T.loc[1,2]
            df['RoadwayTypes']     = L[3].T.loc[1,3]
            df['NumberofLanes']    = L[3].T.loc[1,4]
            df['RoadDivisionType'] = L[3].T.loc[1,5]
            df['SpeedLimit']       = L[3].T.loc[1,6]
            df['AreaType']         = L[3].T.loc[1,7]
            df['AADT']             = L[3].T.loc[1,8]
            df['TimeofDay']        = L[3].T.loc[1,9]
            df['IntType']          = L[3].T.loc[1,11]
            df['IntGeometry']      = L[3].T.loc[1,12]
            df['TrafficControl']   = L[3].T.loc[1,13]
            df['AADTMajor']        = L[3].T.loc[1,14]
            df['AADTMinor']        = L[3].T.loc[1,15]
        except:pass
        try:
            df['DataDateRange'] = L[4].T.loc[1,1]
            df['Municipality']  = L[4].T.loc[1,2]
            df['State']         = L[4].T.loc[1,3]
            df['Country']       = L[4].T.loc[1,4]
            df['Methodology']   = L[4].T.loc[1,5]
        except:pass
        try:
            df['InHSM']     = L[5].T.loc[1,1]
            df['DateAdded'] = L[5].T.loc[1,2]
            df['Comments']  = L[5].T.loc[1,3]
        except:
            pass
        CMF_DF = pd.concat([CMF_DF,df])
        time.sleep(TimeSleep)
        print(cmf_base_url(cmfid))
    print(CMF_DF.shape)
    #CMF_DF[~pd.isnull(CMF_DF.CMF)].to_excel('CMFClearingHouse_CMF.xlsx')
    print(strftime("%Y-%m-%d %H:%M:%S"))


    # Reading Studies
    print(strftime("%Y-%m-%d %H:%M:%S"))
    study_base_url = 'http://www.cmfclearinghouse.org/study_detail.cfm?stid={}'.format
    def CMFStudytoDF(studyid):
        study_base_url = 'http://www.cmfclearinghouse.org/study_detail.cfm?stid={}'.format
        cont = requests.get(study_base_url(studyid)).content
        tree = html.fromstring(cont)
        try:
            title    = tree.xpath('//*[@id="content"]')[0][1].text_content()[13:]
        except: title = ''
        try:
            authur   = tree.xpath('//*[@id="content"]')[0][2].text_content()[9:]
        except: authur = ''
        try:
            date     = tree.xpath('//*[@id="content"]')[0][3].text_content()[18:]
        except: date = ''
        try:
            abstract = tree.xpath('//*[@id="content"]')[0][4].text_content()[10:]
        except: abstract = ''
        try:
            citation = tree.xpath('//*[@id="content"]')[0][5].text_content()[16:]
        except: citation = ''
        df = pd.DataFrame(columns=['StudyID','Title','Authur','Date','Abstract','Citation'])
        df.loc[studyid] = [studyid,title,authur,date,abstract,citation]
        return(df)
    CMFStudy_DF = pd.DataFrame()
    for studyid in range(1,MAxStudyID+1):
        CMFStudy_DF = pd.concat([CMFStudy_DF,CMFStudytoDF(studyid)])
        print(study_base_url(studyid))
        time.sleep(TimeSleep)
    print(CMFStudy_DF.shape)
    #CMFStudy_DF.to_excel('CMFClearingHouse_Study.xlsx')
    print(strftime("%Y-%m-%d %H:%M:%S"))

    def ConvtoInt(v):
        try:
            o = int(v)
        except:
            o = 0
        return(o)
    CMF_DF.StudyID = [ConvtoInt(v) for v in CMF_DF.StudyID]
    CMFStudy_DF = CMFStudy_DF[CMFStudy_DF.Title!='']
    CMF_DF = CMF_DF.join(CMFStudy_DF,on='StudyID',rsuffix='_Study')
    C_DF = CMF_DF[~pd.isnull(CMF_DF.CMF)]
    C_DF = C_DF[C_DF.CMF!=0]
    print(C_DF.shape)

    C_DF.Date = [int(''.join(c for c in str(s) if c.isdigit())) for s in C_DF.Date]
    C_DF = C_DF[[
        'CMFID','StudyID','Title','Title_Study','Authur','Date',
        'DataDateRange','AreaType','TimeofDay','CrashType','CrashSeverity',
        'Country','State','Municipality','Methodology',
        'CMF','StarRating','Category',
        'InHSM','Comments','Abstract',
        'RoadwayTypes','NumberofLanes','RoadDivisionType','SpeedLimit','AADT',
        'IntType','IntGeometry','TrafficControl','AADTMajor','AADTMinor',
        'Description','PreCondition','DateAdded',
        'AdjCMFSE','UnAdjCMFSE','CRF','AdjCRFSE','UnAdjCRFSE',
        'Citation'
    ]]
    C_DF = C_DF.sort_values(by=['Date','Title_Study','Category','Title'])
    C_DF.to_excel(CMFOutput)
def CON_CMFClearingHousetoExcel(WDir,HSMPY_PATH,CMFOutput,MaxCMFID=15000,MAxStudyID=3000,TimeSleep=0.75):
    import sys, os, arcpy, csv, json, math, subprocess
    sys.path.append(HSMPY_PATH)

    pyFN = os.path.join(WDir , 'ScrapeCMFClearinghouse.py')
    OutFile = open(pyFN, 'w')
    pyfile = """from time import gmtime, strftime
print(strftime("%Y-%m-%d %H:%M:%S"))
import os, sys
import atexit
atexit.register(input, 'Press Enter to continue...')
HSMPY_PATH = r'{}'
CMFOutput = r'{}'
MaxCMFID={}
MAxStudyID={}
TimeSleep={}
print("Scrape CMF Clearinghouse")

sys.path.append(HSMPY_PATH) 
import hsmpy3
import arcpy
hsmpy3.common.CMFClearingHousetoExcel(CMFOutput)
print(strftime("%Y-%m-%d %H:%M:%S"))
""".format(HSMPY_PATH,CMFOutput,MaxCMFID,MAxStudyID,TimeSleep)
    OutFile.write(pyfile)
    OutFile.close()
    SubProcess = subprocess.Popen(
                [sys.executable, pyFN],
                shell=False,creationflags = subprocess.CREATE_NEW_CONSOLE)
    return(SubProcess)

# Plots
def PairTable(DF,Rows,Cols,Percentage=False,Plot=False):
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
def TranslateDomains(DF,domain,subset=[]):
    if len(subset)==0:
        subset = DF.columns
    else:
        subset = [c for c in subset if c in DF.columns]
    for c in subset:
        if hasattr(domain,c):
            d = getattr(domain,c)['codes']
            if len(d)>0:
                DF[c] = DF[c].apply(lambda x:'{}. {}'.format(x,d[x]) if x in d.keys() else x)
    return(DF)
def RunInConsole(WDir,Title,func,**kwargs):
    Title = str(Title)
    pyFN = os.path.join(WDir , Title + '.py')
    OutFile = open(pyFN, 'w')
    f_name = '.'.join([func.__module__ ,func.__name__])
    L = []
    for k in kwargs:
        v = kwargs[k]
        if type(v)==str:
            L.append('{}=r"{}"'.format(k,v))
        else:
            L.append('{}={}'.format(k,v))
    f_args = ','.join(L)
    pyfile = """from time import gmtime, strftime
print(strftime("%Y-%m-%d %H:%M:%S"))
print("{}")
import os, sys
import atexit
atexit.register(input, 'Press Enter to continue...')
Code_Repo_Path = r'\\\\Chcfpp01\Groups\HTS\Code_Repository\Python\Libraries'
Site_Packages = os.path.join(Code_Repo_Path, 'Site_Packages')
print(sys.executable)
print(sys.version)
for p in [Code_Repo_Path,Site_Packages]:
    #if not p in sys.path:
        sys.path.append(p)
import hsmpy3
{}({})
print(strftime("%Y-%m-%d %H:%M:%S"))
""".format(Title,f_name,f_args)
    OutFile.write(pyfile)
    OutFile.close()
    SubProcess = subprocess.Popen(
                [sys.executable, pyFN],
                shell=False,creationflags = subprocess.CREATE_NEW_CONSOLE)
    return(SubProcess)
def TranslateDomains_DF(DF,DomainDF,subset=[]):
    if len(subset)==0:
        subset = DF.columns
    else:
        subset = [c for c in subset if c in DF.columns]
    for c in subset:
        if c in DomainDF['field_name'].tolist():
            print(c)
            idf = DomainDF[DomainDF.field_name==c][['code','description']]
            idf.index = idf.code
            d = idf.description.to_dict()
            if len(d)>0:
                
                DF[c] = DF[c].apply(lambda x:'{}. {}'.format(x,d[x]) if x in d.keys() else x)
    return(DF)
def NoArcpy():
    from  osgeo import ogr, osr

    driver = ogr.GetDriverByName('ESRI Shapefile')
    shp = driver.Open(r"C:\Users\mr068144\Downloads\all2018\hwy2018.shp")
    # Get Projection from layer
    layer = shp.GetLayer()
    spatialRef = layer.GetSpatialRef()
    print (spatialRef)
    print(layer.GetFeatureCount())
    for i in range(2):
        feature = layer.GetFeature(i)
        print (feature.ExportToJson())

    driver = ogr.GetDriverByName("OpenFileGDB")
    ds = driver.Open(r"\\chcfpp01\Proj\ILDOT\650511SAFETYPROGRAM\7_BaseData\Master_BaseFiles_Data\9.0_IL_SafetyDB\IL_SafetyDB.gdb", 0)
    layer = ds.GetLayer("Route_2017")
    sr = layer.GetSpatialRef()
    print(sr)
    print(layer.GetFeatureCount())
    for i in range(1,2):
        feature = layer.GetFeature(i)
        print (feature.ExportToJson())    
def TranslateDomains_CSV(DF,DomainCSV,subset=[],add_code_prefix=True):
    import os
    df = pd.DataFrame(columns=['field_name'])
    if os.path.basename(DomainCSV).split('.')[-1].lower()=='csv':
        df = pd.read_csv(DomainCSV)
    if os.path.basename(DomainCSV).split('.')[-1].lower() in ['xls','xlsx']:
        df = pd.read_excel(DomainCSV)
    if len(subset)==0:
        subset = DF.columns
    else:
        subset = [c for c in subset if c in DF.columns]
    for c in subset:
        if c in df.field_name.unique():
            idf = df[df.field_name==c][['code','description']]
            idf.index = idf.code
            d = idf.description.to_dict()
            if len(d)>0:
                if add_code_prefix:
                    DF[c] = DF[c].apply(lambda x:'{}. {}'.format(x,d[x]) if x in d.keys() else x)
                else:
                    DF[c] = DF[c].apply(lambda x:'{}'.format(d[x]) if x in d.keys() else x)
    return(DF)