import arcpy
import pandas as pd
import numpy as np
from time import gmtime, strftime
import os
t_print = lambda x:print('[{}] {}'.format(strftime("%Y-%m-%d %H:%M:%S"),x))

#Spatial References
WGS1984   = arcpy.SpatialReference(4326)
UTM13N    = arcpy.SpatialReference(26913)
NAD1983IL = arcpy.SpatialReference(102672)
NAD1983SC = arcpy.SpatialReference(102733)
NAD1983NC = arcpy.SpatialReference(2264)
NAD1983CT = arcpy.SpatialReference(26956)
NAD1983CO = arcpy.SpatialReference(6428)
NAD1983NM = arcpy.SpatialReference(6529)
NAD1983TX = arcpy.SpatialReference(2277)

def GetSpatialReference(FC):
    import arcpy
    d = arcpy.Describe(FC)
    s = d.spatialReference
    print(s.factoryCode)
    return(arcpy.SpatialReference(s.factoryCode))

# GDB List of Features
def ListFCinGDB(DB):
    import arcpy
    #print('List of datasets in the {}:\n'.format(DB))
    return([datasets for root, dirs, datasets in arcpy.da.Walk(DB)][0])
def ListFCinMDB(DB):
    import pyodbc
    DRV = 'Microsoft Access Driver (*.mdb, *.accdb)'
    con = pyodbc.connect('DRIVER={};DBQ={}'.format(DRV,MDB))
    cur = con.cursor()
    tables = list(cur.tables())    
    cur.close()
    con.close()
    tables = [t[2] for t in tables if t[3]=='TABLE']
    return(tables)

# GDB to Pandas
def FCtoDF_cursor(FC,readGeometry = False,selectedFields=None):
    '''
    uses cursors to read the data
    '''
    import arcpy
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
def FCtoDF_numpy(Table,ListOfFields=[],readGeometry = False):
    '''
    uses numpy to read the data
    '''
    from time import gmtime, strftime
    import arcpy
    def GetDropList(Table,FNames):
        DropList = []
        for f in FNames:
            try:
                arr = arcpy.da.TableToNumPyArray (in_table=Table, field_names=[f],null_value=-1)
            except:
                DropList.append(f)
        print('[{}]  - cannot process these fields: {}'.format(strftime("%Y-%m-%d %H:%M:%S"),DropList))
        FNames = [f for f in FNames if not f in DropList]
        return(FNames)
    try:
        ShapeF = arcpy.Describe(Table).shapeFieldName
    except:
        ShapeF = ""

    try:
        oid_fieldname = arcpy.Describe(Table).OIDFieldName
    except:
        oid_fieldname = ''
    if len(ListOfFields)>0:
        FNames = [f.name for f in arcpy.ListFields(Table) if f.name in ListOfFields]
    else:
         FNames = [f.name for f in arcpy.ListFields(Table)]
    if len(ListOfFields)>0:
        FNames = [f for f in FNames if f in ListOfFields]
    if not oid_fieldname in FNames:
        FNames = [oid_fieldname] + FNames
    if ShapeF in FNames and ShapeF!="":
        print('[{}] converting feature class to numpy array'.format(strftime("%Y-%m-%d %H:%M:%S")))
        try:
            arr = arcpy.da.FeatureClassToNumPyArray (in_table=Table, field_names=FNames,null_value=-1)
        except:
            print('[{}] examining the fields'.format(strftime("%Y-%m-%d %H:%M:%S")))
            FNames = GetDropList(Table,FNames)
            arr = arcpy.da.FeatureClassToNumPyArray (in_table=Table, field_names=FNames,null_value=-1)
        if readGeometry:
            print('[{}] reading the geometry'.format(strftime("%Y-%m-%d %H:%M:%S")))
            shape_series = pd.Series()
            sc = arcpy.SearchCursor(Table,["OID@", "SHAPE@"])
            for row in sc:
                i,s = row.getValue(oid_fieldname),row.getValue(ShapeF)
                shape_series.loc[i] = s        
            del sc
            del row
    else:
        print('[{}] converting table to numpy array'.format(strftime("%Y-%m-%d %H:%M:%S")))
        try:
            arr = arcpy.da.TableToNumPyArray(in_table=Table, field_names=FNames,null_value=-1)
        except:
            print('[{}] examining the fields'.format(strftime("%Y-%m-%d %H:%M:%S")))
            FNames = GetDropList(Table,FNames)
            arr = arcpy.da.TableToNumPyArray(in_table=Table, field_names=FNames,null_value=-1)

    print('[{}] converting numpy array to pandas'.format(strftime("%Y-%m-%d %H:%M:%S")))
    df = pd.DataFrame(data=[[v for v in r] for r in arr],columns= arr.dtype.names)
    if oid_fieldname in df.columns:
        df.index = pd.Index(df[oid_fieldname],name= oid_fieldname + '_IDX')
    if ShapeF in FNames and ShapeF!="" and readGeometry:
        df[ShapeF] = shape_series
    print('[{}] done! - '.format(strftime("%Y-%m-%d %H:%M:%S")) + str(df.shape))
    return(df)
def FCtoDF_json(Table):
    import json
    try: 
        ShapeF = arcpy.Describe(Table).shapeFieldName
    except:
        ShapeF = ""
    
    json_fp = arcpy.env.scratchFolder + '\\fctodf.json'
    arcpy.Delete_management(json_fp)
    t_print('convert features to json')
    arcpy.FeaturesToJSON_conversion(in_features=Table,out_json_file=json_fp,format_json='NOT_FORMATTED',geoJSON='NO_GEOJSON',include_m_values='M_VALUES',include_z_values='Z_VALUES',outputToWGS84='KEEP_INPUT_SR')
    t_print('read json')
    f = open(json_fp)
    j = json.loads(f.read())
    f.close()
    t_print('convert to pandas')
    df = pd.DataFrame.from_dict(j['features'])
    df1 = df.attributes.apply(lambda d:pd.Series(d))
    if 'geometry' in list(df) and ShapeF!='':
        t_print('convert geometry')
        df1[ShapeF] = df.geometry.apply(lambda x:arcpy.AsShape(x,True))
    t_print(df1.shape)
    return(df1)
def ShapeFiletoDF(path):
    import shapefile
    sf_path = path
    sf = shapefile.Reader(sf_path, encoding = 'Shift-JIS')

    fields = [x[0] for x in sf.fields][1:]
    records = [y[:] for y in sf.records()]
    shps = [s.points for s in sf.shapes()]
    sf_df = pd.DataFrame(columns = fields, data = records)
    sf_df['Shape'] = shps
    return(sf_df)

# Pandas to GDB
def DFtoTable_numpy(Input_DF1,Output_Table,Chunks=1,verbose=2):
    import arcpy
    Input_DF = Input_DF1.copy(deep=True)
    not_allowed_characters = list('+ /()')
    def replace_all(text, List):
        for i in List:
            text = text.replace(i, '_')
        return text
    Input_DF.columns = [replace_all(c,not_allowed_characters) for c in Input_DF.columns]    
    date_fields = [c for c in list(Input_DF) if Input_DF[c].dtypes==np.dtype('<M8[ns]')]
    for c in date_fields:
        if verbose in [2]:
            print(' - date field found: {}'.format(c))
        Input_DF[c] = Input_DF[c].astype(str)
        Input_DF.rename(columns={c:c + '_TEXT'},inplace=True)

    if Chunks>1:
        n = Input_DF.shape[0]
        Chunks = min(Chunks,n)
        m = int(n/Chunks)
        L = [(m*(i),m*(i+1)-1) for i in range(Chunks)]        
        if verbose in [2]:
            print('[{}] reading data types'.format(strftime("%Y-%m-%d %H:%M:%S")))
        DT = []
        for c in Input_DF.columns:
            arr = np.array(np.rec.fromrecords(Input_DF[[c]].values))
            arr.dtype.names = [c]
            DT.append(arr.dtype)
        
        for i,l in enumerate(L):
            if verbose in [2]:
                print('[{}] converting df ({} of {}) to np array'.format(strftime("%Y-%m-%d %H:%M:%S"),i+1,Chunks))
            idf = Input_DF.iloc[l[0]:l[1]+1]
            i_out = Output_Table + '_{}'.format(i)
            x = np.array(np.rec.fromrecords(idf.values,dtype=[(i.names[0],str(i[0])) for i in DT]))
            #x.dtype.names = tuple(idf.dtypes.index.tolist())
            if verbose in [2]:
                print('[{}] converting np array to table'.format(strftime("%Y-%m-%d %H:%M:%S")))
            arcpy.Delete_management(i_out)
            arcpy.da.NumPyArrayToTable(x, i_out)
            if verbose in [2]:
                print('[{}] Table Output {}'.format(strftime("%Y-%m-%d %H:%M:%S"),arcpy.GetCount_management(i_out)[0]))
        if verbose in [2]:
            print('[{}] merging tables'.format(strftime("%Y-%m-%d %H:%M:%S")))
        arcpy.Delete_management(Output_Table)
        arcpy.Merge_management([Output_Table + '_{}'.format(i) for i in range(Chunks)],Output_Table)
        if verbose in [1,2]:
            print('[{}] Table Output {}'.format(strftime("%Y-%m-%d %H:%M:%S"),arcpy.GetCount_management(Output_Table)[0]))
        if verbose in [2]:
            print('[{}] deleting chunks'.format(strftime("%Y-%m-%d %H:%M:%S")))
        for i in range(Chunks):
            arcpy.Delete_management(Output_Table + '_{}'.format(i))
    else:
        if verbose in [2]:
            print('[{}] converting df to np array'.format(strftime("%Y-%m-%d %H:%M:%S")))
        x = np.array(np.rec.fromrecords(Input_DF.values))
        x.dtype.names = tuple(Input_DF.dtypes.index.tolist())
        if verbose in [2]:
            print('[{}] converting np array to table'.format(strftime("%Y-%m-%d %H:%M:%S")))
        arcpy.Delete_management(Output_Table)
        arcpy.da.NumPyArrayToTable(x, Output_Table)
        if verbose in [2]:
            print('[{}] Table Output {}'.format(strftime("%Y-%m-%d %H:%M:%S"),arcpy.GetCount_management(Output_Table)[0]))
        if verbose in [1,2]:
            print('[{}] done: {}'.format(strftime("%Y-%m-%d %H:%M:%S"),arcpy.GetCount_management(Output_Table)[0]))
    for c in date_fields:
        cb = '''def myfun(x):
            import pandas as pd
            if pd.isnull(x) or x=='NaT':
                return()
            else:
                return(x)'''
        arcpy.AddField_management(Output_Table,c,'DATE')
        arcpy.CalculateField_management(Output_Table,c,'myfun(!' + c + '_TEXT!)',code_block=cb  )
        arcpy.DeleteField_management(Output_Table,c+'_TEXT')
def DFtoFC_numpy(Input_DF,XYFields,Output_FC,Spatial_reference):
    import arcpy
    print('[{}] converting df to np array'.format(strftime("%Y-%m-%d %H:%M:%S")))
    date_fields = [c for c in list(Input_DF) if Input_DF[c].dtypes==np.dtype('<M8[ns]')]
    for c in date_fields:
        print(' - date field found: {}'.format(c))
        Input_DF[c] = Input_DF[c].astype(str)
        Input_DF.rename(columns={c:c + '_TEXT'},inplace=True)
    try:
        x = np.array(np.rec.fromrecords(Input_DF.values))
        x.dtype.names = tuple(Input_DF.dtypes.index.tolist())
        print('[{}] converting np array to table'.format(strftime("%Y-%m-%d %H:%M:%S")))
        if arcpy.Exists(Output_FC):
            arcpy.Delete_management(Output_FC)
        arcpy.da.NumPyArrayToFeatureClass(x,Output_FC,XYFields,Spatial_reference)
    except:
        print('[{}] error in conversion - checking fields:'.format(strftime("%Y-%m-%d %H:%M:%S")))
        for c in list(Input_DF):
            if not c in XYFields:
                df = Input_DF[[c] + XYFields]
                try:
                    x = np.array(np.rec.fromrecords(df.values))
                    x.dtype.names = tuple(df.dtypes.index.tolist())
                    if arcpy.Exists(Output_FC):
                        arcpy.Delete_management(Output_FC)
                    arcpy.da.NumPyArrayToFeatureClass(x,Output_FC,XYFields,Spatial_reference)
                except:
                    print('[{}] converting {} to string'.format(strftime("%Y-%m-%d %H:%M:%S"),c))
                    Input_DF[c] = Input_DF[c].astype(str)
        x = np.array(np.rec.fromrecords(Input_DF.values))
        x.dtype.names = tuple(Input_DF.dtypes.index.tolist())
        print('[{}] converting np array to table'.format(strftime("%Y-%m-%d %H:%M:%S")))
        if arcpy.Exists(Output_FC):
            arcpy.Delete_management(Output_FC)
        arcpy.da.NumPyArrayToFeatureClass(x,Output_FC,XYFields,Spatial_reference)

    for c in date_fields:
        cb = '''def myfun(x):
            import pandas as pd
            if pd.isnull(x) or x=='NaT':
                return()
            else:
                return(x)'''
        arcpy.AddField_management(Output_FC,c,'DATE')
        arcpy.CalculateField_management(Output_FC,c,'myfun(!' + c + '_TEXT!)',code_block=cb  )
        arcpy.DeleteField_management(Output_FC,c+'_TEXT')

    print('[{}] done: {}'.format(strftime("%Y-%m-%d %H:%M:%S"),arcpy.GetCount_management(Output_FC)[0]))
def DFtoFCorTable_cursor(DF,Table,Overwrite={},shape='',spatialReference='',Continue=False): # Incomplete
    import arcpy
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
                cc = c.replace(' ','_').replace('.','_')
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
                            #print('failed: {}, {}'.format(k.Esri_FieldName,v))
                            pass
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

# Pandas to Route Event
def DFtoRouteEvent(Input_DF,Route_FC,Output_FC,offset_field=''):
    """
    Converts a pandas Dataframe to a route event feature class. 
    Input_DF: pandas dataframe with RID,BMP,EMP + additional columns
    Route_FC: Ployline M feature class with RID (M values should be populated)
    Output_FC: the output feature class, will be overwritten if existed, a polyline segmented based on the input DF
    """
    import sys
    import os
    import pandas as pd
    import numpy as np
    import time
    import shutil
    import arcpy
    ts = strftime("%Y%m%d_%H%M%S")
    Table_Out = Output_FC + '_Table_' + ts
    DFtoTable_numpy(Input_DF,Table_Out,Chunks=1,verbose=1)

    print('[{}] converting table to feature class'.format(strftime("%Y-%m-%d %H:%M:%S")))
    EventLayer = Output_FC + '_Layer_'  + ts
    arcpy.Delete_management(EventLayer)
    arcpy.MakeRouteEventLayer_lr(in_routes = Route_FC,
                                     route_id_field = 'RID', 
                                     in_table = Table_Out, 
                                     in_event_properties = ' '.join(['RID','LINE','BMP','EMP']), 
                                     out_layer = EventLayer, 
                                     offset_field = offset_field, 
                                     add_error_field = "ERROR_FIELD") 

    arcpy.Delete_management(Output_FC)
    arcpy.CopyFeatures_management(in_features=EventLayer,out_feature_class=Output_FC)
    arcpy.Delete_management(Table_Out)
    arcpy.Delete_management(EventLayer)
    print('[{}] done: {}'.format(strftime("%Y-%m-%d %H:%M:%S"),arcpy.GetCount_management(Output_FC)[0]))

# Other
def reorder_fields(table, out_table, field_order, add_missing=True):
    """ 
    Reorders fields in input featureclass/table
    :table:         input table (fc, table, layer, etc)
    :out_table:     output table (fc, table, layer, etc)
    :field_order:   order of fields (objectid, shape not necessary)
    :add_missing:   add missing fields to end if True (leave out if False)
    -> path to output table
    """
    existing_fields = arcpy.ListFields(table)
    existing_field_names = [field.name for field in existing_fields]

    existing_mapping = arcpy.FieldMappings()
    existing_mapping.addTable(table)

    new_mapping = arcpy.FieldMappings()

    def add_mapping(field_name):
        mapping_index = existing_mapping.findFieldMapIndex(field_name)

        # required fields (OBJECTID, etc) will not be in existing mappings
        # they are added automatically
        if mapping_index != -1:
            field_map = existing_mapping.fieldMappings[mapping_index]
            new_mapping.addFieldMap(field_map)

    # add user fields from field_order
    for field_name in field_order:
        if field_name not in existing_field_names:
            raise Exception("Field: {0} not in {1}".format(field_name, table))

        add_mapping(field_name)

    # add missing fields at end
    if add_missing:
        missing_fields = [f for f in existing_field_names if f not in field_order]
        for field_name in missing_fields:
            add_mapping(field_name)

    # use merge with single input just to use new field_mappings
    arcpy.Merge_management(table, out_table, new_mapping)
    return out_table
def MDBtoDF(MDB,Table):
    import pyodbc
    DRV = 'Microsoft Access Driver (*.mdb, *.accdb)'
    con = pyodbc.connect('DRIVER={};DBQ={}'.format(DRV,MDB))
    SQL = 'SELECT * FROM {};'.format(Table)
    df = pd.read_sql(SQL, con)
    con.close()
    return(df)

def DFtoRouteEvent_POINT(Input_DF,Route_FC,Output_FC,offset_field=''):
    """
    Converts a pandas Dataframe to a route event feature class. 
    Input_DF: pandas dataframe with RID,BMP,EMP + additional columns
    Route_FC: Ployline M feature class with RID (M values should be populated)
    Output_FC: the output feature class, will be overwritten if existed, a polyline segmented based on the input DF
    """
    import sys
    import os
    import pandas as pd
    import numpy as np
    import time
    import shutil
    import arcpy
    ts = strftime("%Y%m%d_%H%M%S")
    Table_Out = Output_FC + '_Table_' + ts
    DFtoTable_numpy(Input_DF,Table_Out,Chunks=1,verbose=1)

    print('[{}] converting table to feature class'.format(strftime("%Y-%m-%d %H:%M:%S")))
    EventLayer = Output_FC + '_Layer_'  + ts
    arcpy.Delete_management(EventLayer)
    arcpy.MakeRouteEventLayer_lr(in_routes = Route_FC,
                                     route_id_field = 'RID', 
                                     in_table = Table_Out, 
                                     in_event_properties = ' '.join(['RID','POINT','MP']), 
                                     out_layer = EventLayer, 
                                     offset_field = offset_field, 
                                     add_error_field = "ERROR_FIELD") 

    arcpy.Delete_management(Output_FC)
    arcpy.CopyFeatures_management(in_features=EventLayer,out_feature_class=Output_FC)
    arcpy.Delete_management(Table_Out)
    arcpy.Delete_management(EventLayer)
    print('[{}] done: {}'.format(strftime("%Y-%m-%d %H:%M:%S"),arcpy.GetCount_management(Output_FC)[0]))

def SpatialJoin(target_features, join_features,GDB,join_operation='JOIN_ONE_TO_ONE',join_type='KEEP_ALL',field_mapping='#',match_option='INTERSECT',
                search_radius='', distance_field_name='SPJ_DIST'):
    import arcpy
    ts = strftime("%Y%m%d_%H%M%S")
    spj_fc = GDB + '\\SPJ_'+ ts
    if arcpy.Exists(spj_fc):
        arcpy.Delete_management(spj_fc)
    Res = arcpy.analysis.SpatialJoin(target_features,join_features,spj_fc,
                                     join_operation,join_type,field_mapping,match_option,search_radius,distance_field_name)
    SPJ_DF = FCtoDF_numpy(spj_fc)
    arcpy.Delete_management(spj_fc)
    return({'DF':SPJ_DF,'Results':Res})