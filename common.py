
import pandas as pd
import numpy as np
from time import gmtime, strftime
import os,sys

#NAD1983IL = arcpy.SpatialReference(102672)
#NAD1983SC = arcpy.SpatialReference(102733)
#NAD1983NC = arcpy.SpatialReference(2264)
#WGS1984   = arcpy.SpatialReference(4326)
#NAD1983CT = arcpy.SpatialReference(26956)

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
def PrintSummary(Input,extension='shp'):
    import arcpy
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
def RunInConsole(WDir,Title,func,**kwargs):
    import subprocess
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
import hsmpy31
{}({})
print(strftime("%Y-%m-%d %H:%M:%S"))
""".format(Title,f_name,f_args)
    OutFile.write(pyfile)
    OutFile.close()
    SubProcess = subprocess.Popen(
                [sys.executable, pyFN],
                shell=False,creationflags = subprocess.CREATE_NEW_CONSOLE)
    return(SubProcess)
def AssignTiers(Array,Num_Classes):
    from pysal.viz.mapclassify import Natural_Breaks as nb
    nb_res = nb(Array, k=Num_Classes)
    #display(nb_res)
    df = pd.DataFrame(Array)
    for lim,lim2,tier in zip([min(Array)-1] + list(nb_res.bins)[:-1],list(nb_res.bins),reversed(['Tier{}'.format(i) for i in range(1,Num_Classes+1)])):
        df.loc[df[Array.name]>lim,'Tier'] = tier
        df.loc[df[Array.name]>lim,'TierMean'] = (lim+lim2)/2
    df = df[['Tier','TierMean']]
    return(df)
def ServiceURLtoDF(URL):
    import arcgis
    import pandas as pd
    fl = arcgis.features.FeatureLayer(URL)
    L = []
    for f in fl.query():
        L.append(f)
    DF = pd.DataFrame([pd.Series(l.attributes) for l in L])
    Shape = pd.DataFrame([pd.Series(l.geometry) for l in L])
    CDF = pd.concat([DF,Shape],axis=1)
    return(CDF)
def DataRepo(Owner='',Name='',Type='',readGeometry=True):
    from time import gmtime, strftime
    import datetime
    import psutil
    t_print = lambda x:print('[{}] {}'.format(strftime("%Y-%m-%d %H:%M:%S"),x))
    def print_fileinfo(row):
        path = row.Path
        if row.Type=='feature class':
            size = int(arcpy.GetCount_management(path)[0])
            t_print('{} - Size: {:,.0f} Rows'.format(path,size))
        else:
            size = os.path.getsize(path)/2**20
            d = datetime.datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M:%S")
            t_print('{} - Size: {:,.2f} MB - Last Modified: {}'.format(path,size,d))
    def print_dfinfo(df):
        ava_ram = psutil.virtual_memory().available/2**30
        t_print(' Size: {} - Remaining RAM: {:,.2f} GB'.format(str(df.shape),ava_ram))            

    fp = r"\\chcfpp01\Groups\HTS\Data_Repository\HTS_Data_Registry.xlsx"
    DataRepo_DF = pd.read_excel(fp)
    Mask = DataRepo_DF['Owner']=='ZZZZZZ'
    if Owner!='':
        Mask = Mask | (DataRepo_DF['Owner'] == Owner)
    if Name!='':
        Mask = Mask & (DataRepo_DF['Name'] == Name)
    if Type!='':
        Mask = Mask & (DataRepo_DF['Type'] == Type)
    df = DataRepo_DF[Mask]
    Res = []
    for i,r in df.iterrows():
        print_fileinfo(r)
        if r.Type in ['feature class']:
            df = hsmpy31.common.FCtoDF_numpy(r.Path)
            if readGeometry:
                s = hsmpy31.common.FCtoDF(r.Path,readGeometry=True,selectedFields=['Shape'])
                df['Shape'] = s
            print_dfinfo(df)
            Res.append(df)
        if r.Type in ['shapefile','shp']:
            df = hsmpy31.common.ShapeFiletoDF(r.Path)
            if readGeometry:
                s = hsmpy31.common.FCtoDF(r.Path,readGeometry=True,selectedFields=['Shape'])
                df['Shape'] = s
            print_dfinfo(df)
            Res.append(df)
        if r.Type=='excel':
            df = pd.read_excel(r.Path)
            print_dfinfo(df)
            Res.append(df)
        if r.Type=='csv':
            df = pd.read_csv(r.Path)
            print_dfinfo(df)
            Res.append(df)
    if len(Res)==1:
        Res = Res[0]
    return(Res)
def ConvertDFtoTable(Input_DF,Output_Table,Chunks=1):
    import arcpy
    not_allowed_characters = list('+ /()')
    def replace_all(text, List):
        for i in List:
            text = text.replace(i, '_')
        return text
    Input_DF.columns = [replace_all(c,not_allowed_characters) for c in Input_DF.columns]    
    if Chunks>1:
        n = Input_DF.shape[0]
        Chunks = min(Chunks,n)
        m = int(n/Chunks)
        L = [(m*(i),m*(i+1)-1) for i in range(Chunks)]        
        
        print('[{}] reading data types'.format(strftime("%Y-%m-%d %H:%M:%S")))
        DT = []
        for c in Input_DF.columns:
            arr = np.array(np.rec.fromrecords(Input_DF[[c]].values))
            arr.dtype.names = [c]
            DT.append(arr.dtype)
        
        for i,l in enumerate(L):
            print('[{}] converting df ({} of {}) to np array'.format(strftime("%Y-%m-%d %H:%M:%S"),i+1,Chunks))
            idf = Input_DF.iloc[l[0]:l[1]+1]
            i_out = Output_Table + '_{}'.format(i)
            x = np.array(np.rec.fromrecords(idf.values,dtype=[(i.names[0],str(i[0])) for i in DT]))
            #x.dtype.names = tuple(idf.dtypes.index.tolist())
            print('[{}] converting np array to table'.format(strftime("%Y-%m-%d %H:%M:%S")))
            arcpy.Delete_management(i_out)
            arcpy.da.NumPyArrayToTable(x, i_out)
            print('[{}] Table Output {}'.format(strftime("%Y-%m-%d %H:%M:%S"),arcpy.GetCount_management(i_out)[0]))
        print('[{}] merging tables'.format(strftime("%Y-%m-%d %H:%M:%S")))
        arcpy.Delete_management(Output_Table)
        arcpy.Merge_management([Output_Table + '_{}'.format(i) for i in range(Chunks)],Output_Table)
        print('[{}] Table Output {}'.format(strftime("%Y-%m-%d %H:%M:%S"),arcpy.GetCount_management(Output_Table)[0]))
        print('[{}] deleting chunks'.format(strftime("%Y-%m-%d %H:%M:%S")))
        for i in range(Chunks):
            arcpy.Delete_management(Output_Table + '_{}'.format(i))
    else:
        print('[{}] converting df to np array'.format(strftime("%Y-%m-%d %H:%M:%S")))
        x = np.array(np.rec.fromrecords(Input_DF.values))
        x.dtype.names = tuple(Input_DF.dtypes.index.tolist())
        print('[{}] converting np array to table'.format(strftime("%Y-%m-%d %H:%M:%S")))
        arcpy.Delete_management(Output_Table)
        arcpy.da.NumPyArrayToTable(x, Output_Table)
        print('[{}] Table Output {}'.format(strftime("%Y-%m-%d %H:%M:%S"),arcpy.GetCount_management(Output_Table)[0]))
        print('[{}] done: {}'.format(strftime("%Y-%m-%d %H:%M:%S"),arcpy.GetCount_management(Output_Table)[0]))