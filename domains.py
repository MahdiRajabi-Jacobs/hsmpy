
import pandas as pd
import numpy as np

def GetTranslationDict(Domain_CSV,field_name,add_prefix=False):
    """
    returns a translation dictionary for the filed_name based on the Domain_CSV
    Domain_CSV: string for full path to a CSV file containing at least "field_name", "code", and "description" fields
    field_name: string for filtering the CSV "field_name" column 
    """
    IL_DM = pd.read_csv(Domain_CSV)
    idx = IL_DM.field_name==field_name
    idf = IL_DM.loc[idx]
    if idf.field_type.iloc[0]=='SHORT':
        idf['code'] = idf.code.astype(int)
    if idf.field_type.iloc[0]=='FLOAT':
        idf['code'] = idf.code.astype(float)
    if add_prefix:
        d = pd.Series(index = idf['code'].tolist(),data=['{}_{}'.format(r['code'],r['description']) for i,r in idf.iterrows()]).to_dict()
    else:
        d = pd.Series(index = idf['code'].tolist(),data=idf['description'].tolist()).to_dict()


    return(d)
def GetTranslationDict_DF(Domain_DF,field_name,add_prefix=False):
    """
    returns a translation dictionary for the filed_name based on the Domain_CSV
    Domain_CSV: string for full path to a CSV file containing at least "field_name", "code", and "description" fields
    field_name: string for filtering the CSV "field_name" column 
    """
    IL_DM = Domain_DF
    idx = IL_DM.field_name==field_name
    idf = IL_DM.loc[idx]
    if idf.field_type.iloc[0]=='SHORT':
        idf['code'] = idf.code.astype(float).astype(int)
    if idf.field_type.iloc[0]=='FLOAT':
        idf['code'] = idf.code.astype(float)
    if add_prefix:
        d = pd.Series(index = idf['code'].tolist(),data=['{}_{}'.format(r['code'],r['description']) for i,r in idf.iterrows()]).to_dict()
    else:
        d = pd.Series(index = idf['code'].tolist(),data=idf['description'].tolist()).to_dict()


    return(d)
def GetTranslationDict2_DF(Domain_DF,Field_Domain,field_name,add_prefix=False):
    if field_name in list(Field_Domain.field_name):
        domain_name = Field_Domain[Field_Domain.field_name==field_name].domain_name.iloc[0]
        if domain_name in list(Domain_DF.field_name):
            idf = Domain_DF[Domain_DF.field_name==domain_name]
            if idf.field_type.iloc[0]=='SHORT':
                idf['code'] = idf.code.astype(int)
            if idf.field_type.iloc[0]=='FLOAT':
                idf['code'] = idf.code.astype(float)
            if add_prefix:
                d = pd.Series(index = idf['code'].tolist(),data=['{}_{}'.format(r['code'],r['description']) for i,r in idf.iterrows()]).to_dict()
            else:
                d = pd.Series(index = idf['code'].tolist(),data=idf['description'].tolist()).to_dict()
            return(d)
        return({})
    return({})

def TranslateDomains_CSV(DF,DomainCSV,subset=[],add_code_prefix=True):
    """
    returns the input dataframe with translated fields based on the domain CSV file
    DF: input pandas dataframe
    Domain_CSV: string for full path to a CSV file containing at least "field_name", "code", and "description" fields
    subset: list of columns to be translated
    add_code_prefix: adds the code before the text if True
    """
    import os
    DF = DF.copy(deep=True)
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
            d = GetTranslationDict(DomainCSV,c,add_prefix=False)
            if len(d)>0:
                if add_code_prefix:
                    DF[c] = DF[c].apply(lambda x:'{}. {}'.format(x,d[x]) if x in d.keys() else x)
                else:
                    DF[c] = DF[c].apply(lambda x:'{}'.format(d[x]) if x in d.keys() else x)
    return(DF)
def AddDomainsToGDB(GDB,DomainCSV,replace=False):
    import os
    from time import gmtime, strftime
    import arcpy

    df = pd.DataFrame(columns=['field_name'])
    if os.path.basename(DomainCSV).split('.')[-1].lower()=='csv':
        df = pd.read_csv(DomainCSV)
    if os.path.basename(DomainCSV).split('.')[-1].lower() in ['xls','xlsx']:
        df = pd.read_excel(DomainCSV)

    print('[{}] Add Domains'.format(strftime("%Y-%m-%d %H:%M:%S")))
    ListDomains = list(df.field_name.unique())
    GDBDomains = [d.name for d in arcpy.da.ListDomains(GDB)]
    for domain in ListDomains:
        if domain['name'] in GDBDomains:
            if replace:
                try:
                    arcpy.DeleteDomain_management (in_workspace=GDB, domain_name=domain['name'])
                except:
                    print('[{}]  - Failed to Delete {}'.format(strftime("%Y-%m-%d %H:%M:%S"),domain['name']))
                    continue
            else:
                continue
        print('[{}]  - {}'.format(strftime("%Y-%m-%d %H:%M:%S"),domain['name']))
        arcpy.CreateDomain_management(in_workspace=GDB,
                                      domain_name=domain['name'],
                                      domain_description=domain['alias'],
                                      field_type=domain['type'], 
                                      domain_type="CODED")
        for code in domain['codes'].keys():
            #print(domain['name'],code)
            arcpy.AddCodedValueToDomain_management(GDB,domain['name'],code,domain['codes'][code])
    print('[{}] Done!'.format(strftime("%Y-%m-%d %H:%M:%S")))
def AddDomainsToTable(GDB,Table,DomainCSV,replace=False):
    import os
    from time import gmtime, strftime
    import arcpy

    df = pd.DataFrame(columns=['field_name'])
    if os.path.basename(DomainCSV).split('.')[-1].lower()=='csv':
        df = pd.read_csv(DomainCSV)
    if os.path.basename(DomainCSV).split('.')[-1].lower() in ['xls','xlsx']:
        df = pd.read_excel(DomainCSV)
    def AddDomain(c):
        idx = df.field_name==c
        Table_Name= GDB+'\\Domain_{}'.format(c)
        idf = df.loc[idx][['code','description','field_type']]
        if idf.field_type.iloc[0]=='SHORT':
            idf['code'] = idf.code.astype(int)
        x = np.array(np.rec.fromrecords(idf.values))
        x.dtype.names = tuple(idf.dtypes.index.tolist())
        arcpy.Delete_management(Table_Name)
        arcpy.da.NumPyArrayToTable(x, Table_Name)
        arcpy.TableToDomain_management(Table_Name, 'code', 'description', GDB, c, df.loc[idx].field_alias.iloc[0], 'REPLACE')
        
    GDBDomains = [d.name for d in arcpy.da.ListDomains(GDB)]
    Fields = [f.name for f in arcpy.ListFields(Table)]
    for c in Fields:
        if c in df.field_name.unique():
            if replace:
                if c in GDBDomains:
                    arcpy.DeleteDomain_management(GDB, c)
                AddDomain(c)
            else:
                if not c in GDBDomains:
                    AddDomain(c)
            arcpy.AssignDomainToField_management(Table, c, c) 
            idx = df.field_name==c
            arcpy.AlterField_management(Table, c,new_field_alias= df.loc[idx].field_alias.iloc[0])
            print('[{}]  - {}'.format(strftime("%Y-%m-%d %H:%M:%S"),c))
    print('[{}] Done!'.format(strftime("%Y-%m-%d %H:%M:%S")))

class Domains(object):
    import pandas as pd
    from time import gmtime, strftime
    import datetime
    import psutil
    import os
    import arcpy
    from hsmpy31 import gdb

    t_print = lambda x:print('\x1b[31m[{}]\x1b[39m {}'.format(strftime("%Y-%m-%d %H:%M:%S"),x))
    def __init__(self, path):
        if not os.path.exists(path):
            raise ValueError('provided path [{}] do not exists'.format(path))
        self.path = path
        d = datetime.datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M:%S")
        t_print('initializing at ' + self.path + ', Last Modified: ' + d)
        self.reload()
        t_print(' - {} rows found'.format(self.Data.shape[0]))
    def reload(self):
        filename, ext = os.path.splitext(self.path)
        if ext in ['xlsx','xls']:
            self.Data = pd.read_excel(self.path)
        elif ext in ['csv']:
            self.Data = pd.read_csv(self.path)
        else:
            raise ValueError('file extension [{}] not supported'.format(ext))
    def AddDomains(self,GDB,replace=False):
        t_print('add domains')

        df = self.Data.copy(deep=True)
        df = df.groupby('field_name').agg({'field_alias':'first','field_type':'first','code':list,'description':list})
        df['codes'] = df.apply(lambda row:{int(k):v for k,v in zip(row.code,row.description)} if row.field_type=='SHORT' else {k:v for k,v in zip(row.code,row.description)},axis=1)
        df.drop(columns=['code','description'],inplace=True)
        df = df.reset_index()
        df.rename(columns={'field_name':'name','field_type':'type','field_alias':'alias'},inplace=True)

        ListDomains = df.name.tolist()
        GDBDomains = [d.name for d in arcpy.da.ListDomains(GDB)]
        for i,domain in df.iterrows():
            print(' - ' + domain['name'])
            cdf = pd.Series(domain.codes).reset_index().rename(columns={'index':'code',0:'description'})
            gdb.ConvertDFtoTable(cdf,GDB+'\\domain_temp')
            try:
                arcpy.TableToDomain_management(
                    in_table = GDB+'\\domain_temp',
                    code_field = 'code',description_field = 'description', 
                    in_workspace = GDB, 
                    domain_name = domain['name'], 
                    domain_description = domain['alias'], 
                    update_option = {True:'REPLACE',False:'APPEND'}[replace])
            except:
                pass
            arcpy.Delete_management(GDB+'\\domain_temp')
        t_print('done!')
    def AddDomainsToFields(self,GDB,FCs):
        GDBDomains = [d.name for d in arcpy.da.ListDomains(GDB)]
        not_allowed_characters = list('+ /()')
        def replace_all(text, List):
            for i in List:
                text = text.replace(i, '_')
            return text
        for fc in FCs:
            t_print(fc)
            ddf = pd.DataFrame({'field_name':[f.name for f in arcpy.ListFields(GDB + '\\' + fc)]})
            if ddf.shape[0]>0:
                ddf['AllDomains'] = [GDBDomains]*ddf.shape[0]
                ddf['Domain'] = ddf.apply(lambda row:[d for d in row.AllDomains if replace_all(d,not_allowed_characters)==row.field_name],axis=1)
                ddf = ddf[ddf.Domain.apply(len)>0]
                ddf['Domain'] = ddf['Domain'].apply(lambda x:x[0])
                for field,domain in ddf.set_index('field_name').Domain.iteritems():
                    try:
                        arcpy.AssignDomainToField_management(in_table=GDB + '\\' + fc, field_name=field, domain_name=domain)
                        print(field,domain)
                    except:
                        print('failed: ' + field + ', ' + domain)
