import pandas as pd
import numpy as np
from time import gmtime, strftime
import os
import arcgis
import arcpy

def PortalSignIn(portal='jacobs',password=''):
    p_print = lambda gis:print("Successfully logged into '{}' via the '{}' user".format(gis.properties.portalHostname,gis.properties.user.username)) 
    old_portal = "https://jeg.maps.arcgis.com"
    old_user   = 'Mahdi.Rajabi_JEG'
    new_portal = "https://jacobs.maps.arcgis.com"
    client_id = 'luXV6ayhOYcHMNQQ'

    if portal=='jeg':
        gis = arcgis.gis.GIS(old_portal,old_user,password)
        p_print(gis)        
    if portal =='jacobs':
        gis = arcgis.gis.GIS(new_portal,client_id=client_id)
        p_print(gis)        
    
    act_p = arcpy.GetActivePortalURL()
    print('Active URL: ' + act_p)
    return(gis)


def ServiceLayerToDF(URL):
    import arcpy
    import arcgis
    from arcgis.features import FeatureLayer
    #URL = 'https://geo.dot.gov/server/rest/services/Hosted/Alabama_2018_PR/FeatureServer/0'
    print('[{}] accessing: {}'.format(strftime("%Y-%m-%d %H:%M:%S"),URL))    
    layer = FeatureLayer(URL)
    q = layer.query()    
    print('[{}]  - features found: {}'.format(strftime("%Y-%m-%d %H:%M:%S"),len(q.features)))    
    print('[{}] converting features:'.format(strftime("%Y-%m-%d %H:%M:%S")))    
    L = []
    for f in q.features:
        d = f.as_dict
        s = pd.Series(d['attributes'])
        if 'geometry' in d:
            s['Shape'] = arcpy.AsShape(d['geometry'],esri_json=True)
        L.append(s)
    df = pd.DataFrame(L)
    print('[{}] done: {}'.format(strftime("%Y-%m-%d %H:%M:%S"),df.shape))    
    #df['Shape'] = df['geometry'].apply(lambda s:arcpy.AsShape(s,esri_json=True) if not pd.isnull(s) else np.NaN)
    return(df)
def URLtoJSON(URL):
    crUrl = 'https://geo.dot.gov/server/rest/services/Hosted/Alabama_2018_PR/FeatureServer/0'
    crUrl = URL
    import urllib
    from urllib.request import urlretrieve
    crValues = {'f' : 'json',
    'layers' : '0',
    'returnAttachments' : 'true',
    }
    crData = urllib.parse.urlencode(crValues).encode("utf-8")
    crRequest = urllib.request.Request(crUrl, crData)
    crResponse = urllib.request.urlopen(crRequest)
    crJson = json.load(crResponse)
    return(crJson)

# Service URL to Pandas
# def ServiceURLtoDF(URL):
#     import arcgis
#     import pandas as pd
#     fl = arcgis.features.FeatureLayer(URL)
#     L = []
#     for f in fl.query():
#         L.append(f)
#     DF = pd.DataFrame([pd.Series(l.attributes) for l in L])
#     Shape = pd.DataFrame([pd.Series(l.geometry) for l in L])
#     CDF = pd.concat([DF,Shape],axis=1)
#     return(CDF)