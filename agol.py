import pandas as pd
import numpy as np
from time import gmtime, strftime
import os

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
    return(df)
def DomainsFromURL(URL):
    import arcgis
    from arcgis.features import FeatureLayer
    layer = FeatureLayer(URL)
    L = []
    domains = layer._lyr_domains
    for domain in domains:
        f = list(domain.keys())[0]
        if domain[f]['type']=='codedValue':
            idf = pd.DataFrame(domain[f]['codedValues'])
            idf = idf.rename(columns={'name':'description'})
            idf['fieldname'] = f 
        L.append(idf)
    df = pd.concat(L)
    return(df)
def SigninToPortal():
    # login
    import arcgis
    client_id = 'luXV6ayhOYcHMNQQ'
    new_portal = "https://jacobs.maps.arcgis.com"
    p_print = lambda gis:print("Successfully logged into '{}' via the '{}' user".format(gis.properties.portalHostname,gis.properties.user.username)) 
    gis = arcgis.gis.GIS(new_portal,client_id=client_id,expiration=600)
    p_print(gis)