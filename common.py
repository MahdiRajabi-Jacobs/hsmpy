
import pandas as pd
import numpy as np
from time import gmtime, strftime
import os,sys

def AssignTiers(Array,Num_Classes):
    import jenkspy
    brks = jenkspy.jenks_breaks(Array, Num_Classes)
    nb_res = pd.cut(Array,brks,labels=['Tier{}'.format(i) for i in range(1,Num_Classes+1)])
    df = pd.DataFrame(Array)
    for lim,lim2,tier in zip(brks[:-1],brks[1:],reversed(['Tier{}'.format(i) for i in range(1,Num_Classes+1)])):
        #print(tier,lim,lim2)
        df.loc[df[Array.name]>=lim,'Tier'] = tier
        s = df[(df[Array.name]>=lim) & (df[Array.name]<=lim2)][Array.name]
        df.loc[df[Array.name]>=lim,'TierMean'] = s.mean()
        df.loc[df[Array.name]>=lim,'TierMedian'] = s.median()
        df.loc[df[Array.name]>=lim,'Limit_Left'] = lim
        df.loc[df[Array.name]>=lim,'Limit_Right'] = lim2
    return(df)

def PolylineToStreetView(Shape):
    """
    This function creates a url for Goolge StreetView based on an arcpy.Polyline object.
    The Google StreetView link location is the first point of the line and it will be headed to the Polyline's last point.
    """
    def get_bearing(lat1, long1, lat2, long2):
        dLon = (long2 - long1)
        x = math.cos(math.radians(lat2)) * math.sin(math.radians(dLon))
        y = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(math.radians(dLon))
        brng = np.arctan2(x,y)
        brng = np.degrees(brng)

        return np.round(brng,1)
    import pyproj
    geodesic = pyproj.Geod(ellps='WGS84')
    proj_pl = Shape.projectAs(hsmpy.gdb.WGS1984)
    lat1 = proj_pl.firstPoint.Y
    long1 = proj_pl.firstPoint.X
    lat2 = proj_pl.lastPoint.Y
    long2 = proj_pl.lastPoint.X
    
    #fwd_azimuth,back_azimuth,distance = geodesic.inv(lat1, long1, lat2, long2)
    fwd_azimuth = get_bearing(lat1, long1, lat2, long2)
    url = "https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={},{}&heading={}".format(lat1,long1,fwd_azimuth) # &heading=-45&pitch=38&fov=80
    return(url)