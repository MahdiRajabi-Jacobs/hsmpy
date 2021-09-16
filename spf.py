import arcpy
import sys
import os
class R2U:
    def __init__(self,row):
        self

class predict():
    def __init__(self,facility_type='',aadt=0,aadt_minor=0,length=0,driveway=0,ped_vol=0,lanes_x=0):
        self.spf = self.findspf(self,facility_type)
    def findspf(self,facility_type='',aadt=0,aadt_minor=0,length=0,driveway=0,ped_vol=0,lanes_x=0):
        return (11)


def SPF_Module(URow):
    import math

    # Define SPF Dictionary
    SPF = {
        'TOT'       : 0,        'TCk'       : 0,

        'FI'        : 0,        'FIk'       : 0,
        'FIKAB'     : 0,        'FIKABk'    : 0,

        'MVC'       : 0,        'MVCk'      : 0,
        'MVFIC'     : 0,        'MVFICk'    : 0,
        'MVPDOC'    : 0,        'MVPDOCk'   : 0,

        'MVndC'     : 0,        'MVndCk'    : 0,
        'MVFIndC'   : 0,        'MVFIndCk'  : 0,
        'MVPDOndC'  : 0,        'MVPDOndCk' : 0,

        'MVdC'      : 0,        'MVdCk'     : 0,
        'MVFIdC'    : 0,        'MVFIdCk'   : 0,
        'MVPDOdC'   : 0,        'MVPDOdCk'  : 0,

        'SVC'       : 0,        'SVCk'      : 0,
        'SVFIC'     : 0,        'SVFICk'    : 0,
        'SVPDOC'    : 0,        'SVPDOCk'   : 0,

        'Ped'       : 0,        'Pedk'      : 0}

    # Flag for Warning Message(WM)       
    Flag = False
    WM   = 'SPF Intersections ' 

    # Supported Facility Types in this Function
    SPFTypes = ['R3ST' ,'R4ST' , 'R4SG', 'RM3ST', 'RM4ST', 'RM4SG', 'U3ST', 'U4ST', 'U3SG', 'U4SG', 'U2U', 'U3T', 'U4U', 'U4D', 'U5T']

    # Reading & Checking the type
    iType    = GetVal(URow,F_FType['name'],'')
    if not iType in FTypes:
        Flag = True; WM += ', Type(' + str(iType) + ') not supported'

    # Reading Input Info: AADT, Segment Length, Driveway Info, ...
    AADT_Major = GetVal(URow,F_AADT_Major['name'])
    AADT_Minor = GetVal(URow,F_AADT_Minor['name'])
    AADT       = GetVal(URow,F_AADT      ['name'])
    Length     = URow.getValue('Shape').length

    dMjC       = GetVal(URow,F_dMjC['name'])
    dMnC       = GetVal(URow,F_dMnC['name'])
    dMjI       = GetVal(URow,F_dMjI['name'])
    dMnI       = GetVal(URow,F_dMnI['name'])
    dMjR       = GetVal(URow,F_dMjR['name'])
    dMnR       = GetVal(URow,F_dMnR['name'])
    dO         = GetVal(URow,F_dO  ['name'])

    PedVol     = GetVal(URow,'Ped_Vol', 1)
    PedVol     = GetVal(URow,F_PED_VOL['name'],PedVol)

    LanesX     = GetVal(URow,'Lanesx', 2)
    LanesX     = GetVal(URow,F_LANESX['name'],LanesX)

    # Checking AADT
    if AADT_Major == 0 and AADT_Minor == 0 and AADT == 0:
        Flag = True; WM += ', Zero AADT'

    # Checking Length
    if Length == 0 and not iType in ITypes:
        Flag = True; WM += ', Zero segment Length'

    # Ln's and Ped Volumes conversion

    a   = NaturalLog(AADT_Major)
    b   = NaturalLog(AADT_Minor)
    ab  = NaturalLog(AADT_Major + AADT_Minor)
    if AADT_Major <> 0:
        boa = NaturalLog(float(AADT_Minor) / AADT_Major)
    else:
        boa = 0
    c   = NaturalLog(AADT)
    d   = NaturalLog(Length/5280)

    p = 0
    if PedVol == 1: p = 20
    if PedVol == 2: p = 120
    if PedVol == 3: p = 400
    if PedVol == 4: p = 7500
    if PedVol == 5: p = 1700
    if p <> 0: p = NaturalLog(p)

    # SPF Calculations based on types
    if iType ==  "R3ST":
        if AADT_Major > 19500 or AADT_Major <= 0: Flag = True; WM += ', AADT Major(' + str(AADT_Major) + ') for FID = ' + str(GetVal(URow,"FID")) + ' not in range'
        if AADT_Minor > 4300  or AADT_Minor <= 0: Flag = True; WM += ', AADT Minor(' + str(AADT_Minor) + ') for FID = ' + str(GetVal(URow,"FID")) + ' not in range'
        SPF['TOT']  = math.exp(-9.86 + 0.79 * a + 0.49 * b)
        SPF['TCk'] = 0.54    

    if iType ==  "R4ST":
        if AADT_Major > 14700 or AADT_Major <= 0: Flag = True; WM += ', AADT Major(' + str(AADT_Major) + ') for FID = ' + str(GetVal(URow,"FID")) + ' not in range'
        if AADT_Minor > 3500  or AADT_Minor <= 0: Flag = True; WM += ', AADT Minor(' + str(AADT_Minor) + ') for FID = ' + str(GetVal(URow,"FID")) + ' not in range'
        SPF['TOT']  = math.exp(-8.56 + 0.6 * a + 0.61 * b)
        SPF['TCk'] = 0.24    

    if iType ==  "R4SG":
        if AADT_Major > 25200 or AADT_Major <= 0: Flag = True; WM += ', AADT Major(' + str(AADT_Major) + ') for FID = ' + str(GetVal(URow,"FID")) + ' not in range'
        if AADT_Minor > 12500 or AADT_Minor <= 0: Flag = True; WM += ', AADT Minor(' + str(AADT_Minor) + ') for FID = ' + str(GetVal(URow,"FID")) + ' not in range'
        SPF['TOT']  = math.exp(-5.13 + 0.6 * a + 0.2 * b)
        SPF['TCk'] = 0.11    

    if iType ==  "RM3ST":
        if AADT_Major > 78300 or AADT_Major <= 0: Flag = True; WM += ', AADT Major(' + str(AADT_Major) + ') for FID = ' + str(GetVal(URow,"FID")) + ' not in range'
        if AADT_Minor > 23000 or AADT_Minor <= 0: Flag = True; WM += ', AADT Minor(' + str(AADT_Minor) + ') for FID = ' + str(GetVal(URow,"FID")) + ' not in range'
        SPF['TOT']     = math.exp(-12.526 + 1.204 * a + 0.236 * b)
        SPF['TCk']    = 0.46    
        SPF['FI']     = math.exp(-12.664 + 1.107 * a + 0.272 * b)
        SPF['FIk']    = 0.569
        SPF['FIKAB']  = math.exp(-11.989 + 1.013 * a + 0.228 * b)
        SPF['FIKABk'] = 0.566    
   
    if iType ==  "RM4ST":
        if AADT_Major > 78300 or AADT_Major <= 0: Flag = True; WM += ', AADT Major(' + str(AADT_Major) + ') for FID = ' + str(GetVal(URow,"FID")) + ' not in range'
        if AADT_Minor > 7400  or AADT_Minor <= 0: Flag = True; WM += ', AADT Minor(' + str(AADT_Minor) + ') for FID = ' + str(GetVal(URow,"FID")) + ' not in range'
        SPF['TOT']     = math.exp(-10.008 + 0.848 * a + 0.448 * b)
        SPF['TCk']    = 0.494    
        SPF['FI']     = math.exp(-11.554 + 0.888 * a + 0.525 * b)
        SPF['FIk']    = 0.742
        SPF['FIKAB']  = math.exp(-10.734 + 0.828 * a + 0.412 * b)
        SPF['FIKABk'] = 0.655  

    if iType ==  "RM4SG":
        if AADT_Major > 43500 or AADT_Major <= 0: Flag = True; WM += ', AADT Major(' + str(AADT_Major) + ') for FID = ' + str(GetVal(URow,"FID")) + ' not in range'
        if AADT_Minor > 18500 or AADT_Minor <= 0: Flag = True; WM += ', AADT Minor(' + str(AADT_Minor) + ') for FID = ' + str(GetVal(URow,"FID")) + ' not in range'
        SPF['TOT']     = math.exp(-7.182 + 0.722 * a + 0.337 * b)
        SPF['TCk']    = 0.277    
        SPF['FI']     = math.exp(-6.393 + 0.638 * a + 0.232 * b)
        SPF['FIk']    = 0.218
        SPF['FIKAB']  = math.exp(-12.011 + 1.279 * ab)
        SPF['FIKABk'] = 0.566  
    
    if iType ==  "U3ST":
        if AADT_Major > 45700 or AADT_Major <= 0: Flag = True; WM += ', AADT Major(' + str(AADT_Major) + ') for FID = ' + str(GetVal(URow,"FID")) + ' not in range'
        if AADT_Minor > 9300  or AADT_Minor <= 0: Flag = True; WM += ', AADT Minor(' + str(AADT_Minor) + ') for FID = ' + str(GetVal(URow,"FID")) + ' not in range'
        SPF['MVC']     = math.exp(-13.36 + 1.11 * a + 0.41 * b)
        SPF['MVCk']    = 0.8
        SPF['MVFIC']   = math.exp(-14.01 + 1.16 * a + 0.3 * b)
        SPF['MVFICk']  = 0.69
        SPF['MVPDOC']  = math.exp(-15.38 + 1.2 * a + 0.51 * b)
        SPF['MVPDOCk'] = 0.77
        SPF['MVFIC']   = SPF['MVC'] * SPF['MVFIC'] / (SPF['MVFIC'] + SPF['MVPDOC'])
        SPF['MVPDOC']  = SPF['MVC'] - SPF['MVFIC']

        SPF['SVC']     = math.exp(-6.81 + 0.16 * a + 0.51 * b)
        SPF['SVCk']    = 1.14
        SPF['SVFIC']   = 0.31 * SPF['SVC']
        SPF['SVFICk']  = SPF['SVCk']
        SPF['SVPDOC']  = math.exp(-8.36 + 0.25 * a + 0.55 * b)
        SPF['SVPDOCk'] = 1.29
        SPF['SVFIC']   = SPF['SVC'] * SPF['SVFIC'] / (SPF['SVFIC'] + SPF['SVPDOC'])
        SPF['SVPDOC']  = SPF['SVC'] - SPF['SVFIC']
        
    if iType == "U4ST":
        if AADT_Major > 58100 or AADT_Major <= 0: Flag = True; WM += ', AADT Major(' + str(AADT_Major) + ') for FID = ' + str(GetVal(URow,"FID")) + ' not in range'
        if AADT_Minor > 16400 or AADT_Minor <= 0: Flag = True; WM += ', AADT Minor(' + str(AADT_Minor) + ') for FID = ' + str(GetVal(URow,"FID")) + ' not in range'
        
        SPF['MVC']     = math.exp(-8.9 + 0.82 * a + 0.25 * b)
        SPF['MVCk']    = 0.4
        SPF['MVFIC']   = math.exp(-11.13 + 0.93 * a + 0.28 * b)
        SPF['MVFICk']  = 0.48
        SPF['MVPDOC']  = math.exp(-8.74 + 0.77 * a + 0.23 * b)
        SPF['MVPDOCk'] = 0.4
        SPF['MVFIC']   = SPF['MVC'] * SPF['MVFIC'] / (SPF['MVFIC'] + SPF['MVPDOC'])
        SPF['MVPDOC']  = SPF['MVC'] - SPF['MVFIC']

        SPF['SVC']     = math.exp(-5.33 + 0.33 * a + 0.12 * b)
        SPF['SVCk']    = 0.65
        SPF['SVFIC']   = 0.28 * SPF['SVC']
        SPF['SVFICk']  = SPF['SVCk']
        SPF['SVPDOC']  = math.exp(-7.04 + 0.36 * a + 0.25 * b)
        SPF['SVPDOCk'] = 0.54
        SPF['SVFIC']   = SPF['SVC'] * SPF['SVFIC'] / (SPF['SVFIC'] + SPF['SVPDOC'])
        SPF['SVPDOC']  = SPF['SVC'] - SPF['SVFIC']
       
    if iType == "U3SG":
        if AADT_Major > 46800 or AADT_Major <= 0: Flag = True; WM += ', AADT Major(' + str(AADT_Major) + ') for FID = ' + str(GetVal(URow,"FID")) + ' not in range'
        if AADT_Minor > 5900  or AADT_Minor <= 0: Flag = True; WM += ', AADT Minor(' + str(AADT_Minor) + ') for FID = ' + str(GetVal(URow,"FID")) + ' not in range'
        
        SPF['MVC']     = math.exp(-12.13 + 1.11 * a + 0.26 * b)
        SPF['MVCk']    = 0.33
        SPF['MVFIC']   = math.exp(-11.58 + 1.02 * a + 0.17 * b)
        SPF['MVFICk']  = 0.3
        SPF['MVPDOC']  = math.exp(-13.24 + 1.14 * a + 0.3 * b)
        SPF['MVPDOCk'] = 0.36
        SPF['MVFIC']   = SPF['MVC'] * SPF['MVFIC'] / (SPF['MVFIC'] + SPF['MVPDOC'])
        SPF['MVPDOC']  = SPF['MVC'] - SPF['MVFIC']

        SPF['SVC']     = math.exp(-9.02 + 0.42 * a + 0.4 * b)
        SPF['SVCk']    = 0.36
        SPF['SVFIC']   = math.exp(-9.75 + 0.27 * a + 0.51 * b)
        SPF['SVFICk']  = 0.24
        SPF['SVPDOC']  = math.exp(-9.08 + 0.45 * a + 0.33 * b)
        SPF['SVPDOCk'] = 0.53
        SPF['SVFIC']   = SPF['SVC'] * SPF['SVFIC'] / (SPF['SVFIC'] + SPF['SVPDOC'])
        SPF['SVPDOC']  = SPF['SVC'] - SPF['SVFIC']

        SPF['Ped']     = math.exp(-6.6 + 0.05 * ab + 0.24 * boa + 0.41 * p + 0.09 * LanesX)
        SPF['Pedk']    = 0.52
    
    if iType == "U4SG":
        if AADT_Major > 67700 or AADT_Major <= 0: Flag = True; WM += ', AADT Major(' + str(AADT_Major) + ') for FID = ' + str(GetVal(URow,"FID")) + ' not in range'
        if AADT_Minor > 33400 or AADT_Minor <= 0: Flag = True; WM += ', AADT Minor(' + str(AADT_Minor) + ') for FID = ' + str(GetVal(URow,"FID")) + ' not in range'
        
        SPF['MVC']     = math.exp(-10.99 + 1.07 * a + 0.23 * b)
        SPF['MVCk']    = 0.39
        SPF['MVFIC']   = math.exp(-13.14 + 1.18 * a + 0.22 * b)
        SPF['MVFICk']  = 0.33
        SPF['MVPDOC']  = math.exp(-11.02 + 1.02 * a + 0.24 * b)
        SPF['MVPDOCk'] = 0.44
        SPF['MVFIC']   = SPF['MVC'] * SPF['MVFIC'] / (SPF['MVFIC'] + SPF['MVPDOC'])
        SPF['MVPDOC']  = SPF['MVC'] - SPF['MVFIC']

        SPF['SVC']     = math.exp(-10.21 + 0.68 * a + 0.27 * b)
        SPF['SVCk']    = 0.36
        SPF['SVFIC']   = math.exp(-9.25 + 0.43 * a + 0.29 * b)
        SPF['SVFICk']  = 0.09
        SPF['SVPDOC']  = math.exp(-11.34 + 0.78 * a + 0.25 * b)
        SPF['SVPDOCk'] = 0.44
        SPF['SVFIC']   = SPF['SVC'] * SPF['SVFIC'] / (SPF['SVFIC'] + SPF['SVPDOC'])
        SPF['SVPDOC']  = SPF['SVC'] - SPF['SVFIC']

        SPF['Ped']     = math.exp(-9.53 + 0.4 * ab + 0.26 * boa + 0.45 * p + 0.04 * LanesX)
        SPF['Pedk']    = 0.24

    if iType == "R2U":
        if AADT > 17800 or AADT <= 0: Flag = True; WM += ', AADT (' + str(AADT) + ') for FID = ' + str(GetVal(URow,"FID")) + ' not in range'
        SPF['TOT'] = AADT * Length/5280 * 365 * math.exp(-0.312) / 1000000.0
        SPF['TCk'] = 0.236/(Length/5280)

    if iType == "R4U":
        if AADT > 33200 or AADT <= 0: Flag = True; WM += ', AADT (' + str(AADT) + ') for FID = ' + str(GetVal(URow,"FID")) + ' not in range'
        SPF['TOT'] = math.exp(-9.653 + 1.176 * c + d)
        SPF['TCk'] = 1/math.exp(1.675 + d)
        SPF['FI']  = math.exp(-9.410 + 1.094 * c + d)
        SPF['FIk'] = 1/math.exp(1.796 + d)

    if iType == "R4D":
        if AADT > 89300 or AADT <= 0: Flag = True; WM += ', AADT (' + str(AADT) + ') for FID = ' + str(GetVal(URow,"FID")) + ' not in range'
        SPF['TOT'] = math.exp(-9.025 + 1.049 * c + d)
        SPF['TCk'] = 1/math.exp(1.549 + d)
        SPF['FI']  = math.exp(-8.837 + 0.958 * c + d)
        SPF['FIk'] = 1/math.exp(1.687 + d)

    if iType == "U2U":
        if AADT > 32600 or AADT <= 0: Flag = True; WM += ', AADT (' + str(AADT) + ') for FID = ' + str(GetVal(URow,"FID")) + ' not in range'

        SPF['MVndC']     = math.exp(-15.22 + 1.68 * c + d)
        SPF['MVndCk']    = 0.84
        SPF['MVFIndC']   = math.exp(-16.22 + 1.66 * c + d)
        SPF['MVFIndCk']  = 0.65
        SPF['MVPDOndC']  = math.exp(-15.62 + 1.69 * a + d)
        SPF['MVPDOndCk'] = 0.87
        SPF['MVFIndC']   = SPF['MVndC'] * SPF['MVFIndC'] / (SPF['MVFIndC'] + SPF['MVPDOndC'])
        SPF['MVPDOndC']  = SPF['MVndC'] - SPF['MVFIndC']

        SPF['SVC']      = math.exp(-5.47 + 0.56 * c + d)
        SPF['SVCk']     = 0.81
        SPF['SVFIC']    = math.exp(-3.96 + 0.23 * c + d)
        SPF['SVFICk']   = 0.50
        SPF['SVPDOC']   = math.exp(-6.51 + 0.64 * c + d)
        SPF['SVPDOCk']  = 0.87
        SPF['SVFIC']    = SPF['SVC'] * SPF['SVFIC'] / (SPF['SVFIC'] + SPF['SVPDOC'])
        SPF['SVPDOC']   = SPF['SVC'] - SPF['SVFIC']

        SPF['MVdC']     = ((0.158 * dMjC) + (0.050 * dMnC) + (0.172 * dMjI) + (0.023 * dMnI) + (0.083 * dMjR) + (0.016 * dMnR) + (0.025 * dO)) * (AADT/15000)**1.00 
        SPF['MVdCk']    = 0.81
        SPF['MVFIdC']   = 0.323 * SPF['MVdC']
        SPF['MVFIdCk']  = SPF['MVdCk']
        SPF['MVPDOdC']  = 0.677 * SPF['MVdC']
        SPF['MVPDOdCk'] = SPF['MVdCk']

        SPF['TOT']     = SPF['MVndC']    + SPF['MVdC']    + SPF['SVC']
        SPF['FI']      = SPF['MVFIndC']  + SPF['MVFIdC']  + SPF['SVFIC']

        SPF['MVC']     = SPF['MVndC']    + SPF['MVdC']
        SPF['MVFIC']   = SPF['MVFIndC']  + SPF['MVFIdC']
        SPF['MVPDOC']  = SPF['MVPDOndC'] + SPF['MVPDOdC']

    if iType == "U3T":
        if AADT > 32900 or AADT <= 0: Flag = True; WM += ', AADT (' + str(AADT) + ') for FID = ' + str(GetVal(URow,"FID")) + ' not in range'

        SPF['MVndC']     = math.exp(-12.40 + 1.41 * c + d)
        SPF['MVndCk']    = 0.66
        SPF['MVFIndC']   = math.exp(-16.45 + 1.69 * c + d)
        SPF['MVFIndCk']  = 0.59
        SPF['MVPDOndC']  = math.exp(-11.95 + 1.33 * a + d)
        SPF['MVPDOndCk'] = 0.59
        SPF['MVFIndC']   = SPF['MVndC'] * SPF['MVFIndC'] / (SPF['MVFIndC'] + SPF['MVPDOndC'])
        SPF['MVPDOndC']  = SPF['MVndC'] - SPF['MVFIndC']

        SPF['SVC']     = math.exp(-5.74 + 0.54 * c + d)
        SPF['SVCk']    = 1.37
        SPF['SVFIC']   = math.exp(-6.37 + 0.47 * c + d)
        SPF['SVFICk']  = 1.06
        SPF['SVPDOC']  = math.exp(-6.29 + 0.56 * c + d)
        SPF['SVPDOCk'] = 1.93
        SPF['SVFIC']   = SPF['SVC'] * SPF['SVFIC'] / (SPF['SVFIC'] + SPF['SVPDOC'])
        SPF['SVPDOC']  = SPF['SVC'] - SPF['SVFIC']

        SPF['MVdC']     = ((0.102 * dMjC) + (0.032 * dMnC) + (0.110 * dMjI) + (0.015 * dMnI) + (0.053 * dMjR) + (0.010 * dMnR) + (0.016 * dO)) * (AADT/15000)**1.000 
        SPF['MVdCk']    = 1.10
        SPF['MVFIdC']   = 0.243 * SPF['MVdC']
        SPF['MVFIdCk']  = SPF['MVdCk']
        SPF['MVPDOdC']  = 0.757 * SPF['MVdC']
        SPF['MVPDOdCk'] = SPF['MVdCk']

        SPF['TOT']     = SPF['MVndC']    + SPF['MVdC']    + SPF['SVC']
        SPF['FI']      = SPF['MVFIndC']  + SPF['MVFIdC']  + SPF['SVFIC']

        SPF['MVC']     = SPF['MVndC']    + SPF['MVdC']
        SPF['MVFIC']   = SPF['MVFIndC']  + SPF['MVFIdC']
        SPF['MVPDOC']  = SPF['MVPDOndC'] + SPF['MVPDOdC']

    if iType == "U4U":
        if AADT > 40100 or AADT <= 0: Flag = True; WM += ', AADT (' + str(AADT) + ') for FID = ' + str(GetVal(URow,"FID")) + ' not in range'

        SPF['MVndC']     = math.exp(-11.63 + 1.33 * c + d)
        SPF['MVndCk']    = 1.01
        SPF['MVFIndC']   = math.exp(-12.08 + 1.25 * c + d)
        SPF['MVFIndCk']  = 0.99
        SPF['MVPDOndC']  = math.exp(-12.53 + 1.38 * a + d)
        SPF['MVPDOndCk'] = 1.08
        SPF['MVFIndC']   = SPF['MVndC'] * SPF['MVFIndC'] / (SPF['MVFIndC'] + SPF['MVPDOndC'])
        SPF['MVPDOndC']  = SPF['MVndC'] - SPF['MVFIndC']

        SPF['SVC']     = math.exp(-7.99 + 0.81 * c + d)
        SPF['SVCk']    = 0.91
        SPF['SVFIC']   = math.exp(-7.37 + 0.61 * c + d)
        SPF['SVFICk']  = 0.54
        SPF['SVPDOC']  = math.exp(-8.50 + 0.84 * c + d)
        SPF['SVPDOCk'] = 0.97
        SPF['SVFIC']   = SPF['SVC'] * SPF['SVFIC'] / (SPF['SVFIC'] + SPF['SVPDOC'])
        SPF['SVPDOC']  = SPF['SVC'] - SPF['SVFIC']

        SPF['MVdC']     = ((0.182 * dMjC) + (0.058 * dMnC) + (0.198 * dMjI) + (0.026 * dMnI) + (0.096 * dMjR) + (0.018 * dMnR) + (0.029 * dO)) * (AADT/15000)**1.172 
        SPF['MVdCk']    = 0.81
        SPF['MVFIdC']   = 0.342 * SPF['MVdC']
        SPF['MVFIdCk']  = SPF['MVdCk']
        SPF['MVPDOdC']  = 0.658 * SPF['MVdC']
        SPF['MVPDOdCk'] = SPF['MVdCk']

        SPF['TOT']     = SPF['MVndC']    + SPF['MVdC']    + SPF['SVC']
        SPF['FI']      = SPF['MVFIndC']  + SPF['MVFIdC']  + SPF['SVFIC']

        SPF['MVC']     = SPF['MVndC']    + SPF['MVdC']
        SPF['MVFIC']   = SPF['MVFIndC']  + SPF['MVFIdC']
        SPF['MVPDOC']  = SPF['MVPDOndC'] + SPF['MVPDOdC']

    if iType == "U4D":
        if AADT > 66000 or AADT <= 0: Flag = True; WM += ', AADT (' + str(AADT) + ') for FID = ' + str(GetVal(URow,"FID")) + ' not in range'

        SPF['MVndC']     = math.exp(-12.34 + 1.36 * c + d)
        SPF['MVndCk']    = 1.32
        SPF['MVFIndC']   = math.exp(-12.76 + 1.28 * c + d)
        SPF['MVFIndCk']  = 1.31
        SPF['MVPDOndC']  = math.exp(-12.81 + 1.38 * a + d)
        SPF['MVPDOndCk'] = 1.34
        SPF['MVFIndC']   = SPF['MVndC'] * SPF['MVFIndC'] / (SPF['MVFIndC'] + SPF['MVPDOndC'])
        SPF['MVPDOndC']  = SPF['MVndC'] - SPF['MVFIndC']

        SPF['SVC']     = math.exp(-5.05 + 0.47 * c + d)
        SPF['SVCk']    = 0.86
        SPF['SVFIC']   = math.exp(-8.71 + 0.66 * c + d)
        SPF['SVFICk']  = 0.28
        SPF['SVPDOC']  = math.exp(-5.04 + 0.45 * c + d)
        SPF['SVPDOCk'] = 1.06
        SPF['SVFIC']   = SPF['SVC'] * SPF['SVFIC'] / (SPF['SVFIC'] + SPF['SVPDOC'])
        SPF['SVPDOC']  = SPF['SVC'] - SPF['SVFIC']

        SPF['MVdC']     = ((0.033 * dMjC) + (0.011 * dMnC) + (0.036 * dMjI) + (0.005 * dMnI) + (0.018 * dMjR) + (0.003 * dMnR) + (0.005 * dO)) * (AADT/15000)**1.106 
        SPF['MVdCk']    = 1.39
        SPF['MVFIdC']   = 0.284 * SPF['MVdC']
        SPF['MVFIdCk']  = SPF['MVdCk']
        SPF['MVPDOdC']  = 0.716 * SPF['MVdC']
        SPF['MVPDOdCk'] = SPF['MVdCk']

        SPF['TOT']     = SPF['MVndC']    + SPF['MVdC']    + SPF['SVC']
        SPF['FI']      = SPF['MVFIndC']  + SPF['MVFIdC']  + SPF['SVFIC']

        SPF['MVC']     = SPF['MVndC']    + SPF['MVdC']
        SPF['MVFIC']   = SPF['MVFIndC']  + SPF['MVFIdC']
        SPF['MVPDOC']  = SPF['MVPDOndC'] + SPF['MVPDOdC']

    if iType == "U5T":
        if AADT > 53800 or AADT <= 0: Flag = True; WM += ', AADT (' + str(AADT) + ') for FID = ' + str(GetVal(URow,"FID")) + ' not in range'

        SPF['MVndC']     = math.exp(-9.70 + 1.17 * c + d)
        SPF['MVndCk']    = 0.81
        SPF['MVFIndC']   = math.exp(-10.47 + 1.12 * c + d)
        SPF['MVFIndCk']  = 0.62
        SPF['MVPDOndC']  = math.exp(-9.97 + 1.17 * a + d)
        SPF['MVPDOndCk'] = 0.88
        SPF['MVFIndC']   = SPF['MVndC'] * SPF['MVFIndC'] / (SPF['MVFIndC'] + SPF['MVPDOndC'])
        SPF['MVPDOndC']  = SPF['MVndC'] - SPF['MVFIndC']

        SPF['SVC']     = math.exp(-4.82 + 0.54 * c + d)
        SPF['SVCk']    = 0.52
        SPF['SVFIC']   = math.exp(-4.43 + 0.35 * c + d)
        SPF['SVFICk']  = 0.36
        SPF['SVPDOC']  = math.exp(-5.83 + 0.61 * c + d)
        SPF['SVPDOCk'] = 0.55
        SPF['SVFIC']   = SPF['SVC'] * SPF['SVFIC'] / (SPF['SVFIC'] + SPF['SVPDOC'])
        SPF['SVPDOC']  = SPF['SVC'] - SPF['SVFIC']

        SPF['MVdC']     = ((0.165 * dMjC) + (0.053 * dMnC) + (0.181 * dMjI) + (0.024 * dMnI) + (0.087 * dMjR) + (0.016 * dMnR) + (0.027 * dO)) * (AADT/15000)**1.172 
        SPF['MVdCk']    = 0.10
        SPF['MVFIdC']   = 0.269 * SPF['MVdC']
        SPF['MVFIdCk']  = SPF['MVdCk']
        SPF['MVPDOdC']  = 0.731 * SPF['MVdC']
        SPF['MVPDOdCk'] = SPF['MVdCk']

        SPF['TOT']     = SPF['MVndC']    + SPF['MVdC']    + SPF['SVC']
        SPF['FI']      = SPF['MVFIndC']  + SPF['MVFIdC']  + SPF['SVFIC']

        SPF['MVC']     = SPF['MVndC']    + SPF['MVdC']
        SPF['MVFIC']   = SPF['MVFIndC']  + SPF['MVFIdC']
        SPF['MVPDOC']  = SPF['MVPDOndC'] + SPF['MVPDOdC']

    if iType == "R4F":
        if AADT > 73000 or AADT <= 0: Flag = True; WM += ', AADT (' + str(AADT) + ') for FID = ' + str(GetVal(URow,"FID")) + ' not in range'
        el = Length/5280
        SPF['MVFIC']   = el * math.exp(-5.975 + 1.492 * NaturalLog(AADT/1000))
        SPF['MVFICk']  = 1/(17.6*el)
        SPF['MVPDOC']  = el * math.exp(-6.880 + 1.936 * NaturalLog(AADT/1000))
        SPF['MVPDOCk'] = 1/(18.8*el)
        SPF['MVC']     = SPF['MVFIC'] + SPF['MVPDOC']

        SPF['SVFIC']   = el * math.exp(-2.126 + 0.646 * NaturalLog(AADT/1000))
        SPF['SVFICk']  = 1/(30.1*el)
        SPF['SVPDOC']  = el * math.exp(-2.235 + 0.876 * NaturalLog(AADT/1000))
        SPF['SVPDOCk'] = 1/(20.7*el)
        SPF['SVC']     = SPF['SVFIC'] + SPF['SVPDOC']

        SPF['TOT']     = SPF['MVC']    + SPF['SVC']
        SPF['FI']      = SPF['MVFIC']  + SPF['SVFIC']

    if iType == "U4F":
        if AADT > 110000 or AADT <= 0: Flag = True; WM += ', AADT (' + str(AADT) + ') for FID = ' + str(GetVal(URow,"FID")) + ' not in range'
        el = Length/5280

        SPF['MVFIC']   = el * math.exp(-5.470 + 1.492 * NaturalLog(AADT/1000))
        SPF['MVFICk']  = 1/(17.6*el)
        SPF['MVPDOC']  = el * math.exp(-6.548 + 1.936 * NaturalLog(AADT/1000))
        SPF['MVPDOCk'] = 1/(18.8*el)
        SPF['MVC']     = SPF['MVFIC'] + SPF['MVPDOC']

        SPF['SVFIC']   = el * math.exp(-2.126 + 0.646 * NaturalLog(AADT/1000))
        SPF['SVFICk']  = 1/(30.1*el)
        SPF['SVPDOC']  = el * math.exp(-2.235 + 0.876 * NaturalLog(AADT/1000))
        SPF['SVPDOCk'] = 1/(20.7*el)
        SPF['SVC']     = SPF['SVFIC'] + SPF['SVPDOC']

        SPF['TOT']     = SPF['MVC']    + SPF['SVC']
        SPF['FI']      = SPF['MVFIC']  + SPF['SVFIC']

    if iType == "U6F":
        if AADT > 180000 or AADT <= 0: Flag = True; WM += ', AADT (' + str(AADT) + ') for FID = ' + str(GetVal(URow,"FID")) + ' not in range'
        el = Length/5280

        SPF['MVFIC']   = el * math.exp(-5.587 + 1.492 * NaturalLog(AADT/1000))
        SPF['MVFICk']  = 1/(17.6*el)
        SPF['MVPDOC']  = el * math.exp(-6.809 + 1.936 * NaturalLog(AADT/1000))
        SPF['MVPDOCk'] = 1/(18.8*el)
        SPF['MVC']     = SPF['MVFIC'] + SPF['MVPDOC']

        SPF['SVFIC']   = el * math.exp(-2.055 + 0.646 * NaturalLog(AADT/1000))
        SPF['SVFICk']  = 1/(30.1*el)
        SPF['SVPDOC']  = el * math.exp(-2.274 + 0.876 * NaturalLog(AADT/1000))
        SPF['SVPDOCk'] = 1/(20.7*el)
        SPF['SVC']     = SPF['SVFIC'] + SPF['SVPDOC']

        SPF['TOT']     = SPF['MVC']    + SPF['SVC']
        SPF['FI']      = SPF['MVFIC']  + SPF['SVFIC']

    # Warning Message if flagged
    if Flag: arcpy.AddWarning(WM)
    return SPF
## CMFs
def CMFSkew(URow):
    import math

    Flag = False
    WM   = 'CMF Skew Angle ' 

    CMF_Skew = {"TOT": 1,"FI": 1}

    iType    = GetVal(URow,F_FType['name'])
    if not iType in FTypes:
        Flag = True; WM += ', Type(' + str(iType) + ') not supported'

    Skew1 = GetVal(URow,'Skew1')
    Skew1 = GetVal(URow,F_SKEW1['name'],Skew1)

    Skew2 = GetVal(URow,'Skew2')
    Skew2 = GetVal(URow,F_SKEW2['name'],Skew2)


    if iType == "R3ST":
        if Skew1 >= 90 or Skew1 < 0: Flag = True; WM += ', Skew Angle(' + str(Skew1) + ') for FID = ' + str(GetVal(URow,"FID")) + ' not in range'
        CMF_Skew['TOT'] = math.exp(0.004 * Skew1)
        
    if iType == "R4ST":
        if Skew2 >= 90 or Skew2 < 0: Flag = True; WM += ', Skew Angle(' + str(Skew2) + ') for FID = ' + str(GetVal(URow,"FID")) + ' not in range'
        a = math.exp(0.0054 * Skew1)
        b = math.exp(0.0054 * Skew2)
        CMF_Skew['TOT'] = (a + b) / 2
        
    if iType == "RM3ST":
        if Skew1 >= 90 or Skew1 < 0: Flag = True; WM += ', Skew Angle(' + str(Skew1) + ') for FID = ' + str(GetVal(URow,"FID")) + ' not in range'
        CMF_Skew['TOT'] = 0.016 * Skew1 / (0.98 + 0.016 * Skew1) + 1
        CMF_Skew['FI' ] = 0.017 * Skew1 / (0.52 + 0.017 * Skew1) + 1
        
    if iType == "RM4ST":
        if Skew2 >= 90 or Skew2 < 0: Flag = True; WM += ', Skew Angle(' + str(Skew2) + ') for FID = ' + str(GetVal(URow,"FID")) + ' not in range'
        a = 0.053 * Skew1 / (1.43 + 0.053 * Skew1) + 1
        b = 0.053 * Skew1 / (1.43 + 0.053 * Skew1) + 1
        CMF_Skew['TOT'] = (a + b) / 2
        a = 0.048 * Skew1 / (0.72 + 0.048 * Skew1) + 1
        b = 0.048 * Skew1 / (0.72 + 0.048 * Skew1) + 1
        CMF_Skew['FI'] = (a + b) / 2

    if Flag: arcpy.AddWarning(WM)
    return CMF_Skew
def CMFLTL(URow):

    Flag = False
    WM   = 'CMF LT Lane ' 

    iType    = GetVal(URow,F_FType['name'])
    if not iType in FTypes:
        Flag = True; WM += ', Type(' + str(iType) + ') not supported'

    LTL = GetVal(URow,F_LTL['name'])

    CMF_LTL = {
        "TOT": 1,
        "FI": 1}

    if iType == "R3ST":
        if LTL > 1 or LTL < 0: Flag = True; WM += ', LTL (' + str(LTL) + ') for FID = ' + str(GetVal(URow,"FID")) + ' not in range'
        CMF_LTL['TOT'] = 0.56 ** LTL
    
    if iType == "R4ST":
        if LTL > 2 or LTL < 0: Flag = True; WM += ', LTL (' + str(LTL) + ') for FID = ' + str(GetVal(URow,"FID")) + ' not in range'
        CMF_LTL['TOT'] = 0.72 ** LTL
    
    if iType == "R4SG":
        if LTL > 4 or LTL < 0: Flag = True; WM += ', LTL (' + str(LTL) + ') for FID = ' + str(GetVal(URow,"FID")) + ' not in range'
        CMF_LTL['TOT'] = 0.82 ** LTL
    
    if iType == "RM3ST":
        if LTL > 1 or LTL < 0: Flag = True; WM += ', LTL (' + str(LTL) + ') for FID = ' + str(GetVal(URow,"FID")) + ' not in range'
        CMF_LTL['TOT'] = 0.56 ** LTL
        CMF_LTL['FI' ] = 0.45 ** LTL

    if iType == "RM4ST":
        if LTL > 2 or LTL < 0: Flag = True; WM += ', LTL (' + str(LTL) + ') for FID = ' + str(GetVal(URow,"FID")) + ' not in range'
        CMF_LTL['TOT'] = 0.72 ** LTL
        CMF_LTL['FI' ] = 0.65 ** LTL

    if iType == "U3ST":
        if LTL > 1 or LTL < 0: Flag = True; WM += ', LTL (' + str(LTL) + ') for FID = ' + str(GetVal(URow,"FID")) + ' not in range'
        CMF_LTL['TOT'] = 0.67 ** LTL

    if iType == "U3SG":
        if LTL > 2 or LTL < 0: Flag = True; WM += ', LTL (' + str(LTL) + ') for FID = ' + str(GetVal(URow,"FID")) + ' not in range'
        CMF_LTL['TOT'] = 0.93 ** LTL

    if iType == "U4ST":
        if LTL > 2 or LTL < 0: Flag = True; WM += ', LTL (' + str(LTL) + ') for FID = ' + str(GetVal(URow,"FID")) + ' not in range'
        CMF_LTL['TOT'] = 0.73 ** LTL

    if iType == "U4SG":
        if LTL > 4 or LTL < 0: Flag = True; WM += ', LTL (' + str(LTL) + ') for FID = ' + str(GetVal(URow,"FID")) + ' not in range'
        CMF_LTL['TOT'] = 0.9 ** LTL

    if Flag: arcpy.AddWarning(WM)
    return CMF_LTL
def CMFRTL(URow):

    Flag = False
    WM   = 'CMF RT Lane ' 

    iType    = GetVal(URow,F_FType['name'])
    if not iType in FTypes:
        Flag = True; WM += ', Type(' + str(iType) + ') not supported'

    RTL = GetVal(URow,F_RTL['name'])

    CMF_RTL = {
        "TOT": 1,
        "FI" : 1}

    if iType == "R3ST":
        if RTL > 2 or RTL < 0: Flag = True; WM += ', RTL (' + str(RTL) + ') for FID = ' + str(GetVal(URow,"FID")) + ' not in range'
        CMF_RTL['TC'] = 0.86 ** RTL
    
    if iType == "R4ST":
        if RTL > 4 or RTL < 0: Flag = True; WM += ', RTL (' + str(RTL) + ') for FID = ' + str(GetVal(URow,"FID")) + ' not in range'
        CMF_RTL['TOT'] = 0.86 ** RTL
    
    if iType == "R4SG":
        if RTL > 4 or RTL < 0: Flag = True; WM += ', RTL (' + str(RTL) + ') for FID = ' + str(GetVal(URow,"FID")) + ' not in range'
        CMF_RTL['TOT'] = 0.96 ** RTL
    
    if iType == "RM3ST":
        if RTL > 2 or RTL < 0: Flag = True; WM += ', RTL (' + str(RTL) + ') for FID = ' + str(GetVal(URow,"FID")) + ' not in range'
        CMF_RTL['TOT'] = 0.86 ** RTL
        CMF_RTL['FI' ] = 0.77 ** RTL

    if iType == "RM4ST":
        if RTL > 4 or RTL < 0: Flag = True; WM += ', RTL (' + str(RTL) + ') for FID = ' + str(GetVal(URow,"FID")) + ' not in range'
        CMF_RTL['TOT'] = 0.86 ** RTL
        CMF_RTL['FI' ] = 0.77 ** RTL
    
    if iType == "U3ST":
        if RTL > 2 or RTL < 0: Flag = True; WM += ', RTL (' + str(RTL) + ') for FID = ' + str(GetVal(URow,"FID")) + ' not in range'
        CMF_RTL['TOT'] = 0.86 ** RTL

    if iType == "U3SG":
        if RTL > 2 or RTL < 0: Flag = True; WM += ', RTL (' + str(RTL) + ') for FID = ' + str(GetVal(URow,"FID")) + ' not in range'
        CMF_RTL['TOT'] = 0.96 ** RTL

    if iType == "U4ST":
        if RTL > 4 or RTL < 0: Flag = True; WM += ', RTL (' + str(RTL) + ') for FID = ' + str(GetVal(URow,"FID")) + ' not in range'
        CMF_RTL['TOT'] = 0.86 ** RTL

    if iType == "U4SG":
        if RTL > 4 or RTL < 0: Flag = True; WM += ', RTL (' + str(RTL) + ') for FID = ' + str(GetVal(URow,"FID")) + ' not in range'
        CMF_RTL['TOT'] = 0.96 ** RTL

    if Flag: arcpy.AddWarning(WM)
    return CMF_RTL
def CMFLighting(URow = '', iType = '', Lighting = 0):

    Flag = False
    WM   = 'CMF Lighting ' 

    if URow:
        iType = GetVal(URow,F_FType['name'])
        Lighting = GetVal(URow,'Lighting')
        Lighting = GetVal(URow,F_LIGHTING['name'],Lighting)
    if not iType in FTypes:
        Flag = True; WM += ', Type(' + str(iType) + ') not supported'

    

    Pni = 0
    CMF_Lighting = 1

    if Lighting <> 1 and Lighting <> 0: Flag = True; WM += ', Lighting (' + str(Lighting) + ') for FID = ' + str(GetVal(URow,"FID")) + ' not in range'

    if Lighting == 1:
        if iType == "R3ST":
            Pni = 0.26
            CMF_Lighting = 1.0 - 0.38 * Pni

        if iType == "R4ST":
            Pni = 0.244
            CMF_Lighting = 1.0 - 0.38 * Pni
    
        if iType == "R4SG":
            Pni = 0.286
            CMF_Lighting = 1.0 - 0.38 * Pni
    
        if iType == "RM3ST":
            Pni = 0.276
            CMF_Lighting = 1.0 - 0.38 * Pni

        if iType == "RM4ST":
            Pni = 0.273
            CMF_Lighting = 1.0 - 0.38 * Pni
        
        if iType == "U3ST":
            Pni = 0.238
            CMF_Lighting = 1.0 - 0.38 * Pni
        
        if iType == "U3SG":
            Pni = 0.229
            CMF_Lighting = 1.0 - 0.38 * Pni
        
        if iType == "U4ST":
            Pni = 0.235
            CMF_Lighting = 1.0 - 0.38 * Pni
        
        if iType == "U4SG":
            Pni = 0.235
            CMF_Lighting = 1.0 - 0.38 * Pni

        if iType == "U2U":
            Pinr = 0.424
            Ppnr = 0.576
            Pnr  = 0.316
            CMF_Lighting = 1.0 - (Pnr * (1.0 - 0.72 * Pinr - 0.83 * Ppnr))

        if iType == "U3T":
            Pinr = 0.429
            Ppnr = 0.571
            Pnr  = 0.304
            CMF_Lighting = 1.0 - (Pnr * (1.0 - 0.72 * Pinr - 0.83 * Ppnr))

        if iType == "U4U":
            Pinr = 0.517
            Ppnr = 0.483
            Pnr  = 0.365
            CMF_Lighting = 1.0 - (Pnr * (1.0 - 0.72 * Pinr - 0.83 * Ppnr))

        if iType == "U4D":
            Pinr = 0.364
            Ppnr = 0.636
            Pnr  = 0.410
            CMF_Lighting = 1.0 - (Pnr * (1.0 - 0.72 * Pinr - 0.83 * Ppnr))

        if iType == "U5T":
            Pinr = 0.432
            Ppnr = 0.568
            Pnr  = 0.274
            CMF_Lighting = 1.0 - (Pnr * (1.0 - 0.72 * Pinr - 0.83 * Ppnr))

        if iType == "R2U":
            Pinr = 0.382
            Ppnr = 0.618
            Pnr  = 0.370
            CMF_Lighting = 1.0 - (Pnr * (1.0 - 0.72 * Pinr - 0.83 * Ppnr))

        if iType == "R4U":
            Pinr = 0.361
            Ppnr = 0.639
            Pnr  = 0.255
            CMF_Lighting = 1.0 - (Pnr * (1.0 - 0.72 * Pinr - 0.83 * Ppnr))

        if iType == "R4D":
            Pinr = 0.323
            Ppnr = 0.677
            Pnr  = 0.426
            CMF_Lighting = 1.0 - (Pnr * (1.0 - 0.72 * Pinr - 0.83 * Ppnr))
    
    if Flag: arcpy.AddWarning(WM)
    return CMF_Lighting
def CMFLTP(URow):

    Flag = False
    WM   = 'CMF LT Phasing ' 

    iType    = GetVal(URow,F_FType['name'])
    if not iType in FTypes:
        Flag = True; WM += ', Type(' + str(iType) + ') not supported'

    LTPList = [
        GetVal(URow,F_LTP1['name']),
        GetVal(URow,F_LTP2['name']),
        GetVal(URow,F_LTP3['name']),
        GetVal(URow,F_LTP4['name'])]
    
    T = []
    for LTP in LTPList:
        if   LTP == 0: T.append(1.00)
        elif LTP == 1: T.append(0.99)
        elif LTP == 2: T.append(0.94)
        else: WM += ', LTP (' + str(i) + ') for FID = ' + str(GetVal(URow,"FID")) + ' not in range'

    CMF_LTP = 1
    for i in T:
        CMF_LTP = CMF_LTP * i

    if Flag: arcpy.AddWarning(WM)
    return CMF_LTP
def CMFNoRTR(URow):

    Flag = False
    WM   = 'CMF NoRTR ' 

    iType    = GetVal(URow,F_FType['name'])
    if not iType in FTypes:
        Flag = True; WM += ', Type(' + str(iType) + ') not supported'

    NoRTR    = GetVal(URow,F_No_RTOR['name'])

    CMF_NoRTR = 1
    if iType == "U3SG":
        if NoRTR > 2 or NoRTR < 0: Flag = True; WM += ', NoRTR (' + str(NoRTR) + ') for FID = ' + str(GetVal(URow,"FID")) + ' not in range'
        CMF_NoRTR = 0.98 ** NoRTR

    if iType == "U4SG":
        if NoRTR > 4 or NoRTR < 0: Flag = True; WM += ', NoRTR (' + str(NoRTR) + ') for FID = ' + str(GetVal(URow,"FID")) + ' not in range'
        CMF_NoRTR = 0.98 ** NoRTR

    if Flag: arcpy.AddWarning(WM)
    return CMF_NoRTR
def CMFBus(URow):

    Flag = False
    WM   = 'CMF Bus ' 

    iType    = GetVal(URow,F_FType['name'])
    if not iType in FTypes:
        Flag = True; WM += ', Type(' + str(iType) + ') not supported'

    Bus    = GetVal(URow,'Bus_Stops')
    Bus    = GetVal(URow,F_BUS_STOPS['name'],Bus)

    CMF_Bus = 1

    if iType <> "U3SG" and iType <> "U4SG": Flag = True; WM += ', Type(' + str(iType) + ') not appropriate'

    if Bus ==0:
        CMF_Bus = 1

    elif Bus == 1 or Bus == 2:
        CMF_Bus = 2.78
        
    elif Bus >= 3:
        CMF_Bus = 4.15

    else:
        Flag = True; WM += ', Bus(' + str(Bus) + ') not supported'
    
    if Flag: arcpy.AddWarning(WM)
    return CMF_Bus            
def CMFSchool(URow):

    Flag = False
    WM   = 'CMF School ' 

    iType    = GetVal(URow,F_FType['name'])
    if not iType in FTypes:
        Flag = True; WM += ', Type(' + str(iType) + ') not supported'

    School    = GetVal(URow,'Schools')
    School    = GetVal(URow,F_SCHOOLS['name'],School)

    CMF_School = 1

    if iType <> "U3SG" and iType <> "U4SG": Flag = True; WM += ', Type(' + str(iType) + ') not appropriate'

    if School == 0:
        CMF_School = 1

    if School >= 1:
        CMF_School = 1.35
        
    if Flag: arcpy.AddWarning(WM)
    return CMF_School
def CMFAlco(URow):

    Flag = False
    WM   = 'CMF Alco ' 

    iType    = GetVal(URow,F_FType['name'])
    if not iType in FTypes:
        Flag = True; WM += ', Type(' + str(iType) + ') not supported'

    Alco    = GetVal(URow,'Alco_Sales')
    Alco    = GetVal(URow,F_ALCO_SALES['name'],Alco)
    
    CMF_Alco = 1
    if iType <> "U3SG" and iType <> "U4SG": Flag = True; WM += ', Type(' + str(iType) + ') not appropriate'

    if Alco == 0:
        CMF_Alco = 1

    elif Alco == 1:
        CMF_Alco = 1.12
        
    elif Alco == 2:
        CMF_Alco = 1.56

    else:
        Flag = True; WM += ', Alco(' + str(Alco) + ') not supported'
        

    if Flag: arcpy.AddWarning(WM)
    return CMF_Alco        
def CMFLaneWidth(iType,LaneWidth,AADT):
    import math 
    Flag = False
    WM   = 'CMF LaneWidth ' 

    if not iType in ['R2U', 'R4U', 'R4D', 'U2U', 'U3T', 'U4D', 'U4U', 'U5T','R4F','U4F','U6F']: Flag = True; WM += ', Type(' + str(iType) + ') not appropriate'
    
    CMF_LaneWidth = 1

    if iType in ['R2U']:
        if LaneWidth >= 12:
                CMF_LaneWidth = 1
        
        if LaneWidth >= 11 and LaneWidth < 12:
            if   AADT <= 400:
                CMF_LaneWidth = 1 + 0.01 * (12-LaneWidth)/1.0
            elif AADT > 400 and AADT <= 2000:
                CMF_LaneWidth = 1 + (0.01 + 2.5 * 10**(-5) * (AADT-400)) * (12-LaneWidth)/1.0
            elif AADT > 2000:
                CMF_LaneWidth = 1 + 0.05 * (12-LaneWidth)/1.0

        if LaneWidth >= 10 and LaneWidth < 11:
            if   AADT <= 400:
                CMF_LaneWidth = 1.01 + 0.01 * (11-LaneWidth)/1.0
            elif AADT > 400 and AADT <= 2000:
                CMF_LaneWidth = 1.01 + (0.01 + 1.75 * 10**(-4) * (AADT-400)) * (11-LaneWidth)/1.0
            elif AADT > 2000:
                CMF_LaneWidth = 1.05 + 0.25 * (11-LaneWidth)/1.0

        if LaneWidth <= 9:
            if   AADT <= 400:
                CMF_LaneWidth = 1.05
            elif AADT > 400 and AADT <= 2000:
                CMF_LaneWidth = 1.05 + 2.81 * 10**(-4) * (AADT-400)
            elif AADT > 2000:
                CMF_LaneWidth = 1.50

    if iType in ['R4U']:
        Pra = 0.582
        if LaneWidth >= 12:
                CMF_LaneWidth = 1
        
        if LaneWidth >= 11 and LaneWidth < 12:
            if   AADT <= 400:
                CMF_LaneWidth = 1 + 0.01 * (12-LaneWidth)/1.0
            elif AADT > 400 and AADT <= 2000:
                CMF_LaneWidth = 1 + (0.01 + 1.88 * 10**(-5) * (AADT-400)) * (12-LaneWidth)/1.0
            elif AADT > 2000:
                CMF_LaneWidth = 1 + 0.04 * (12-LaneWidth)/1.0

        if LaneWidth >= 10 and LaneWidth < 11:
            if   AADT <= 400:
                CMF_LaneWidth = 1.01 + 0.01 * (11-LaneWidth)/1.0
            elif AADT > 400 and AADT <= 2000:
                CMF_LaneWidth = 1.01 + (0.01 + 1.31 * 10**(-4) * (AADT-400)) * (11-LaneWidth)/1.0
            elif AADT > 2000:
                CMF_LaneWidth = 1.04 + 0.19 * (11-LaneWidth)/1.0

        if LaneWidth <= 9:
            if   AADT <= 400:
                CMF_LaneWidth = 1.04
            elif AADT > 400 and AADT <= 2000:
                CMF_LaneWidth = 1.04 + 2.13 * 10**(-4) * (AADT-400)
            elif AADT > 2000:
                CMF_LaneWidth = 1.38
        CMF_LaneWidth = (CMF_LaneWidth-1) * Pra + 1

    if iType in ['R4D']:
        Pra = 0.768
        if LaneWidth >= 12:
                CMF_LaneWidth = 1
        
        if LaneWidth >= 11 and LaneWidth < 12:
            if   AADT <= 400:
                CMF_LaneWidth = 1 + 0.01 * (12-LaneWidth)/1.0
            elif AADT > 400 and AADT <= 2000:
                CMF_LaneWidth = 1 + (0.01 + 1.25 * 10**(-5) * (AADT-400)) * (12-LaneWidth)/1.0
            elif AADT > 2000:
                CMF_LaneWidth = 1 + 0.03 * (12-LaneWidth)/1.0

        if LaneWidth >= 10 and LaneWidth < 11:
            if   AADT <= 400:
                CMF_LaneWidth = 1.01 + 0.00 * (11-LaneWidth)/1.0
            elif AADT > 400 and AADT <= 2000:
                CMF_LaneWidth = 1.01 + (0.01 + 8.75 * 10**(-5) * (AADT-400)) * (11-LaneWidth)/1.0
            elif AADT > 2000:
                CMF_LaneWidth = 1.03 + 0.12 * (11-LaneWidth)/1.0

        if LaneWidth <= 9:
            if   AADT <= 400:
                CMF_LaneWidth = 1.03
            elif AADT > 400 and AADT <= 2000:
                CMF_LaneWidth = 1.03 + 1.38 * 10**(-4) * (AADT-400)
            elif AADT > 2000:
                CMF_LaneWidth = 1.25
        CMF_LaneWidth = (CMF_LaneWidth-1) * Pra + 1

    if iType in ['R4F','U4F','U6F']:
        a = -0.0376
        b =  0.963
        if LaneWidth >= 13:
            CMF_LaneWidth = b
        if LaneWidth < 13:
            CMF_LaneWidth = math.exp(a*(LaneWidth-12))
    if Flag: arcpy.AddWarning(WM)
    return CMF_LaneWidth
def CMFShoulderWidthType(iType,ShWidth,ShType,AADT):

    Flag = False
    WM   = 'CMF ShoulderWidth ' 

    if not iType in ['R2U', 'R4U', 'R4D', 'U2U', 'U3T', 'U4D', 'U4U', 'U5T']: Flag = True; WM += ', Type(' + str(iType) + ') not appropriate'

    CMF_ShWidth = 1
    CMF_ShType  = 1
    Output      = 1
    Pra         = 0.617
    if iType in ['R2U','R4U']:
        if ShWidth >= 0 and ShWidth < 1:
            if   AADT <= 400:
                CMF_ShWidth = 1.1
            elif AADT > 400 and AADT <= 2000:
                CMF_ShWidth = 1.1 + (2.5 * 10**(-4) * (AADT-400))
            elif AADT > 2000:
                CMF_ShWidth = 1.5 - (1.5-1.4) * (ShWidth-0)

        if ShWidth >= 1 and ShWidth < 3:
            if   AADT <= 400:
                CMF_ShWidth = 1.07
            elif AADT > 400 and AADT <= 2000:
                CMF_ShWidth = 1.07 + (1.43 * 10**(-4) * (AADT-400))
            elif AADT > 2000:
                CMF_ShWidth = 1.4 - (1.4-1.23) * (ShWidth-1)/(3-1)

        if ShWidth >= 3 and ShWidth < 5:
            if   AADT <= 400:
                CMF_ShWidth = 1.02
            elif AADT > 400 and AADT <= 2000:
                CMF_ShWidth = 1.02 + (8.125 * 10**(-5) * (AADT-400))
            elif AADT > 2000:
                CMF_ShWidth = 1.23 - (1.23-1.08) * (ShWidth-3)/(5-3)

        if ShWidth >= 5 and ShWidth < 7:
                CMF_ShWidth = 1.08 - (1.08-0.94) * (ShWidth-5)/(5-7)

        if ShWidth <= 7:
            if   AADT <= 400:
                CMF_ShWidth = 0.98
            elif AADT > 400 and AADT <= 2000:
                CMF_ShWidth = 0.98 - 6.875 * 10**(-5) * (AADT-400)
            elif AADT > 2000:
                CMF_ShWidth = 0.87

        if ShType == 1: #'Paved':
            CMF_ShType = 1

        if ShType == 2: #'Gravel':
            if ShWidth < 2:
                CMF_ShType = 1
            if ShWidth >= 2 and ShWidth < 6:
                CMF_ShType = 1.01
            if ShWidth >= 6:
                CMF_ShType = 1.02

        if ShType == 3: #'Composite':
            if ShWidth < 1:
                CMF_ShType = 1
            if ShWidth >= 1 and ShWidth < 2:
                CMF_ShType = 1.01
            if ShWidth >= 2 and ShWidth < 4:
                CMF_ShType = 1.02
            if ShWidth >= 4 and ShWidth < 6:
                CMF_ShType = 1.03
            if ShWidth >= 6 and ShWidth < 8:
                CMF_ShType = 1.04
            if ShWidth >= 8:
                CMF_ShType = 1.06

        if ShType == 4: #'Turf':
            if ShWidth < 1:
                CMF_ShType = 1
            if ShWidth >= 1 and ShWidth < 2:
                CMF_ShType = 1.01
            if ShWidth >= 2 and ShWidth < 3:
                CMF_ShType = 1.03
            if ShWidth >= 3 and ShWidth < 4:
                CMF_ShType = 1.04
            if ShWidth >= 4 and ShWidth < 6:
                CMF_ShType = 1.05
            if ShWidth >= 6 and ShWidth < 8:
                CMF_ShType = 1.08
            if ShWidth >= 6:
                CMF_ShType = 1.11
        Output = (CMF_ShWidth * CMF_ShType - 1) * Pra + 1
    
    if iType in ['R4D']:
        if ShWidth >= 8:
            Output = 1.00

        if ShWidth >= 6 and ShWidth < 8:
            Output = 1.04

        if ShWidth >= 4 and ShWidth < 6:
            Output = 1.09

        if ShWidth >= 2 and ShWidth < 4:
            Output = 1.13

        if ShWidth >= 0 and ShWidth < 2:
            Output = 1.18

    if Flag: arcpy.AddWarning(WM)
    return Output
def CMFInsideShoulderW(iType,ShWidth):
    import math
    Flag = False
    WM   = 'CMF Inside ShoulderWidth ' 

    if not iType in ['R4F','U4F','U6F']: Flag = True; WM += ', Type(' + str(iType) + ') not appropriate'

    CMF_ShWidth = {'FI':1,'PDO':1}
    CMF_ShWidth['FI' ] = math.exp(-0.0172*(ShWidth-6))
    CMF_ShWidth['PDO'] = math.exp(-0.0153*(ShWidth-6))

    if Flag: arcpy.AddWarning(WM)
    return CMF_ShWidth
def CMFOutsideShoulderW(iType,ShWidth):
    import math
    Flag = False
    WM   = 'CMF Outside ShoulderWidth ' 

    if not iType in ['R4F','U4F','U6F']: Flag = True; WM += ', Type(' + str(iType) + ') not appropriate'

    CMF_ShWidth = {'FI':1,'PDO':1}
    CMF_ShWidth['FI' ] = math.exp(-0.0897*(ShWidth-10))
    CMF_ShWidth['PDO'] = math.exp(-0.0840*(ShWidth-10))

    if Flag: arcpy.AddWarning(WM)
    return CMF_ShWidth
def CMFHorCurve(iType,Lc,R,S):
    # Lc in miles
    # R in Feet
    Flag = False
    WM   = 'CMF Hor Curvature ' 

    if not iType in ['R2U', 'R4U', 'R4D', 'U2U', 'U3T', 'U4D', 'U4U', 'U5T','R4F','U4F','U6F']: Flag = True; WM += ', Type(' + str(iType) + ') not appropriate'

    if Lc < 100/5280: Lc = 100/5280
    if R < 100: R = 100
    
    CMF_HorCur = {'TOT':1,'MVFI':1,'MVPDO':1,'SVFI':1,'SVPDO':1}
    def FreewayCMF(a,R,P):
        return(1+a*((5730.0/R)**2)*P)

    if iType in ['R2U']:
        CMF_HorCur['TOT'] = (1.55*Lc + 80.2/R - 0.012 * S)/(1.55*Lc)
        if CMF_HorCur['TOT'] < 1: CMF_HorCur['TOT'] = 1
    if iType in ['R4F','U4F','U6F']:
        if R < 1000: R = 1000
        CMF_HorCur['MVFI' ] = FreewayCMF(0.0172,R,1)
        CMF_HorCur['MVPDO'] = FreewayCMF(0.0340,R,1)
        CMF_HorCur['SVFI' ] = FreewayCMF(0.0179,R,1)
        CMF_HorCur['SVPDO'] = FreewayCMF(0.0626,R,1)

    if Flag: arcpy.AddWarning(WM)
    return CMF_HorCur
def CMFGrade(iType,G):

    Flag = False
    WM   = 'CMF Grade' 

    if not iType in ['R2U', 'R4U', 'R4D', 'U2U', 'U3T', 'U4D', 'U4U', 'U5T']: Flag = True; WM += ', Type(' + str(iType) + ') not appropriate'

    
    CMF_Grade = 1

    if iType in ['R2U']:
        if G <= 0.03:
            CMF_Grade = 1.00
        if G > 0.03 and G <= .06:
            CMF_Grade = 1.10
        if G > 0.06:
            CMF_Grade = 1.16

    if Flag: arcpy.AddWarning(WM)
    return CMF_Grade
def CMFDrivewayDensity(iType,DD,AADT):
    import math

    def NaturalLog(A):
        if A > 0:
            return math.log1p(A - 1)
        else:
            return 0
    Flag = False
    WM   = 'CMF Driveway Density ' 

    if not iType in ['R2U', 'R4U', 'R4D', 'U2U', 'U3T', 'U4D', 'U4U', 'U5T']: Flag = True; WM += ', Type(' + str(iType) + ') not appropriate'
    AADT = NaturalLog(AADT)
    CMF_DrwDens = 1

    if iType in ['R2U']:
        CMF_DrwDens = (0.322 + DD*(0.05 - .005*AADT))/(0.322 + 5*(0.05 - .005*AADT))
        if CMF_DrwDens < 1: CMF_DrwDens = 1
    if Flag: arcpy.AddWarning(WM)
    return CMF_DrwDens
def CMFRHR(iType,RHR):
    import math 
    Flag = False
    WM   = 'CMF RHR' 

    if not iType in ['R2U', 'R4U', 'R4D', 'U2U', 'U3T', 'U4D', 'U4U', 'U5T']: Flag = True; WM += ', Type(' + str(iType) + ') not appropriate'

    CMF_RHR = 1

    if iType in ['R2U']:
        CMF_RHR = math.exp(-0.6869+0.0668*RHR)/math.exp(-0.4865)

    if Flag: arcpy.AddWarning(WM)
    return CMF_RHR
def CMFOnStreetParking(iType,OSPParkT,OSPAreaT,OSPPropor):
    # Code Dictionary:
    # 1.	Type : Type of parking (1= Angle Parking; 2= Parallel Parking)
    # 2.	Area_Type: (1=Residential; 2=Industrial; 3=Commercial; 4=Institutional; 5=Other)
    # Fields to read from:
    # F_OSPPropor: Proporsion (ParkL/SegL)
    # F_OSPParkT: Parking type
    # F_OSPAreaT: Area Type
    
    Flag = False
    WM   = 'CMF On Street Parking' 

    #iType    = GetVal(URow,F_FType['name'])
    if not iType in ['R2U', 'R4U', 'R4D', 'U2U', 'U3T', 'U4D', 'U4U', 'U5T']:
        Flag = True; WM += ', Type(' + str(iType) + ') not supported'

    #OSPPropor  = GetVal(URow,F_OSPPropor['name'])
    #OSPParkT   = GetVal(URow,F_OSPParkT ['name'])
    #OSPAreaT   = GetVal(URow,F_OSPAreaT ['name'])

    Factor        = 1
    if iType in ['U2U']:
        if OSPParkT in [2]: # Parallel
            if OSPAreaT in [1,5]: # Resi Other
                Factor = 1.465
            if OSPAreaT in [2,3,4]: # Comm Indus Instit
                Factor = 2.074

        if OSPParkT in [1]: # Angle
            if OSPAreaT in [1,5]: # Resi Other
                Factor = 3.428
            if OSPAreaT in [2,3,4]: # Comm Indus Instit
                Factor = 4.853

    if iType in ['U3T']:
        if OSPParkT in [2]: # Parallel
            if OSPAreaT in [1,5]: # Resi Other
                Factor = 1.465
            if OSPAreaT in [2,3,4]: # Comm Indus Instit
                Factor = 2.074

        if OSPParkT in [1]: # Angle
            if OSPAreaT in [1,5]: # Resi Other
                Factor = 3.428
            if OSPAreaT in [2,3,4]: # Comm Indus Instit
                Factor = 4.853

    if iType in ['U4U']:
        if OSPParkT in [2]: # Parallel
            if OSPAreaT in [1,5]: # Resi Other
                Factor = 1.100
            if OSPAreaT in [2,3,4]: # Comm Indus Instit
                Factor = 2.709

        if OSPParkT in [1]: # Angle
            if OSPAreaT in [1,5]: # Resi Other
                Factor = 2.574
            if OSPAreaT in [2,3,4]: # Comm Indus Instit
                Factor = 3.999

    if iType in ['U4D']:
        if OSPParkT in [2]: # Parallel
            if OSPAreaT in [1,5]: # Resi Other
                Factor = 1.100
            if OSPAreaT in [2,3,4]: # Comm Indus Instit
                Factor = 2.709

        if OSPParkT in [1]: # Angle
            if OSPAreaT in [1,5]: # Resi Other
                Factor = 2.574
            if OSPAreaT in [2,3,4]: # Comm Indus Instit
                Factor = 3.999

    if iType in ['U5T']:
        if OSPParkT in [2]: # Parallel
            if OSPAreaT in [1,5]: # Resi Other
                Factor = 1.100
            if OSPAreaT in [2,3,4]: # Comm Indus Instit
                Factor = 2.709

        if OSPParkT in [1]: # Angle
            if OSPAreaT in [1,5]: # Resi Other
                Factor = 2.574
            if OSPAreaT in [2,3,4]: # Comm Indus Instit
                Factor = 3.999

    CMF_OnStrPark = 1 + OSPPropor * (Factor - 1.0)

    if Flag: arcpy.AddWarning(WM)
    return CMF_OnStrPark
def CMFMedianWidth(iType,MW):
   
    Flag = False
    WM   = 'CMF Median Width' 

    #iType    = GetVal(URow,F_FType['name'])
    if not iType in ['R2U', 'R4U', 'R4D', 'U2U', 'U3T', 'U4D', 'U4U', 'U5T','R4F','U4F','U6F']:
        #Flag = True; 
        WM += ', Type(' + str(iType) + ') not supported'

    CMF_MW        = 1
    if iType in ['R4D']:
        if MW <= 10:
            CMF_MW = 1.04
        if MW > 10  and MW <= 20:
            CMF_MW = (1.04-1.02)/(10-20)*(MW-20)+1.02
        if MW > 20  and MW <= 30:
            CMF_MW = (1.02-1.00)/(20-30)*(MW-30)+1.00
        if MW > 30  and MW <= 40:
            CMF_MW = (1.00-0.99)/(30-40)*(MW-40)+0.99
        if MW > 40  and MW <= 50:
            CMF_MW = (0.99-0.97)/(40-50)*(MW-50)+0.97
        if MW > 50  and MW <= 60:
            CMF_MW = (0.97-0.96)/(50-60)*(MW-60)+0.96
        if MW > 60  and MW <= 70:
            CMF_MW = 0.96
        if MW > 70  and MW <= 80:
            CMF_MW = (0.96-0.95)/(70-80)*(MW-80)+0.95
        if MW > 90  and MW <= 100:
            CMF_MW = 0.94
        if MW > 100: 
            CMF_MW = 0.94

    if iType in ['U4D']:
        if MW <= 10:
            CMF_MW = 1.01
        if MW > 10  and MW <= 15:
            CMF_MW = (1.01-1.00)/(10-15)*(MW-15)+1.00
        if MW > 15  and MW <= 20:
            CMF_MW = (1.00-0.99)/(15-20)*(MW-20)+0.99
        if MW > 20  and MW <= 80:
            CMF_MW = (0.99-0.93)/(20-80)*(MW-80)+0.93
        if MW > 80  and MW <= 90:
            CMF_MW = 0.93
        if MW > 90  and MW <= 100:
            CMF_MW = (0.93-0.92)/(90-100)*(MW-100)+0.92
        if MW > 100: 
            CMF_MW = 0.92

    if iType in ['R4F','U4F','U6F']:
        CMF_MV = 1
    if Flag: arcpy.AddWarning(WM)
    if CMF_MW == 1:
        if MW <> 0:
           CMF_MW = 1.0001
        else:
            CMF_MW = 1
    return CMF_MW
def CMFMedianBarrier(iType,MW,MType,SWi):
    import math
    Flag = False
    WM   = 'CMF Median Width and Median Barrier ' 

    if not iType in ['R4F','U4F','U6F']: Flag = True; WM += ', Type(' + str(iType) + ') not appropriate'

    CMF_MB = {'MVFI':1,'MVPDO':1,'SVFI':1,'SVPDO':1}
    if MW>90: MW=90
    if MW<9: MW = 9
    if SWi<2: SWi = 2
    if SWi>12: SWi = 12

    if MType in [1,2,3]:
        CMF_MB['MVFI' ] = math.exp(-0.00302*(MW-2*SWi-48))
        CMF_MB['MVPDO'] = math.exp(-0.00291*(MW-2*SWi-48))
        CMF_MB['SVFI' ] = math.exp(+0.00102*(MW-2*SWi-48))
        CMF_MB['SVPDO'] = math.exp(-0.00289*(MW-2*SWi-48))
    if MType in [4,5,6]:
        if MW>34: MW=34
        if MW<9: MW = 9
        CMF_MB['MVFI' ] = math.exp(-0.00302*(MW-48)) * math.exp(0.131/(MW/2))
        CMF_MB['MVPDO'] = math.exp(-0.00291*(MW-48)) * math.exp(0.169/(MW/2))
        CMF_MB['SVFI' ] = math.exp(+0.00102*(MW-48)) * math.exp(0.131/(MW/2))
        CMF_MB['SVPDO'] = math.exp(-0.00289*(MW-48)) * math.exp(0.169/(MW/2))

    if Flag: arcpy.AddWarning(WM)
    return CMF_MB
def CMFFixedObjects(iType,Density,Offset):
   
    Flag = False
    WM   = 'CMF Fixed Objects'

    #iType    = GetVal(URow,F_FType['name'])
    if not iType in ['R2U', 'R4U', 'R4D', 'U2U', 'U3T', 'U4D', 'U4U', 'U5T']:
        #Flag = True; 
        WM += ', Type(' + str(iType) + ') not supported'

    #OSPPropor  = GetVal(URow,F_OSPPropor['name'])
    #OSPParkT   = GetVal(URow,F_OSPParkT ['name'])
    #OSPAreaT   = GetVal(URow,F_OSPAreaT ['name'])

    Pfo    = 0
    if iType in ['U2U']:
        Pfo = 0.059
    if iType in ['U3T']:
        Pfo = 0.034
    if iType in ['U4U']:
        Pfo = 0.037
    if iType in ['U4D']:
        Pfo = 0.036
    if iType in ['U5T']:
        Pfo = 0.016

    Foffset = 0
    if Offset <= 2:
        Foffset = 0.232
    if Offset > 2 and Offset <= 5:
        Foffset = (0.133 - 0.232) / (5 - 2) * (Offset - 5) + 0.133
    if Offset > 5 and Offset <= 10:
        Foffset = (0.087 - 0.133) / (10 - 5) * (Offset - 10) + 0.087
    if Offset > 10 and Offset <= 15:
        Foffset = (0.068 - 0.087) / (15 - 10) * (Offset - 15) + 0.068
    if Offset > 15 and Offset <= 20:
        Foffset = (0.057 - 0.068) / (20 - 15) * (Offset - 20) + 0.057
    if Offset > 20 and Offset <= 25:
        Foffset = (0.049 - 0.057) / (25 - 20) * (Offset - 25) + 0.049
    if Offset > 25 and Offset <= 30:
        Foffset = (0.044 - 0.049) / (30 - 25) * (Offset - 30) + 0.044
    if Offset > 30: 
        Foffset = 0.044

    CMF_FO = Foffset * Density * Pfo + (1-Pfo)
    if CMF_FO < 1: 
        if Density <> 0: 
            CMF_FO = 1.00001
        else:
            CMF_FO = 1
    if Flag: arcpy.AddWarning(WM)
    return CMF_FO
def CMFLaneChange(iType,Dict):
    import math
    Flag = False
    WM   = 'CMF Lane Change'

    #iType    = GetVal(URow,F_FType['name'])
    if not iType in ['R4F', 'U4F', 'U6F']:
        #Flag = True; 
        WM += ', Type(' + str(iType) + ') not supported'

    a=0.175;b=12.56;c=0.001;d=-0.272
    Pwevinc = Dict['LsWevInc']/Dict['Ls']
    Pwevdec = Dict['LsWevDec']/Dict['Ls']
    fwevinc = 1-Pwevinc+Pwevinc*math.exp(a/Dict['LWevInc'])
    fwevdec = 1-Pwevdec+Pwevdec*math.exp(a/Dict['LWevDec'])
    A = -b*Dict['XbEnt']+d*NaturalLog(c*Dict['AADTbEnt'])
    flcinc1 = 1 + math.exp(A)/b/Dict['Ls']*(1-math.exp(-b*Dict['Ls']))
    A = -b*Dict['XeExt']+d*NaturalLog(c*Dict['AADTeExt'])
    flcinc2 = 1 + math.exp(A)/b/Dict['Ls']*(1-math.exp(-b*Dict['Ls']))
    flcinc = flcinc1*flcinc2

    A = -b*Dict['XeEnt']+d*NaturalLog(c*Dict['AADTeEnt'])
    flcdec1 = 1 + math.exp(A)/b/Dict['Ls']*(1-math.exp(-b*Dict['Ls']))
    A = -b*Dict['XbExt']+d*NaturalLog(c*Dict['AADTbExt'])
    flcdec2 = 1 + math.exp(A)/b/Dict['Ls']*(1-math.exp(-b*Dict['Ls']))
    flcdec = flcdec1*flcdec2

    CMF_LC_FI = 0.5*fwevinc*flcinc + 0.5*fwevdec*flcdec

    a=0.123;b=13.46;c=0.001;d=-0.283
    fwevinc = 1-Pwevinc+Pwevinc*math.exp(a/Dict['LWevInc'])
    fwevdec = 1-Pwevdec+Pwevdec*math.exp(a/Dict['LWevDec'])
    A = -b*Dict['XbEnt']+d*NaturalLog(c*Dict['AADTbEnt'])
    flcinc1 = 1 + math.exp(A)/b/Dict['Ls']*(1-math.exp(-b*Dict['Ls']))
    A = -b*Dict['XeExt']+d*NaturalLog(c*Dict['AADTeExt'])
    flcinc2 = 1 + math.exp(A)/b/Dict['Ls']*(1-math.exp(-b*Dict['Ls']))
    flcinc = flcinc1*flcinc2

    A = -b*Dict['XeEnt']+d*NaturalLog(c*Dict['AADTeEnt'])
    flcdec1 = 1 + math.exp(A)/b/Dict['Ls']*(1-math.exp(-b*Dict['Ls']))
    A = -b*Dict['XbExt']+d*NaturalLog(c*Dict['AADTbExt'])
    flcdec2 = 1 + math.exp(A)/b/Dict['Ls']*(1-math.exp(-b*Dict['Ls']))
    flcdec = flcdec1*flcdec2

    CMF_LC_PDO = 0.5*fwevinc*flcinc + 0.5*fwevdec*flcdec
    
    CMF_LC = {'FI':1,'PDO':1}
    CMF_LC['FI' ] = CMF_LC_FI
    CMF_LC['PDO'] = CMF_LC_PDO
    if Flag: arcpy.AddWarning(WM)
    return CMF_LC
def CMFCZandOB(iType,Pob,Whc,Wocb,Ws):
    import math
    Flag = False
    WM   = 'CMF Outside Clearance'

    #iType    = GetVal(URow,F_FType['name'])
    if not iType in ['R4F', 'U4F', 'U6F']:
        #Flag = True; 
        WM += ', Type(' + str(iType) + ') not supported'

    
    CMF_OC_FI = (1-Pob)*math.exp(-0.00451*(Whc-Ws-20))+Pob*math.exp(-.00451*(Wocb-20))

    a=0.131
    CMF_OB_FI  = (1-Pob)+Pob*math.exp(a/Wocb)
    a=0.169
    CMF_OB_PDO = (1-Pob)+Pob*math.exp(a/Wocb)
    
    CMF_OB = {'FI':1,'PDO':1}
    CMF_OB['FI' ] = CMF_OB_FI
    CMF_OB['PDO'] = CMF_OB_PDO
    if Flag: arcpy.AddWarning(WM)
    return [CMF_OC_FI,CMF_OB]
def CMFHighVolume(iType,Phv):
    import math
    Flag = False
    WM   = 'CMF Outside Clearance'

    #iType    = GetVal(URow,F_FType['name'])
    if not iType in ['R4F', 'U4F', 'U6F']:
        #Flag = True; 
        WM += ', Type(' + str(iType) + ') not supported'

    if Phv<0:Phv=0
    if Phv>1:Phv=1
    a=0.350
    CMF_HV_MVFI  = math.exp(a*Phv)
    a=0.283
    CMF_HV_MVPDO = math.exp(a*Phv)
    a=-0.0675
    CMF_HV_SVFI  = math.exp(a*Phv)
    a=-0.611
    CMF_HV_SVPDO = math.exp(a*Phv)
    
    CMF_HV = {'MVFI':CMF_HV_MVFI,'MVPDO':CMF_HV_MVPDO,'SVFI':CMF_HV_SVFI,'SVPDO':CMF_HV_SVPDO}

    if Flag: arcpy.AddWarning(WM)
    return CMF_HV


def PredictedCrash(Input, Expected, Output):
    arcpy.AddMessage(" ")
    arcpy.AddMessage("    Import Predicted Crash")

    OutDic = OutputParser(Output, 'shp')
    Output = OutDic['folder'] + '\\' + OutDic['file']

    arcpy.AddMessage("    Copy Features: Sites Layer")
    SitesLayer = CopyFeatures(Input,'PCSites')
    TotalSites = int(str(arcpy.GetCount_management(SitesLayer)))
    arcpy.AddMessage("     - Total number of Items Found: " + str(TotalSites))

    arcpy.AddMessage("    Add Fields: Sites Layer")
    PreCrashDic = {
        'R3ST':  [F_CMFSkew, F_CMFLTL, F_CMFRTL, F_CMFLight, F_TOT_PC, F_TOTk_PC],
        'R4ST':  [F_CMFSkew, F_CMFLTL, F_CMFRTL, F_CMFLight, F_TOT_PC, F_TOTk_PC],
        'R4SG':  [           F_CMFLTL, F_CMFRTL, F_CMFLight, F_TOT_PC, F_TOTk_PC],
        'RM3ST': [F_CMFSkew, F_CMFLTL, F_CMFRTL, F_CMFLight, F_TOT_PC, F_TOTk_PC, F_FI_PC, F_FIk_PC, F_FIKAB_PC, F_FIKABk_PC],
        'RM4ST': [F_CMFSkew, F_CMFLTL, F_CMFRTL, F_CMFLight, F_TOT_PC, F_TOTk_PC, F_FI_PC, F_FIk_PC, F_FIKAB_PC, F_FIKABk_PC],
        'RM4SG': [                                           F_TOT_PC, F_TOTk_PC, F_FI_PC, F_FIk_PC, F_FIKAB_PC, F_FIKABk_PC],
        'U3ST' : [F_CMFLTL, F_CMFRTL, F_CMFLight,                                                         F_MV_PC, F_MVk_PC, F_MVFI_PC, F_MVFIk_PC, F_MVPDO_PC, F_MVPDOk_PC, F_SV_PC, F_SVk_PC, F_SVFI_PC, F_SVFIk_PC, F_SVPDO_PC, F_SVPDOk_PC],
        'U4ST' : [F_CMFLTL, F_CMFRTL, F_CMFLight,                                                         F_MV_PC, F_MVk_PC, F_MVFI_PC, F_MVFIk_PC, F_MVPDO_PC, F_MVPDOk_PC, F_SV_PC, F_SVk_PC, F_SVFI_PC, F_SVFIk_PC, F_SVPDO_PC, F_SVPDOk_PC],
        'U3SG' : [F_CMFLTL, F_CMFRTL, F_CMFLight, F_CMFLTP, F_CMFNoRTR,                                   F_MV_PC, F_MVk_PC, F_MVFI_PC, F_MVFIk_PC, F_MVPDO_PC, F_MVPDOk_PC, F_SV_PC, F_SVk_PC, F_SVFI_PC, F_SVFIk_PC, F_SVPDO_PC, F_SVPDOk_PC],
        'U4SG' : [F_CMFLTL, F_CMFRTL, F_CMFLight, F_CMFLTP, F_CMFNoRTR, F_CMFBus, F_CMFSchool, F_CMFAlco, F_MV_PC, F_MVk_PC, F_MVFI_PC, F_MVFIk_PC, F_MVPDO_PC, F_MVPDOk_PC, F_SV_PC, F_SVk_PC, F_SVFI_PC, F_SVFIk_PC, F_SVPDO_PC, F_SVPDOk_PC, F_Ped_PC, F_Pedk_PC],
        'R2U':   [F_TOT_PC, F_CMFGrade],
        'R4U':   [F_TOT_PC],
        'R4D':   [F_TOT_PC],
        'U2U':   [F_TOT_PC, F_FI_PC, F_MV_PC, F_MVFI_PC, F_MVPDO_PC, F_MVd_PC, F_MVdk_PC, F_MVFId_PC, F_MVPDO_PC, F_MVnd_PC, F_MVndk_PC, F_MVFInd_PC, F_MVFIndk_PC, F_MVPDOnd_PC, F_MVPDOndk_PC, F_SV_PC, F_SVk_PC, F_SVFI_PC, F_SVFIk_PC, F_SVPDO_PC, F_SVPDOk_PC],
        'U3T':   [F_TOT_PC, F_FI_PC, F_MV_PC, F_MVFI_PC, F_MVPDO_PC, F_MVd_PC, F_MVdk_PC, F_MVFId_PC, F_MVPDO_PC, F_MVnd_PC, F_MVndk_PC, F_MVFInd_PC, F_MVFIndk_PC, F_MVPDOnd_PC, F_MVPDOndk_PC, F_SV_PC, F_SVk_PC, F_SVFI_PC, F_SVFIk_PC, F_SVPDO_PC, F_SVPDOk_PC],
        'U4D':   [F_TOT_PC, F_FI_PC, F_MV_PC, F_MVFI_PC, F_MVPDO_PC, F_MVd_PC, F_MVdk_PC, F_MVFId_PC, F_MVPDO_PC, F_MVnd_PC, F_MVndk_PC, F_MVFInd_PC, F_MVFIndk_PC, F_MVPDOnd_PC, F_MVPDOndk_PC, F_SV_PC, F_SVk_PC, F_SVFI_PC, F_SVFIk_PC, F_SVPDO_PC, F_SVPDOk_PC],
        'U4U':   [F_TOT_PC, F_FI_PC, F_MV_PC, F_MVFI_PC, F_MVPDO_PC, F_MVd_PC, F_MVdk_PC, F_MVFId_PC, F_MVPDO_PC, F_MVnd_PC, F_MVndk_PC, F_MVFInd_PC, F_MVFIndk_PC, F_MVPDOnd_PC, F_MVPDOndk_PC, F_SV_PC, F_SVk_PC, F_SVFI_PC, F_SVFIk_PC, F_SVPDO_PC, F_SVPDOk_PC],
        'U5T':   [F_TOT_PC, F_FI_PC, F_MV_PC, F_MVFI_PC, F_MVPDO_PC, F_MVd_PC, F_MVdk_PC, F_MVFId_PC, F_MVPDO_PC, F_MVnd_PC, F_MVndk_PC, F_MVFInd_PC, F_MVFIndk_PC, F_MVPDOnd_PC, F_MVPDOndk_PC, F_SV_PC, F_SVk_PC, F_SVFI_PC, F_SVFIk_PC, F_SVPDO_PC, F_SVPDOk_PC],
        'R4F':   [F_TOT_PC, F_FI_PC, F_MV_PC, F_MVFI_PC, F_MVPDO_PC, F_SV_PC , F_SVFI_PC, F_SVPDO_PC, F_CMFLW, F_CMFMWBMFI, F_CMFMWBMPD, F_CMFMWBSFI, F_CMFMWBSPD, F_CMFInSWFI, F_CMFInSWPD, F_CMFOutSWFI, F_CMFOutSWPD, F_CMFCZFI, F_CMFOBFI, F_CMFOBPDO, F_CMFRSSVFI, F_CMFHVMVFI, F_CMFHVMVPDO, F_CMFHVSVFI, F_CMFHVSVPDO],
        'U4F':   [F_TOT_PC, F_FI_PC, F_MV_PC, F_MVFI_PC, F_MVPDO_PC, F_SV_PC , F_SVFI_PC, F_SVPDO_PC, F_CMFLW, F_CMFMWBMFI, F_CMFMWBMPD, F_CMFMWBSFI, F_CMFMWBSPD, F_CMFInSWFI, F_CMFInSWPD, F_CMFOutSWFI, F_CMFOutSWPD, F_CMFCZFI, F_CMFOBFI, F_CMFOBPDO, F_CMFRSSVFI, F_CMFHVMVFI, F_CMFHVMVPDO, F_CMFHVSVFI, F_CMFHVSVPDO],
        'U6F':   [F_TOT_PC, F_FI_PC, F_MV_PC, F_MVFI_PC, F_MVPDO_PC, F_SV_PC , F_SVFI_PC, F_SVPDO_PC, F_CMFLW, F_CMFMWBMFI, F_CMFMWBMPD, F_CMFMWBSFI, F_CMFMWBSPD, F_CMFInSWFI, F_CMFInSWPD, F_CMFOutSWFI, F_CMFOutSWPD, F_CMFCZFI, F_CMFOBFI, F_CMFOBPDO, F_CMFRSSVFI, F_CMFHVMVFI, F_CMFHVMVPDO, F_CMFHVSVFI, F_CMFHVSVPDO]
        }
    
    Dic = {}
    for L in PreCrashDic.values():
        Dic.update({K['name']:'' for K in L})
    for Field in Dic.keys():
        arcpy.DeleteField_management(SitesLayer,Field)

    Flag          = {Type: False for Type in PreCrashDic.keys()}
    ExistingTypes = {GetVal(SRow, F_FType['name']): True for SRow in arcpy.SearchCursor(SitesLayer)}
    for Type in ExistingTypes.keys():
        if Type in Flag.keys():
            Flag[Type] = True

    for Type in Flag.keys():
        if Flag[Type]:
            for Field in PreCrashDic[Type]:
                AddField (SitesLayer,Field,'append')
                if Field in [F_CMFGrade]:
                    CalField(SitesLayer,Field,1)

    arcpy.AddMessage("    Update Cursor: Sites Layer")
    UC = arcpy.UpdateCursor(SitesLayer)
    for URow in UC:

        Type        = GetVal(URow,F_FType['name'])

        if Type in Flag.keys():

            SPF = SPF_Module(URow)

            if Type == 'R3ST':

                CMF_Lighting = CMFLighting(URow)
                CMF_Skew     = CMFSkew(URow)
                CMF_LTL      = CMFLTL(URow)
                CMF_RTL      = CMFRTL(URow)

                TOT_PC = SPF['TOT'] * CMF_Skew['TOT'] * CMF_LTL['TOT'] * CMF_RTL['TOT'] * CMF_Lighting


                URow.setValue(F_CMFLight['name'], CMF_Lighting)
                URow.setValue(F_CMFSkew ['name'], CMF_Skew['TOT'])
                URow.setValue(F_CMFLTL  ['name'], CMF_LTL['TOT'])
                URow.setValue(F_CMFRTL  ['name'], CMF_RTL['TOT'])
                URow.setValue(F_TOT_PC  ['name'], TOT_PC * GetVal(URow,'CCMF',1))
                URow.setValue(F_TOTk_PC ['name'], SPF['TCk'])

            if Type == 'R4ST':

                CMF_Lighting = CMFLighting(URow)
                CMF_Skew     = CMFSkew(URow)
                CMF_LTL      = CMFLTL(URow)
                CMF_RTL      = CMFRTL(URow)

                TOT_PC = SPF['TOT'] * CMF_Skew['TOT'] * CMF_LTL['TOT'] * CMF_RTL['TOT'] * CMF_Lighting

                URow.setValue(F_CMFLight['name'], CMF_Lighting)
                URow.setValue(F_CMFSkew ['name'], CMF_Skew['TOT'])
                URow.setValue(F_CMFLTL  ['name'], CMF_LTL['TOT'])
                URow.setValue(F_CMFRTL  ['name'], CMF_RTL['TOT'])
                URow.setValue(F_TOT_PC  ['name'], TOT_PC * GetVal(URow,'CCMF',1))
                URow.setValue(F_TOTk_PC ['name'], SPF['TCk'])

            if Type == 'R4SG':

                CMF_Lighting = CMFLighting(URow)
                CMF_LTL      = CMFLTL(URow)
                CMF_RTL      = CMFRTL(URow)

                TOT_PC    = SPF['TOT'] * CMF_LTL['TOT'] * CMF_RTL['TOT'] * CMF_Lighting

                URow.setValue(F_CMFLight['name'], CMF_Lighting)
                URow.setValue(F_CMFLTL  ['name'], CMF_LTL['TOT'])
                URow.setValue(F_CMFRTL  ['name'], CMF_RTL['TOT'])
                URow.setValue(F_TOT_PC  ['name'], TOT_PC * GetVal(URow,'CCMF',1))
                URow.setValue(F_TOTk_PC ['name'], SPF['TCk'])

            if Type == 'RM3ST':

                CMF_Lighting = CMFLighting(URow)
                CMF_Skew     = CMFSkew(URow)
                CMF_LTL      = CMFLTL(URow)
                CMF_RTL      = CMFRTL(URow)

                TOT_PC   = SPF['TOT'  ] * CMF_Skew['TOT'] * CMF_LTL['TOT'] * CMF_RTL['TOT'] * CMF_Lighting
                FI_PC    = SPF['FI'   ] * CMF_Skew['FI' ] * CMF_LTL['FI' ] * CMF_RTL['FI' ] * CMF_Lighting
                FIKAB_PC = SPF['FIKAB'] * CMF_Skew['FI' ] * CMF_LTL['FI' ] * CMF_RTL['FI' ] * CMF_Lighting

                URow.setValue(F_CMFLight['name'], CMF_Lighting)
                URow.setValue(F_CMFSkew ['name'], CMF_Skew['TOT'])
                URow.setValue(F_CMFLTL  ['name'], CMF_LTL['TOT'])
                URow.setValue(F_CMFRTL  ['name'], CMF_RTL['TOT'])
                URow.setValue(F_TOT_PC   ['name'], TOT_PC  )
                URow.setValue(F_TOTk_PC  ['name'], SPF['TCk'])
                URow.setValue(F_FI_PC    ['name'], FI_PC   )
                URow.setValue(F_FIk_PC   ['name'], SPF['FIk'])
                URow.setValue(F_FIKAB_PC ['name'], FIKAB_PC)
                URow.setValue(F_FIKABk_PC['name'], SPF['FIKABk'])

            if Type == 'RM4ST':

                CMF_Lighting = CMFLighting(URow)
                CMF_Skew     = CMFSkew(URow)
                CMF_LTL      = CMFLTL(URow)
                CMF_RTL      = CMFRTL(URow)

                TOT_PC   = SPF['TOT'  ] * CMF_Skew['TOT'] * CMF_LTL['TOT'] * CMF_RTL['TOT'] * CMF_Lighting
                FI_PC    = SPF['FI'   ] * CMF_Skew['FI' ] * CMF_LTL['FI' ] * CMF_RTL['FI' ] * CMF_Lighting
                FIKAB_PC = SPF['FIKAB'] * CMF_Skew['FI' ] * CMF_LTL['FI' ] * CMF_RTL['FI' ] * CMF_Lighting

                URow.setValue(F_CMFLight['name'], CMF_Lighting)
                URow.setValue(F_CMFSkew ['name'], CMF_Skew['TOT'])
                URow.setValue(F_CMFLTL  ['name'], CMF_LTL['TOT'])
                URow.setValue(F_CMFRTL  ['name'], CMF_RTL['TOT'])
                URow.setValue(F_TOT_PC   ['name'], TOT_PC  )
                URow.setValue(F_TOTk_PC  ['name'], SPF['TCk'])
                URow.setValue(F_FI_PC    ['name'], FI_PC   )
                URow.setValue(F_FIk_PC   ['name'], SPF['FIk'])
                URow.setValue(F_FIKAB_PC ['name'], FIKAB_PC)
                URow.setValue(F_FIKABk_PC['name'], SPF['FIKABk'])

            if Type == 'RM4SG':

                TOT_PC   = SPF['TOT'  ]
                FI_PC    = SPF['FI'   ]
                FIKAB_PC = SPF['FIKAB']

                URow.setValue(F_TOT_PC   ['name'], TOT_PC  )
                URow.setValue(F_TOTk_PC  ['name'], SPF['TCk'])
                URow.setValue(F_FI_PC    ['name'], FI_PC   )
                URow.setValue(F_FIk_PC   ['name'], SPF['FIk'])
                URow.setValue(F_FIKAB_PC ['name'], FIKAB_PC)
                URow.setValue(F_FIKABk_PC['name'], SPF['FIKABk'])

            if Type == 'U3ST':

                CMF_Lighting = CMFLighting(URow)
                CMF_LTL = CMFLTL(URow)
                CMF_RTL = CMFRTL(URow)

                MV_PC    = SPF['MVC'   ] * CMF_LTL['TOT'] * CMF_RTL['TOT'] * CMF_Lighting
                MVFI_PC  = SPF['MVFIC' ] * CMF_LTL['TOT'] * CMF_RTL['TOT'] * CMF_Lighting
                MVPDO_PC = SPF['MVPDOC'] * CMF_LTL['TOT'] * CMF_RTL['TOT'] * CMF_Lighting
                SV_PC    = SPF['SVC'   ] * CMF_LTL['TOT'] * CMF_RTL['TOT'] * CMF_Lighting
                SVFI_PC  = SPF['SVFIC' ] * CMF_LTL['TOT'] * CMF_RTL['TOT'] * CMF_Lighting
                SVPDO_PC = SPF['SVPDOC'] * CMF_LTL['TOT'] * CMF_RTL['TOT'] * CMF_Lighting
                TOT_PC   = MV_PC   + SV_PC
                FI_PC    = MVFI_PC + SVFI_PC

                URow.setValue(F_CMFLight['name'], CMF_Lighting)
                URow.setValue(F_CMFLTL  ['name'], CMF_LTL['TOT'])
                URow.setValue(F_CMFRTL  ['name'], CMF_RTL['TOT'])
                URow.setValue(F_TOT_PC    ['name'], TOT_PC * GetVal(URow,'CCMF',1))
                URow.setValue(F_FI_PC     ['name'], FI_PC)
                URow.setValue(F_MV_PC     ['name'], MV_PC)
                URow.setValue(F_MVk_PC    ['name'], SPF['MVCk'])
                URow.setValue(F_MVFI_PC   ['name'], MVFI_PC)
                URow.setValue(F_MVFIk_PC  ['name'], SPF['MVFICk'])
                URow.setValue(F_MVPDO_PC  ['name'], MVPDO_PC)
                URow.setValue(F_MVPDOk_PC ['name'], SPF['MVPDOCk'])
                URow.setValue(F_SV_PC     ['name'], SV_PC)
                URow.setValue(F_SVk_PC    ['name'], SPF['SVCk'])
                URow.setValue(F_SVFI_PC   ['name'], SVFI_PC)
                URow.setValue(F_SVFIk_PC  ['name'], SPF['SVFICk'])
                URow.setValue(F_SVPDO_PC  ['name'], SVPDO_PC)
                URow.setValue(F_SVPDOk_PC ['name'], SPF['SVPDOCk'])

            if Type == 'U4ST':

                CMF_Lighting = CMFLighting(URow)
                CMF_LTL = CMFLTL(URow)
                CMF_RTL = CMFRTL(URow)

                MV_PC    = SPF['MVC'   ] * CMF_LTL['TOT'] * CMF_RTL['TOT'] * CMF_Lighting
                MVFI_PC  = SPF['MVFIC' ] * CMF_LTL['TOT'] * CMF_RTL['TOT'] * CMF_Lighting
                MVPDO_PC = SPF['MVPDOC'] * CMF_LTL['TOT'] * CMF_RTL['TOT'] * CMF_Lighting
                SV_PC    = SPF['SVC'   ] * CMF_LTL['TOT'] * CMF_RTL['TOT'] * CMF_Lighting
                SVFI_PC  = SPF['SVFIC' ] * CMF_LTL['TOT'] * CMF_RTL['TOT'] * CMF_Lighting
                SVPDO_PC = SPF['SVPDOC'] * CMF_LTL['TOT'] * CMF_RTL['TOT'] * CMF_Lighting
                TOT_PC   = MV_PC   + SV_PC
                FI_PC    = MVFI_PC + SVFI_PC

                URow.setValue(F_CMFLight['name'], CMF_Lighting)
                URow.setValue(F_CMFLTL  ['name'], CMF_LTL['TOT'])
                URow.setValue(F_CMFRTL  ['name'], CMF_RTL['TOT'])
                URow.setValue(F_TOT_PC    ['name'], TOT_PC * GetVal(URow,'CCMF',1))
                URow.setValue(F_FI_PC     ['name'], FI_PC)
                URow.setValue(F_MV_PC     ['name'], MV_PC)
                URow.setValue(F_MVk_PC    ['name'], SPF['MVCk'])
                URow.setValue(F_MVFI_PC   ['name'], MVFI_PC)
                URow.setValue(F_MVFIk_PC  ['name'], SPF['MVFICk'])
                URow.setValue(F_MVPDO_PC  ['name'], MVPDO_PC)
                URow.setValue(F_MVPDOk_PC ['name'], SPF['MVPDOCk'])
                URow.setValue(F_SV_PC     ['name'], SV_PC)
                URow.setValue(F_SVk_PC    ['name'], SPF['SVCk'])
                URow.setValue(F_SVFI_PC   ['name'], SVFI_PC)
                URow.setValue(F_SVFIk_PC  ['name'], SPF['SVFICk'])
                URow.setValue(F_SVPDO_PC  ['name'], SVPDO_PC)
                URow.setValue(F_SVPDOk_PC ['name'], SPF['SVPDOCk'])

            if Type == 'U3SG':

                CMF_Lighting = CMFLighting(URow)
                CMF_LTL = CMFLTL(URow)
                CMF_RTL = CMFRTL(URow)
                CMF_LTP = CMFLTP(URow)
                CMF_No_RTR = CMFNoRTR(URow)
    
                MV_PC    = SPF['MVC'   ] * CMF_LTL['TOT'] * CMF_RTL['TOT'] * CMF_Lighting * CMF_LTP * CMF_No_RTR
                MVFI_PC  = SPF['MVFIC' ] * CMF_LTL['TOT'] * CMF_RTL['TOT'] * CMF_Lighting * CMF_LTP * CMF_No_RTR
                MVPDO_PC = SPF['MVPDOC'] * CMF_LTL['TOT'] * CMF_RTL['TOT'] * CMF_Lighting * CMF_LTP * CMF_No_RTR
                SV_PC    = SPF['SVC'   ] * CMF_LTL['TOT'] * CMF_RTL['TOT'] * CMF_Lighting * CMF_LTP * CMF_No_RTR
                SVFI_PC  = SPF['SVFIC' ] * CMF_LTL['TOT'] * CMF_RTL['TOT'] * CMF_Lighting * CMF_LTP * CMF_No_RTR
                SVPDO_PC = SPF['SVPDOC'] * CMF_LTL['TOT'] * CMF_RTL['TOT'] * CMF_Lighting * CMF_LTP * CMF_No_RTR
                TOT_PC   = MV_PC   + SV_PC
                FI_PC    = MVFI_PC + SVFI_PC

                URow.setValue(F_CMFLight['name'], CMF_Lighting)
                URow.setValue(F_CMFLTL  ['name'], CMF_LTL['TOT'])
                URow.setValue(F_CMFRTL  ['name'], CMF_RTL['TOT'])
                URow.setValue(F_CMFLTP  ['name'], CMF_LTP)
                URow.setValue(F_CMFNoRTR['name'], CMF_No_RTR)
                URow.setValue(F_TOT_PC    ['name'], TOT_PC * GetVal(URow,'CCMF',1))
                URow.setValue(F_FI_PC     ['name'], FI_PC)
                URow.setValue(F_MV_PC     ['name'], MV_PC)
                URow.setValue(F_MVk_PC    ['name'], SPF['MVCk'])
                URow.setValue(F_MVFI_PC   ['name'], MVFI_PC)
                URow.setValue(F_MVFIk_PC  ['name'], SPF['MVFICk'])
                URow.setValue(F_MVPDO_PC  ['name'], MVPDO_PC)
                URow.setValue(F_MVPDOk_PC ['name'], SPF['MVPDOCk'])
                URow.setValue(F_SV_PC     ['name'], SV_PC)
                URow.setValue(F_SVk_PC    ['name'], SPF['SVCk'])
                URow.setValue(F_SVFI_PC   ['name'], SVFI_PC)
                URow.setValue(F_SVFIk_PC  ['name'], SPF['SVFICk'])
                URow.setValue(F_SVPDO_PC  ['name'], SVPDO_PC)
                URow.setValue(F_SVPDOk_PC ['name'], SPF['SVPDOCk'])

            if Type == 'U4SG':

                CMF_Lighting = CMFLighting(URow)
                CMF_LTL = CMFLTL(URow)
                CMF_RTL = CMFRTL(URow)
                CMF_LTP = CMFLTP(URow)
                CMF_No_RTR = CMFNoRTR(URow)
                CMF_Bus = CMFBus(URow)
                CMF_School = CMFSchool(URow)
                CMF_Alco = CMFAlco(URow)

                MV_PC    = SPF['MVC'   ] * CMF_LTL['TOT'] * CMF_RTL['TOT'] * CMF_Lighting * CMF_LTP * CMF_No_RTR
                MVFI_PC  = SPF['MVFIC' ] * CMF_LTL['TOT'] * CMF_RTL['TOT'] * CMF_Lighting * CMF_LTP * CMF_No_RTR
                MVPDO_PC = SPF['MVPDOC'] * CMF_LTL['TOT'] * CMF_RTL['TOT'] * CMF_Lighting * CMF_LTP * CMF_No_RTR
                SV_PC    = SPF['SVC'   ] * CMF_LTL['TOT'] * CMF_RTL['TOT'] * CMF_Lighting * CMF_LTP * CMF_No_RTR
                SVFI_PC  = SPF['SVFIC' ] * CMF_LTL['TOT'] * CMF_RTL['TOT'] * CMF_Lighting * CMF_LTP * CMF_No_RTR
                SVPDO_PC = SPF['SVPDOC'] * CMF_LTL['TOT'] * CMF_RTL['TOT'] * CMF_Lighting * CMF_LTP * CMF_No_RTR
                Ped_PC   = SPF['Ped'   ] * CMF_Bus * CMF_School * CMF_Alco
                TOT_PC   = MV_PC   + SV_PC   + Ped_PC
                FI_PC    = MVFI_PC + SVFI_PC

                URow.setValue(F_CMFLight['name'], CMF_Lighting)
                URow.setValue(F_CMFLTL  ['name'], CMF_LTL['TOT'])
                URow.setValue(F_CMFRTL  ['name'], CMF_RTL['TOT'])
                URow.setValue(F_CMFLTP  ['name'], CMF_LTP)
                URow.setValue(F_CMFNoRTR['name'], CMF_No_RTR)
                URow.setValue(F_CMFBus  ['name'], CMF_Bus)
                URow.setValue(F_CMFSchool['name'], CMF_School)
                URow.setValue(F_CMFAlco  ['name'], CMF_Alco)
                URow.setValue(F_TOT_PC    ['name'], TOT_PC * GetVal(URow,'CCMF',1))
                URow.setValue(F_FI_PC     ['name'], FI_PC)
                URow.setValue(F_MV_PC     ['name'], MV_PC)
                URow.setValue(F_MVk_PC    ['name'], SPF['MVCk'])
                URow.setValue(F_MVFI_PC   ['name'], MVFI_PC)
                URow.setValue(F_MVFIk_PC  ['name'], SPF['MVFICk'])
                URow.setValue(F_MVPDO_PC  ['name'], MVPDO_PC)
                URow.setValue(F_MVPDOk_PC ['name'], SPF['MVPDOCk'])
                URow.setValue(F_SV_PC     ['name'], SV_PC)
                URow.setValue(F_SVk_PC    ['name'], SPF['SVCk'])
                URow.setValue(F_SVFI_PC   ['name'], SVFI_PC)
                URow.setValue(F_SVFIk_PC  ['name'], SPF['SVFICk'])
                URow.setValue(F_SVPDO_PC  ['name'], SVPDO_PC)
                URow.setValue(F_SVPDOk_PC ['name'], SPF['SVPDOCk'])
                URow.setValue(F_Ped_PC    ['name'], Ped_PC)
                URow.setValue(F_Pedk_PC   ['name'], SPF['Pedk'])
        
            if Type in ['R2U']:

                CMF_Grade = CMFGrade('R2U',GetVal(URow,F_Grade['name'],0))

                CMF_LW    = GetVal(URow,F_CMFLW['name'],1)
                CMF_SW    = GetVal(URow,F_CMFSW['name'],1)
                CMF_HurC  = GetVal(URow,F_CMFHorCur['name'],1)
                CMF_DD    = GetVal(URow,F_CMFDrwDens['name'],1)
                CMF_RHR   = GetVal(URow,F_CMFRHR['name'],1)
                CMF_Light = GetVal(URow,F_CMFLight['name'],1)
                
                URow.setValue(F_TOT_PC     ['name'], SPF['TOT']      * CMF_LW * CMF_SW * CMF_HurC * CMF_Grade * CMF_DD * CMF_RHR * CMF_Light)

                URow.setValue(F_CMFGrade['name'], CMF_Grade)

            if Type in ['R4D']:

                CMF_LW    = GetVal(URow,F_CMFLW['name'],1)
                CMF_SW    = GetVal(URow,F_CMFSW['name'],1)
                CMF_MW    = GetVal(URow,F_CMFMW['name'],1)
                CMF_Light = GetVal(URow,F_CMFLight['name'],1)
                
                URow.setValue(F_TOT_PC     ['name'], SPF['TOT']      * CMF_LW * CMF_SW * CMF_MW * CMF_Light)
                URow.setValue(F_FI_PC      ['name'], SPF['FI' ]      * CMF_LW * CMF_SW * CMF_MW * CMF_Light)

            if Type in ['R4U']:

                CMF_LW    = GetVal(URow,F_CMFLW['name'],1)
                CMF_SW    = GetVal(URow,F_CMFSW['name'],1)
                CMF_Light = GetVal(URow,F_CMFLight['name'],1)
                
                URow.setValue(F_TOT_PC     ['name'], SPF['TOT']      * CMF_LW * CMF_SW * CMF_Light)
                URow.setValue(F_FI_PC      ['name'], SPF['FI' ]      * CMF_LW * CMF_SW * CMF_Light)

            if Type in ['U2U', 'U3T', 'U4D', 'U4U', 'U5T']:

                CMF_OSP   = GetVal(URow,F_CMFOSP['name'],1)
                CMF_MW    = GetVal(URow,F_CMFMW['name'],1)
                CMF_Light = GetVal(URow,F_CMFLight['name'],1)
                CMF_FO    = GetVal(URow,F_CMFFO['name'],1)

                URow.setValue(F_TOT_PC     ['name'], SPF['TOT']      * CMF_OSP * CMF_FO * CMF_MW * CMF_Light * GetVal(URow,'CCMF',1))
                URow.setValue(F_FI_PC      ['name'], SPF['FI']       * CMF_OSP * CMF_FO * CMF_MW * CMF_Light)
                URow.setValue(F_MV_PC      ['name'], SPF['MVC']      * CMF_OSP * CMF_FO * CMF_MW * CMF_Light)
                URow.setValue(F_MVFI_PC    ['name'], SPF['MVFIC']    * CMF_OSP * CMF_FO * CMF_MW * CMF_Light)
                URow.setValue(F_MVPDO_PC   ['name'], SPF['MVPDOC']   * CMF_OSP * CMF_FO * CMF_MW * CMF_Light)

                URow.setValue(F_MVd_PC     ['name'], SPF['MVdC']     * CMF_OSP * CMF_FO * CMF_MW * CMF_Light)
                URow.setValue(F_MVFId_PC   ['name'], SPF['MVFIdC']   * CMF_OSP * CMF_FO * CMF_MW * CMF_Light)
                URow.setValue(F_MVPDOd_PC  ['name'], SPF['MVPDOdC']  * CMF_OSP * CMF_FO * CMF_MW * CMF_Light)
                URow.setValue(F_MVdk_PC    ['name'], SPF['MVdCk'])

                URow.setValue(F_MVnd_PC    ['name'], SPF['MVndC']    * CMF_OSP * CMF_FO * CMF_MW * CMF_Light)
                URow.setValue(F_MVFInd_PC  ['name'], SPF['MVFIndC']  * CMF_OSP * CMF_FO * CMF_MW * CMF_Light)
                URow.setValue(F_MVPDOnd_PC ['name'], SPF['MVPDOndC'] * CMF_OSP * CMF_FO * CMF_MW * CMF_Light)
                URow.setValue(F_MVndk_PC   ['name'], SPF['MVndCk'])
                URow.setValue(F_MVFIndk_PC ['name'], SPF['MVFIndCk'])
                URow.setValue(F_MVPDOndk_PC['name'], SPF['MVPDOndCk'])

                URow.setValue(F_SV_PC      ['name'], SPF['SVC']      * CMF_OSP * CMF_FO * CMF_MW * CMF_Light)
                URow.setValue(F_SVFI_PC    ['name'], SPF['SVFIC']    * CMF_OSP * CMF_FO * CMF_MW * CMF_Light)
                URow.setValue(F_SVPDO_PC   ['name'], SPF['SVPDOC']   * CMF_OSP * CMF_FO * CMF_MW * CMF_Light)
                URow.setValue(F_SVk_PC     ['name'], SPF['SVCk'])
                URow.setValue(F_SVFIk_PC   ['name'], SPF['SVFICk'])
                URow.setValue(F_SVPDOk_PC  ['name'], SPF['SVPDOCk'])

            if Type in ['R4F', 'U4F', 'U6F']:
                CMFHC = {'MVFI' :GetVal(URow,F_CMFHCMVFI ['name'],1),
                         'MVPDO':GetVal(URow,F_CMFHCMVPD ['name'],1),
                         'SVFI' :GetVal(URow,F_CMFHCSVFI ['name'],1),
                         'SVPDO':GetVal(URow,F_CMFHCSVPD ['name'],1)}

                LaneWidth = GetVal(URow,F_Lane_Width['name'],12)
                InShWidth  = (GetVal(URow,F_Sh_Wid_RI['name'],6 )+GetVal(URow,F_Sh_Wid_LI['name'],6 ))/2
                OutShWidth = (GetVal(URow,F_Sh_Wid_RO['name'],10)+GetVal(URow,F_Sh_Wid_LO['name'],10))/2

                CMFLW = CMFLaneWidth(Type,LaneWidth,0)
                URow.setValue(F_CMFLW['name'],CMFLW) 

                CMFMWB = CMFMedianBarrier(Type,GetVal(URow,'MEDIAN_WID',60),GetVal(URow,'MEDIAN_ID',1),InShWidth)
                URow.setValue(F_CMFMWBMFI['name'],CMFMWB['MVFI' ]) 
                URow.setValue(F_CMFMWBMPD['name'],CMFMWB['MVPDO']) 
                URow.setValue(F_CMFMWBSFI['name'],CMFMWB['SVFI' ]) 
                URow.setValue(F_CMFMWBSPD['name'],CMFMWB['SVPDO']) 

                CMFInSW  = CMFInsideShoulderW (Type,InShWidth )
                CMFOutSW = CMFOutsideShoulderW(Type,OutShWidth)
                URow.setValue(F_CMFInSWFI['name'],CMFInSW['FI' ]) 
                URow.setValue(F_CMFInSWPD['name'],CMFInSW['PDO']) 
                URow.setValue(F_CMFOutSWFI['name'],CMFOutSW['FI' ]) 
                URow.setValue(F_CMFOutSWPD['name'],CMFOutSW['PDO']) 

                #Clearzone
                CMFCZ = GetVal(URow,F_CMFCZFI ['name'],1)
                CMFOB = {'SVFI' :GetVal(URow,F_CMFOBFI ['name'],1),
                         'SVPDO':GetVal(URow,F_CMFOBPDO['name'],1)}
                CMFRS = 0.923
                URow.setValue(F_CMFRSSVFI['name'],CMFRS) 


                Phv = GetVal(URow,F_PHighVol['name'],0)
                CMF_HV = CMFHighVolume(Type,Phv)
                URow.setValue(F_CMFHVSVFI ['name'],CMF_HV['SVFI' ]) 
                URow.setValue(F_CMFHVSVPDO['name'],CMF_HV['SVPDO']) 
                URow.setValue(F_CMFHVMVFI ['name'],CMF_HV['MVFI' ]) 
                URow.setValue(F_CMFHVMVPDO['name'],CMF_HV['MVPDO']) 


                SPF_MVFI  = SPF['MVFIC' ]*CMFHC['MVFI' ]*CMFLW*CMFInSW['FI' ]                *CMFMWB['MVFI' ] * CMF_HV['MVFI' ]
                SPF_MVPDO = SPF['MVPDOC']*CMFHC['MVPDO']      *CMFInSW['PDO']                *CMFMWB['MVPDO'] * CMF_HV['MVPDO']
                SPF_SVFI  = SPF['SVFIC' ]*CMFHC['SVFI' ]*CMFLW*CMFInSW['FI' ]*CMFOutSW['FI' ]*CMFMWB['SVFI' ]*CMFOB['SVFI' ]*CMFCZ*CMFRS * CMF_HV['SVFI' ]
                SPF_SVPDO = SPF['SVPDOC']*CMFHC['SVPDO']      *CMFInSW['PDO']*CMFOutSW['PDO']*CMFMWB['SVPDO']*CMFOB['SVPDO']             * CMF_HV['SVPDO']
                URow.setValue(F_MVFI_PC    ['name'], SPF_MVFI)
                URow.setValue(F_MVPDO_PC   ['name'], SPF_MVPDO)

                URow.setValue(F_SVFI_PC    ['name'], SPF_SVFI)
                URow.setValue(F_SVPDO_PC   ['name'], SPF_SVPDO)

                URow.setValue(F_MV_PC      ['name'], SPF_MVFI+SPF_MVPDO)
                URow.setValue(F_SV_PC      ['name'], SPF_SVFI+SPF_SVPDO)
                URow.setValue(F_TOT_PC     ['name'], SPF_MVFI+SPF_MVPDO+SPF_SVFI+SPF_SVPDO)
                URow.setValue(F_FI_PC      ['name'], SPF_SVFI+SPF_MVFI)

        UC.updateRow(URow)

    Output = arcpy.CopyFeatures_management(SitesLayer,Output)

    arcpy.AddMessage("   --> Done.")
    arcpy.AddMessage(" ")


# Roadway segment analysis
def HorCurvature(Dic):
    import math
    Inf = 10**10 
    def Radius(P1,P2,P3):
        Inf = 10**10
        x1 = P1[0];x2 = P2[0];x3 = P3[0]
        y1 = P1[1];y2 = P2[1];y3 = P3[1]
        if y1==y2 and y2==y3:
            R = Inf
        elif y1==y2 and y2<>y3:
            m2 = -(x3-x2)/(y3-y2)
            xm1 = (x1+x2)/2
            ym1 = (y1+y2)/2
            xm2 = (x3+x2)/2
            ym2 = (y3+y2)/2
            xc = xm1
            yc = m2*(xc-xm2)+ym2
            R  = math.sqrt((xc-x1)**2+(yc-y1)**2)
        elif y2==y3 and y1<>y2:
            m1 = -(x2-x1)/(y2-y1)
            xm1 = (x1+x2)/2
            ym1 = (y1+y2)/2
            xm2 = (x3+x2)/2
            ym2 = (y3+y2)/2
            xc = xm2
            yc = m1*(xc-xm1)+ym1
            R  = math.sqrt((xc-x1)**2+(yc-y1)**2)
        elif y1<>y2 and y3<>y2:
            if y3 == y1:
                R = Inf
            elif y3<>y1:
                if (x3-x1)/(y3-y1) == (x2-x1)/(y2-y1):
                    R = Inf
                else:
                    m1 = -(x2-x1)/(y2-y1)
                    m2 = -(x3-x2)/(y3-y2)
                    xm1 = (x1+x2)/2
                    ym1 = (y1+y2)/2
                    xm2 = (x3+x2)/2
                    ym2 = (y3+y2)/2
                    xc = (ym1-ym2+m2*xm2-m1*xm1)/(m2-m1)
                    yc = m1*(xc-xm1)+ym1
                    R  = math.sqrt((xc-x1)**2+(yc-y1)**2)
        return(R)
    def Length(P1,P2):
        return(math.sqrt((P2[0]-P1[0])**2+(P2[1]-P1[1])**2))
    Vertices = Dic['Path']
    R = []
    L = Dic['Length']
    for i in range(2,len(Vertices)):
        l1 = Length(Vertices[i-2],Vertices[i-1])
        l2 = Length(Vertices[i-1],Vertices[i  ])
        R.append(Radius(Vertices[i-2],Vertices[i-1],Vertices[i]))
    if L<0.05*5280:
        R = []
    if len(R) == 0:
        R.append(Inf)
    return(R)
def VerCurvature(Dic):
    L = Dic['Length']
    Z = Dic['MeanZ' ]
    Lf = 0
    Zf = 0
    for List in Dic['FirstPoint']:
        Lf += List[0]/len(Dic['FirstPoint'])
        Zf += List[1]/len(Dic['FirstPoint'])
    Ll = 0
    Zl = 0
    for List in Dic['LastPoint']:
        Ll += List[0]/len(Dic['LastPoint'])
        Zl += List[1]/len(Dic['LastPoint'])
    if Lf==0 and Ll==0:
        G = 0
    if Lf<>0 and Ll<>0:
        G = max([abs(Z-Zf)/(L+Lf)*2,abs(Z-Zl)/(L+Ll)*2])
    if Lf==0 and Ll<>0:
        G = abs(Z-Zl)/(L+Ll)*2
    if Lf<>0 and Ll==0:
        G = abs(Z-Zf)/(L+Lf)*2
    if G>1:
        arcpy.AddMessage(str([L,Z,Ll,Zl,Lf,Zf,G]))
    return(G)

## Import Data Elements
def ImportGrade(SiteInput,MeanZField,Output):
    import json

    arcpy.AddMessage(" ")
    arcpy.AddMessage("    Import Grade")

    OutDic = OutputParser(Output, 'shp')
    Output = OutDic['folder'] + '\\' + OutDic['file']

    arcpy.AddMessage("    Copy Features: Roads Layer")
    SiteLayer = CopyFeatures(SiteInput,'CURVEInput')
    JoinLayer = CopyFeatures(SiteLayer,'CURVEJoin')
    TotalSites = arcpy.GetCount_management(SiteLayer)
    arcpy.AddMessage("     - Total number of Items Found: " + str(TotalSites))
    
    arcpy.AddMessage("    Add Fields: Roads Layer")
    FieldList1 = [F_TargetFID,F_Grade]
    for field in FieldList1:
        AddField(SiteLayer,field)

    FieldList2 = [F_JoinFID]
    for field in FieldList2:
        AddField(JoinLayer,field)

    arcpy.AddMessage("    Calculate Fields: Roads Layer")
    i = 0
    UC = arcpy.UpdateCursor(SiteLayer)
    for URow in UC:
        URow.setValue(F_TargetFID['name'],i)
        i += 1
        UC.updateRow(URow)

    i = 0
    UC = arcpy.UpdateCursor(JoinLayer)
    for URow in UC:
        URow.setValue(F_JoinFID['name'],i)
        i += 1
        UC.updateRow(URow)

    arcpy.AddMessage("    Spatial Join: Sites Layer")
    USPJ = SpatialJoin(SiteLayer, JoinLayer, 'CurveSPJ', {'join_operation': "JOIN_ONE_TO_MANY", 'join_type': "KEEP_ALL", 'field_mapping': '', 'match_option': 'INTERSECT', 'search_radius': '', 'distance_field_name': ''})


    arcpy.AddMessage("    Search Cursor: Roads Layer")
    ConnDic  = {SRow.getValue(F_TargetFID['name']):[] for SRow in arcpy.SearchCursor(USPJ)}
    SC = arcpy.SearchCursor(USPJ)
    for SRow in SC:
        TFID = SRow.getValue(F_TargetFID['name'])
        JFID = SRow.getValue(F_JoinFID  ['name'])
        if (not JFID in ConnDic[TFID]) and (JFID<>TFID):
           ConnDic[TFID].append(JFID)

    PointDic = {SRow.getValue(F_TargetFID['name']):{'FirstPoint':SRow.getValue('Shape').firstPoint,'LastPoint':SRow.getValue('Shape').lastPoint                        } for SRow in arcpy.SearchCursor(SiteLayer)}
    FinalDic = {SRow.getValue(F_TargetFID['name']):{'Length'    :SRow.getValue('Shape').length    ,'MeanZ'    :SRow.getValue(MeanZField),'Path': json.loads(SRow.getValue('Shape').JSON)['paths'][0],'FirstPoint':[],'LastPoint':[]} for SRow in arcpy.SearchCursor(SiteLayer)}


    for TFID in FinalDic.keys():
        for JFID in ConnDic[TFID]:
            if PointDic[TFID]['FirstPoint'].equals(PointDic[JFID]['FirstPoint']) or PointDic[TFID]['FirstPoint'].equals(PointDic[JFID]['LastPoint']):
                FinalDic[TFID]['FirstPoint'].append([FinalDic[JFID]['Length'],FinalDic[JFID]['MeanZ']])
            if PointDic[TFID]['LastPoint'].equals(PointDic[JFID]['FirstPoint']) or PointDic[TFID]['LastPoint'].equals(PointDic[JFID]['LastPoint']):
                FinalDic[TFID]['LastPoint'].append([FinalDic[JFID]['Length'],FinalDic[JFID]['MeanZ']])

    arcpy.AddMessage("    Update Cursor: Roads Layer")
    UC = arcpy.UpdateCursor(SiteLayer)
    for URow in UC:
        TFID = URow.getValue(F_TargetFID['name'])        
        G = VerCurvature(FinalDic[TFID])
        URow.setValue(F_Grade ['name'],G)
        
        UC.updateRow(URow)
    arcpy.DeleteField_management(SiteLayer,F_TargetFID['name'])

    arcpy.CopyFeatures_management(SiteLayer, Output)
    arcpy.AddMessage("   --> Done.")
    arcpy.AddMessage(" ")
def ImportCurvature(SiteInput,Output):
    import json

    arcpy.AddMessage(" ")
    arcpy.AddMessage("    Import Curvature")

    OutDic = OutputParser(Output, 'shp')
    Output = OutDic['folder'] + '\\' + OutDic['file']

    arcpy.AddMessage("    Copy Features: Roads Layer")
    SiteLayer = CopyFeatures(SiteInput,'CURVEInput')
    JoinLayer = CopyFeatures(SiteLayer,'CURVEJoin')
    TotalSites = arcpy.GetCount_management(SiteLayer)
    arcpy.AddMessage("     - Total number of Items Found: " + str(TotalSites))
    
    arcpy.AddMessage("    Add Fields: Roads Layer")
    FieldList1 = [F_TargetFID,F_HorCur,F_CMFHorCur, F_CMFHCMVFI, F_CMFHCMVPD, F_CMFHCSVFI, F_CMFHCSVPD]
    for field in FieldList1:
        AddField(SiteLayer,field)

    FieldList2 = [F_JoinFID]
    for field in FieldList2:
        AddField(JoinLayer,field)

    arcpy.AddMessage("    Calculate Fields: Roads Layer")
    i = 0
    UC = arcpy.UpdateCursor(SiteLayer)
    for URow in UC:
        URow.setValue(F_TargetFID['name'],i)
        i += 1
        UC.updateRow(URow)

    i = 0
    UC = arcpy.UpdateCursor(JoinLayer)
    for URow in UC:
        URow.setValue(F_JoinFID['name'],i)
        i += 1
        UC.updateRow(URow)

    arcpy.AddMessage("    Spatial Join: Sites Layer")
    USPJ = SpatialJoin(SiteLayer, JoinLayer, 'CurveSPJ', {'join_operation': "JOIN_ONE_TO_MANY", 'join_type': "KEEP_ALL", 'field_mapping': '', 'match_option': 'INTERSECT', 'search_radius': '', 'distance_field_name': ''})


    arcpy.AddMessage("    Search Cursor: Roads Layer")
    ConnDic  = {SRow.getValue(F_TargetFID['name']):[] for SRow in arcpy.SearchCursor(USPJ)}
    SC = arcpy.SearchCursor(USPJ)
    for SRow in SC:
        TFID = SRow.getValue(F_TargetFID['name'])
        JFID = SRow.getValue(F_JoinFID  ['name'])
        if (not JFID in ConnDic[TFID]) and (JFID<>TFID):
           ConnDic[TFID].append(JFID)

    PointDic = {SRow.getValue(F_TargetFID['name']):{'FirstPoint':SRow.getValue('Shape').firstPoint,'LastPoint':SRow.getValue('Shape').lastPoint                        } for SRow in arcpy.SearchCursor(SiteLayer)}
    FinalDic = {SRow.getValue(F_TargetFID['name']):{'Length'    :SRow.getValue('Shape').length    ,'Path': json.loads(SRow.getValue('Shape').JSON)['paths'][0],'FirstPoint':[],'LastPoint':[]} for SRow in arcpy.SearchCursor(SiteLayer)}


    for TFID in FinalDic.keys():
        for JFID in ConnDic[TFID]:
            if PointDic[TFID]['FirstPoint'].equals(PointDic[JFID]['FirstPoint']) or PointDic[TFID]['FirstPoint'].equals(PointDic[JFID]['LastPoint']):
                FinalDic[TFID]['FirstPoint'].append([FinalDic[JFID]['Length'],1])
            if PointDic[TFID]['LastPoint'].equals(PointDic[JFID]['FirstPoint']) or PointDic[TFID]['LastPoint'].equals(PointDic[JFID]['LastPoint']):
                FinalDic[TFID]['LastPoint'].append([FinalDic[JFID]['Length'],1])

    arcpy.AddMessage("    Update Cursor: Roads Layer")
    UC = arcpy.UpdateCursor(SiteLayer)
    for URow in UC:
        TFID = URow.getValue(F_TargetFID['name'])        
        R = HorCurvature(FinalDic[TFID])
        URow.setValue(F_HorCur['name'],1/(float(sum(R)/max(len(R),1)))*5280)
        CMF = CMFHorCurve(URow.getValue(F_FType['name']),URow.getValue('Shape').length,1/(float(sum(R)/max(len(R),1)))*5280,0)
        URow.setValue(F_CMFHorCur['name'],CMF['TOT'])
        URow.setValue(F_CMFHCMVFI['name'],CMF['MVFI'])
        URow.setValue(F_CMFHCMVPD['name'],CMF['MVPDO'])
        URow.setValue(F_CMFHCSVFI['name'],CMF['SVFI'])
        URow.setValue(F_CMFHCSVPD['name'],CMF['SVPDO'])

        UC.updateRow(URow)
    arcpy.DeleteField_management(SiteLayer,F_TargetFID['name'])

    arcpy.CopyFeatures_management(SiteLayer, Output)
    arcpy.AddMessage("   --> Done.")
    arcpy.AddMessage(" ")
def ImportDriveway(SiteInput,CMFInput,DWField,DBuffer,Output):
    arcpy.AddMessage(" ")
    arcpy.AddMessage("    Import Driveway")

    OutDic = OutputParser(Output, 'shp')
    Output = OutDic['folder'] + '\\' + OutDic['file']

    arcpy.AddMessage("    Copy Features: Sites Layer")
    SiteLayer = CopyFeatures(SiteInput,'DWSite')
    TotalSites = arcpy.GetCount_management(SiteLayer)
    arcpy.AddMessage("     - Total number of Sites Found: " + str(TotalSites))

    arcpy.AddMessage("    Copy Features: Driveway Layer")
    CMFLayer = CopyFeatures(CMFInput,'DWCMF')
    TotalCMF = arcpy.GetCount_management(CMFLayer)
    arcpy.AddMessage("     - Total number of Driveway Items Found: " + str(TotalCMF))

    arcpy.AddMessage("    Add Fields: Sites Layer")
    FieldList1 = [F_RouteFID, F_dMjC   , F_dMnC, F_dMjI, F_dMnI, F_dMjR, F_dMnR, F_dO, F_DrwDens, F_CMFDrwDens]
    for Field in FieldList1:
        AddField(SiteLayer,Field)
    # Save FIDs in RouteFID
    UC = arcpy.UpdateCursor(SiteLayer)
    for URow in UC:
        FID = GetFID(URow)
        URow.setValue(F_RouteFID['name'],FID)
        UC.updateRow(URow)
    # Creating a dictionary with keys = Sites FIDs
    FIDIntersect = {SRow.getValue(F_RouteFID['name']):[] for SRow in arcpy.SearchCursor(SiteLayer)}

    arcpy.AddMessage("    Add Fields: Driveway Layer")
    FieldList2 = [F_RouteFID, F_JoinFID]
    for Field in FieldList2:
        AddField(CMFLayer,Field)
    # Save Driveway FIDs in JoinFID
    UC = arcpy.UpdateCursor(CMFLayer)
    for URow in UC:
        FID = GetFID(URow)
        URow.setValue(F_JoinFID['name'],FID)
        UC.updateRow(URow)

    arcpy.AddMessage("    Buffer: Sites Layer")
    SBuffer = Buffer(SiteLayer,"DWBuffer",{'distance': DBuffer, 'line_side': 'FULL', 'line_end_type': 'FLAT', 'dissolve_option': 'NONE', 'dissolve_field': '#'})

    arcpy.AddMessage("    Intersect: Buffer + Driveway Layer")
    BI = Intersect(SBuffer + ';' + CMFLayer, "DWIntersect", {'join_attributes': "ALL", 'cluster_tolerance': '', 'output_type': "POINT"})
    # Save Intersecting driveways for each Site
    for SRow in arcpy.SearchCursor(BI):
        FIDIntersect[SRow.getValue(F_RouteFID['name'])].append(SRow.getValue(F_JoinFID['name']))

    arcpy.AddMessage("    Update Cursor: Site Layer & Driveway Layer")
    # Updates Sites layer for number of driveways for each type; and updates
    # driveways layer with corresponding site layer
    SitesWithData = 0
    UC1 = arcpy.UpdateCursor(SiteLayer)
    for URow1 in UC1:
        SFID = URow1.getValue(F_RouteFID['name'])
        T = {'dMjC': 0, 'dMnC': 0, 'dMjI': 0, 'dMnI': 0, 'dMjR': 0, 'dMnR': 0, 'dO': 0}
        Flag = True
        for CMFFID in FIDIntersect[SFID]:
            UC2 = arcpy.UpdateCursor(CMFLayer,'"' + F_JoinFID['name'] + '" = ' + str(CMFFID))
            URow2 = UC2.next()                
            URow2.setValue(F_RouteFID['name'], SFID)
            try:
                DW = int(URow2.getValue(DWField))
            except:
                arcpy.AddWarning("    Could not change " + DWField + ' = ' + str(URow2.getValue(DWField)) + ' to integer; Default value assumed: 7')
                DW = 7

            if   DW == 1:
                T['dMjC'] += 1
            elif DW == 2:
                T['dMnC'] += 1
            elif DW == 3:
                T['dMjI'] += 1
            elif DW == 4:
                T['dMnI'] += 1
            elif DW == 5:
                T['dMjR'] += 1
            elif DW == 6:
                T['dMnR'] += 1
            else:
                T['dO']   += 1
            if Flag:
                SitesWithData += 1
                Flag = False
            UC2.updateRow(URow2)
            del UC2
        Length = URow1.getValue('Shape').length/5280
        URow1.setValue(F_dMjC['name'],T['dMjC'])
        URow1.setValue(F_dMnC['name'],T['dMnC'])
        URow1.setValue(F_dMjI['name'],T['dMjI'])
        URow1.setValue(F_dMnI['name'],T['dMnI'])
        URow1.setValue(F_dMjR['name'],T['dMjR'])
        URow1.setValue(F_dMnR['name'],T['dMnR'])
        URow1.setValue(F_dO['name'],T['dO'])
        URow1.setValue(F_DrwDens['name'],sum(T.values())/Length)
        URow1.setValue(F_CMFDrwDens['name'],CMFDrivewayDensity(URow1.getValue(F_FType['name']),sum(T.values())/Length,URow1.getValue(F_AADT['name'])))
        
        UC1.updateRow(URow1)
        if not Flag:
            arcpy.AddMessage('     - FID: ' + str(SFID) + ', Tot Driveways: ' + str(sum(T.values())))
    arcpy.AddMessage("    Total Sites with Data: " + str(SitesWithData))

    for Field in [F_RouteFID]:
        arcpy.DeleteField_management(SiteLayer,Field['name'])

    for Field in [F_JoinFID]:
        arcpy.DeleteField_management(CMFLayer,Field['name'])

    arcpy.CopyFeatures_management(SiteLayer, Output)
    arcpy.AddMessage("   --> Done.")
    arcpy.AddMessage(" ")
def ImportFixedObjects(SiteInput, CMFDataInput, DCField, FOField, OffsetField, Output):
    arcpy.AddMessage(" ")
    arcpy.AddMessage("    Import Fixed Objects")

    OutDic = OutputParser(Output, 'shp')
    Output = OutDic['folder'] + '\\' + OutDic['file']

    arcpy.AddMessage("    Copy Features: Sites Layer")
    SiteLayer = CopyFeatures(SiteInput,'SiteInput')
    TotalSites = arcpy.GetCount_management(SiteLayer)
    arcpy.AddMessage("     - Total number of Sites Found: " + str(TotalSites))

    arcpy.AddMessage("    Copy Features: Fixed Object Data Layer")
    CMFDataLayer = CopyFeatures(CMFDataInput,'FOInput')
    TotalCMF = arcpy.GetCount_management(CMFDataLayer)
    arcpy.AddMessage("     - Total number of Fixed Objects Found: " + str(TotalCMF))

    arcpy.AddMessage("    Add Fields: Sites Layer")
    FieldList1 = [F_RouteFID, F_CMFFO, F_FODensity, F_FOOffset]
    for Field in FieldList1:
        AddField(SiteLayer,Field,'append')
    UC = arcpy.UpdateCursor(SiteLayer)
    for URow in UC:
        FID = GetFID(URow)
        URow.setValue(F_RouteFID['name'],FID)
        UC.updateRow(URow)

    arcpy.AddMessage("    Add Fields: Fixed Objects Layer")
    FieldList2 = [F_RouteFID, F_SegLength, F_CMFFID, F_CMFLength]
    for Field in FieldList2:
        AddField(CMFDataLayer,Field)
    UC = arcpy.UpdateCursor(CMFDataLayer)
    for URow in UC:
        FID = GetFID(URow)
        URow.setValue(F_CMFFID['name'],FID)
        UC.updateRow(URow)

    arcpy.AddMessage("    Buffer: Sites Layer")
    FBuffer = 30 + 1.1 * MaximumValue(SiteLayer, F_RBuffer['name'])
    if FBuffer<250: FBuffer = 250
    LeftBuffer = Buffer(SiteLayer,"FOLeftBuffer" ,{'distance': FBuffer, 'line_side': 'LEFT' , 'line_end_type': 'FLAT', 'dissolve_option': 'NONE', 'dissolve_field': '#'})
    RightBuffer = Buffer(SiteLayer,"FORightBuffer",{'distance': FBuffer, 'line_side': 'RIGHT', 'line_end_type': 'FLAT', 'dissolve_option': 'NONE', 'dissolve_field': '#'})

    arcpy.AddMessage("    Intersect: Sites Layer + Fixed Obj Layer")
    LeftBI = Intersect(LeftBuffer + ';' + CMFDataLayer,"FOLeftIntersect" ,{'join_attributes': "ALL", 'cluster_tolerance': '', 'output_type': "LINE"})
    RightBI = Intersect(RightBuffer + ';' + CMFDataLayer,"FORightIntersect",{'join_attributes': "ALL", 'cluster_tolerance': '', 'output_type': "LINE"})

    arcpy.AddMessage("    Search Cursor: Sites Layer, Update Cursor: Fixed Obj Layer")
    # Saves a list of fixed objects in NumFO array, updates road FID in Fixed obj
    # layer
    def GetValFloat(Row, Field, Default):
         try:
             return float(Row.getValue(Field))
         except:
             arcpy.AddMessage("    Could not change " + Field + ' = ' + str(Row.getValue(Field)) + ' to float; Default value assumed: ' + str(Default))
    NumFO = []
    SitesWithData = 0
    SC1 = arcpy.UpdateCursor(SiteLayer)
    for SRow1 in SC1:
        SFID = SRow1.getValue(F_RouteFID['name'])
        Flag = True
        Left = []
        SC2 = arcpy.SearchCursor(LeftBI,'"' + F_RouteFID['name'] + '" = ' + str(SFID))
        for SRow2 in SC2:
            CMFFID = SRow2.getValue(F_CMFFID['name'])
            UC = arcpy.UpdateCursor(CMFDataLayer,'"' + F_CMFFID['name'] + '" = ' + str(CMFFID))
            URow = UC.next()
            DCT = int(GetValFloat(URow,DCField,1))        
            URow.setValue(F_RouteFID['name'],SFID)
            CMFLength = SRow2.getValue("Shape_Length")
            if DCT == 1:
                FO = int(GetValFloat(URow,FOField,0))
                Offset = int(GetValFloat(URow,OffsetField,15))
                if FO > int(CMFLength / 70) + 1: FO = int(CMFLength / 70) + 1        
            if DCT == 2:
                FO = 1
                Offset = CMFLength
            UC.updateRow(URow)
            Left.append([FO,Offset])
            if Flag:
                SitesWithData += 1
                Flag = False
        Right = []
        SC2 = arcpy.SearchCursor(RightBI,'"' + F_RouteFID['name'] + '" = ' + str(SFID))
        for SRow2 in SC2:
            CMFFID = SRow2.getValue(F_CMFFID['name'])
            UC = arcpy.UpdateCursor(CMFDataLayer,'"' + F_CMFFID['name'] + '" = ' + str(CMFFID))
            URow = UC.next()                
            DCT = int(GetValFloat(URow,DCField,1))        
            URow.setValue(F_RouteFID['name'],SFID)
            CMFLength = SRow2.getValue("Shape_Length")
            if DCT == 1:
                FO = int(GetValFloat(URow,FOField,0))
                Offset = int(GetValFloat(URow,OffsetField,15))
                if FO > int(CMFLength / 70) + 1: FO = int(CMFLength / 70) + 1        
            if DCT == 2:
                FO = 1
                Offset = CMFLength
            UC.updateRow(URow)
            Right.append([FO,Offset])
            if Flag:
                SitesWithData += 1
                Flag = False
        #if not Flag:arcpy.AddMessage(str([SFID,Left,Right]))
        NumFO.append([Left,Right])
    arcpy.AddMessage("    Total Sites with Data: " + str(SitesWithData))

    arcpy.AddMessage("    Update Cursor: Sites Layer")
    i = 0
    UC = arcpy.UpdateCursor(SiteLayer)
    for URow in UC:
        FType = URow.getValue(F_FType['name'])
        SegLength = URow.getValue('Shape').length

        # Counting Objects
        LTotObj = 0
        for Obj in NumFO[i][0]: LTotObj += Obj[0]
        RTotObj = 0
        for Obj in NumFO[i][1]: RTotObj += Obj[0]

        # Equivalent offset
        LSoM = 0 # Summation of Multiplications of # of Obj and Offsets for Left Side
        for Obj in NumFO[i][0]: LSoM += Obj[0] * Obj[1]  # # of Obj * Offset of Obj
        if LTotObj == 0:  # Left Equivalent Offset
            LEOffset = 0
        else:
            LEOffset = LSoM / LTotObj

        RSoM = 0 # Summation of Multiplications of # of Obj and Offsets for Right Side
        for Obj in NumFO[i][1]: RSoM += Obj[0] * Obj[1]  # # of Obj * Offset of Obj
        if RTotObj == 0:  # Right Equivalent Offset
            REOffset = 0
        else:
            REOffset = RSoM / RTotObj

        # Adjustment for maximum total number of objects
        MaxObj = int(SegLength / 70) + 1
        if LTotObj > MaxObj: LTotObj = MaxObj
        if RTotObj > MaxObj: RTotObj = MaxObj

        # Aggregating Both Sides:
        if SegLength == 0:
            Density = 0
        else:
            Density = (LTotObj + RTotObj) / (SegLength / 5280)
    
        if (LTotObj + RTotObj) == 0:
            EOffset = 0
        else:
            EOffset = (LEOffset * LTotObj + REOffset * RTotObj) / (LTotObj + RTotObj)
        #arcpy.AddMessage(str([FType,Density,LEOffset,REOffset,EOffset]))
        
        CMFFO = CMFFixedObjects(FType,Density,EOffset)
        URow.setValue(F_FODensity['name'],Density)
        if EOffset == 0: EOffset = 15
        URow.setValue(F_FOOffset['name'],EOffset)
        URow.setValue(F_CMFFO['name'],CMFFO)
        UC.updateRow(URow)
        i += 1

    for Field in [F_RouteFID]:
        arcpy.DeleteField_management(SiteLayer,Field['name'])

    arcpy.CopyFeatures_management(SiteLayer, Output)
    arcpy.AddMessage("   --> Done.")
    arcpy.AddMessage(" ")
def ImportLighting(SiteInput, CMFDataInput, LBuffer, Output):
    arcpy.AddMessage(" ")
    arcpy.AddMessage("    Lighting")

    OutDic = OutputParser(Output, 'shp')
    Output = OutDic['folder'] + '\\' + OutDic['file']

    arcpy.AddMessage("    Copy Features: Site Layer")
    SiteLayer = CopyFeatures(SiteInput,'SiteInput')
    TotalSites = arcpy.GetCount_management(SiteLayer)
    arcpy.AddMessage("     - Total Items Found: " + str(TotalSites))

    arcpy.AddMessage("    Copy Features: Lighting Data Layer")
    CMFDataLayer = CopyFeatures(CMFDataInput,'LightingInput')
    TotalCMF = arcpy.GetCount_management(CMFDataLayer)
    arcpy.AddMessage("     - Total Items Found: " + str(TotalCMF))

    arcpy.AddMessage("    Adding Fields")

    FieldList1 = [F_RouteFID, F_Mileage, F_CMFLight, F_LIGHTING]
    FieldList2 = [F_RouteFID, F_CMFFID]

    for Field in FieldList1:
        AddField(SiteLayer,Field,'append')
    arcpy.AddMessage("    Calculating Fields")
    UC = arcpy.UpdateCursor(SiteLayer)
    for URow in UC:
        FID = GetFID(URow)
        URow.setValue(F_RouteFID['name'],FID)
        UC.updateRow(URow)
    FIDMileage = {SRow.getValue(F_RouteFID['name']):[] for SRow in arcpy.SearchCursor(SiteLayer)}

    arcpy.AddMessage("    Locate Features Along Routes: SiteLayer + Lighting Data Layer")
    LFAR = LocateFeaturesAlongRoutes(CMFDataLayer,SiteLayer,  F_RouteFID['name'], str(LBuffer) + " Feet", "LLFAR", F_RouteFID['name'] + " POINT " + F_Mileage['name'])

    arcpy.AddMessage("    Finding Duplicates")
    for SRow in arcpy.SearchCursor(LFAR):
        FIDMileage[SRow.getValue(F_RouteFID['name'])].append(SRow.getValue(F_Mileage['name']) * 5280)

    for Field in FieldList2:
        AddField(CMFDataLayer,Field)


    UC = arcpy.UpdateCursor(CMFDataLayer)
    for URow in UC:
        FID = GetFID(URow)
        URow.setValue(F_CMFFID['name'],FID)
        UC.updateRow(URow)


    arcpy.AddMessage("    Buffer: SiteLayer")
    SBuffer = Buffer(SiteLayer,"LBuffer",{'distance': str(LBuffer) + " Feet", 'line_side': 'FULL', 'line_end_type': 'FLAT', 'dissolve_option': 'NONE', 'dissolve_field': '#'})

    arcpy.AddMessage("    Intersect: Buffer + CMF Layer")
    BI = Intersect(SBuffer + ';' + CMFDataLayer,"LIntersect",{'join_attributes': "ALL", 'cluster_tolerance': '', 'output_type': "POINT"})




    #arcpy.AddMessage(" Route Event Layer: SiteLayer + Lighting Data Layer")
    #REL = MakeRouteEventLayer(SiteLayer, F_RouteFID['name'], LFAR,
    #F_RouteFID['name'] + " POINT " + F_Mileage['name'], "LMREL")
    arcpy.AddMessage("    Reading Info")
    AllL = []
    SitesWithData = 0
    SC1 = arcpy.UpdateCursor(SiteLayer)
    for SRow1 in SC1:
        SFID = SRow1.getValue(F_RouteFID['name'])
        SC2 = arcpy.SearchCursor(BI,'"' + F_RouteFID['name'] + '" = ' + str(SFID))
        T = []
        Flag = True
        for SRow2 in SC2:
            CMFFID = SRow2.getValue(F_CMFFID['name'])
            UC = arcpy.UpdateCursor(CMFDataLayer,'"' + F_CMFFID['name'] + '" = ' + str(CMFFID))
            URow = UC.next()
            URow.setValue(F_RouteFID['name'],SFID)
            #L = int(GetVal(URow,LField,0))
            L = 1
            UC.updateRow(URow)
            if L == 1:
                T.append(L)
                if Flag:
                    SitesWithData += 1
                    Flag = False
        #arcpy.AddMessage(str(T))
        AllL.append(T)
    arcpy.AddMessage("    Total Sites with Data: " + str(SitesWithData))

    #arcpy.AddMessage(" Calculating CMF")
    #i = 0
    #UC = arcpy.UpdateCursor(SiteLayer)
    #for URow in UC:
    #    FType = URow.getValue(F_FType['name'])
    #    SLen = URow.getValue("Shape").length
    #    CMFL = 1
    #    L = len(AllL[i])
    #    #arcpy.AddMessage(str(L))
    #    if L > 0:
    #        Factor = L*200/SLen #Assuming each light pole is effective for 200ft
    #        length of roadway
    #        if Factor > 1: Factor = 1
    #        CMFL = 1 - (Factor * (1-CMFLighting('',FType,1)))
    #    URow.setValue(F_CMFLight['name'],CMFL)
    #    UC.updateRow(URow)
    #    i += 1
    def FindFactor(SegLen, MList, Cov):
        def MergeIntervals(Int1,Int2):
            if Int1[1] > Int2[0]:
                return [Int1[0],Int2[1]]
            else:
                return ''
        def NewIntervals(OldInts):
            if len(OldInts) > 1:
                NewInts = []
            else:
                return OldInts
            for i in range(0,len(OldInts) - 1):
                Res = MergeIntervals(OldInts[i], OldInts[i + 1])
                if Res <> '':
                    if len(NewInts) > 0:
                        if Res[1] > NewInts[len(NewInts) - 1][1]:
                            NewInts.append(Res)
                    else:
                        NewInts.append(Res)
                else:
                    if len(NewInts) > 0:
                        if OldInts[i][1] > NewInts[len(NewInts) - 1][1]:
                            NewInts.append(OldInts[i])
                    else:
                        NewInts.append(OldInts[i])
                    if i == len(OldInts) - 2:
                        if len(NewInts) > 0:
                            if OldInts[i + 1][1] > NewInts[len(NewInts) - 1][1]:
                                NewInts.append(OldInts[i + 1])
                        else:
                            NewInts.append(OldInts[i + 1])
            return NewInts

        MList.sort()
        Ints = []
        for M in MList:
            Ints.append([M - Cov / 2.0, M + Cov / 2.0])
        L1 = len(Ints)
        NI = NewIntervals(Ints)
        while len(NI) <> L1:
            L1 = len(NI)
            NI = NewIntervals(NI)
        L = []
        for Int in NI:
            L.append(Int[1] - Int[0])
        TotL = sum(L)
        Factor = TotL / SegLen
        if Factor > 1: Factor = 1
        return Factor

    arcpy.AddMessage("    Calculating CMF")
    i = 0
    UC = arcpy.UpdateCursor(SiteLayer)
    for URow in UC:
        FType = URow.getValue(F_FType['name'])
        RID = URow.getValue(F_RouteFID['name'])
        SLen = URow.getValue("Shape").length
        CMFL = 1
        Factor = 0
        L = len(FIDMileage[RID])
        if L > 0: 
        
            Factor = FindFactor(SLen,FIDMileage[RID],200) #Assuming each light pole is effective for 200ft length of roadway
            CMFL = 1 - (Factor * (1 - CMFLighting('',FType,1)))
        URow.setValue(F_LIGHTING['name'],Factor)
        URow.setValue(F_CMFLight['name'],CMFL)
        UC.updateRow(URow)
        i += 1
    del UC

    arcpy.CopyFeatures_management(SiteLayer, Output)
    for Field in [F_RouteFID['name'], F_Mileage['name'], 'MEAS', 'RID']:
        arcpy.DeleteField_management(Output,Field)
    arcpy.AddMessage("   --> Done.")
    arcpy.AddMessage(" ")
def ImportMedianWidth(SiteInput, CMFDataInput, MWField, Output):
    arcpy.AddMessage(" ")
    arcpy.AddMessage("    Median Width")

    OutDic = OutputParser(Output, 'shp')
    Output = OutDic['folder'] + '\\' + OutDic['file']

    arcpy.AddMessage("    Copy Features: Site Layer")
    SiteLayer = CopyFeatures(SiteInput,'SiteInput')
    TotalSites = arcpy.GetCount_management(SiteLayer)
    arcpy.AddMessage("     - Total number of Items Found: " + str(TotalSites))

    arcpy.AddMessage("    Copy Features: Median Width Layer")
    CMFDataLayer = CopyFeatures(CMFDataInput,'MWInput')
    TotalSites = arcpy.GetCount_management(CMFDataLayer)
    arcpy.AddMessage("     - Total number of Items Found: " + str(TotalSites))

    arcpy.AddMessage("    Adding Fields")

    FieldList1 = [F_FType, F_RouteFID, F_CMFMW, F_Median_Wid]
    FieldList2 = [F_RouteFID, F_SegLength, F_CMFFID]

    for Field in FieldList1:
        AddField(SiteLayer,Field,'append')

    for Field in FieldList2:
        AddField(CMFDataLayer,Field)

    arcpy.AddMessage("    Calculating Fields")
    UC = arcpy.UpdateCursor(SiteLayer)
    for URow in UC:
        FID = GetFID(URow)
        URow.setValue(F_RouteFID['name'],FID)
        UC.updateRow(URow)

    UC = arcpy.UpdateCursor(CMFDataLayer)

    for URow in UC:
        FID = GetFID(URow)
        URow.setValue(F_CMFFID['name'],FID)
        UC.updateRow(URow)

    arcpy.AddMessage("    Performing Buffer")
    FBuffer = 250
    SBuffer = Buffer(SiteLayer,"MWBuffer",{'distance': FBuffer, 'line_side': 'FULL', 'line_end_type': 'FLAT', 'dissolve_option': 'NONE', 'dissolve_field': '#'})

    arcpy.AddMessage("    Performing Intersect")
    BI = Intersect(SBuffer + ';' + CMFDataLayer,"MWIntersect",{'join_attributes': "ALL", 'cluster_tolerance': '', 'output_type': "POINT"})

    arcpy.AddMessage("    Reading Info")
    AllMW = []
    SitesWithData = 0
    SC1 = arcpy.UpdateCursor(SiteLayer)
    for SRow1 in SC1:
        SFID = SRow1.getValue(F_RouteFID['name'])
        SC2 = arcpy.SearchCursor(BI,'"' + F_RouteFID['name'] + '" = ' + str(SFID))
        T = []
        Flag = True
        for SRow2 in SC2:
            CMFFID = SRow2.getValue(F_CMFFID['name'])
            UC = arcpy.UpdateCursor(CMFDataLayer,'"' + F_CMFFID['name'] + '" = ' + str(CMFFID))
            URow = UC.next()
                
            URow.setValue(F_RouteFID['name'],SFID)

            MW = float(GetVal(URow,MWField,15))
            UC.updateRow(URow)
            T.append(MW)
            if Flag:
                SitesWithData += 1
                Flag = False
        AllMW.append(T)
    arcpy.AddMessage("    Total Sites with Data: " + str(SitesWithData))

    arcpy.AddMessage("    Calculating CMF")
    i = 0
    UC = arcpy.UpdateCursor(SiteLayer)
    for URow in UC:
        FType = URow.getValue(F_FType['name'])
        CMFMW = 0
        for MW in AllMW[i]:
            CMFMW += CMFMedianWidth(FType,MW)
        if CMFMW == 0:
            CMFMW = 1
        else:
            CMFMW = CMFMW / len(AllMW[i])
        URow.setValue(F_CMFMW['name'],CMFMW)
        URow.setValue(F_Median_Wid['name'],float(sum(AllMW[i]))/max(len(AllMW[i]),1))
        UC.updateRow(URow)
        i += 1

    arcpy.CopyFeatures_management(SiteLayer, Output)
    arcpy.AddMessage("   --> Done.")
    arcpy.AddMessage(" ")
def ImportOnStreetParking(SiteInput, CMFDataInput, ParkTypeField, AreaTypeField, Output):
    arcpy.AddMessage(" ")
    arcpy.AddMessage("    On Street Parking Import")

    OutDic = OutputParser(Output, 'shp')
    Output = OutDic['folder'] + '\\' + OutDic['file']

    arcpy.AddMessage("    Copy Features: Site Layer")
    SiteLayer = CopyFeatures(SiteInput,'OSPSite')
    TotalSites = arcpy.GetCount_management(SiteLayer)
    arcpy.AddMessage("     - Total number of Sites Found: " + str(TotalSites))

    arcpy.AddMessage("    Copy Features: CMF Data Layer")
    CMFDataLayer = CopyFeatures(CMFDataInput,'OSPData')
    TotalCMF = arcpy.GetCount_management(CMFDataLayer)
    arcpy.AddMessage("     - Total number of CMF Items Found: " + str(TotalCMF))

    arcpy.AddMessage("    Add Fields: Site Layer")
    FieldList1 = [F_RouteFID, F_CMFOSP, F_OSPType, F_OSPProp]
    for Field in FieldList1:
        AddField(SiteLayer,Field,'append')
    UC = arcpy.UpdateCursor(SiteLayer)
    for URow in UC:
        FID = GetFID(URow)
        URow.setValue(F_RouteFID['name'],FID)
        UC.updateRow(URow)

    arcpy.AddMessage("    Add Fields: On Street Parking Layer")
    FieldList2 = [F_RouteFID, F_SegLength, F_CMFFID, F_CMFLength]
    for Field in FieldList2:
        AddField(CMFDataLayer,Field)
    UC = arcpy.UpdateCursor(CMFDataLayer)
    for URow in UC:
        FID = GetFID(URow)
        URow.setValue(F_CMFFID['name'],FID)
        UC.updateRow(URow)

    arcpy.AddMessage("    Buffer: Sites Layer")
    FBuffer = 250
    SBuffer = Buffer(SiteLayer,"OSPBuffer",{'distance': FBuffer, 'line_side': 'FULL', 'line_end_type': 'FLAT', 'dissolve_option': 'NONE', 'dissolve_field': '#'})

    arcpy.AddMessage("    Intersect: Buffer + On Street Parking Layer")
    BI = Intersect(SBuffer + ';' + CMFDataLayer,"OSPIntersect",{'join_attributes': "ALL", 'cluster_tolerance': '', 'output_type': "LINE"})

    arcpy.AddMessage("    Update Cursor: Sites Layer")
    def GetOSP(Row, Field,DefaultVal):
        try:
            return int(Row.getValue(Field))
        except:
            return DefaultVal
    
    OSPPropor = []
    SitesWithData = 0
    SC1 = arcpy.UpdateCursor(SiteLayer)
    for SRow1 in SC1:
        SFID = SRow1.getValue(F_RouteFID['name'])
        FType = SRow1.getValue(F_FType['name'])
        if not FType: FType = ' '
        CMFOSP = 1
        SC2 = arcpy.SearchCursor(BI,'"' + F_RouteFID['name'] + '" = ' + str(SFID))
        T = [[0,0,0,0,0],[0,0,0,0,0]]
        Flag = True
        for SRow2 in SC2:
            CMFFID = SRow2.getValue(F_CMFFID['name'])
            UC = arcpy.UpdateCursor(CMFDataLayer,'"' + F_CMFFID['name'] + '" = ' + str(CMFFID))
            URow = UC.next()
                
            URow.setValue(F_RouteFID['name'],SFID)

            CMFLength = SRow2.getValue("Shape_Length")
            URow.setValue(F_CMFLength['name'],CMFLength)
            URow.setValue(F_SegLength['name'],SRow1.getValue("Shape").length)

            AreaT = GetOSP(URow,AreaTypeField,5)
            ParkT = GetOSP(URow,ParkTypeField,2)
            Propor = float(URow.getValue(F_CMFLength['name'])) / float(URow.getValue(F_SegLength['name'])) / 2
        
            UC.updateRow(URow)
            T[ParkT - 1][AreaT - 1] = T[ParkT - 1][AreaT - 1] + Propor
            if Flag:
                SitesWithData += 1
                Flag = False
        OSPP = 0
        OSPT = 0
        MaxCMF = 1
        if FType in ['U2U','U4D','U4U','U3T','U5T']:
            for ParkT in [1,2]:
                for AreaT in [1,2,3,4,5]:
                    t =  CMFOnStreetParking(FType,ParkT,AreaT,T[ParkT - 1][AreaT - 1])
                    if t>MaxCMF:
                        MaxCMF = t
                        OSPP = T[ParkT - 1][AreaT - 1]
                        OSPT = 1*(ParkT==2 and AreaT in [1,5]) + 2*(ParkT==2 and AreaT in [2,3,4]) + 3*(ParkT==2 and AreaT in [1,5]) + 4*(ParkT==2 and AreaT in [2,3,4])
                    CMFOSP = CMFOSP * t
        SRow1.setValue(F_CMFOSP['name'],CMFOSP)
        SRow1.setValue(F_OSPProp['name'],OSPP)
        SRow1.setValue(F_OSPType['name'],OSPT)
        SC1.updateRow(SRow1)
        if CMFOSP <> 1:
            arcpy.AddMessage('    FID: ' + str(SFID) + ', Type: ' + FType + ', OSP: ' + str(T) + ', CMF: ' + str(CMFOSP))

    arcpy.AddMessage("    Total Sites with Data: " + str(SitesWithData))
    arcpy.CopyFeatures_management(SiteLayer, Output)
    arcpy.AddMessage("   --> Done.")
    arcpy.AddMessage(" ")
def ImportRHR(SiteInput, CMFDataInput, RHRField, Output):
    arcpy.AddMessage(" ")
    arcpy.AddMessage("    RHR Import")

    OutDic = OutputParser(Output, 'shp')
    Output = OutDic['folder'] + '\\' + OutDic['file']

    arcpy.AddMessage("    Copy Features: Site Layer")
    SiteLayer = CopyFeatures(SiteInput,'OSPSite')
    TotalSites = arcpy.GetCount_management(SiteLayer)
    arcpy.AddMessage("     - Total number of Sites Found: " + str(TotalSites))

    arcpy.AddMessage("    Copy Features: CMF Data Layer")
    CMFDataLayer = CopyFeatures(CMFDataInput,'OSPData')
    TotalCMF = arcpy.GetCount_management(CMFDataLayer)
    arcpy.AddMessage("     - Total number of CMF Items Found: " + str(TotalCMF))

    arcpy.AddMessage("    Add Fields: Site Layer")
    FieldList1 = [F_RouteFID, F_RHR, F_CMFRHR]
    for Field in FieldList1:
        AddField(SiteLayer,Field,'append')
    UC = arcpy.UpdateCursor(SiteLayer)
    for URow in UC:
        FID = GetFID(URow)
        URow.setValue(F_RouteFID['name'],FID)
        UC.updateRow(URow)

    arcpy.AddMessage("    Add Fields: RHR Layer")
    FieldList2 = [F_RouteFID, F_SegLength, F_CMFFID]
    for Field in FieldList2:
        AddField(CMFDataLayer,Field)
    UC = arcpy.UpdateCursor(CMFDataLayer)
    for URow in UC:
        FID = GetFID(URow)
        URow.setValue(F_CMFFID['name'],FID)
        UC.updateRow(URow)

    arcpy.AddMessage("    Buffer: Sites Layer")
    FBuffer = 250
    SBuffer = Buffer(SiteLayer,"OSPBuffer",{'distance': FBuffer, 'line_side': 'FULL', 'line_end_type': 'ROUND', 'dissolve_option': 'NONE', 'dissolve_field': '#'})

    arcpy.AddMessage("    Intersect: Buffer + On Street Parking Layer")
    BI = Intersect(SBuffer + ';' + CMFDataLayer,"OSPIntersect",{'join_attributes': "ALL", 'cluster_tolerance': '', 'output_type': "POINT"})

    arcpy.AddMessage("    Update Cursor: Sites Layer")
    def GetOSP(Row, Field,DefaultVal):
        try:
            return int(Row.getValue(Field))
        except:
            return DefaultVal
    
    OSPPropor = []
    SitesWithData = 0
    SC1 = arcpy.UpdateCursor(SiteLayer)
    for SRow1 in SC1:
        SFID = SRow1.getValue(F_RouteFID['name'])
        FType = SRow1.getValue(F_FType['name'])
        if not FType: FType = ' '

        SC2 = arcpy.SearchCursor(BI,'"' + F_RouteFID['name'] + '" = ' + str(SFID))

        Flag = True
        RHR = []
        for SRow2 in SC2:
            CMFFID = SRow2.getValue(F_CMFFID['name'])
            UC = arcpy.UpdateCursor(CMFDataLayer,'"' + F_CMFFID['name'] + '" = ' + str(CMFFID))
            URow = UC.next()
                
            URow.setValue(F_RouteFID['name'],SFID)
            URow.setValue(F_SegLength['name'],SRow1.getValue("Shape").length)

            RHR.append(GetOSP(URow,RHRField,1))
        
            UC.updateRow(URow)

        t = float(sum(RHR))/max(len(RHR),1)
        if t == 0: t=1
        SRow1.setValue(F_RHR['name'],t)
        SRow1.setValue(F_CMFRHR['name'],CMFRHR(FType,t))
        
        SC1.updateRow(SRow1)
        if len(RHR) > 1:
            arcpy.AddMessage('    FID: ' + str(SFID) + ', Type: ' + FType + ', RHR: ' + str(RHR) + ', Average: ' + str(t))

    arcpy.AddMessage("    Total Sites with Data: " + str(SitesWithData))
    arcpy.CopyFeatures_management(SiteLayer, Output)
    arcpy.AddMessage("   --> Done.")
    arcpy.AddMessage(" ")
def ImportShoulderWidth(SiteInput, CMFDataInput, ShWField, ShTField, Output):
    arcpy.AddMessage(" ")
    arcpy.AddMessage("    Shoulder Width Import")

    OutDic = OutputParser(Output, 'shp')
    Output = OutDic['folder'] + '\\' + OutDic['file']

    arcpy.AddMessage("    Copy Features: Site Layer")
    SiteLayer = CopyFeatures(SiteInput,'OSPSite')
    TotalSites = arcpy.GetCount_management(SiteLayer)
    arcpy.AddMessage("     - Total number of Sites Found: " + str(TotalSites))

    arcpy.AddMessage("    Copy Features: CMF Data Layer")
    CMFDataLayer = CopyFeatures(CMFDataInput,'OSPData')
    TotalCMF = arcpy.GetCount_management(CMFDataLayer)
    arcpy.AddMessage("     - Total number of CMF Items Found: " + str(TotalCMF))

    arcpy.AddMessage("    Add Fields: Site Layer")
    FieldList1 = [F_RouteFID, F_Shuold_Wid, F_Shuold_Typ, F_CMFSW]
    for Field in FieldList1:
        AddField(SiteLayer,Field,'append')
    UC = arcpy.UpdateCursor(SiteLayer)
    for URow in UC:
        FID = GetFID(URow)
        URow.setValue(F_RouteFID['name'],FID)
        UC.updateRow(URow)

    arcpy.AddMessage("    Add Fields: ShW Layer")
    FieldList2 = [F_RouteFID, F_SegLength, F_CMFFID, F_CMFSW]
    for Field in FieldList2:
        AddField(CMFDataLayer,Field)
    UC = arcpy.UpdateCursor(CMFDataLayer)
    for URow in UC:
        FID = GetFID(URow)
        URow.setValue(F_CMFFID['name'],FID)
        UC.updateRow(URow)

    arcpy.AddMessage("    Buffer: Sites Layer")
    FBuffer = 250
    SBuffer = Buffer(SiteLayer,"OSPBuffer",{'distance': FBuffer, 'line_side': 'FULL', 'line_end_type': 'ROUND', 'dissolve_option': 'NONE', 'dissolve_field': '#'})

    arcpy.AddMessage("    Intersect: Buffer + On Street Parking Layer")
    BI = Intersect(SBuffer + ';' + CMFDataLayer,"OSPIntersect",{'join_attributes': "ALL", 'cluster_tolerance': '', 'output_type': "POINT"})

    arcpy.AddMessage("    Update Cursor: Sites Layer")
    def GetOSP(Row, Field,DefaultVal):
        try:
            return int(Row.getValue(Field))
        except:
            return DefaultVal
    
    OSPPropor = []
    SitesWithData = 0
    SC1 = arcpy.UpdateCursor(SiteLayer)
    for SRow1 in SC1:
        SFID = SRow1.getValue(F_RouteFID['name'])
        FType = SRow1.getValue(F_FType['name'])
        if not FType: FType = ' '

        SC2 = arcpy.SearchCursor(BI,'"' + F_RouteFID['name'] + '" = ' + str(SFID))

        Flag = True
        RHR = []
        ShT = []
        for SRow2 in SC2:
            CMFFID = SRow2.getValue(F_CMFFID['name'])
            UC = arcpy.UpdateCursor(CMFDataLayer,'"' + F_CMFFID['name'] + '" = ' + str(CMFFID))
            URow = UC.next()
                
            URow.setValue(F_RouteFID['name'],SFID)
            URow.setValue(F_SegLength['name'],SRow1.getValue("Shape").length)

            RHR.append(URow.getValue(ShWField))
            a = URow.getValue(ShTField)
            b= 1
            if a == 'Paved': b = 1
            if a == 'Gravel': b = 2
            if a == 'Composite': b = 3
            if a == 'Turf': b = 4
            ShT.append(b)
        
            UC.updateRow(URow)

        t1 = float(sum(RHR))/max(len(RHR),1)
        try:
            t2 = min(ShT)
        except:
            t2 = 0
        SRow1.setValue(F_Shuold_Wid['name'],t1)
        SRow1.setValue(F_Shuold_Typ['name'],t2)
        SRow1.setValue(F_CMFSW['name'],CMFShoulderWidthType(FType,t1,t2,SRow1.getValue(F_AADT['name'])))
        SC1.updateRow(SRow1)
        if len(RHR) > 1:
            arcpy.AddMessage('    FID: ' + str(SFID) + ', Type: ' + FType + ', ShW: ' + str(RHR) + ', Average: ' + str(t1))

    arcpy.AddMessage("    Total Sites with Data: " + str(SitesWithData))
    arcpy.CopyFeatures_management(SiteLayer, Output)
    arcpy.AddMessage("   --> Done.")
    arcpy.AddMessage(" ")
def ImportLaneWidth(SiteInput, CMFDataInput, LnWField, Output):
    arcpy.AddMessage(" ")
    arcpy.AddMessage("    Shoulder Width Import")

    OutDic = OutputParser(Output, 'shp')
    Output = OutDic['folder'] + '\\' + OutDic['file']

    arcpy.AddMessage("    Copy Features: Site Layer")
    SiteLayer = CopyFeatures(SiteInput,'OSPSite')
    TotalSites = arcpy.GetCount_management(SiteLayer)
    arcpy.AddMessage("     - Total number of Sites Found: " + str(TotalSites))

    arcpy.AddMessage("    Copy Features: CMF Data Layer")
    CMFDataLayer = CopyFeatures(CMFDataInput,'OSPData')
    TotalCMF = arcpy.GetCount_management(CMFDataLayer)
    arcpy.AddMessage("     - Total number of CMF Items Found: " + str(TotalCMF))

    arcpy.AddMessage("    Add Fields: Site Layer")
    FieldList1 = [F_RouteFID, F_Lane_Width, F_CMFLW]
    for Field in FieldList1:
        AddField(SiteLayer,Field,'append')
    UC = arcpy.UpdateCursor(SiteLayer)
    for URow in UC:
        FID = GetFID(URow)
        URow.setValue(F_RouteFID['name'],FID)
        UC.updateRow(URow)

    arcpy.AddMessage("    Add Fields: LnW Layer")
    FieldList2 = [F_RouteFID, F_SegLength, F_CMFFID]
    for Field in FieldList2:
        AddField(CMFDataLayer,Field)
    UC = arcpy.UpdateCursor(CMFDataLayer)
    for URow in UC:
        FID = GetFID(URow)
        URow.setValue(F_CMFFID['name'],FID)
        UC.updateRow(URow)

    arcpy.AddMessage("    Buffer: Sites Layer")
    FBuffer = 250
    SBuffer = Buffer(SiteLayer,"OSPBuffer",{'distance': FBuffer, 'line_side': 'FULL', 'line_end_type': 'ROUND', 'dissolve_option': 'NONE', 'dissolve_field': '#'})

    arcpy.AddMessage("    Intersect: Buffer + On Street Parking Layer")
    BI = Intersect(SBuffer + ';' + CMFDataLayer,"OSPIntersect",{'join_attributes': "ALL", 'cluster_tolerance': '', 'output_type': "POINT"})

    arcpy.AddMessage("    Update Cursor: Sites Layer")
    def GetOSP(Row, Field,DefaultVal):
        try:
            return float(Row.getValue(Field))
        except:
            return DefaultVal
    
    OSPPropor = []
    SitesWithData = 0
    SC1 = arcpy.UpdateCursor(SiteLayer)
    for SRow1 in SC1:
        SFID = SRow1.getValue(F_RouteFID['name'])
        FType = SRow1.getValue(F_FType['name'])
        if not FType: FType = ' '

        SC2 = arcpy.SearchCursor(BI,'"' + F_RouteFID['name'] + '" = ' + str(SFID))

        Flag = True
        RHR = []
        ShT = []
        for SRow2 in SC2:
            CMFFID = SRow2.getValue(F_CMFFID['name'])
            UC = arcpy.UpdateCursor(CMFDataLayer,'"' + F_CMFFID['name'] + '" = ' + str(CMFFID))
            URow = UC.next()
                
            URow.setValue(F_RouteFID['name'],SFID)
            URow.setValue(F_SegLength['name'],SRow1.getValue("Shape").length)

            RHR.append(GetOSP(URow,LnWField,12))
    
            UC.updateRow(URow)

        t1 = float(sum(RHR))/max(len(RHR),1)
        SRow1.setValue(F_Lane_Width['name'],t1)
        SRow1.setValue(F_CMFLW['name'],CMFLaneWidth(FType,t1,SRow1.getValue(F_AADT['name'])))
        SC1.updateRow(SRow1)
        if len(RHR) > 1:
            arcpy.AddMessage('    FID: ' + str(SFID) + ', Type: ' + FType + ', LnW: ' + str(RHR) + ', Average: ' + str(t1))

    arcpy.AddMessage("    Total Sites with Data: " + str(SitesWithData))
    arcpy.CopyFeatures_management(SiteLayer, Output)
    arcpy.AddMessage("   --> Done.")
    arcpy.AddMessage(" ")
def ImportLaneChange(SiteInput,WeavingInput,AllSegInput, Output):
    arcpy.AddMessage(" ")
    arcpy.AddMessage("    Lane Change Import")

    OutDic = OutputParser(Output, 'shp')
    Output = OutDic['folder'] + '\\' + OutDic['file']

    arcpy.AddMessage("    Copy Features: Site Layer")
    SiteLayer = CopyFeatures(SiteInput,'LCSite')
    TotalSites = arcpy.GetCount_management(SiteLayer)
    arcpy.AddMessage("     - Total number of Sites Found: " + str(TotalSites))

    arcpy.AddMessage("    Copy Features: CMF Data Layer")
    CMFDataLayer = CopyFeatures(WeavingInput,'LCData')
    TotalCMF = arcpy.GetCount_management(CMFDataLayer)
    arcpy.AddMessage("     - Total number of CMF Items Found: " + str(TotalCMF))

    arcpy.AddMessage("    Add Fields: Site Layer")
    FieldList1 = [F_RouteFID, F_CMFLChgFI,F_CMFLChgPDO]
    for Field in FieldList1:
        AddField(SiteLayer,Field,'append')
    UC = arcpy.UpdateCursor(SiteLayer)
    for URow in UC:
        FID = GetFID(URow)
        URow.setValue(F_RouteFID['name'],FID)
        UC.updateRow(URow)

    SiteDict = {Row.getValue(F_RouteFID['name']):{'LsWevInc':0,'LsWevDec':0,'LWevInc':1,'LWevDec':1,'Ls':Row.getValue('Shape').length/5280,
                                                  'XbEnt':10,'XbExt':10,'XeEnt':10,'XeExt':10,
                                                  'AADTbEnt':1000,'AADTbExt':1000,'AADTeEnt':1000,'AADTeExt':1000, 'Oneway':Row.getValue(F_Oneway['name'])} 
                for Row in arcpy.SearchCursor(SiteLayer)}

    arcpy.AddMessage("    Add Fields: Weaving Layer")
    FieldList2 = [F_RouteFID, F_SegLength, F_CMFFID]
    for Field in FieldList2:
        AddField(CMFDataLayer,Field)
    UC = arcpy.UpdateCursor(CMFDataLayer)
    for URow in UC:
        FID = GetFID(URow)
        URow.setValue(F_CMFFID['name'],FID)
        UC.updateRow(URow)

    CMFDict = {Row.getValue(F_CMFFID['name']):{'L':Row.getValue('Shape').length/5280} 
                for Row in arcpy.SearchCursor(CMFDataLayer)}

    arcpy.AddMessage("    Buffer: Sites Layer")
    FBuffer = 25
    SBuffer = Buffer(SiteLayer,"LCBuffer",{'distance': FBuffer, 'line_side': 'FULL', 'line_end_type': 'FLAT', 'dissolve_option': 'NONE', 'dissolve_field': '#'})

    arcpy.AddMessage("    Intersect: Buffer + Weaving Layer")
    BI = Intersect(SBuffer + ';' + CMFDataLayer,"LCIntersect",{'join_attributes': "ALL", 'cluster_tolerance': '', 'output_type': "POINT"})
    IntDict = {Row.getValue(F_RouteFID['name']):Row.getValue(F_CMFFID['name']) for Row in arcpy.SearchCursor(BI)}
    for fid in IntDict.keys():
        if SiteDict[fid]['Oneway'] == 0:
            SiteDict[fid]['LWevInc'] = CMFDict[IntDict[fid]]['L']
        if SiteDict[fid]['Oneway'] == 1:
            SiteDict[fid]['LWevDec'] = CMFDict[IntDict[fid]]['L']

    arcpy.AddMessage("    Split Line: Weaving Layer + Intersect")
    CMFSplit = SplitLineAtPoint(CMFDataLayer,BI,'SPLWev','20 Feet')
    arcpy.CalculateField_management(CMFSplit,F_SegLength['name'],'!Shape!.length','PYTHON_9.3')

    arcpy.AddMessage("    Spatial Join: Buffer + Split Layer")
    SPJ = SpatialJoin(SBuffer, CMFSplit, 'LCSPJ',{'join_operation':"JOIN_ONE_TO_MANY",'join_type':"KEEP_COMMON",'field_mapping':'','match_option':'CONTAINS','search_radius':'','distance_field_name':''})
    
    SPJDict = {Row.getValue(F_RouteFID['name']):Row.getValue(F_SegLength['name'])/5280 for Row in arcpy.SearchCursor(SPJ)}
    for fid in SPJDict.keys():
        if SiteDict[fid]['Oneway'] == 0:
            SiteDict[fid]['LsWevInc'] = SPJDict[fid]
        if SiteDict[fid]['Oneway'] == 1:
            SiteDict[fid]['LsWevDec'] = SPJDict[fid]

    arcpy.AddMessage("    Update Cursor: Sites Layer")
    SC = arcpy.UpdateCursor(SiteLayer)
    for SRow in SC:
        SFID = SRow.getValue(F_RouteFID['name'])
        FType = SRow.getValue(F_FType['name'])
        if not FType: FType = ' '

        Flag = True
        CMFLC = CMFLaneChange(FType,SiteDict[SFID])
        SRow.setValue(F_CMFLChgFI['name'],CMFLC['FI'])
        SRow.setValue(F_CMFLChgPDO['name'],CMFLC['PDO'])
        SC.updateRow(SRow)

    arcpy.CopyFeatures_management(SiteLayer, Output)
    arcpy.AddMessage("   --> Done.")
    arcpy.AddMessage(" ")
def ImportClearZone(SiteInput,CZInput,OBInput,Output):
    arcpy.AddMessage(" ")
    arcpy.AddMessage("    Clear Zone Import")

    OutDic = OutputParser(Output, 'shp')
    Output = OutDic['folder'] + '\\' + OutDic['file']

    arcpy.AddMessage("    Copy Features: Site Layer")
    SiteLayer = CopyFeatures(SiteInput,'CZSite')
    TotalSites = arcpy.GetCount_management(SiteLayer)
    arcpy.AddMessage("     - Total number of Sites Found: " + str(TotalSites))

    arcpy.AddMessage("    Copy Features: ClearZone Data Layer")
    CMFDataLayer = CopyFeatures(CZInput,'CZData')
    TotalCMF = arcpy.GetCount_management(CMFDataLayer)
    arcpy.AddMessage("     - Total number of CMF Items Found: " + str(TotalCMF))

    arcpy.AddMessage("    Copy Features: Outside Barrier Data Layer")
    OBDataLayer = CopyFeatures(OBInput,'OBData')
    TotalCMF = arcpy.GetCount_management(OBDataLayer)
    arcpy.AddMessage("     - Total number of CMF Items Found: " + str(TotalCMF))

    arcpy.AddMessage("    Add Fields: Site Layer")
    FieldList1 = [F_RouteFID,F_CZWidth,F_CMFCZFI,F_CMFOBFI,F_CMFOBPDO]
    for Field in FieldList1:
        AddField(SiteLayer,Field,'append')
    UC = arcpy.UpdateCursor(SiteLayer)
    for URow in UC:
        FID = GetFID(URow)
        URow.setValue(F_RouteFID['name'],FID)
        UC.updateRow(URow)

    SDict = {Row.getValue(F_RouteFID['name']):{'CMFFID':-1,'CZWidth':30,
                                               'SHW_LO':Row.getValue(F_Sh_Wid_LO['name']),'SHW_RO':Row.getValue(F_Sh_Wid_RO['name'])
                                               ,'SW_L':Row.getValue(F_L_Sur_W['name']),'SW_R':Row.getValue(F_R_Sur_W['name']),
                                               'Oneway':Row.getValue(F_Oneway['name'])} for Row in arcpy.SearchCursor(SiteLayer)}
    
    arcpy.AddMessage("    Buffer: Sites Layer")
    FBuffer = 300
    SBuffer = Buffer(SiteLayer,"LCBuffer",{'distance': FBuffer, 'line_side': 'FULL', 'line_end_type': 'FLAT', 'dissolve_option': 'NONE', 'dissolve_field': '#'})

    arcpy.AddMessage("    Intersect: Buffer + Weaving Layer")
    BI = Intersect(SBuffer + ';' + CMFDataLayer,"LCIntersect",{'join_attributes': "ALL", 'cluster_tolerance': '', 'output_type': "POINT"})

    arcpy.AddMessage("    Split Line: CMF Layer + Intersect")
    CMFSplit = SplitLineAtPoint(CMFDataLayer,BI,'SPLWev','5 Feet')

    arcpy.AddMessage("    Add Fields: CMF Layer")
    FieldList2 = [F_CMFFID]
    for Field in FieldList2:
        AddField(CMFSplit,Field)
    UC = arcpy.UpdateCursor(CMFSplit)
    i = 0
    for URow in UC:
        i = i + 1
        URow.setValue(F_CMFFID['name'],i)
        UC.updateRow(URow)
    CMFDict = {Row.getValue(F_CMFFID['name']):{'Dist':[],'SFID':-1} for Row in arcpy.SearchCursor(CMFSplit)}

    arcpy.AddMessage("    Spatial Join: Buffer + Split Layer")
    SPJ1 = SpatialJoin(SBuffer , CMFSplit , 'LCSPJB',{'join_operation':"JOIN_ONE_TO_MANY",'join_type':"KEEP_COMMON",'field_mapping':'','match_option':'CONTAINS','search_radius':'','distance_field_name':''})
    SPJ2 = SpatialJoin(CMFSplit, SiteLayer, 'LCSPJC',{'join_operation':"JOIN_ONE_TO_ONE" ,'join_type':"KEEP_COMMON",'field_mapping':'','match_option':'CLOSEST','search_radius':'','distance_field_name':''})

    for SRow in arcpy.SearchCursor(SPJ2):
        fid = SRow.getValue(F_CMFFID['name'])
        CMFDict[fid]['SFID'] = SRow.getValue(F_RouteFID['name'])  
        #arcpy.AddMessage(str([CMFDict[fid]['SFID'],fid]))
    for SRow in arcpy.SearchCursor(SPJ1):
        SFID   = SRow.getValue(F_RouteFID['name'])
        CMFFID = SRow.getValue(F_CMFFID  ['name'])
        #arcpy.AddMessage(str([SFID,CMFFID]))
        if SFID == CMFDict[CMFFID]['SFID']:
            SDict[SFID]['CMFFID'] = SRow.getValue(F_CMFFID['name'])  
            #arcpy.AddMessage(str([SDict[fid]['CMFFID']]))
    
    V2P = VerticesToPoints(CMFSplit,'CLV2P','ALL')

    arcpy.AddMessage("    Near Analysis: Split Layer")
    arcpy.Near_analysis(V2P,SiteLayer,str(FBuffer)+' Feet')

    for SRow in arcpy.SearchCursor(V2P):
        fid = SRow.getValue(F_CMFFID['name'])
        CMFDict[fid]['Dist'].append(SRow.getValue('NEAR_DIST'))  

    for fid in SDict.keys():
        cmffid = SDict[fid]['CMFFID']
        if cmffid in CMFDict.keys():
            L = CMFDict[cmffid]['Dist']
            MD = sum(L)/len(L)
            MSW = (SDict[fid]['SW_L']+SDict[fid]['SW_R'])/2
            MOS = (SDict[fid]['SHW_LO']+SDict[fid]['SHW_RO'])/2
            SDict[fid]['CZWidth'] = MD - MSW/2 - MOS
            if SDict[fid]['CZWidth']>30:SDict[fid]['CZWidth']=30
            arcpy.AddMessage(str([MD - MSW/2 - MOS]))

    arcpy.AddMessage("    Update Cursor: Sites Layer")
    SC = arcpy.UpdateCursor(SiteLayer)
    for SRow in SC:
        SFID = SRow.getValue(F_RouteFID['name'])
        FType = SRow.getValue(F_FType['name'])
        if not FType: FType = ' '

        Flag = True
        CMFList = CMFCZandOB(FType,0,SDict[SFID]['CZWidth'],30,(SDict[SFID]['SHW_LO']+SDict[SFID]['SHW_RO'])/2)
        SRow.setValue(F_CZWidth ['name'],SDict[SFID]['CZWidth'])
        SRow.setValue(F_CMFCZFI ['name'],CMFList[0])
        SRow.setValue(F_CMFOBFI ['name'],CMFList[1]['FI'])
        SRow.setValue(F_CMFOBPDO['name'],CMFList[1]['PDO'])
        SC.updateRow(SRow)

    arcpy.CopyFeatures_management(SiteLayer, Output)
    arcpy.AddMessage("   --> Done.")
    arcpy.AddMessage(" ")

def FittSPF_Seg_2(Sim_Data):
    LL_DS = pd.Series()
    for lw in [10,11,12]:
        Sim_Data['LW{}'.format(lw)] = Sim_Data.LN_WTH.apply(lambda x:1 if x==lw else 0)
    fm_string = ' + '.join(["N_Obs ~ AADT_Log"] + ['LW{}'.format(i) for i in [10,11,12]])
    for a in np.linspace(0.05, 3, 50) :
        model = smf.glm(formula = fm_string, data=Sim_Data, family=sm.families.NegativeBinomial(alpha=a),offset=Sim_Data.Length_Log).fit()
        LL_DS.loc[a] = model.llf
    LL_DS.plot()
    plt.grid()
    sel_a = LL_DS[LL_DS==LL_DS.max()].index[0]
    plt.vlines(sel_a,LL_DS.min(),LL_DS.max(),'red')
    plt.show()
    model = smf.glm(formula = fm_string, data=Sim_Data, family=sm.families.NegativeBinomial(alpha=sel_a),offset=Sim_Data.Length_Log).fit()    
    #display(model.summary())
    #Sim_Data.LN_WTH = Sim_Data.LN_WTH + 12
    CMF = lambda row:np.prod([np.exp(row[lw]*model.params.loc[lw]) for lw in ['LW10','LW11','LW12']])/np.exp(model.params.loc['LW12'])
    SPF1 = lambda row:row.AADT**model.params.loc['AADT_Log']*row.Length*np.exp(model.params.loc['Intercept'])
    #SPF2 = lambda row:np.exp(row.AADT_Log*model.params.loc['AADT_Log']+row.Length_Log+model.params.loc['Intercept'])
    #SPF3 = lambda row:model.predict(row).tolist()[0]*row.Length
    #display(model.summary())
    Sim_Data['overdisperssion'] = sel_a
    #Sim_Data['Fitted'] = model.fittedvalues
    Sim_Data['Pred_CF'] = Sim_Data.apply(SPF1,axis=1) * Sim_Data.apply(CMF,axis=1)
    #Sim_Data['Pred_CF2'] = Sim_Data.apply(SPF2,axis=1)
    #Sim_Data['Pred_CF3'] = Sim_Data.apply(SPF3,axis=1)

    Sim_Data['Constant'] = 1
    sel_flds = ['Constant','AADT_Log','LW10','LW11','LW12']
    Sim_Data['Variance'] = Sim_Data.apply(lambda row:np.dot(np.dot(row[sel_flds].to_numpy(),model.cov_params()),row[sel_flds].to_numpy().T),axis=1)
    Sim_Data['Std_Err'] = Sim_Data.Variance.apply(np.sqrt)
    Sim_Data['Pred_CF_Per_Mile'] = Sim_Data.Pred_CF / Sim_Data.Length
    Sim_Data['CF'] = Sim_Data.Pred_CF / Sim_Data.Length

    Sim_Data['Upper'] = Sim_Data.Pred_CF_Per_Mile.apply(np.log) + Sim_Data.Std_Err * 1.96
    Sim_Data['Lower'] = Sim_Data.Pred_CF_Per_Mile.apply(np.log) - Sim_Data.Std_Err * 1.96
    Sim_Data['Upper'] = Sim_Data.Upper.apply(np.exp)
    Sim_Data['Lower'] = Sim_Data.Lower.apply(np.exp)

    Sim_Data['Residuals'] = Sim_Data.N_Obs - Sim_Data.Pred_CF
    Sim_Data.sort_values('AADT',inplace=True)
    Sim_Data['CumRes'] = Sim_Data.Residuals.cumsum()
    Sim_Data['CumResSquared'] = Sim_Data.Residuals.pow(2).cumsum()
    Sim_Data['Var'] = Sim_Data.CumResSquared * (1-Sim_Data.CumResSquared/Sim_Data.CumResSquared.max())
    Sim_Data['Lim'] = 2 * Sim_Data.Var.pow(0.5)
    Sim_Data.drop(columns=['CumResSquared','Var'],inplace=True)
    
    Predicted_DF = pd.DataFrame([
            pd.Series(np.linspace(Sim_Data.AADT.min(),Sim_Data.AADT.max(),1000),name='AADT')
        ]).T
    Predicted_DF['Length'] = 1
    Predicted_DF['AADT_Log'] = Predicted_DF.AADT.apply(np.log)
    Predicted_DF['Length_Log'] = Predicted_DF.Length.apply(np.log)
    Predicted_DF['LN_WTH'] = 12

    Predicted_DF['Pred_CF'] = Predicted_DF.apply(SPF1,axis=1)
        #Predicted_DF['Pred_CF2'] = Predicted_DF.apply(SPF2,axis=1)
        #Predicted_DF['Pred_CF3'] = Predicted_DF.apply(SPF3,axis=1)


    Predicted_DF['Constant'] = 1
    #sel_flds = ['Constant','AADT_Log','LN_WTH']
    Predicted_DF['Variance'] = Predicted_DF.apply(lambda row:np.dot(np.dot(row[sel_flds].to_numpy(),model.cov_params()),row[sel_flds].to_numpy().T),axis=1)
    Predicted_DF['Std_Err'] = Predicted_DF.Variance.apply(np.sqrt)
    Predicted_DF['Upper'] = Predicted_DF.Pred_CF.apply(np.log) + Predicted_DF.Std_Err * 1.96
    Predicted_DF['Lower'] = Predicted_DF.Pred_CF.apply(np.log) - Predicted_DF.Std_Err * 1.96
    Predicted_DF['Upper'] = Predicted_DF.Upper.apply(np.exp)
    Predicted_DF['Lower'] = Predicted_DF.Lower.apply(np.exp)
    return(model,Sim_Data,Predicted_DF)

def PlotAADT(DF):
    DF.AADT_Log.plot.hist()
    plt.grid()
    plt.show()
def PlotCrash(df):
    df['CF'] = df.N_Obs/df.Length
    plt.hist(df.CF.tolist(),density=True,bins=20)
    Mean = df.CF.mean()
    Var  = df.CF.var() 
    if Var>Mean:
        Samples = 1000
        p = 1-Mean/Var
        r = Mean*(1-p)/p
        L = np.random.negative_binomial(r, 1-p, Samples)
        S = pd.Series(L)
        act_mode = stats.mode(L)[0][0]
        thr_mode = p*(r-1)/(1-p)
        act_mean = np.mean(L)
        act_var = np.var(L)
        thr_mean = r*p/(1-p)
        thr_var = r*p/(1-p)**2
        idf = pd.DataFrame(list(range(int(df.CF.max())+1)),columns=['X'])
        idf.index = idf.X
        idf['PMF'] = idf.X.apply(lambda x:stats.nbinom.pmf(x,r,1-p))
        plt.plot(idf.PMF,'-')
        plt.vlines(act_mean,0,idf.PMF.max(),'orange')
        plt.vlines(thr_mean,0,idf.PMF.max(),'blue')
        vc = S.value_counts(True)
        vc_s = '; '.join(['{}({:0.0f}%)'.format(i,r*100) for i,r in vc.iloc[:3].iteritems()])
    plt.grid()
    #plt.title(PG_dict[pg] + '\n' + vc_s + '; {:0.3f}'.format(act_mean))
    plt.show()
def PlotSPF(res):
    plt.scatter(res[0]['AADT'],res[0]['CF'],label='Regression Dataset')
    plt.plot(res[2].AADT,res[2].Pred_CF,'red',label='Fitted SPF')
    plt.plot(res[2].AADT,res[2].Upper,'green',label='95% Interval Limits')
    plt.legend()
    plt.plot(res[2].AADT,res[2].Lower,'green')
    plt.grid()
    plt.xlabel('AADT')
    plt.ylabel('Crash Frequency per Mile per Year')
    plt.show()    
def CUREPlot(df):
    plt.plot(df.AADT.tolist(),df.Lim.tolist(),'g')
    plt.plot(df.AADT.tolist(),(-df.Lim).tolist(),'g')
    plt.plot(df.AADT.tolist(),df.CumRes.tolist(),'b')
    plt.hlines(0,df.AADT.min(),df.AADT.max(),'black')
    plt.grid()
    plt.show()       
def DerivedCMF(cmf,model):
    o = []
    for i in cmf['levels']:
        l = cmf['label'] + str(i)
        CMF = np.exp(model.params.loc[l])/np.exp(model.params.loc[cmf['label']+str(cmf['base'])])    
        o.append(CMF)
    df = pd.DataFrame({
        'Level':cmf['levels'],
        'CMF':o
    })
    df['BaseValue'] = cmf['base']
    df['Label'] = cmf['label']
    df = df[['Label','BaseValue','Level','CMF']]
    return(df)