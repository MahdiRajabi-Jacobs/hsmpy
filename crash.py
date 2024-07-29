#from multiprocessing.sharedctypes import Value
import pandas as pd
import numpy as np
from scipy import stats

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
        elif all_crashes == 0:
            p = 0
        # Else, calculate the probability using the beta distribution survival function
        else:
            p = stats.beta.sf(self.mean_proportion, self.alpha + type_crashes, self.beta + all_crashes - type_crashes)
            if np.isnan(p):
                raise ValueError("Invalid result!")
        return p

class BinaryField(object):
    def __init__(self, name, mask_expressions=[], cid_list=[],uid_list=[],pid_list=[]):
        self.name = name
        self.mask_expressions = mask_expressions
        self.cid_list = cid_list
        self.uid_list = uid_list
        self.pid_list = pid_list
class MaskExpression(object):
    def __init__(self, mask, level='crash', logic='or', file='crash'):
        self.mask = mask
        self.level = level
        self.logic = logic
        self.file = file
class CrashDataBinaries(object):
    def __init__(self, csh, unt, per, cid='CID', uid='CID_UID', pid='CID_UID_PID'):
        # Log data inputs
        self.csh_source = csh
        self.unt_source = unt
        self.per_source = per
        self.csh_len = csh.shape[0]
        self.unt_len = unt.shape[0]
        self.per_len = per.shape[0]
        if len(set([self.csh_len,self.unt_len,self.per_len]))!=3:
            raise ValueError('This tool is optimized to work only if the length of crash, unit, and person files differ')
        # Log ID field names
        self.cid = cid
        self.uid = uid
        self.pid = pid
        # Initialize emphasis areas
        self.file_selector = {self.csh_len:'crash',self.unt_len:'unit',self.per_len:'person'}
        self.eas = []
        self.ea_names = []
        
    def create_mask(self,m):
        if 'mask' in m.keys():
            if m['mask'].shape[0] in self.file_selector.keys():
                file = self.file_selector[m['mask'].shape[0]]
            else:
                raise ValueError('mask length ({}) not matching any of files (crash:{}, unit:{}, person:{})'.format(m['mask'].shape[0],self.csh_len,self.unt_len,self.per_len))
        else:
            raise ValueError('No masks passed')
        logic='or'
        if 'logic' in m.keys():
            if m['logic']=='and':
                logic = 'and'
        level = file
        if 'level' in m.keys():
            level = m['level']
        return(MaskExpression(mask=m['mask'],logic=logic,file=file,level=level))
    def map_ids(self,ids,from_id,to_id):
        if from_id=='crash':
            if to_id=='cid':
                return(sorted(set(ids)))
            if to_id=='uid':
                mask = self.unt_source[self.cid].isin(ids)
                return(sorted(set(self.unt_source[mask][self.uid])))
            if to_id=='pid':
                mask = self.per_source[self.cid].isin(ids)
                return(sorted(set(self.per_source[mask][self.pid])))
        if from_id=='vehicle':
            if to_id=='cid':
                mask = self.unt_source[self.uid].isin(ids)
                return(sorted(set(self.unt_source[mask][self.cid])))
            if to_id=='uid':
                return(sorted(set(ids)))
            if to_id=='pid':
                mask = self.per_source[self.uid].isin(ids)
                return(sorted(set(self.per_source[mask][self.pid])))
        if from_id=='person':
            if to_id=='cid':
                mask = self.per_source[self.pid].isin(ids)
                return(sorted(set(self.per_source[mask][self.cid])))
            if to_id=='uid':
                mask = self.per_source[self.pid].isin(ids)
                return(sorted(set(self.per_source[mask][self.uid])))
            if to_id=='pid':
                return(sorted(set(ids)))
    def ids_from_mask(self,mask_exp):
        if mask_exp.level=='crash':
            if mask_exp.file=='crash':
                return(sorted(set(self.csh_source[mask_exp.mask][self.cid])))
            if mask_exp.file=='unit':
                return(sorted(set(self.unt_source[mask_exp.mask][self.cid])))
            if mask_exp.file=='person':
                return(sorted(set(self.per_source[mask_exp.mask][self.cid])))
        if mask_exp.level=='vehicle':
            if mask_exp.file=='crash':
                cids = set(self.csh_source[mask_exp.mask][self.cid])
                return(sorted(set(self.unt_source[self.unt_source[self.cid].isin(cids)][self.uid])))
            if mask_exp.file=='unit':
                return(sorted(set(self.unt_source[mask_exp.mask][self.uid])))
            if mask_exp.file=='person':
                return(set(self.per_source[mask_exp.mask][self.uid]))
        if mask_exp.level=='person':
            if mask_exp.file=='crash':
                cids = set(self.csh_source[mask_exp.mask][self.cid])
                return(sorted(set(self.per_source[self.per_source[self.cid].isin(cids)][self.pid])))
            if mask_exp.file=='unit':
                uids = set(self.unt_source[mask_exp.mask][self.uid])
                return(sorted(set(self.per_source[self.per_source[self.uid].isin(uids)][self.pid])))
            if mask_exp.file=='person':
                return(sorted(set(self.per_source[mask_exp.mask][self.pid])))
    def csh_flags(self, joined=True, positive=1, negative=0):
        # Create flagging columns
        cols = [self.csh_source[c] for c in list(self.csh_source) if not c in self.ea_names] if joined else [] 
        for ea in self.eas:
            mask = self.csh_source[self.cid].isin(ea.cid_list)
            cols.append(mask.replace({True: positive, False: negative}).rename(ea.name))
        return pd.concat(cols, axis=1)
    def unt_flags(self, joined=True, positive=1, negative=0):
        # Create flagging columns
        cols = [self.unt_source[c] for c in list(self.unt_source) if not c in self.ea_names]
        for ea in self.eas:
            mask = self.unt_source[self.uid].isin(ea.uid_list)
            cols.append(mask.replace({True: positive, False: negative}).rename(ea.name))
        # Consolidate flagging columns
        return pd.concat(cols, axis=1)
    def per_flags(self, joined=True, positive=1, negative=0):
        # Create flagging columns
        cols = [self.per_source[c] for c in list(self.per_source) if not c in self.ea_names] 
        for ea in self.eas:
            mask = self.per_source[self.pid].isin(ea.pid_list)
            cols.append(mask.replace({True: positive, False: negative}).rename(ea.name))
        # Consolidate flagging columns
        return pd.concat(cols, axis=1)
    def define_new(self, masks, name):
        if len(masks)==0:
            return()
        mask_expressions = [self.create_mask(m) for m in masks]
        mask_exp0 = mask_expressions[0]
        ids = self.ids_from_mask(mask_exp=mask_exp0)
        cid_list = self.map_ids(ids=ids,from_id=mask_exp0.level,to_id='cid')
        uid_list = self.map_ids(ids=ids,from_id=mask_exp0.level,to_id='uid')
        pid_list = self.map_ids(ids=ids,from_id=mask_exp0.level,to_id='pid')
        for mask_exp in mask_expressions[1:]:
            ids = self.ids_from_mask(mask_exp=mask_exp)
            cid_list1 = self.map_ids(ids=ids,from_id=mask_exp.level,to_id='cid')
            uid_list1 = self.map_ids(ids=ids,from_id=mask_exp.level,to_id='uid')
            pid_list1 = self.map_ids(ids=ids,from_id=mask_exp.level,to_id='pid')
            if mask_exp.logic=='and':
                cid_list = sorted(set(cid_list).intersection(set(cid_list1)))
                uid_list = sorted(set(uid_list).intersection(set(uid_list1)))
                pid_list = sorted(set(pid_list).intersection(set(pid_list1)))
            if mask_exp.logic=='or':
                cid_list = sorted(set(cid_list).union(set(cid_list1)))
                uid_list = sorted(set(uid_list).union(set(uid_list1)))
                pid_list = sorted(set(pid_list).union(set(pid_list1)))

        self.eas.append(BinaryField(name=name,mask_expressions= mask_expressions,cid_list=cid_list,uid_list=uid_list,pid_list=pid_list))
        self.ea_names.append(name)
