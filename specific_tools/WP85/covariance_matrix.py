import numpy as np
import copy
import os, sys, re
import yaml
from yaml import load, add_implicit_resolver, add_constructor, Loader, dump, Dumper
yaml.warnings({'YAMLLoadWarning': False})
try:
    from yaml import CLoader, CDumper
except:
    CLoader, CDumper = Loader, Dumper

random_noise_fraction_std = 1.e-6

def load_yaml(filename, env_vars=False, complicated_input=False):
    with open(filename) as descr:
        if env_vars or complicated_input:
            dico = load(descr, Loader=Loader)
        else:
            dico = load(descr, Loader=CLoader)
    if 'include' in dico:
        if isinstance(dico['include'], list):
            for el in dico['include']:
                if os.path.exists(el):
                    dico.update(load_yaml(el, env_vars=env_vars, complicated_input=complicated_input))
                else:
                    raise IOError('file %s does not exist'%el)
        elif isinstance(dico['include'], dict):
            for el in dico['include']:
                if os.path.exists(dico['include'][el]):
                    dico.update({el: load_yaml(dico['include'][el], env_vars=env_vars, complicated_input=complicated_input)})
                else:
                    raise IOError('file %s does not exist'%dico['include'][el])
        else:
            if os.path.exists(dico['include']):
                dico.update(load_yaml(dico['include'], env_vars=env_vars, complicated_input=complicated_input))
            else:
                raise IOError('file %s does not exist'%dico['include'])
    return dico

def list_form(list_in):
    if type(list_in) == list:
        list_out = list_in
    else:
        list_out = [list_in]
    return list_out

def check_dict(dico, keys, check_none=True, prefix=None):
    none_keys = []
    not_present_keys = []
    for key in keys:
        if key not in dico:
            not_present_keys.append(key)
        else:
            if dico[key] is None:
                none_keys.append(key)
    msg = ''
    if len(not_present_keys) > 0:
        msg += 'Missing keys: %s'%(', '.join(not_present_keys))
    if (len(none_keys) > 0) and check_none:
        if len(msg) > 0:
            msg += '\n'
        msg += 'None keys: %s'%(', '.join(none_keys))
    if len(msg) > 0:
        if prefix is not None:
            msg = '%s%s'%(prefix, msg)
        raise Exception(msg)

def load_errors(error_file):
    dico = load_yaml(error_file, env_vars=True)
    if dico['errors'] is not None:
        dico['errors'] = list_form(dico['errors'])
    else:
        dico['errors'] = []
    for ii in range(len(dico['errors'])):
        subdict = dico['errors'][ii]
        prefix = 'errors: %d: '%ii
        check_dict(subdict, ['type', 'parameters'], check_none=True, prefix=prefix)
        prefix = 'errors: %d (type %s): '%(ii, subdict['type'])
        if subdict['type'] == 'noise':
            check_dict(subdict['parameters'], ['value', 'span'], check_none=True, prefix=prefix)
            check_dict(subdict['parameters'], ['time_min', 'time_max', 'value', 'bias_type', 'span', 'conversion_factor'], check_none=False, prefix=prefix)
        elif subdict['type'] == 'bias':
            check_dict(subdict['parameters'], ['value'], check_none=True, prefix=prefix)
            check_dict(subdict['parameters'], ['time', 'value', 'bias_type', 'conversion_factor'], check_none=False, prefix=prefix)
        elif subdict['type'] == 'drift':
            check_dict(subdict['parameters'], ['value'], check_none=True, prefix=prefix)
            check_dict(subdict['parameters'], ['time_min', 'time_max', 'value', 'bias_type', 'conversion_factor'], check_none=False, prefix=prefix)
        else:
            raise IOError('%serror type %s is unknown'%(prefix, subdict['type']))
        if dico['errors'][ii]['parameters']['conversion_factor'] is None:
            dico['errors'][ii]['parameters']['conversion_factor'] = 1.0
    return dico['errors']

class ErrorCovariance():
    '''
    a class to provide error covariance matrix implementation
    '''
    def __init__(self, x, matrix=None, individual_errors=False):
        '''
        constructeur
        Just provide your covariance matrix if you want it to be transformed as ErrorCovariance object. 
        x: time vector
        '''
        self.times = x
        self.size = len(x)
        if matrix is None:
            self.covar = np.zeros((self.size, self.size)).astype(np.float64)
        else:
            self.covar=matrix
        self.individual_errors = individual_errors
        self.qerror_list = []


    def add_noise_time(self, level, period_in, t_min=None, t_max=None, bias_type='centered', corr_int=None):
        '''
        adds noise covariance to the matrix
        '''
        
        period = period_in
        
        if bias_type is None:
            bias_type = 'centered'
            
        if t_min is not None:
            if t_min >= self.times[0] and t_min <= self.times[-1]:
                idmin = np.where(self.times >= t_min)[0][0]
            elif t_min > self.times[-1]:
                return 0
            else:
                idmin = 0
        else:
            t_min = self.times[0]
            idmin = 0
        if t_max is not None:
            if t_max >= self.times[0] and t_max <= self.times[-1]:
                idmax = np.where(self.times <= t_max)[0][-1]+1
            elif t_max < self.times[0]:
                return 0
            else:
                idmax = self.size
        else:
            t_max = self.times[-1]
            idmax = self.size

        dt = np.mean(np.diff(self.times[idmin:idmax]))
        timeMatrix = np.tile(self.times[idmin:idmax],(idmax-idmin,1))
        bias_type_l = bias_type.split('_')
        
        if len(bias_type_l)==1:
            if bias_type == 'nobias':
                errorCovariance = np.zeros((self.size, self.size)).astype(np.float64)
                if period < dt/4.:
                    errorCovariance[idmin:idmax,idmin:idmax] += np.diag(np.ones(idmax-idmin))
                else:
                    errorCovariance[idmin:idmax,idmin:idmax] += np.exp(-0.5*((timeMatrix-timeMatrix.T)/period)**2)
            else:
                practical_times = np.array(self.times)
                t_min = self.times[idmin]
                t_max = self.times[idmax-1]
                practical_times[0:idmin] = t_min
                practical_times[idmax:] = t_max
                if period < dt/4.:
                    errorCovariance = self.propagate_bounds(np.diag(np.ones(idmax-idmin)), idmin, idmax, self.size)
                else:
                    errorCovariance = self.propagate_bounds(np.exp(-0.5*((timeMatrix-timeMatrix.T)/period)**2), idmin, idmax, self.size)
                if bias_type == 'centered':
                    #mk2 term
                    if corr_int is None:
                        corr_int = np.min([1., 1.01*random_corr_int(t_max-t_min, period, n=1000, empirical=True)])
                    errorCovariance += ((t_min-self.times[0])**2+(self.times[-1]-t_max)**2+corr_int*(t_max-t_min)**2+ \
                        2.*(self.times[-1]-t_max+t_min-self.times[0])*gaussian_int(t_min, t_max, t_min, period, n=1000)+ \
                        2.*(self.times[-1]-t_max)*(t_min-self.times[0])*np.exp(-0.5*((t_max-t_min)/period)**2))/ \
                        (self.times[-1]-self.times[0])**2
                    #-mk*fki*fkj term
                    vector_cov = (t_min-self.times[0])*np.exp(-0.5*((practical_times-t_min)/period)**2)
                    vector_cov += (self.times[-1]-t_max)*np.exp(-0.5*((practical_times-t_max)/period)**2)
                    vector_cov[idmin:idmax] += np.array([gaussian_int(t_min, t_max, self.times[ii], period) for ii in range(idmin,idmax)])
                    vector_cov[0:idmin] += gaussian_int(t_min, t_max, t_min, period)
                    vector_cov[idmax:] += gaussian_int(t_min, t_max, t_max, period)
                    matrix_cov = np.tile(vector_cov, (self.size,1))
                    errorCovariance -= (matrix_cov+matrix_cov.T)/(self.times[-1]-self.times[0])
                elif bias_type == 'right':
                    #mk2 term
                    errorCovariance += 1.
                    #-mk*fki*fkj term
                    vector_cov = np.exp(-0.5*((practical_times-t_min)/period)**2)
                    matrix_cov = np.tile(vector_cov, (self.size,1))
                    errorCovariance -= (matrix_cov+matrix_cov.T)
                elif bias_type == 'left':
                    #mk2 term
                    errorCovariance += 1.
                    #-mk*fki*fkj term
                    vector_cov = np.exp(-0.5*((practical_times-t_max)/period)**2)
                    matrix_cov = np.tile(vector_cov, (self.size,1))
                    errorCovariance -= (matrix_cov+matrix_cov.T)
                else:
                    raise Exception('bias_type %s unknown'%bias_type)
            self.covar += level*errorCovariance
            if self.individual_errors:
                self.qerror_list.append(self.enveloppe_uncertainty_from_covar_matrix(self.times, level*errorCovariance, n=100))
        elif len(bias_type_l)>1:
            if bias_type_l[1] == 'empirical':
                if len(bias_type_l) > 2:
                    ngen = int(bias_type_l[2])
                else:
                    ngen = 1000
                #############################################
                gen_matrix = np.zeros((self.size, ngen))
                if period < dt/4.:
                    gen_matrix[idmin:idmax,:] = np.random.randn(idmax-idmin, ngen)
                else:
                    gen_covar = ErrorCovariance(self.times[idmin:idmax], matrix=np.exp(-0.5*((timeMatrix-timeMatrix.T)/period)**2))
                    gen_covar.add_randomNoise(fraction=random_noise_fraction_std)
                    gen_matrix[idmin:idmax,:] = np.dot(np.linalg.cholesky(gen_covar.covar), np.random.randn(idmax-idmin, ngen))
                if idmin > 0:
                    gen_matrix[0:idmin,:] = np.tile(gen_matrix[idmin,:], (idmin,1))
                if idmax < self.size:
                    gen_matrix[idmax:,:] = np.tile(gen_matrix[idmax-1,:], (self.size-idmax,1))
                #############################################

                if bias_type_l[0] == 'right':
                    errorCovariance = np.cov(gen_matrix-np.tile(np.mean(gen_matrix[0:idmin+1,:], axis=0), (self.size,1)))
                elif bias_type_l[0] == 'left':
                    errorCovariance = np.cov(gen_matrix-np.tile(np.mean(gen_matrix[idmax-1:,:], axis=0), (self.size,1)))
                elif bias_type_l[0] == 'centered':
                    errorCovariance = np.cov(gen_matrix-np.tile(np.mean(gen_matrix, axis=0), (self.size,1)))
                elif bias_type_l[0] == 'nobias':
                    errorCovariance = np.cov(gen_matrix)
                else:
                    raise Exception('bias_type %s unknown'%bias_type)
                self.covar += level*errorCovariance
                if self.individual_errors:
                    self.qerror_list.append(self.enveloppe_uncertainty_from_covar_matrix(self.times, level*errorCovariance, n=100))
            else:
                raise Exception('bias_type %s unknown'%bias_type)


        return 0



    def add_bias(self, level, timing, bias_type='centered'):
        '''
        adds a bias error covariance to the matrix
        '''
        if bias_type is None:
            bias_type = 'centered'
        matrix = np.zeros((self.size, self.size)).astype(np.float64)
        index = self._time2Index(timing)
        tot_time = self.times[-1]-self.times[0]
        left_time = timing-self.times[0]
        if bias_type == 'left':
            matrix[0:index,0:index] = level
        elif bias_type == 'right':
            matrix[index:self.size, index:self.size] = level
        elif bias_type == 'centered':
            matrix[0:index, 0:index] = (tot_time-left_time)**2
            matrix[index:self.size, index:self.size] = left_time**2
            matrix[0:index, index:self.size] = -left_time*(tot_time-left_time)
            matrix[index:self.size, 0:index] = -left_time*(tot_time-left_time)
            matrix *= level/(tot_time**2)
        else:
            raise Exception('bias_type %s unknown'%bias_type)
        self.covar += matrix
        if self.individual_errors:
            self.qerror_list.append(self.enveloppe_uncertainty_from_covar_matrix(self.times, matrix, n=100))
        return 0


    def add_drift(self, level, t_min=None, t_max=None, bias_type='centered'):
        '''
        adds a drift error covariance to the matrix
        '''
        if bias_type is None:
            bias_type = 'centered'
        if t_min is not None:
            if t_min >= self.times[0] and t_min <= self.times[-1]:
                idmin = np.where(self.times >= t_min)[0][0]
            elif t_min > self.times[-1]:
                return 0
            else:
                idmin = 0
        else:
            t_min = self.times[0]
            idmin = 0
        if t_max is not None:
            if t_max >= self.times[0] and t_max <= self.times[-1]:
                idmax = np.where(self.times <= t_max)[0][-1]+1
            elif t_max < self.times[0]:
                return 0
            else:
                idmax = self.size
        else:
            t_max = self.times[-1]
            idmax = self.size
        tot_time = self.times[-1]-self.times[0]
        left_time = 0.5*(t_min+t_max)-self.times[0]
        timeMatrix = np.tile(self.times[idmin:idmax],(idmax-idmin,1))
        if bias_type == 'centered':
            t_ref = t_min+(t_max-t_min)*(tot_time-left_time)/tot_time
        elif bias_type == 'left':
            t_ref = self.times[idmax-1]
        elif bias_type == 'right':
            t_ref = self.times[idmin]
        else:
            raise Exception('bias_type %s unknown'%bias_type)
        
        matrix = np.zeros((self.size, self.size)).astype(np.float64)
        matrix[idmin:idmax,idmin:idmax] = np.multiply(timeMatrix-t_ref, timeMatrix.T-t_ref)
        matrix[0:idmin, idmin:idmax] = np.tile((self.times[idmin:idmax]-t_ref)*(t_min-t_ref), (idmin,1))
        matrix[idmax:, idmin:idmax] = np.tile((self.times[idmin:idmax]-t_ref)*(t_max-t_ref), (self.size-idmax,1))
        matrix[idmin:idmax, 0:idmin] = np.tile((self.times[idmin:idmax]-t_ref)*(t_min-t_ref), (idmin,1)).T
        matrix[idmin:idmax, idmax:] = np.tile((self.times[idmin:idmax]-t_ref)*(t_max-t_ref), (self.size-idmax,1)).T
        matrix[0:idmin,0:idmin] = (t_min-t_ref)**2
        matrix[0:idmin,idmax:] = (t_max-t_ref)*(t_min-t_ref)
        matrix[idmax:,0:idmin] = (t_max-t_ref)*(t_min-t_ref)
        matrix[idmax:,idmax:] = (t_max-t_ref)**2
        self.covar += level*matrix
        if self.individual_errors:
            self.qerror_list.append(self.enveloppe_uncertainty_from_covar_matrix(self.times, level*matrix, n=100))
        return 0


    def add_randomNoise(self, matrix=None, return_matrix=False, fraction=random_noise_fraction_std, method='diagonal'):
        '''
        add random noise on the diagonal of the matrix
        '''

        if method is None:
            method = 'diagonal'
        if matrix is None:
            matrix = self.covar
        else:
            return_matrix = True
        shp = np.shape(matrix)
        if shp[0] != shp[1]:
            raise Exception('square matrix required...')
        level = fraction*np.max(matrix)
        if method == 'diagonal':
            matrix += np.diag(level*np.random.rand(shp[0]))
        elif method == 'noiselike':
            timeMatrix = np.tile(np.linspace(0.,1.,shp[0]), (shp[0], 1))
            matrix += np.dot(np.diag(level*np.random.rand(shp[0])),np.exp(-0.5*((timeMatrix-timeMatrix.T))**2))
        elif method == 'overall':
            matrix += level*np.random.rand(shp[0],shp[0])
        elif method == 'nearest':
            matrix = nearest_positive_definite_matrix(matrix)
        if return_matrix:
            return matrix
        else:
            self.covar = matrix
            return 0
        
    def is_pos_def(self, matrix=None):
        # ~ if matrix is not None:
            # ~ return np.all(np.linalg.eigh(matrix)[0] > 0)
        # ~ else:
            # ~ return np.all(np.linalg.eigh(self.covar)[0] > 0)
        if matrix is not None:
            return is_positive_definite(matrix)
        else:
            return is_positive_definite(self.covar)
            
    def is_symmetric(self, matrix=None):
        if matrix is not None:
            return (matrix.transpose() == matrix).all()
        else:
            return (self.covar.transpose() == self.covar).all()

    def _time2TimeStep(self, period):
        '''
        conversion period en time_units -> intervalles
        '''
        self.timeStep = self.times[1] - self.times[0]
        return float(period/self.timeStep)

    def _time2Index(self, date):
        '''
        conversion date -> index
        '''
        if date < self.times[0] or date > self.times[-1]:
            print("ERROR: date %s is out of bounds" %(date))
        else:
            return np.where(self.times >= date)[0][0]
            


def make_covariance_matrix(err0, t, individual_errors=False, add_diagonal_noise_fraction=random_noise_fraction_std, check_definite_positive=True, method_def_pos='diagonal'):
    
    err = copy.deepcopy(err0)
    
    if method_def_pos is None:
        method_def_pos = 'diagonal'
    
    if err is None:
        return None
    else:
        if len(err) == 0:
            return None

    covar = ErrorCovariance(t, individual_errors=individual_errors)
    for ii in range(len(err)):
        
        #complete undefined time periods
        if 'time_min' in err[ii]['parameters']:
            if err[ii]['parameters']['time_min'] is None:
                err[ii]['parameters']['time_min'] = t[0]
        if 'time_max' in err[ii]['parameters']:
            if err[ii]['parameters']['time_max'] is None:
                err[ii]['parameters']['time_max'] = t[-1]   
        
        #compute covariance values
        val = err[ii]['parameters']['value']*err[ii]['parameters']['conversion_factor']
        if err[ii]['type'] == 'noise':
            covar.add_noise_time(val**2, err[ii]['parameters']['span'], t_min=err[ii]['parameters']['time_min'], t_max=err[ii]['parameters']['time_max'], bias_type=err[ii]['parameters']['bias_type'])
        elif err[ii]['type'] == 'bias':
            if err[ii]['parameters']['time'] >= t[0] and err[ii]['parameters']['time'] <= t[-1]:
                t_close = t[np.argmin(np.abs(t-err[ii]['parameters']['time']))]
                covar.add_bias(val**2, t_close, bias_type=err[ii]['parameters']['bias_type'])
        elif err[ii]['type'] == 'drift':
            covar.add_drift(val**2, t_min=err[ii]['parameters']['time_min'], t_max=err[ii]['parameters']['time_max'], bias_type=err[ii]['parameters']['bias_type'])
            
    tries_making_defpos = 10
    if check_definite_positive:
        valid = True
        if not covar.is_pos_def():
            valid = False
            if add_diagonal_noise_fraction not in [None, False]:
                if add_diagonal_noise_fraction > 0.:
                    # ~ print('Adding noise to diagonal')
                    itry = 0
                    while((not valid) and itry < tries_making_defpos):
                        itry += 1
                        #add random noise on matrix diagonal => makes for a positive definite matrix and allows for GLS
                        covar.add_randomNoise(fraction=add_diagonal_noise_fraction, method=method_def_pos)
                        valid = covar.is_pos_def()
        if not valid:
            raise Exception('covariance matrix is not positive definite')

    return covar
