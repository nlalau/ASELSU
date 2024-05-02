import yaml
from yaml import load

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
