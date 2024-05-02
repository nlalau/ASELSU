import xarray as xr 
import numpy as np 
import netCDF4
from datetime import datetime,timedelta
import time
from scipy.interpolate import griddata,interp1d
from ASELSU.common_tools.time_tools import julianday_to_decimalyears_array,datetime_to_decimalyears

## OLS and GLS functions

def inversion(m):
    lmin = np.linalg.eigh(m)[0][0]
    m = np.linalg.pinv(m/lmin)/lmin
    return m
    
## OLS
def formulaOLS(H, Y, covar_matrix_for_error=None, use_fit_uncertainty=False):
    A = np.dot(np.transpose(H), H)
    A_inv = inversion(A)
    R = np.dot(A_inv, np.dot(np.transpose(H), Y))
    if covar_matrix_for_error is not None:
        A_bis = np.dot(np.transpose(H), np.dot(covar_matrix_for_error, H))
        if not use_fit_uncertainty:
            sigma2 = 1.
        else:
            DOF = np.float64(np.shape(H)[0]-2)
            Yres = Y-np.dot(H, R)
            sigma2 = np.dot(np.transpose(Yres), np.dot(inversion(covar_matrix_for_error), Yres))/DOF
        pcov = sigma2*np.dot(np.dot(A_inv, A_bis), A_inv)
        U = np.sqrt(np.diag(pcov))
    else:
        Yres = Y-np.dot(H, R)
        DOF = np.float64(np.shape(H)[0]-2)
        Serr = np.float64(np.sum(np.square(Yres)))
        pcov = (Serr/DOF)*A_inv
        U = np.sqrt(np.diag(pcov))
    return R,U, pcov
    
def polynomialOLS(X, Y, order=1, periods=[], out_comp=False, covar_matrix_for_error=None, debug=False, use_fit_uncertainty=False, additional_H_dict=None):
    '''Ordinary Least Squares -can be used with Y = constant vector [np.zeros(len(time), dtype=np.float64)] whereas GLS imperatively needs Y 
    
    Parameters
    ----------
    X : array
        time (size N)
    Y : array
        variable (size N)
    order : int
        if == 1 then look for ax+b fitting
    periods : array                
        if not empty, compute fitting after removing a sinusoidal signal of period x. Must be of the same unit as time - usually in days, eg: [365.25, 365.25/2] if you want to remove annual and semi-annual cycles.
        Adjusts cosine and sine functions of each period you precised in "periods". For 2 periods, 
    out_comp : bool
        if True more information are given in output 
    covar_matrix_for_error: array  
        covariance matrix of errors on Y (size N*N)
        
    Returns
    -------
    dico : dict
        Contains 'coefficients': (order+1) vector which contains values of estimated bias, trend, acceleration /!\ need to fill Y with the real variable to get right values
                'uncertainties': (order+1) vector which contains uncertainties of bias, trend, acceleration, ...
                'detail'
    '''

    n0 = order+1+2*len(periods)
    if additional_H_dict is not None:
        H = np.concatenate((np.empty((len(X), n0)), additional_H_dict['matrix']), axis=1)
        n0 = np.shape(H)[1]
    else:
        H = np.empty((len(X), n0))
    #we construct the prediction matrix H so that values are between [0,1] for numerical stability, and X centered
    x_mean = X.mean()
    X_adim = (X-x_mean)/(X[-1]-X[0])
    for ii in range(order+1):
        H[:,ii] = X_adim**ii
    for ii in range(len(periods)):
        H[:,order+1+2*ii] = np.cos(2.*np.pi*X/periods[ii])
        H[:,order+1+2*ii+1] = np.sin(2.*np.pi*X/periods[ii])
    R,U,_ = formulaOLS(H, Y, covar_matrix_for_error=covar_matrix_for_error, use_fit_uncertainty=use_fit_uncertainty)
    dico = dict()
    if out_comp:
        Ymod = np.dot(H, R)
        dico.update({'reconstruction': Ymod, 'residue': Y-Ymod, 'detail': [R[ii]*H[:,ii] for ii in range(n0)]})
    for ii in range(order+1):
        R[ii] *= (X[-1]-X[0])**(-ii)
        U[ii] *= (X[-1]-X[0])**(-ii)
    if additional_H_dict is not None:
        R[order+1+2*len(periods):n0] *= additional_H_dict['coeff_adim']
        U[order+1+2*len(periods):n0] *= additional_H_dict['coeff_adim']
    dico['coefficients'] = R
    dico['uncertainties'] = U
    dico['xref'] = x_mean
    dico['order'] = order
    dico['periods'] = periods
    dico['additional_H_dict'] = additional_H_dict
    if debug:
        print('times', X.min(), X.mean(), X.max(), 'y', Y.min(), Y.mean(), Y.max(), \
            'covar', np.diag(covar_matrix_for_error).min(), np.diag(covar_matrix_for_error).mean(), np.diag(covar_matrix_for_error).max())
        if len(dico['uncertainties']) > 2:
            print(dico['uncertainties'][1]*365.25*1000., dico['uncertainties'][2]*365.25**2*1000.)
        else:
            print(dico['uncertainties'][1]*365.25*1000.)
    
    return dico

## GLS
def formulaGLS(H, Y, covar_matrix, use_fit_uncertainty=False):
    covar_matrix_inv = inversion(covar_matrix)
    A = np.dot(np.transpose(H), np.dot(covar_matrix_inv, H))
    A_inv = inversion(A)
    R = np.dot(np.dot(A_inv, np.transpose(H)), np.dot(covar_matrix_inv, Y))
    if not use_fit_uncertainty:
        sigma2 = 1.
    else:
        DOF = np.float64(np.shape(H)[0]-2)
        Yres = Y-np.dot(H, R)
        sigma2 = np.dot(np.transpose(Yres), np.dot(inversion(covar_matrix), Yres))/DOF
    U = np.sqrt(sigma2*np.diag(A_inv))
    return R,U
    
def polynomialGLS(X, Y, covar_matrix, order=1, periods=[], out_comp=False, use_fit_uncertainty=False):
    
    n0 = order+1+2*len(periods)
    H = np.empty((len(X), n0))
    # prediction matrix H contruct so that values are between [0,1] for numerical stability, and X centered
    x_mean = X.mean()
    X_adim = (X-x_mean)/(X[-1]-X[0])
    for ii in range(order+1):
        H[:,ii] = X_adim**ii
    for ii in range(len(periods)):
        H[:,order+1+2*ii] = np.cos(2.*np.pi*X/periods[ii])
        H[:,order+1+2*ii+1] = np.sin(2.*np.pi*X/periods[ii])

    R,U = formulaGLS(H, Y, covar_matrix, use_fit_uncertainty=use_fit_uncertainty)
    
    dico = dict()
    if out_comp:
        Ymod = np.dot(H, R)
        dico.update({'reconstruction': Ymod, 'residue': Y-Ymod, 'detail': [R[ii]*H[:,ii] for ii in range(n0)]})
    for ii in range(order+1):
        R[ii] *= (X[-1]-X[0])**(-ii)
        U[ii] *= (X[-1]-X[0])**(-ii)
    dico['coefficients'] = R
    dico['uncertainties'] = U
    dico['xref'] = x_mean
    dico['order'] = order
    dico['periods'] = periods
    
    return dico
    
## Corrections functions
### TOPEX-A correction
def tpa_drift_correction(tpa_corr_file, jd, method='v-shape'):
    """ Compute TOPEX-A drift correction for GMSL time series
    V-shape correction from Cazenave and the WCRP global sea level budget group, ESSD, 2018
    
    Input
    -----
    jd: numpy array
        time array [in julian days]
    method: string ['linear'|'v-shape'(default)|'v-smoothed']
        TPA drift correction method
    Returns
    -------
    tpa_corr: numpy array
        Correction for TPA drift [in meters]
    """

    dy = julianday_to_decimalyears_array(jd)
    tpa_corr = np.zeros(jd.shape)

    if method.lower() == 'linear':
        ind = np.where(dy<1999)[0]
        tpa_corr[ind] = 0.0015*(dy[ind]-dy[ind[-1]])
    elif method.lower() == 'v-shape' or method.lower() == 'v':
        date1 = datetime_to_decimalyears(datetime(1995,7,31))
        date2 = datetime_to_decimalyears(datetime(1999,2,28))
        ind1 = np.where(dy<=date1)[0]
        ind2 = np.where(dy<=date2)[0]
        tpa_corr[ind2] = 0.003*(dy[ind2]-dy[ind2[-1]])
        tpa_corr[ind1] = tpa_corr[ind1]-0.004*(dy[ind1]-dy[ind1[-1]])
    elif 'smoothed' in method.lower():
        ds_tpa = xr.open_dataset(tpa_corr_file,decode_times=False)
        jd_tpa = ds_tpa.time.values
        tpa_corr_init = ds_tpa.tpa_corr.values
        tpa_corr = np.interp(jd, jd_tpa, tpa_corr_init, left=tpa_corr_init[0], right=tpa_corr_init[-1])
    else:
        raise Exception('Unknown method in correct_for_tpa_drift function.')

    return tpa_corr


def correct_gmsl_for_tpa_drift(tpa_corr_file, jd, gmsl, method='v-shape'):
    """ Correct GMSL time series for TOPEX-A drift
    
    Input
    -----
    jd: numpy array
        time array [in julian days]
    gmsl: numpy array
        GMSL time series to correct [in meters]
    method: string ['linear'|'v-shape'(default)|'v-smoothed']
        TPA drift correction method
    Returns
    -------
    gmsl_corr: numpy array
        GMSL corrected for TPA drift [in meters]
    """

    tpa_corr = tpa_drift_correction(tpa_corr_file, jd,method=method)
    gmsl_corr = gmsl-tpa_corr

    return gmsl_corr

### Correction of Jason-3 wet troposphere correction drift 
def correct_for_jason3_wtc_drift(ncfile, jd, gmsl):
    """ Correct GMSL time series for Jason-3 radiometer drift
    
    Input
    -----
    ncfile: str
        path to j3_wtc_drift_correction_cdr_al_s3a.nc file containing Jason-3 correction
    jd: numpy array
        time array [in julian days]
    gmsl: numpy array
        GMSL time series to correct [in meters]
    Returns
    -------
    gmsl_corr: numpy array
        GMSL corrected for Jason-3 WTC drift [in meters]
    """

    ds = xr.open_dataset(ncfile,decode_times=False)
    jd_j3_corr = ds.time.values
    j3_corr = np.ma.masked_invalid(ds.j3_corr)

    j3_corr_interp = np.interp(jd, jd_j3_corr, j3_corr, left=0, right=j3_corr[-1])
    gmsl_corr = gmsl-j3_corr_interp

    return gmsl_corr
