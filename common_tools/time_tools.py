import xarray as xr 
import numpy as np 
import netCDF4
from datetime import datetime,timedelta
import time
from scipy.interpolate import griddata,interp1d
import matplotlib.pyplot as plt
import matplotlib as mpl

def datetime_to_decimalyears(datetime_obj):
    
    def sinceEpoch(datetime_obj): # returns seconds since epoch
        return time.mktime(datetime_obj.timetuple())
    s = sinceEpoch
    year = datetime_obj.year
    startOfThisYear = datetime(year=year, month=1, day=1)
    startOfNextYear = datetime(year=year+1, month=1, day=1)
    yearElapsed = s(datetime_obj) - s(startOfThisYear)
    yearDuration = s(startOfNextYear) - s(startOfThisYear)
    fraction = yearElapsed/yearDuration
    return datetime_obj.year + fraction

def decimalyears_to_datetime_array(decyear_array):
    year = [int(el) for el in decyear_array]
    rem = decyear_array - year
    base = [datetime(el, 1, 1) for el in year]
    return np.array([el + timedelta(seconds=(el.replace(year=el.year + 1) - el).total_seconds() * rem[iel]) for iel, el in enumerate(base)])

def datetime_to_decimalyears_array(datetime_array):
    return np.array([datetime_to_decimalyears(el) for el in datetime_array])

def julianday_to_datetime_array(julianday_array):
    return datetime(1950,1,1) + np.array([timedelta(days=float(el)) for el in julianday_array])

def julianday_to_decimalyears_array(jday_ar):
    return datetime_to_decimalyears_array(julianday_to_datetime_array(jday_ar))

def datetime_to_julianday_array(datetime_array):
    return np.array([el.total_seconds() for el in datetime_array - datetime(1950,1,1)], dtype=np.float64)/(24.*3600.)

def decimalyears_to_julianday_array(dec_ar):
    return datetime_to_julianday_array(decimalyears_to_datetime_array(dec_ar))
