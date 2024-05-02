from mpl_toolkits.axes_grid1 import make_axes_locatable
import palettable
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np 

cmap_div = palettable.scientific.diverging.Vik_20.mpl_colormap
cmap_seq = palettable.scientific.sequential.Imola_20.mpl_colormap
cmap_pos = plt.cm.YlOrRd
cmap_neg = plt.cm.GnBu_r
colors1 = np.vstack((cmap_neg(np.linspace(0, 1, 128)),cmap_pos(np.linspace(0, 1, 128))))
cmap_disc = colors.LinearSegmentedColormap.from_list('diverging',colors1)

def plot_grid(x, y, z, imgfile=None, vmin=None, vmax=None, xlabel=None, ylabel=None, title=None, levels=None, labels=None, \
    xmin = None, xmax = None, ymin = None, ymax = None, extend = 'both', \
    xticks = None, \
    cbar_ticks=None, scalemode='lin', add_lines=[], colormap=cmap_seq, cbar_label=None, \
    probability_mode=False, grey_background=False, \
    hatches_values=None, hatches=None, hatches_pattern=None):
    """ Plot grid in general and uncertainty trees in particular"""
 
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.grid(True)
    
    create_plot_grid(x, y, z, fig, ax, vmin=vmin, vmax=vmax, xlabel=xlabel, ylabel=ylabel, title=title, \
    levels=levels, labels=labels, \
    xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, extend = extend, \
    cbar_ticks=cbar_ticks, scalemode=scalemode, add_lines=add_lines, colormap=colormap, cbar_label=cbar_label, \
    probability_mode=probability_mode, grey_background=grey_background, \
    hatches_values=hatches_values, hatches=hatches, hatches_pattern=hatches_pattern)
    
    if imgfile!=None:
        plt.savefig(imgfile,bbox_inches='tight')
        print(imgfile)
        fig.clf()
        plt.close()
    plt.show()
    
def create_plot_grid(x, y, z, fig, ax, vmin=None, vmax=None, xlabel=None, ylabel=None, title=None, levels=None, labels=None, \
    xmin = None, xmax = None, ymin = None, ymax = None, extend = 'both', \
    xticks = None, \
    cbar_ticks=None, scalemode='lin', add_lines=[], colormap=cmap_seq, cbar_label=None, \
    probability_mode=False, grey_background=False, \
    hatches_values=None, hatches=None, hatches_pattern=None):
    """ Plot grid in general and uncertainty trees in particular"""
        
    if len(np.shape(x)) != 1 or len(np.shape(y)) != 1 or len(np.shape(z)) != 2:
        raise IOError('Requires x,y as 1D arrays and z as 2D array')
    if np.shape(z)[1] != np.shape(x)[0] or np.shape(z)[0] != np.shape(y)[0]:
        raise IOError('Shape of z must be nx*ny')
    
    if not ((scalemode in ['lin', 'log']) or ('power' in scalemode)):
        raise IOError('scalemode unknown: %s'%scalemode)
    if 'power' in scalemode:
        if len(scalemode) <= 5:
            raise IOError('scalemode power requires powervalue.\nExemple: power5.0')
        powervalue = float(scalemode[5:])
    else:
        powervalue = None
        
    #probability mode
    add_text_probability_mode = False
    if probability_mode:
        vmin, vmax = 0., 100.
        if levels is None:
            levels = np.concatenate((np.zeros(1), np.linspace(68.,90.,20), np.linspace(90.,95.,20)[1:], np.linspace(95.,100.,20)[1:]), axis=0)
            add_text_probability_mode = True
        if labels is None:
            labels = [68., 90., 95.]
        if cbar_ticks is None:
            cbar_ticks = [68., 90., 95., 100.]
    else:
        if vmin is None:
            vmin = np.min(z)
        if vmax is None:
            vmax = np.max(z)
    #z.mask = np.logical_or(z.mask, np.logical_or(z>vmax, z<vmin))
    x2d, y2d = np.meshgrid(x,y)
    
    #grey background special
    if grey_background:
        x2d_grey, y2d_grey = np.meshgrid(np.linspace(x.min(), x.max(), 10), np.linspace(y.min(), y.max()*1.05, 10))
        ax.contourf(x2d_grey, y2d_grey, np.ones(np.shape(x2d_grey)), levels=np.linspace(0.,6.,4), cmap=plt.get_cmap('Greys'), alpha=1., ls='')
        
    cmap = plt.get_cmap(colormap)
    if levels is None:
        if scalemode == 'log':
            if vmin < 1.e-6:
                vmin = 1.e6
                z.mask = np.logical_or(z.mask, z<vmin)
            levels = np.logspace(np.log10(vmin),np.log10(vmax),30)
        elif scalemode == 'lin':
            levels = np.linspace(vmin,vmax,30)
        elif powervalue is not None:
            levels = vmin+(vmax-vmin)*np.linspace(0.0,1.0,30)**powervalue
    if hatches is not None:
        if hatches_values is None:
            hatches_values = z
        if scalemode == 'log':
            hatches = np.log10(hatches)
    if scalemode == 'log':
        im = ax.contourf(x2d, y2d, z, levels=levels, cmap=cmap, alpha=1., norm=LogNorm(), extend=extend)
        if hatches is not None:
            #im3 = ax.contour(x2d, y2d, z, levels=limit, color='w', alpha=1., ls='-', norm=LogNorm())
            mpl.rcParams['hatch.linewidth'] = 1
            im3 = ax.contour(x2d, y2d, hatches_values, colors='w',linewidths=2,levels=hatches, norm=LogNorm(), extend='both')
            im4 = ax.contourf(x2d, y2d, hatches_values, colors='none',edgecolor='w',levels=hatches, hatches=hatches_pattern, norm=LogNorm(), extend='both')
            for i, collection in enumerate(im4.collections):
                collection.set_edgecolor('w')
        else:
            im2 = ax.contour(x2d, y2d, z, levels=levels, cmap=cmap, alpha=1., ls='--', extend=extend, norm=LogNorm())
    elif scalemode == 'lin':
        im = ax.contourf(x2d, y2d, z, levels=levels, cmap=cmap, alpha=1., extend=extend)
        if hatches is not None:
            #im3 = ax.contour(x2d, y2d, z, levels=limit, color='w', alpha=1., ls='-')
            mpl.rcParams['hatch.linewidth'] = 1
            im3 = ax.contour(x2d, y2d, hatches_values, colors='w',linewidths=2,levels=hatches, extend='both')
            im4 = ax.contourf(x2d, y2d, hatches_values, colors='none',levels=hatches,hatches=hatches_pattern, extend='both')
            for i, collection in enumerate(im4.collections):
                collection.set_edgecolor('w')
        else:
            im2 = ax.contour(x2d, y2d, z, levels=levels, cmap=cmap, alpha=1., ls='--', extend=extend)
    elif powervalue is not None:
        im = ax.contourf(x2d, y2d, z, levels=levels, cmap=cmap, alpha=1., norm=PowerNorm(gamma=powervalue), extend='both')
        im2 = ax.contour(x2d, y2d, z, levels=levels, cmap=cmap, alpha=1., ls='--', norm=PowerNorm(gamma=powervalue))
    if labels is not None:
        if labels == 'standard':
            ax.clabel(im2, inline=True, colors='black', fontsize=12, fmt='%s')
        else:
            levels_ar = np.array(im2.levels)
            labels_select = []
            for el in labels:
                if el>=vmin and el<=vmax and el>=levels_ar.min() and el<=levels_ar.max():
                    labels_select.append(np.argmin(np.abs(levels_ar-el)))
            labels_select = list(set(labels_select))
            ax.clabel(im2, [round(im2.levels[ii],2) for ii in labels_select], inline=True, colors='black', fontsize=12, fmt='%s')
    
    if xmin is None:
        xmin = x.min()
    if xmax is None:
        xmax = x.max()
    if ymin is None:
        ymin = 0.
    if ymax is None:
        ymax = y.max()*1.05
    ax.set_xlim([xmin,xmax])
    if xticks is not None:
        ax.set_xticks(xticks)
    ax.set_ylim([ymin,ymax])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    cbar = fig.colorbar(im, orientation='vertical', cax=cax)
    if cbar_ticks is not None:
        cbar.set_ticks([el for el in cbar_ticks if el>=vmin and el<=vmax])
        cbar.set_ticklabels(['%.2f'%el for el in cbar_ticks if el>=vmin and el<=vmax])
    if cbar_label is not None:
        cbar.set_label(cbar_label,fontsize=18)
    
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    if add_text_probability_mode:
        ax.text(xlims[0]+1.08*(xlims[1]-xlims[0]), ylims[0]+0.2*(ylims[1]-ylims[0]), 'Likely', rotation=90., fontsize=12, fontweight='bold', ha='left', va='center')
        ax.text(xlims[0]+1.08*(xlims[1]-xlims[0]), ylims[0]+0.51*(ylims[1]-ylims[0]), 'Very likely', rotation=90., fontsize=12, fontweight='bold', ha='left', va='center')
        ax.text(xlims[0]+1.08*(xlims[1]-xlims[0]), ylims[0]+0.82*(ylims[1]-ylims[0]), 'Near certain', rotation=90., fontsize=12, fontweight='bold', ha='left', va='center')
    
    for add_line in add_lines:
        ax.plot(add_line[0], add_line[1], color='black', lw=2)

    if title is not None:
        ax.set_title(title, fontsize=22)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=18)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=18)
    for item in (ax.get_xticklabels() + ax.get_yticklabels() + cax.get_yticklabels()):
        item.set_fontsize(18)

def nan_array(tu):
    ar = np.empty(tu)
    ar[:] = np.nan
    return ar

def classic_mask(var, v_min=None, v_max=None):
    if v_min is None:
        v_min = -1e18
    if v_max is None:
        v_max = 1e18
    var = np.ma.masked_invalid(var)
    var = np.ma.masked_outside(var, v_min, v_max)
    return var

def mask_logic(l, operation):
    """Input list of masks"""
    if operation not in ['or', 'and']:
        raise Exception('operation parameter must be in [or,and]')
    n = len(l)
    if n == 1:
        return np.copy(l[0])
    elif n > 1:
        if np.any(np.array([np.shape(el) for el in l]) != np.shape(l[0])):
            raise Exception('shapes mismatch')
        mask = np.copy(l[0])
        if operation == 'or':
            for ii in range(1, n):
                mask = np.logical_or(mask, l[ii])
        elif operation == 'and':
            for ii in range(1, n):
                mask = np.logical_and(mask, l[ii])
        return mask
    else:
        raise Exception('Empty list')
        

def sigma_to_confidence_interval_object(min_sigma, max_sigma, number_measures):
    if min_sigma < 0. or max_sigma <= 0. or number_measures < 2:
        raise IOError('min_sigma must be >=0 and max_sigma must be >0 and number_measures must be >=2')
    x = np.concatenate((np.zeros(1).astype(np.float64), np.logspace(np.log10(min_sigma),np.log10(max_sigma),number_measures).astype(np.float64)), axis=0)
    semigauss = np.exp(-0.5*(x**2))
    y = np.zeros(number_measures+1).astype(np.float64)
    tot = 0.5*np.sqrt(2.0*np.pi)
    u = np.float64(0.)
    for ii in range(1,number_measures):
        u += 0.5*(semigauss[ii-1]+semigauss[ii])*(x[ii]-x[ii-1])
        y[ii] = u/tot
    return interp1d(x,y,kind='linear', bounds_error=False,fill_value=1.)
    
def fractal_trend_uncertainties_interp(times_in, dico_errors, eval_dico, interp_dico, covar_mat = None, least_squares_method='OLS', yvar=None, verbose=0):
    """ Compute uncertainties at all time scales for uncertainty trees

    Parameters
    ----------
    times_in:
        time array in units used in error description dictionnary (usually julian days)
    dico_errors:
        dictionnary of error description (from load_error_list_prescription_from_file)
    eval_dico:
        dictionnary of parameters for the computation of initial uncertainty data
    interp_dico:
        dictionnary of parameters for the interpolation of initial uncertainty data
    covar_mat:
        numpy array of covariance matrix (instead of dico_errors)
    least_squares_method:
        least-squares method ['OLS'|'GLS']
    """

    mode_dd = yvar is not None
    if mode_dd:
        if np.shape(yvar) != np.shape(times_in):
            raise IOError('time vector and yvar vector do not have identical sizes')
    else:
        yvar = np.ones(np.shape(times_in))
    
    n_times = len(times_in)
    time_min = times_in[0]
    time_max = times_in[-1]
    dtime = np.mean(np.diff(times_in))
    if n_times < 2:
        raise IOError('input matrix must be at least 2x2')

    # Uncertainty evaluation     
    if verbose > 0:
        print('Evaluation stage...')
    times_eval = []
    timespans_eval = []
    nspans = eval_dico['number_periods']
    if nspans < 0:
        raise RuntimeError('nspans<0')
    if eval_dico['period_min'] is None:
        eval_dico['period_min'] = dtime*1.5
    spans = np.linspace(eval_dico['period_min'], time_max-time_min-dtime*1.5, nspans)
    for it_span in range(nspans):
        tmin = time_min+spans[it_span]/2.
        tmax = time_max-spans[it_span]/2.
        nt = int(np.max([3, np.min([int(np.floor((tmax-tmin)/dtime)), eval_dico['number_dates']*(nspans-it_span)/(nspans*1.)])]))
        if nt%2 == 0:
            nt += 1
        for it in range(nt):
            timespans_eval.append(spans[it_span])
            times_eval.append(tmin+(tmax-tmin)*it/(nt-1.))
    times_eval = np.array(times_eval).astype(np.float64)
    timespans_eval = np.array(timespans_eval).astype(np.float64)
    nn = len(times_eval)
    var_uncertainties_eval = nan_array(nn)
    var_trends_eval = nan_array(nn)
    var_acc_uncertainties_eval = nan_array(nn)
    var_acc_eval = nan_array(nn)    
    if mode_dd:
        var_trends_eval = nan_array(nn)
        var_drift_probabilities_eval = nan_array(nn)
        f_sigma2confint = sigma_to_confidence_interval_object(0.01, 10.,10000)

    #covar_errors
    var_out_filled = False
    if dico_errors is not None or covar_mat is not None:
        count = 0
        count1 = 0
        for it in range(nn):
            count += 1
            if verbose > 1:
                print('Errors: %d/%d (%d valid)'%(count, nn, count1))
            tmin = times_eval[it]-timespans_eval[it]/2.
            tmax = times_eval[it]+timespans_eval[it]/2.
            valid = (tmin >= times_in[0]) and (tmax <= times_in[-1])
            if valid:
                it_min = np.searchsorted(times_in, tmin, side='right')-1
                it_max = np.searchsorted(times_in, tmax, side='right')
                if it_min < 0:
                    it_min = 0
                if it_max > n_times:
                    it_max = n_times
                if it_max-it_min > 2:
                    if dico_errors is not None:
                        covar_errors = make_covariance_matrix(dico_errors, times_in[it_min:it_max], individual_errors=False)
                        covar = covar_errors.covar
                    elif covar_mat is not None:
                        covar = covar_mat[it_min:it_max,it_min:it_max]
                    full_period = times_in[it_max-1]-times_in[it_min]
                    #H: prediction matrix with annual and semi-annual components
                    #~ H = np.vstack((np.ones(it_max-it_min), times_in[it_min:it_max]/full_period, np.cos(times_in[it_min:it_max]*2.0*np.pi/365.25), np.sin(times_in[it_min:it_max]*2.0*np.pi/365.25))).T
                    #H: prediction matrix without annual and semi-annual components => produces better results (less noise on short periods) but requires annual and semi-annual signals to be removes beforehand
                    H = np.vstack((np.ones(it_max-it_min), times_in[it_min:it_max]/full_period)).T
                    #~ try:
                    if least_squares_method.upper() == 'GLS':
                        est = polynomialGLS(times_in[it_min:it_max]/full_period, yvar[it_min:it_max], covar, 
                                            order=2, periods=[], out_comp=True, use_fit_uncertainty=False)
                    elif least_squares_method.upper() == 'OLS':
                        est = polynomialOLS(decimalyears_to_julianday_array(times_in[it_min:it_max]), yvar[it_min:it_max], 
                                            order=2, periods=[], out_comp=True, covar_matrix_for_error=covar)
                        #R, U, _ = formulaOLS(H, yvar[it_min:it_max], covar_matrix_for_error=covar)
                    else:
                        raise IOError('least squares method %s unknown'%least_squares_method)
                    U = est['uncertainties']
                    R = est['coefficients']
                    var_uncertainties_eval[it] = U[1]#/full_period
                    var_trends_eval[it] = R[1]#/full_period
                    var_acc_uncertainties_eval[it] = U[2]#/full_period
                    var_acc_eval[it] = R[2]#/full_period

                    if np.abs(var_trends_eval[it]) > 100.0:
                        var_trends_eval[it] = np.nan
                    if mode_dd:
                        var_trends_eval[it] = R[1]#/full_period
                        if np.abs(var_trends_eval[it]) > 100.0:
                            var_trends_eval[it] = np.nan
                        else:
                            var_drift_probabilities_eval[it] = f_sigma2confint(np.abs(var_trends_eval[it]/var_uncertainties_eval[it]))
                        #print('Trend uncertainty: %.5f, Trend %.5f, Drift probability: %.5f'%(var_uncertainties_eval[it]*365.25*1000., var_trends_eval[it]*365.25*1000., \
                        #    var_drift_probabilities_eval[it]*100.))
                    #else:
                        #print('Trend uncertainty: %.5f'%(var_uncertainties_eval[it]*365.25*1000.))
                        #print('Trend uncertainty: %.5f, Trend %.5f'%(var_uncertainties_eval[it]*365.25*1000., var_trends_eval[it]*365.25*1000.))
                    count1 += 1
                    #~ except:
                        #~ if verbose > 1:
                            #~ print('failed')
                        #~ pass
        var_uncertainties_eval = classic_mask(var_uncertainties_eval)
        var_trends_eval = classic_mask(var_trends_eval)
        var_acc_uncertainties_eval = classic_mask(var_acc_uncertainties_eval)
        var_acc_eval = classic_mask(var_acc_eval)
        if mode_dd:
            var_trends_eval = classic_mask(var_trends_eval)
            var_drift_probabilities_eval = classic_mask(var_drift_probabilities_eval)
        if verbose > 0:
            print('Errors: %d valid, %d non valid'%(np.sum(~var_uncertainties_eval.mask), np.sum(var_uncertainties_eval.mask)))
        var_eval_filled = True

    if mode_dd:
        mask = mask_logic([var_uncertainties_eval.mask, var_trends_eval.mask, var_drift_probabilities_eval.mask], 'or')
        times_eval = times_eval[~mask]
        timespans_eval = timespans_eval[~mask]
        var_uncertainties_eval = var_uncertainties_eval[~mask]
        var_trends_eval = var_trends_eval[~mask]
        var_acc_uncertainties_eval = var_acc_uncertainties_eval[~mask]
        var_acc_eval = var_acc_eval[~mask]
        var_drift_probabilities_eval = var_drift_probabilities_eval[~mask]
    else:
        times_eval = times_eval[~var_uncertainties_eval.mask]
        timespans_eval = timespans_eval[~var_uncertainties_eval.mask]
        var_trends_eval = var_trends_eval[~var_uncertainties_eval.mask]
        var_uncertainties_eval = var_uncertainties_eval[~var_uncertainties_eval.mask]
        var_acc_uncertainties_eval = var_acc_uncertainties_eval[~var_uncertainties_eval.mask]
        var_acc_eval = var_acc_eval[~var_uncertainties_eval.mask]

    if verbose > 0:
        print('Overall errors: %d valid, %d non valid'%(np.sum(~var_uncertainties_eval.mask), np.sum(var_uncertainties_eval.mask)))


    # Interpolation 
    if verbose > 0:
        print('Interpolation stage...')
    #interpolate
    times_interp = np.linspace(time_min, time_max, interp_dico['number_dates'])
    timespans_interp = np.linspace(0., time_max-time_min, interp_dico['number_periods'])
    Grid_X, Grid_Y = np.meshgrid(timespans_interp, times_interp)
    var_uncertainties_interp = classic_mask(griddata(np.transpose([timespans_eval, times_eval]), var_uncertainties_eval, (Grid_X, Grid_Y), method='linear').T)
    var_nearest = griddata(np.transpose([timespans_eval, times_eval]), var_uncertainties_eval, (Grid_X, Grid_Y), method='nearest').T
    mask = np.array(var_uncertainties_interp.mask)
    var_uncertainties_interp[var_uncertainties_interp.mask] = var_nearest[var_uncertainties_interp.mask]
    var_uncertainties_interp.mask = mask
    
    var_acc_uncertainties_interp = classic_mask(griddata(np.transpose([timespans_eval, times_eval]), var_acc_uncertainties_eval, (Grid_X, Grid_Y), method='linear').T)
    var_nearest = griddata(np.transpose([timespans_eval, times_eval]), var_acc_uncertainties_eval, (Grid_X, Grid_Y), method='nearest').T
    mask = np.array(var_acc_uncertainties_interp.mask)
    var_acc_uncertainties_interp[var_acc_uncertainties_interp.mask] = var_nearest[var_acc_uncertainties_interp.mask]
    var_acc_uncertainties_interp.mask = mask

    var_trends_interp = classic_mask(griddata(np.transpose([timespans_eval, times_eval]), var_trends_eval, (Grid_X, Grid_Y), method='linear').T)
    var_acc_interp = classic_mask(griddata(np.transpose([timespans_eval, times_eval]), var_acc_eval, (Grid_X, Grid_Y), method='linear').T)
    if mode_dd:
        #trends
        var_trends_interp = classic_mask(griddata(np.transpose([timespans_eval, times_eval]), var_trends_eval, (Grid_X, Grid_Y), method='linear').T)
        var_nearest = griddata(np.transpose([timespans_eval, times_eval]), var_trends_eval, (Grid_X, Grid_Y), method='nearest').T
        mask = np.array(var_trends_interp.mask)
        var_trends_interp[var_trends_interp.mask] = var_nearest[var_trends_interp.mask]
        var_trends_interp.mask = mask
        #drift
        var_drift_probabilities_interp = classic_mask(griddata(np.transpose([timespans_eval, times_eval]), var_drift_probabilities_eval, (Grid_X, Grid_Y), method='linear').T)
        var_nearest = griddata(np.transpose([timespans_eval, times_eval]), var_drift_probabilities_eval, (Grid_X, Grid_Y), method='nearest').T
        mask = np.array(var_drift_probabilities_interp.mask)
        var_drift_probabilities_interp[var_drift_probabilities_interp.mask] = var_nearest[var_drift_probabilities_interp.mask]
        var_drift_probabilities_interp.mask = mask
        #acc
        var_acc_interp = classic_mask(griddata(np.transpose([timespans_eval, times_eval]), var_acc_eval, (Grid_X, Grid_Y), method='linear').T)
        var_nearest = griddata(np.transpose([timespans_eval, times_eval]), var_acc_eval, (Grid_X, Grid_Y), method='nearest').T
        mask = np.array(var_acc_interp.mask)
        var_acc_interp[var_acc_interp.mask] = var_nearest[var_acc_interp.mask]
        var_acc_interp.mask = mask
            
    if mode_dd:
        return {'time': (times_interp), 'period': timespans_interp, 'uncertainties': var_uncertainties_interp, 'time_eval': times_eval, 'period_eval': timespans_eval, \
                'uncertainties_eval': var_uncertainties_eval, \
                'trends': var_trends_interp, 'trends_eval': var_trends_eval,
                'acc_uncertainties':var_acc_uncertainties_interp, 'acc_uncertainties_eval': var_acc_uncertainties_eval, \
                'acc': var_acc_interp, 'acc_eval': var_acc_eval,
                'drift_probabilities': var_drift_probabilities_interp, 'drift_probabilities_eval': var_drift_probabilities_eval}
    else:
        return {'time': times_interp, 'period': timespans_interp, 'uncertainties': var_uncertainties_interp, 'time_eval': times_eval, \
            'period_eval': timespans_eval, 'uncertainties_eval': var_uncertainties_eval,'trends':var_trends_interp}
    

def timeserie_plot(time_vec, msl,msl_orig,covar,lGLS,label, lGLS_2=None,label2=None, output_path=None):
    u = np.sqrt(np.diag(covar))

    dates = np.array([datetime(1950,1,1) + timedelta(el) for el in time_vec])
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1,1,1)
    ax.plot(dates, msl*1000., ls='-', lw=1, color='black', label='GMSL')
    ax.plot(dates, msl_orig*1000., ls='--', lw=1, color='black', label='GMSL without correction')
    ax.plot(dates, lGLS['reconstruction']*1000., ls='--', lw=1, marker=None, color='red', label='                    %s: \nTrend = %.2f +/- %.2f mm/yr \nAcceleration = %.2f +/- %.2f mm/yr/dec'%(label,lGLS['coefficients'][1]*1000.*365.25, \
        lGLS['uncertainties'][1]*1000.*365.25, lGLS['coefficients'][2]*1000.*365.25*(365.25*10)*2., lGLS['uncertainties'][2]*1000.*365.25*(365.25*10)*2.))   
    if lGLS_2 != None:
        ax.plot(dates, lGLS_2['reconstruction']*1000., ls='-.', lw=1, marker=None, color='blue', label='                    %s: \nTrend = %.2f +/- %.2f mm/yr \nAcceleration = %.2f +/- %.2f mm/yr/dec'%(label2,lGLS_2['coefficients'][1]*1000.*365.25, \
            lGLS_2['uncertainties'][1]*1000.*365.25, lGLS_2['coefficients'][2]*1000.*365.25*(365.25*10)*2., lGLS_2['uncertainties'][2]*1000.*365.25*(365.25*10)*2.))    
    
    plt.fill_between(dates, (msl-u)*1000., (msl+u)*1000., ls='-', lw=1, color='k', alpha=0.2, zorder=100.)
    ax.set_xlim([dates[0], dates[-1]])
    ax.set_axisbelow(True)
    ax.grid(True)
    ax.set_ylabel('GMSL (mm)', fontsize=12)
    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(15)
    ax.legend(fontsize=12, loc='upper left', fancybox=True, shadow=True)
    if output_path!=None:
        plt.savefig(output_path, dpi=600, bbox_inches = 'tight', pad_inches = 0)
    plt.show()