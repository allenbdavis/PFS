from lmfit import minimize, Parameters, report_fit
from astroML.time_series import lomb_scargle
from tqdm import tqdm
import numpy as np
import orbit


def fit_Keplerian(t, rv, err=None, report=False,
                  P=None, e=None, tp=None, w=None, K=None, v0=None):
    """
    Parameters
    ----------
    t: 1D array
    rv: 1D array
    err: 1D array or None
    report: bool

    Returns
    -------
    results.param: Parameters
    """
    if P is None:
        P = [None, 0, np.inf]
    if e is None:
        e = [0.2, 0.001, 0.999]
    if tp is None:
        tp = [t[0], -np.inf, np.inf]
    if w is None:
        w = [0., -np.inf, np.inf]
    if K is None:
        K = [1., 0., np.inf]
    if v0 is None:
        v0 = [0., -np.inf, np.inf]

    if err is None or 0. in err:
        err = None
        print('Warning, error bars will be ignored in fitting, since at least one data point as a reported error of 0.')

    # Define objective function: returns the array to be minimized
    def func(params, t, data, err=None):
        P_i = params['P'].value
        e_i = params['e'].value
        tp_i = params['tp'].value
        w_i = params['w'].value
        K_i = params['K'].value
        v0_i = params['v0'].value

        model = orbit.get_RV(P_i, K_i, e_i, tp_i, w_i, v0_i, t, method='rvos')
        if err is None:
            return model - data
        else:
            return (model - data) / err

    # Create a set of Parameters
    params = Parameters()
    # params.add('P', value=P_guess, min=P_guess-1, max=P_guess+1)
    # params.add('e', value=0.1, min=0.01, max=0.9)
    # params.add('tp', value=t[0])
    # params.add('w', value=0.)
    # params.add('K', value=250, min=0., max=400.)
    # params.add('v0', value=0.)

    # User guesses
    params.add('P', value=P[0], min=P[1], max=P[2])
    params.add('e', value=e[0], min=e[1], max=e[2])
    params.add('tp', value=tp[0], min=tp[1], max=tp[2])
    params.add('w', value=w[0], min=w[1], max=w[2])
    params.add('K', value=K[0], min=K[1], max=K[2])
    params.add('v0', value=v0[0], min=v0[1], max=v0[2])

    # ### Perfect guesses
    # params.add('P', value=21.22)
    # params.add('e', value=0.67)
    # params.add('tp', value=13738.529)
    # params.add('w', value=121. * (np.pi / 180.))
    # params.add('K', value=275.)
    # params.add('v0', value=0.)

    ### Slightly off guesses
    # params.add('P', value=P_guess, min=P_guess-1, max=P_guess+1)
    # params.add('e', value=0.67, min=0.05, max=1.)
    # params.add('tp', value=13736)
    # params.add('w', value=120. * (np.pi / 180.))  # Maybe I should leave w unbounded, and then just regularlize it afterwards
    # params.add('K', value=220., min=0)
    # params.add('v0', value=0.3)

    # Do fit, here with leastsq model
    result = minimize(func, params, args=(t, rv, err))

    # Output
    if report:
        report_fit(result)

    return result.params


def bootstrap_MC(t, rv, err, n_iter=100,
                 P=None, e=None, tp=None, w=None, K=None, v0=None):
    if P is None:
        P = [None, 0, np.inf]
    if e is None:
        e = [0.2, 0.001, 0.999]
    if tp is None:
        tp = [t[0], -np.inf, np.inf]
    if w is None:
        w = [0., -np.inf, np.inf]
    if K is None:
        K = [1., 0., np.inf]
    if v0 is None:
        v0 = [0., -np.inf, np.inf]

    results_boot = []
    for i in tqdm(range(n_iter)):
        rv_i = np.random.normal(loc=rv, scale=err)
        result = fit_Keplerian(t, rv_i, err=err, P=P, e=e, tp=tp, w=w, K=K, v0=v0, report=False)
        results_boot.append(result)

    return results_boot


def LSP(dates, RVs, errs, pmin=0.1, pmax=3000., n=10000):
    period_list = np.linspace(pmin, pmax, n)
    freqs = 2. * np.pi / period_list
    PG = lomb_scargle(dates, RVs, errs, freqs, generalized=True)
    return period_list, PG


def fit_orbit(system, P_G, periodList, times, rvs, niter, n_pls=None, guesses=None, ignorePeriods=None, perThres=None,
              flag='boot', report=False):
    """
    Use LM fit to determine orbital parameters from RV data.

    Parameters # UPDATE THESE
    ----------
    star:
    P_G:
    periodList:
    obs_data:
    niter:
    n_pls:
    guesses:
    ignorePeriods:
    perThres:
    flag:

    Returns
    -------
    """
    # Get number of planets to be fit
    if n_pls is None:
        n_pls = len(system.planets)

    # Define objective function: returns the array to be minimized
    def func(params, t, data):
        # Arrays that will one parameter for each planet
        P = [None] * n_pls
        e = [None] * n_pls
        tp = [None] * n_pls
        h = [None] * n_pls
        c = [None] * n_pls
        f = [None] * n_pls

        # Deal with each planet
        for i in range(0, n_pls):
            tag = str(i)
            P[i] = params['P' + tag].value
            e[i] = params['e' + tag].value
            tp[i] = params['tp' + tag].value
            h[i] = params['h' + tag].value
            c[i] = params['c' + tag].value
            f[i] = get_true_anomaly(P[i], tp[i], e[i], t)

        assert None not in P and None not in e and None not in tp and \
            None not in h and None not in c and None not in f,\
            "None-type object still in at least one of the parameter arrays"

        # Start with one planet, then add on any others
        model = h[0] * np.cos(f[0]) + c[0] * np.sin(f[0])
        for i in range(1, n_pls):
            model += h[i] * np.cos(f[i]) + c[i] * np.sin(f[i])
        model += params['v0'].value  # constant offset

        return model - data

    # Initialize array of parameters
    params_array = [None] * niter
    result_array = [None] * niter
    chisq_array = [None] * niter

    # Generate list of period peaks sorted by periodogram power
    PG_copy = np.copy(P_G)  # make copies so originals aren't mutated by sorting/zipping
    pers_copy = np.copy(periodList)  # make copies so originals aren't mutated by sorting/zipping
    PG_peaks = [pers_copy for (PG_copy, pers_copy) in
                sorted(zip(PG_copy, pers_copy))]  # periods in order of ascending pgram power
    maxima_pers = periodList[
        np.r_[True, P_G[1:] > P_G[:-1]] & np.r_[P_G[:-1] > P_G[1:], True]]  # finds periods that are local maxima
    toRemove = set(PG_peaks) ^ set(maxima_pers)  # prepare to remove periods that are not in both arrays (^ = XOR)
    for el in toRemove:
        try:
            PG_peaks.remove(el)
        except ValueError:
            pass

    # Now turning PG_peaks into a numpy array. I know it's kludgy, but I cannot make remove work for it.
    # And the np.where() only works on numpy arrays.
    PG_peaks = np.array(PG_peaks)

    # Exclude peaks we want to ignore for fitting
    if ignorePeriods is not None:
        assert type(ignorePeriods) is list and \
               [np.size(el) for el in ignorePeriods] == [2] * (int(np.size(ignorePeriods) / 2)), \
            'ignorePeriods must be of the form: [[min,max],[mix,max],...].'
        for pair in ignorePeriods:
            minval, maxval = pair[0], pair[1]
            PG_peaks = PG_peaks[np.where((PG_peaks < minval) | (PG_peaks > maxval))]

    topNguesses = PG_peaks[-n_pls:]

    # Do a bunch of fits
    for n in range(0, niter):
        # Choose some periods to guess for each planet
        p_guesses = np.copy(topNguesses)
        np.random.shuffle(p_guesses)
        params = Parameters()
        for m in range(0, n_pls):
            # Create a set of Parameters
            tag = str(m)
            if perThres is None:
                params.add('P' + tag, value=p_guesses[m], min=0)
            else:
                params.add('P' + tag, value=p_guesses[m], min=p_guesses[m] * (1 - perThres),
                           max=p_guesses[m] * (1 + perThres))
            params.add('e' + tag, value=0.3, min=0., max=0.9)
            params.add('tp' + tag, value=times[0] + (np.random.random() * p_guesses[m]))
            params.add('h' + tag, value=0)  #4. * np.random.random() - 2.)
            params.add('c' + tag, value=0)  #4. * np.random.random() - 2.)

        params.add('v0', value=0.5 * np.random.random() - 0.25)  # offset parameter; just 1 of these

        # Do fit, here with leastsq model
        result = minimize(func, params, args=(times, rvs))

        # Get chisq, which we will use to decide if this is the best fit
        chisq = np.sum(result.residual ** 2)

        # Save params, result, and chisq
        params_array[n] = params
        result_array[n] = result
        chisq_array[n] = chisq

    # Continue, now using the best fit according to chisq
    n_best = np.argmin(chisq_array)
    params = params_array[n_best]
    result = result_array[n_best]

    # Calculate final result
    final = rvs + result.residual

    # Write error report
    if report:
        report_fit(params)

    # Fitted params = [P,e,tp,h,c]*n_pls + v0
    # For each planet, extract the astrophysical parameters from the fit and save them
    p_opt = [None] * n_pls
    e_opt = [None] * n_pls
    tp_opt = [None] * n_pls
    w_opt = [None] * n_pls
    K_opt = [None] * n_pls
    msini_fit = [None] * n_pls
    params_out = [None] * n_pls
    RV_fit_pl = [None] * n_pls  # one planet's component of the total RV fit

    for m in range(0, n_pls):
        tag = str(m)
        p_opt[m] = params['P' + tag].value
        e_opt[m] = params['e' + tag].value
        tp_opt[m] = params['tp' + tag].value % p_opt[m]  # take the first tp in the observation window for consistency
        w_opt[m] = (np.arctan(-params['c' + tag].value / params['h' + tag].value)) % (2. * np.pi)  # [0-2pi) rads

        # Ensure that the sign of Sin(w) == the sign of the numerator: -c
        # Deals with the ambiguity in taking the arctan, above.
        # This is a condition specified by Wright & Howard 2009
        if not np.sign(np.sin(w_opt[m])) == np.sign(-params['c' + tag].value):
            w_opt[m] = (w_opt[m] - np.pi) % (2. * np.pi)

        K_opt[m] = np.sqrt(params['h' + tag].value ** 2 + params['c' + tag].value ** 2)
        msini_fit[m] = (1. / mEarth) * (K_opt[m]) * ((2. * np.pi * G / (p_opt[m] * day)) ** (-1. / 3.)) * (
                    (system.star.mass * mSun) ** (2. / 3.)) * (np.sqrt(1 - e_opt[m] ** 2.))

        params_out[m] = np.array([p_opt[m], e_opt[m], tp_opt[m], w_opt[m], K_opt[m], msini_fit[m]])

        f_opt = get_true_anomaly(p_opt[m], tp_opt[m], e_opt[m],
                                 times)  # this is a temp variable, so not saved in an array
        RV_fit_pl[m] = K_opt[m] * (np.cos(w_opt[m] + f_opt) + e_opt[m] * np.cos(w_opt[m]))

    assert not None in p_opt and not None in e_opt and not None in tp_opt and not None in w_opt \
           and not None in K_opt and not None in msini_fit and not None in params_out and not None in RV_fit_pl, \
        "None-type object still in at least one of the parameter arrays after fitting."

    RV_fit = np.sum(RV_fit_pl, axis=0)  # total fitted RV

    if "boot" == flag:

        return params_out
    else:
        assert False
        # star.planets_fit = [None] * n_pls
        # for i in range(0, n_pls):
        #     p = Planet(star, msini_fit[i], p_opt[i], e_opt[i], 90., w_opt[i], tp_opt[i], isReal=False)
        #     star.planets_fit[i] = p
        # star.nPlanets_fit = n_pls
        # star.RV_fit = RV_fit
        # # star.params_out = params_out
        # # star.params_out_print = np.array([p_opt,e_opt,tp_opt,np.array(w_opt)*180./pi,K_opt,msini_fit])
        # star.params_LMfit = params