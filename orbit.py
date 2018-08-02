import numpy as np
from constants import *
from bodies import System, Star, Planet
from lmfit import minimize, Parameters, Parameter, report_fit
from scipy.interpolate import interp1d


def calcTrueAnomaly(P, tp, e, t):
    """
    Calculates the true anomaly from the time and orbital elements.

    P:
    tp:
    e:
    t:
    :return:
    """
    phase = (t-tp)/P #phase at each obsList time
    M = 2.*np.pi*(phase - np.floor(phase)) #Mean Anom array: at each obsList time
    E1 = calcKepler(M, np.array([e]))

    n1 = 1. + e
    n2 = 1. - e

    #True Anomaly:
    return 2.*np.arctan(np.sqrt(n1/n2)*np.tan(E1/2.))


def calcKepler(Marr_in, eccarr_in):
    """
    Algorithm adapted from kepler.pro, referenced in Wright & Howard 2009.
    Returns Eccentric anomaly, given mean anomaly and eccentricity.

    Parameters
    ----------
    Marr_in:
    eccarr_in:

    Returns
    -------
    """

    nm = np.size(Marr_in)
    nec = np.size(eccarr_in)

    if nec == 1 and nm > 1:
        eccarr = eccarr_in #[eccarr_in for x in range(nm)]
    else:
        eccarr = eccarr_in

    if nec > 1 and nm == 1:
        Marr = Marr_in #[Marr_in for x in range(nec)]
    else:
        Marr = Marr_in

    conv = 1.E-12 #threshold for convergence
    k = 0.85 #some parameter for guessing ecc
    ssm = np.sign(np.sin(Marr))
    Earr = Marr+(ssm*k*eccarr)  #first guess at E
    fiarr = (Earr-(eccarr*np.sin(Earr))-Marr)  #E - e*sin(E)-M    ; should go to 0 when converges
    convd = np.where(abs(fiarr) > conv) #which indices are unconverged

    count = 0
    while np.size(convd) > 0:
        count += 1

        # M = np.copy(Marr[convd]) #we only run the unconverged elements
        ecc = eccarr #[convd] ??
        E = np.copy(Earr[convd])
        fi = np.copy(fiarr[convd])

        fip = 1.-ecc*np.cos(E) #;d/dE(fi) ;i.e.,  fi^(prime)
        fipp = ecc*np.sin(E)  #;d/dE(d/dE(fi)) ;i.e.,  fi^(\prime\prime)
        fippp = 1.-fip #;d/dE(d/dE(d/dE(fi))) ;i.e.,  fi^(\prime\prime\prime)

        d1 = -fi/fip                             #;first order correction to E
        d2 = -fi/(fip+(d1*fipp/2.))                #;second order correction to E
        d3 = -fi/(fip+(d2*fipp/2.)+(d2*d2*fippp/6.)) #;third order correction to E

        E += d3 #apply correction to E

        Earr[convd] = E #update values

        fiarr = (Earr-eccarr*np.sin(Earr)-Marr)     #;how well did we do?
        convd = np.where(abs(fiarr) > conv)   #;test for convergence; update indices

        if count > 100:
            #print("WARNING!  Kepler's equation not solved!!!")
            break

    return Earr

def get_RV(P, K, e, tp, w, v0, t):
    """
    Get ideal radial velocity curve of a star that has a planet with these orbital elements.

    Parameters
    ----------
    P: float
        Period (days)
    K: float
        Doppler semi-amplitude (m/s)
    e: float
        Eccentricity (0 <= e < 1)
    tp: float
        Periastron passage (JD)
    w: float
        Argument of periastron of the ??? (radians)  # planet or star???
    v0: float
        Velocity offset of system (m/s)
    t: list or ndarray
        List of observing epochs (JD)

    Returns
    -------
    RV: 1D numpy array
        List of radial velocities (m/s). Positive velocities correspond to star moving away from observer.
    """
    if type(t) is not np.ndarray:
        t = np.array(t)

    nu = get_true_anomaly(P, tp, e, t)
    RV = K * (np.cos(nu+w) + e*np.cos(w)) + v0
    return RV


def correct_eccentric_anomaly(E, e):
    """
    ???

    Parameters
    ----------
    E: float
        Eccentric anomaly
    e: float
        Eccentricity

    Returns
    -------

    """
    val = np.arccos((np.cos(E) - e) / (1 - e * np.cos(E)))
    if E > np.pi:
        val_corrected = (2*np.pi) - val
    else:
        val_corrected = val
    return val_corrected


def get_true_anomaly(P, tp, e, t):
    """
    Rather than solving Kepler's equation, this interpolates the solution. Slower.

    Parameters
    ----------
    P:
    tp:
    e:
    t:

    Returns
    -------
    """
    M = (2 * np.pi / P) * np.mod(tp - t, P)  # M at evenly spaced times
    E_interp = np.arange(-2 * np.pi, 2 * np.pi * 2, .05)
    M_interp = E_interp - e * np.sin(E_interp)  # Ms evaluated at the list of Es
    f = interp1d(M_interp, E_interp, kind='cubic')  # Es = f(Ms)
    E = f(M)
    nu = np.array([correct_eccentric_anomaly(E_i, e) for E_i in E])
    return nu


def fit_orbit(star, P_G, periodList, obs_data, niter, n_pls=None, guesses=None, ignorePeriods=None, perThres=None,
              flag=None):
    """
    Use LM fit to determine orbital parameters from RV data.

    Parameters
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
    if n_pls == None:
        n_pls = np.size(star.planets)

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

        assert not None in P and not None in e and not None in tp and not None in h and not None in c and not None in f, \
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
    if not ignorePeriods == None:

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
            if perThres == None:
                params.add('P' + tag, value=p_guesses[m], min=0)
            else:
                params.add('P' + tag, value=p_guesses[m], min=p_guesses[m] * (1 - perThres),
                           max=p_guesses[m] * (1 + perThres))
            params.add('e' + tag, value=0.7 * np.random.random(), min=0, max=1)
            params.add('tp' + tag, value=star.t0 + (np.random.random() * p_guesses[m]))
            params.add('h' + tag, value=4. * np.random.random() - 2.)
            params.add('c' + tag, value=4. * np.random.random() - 2.)

        params.add('v0', value=0.5 * np.random.random() - 0.25)  # offset parameter; just 1 of these

        # Do fit, here with leastsq model
        result = minimize(func, params, args=(star.obsList, obs_data))

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
    final = obs_data + result.residual

    # Write error report
    # report_fit(params)

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
        w_opt[m] = (np.arctan(-params['c' + tag].value / params['h' + tag].value)) % (2. * pi)  # [0-2pi) rads

        # Ensure that the sign of Sin(w) == the sign of the numerator: -c
        # Deals with the ambiguity in taking the arctan, above.
        # This is a condition specified by Wright & Howard 2009
        if not np.sign(np.sin(w_opt[m])) == np.sign(-params['c' + tag].value):
            w_opt[m] = (w_opt[m] - pi) % (2. * pi)

        K_opt[m] = np.sqrt(params['h' + tag].value ** 2 + params['c' + tag].value ** 2)
        msini_fit[m] = (1. / mEarth) * (K_opt[m]) * ((2. * pi * G / (p_opt[m] * day)) ** (-1. / 3.)) * (
                    (star.massStar * mSun) ** (2. / 3.)) * (np.sqrt(1 - e_opt[m] ** 2.))

        params_out[m] = np.array([p_opt[m], e_opt[m], tp_opt[m], w_opt[m], K_opt[m], msini_fit[m]])

        f_opt = get_true_anomaly(p_opt[m], tp_opt[m], e_opt[m],
                                 star.obsList)  # this is a temp variable, so not saved in an array
        RV_fit_pl[m] = K_opt[m] * (np.cos(w_opt[m] + f_opt) + e_opt[m] * np.cos(w_opt[m]))

    assert not None in p_opt and not None in e_opt and not None in tp_opt and not None in w_opt \
           and not None in K_opt and not None in msini_fit and not None in params_out and not None in RV_fit_pl, \
        "None-type object still in at least one of the parameter arrays after fitting."

    RV_fit = np.sum(RV_fit_pl, axis=0)  # total fitted RV

    if "boot" == flag:
        return params_out
    else:
        star.planets_fit = [None] * n_pls
        for i in range(0, n_pls):
            p = Planet(star, msini_fit[i], p_opt[i], e_opt[i], 90., w_opt[i], tp_opt[i], isReal=False)
            star.planets_fit[i] = p
        star.nPlanets_fit = n_pls
        star.RV_fit = RV_fit
        # star.params_out = params_out
        # star.params_out_print = np.array([p_opt,e_opt,tp_opt,np.array(w_opt)*180./pi,K_opt,msini_fit])
        star.params_LMfit = params


if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt
    print('Test output')

    """
    Based on HD 17156, Fig 3 of Fischer et al. 2007.
    Putting in the parameters from Table 3 gives the right curve, except
    it's like it's going backwards in phase. Is this a mistake with the code,
    or some planet vs. star effect? I think it's a code problem.
    It does agree with the plot in the paper when I flip all the eccentric anomalies.
    """
    P_test = 21.22
    tp_test = 13738.529
    e_test = 0.67
    w_test = 121. * (np.pi/180.)  # convert to radians
    K_test = 275.
    v0_test = 0.

    t_test = np.arange(13746, 13746+P_test, .001)
    start_time = time.time()
    rv_test = get_RV(P=P_test, K=K_test, e=e_test, tp=tp_test, w=w_test, v0=v0_test, t=t_test)
    print("--- %s seconds ---" % (time.time() - start_time))
    plt.figure()
    plt.plot(t_test, rv_test)
    plt.xlabel('JD')
    plt.ylabel('RV (m/s)')
    plt.title('Simulating HD 17156 from Fischer et al. 2007, fig. 3')
    # plt.show()

    ### ALTERNATIVELY; this method matches with the paper. And it runs about 20x faster.
    start_time = time.time()
    nu_test = calcTrueAnomaly(P_test, tp_test, e_test, t_test)
    rv_test_RVOS = K_test * (np.cos(nu_test + w_test) + e_test * np.cos(w_test)) + v0_test
    print("--- %s seconds ---" % (time.time() - start_time))

    # plt.figure()
    plt.plot(t_test, rv_test_RVOS)
    plt.show()
