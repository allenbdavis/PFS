import numpy as np
from constants import *
from bodies import System, Star, Planet
from scipy.interpolate import interp1d
import fitting as fit

def calcTrueAnomaly(P, tp, e, t):
    """
    Calculates the true anomaly from the time and orbital elements.

    P:
    tp:
    e:
    t:
    :return:
    """
    phase = (t-tp)/P
    M = 2.*np.pi*(phase - np.floor(phase))  # mean anomaly
    E1 = calcKepler(M, np.array([e]))

    n1 = 1. + e
    n2 = 1. - e

    #True Anomaly:
    true_anomaly = 2.*np.arctan(np.sqrt(n1/n2)*np.tan(E1/2.))
    return true_anomaly


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


def get_RV(P, K, e, tp, w, v0, t, method='rvos'):
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
    method: str
        How will true anomaly be determined? Options are 'rvos' and 'new'

    Returns
    -------
    RV: 1D numpy array
        List of radial velocities (m/s). Positive velocities correspond to star moving away from observer.
    """
    if type(t) is not np.ndarray:
        t = np.array(t)

    if method == 'new':
        nu = get_true_anomaly(P, tp, e, t)
    elif method == 'rvos':
        nu = calcTrueAnomaly(P, tp, e, t)
    else:
        assert False

    RV = K * (np.cos(nu + w) + e * np.cos(w)) + v0
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
    v0_test = 20.

    t_test = np.arange(13746, 13746+P_test, .001)

    # start_time = time.time()
    # rv_test = get_RV(P=P_test, K=K_test, e=e_test, tp=tp_test, w=w_test, v0=v0_test, t=t_test, method='new')
    # print("--- %s seconds ---" % (time.time() - start_time))
    # plt.figure()
    # plt.plot(t_test, rv_test)
    # # plt.show()

    ### ALTERNATIVELY; this method matches with the paper. And it runs 10-20x faster.
    start_time = time.time()
    nu_test = calcTrueAnomaly(P_test, tp_test, e_test, t_test)
    rv_test_RVOS = get_RV(P=P_test, K=K_test, e=e_test, tp=tp_test, w=w_test, v0=v0_test, t=t_test, method='rvos')
    print("--- %s seconds ---" % (time.time() - start_time))

    plt.figure()
    plt.plot(t_test, rv_test_RVOS)
    plt.title('Simulating HD 17156 from Fischer et al. 2007, fig. 3')
    plt.xlabel('JD')
    plt.ylabel('RV (m/s)')

    # Fitting planet
    results = fit.fit_Keplerian(t_test, rv_test_RVOS, P_test, report=True)
    P_fit = results['P']
    K_fit = results['K']
    e_fit = results['e']
    tp_fit = results['tp']
    w_fit = results['w']
    v0_fit = results['v0']
    rv_fit = get_RV(P=P_fit, K=K_fit, e=e_fit, tp=tp_fit, w=w_fit, v0=v0_fit, t=t_test)

    plt.plot(t_test, rv_fit, '--')
    plt.show()

