import numpy as np
from bodies import *
import orbit


def observe_system(system, run_length, weather=0., noise=0.):
    """
    Generate time-series of mock RV observations of a system.

    Parameters
    ----------
    system: System
        System object
    run_length: int
        Number of days over which to observe (before accounting for weather)
    weather: float
        What fraction of nights are missed due to poor weather
    noise: float
        Scale of white-noise (m/s)

    Returns
    -------
    times: 1D array
        List of JDs when observations occurred successfully
    RVs: 1D array
        List of corresponding radial velocities for the system
    estimated_errors: 1D array
        List of 1-sigma uncertainties for each data point
    """

    times = []
    i = 0
    while i < run_length:
        if np.random.random() < 1. - weather:
            times.append(i)
        i += 1
    times = np.array(times)
    estimated_errors = np.array([noise for _ in range(len(times))])
    actual_errors = np.random.normal(loc=0, scale=noise, size=len(times))
    RVs = get_system_RVs(system, times) + actual_errors
    return times, RVs, estimated_errors


def get_system_RVs(system, times):
    """
    Return RVs for the system at the requested times.

    Parameters
    ----------
    system: System
    times: list or ndarray
        JDs

    Returns
    -------
    RVs: 1D array
    """

    star = system.star
    RVs = np.array([star.v0 for _ in times])
    for pl in system.planets:
        RVs = RVs + orbit.get_RV(P=pl.period, K=pl.K, e=pl.ecc, tp=pl.tp, w=pl.w, v0=star.v0, t=times)
    return RVs

if __name__ == '__main__':
    """
    Test observations.
    Eventually should take input text file with system parameters.
    """

    # Create star
    star = Star(mass=1.2, v0=1., t0=0., name="Allen's star")

    # Create Planets
    planet_b = Planet(period=22.3, ecc=0.05, K=17.3)

    # Create System
    system = System(star, [planet_b])

    # Create time-series
    times, RVs, errors = observe_system(system=system, run_length=180, weather=0.5, noise=3.)

    # Ideal orbit
    times_ideal = np.arange(times[0], times[-1]+0.01, 0.01)
    RVs_ideal = get_system_RVs(system, times_ideal)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.errorbar(times, RVs, yerr=errors, marker='o', linestyle='', label='observed')
    plt.plot(times_ideal, RVs_ideal, marker='', linestyle='-', label='true')
    plt.title('')
    plt.xlabel('JD')
    plt.ylabel('RV (m/s)')

    # Fit the planet
    results = orbit.fit_Keplerian(times, RVs, report=True,
                                  P=[22, 21, 23],
                                  e=[0.2, 0.01, 0.9],
                                  tp=None,
                                  w=[0, -np.inf, np.inf],
                                  K=[1, 0, 200],
                                  v0=[0., -200, 200])
    P_fit = results['P']
    e_fit = results['e']
    tp_fit = results['tp']
    w_fit = results['w']
    K_fit = results['K']
    v0_fit = results['v0']
    many_times = np.arange(times[0], times[-1], 0.05)
    rv_fit = orbit.get_RV(P=P_fit, K=K_fit, e=e_fit, tp=tp_fit, w=w_fit, v0=v0_fit, t=many_times)
    plt.plot(many_times, rv_fit, '--', label='fit')
    plt.legend()

    results_boot = orbit.bootstrap_MC(times, RVs, errors, n_iter=100,
                                      P=[P_fit, P_fit-1, P_fit+1],
                                      e=[e_fit, 0.01, 0.9],
                                      tp=[tp_fit, -np.inf, np.inf],
                                      w=[w_fit, -np.inf, np.inf],
                                      K=[K_fit, 0, K_fit*3],
                                      v0=[v0_fit, v0_fit-30, v0_fit+30])

    for result_i in results_boot:
        rv_fit_i = orbit.get_RV(P=result_i['P'], K=result_i['K'], e=result_i['e'], tp=result_i['tp'],
                                w=result_i['w'], v0=result_i['v0'], t=many_times)
        plt.plot(many_times, rv_fit_i, '-', color='k', linewidth=1, alpha=0.05)

    plt.show()

    # MOVE TO NEW .py
    def gaus(x, A, mu, sig, C):
        return A * np.exp(-(x - mu) ** 2 / (2. * sig * sig)) + C

    # MOVE TO NEW .py
    def regularize(vals, center, i=0):
        center = np.mod(center, 2.*np.pi)  # [0,2pi)
        vals = np.mod(vals, 2.*np.pi)  # [0,2pi)
        new_vals = []
        for v in vals:
            if v > center + np.pi:
                new_vals.append(v - 2.*np.pi)
            elif v < center - np.pi:
                new_vals.append(v + 2.*np.pi)
            else:
                new_vals.append(v)

        if i > 4:
            return new_vals
        else:
            return regularize(new_vals, np.mean(new_vals), i=i+1)

    # uncertainty plots
    dic = {'P': 'period', 'e': 'ecc', 'w': 'w', 'tp': 'tp', 'K': 'K', 'v0': 'v0'}
    for par in results.keys():
        plt.figure()
        values = [result_i[par] for result_i in results_boot]
        if par == 'w':
            values = regularize(values, np.mean(values))
        plt.hist(values)
        if not par == 'v0':
            true_val = getattr(planet_b, dic[par])
        else:
            true_val = getattr(star, dic[par])
        plt.axvline(true_val, color='C1')
        plt.title('input: {par} = {val:.2f}\n fit: {par} = {best:.2f} +/- {unc:.2f}'.format(par=par, val=true_val,
                                                                                best=results[par].value, unc=np.std(values)))
        plt.ylabel('frequency')
        plt.xlabel(par)

    plt.show()