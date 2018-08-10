import numpy as np
from bodies import System, Star, Planet
import orbit
import fitting as fit
import plots


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
    the_star = Star(mass=1.2, v0=0., t0=0., name="Allen's star")

    # Create Planets
    planet_b = Planet(period=60.52, ecc=0.1, mass=10., random=True)

    # Create System
    the_system = System(the_star, [planet_b])

    # Create time-series
    t, rv, errors = observe_system(system=the_system, run_length=180, weather=0.5, noise=.2)

    # Ideal orbit
    times_ideal = np.arange(t[0], t[-1]+0.01, 0.01)
    RVs_ideal = get_system_RVs(the_system, times_ideal)

    # Fit the planet
    params_fit, residuals = fit.fit_Keplerian(t, rv, report=True,
                                              P=[60, 58, 62],
                                              e=[0.2, 0.01, 0.4],
                                              tp=None,
                                              w=[0, -np.inf, np.inf],
                                              K=[1, 0, 200],
                                              v0=[0., -200, 200])
    P_fit = params_fit['P']
    e_fit = params_fit['e']
    tp_fit = params_fit['tp']
    w_fit = params_fit['w']
    K_fit = params_fit['K']
    v0_fit = params_fit['v0']
    params_boot = fit.bootstrap_MC(t, rv, residuals, n_iter=200, replacement=False,
                                   P=[P_fit, P_fit-1, P_fit+1],
                                   e=[e_fit, 0.01, 0.9],
                                   tp=[tp_fit, -np.inf, np.inf],
                                   w=[w_fit, -np.inf, np.inf],
                                   K=[K_fit, 0, K_fit*3],
                                   v0=[v0_fit, v0_fit-30, v0_fit+30])

    plots.plot_fit(t, rv, errors=errors, system=the_system, param_fit=params_fit, param_boots=params_boot,
                   phase_fold=True)
    plots.plot_uncertainties(params_fit, params_boot, star=the_star, planet=planet_b)
