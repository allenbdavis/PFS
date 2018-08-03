import numpy as np
from bodies import *
import orbit


def observe_system(system, run_length, weather=0.):
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

    Returns
    -------
    times: 1D array
        List of JDs when observations occurred successfully
    RVs: 1D array
        List of corresponding radial velocities for the system
    """

    times = []
    i = 0
    while i < run_length:
        if np.random.random() < 1. - weather:
            times.append(i)
        i += 1
    times = np.array(times)

    RVs = get_system_RVs(system, times)

    return times, RVs


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
    star = Star(mass=1.2, v0=47., name="Allen's star")

    # Create Planets
    planet_b = Planet(period=22.3, ecc=0.2, K=17.3)

    # Create System
    system = System(star, [planet_b])

    # Create time-series
    times, RVs = observe_system(system=system, run_length=180, weather=0.5)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(times, RVs, 'o-')
    plt.xlabel('JD')
    plt.ylabel('RV (m/s)')
    plt.show()
