import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import observer
import orbit


def plot_fit(t_obs, rv_obs, errors=None, system=None, param_fit=None, param_boots=None, phase_fold=False):
    """
    Plot the observed RVs, true RVs, fitted RVs, and residuals.

    Parameters
    ----------
    t_obs: 1D array
        List of observation epochs (JD)
    rv_obs: 1D array
        List of RV observations
    errors: 1D array or None
        List of 1 sigma errors for the RVs
    system: System
        System object
    param_fit: Parameters
        Fitted parameters for the system
    param_boots: list of Parameters
        List of fitted parameters for bootstrapped uncertainties
    phase_fold: bool
        Create phase folded-plot

    Returns
    -------
    Generates plots
    """

    def get_phase_fold(t, p):
        return (t % p) / p

    phase = np.linspace(0, 1., 200)
    phase_obs = get_phase_fold(t_obs, system.planets[0].period)
    if phase_fold:
        t_full = np.linspace(0, system.planets[0].period, 200)  # evenly & highly sampled days for one period
    else:
        t_full = np.linspace(t_obs[0], t_obs[-1], 200)  # evenly & highly sampled JDs

    fig = plt.figure(figsize=(8, 4))
    nrow = 2
    ncol = 1
    gs = gridspec.GridSpec(nrow, ncol, height_ratios=[3, 1], wspace=0.0, hspace=0.0)

    assert not(param_fit is None and param_boots is not None), "Must provide param_fit is param_boots is given."

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.errorbar((t_full if not phase_fold else phase_obs), rv_obs, yerr=errors, color='C0', marker='o',
                 linestyle='', label='observed', zorder=3)

    if system is not None:
        rv_true = observer.get_system_RVs(system, t_full)
        ax1.plot((t_full if not phase_fold else phase), rv_true, '-', color='C1', label='true', zorder=1)

    if param_fit is not None:
        rv_fit = orbit.get_RV(P=param_fit['P'], K=param_fit['K'], e=param_fit['e'], tp=param_fit['tp'],
                              w=param_fit['w'], v0=param_fit['v0'], t=t_full)
        ax1.plot((t_full if not phase_fold else phase), rv_fit, '--', color='C2', label='fit', zorder=2)

    if param_boots is not None:
        for i, param_boot in enumerate(param_boots):
            rv_boot = orbit.get_RV(P=param_boot['P'], K=param_boot['K'], e=param_boot['e'], tp=param_boot['tp'],
                                   w=param_boot['w'], v0=param_boot['v0'], t=t_full)
            ax1.plot((t_full if not phase_fold else phase), rv_boot, '-', color='k',
                     linewidth=1, alpha=0.05, label=('bootstrap' if i == 0 else ''), zorder=0)

    ax1.set_ylabel('RV (m/s)')
    ax1.legend()
    ax1.grid(linestyle='--', alpha=0.3)

    # Plot residuals
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    rv_fit_obs_samples = orbit.get_RV(P=param_fit['P'], K=param_fit['K'], e=param_fit['e'], tp=param_fit['tp'],
                                      w=param_fit['w'], v0=param_fit['v0'], t=t_obs)
    residuals = rv_obs - rv_fit_obs_samples
    ax2.errorbar((t_obs if not phase_fold else phase_obs), residuals, yerr=errors, marker='o', color='k',
                 linestyle='', label='residuals', zorder=1)

    ax2.axhline(0, color='red', linestyle='--', zorder=0)
    ax2.set_xlabel('JD' if not phase_fold else 'Phase')
    ax2.set_ylabel('Residuals (m/s)')
    ax2.grid(linestyle='--', alpha=0.3)

    plt.show()


def plot_uncertainties(param_fit, param_boots, star=None, planet=None):
    """
    Plot bootstrapped parameters for all the fitted parameters. Currently only works for one planet.

    Parameters
    ----------
    param_fit: Parameters
        Fitted parameters for the system
    param_boots: list of Parameters
        List of fitted parameters for bootstrapped uncertainties
    star: Star
        Star object
    planet: Planet
        Planet object

    Returns
    -------
    Generates plot
    """
    plt.figure(figsize=(8, 4))
    nrow = 2
    ncol = 3
    gs = gridspec.GridSpec(nrow, ncol)

    def regularize(vals, center, i=0, imax=4):
        """
        Recenter angles to between 0 and 2pi, and then wrap them around to be closet to the mean value.
        Calls recursively.
        """
        center = np.mod(center, 2. * np.pi)  # [0,2pi)
        vals = np.mod(vals, 2. * np.pi)  # [0,2pi)
        new_vals = []
        for v in vals:
            if v > center + np.pi:
                new_vals.append(v - 2. * np.pi)
            elif v < center - np.pi:
                new_vals.append(v + 2. * np.pi)
            else:
                new_vals.append(v)
        if i > imax:
            return new_vals
        else:
            return regularize(new_vals, np.mean(new_vals), i=i + 1)

    dic = {'P': 'period', 'e': 'ecc', 'w': 'w', 'tp': 'tp', 'K': 'K', 'v0': 'v0'}
    count = 0
    for par in param_fit.keys():
        which_row = count // ncol
        which_col = count % ncol
        ax = plt.subplot((gs[which_row, which_col]))
        values = [result_i[par] for result_i in param_boots]
        if par == 'w':
            values = regularize(values, np.mean(values))
        ax.hist(values)
        if star is not None and planet is not None:
            if not par == 'v0':
                true_val = getattr(planet, dic[par])
            else:
                true_val = getattr(star, dic[par])
            ax.axvline(true_val, color='C1')
            ax.set_title('input: {par} = {val:.2f}\nfit: {par} = {best:.2f} +/- {unc:.2f}'.
                         format(par=par, val=true_val, best=param_fit[par].value, unc=np.std(values)))
        else:
            ax.set_title('fit: {par} = {best:.2f} +/- {unc:.2f}'.
                         format(par=par, best=param_fit[par].value, unc=np.std(values)))
        ax.set_ylabel('frequency')
        ax.set_xlabel(par)
        count += 1
    plt.show()
