import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import observer
import orbit


def plot_fit(t_obs, rv_obs, errors=None, system=None, param_fit=None, param_boots=None):
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

    Returns
    -------
    Generates plots
    """

    plt.figure(figsize=(8, 4))
    nrow = 2
    ncol = 1
    gs = gridspec.GridSpec(nrow, ncol, height_ratios=[3, 1], wspace=0.0, hspace=0.0)

    assert not(param_fit is None and param_boots is not None), "Must provide param_fit is param_boots is given."

    ax = plt.subplot((gs[0, 0]))
    if errors is not None:
        ax.errorbar(t_obs, rv_obs, yerr=errors, color='C0', marker='o', linestyle='', label='observed')
    else:
        ax.errorbar(t_obs, rv_obs, color='C0', marker='o', linestyle='', label='observed')

    t_full = np.arange(t_obs[0], t_obs[-1], 0.05)

    if system is not None:
        rv_true = observer.get_system_RVs(system, t_full)
        ax.plot(t_full, rv_true, '-', color='C1', label='true')

    if param_fit is not None:
        rv_fit = orbit.get_RV(P=param_fit['P'], K=param_fit['K'], e=param_fit['e'], tp=param_fit['tp'],
                              w=param_fit['w'], v0=param_fit['v0'], t=t_full)
        ax.plot(t_full, rv_fit, '--', color='C2', label='fit')

    if param_boots is not None:
        for i, param_boot in enumerate(param_boots):
            rv_boot = orbit.get_RV(P=param_boot['P'], K=param_boot['K'], e=param_boot['e'], tp=param_boot['tp'],
                                   w=param_boot['w'], v0=param_boot['v0'], t=t_full)
            if i == 0:
                ax.plot(t_full, rv_boot, '-', color='k', linewidth=1, alpha=0.05, label='bootstrap')
            else:
                ax.plot(t_full, rv_boot, '-', color='k', linewidth=1, alpha=0.05)
    ax.set_ylabel('RV (m/s)')
    ax.legend()
    ax.grid(linestyle='--', alpha=0.3)

    # Plot residuals
    ax = plt.subplot((gs[1, 0]))
    rv_fit_obs_samples = orbit.get_RV(P=param_fit['P'], K=param_fit['K'], e=param_fit['e'], tp=param_fit['tp'],
                                      w=param_fit['w'], v0=param_fit['v0'], t=t_obs)
    residuals = rv_obs - rv_fit_obs_samples
    ax.errorbar(t_obs, residuals, yerr=errors, marker='o', color='k', linestyle='', label='residuals')

    ax.set_xlabel('JD')
    ax.set_ylabel('RV (m/s)')
    ax.legend()
    ax.grid(linestyle='--', alpha=0.3)

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
    gs = gridspec.GridSpec(nrow, ncol)#, height_ratios=[1, 1], width_ratios=[1, 1, 1])#,
                           #wspace=0.5, hspace=0.5)

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
