import neutron_baryon_ratio as nbratio
import variations_neutron_baryon_ratio as variations
import calculate_capture_time as capturetime
import helper_functions as helper

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib

matplotlib.rcParams['text.usetex'] = True


def lambda_np(y, tau_n):
    return 255/tau_n * (12/y**5 + 6/y**4 + 1/y**3)


def total_reaction_rate(y, tau_n):
    return (1 + np.exp(-y)) * lambda_np(y, tau_n)


def I_integrand(y, nbr_density_ratio):
    delta_m = helper.PhysicalConstants().delta_m
    a = 60 * 1.21/(43/4 * delta_m * np.pi**4) * nbr_density_ratio
    return ((1 + np.exp(-y)) * (12/y**4 + 6/y**3 + 1/y**2)
            * 1/np.sqrt(a * y + 1))


def I_yy(lower_bound, upper_bound, nbr_density_ratio):
    b = nbratio.NeutronBaryonRatio(43/4).b
    return np.exp(-b * integrate.quad(I_integrand, lower_bound,
                                      upper_bound,
                                      args=(nbr_density_ratio))[0])


def neutron_to_baryon_ratio(y):
    X_no_decay = nbratio.NeutronBaryonRatio().X_no_n_decay(y)
    X_eq = nbratio.NeutronBaryonRatio().X_eq(y)
    freeze_out_correction = X_no_decay - X_eq

    return X_eq, freeze_out_correction, X_no_decay


def radiative_energy_density(T):
    if T > 0.511/2:
        return 43/4 * np.pi**2/30 * T**4
    else:
        return 13 * np.pi**2/30 * T**4


def time_from_temp(T):
    return helper.UnitConversions().temp_to_time(T, 13)


def saha_equation(z):
    return capturetime.CalculateCaptureTime(0.15).G_np(z)


def capture_condition(z):
    capture_time = capturetime.CalculateCaptureTime(0.15)
    return capture_time.neutron_capture_condition(z) - 1
