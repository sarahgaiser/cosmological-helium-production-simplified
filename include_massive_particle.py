import scipy.integrate as integrate
import numpy as np
import matplotlib.pyplot as plt

import helper_functions as helper
import neutron_baryon_ratio as nbratio
import variations_neutron_baryon_ratio as variations
import calculate_capture_time as capturetime


def temp_to_time_nx(temp, nbr_density_ratio, N_eff=13):
    """Modified time-temperature relation including massive particle.

    Args:
        temp: temperature in MeV
        nbr_density_ratio: (n_x*m_x)/n_gamma
        N_eff: number of relativistic degrees of freedom
               after e+e- annihilation
    Returns:
        time in s
    """
    a = 30 * 1.21/(N_eff * np.pi**4) * 11/2 * nbr_density_ratio
    prefactor = (np.sqrt(45/(4 * np.pi**3 * N_eff))
                 * helper.PhysicalConstants().M_Pl)

    def integrand(T):
        return 1/(T * np.sqrt(a * T**3 + T**4))

    time = prefactor * integrate.quad(integrand, temp, np.inf)[0]
    return helper.UnitConversions().per_MeV_in_seconds(time * (11/4)**(2/3))


def neutron_abundance_nx(y_eval, nbr_density_ratio, N=43/4):
    """Modified neutron abundance including massive particle.

    Args:
        y_eval: value of y at which neutron abundance is calculated
        nbr_density_ratio: (n_x*m_x)/n_gamma
        N: number of relativistic degrees of freedom before e+e- annihilation
    Returns:
        neutron abundance at y_eval
    """
    b = nbratio.NeutronBaryonRatio(N).b

    def I_yy(lower_bound, upper_bound):
        delta_m = helper.PhysicalConstants().delta_m
        a = 60 * 1.21/(N * delta_m * np.pi**4) * nbr_density_ratio

        def integrand(y):
            return ((1 + np.exp(-y)) * (12/y**4 + 6/y**3 + 1/y**2)
                    * 1/np.sqrt(a * y + 1))

        return np.exp(-b * integrate.quad(integrand, lower_bound,
                                          upper_bound)[0])

    def integrand(y, upper_bound):
        return (np.exp(y) * nbratio.NeutronBaryonRatio().X_eq(y)**2
                * I_yy(y, upper_bound))

    return (nbratio.NeutronBaryonRatio().X_eq(y_eval)
            + integrate.quad(integrand, 0, y_eval, args=(y_eval))[0])


def helium_abundance_nx(nbr_density_ratio, T_c=0.084, y_eval=100):
    """Modified helium abundance including massive particle.

    Args:
        nbr_density_ratio: (n_x*m_x)/n_gamma
        T_c: capture time of neutrons
        y_eval: value of y at which asymptotic value of X(y) is reached
    Returns:
        helium abundance at given value of (n_x*m_x)/n_gamma
    """
    time = temp_to_time_nx(T_c, nbr_density_ratio)
    n_abundance = neutron_abundance_nx(y_eval, nbr_density_ratio)

    return 2 * n_abundance * np.exp(-time/helper.PhysicalConstants().tau_n)
