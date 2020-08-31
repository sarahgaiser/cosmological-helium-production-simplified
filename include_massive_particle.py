import scipy.integrate as integrate
import numpy as np
import matplotlib.pyplot as plt

import helper_functions as helper
import neutron_baryon_ratio as nbratio
import variations_neutron_baryon_ratio as variations
import calculate_capture_time as capturetime


class UnknownParticleX:
    def __init__(self, m_x, tau_x, g_x):
        self.m = m_x  # mass of particle in MeV
        self.tau = tau_x  # mean life of particle in s
        self.g = g_x  # spin degrees of freedom of particle


def factor_f(unknownX, N_eff):
    return (30 * unknownX.m * unknownX.g/(N_eff * np.pi**2)
            * np.sqrt(0.47 * unknownX.m/(2 * np.pi))**3)


def temp_to_time_including_X(temp, unknownX, N_eff):
    prefactor = (np.sqrt(45 * helper.PhysicalConstants().M_Pl**2
                         * 1/(4 * np.pi**3 * N_eff))
                 * (11/4)**(2/3))

    f = factor_f(unknownX, N_eff)
    m_x = unknownX.m

    def integrand(T):
        return 1/(T * np.sqrt(f * np.sqrt(T)**3 * np.exp(-m_x/(0.47*T))
                              + T**4))

    time_in_per_MeV = prefactor * integrate.quad(integrand, temp, np.inf)[0]
    return helper.UnitConversions().per_MeV_in_seconds(time_in_per_MeV)


def integration_constant_t0(temp, unknownX, N, N_eff):
    def rho(T, n):
        prefactor = n * np.pi**2/30
        f = factor_f(unknownX, n)

        return prefactor * (f * np.sqrt(T)**3 * np.exp(-unknownX.m/(0.47*T))
                            + T**4)

    def integrand(T):
        return 1/T * (rho(T, N) - rho(T, N_eff))/np.sqrt(rho(T, N_eff))**3

    prefactor = (-0.5 * np.sqrt(3/(8 * np.pi)) * (11/4)**(2/3)
                 * helper.PhysicalConstants().M_Pl)

    time = prefactor * integrate.quad(integrand, temp, np.inf)[0]

    return helper.UnitConversions().per_MeV_in_seconds(time)


def modified_I(lower_bound, upper_bound, unknownX, N):
    def integrand(y, unknownX, N):
        delta_m = helper.PhysicalConstants().delta_m
        f = factor_f(unknownX, N)
        m_x = unknownX.m

        return ((1 + np.exp(-y)) * (12/y**5 + 6/y**4 + 1/y**3)
                * 1/np.sqrt(f * 1/np.sqrt(delta_m)**5 * np.sqrt(y)
                            * np.exp(-m_x/delta_m * y/0.47) + 1/y**2))

    b = nbratio.NeutronBaryonRatio().b
    integral = integrate.quad(integrand, lower_bound, upper_bound,
                              args=(unknownX, N))[0]

    return np.exp(-b * integral)


def modified_X(y, unknownX, N):
    def integrand(x, y, unknownX, N):
        return (np.exp(x) * nbratio.NeutronBaryonRatio().X_eq(x)**2
                * modified_I(x, y, unknownX, N))

    return (nbratio.NeutronBaryonRatio().X_eq(y)
            + integrate.quad(integrand, 0.001, y,
                             args=(y, unknownX, N))[0])


def temp_to_time_nx(temp, nbr_density_ratio, N_eff=13):
    a = 30 * 1.21/(N_eff * np.pi**4) * 11/2 * nbr_density_ratio
    prefactor = (np.sqrt(45/(4 * np.pi**3 * N_eff))
                 * helper.PhysicalConstants().M_Pl)

    def integrand(T):
        return 1/(T * np.sqrt(a * T**3 + T**4))

    time = prefactor * integrate.quad(integrand, temp, np.inf)[0]
    return helper.UnitConversions().per_MeV_in_seconds(time * (11/4)**(2/3))


def neutron_abundance_nx(y_eval, nbr_density_ratio, N=43/4):
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


def helium_abundance_nx(nbr_density_ratio):
    time = temp_to_time_nx(0.086, nbr_density_ratio)
    n_abundance = neutron_abundance_nx(100, nbr_density_ratio)

    return 2 * n_abundance * np.exp(-time/helper.PhysicalConstants().tau_n)


# helium_file = open("HeMassFractionUsingnx.txt", 'a')

# nx_values = np.arange(100, 1000, 10)
# for nx in nx_values:
#     he_abundance = helium_abundance_nx(nx)
#     entry = str(nx) + " " + str(he_abundance) + "\n"
#     helium_file.write(entry)

# helium_file.close()
