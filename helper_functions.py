import numpy as np


class PhysicalConstants:
    """A class containing all needed physical constants.

    The values are taken from
    1) https://pdg.lbl.gov/2020/reviews/rpp2020-rev-phys-constants.pdf
    2) https://pdg.lbl.gov/2013/reviews/rpp2013-rev-astrophysical-constants.pdf
    3) https://physics.nist.gov/cgi-bin/cuu/Value?eqmdc2mev
    4) Rev. Mod. Phys., Vol 61, No. 1, January 1989 (the paper)
    """

    # masses
    M_Pl = 1.221e22  # in MeV, (2)
    m_proton = 938.27208  # in MeV, (1)
    m_neutron = 939.56541  # in MeV, (1)
    m_deuteron = 1875.61294  # in MeV, (3)

    # mass differences
    delta_m = m_neutron - m_proton
    epsilon_D = m_proton + m_neutron - m_deuteron

    # neutron mean life
    tau_n = 879.4  # in s, (1)
    tau_n_old = 896  # in s, (4)

    # other constants
    fine_alpha = 7.29735e-3  # fine structure constant alpha, (1)
    eta_zero = 5e-10  # baryon-to-photon ratio, (4)


class UnitConversions:
    """Class including several unit conversions.

    This class includes functions to convert natural units to SI units
    and vice versa, as well as functions to translate temperature
    to time, and similar.
    """

    def seconds_in_per_MeV(self, time_in_s):
        """Conversion of time from SI to natural units.

        Args:
            time_in_s: time in seconds
        Returns:
            time in units of 1/MeV
        """
        return 0.152e22*time_in_s

    def per_MeV_in_seconds(self, time_in_per_MeV):
        """Conversion of time from natural to SI units.

        Args:
            time_in_per_MeV: time in units of 1/MeV
        Returns:
            time in units of s
        """
        return 6.58e-22*time_in_per_MeV

    def temp_to_time(self, temp, N_eff):
        """Conversion from temperature in MeV to time in 1/MeV.

        Args:
            temp: temperature in MeV
            N_eff: number of effective relativistic degrees of freedom
        Returns:
            age of universe in 1/MeV at given temperature
        """
        return (np.sqrt(45/(16 * np.pi**3 * N_eff)) * (11/4)**(2/3)
                * PhysicalConstants().M_Pl/temp**2
                + self.seconds_in_per_MeV(2))


def calculate_b(phys_const, a, N):
    """Calculation of parameter b in K(y).

    b in a number that controls lepton-nucleon interaction rate.
    Args:
        phys_const: member of PhysicalConstants class
        a: number that parameterizes neutron decay rate
        N: number of relativistic degrees of freedom
    Returns:
        value of b
    """
    unitconv = UnitConversions()
    mean_life_in_MeV = unitconv.seconds_in_per_MeV(phys_const.tau_n)
    return (a * phys_const.M_Pl * np.sqrt(45./(4*np.pi**3*N))
            * 1/(phys_const.delta_m**2 * mean_life_in_MeV))


def K(y, b):
    """Function that encodes lepton-nucleon interaction rates."""
    return b*(4/y**3 + 3/y**2 + 1/y + (4/y**3 + 1/y**2) * np.exp(-y))


def G(y, b):
    """Helper function needed in variation of X with chem potential."""
    return (b/2 * (4/y**3 + 3/y**2 + 1/y - (4/y**3 + 1/y**2) * np.exp(-y))
            - 2/(1+np.exp(y)))


def calculate_R_DD_constant_factors():
    """
    Calculate const factors in equation for R_DD.

    R_DD is of the following form:
    R_DD = R_0 * eta/eta_zero * 1/sqrt(N_eff)
               * 1/z**(4/3) * exp(-R_1 * z**(1/3))

    Returns:
        R_0, R_1: see above
    """
    constants = PhysicalConstants()
    C1 = (11/4)**(2/3) * 1.202 * constants.M_Pl
    C2 = (0.87 * 4 * np.pi * constants.fine_alpha
          * 1/np.sqrt(3 * constants.epsilon_D * constants.m_proton))
    C3 = (2 * np.pi * constants.fine_alpha
          * constants.epsilon_D/constants.m_proton)**(1/3)
    R_1 = 3 * np.pi * constants.fine_alpha * (constants.m_proton/(2 * np.pi
                                              * constants.fine_alpha
                                              * constants.epsilon_D))**(1/3)

    R_0 = constants.eta_zero * np.sqrt(45/np.pi**7) * C1 * C2 * C3

    return R_0, R_1
