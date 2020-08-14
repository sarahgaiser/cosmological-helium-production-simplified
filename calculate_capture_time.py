from scipy.optimize import fsolve
import numpy as np
import helper_functions as helper


class CalculateCaptureTime:
    """Class including all the functions needed to calculate t_c.

    t_c is the time of neutron capture in helium. There are several
    steps needed to derive this value which are summarized in this class.

    In __init__, the neutron-to-baryon ratio without neutron decay,
    effective number of relativistic degrees of freedom after
    electron-positron-annihilation, N_eff, and the
    baryon-to-photon ratio, eta, are set. It also initalizes an instance of
    the PhysicalConstants and UnitConversions classes each.

    Args:
        X_no_decay: neutron-to-baryon ratio without neutron decay
        N_eff: effective number of relativistic degrees of freedom
               after e-e+ annihilation
        eta: baryon-to-photon ratio
    """
    def __init__(self, X_no_decay, N_eff=13, eta=5e-10):
        self.X_no_decay = X_no_decay
        self.N_eff = N_eff
        self.eta = eta

        self.phys_const = helper.PhysicalConstants()
        self.conversions = helper.UnitConversions()
        self.R_0, self.R_1 = helper.calculate_R_DD_constant_factors()

    def rate_DD(self, z):
        """Reaction rate of n capture to He.

        R_DD includes all relevant reaction rates of
        n+p->D+photon, D+D->T+p, D+T->He+n.

        Args:
            z: scaled temperature variable, z=epsilon_D/T
        Returns:
            reaction rate for given z
        """
        return (self.R_0 * 1/np.sqrt(self.N_eff)
                * self.eta/self.phys_const.eta_zero
                * 1/z**(4/3) * np.exp(- self.R_1 * z**(1/3)))

    def G_np(self, z):
        """Part of Saha equation.

        Neutron, proton and deuteron number densities follow the Saha equation
        of the form:
        X_n * X_p / X_D = G_np

        Args:
            z: scaled temperature variable, z=epsilon_D/T
        Returns:
            X_n * X_p / X_D at given z
        """
        prefactor = (np.sqrt(np.pi)/(12 * 1.202) * 1/self.eta
                     * (self.phys_const.m_proton
                        / self.phys_const.epsilon_D)**(3/2))
        return prefactor * z**(3/2) * np.exp(-z)

    def X_deuteron_1(self, z):
        """First approximation of deuteron population.

        X_deuteron_1 = 1/G_np * X_p * X_n
        with X_n and X_p neutron(proton)-to-baryon density without n decay,
        and X_n + X_p = 1

        Args:
            z: scaled temperature variable, z=epsilon_D/T
        Returns:
            first approximation of deuteron-to-baryon ratio
        """
        return (1/self.G_np(z) * self.X_no_decay
                * (1 - self.X_no_decay))

    def neutron_capture_condition(self, z):
        """Condition on z at onset of neutron capture.

        The root of the capture condition gives the value of z that
        correponds to the onset of neutron capture.

        Args:
            z: scaled temperature variable, z=epsilon_D/T
        Returns:
            result of capture condition at given value of z
        """
        return 2 * self.X_deuteron_1(z) * self.rate_DD(z) - 1

    def calculate_capture_temp(self):
        """Calculates the capture temperature using the capture condition.

        The root of neutron_capture_condition is calculated and the
        corresponding z value is converted to temperature

        Returns:
            temperature of universe at neutron capture in He
        """
        return (self.phys_const.epsilon_D
                * 1/fsolve(self.neutron_capture_condition, 26)[0])

    def calculate_capture_time(self):
        """Calculates capture time of neutron.

        The capture time corresponds to the age of the universe at the capture
        of neutrons in He. This function calls calculate_capture_temp and
        transforms the capture temperature to time.

        Returns:
            the capture time t_c of neutron in helium in seconds
        """
        capture_temp = self.calculate_capture_temp()
        capture_time_MeV = self.conversions.temp_to_time(capture_temp,
                                                         self.N_eff)
        return self.conversions.per_MeV_in_seconds(capture_time_MeV)
