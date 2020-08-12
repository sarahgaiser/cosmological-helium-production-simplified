import scipy.integrate as integrate
from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
import helper_functions as helper


class NeutronBaryonRatio:
    """Class containing important equations for neutron-to-baryon ratio.

    This class includes formulas for the neutron-to-baryon ratio X
    in equilibrium, X_eq; out of equilibrium without neutron decay,
    X_no_n_decay; and with neutron decay, X_with_n_decay.

    __init__ takes (and sets) the number of effective relativistic
    degrees of freedom N before electron-positron-annihilation,
    sets the parameter a=255 and b that are used to express the
    interaction rate of the neutron. In addition, it initalizes an instance
    of the PhysicalConstants class.

    Args:
        N: number of effective relativistic degrees of freedom N before
           e-e+ annihilation
    """
    def __init__(self, N=43./4):
        self.phys_const = helper.PhysicalConstants()

        self.N = N
        self.a = 255
        self.b = helper.calculate_b(self.phys_const, self.a, self.N)

    def X_eq(self, y):
        """Equilibrium neutron-to-baryon ratio.

        Args:
            y: dimensionless temperature variable, y=delta_m/T
        Returns:
            value of equil. neutron-to-baryon ratio at y
        """
        return 1/(1+np.exp(y))

    def X_no_n_decay(self, y=100):
        """Out of equilibrium neutron-to-baryon ratio without n decay.

        Args:
            y: dimensionless temp. at which to evaluate X,
               coincides with upper limit of integration of X_integrand;
               the default of y=100 corresponds to y->inifinty
        Returns:
            value of out of equil. neutron-to-baryon ratio for given y
        """
        def X_integrand(self, x, y):
            """Intregrand in calculation of out of equil. n-to-baryon ratio.

            Args:
                x: variable to be integrated over (dimensionless temp.)
                y: fixed value of dim. temperature,
                same as upper limit of integration
            Returns:
                value of integrand at given x and y
            """
            return self.X_eq(x)**2 * np.exp(x + helper.K(y, self.b)
                                            - helper.K(x, self.b))

        return self.X_eq(y) + integrate.quad(X_integrand, 0, y, args=(y))[0]

    def X_with_n_decay(self, calculate_t_c=True, t_c=0):
        """Out of equilibrium neutron-to-baryon ratio including n decay.

        This determines the neutron-to-baryon ratio at the time t_c of neutron
        capture in He, using the formula

        X_no_n_decay(infinity) * exp(-t_c/tau_n)

        where tau_n is the neutron lifetime, and X_no_decay is evaluated at its
        asymptotic value.

        Args:
            calculate_t_c: set to True if t_c is to be calculated
            t_c: capture time of neutron; if calculate_t_c is set to False,
                 this value will be used to calculate X
        Returns:
            value of neutron-to-baryon ratio at t_c
        """
        if calculate_t_c:
            calc_t_c = CalculateCaptureTime()
            t_c = calc_t_c.calculate_capture_time()
        return np.exp(-t_c/self.phys_const.tau_n) * self.X_no_n_decay()

    def He_mass_fraction(self, include_n_decay=True,
                         calculate_t_c=True, t_c=0):
        """Calculates primordial helium mass fraction.

        The primordial helium mass fraction is given by 2*X
        with X being the neutron-to-baryon-ratio.

        Args:
            include_n_decay: set to true if neutron decay is to be included
            calculate_t_c: set to True if neutron cature time t_c
                           is to be calculated
            t_c: capture time of neutron; if calculate_t_c is set to False,
                 this value will be used
        Returns:
            primordial He mass fraction with or without including the effects
            of neutron decay
        """
        if include_n_decay:
            return 2*self.X_with_n_decay(calculate_t_c, t_c)
        else:
            return 2*self.X_no_n_decay()


class VariationsOfNeutronBaryonRatio(NeutronBaryonRatio):
    def __init__(self, N_eff=13, N=43./4):
        self.N_eff = N_eff
        NeutronBaryonRatio.__init__(self, N)

    def variation_with_b_coefficient(self):
        def C_integrand(y):
            return (np.exp(y - helper.K(y, self.b)) * self.X_eq(y)**2
                    * helper.K(y, self.b))

        return 1./self.X_no_n_decay() * integrate.quad(C_integrand, 0, 100)[0]

    def variation_with_alpha(self):
        def X_1_integrand(y):
            return (np.exp(y - helper.K(y, self-b)) * self.X_eq(y)**2
                    * helper.G(y, self.b))

        return integrate.quad(X_1_integrand, 0, 100)[0]


class CalculateCaptureTime:
    """Class including all the functions needed to calculate t_c.

    t_c is the time of neutron capture in helium. There are several
    steps needed to derive this value which are summarized in this class.

    In __init__, the effective number of relativistic degrees of freedom
    after electron-positron-annihilation, N_eff,
    the baryon-to-photon ratio, eta, and the effective number of relativistic
    degrees of freedom before e-e+ annihilation, N, are set.
    It also initalizes an instance of the NeutronBaryonRatio, PhysicalConstants
    and UnitConversions classes each.

    Args:
        N_eff: effective number of relativistic degrees of freedom
               after e-e+ annihilation
        eta: baryon-to-photon ratio
        N: effective number of relativistic degrees of freedom
           before e-e+ annihilation
    """
    def __init__(self, N_eff=13, eta=5e-10, N=43./4):
        self.N_eff = N_eff
        self.eta = eta
        self.nbratio = NeutronBaryonRatio(N)
        self.phys_const = helper.PhysicalConstants()
        self.conversions = helper.UnitConversions()

    def rate_DD(self, z):
        """Reaction rate of n capture to He.

        R_DD includes all relevant reaction rates of
        n+p->D+photon, D+D->T+p, D+T->He+n.

        Args:
            z: scaled temperature variable, z=epsilon_D/T
        Returns:
            reaction rate for given z
        """
        return (2.35e7 * self.eta/self.phys_const.eta_zero
                * 1/z**(4/3) * np.exp(- 1.44 * z**(1/3)))

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
        return (1/self.G_np(z) * self.nbratio.X_no_n_decay()
                * (1 - self.nbratio.X_no_n_decay()))

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

        The root of <neutron_capture_condition is calculated and the
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


nbratio = NeutronBaryonRatio()
calc_t_c = CalculateCaptureTime()
print(nbratio.X_no_n_decay())
print(calc_t_c.calculate_capture_time())
print(nbratio.X_with_n_decay())
print(nbratio.He_mass_fraction())

int_max = 100
y_steps = np.arange(0.000001, 10, 0.001)

# print(fsolve(difference, 2.1))

# print(X_simps(int_max, 0.251))
# print(X_quad(10, 0.251))

# X_quad_results = np.zeros(y_steps.size)
# for i, y in enumerate(y_steps):
#    X_quad_results[i] = X_quad(y, 0.251)[0]

# plt.figure()
# plt.plot(y_steps, X_quad_results, 'k', y_steps, X_eq(y_steps), 'r--',
#          y_steps, X_quad_results-X_eq(y_steps), 'b--')
# plt.show()
