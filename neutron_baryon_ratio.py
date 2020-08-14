import scipy.integrate as integrate
import numpy as np
import helper_functions as helper

import calculate_capture_time as capturetime


class NeutronBaryonRatio:
    """Class containing important equations for neutron-to-baryon ratio.

    This class includes formulas for the neutron-to-baryon ratio X
    in equilibrium, X_eq; out of equilibrium without neutron decay,
    X_no_n_decay; and with neutron decay, X_with_n_decay.

    __init__ takes (and sets) the number of effective relativistic
    degrees of freedom N before electron-positron-annihilation, and after
    then called N_eff and the baryon-to-photon ratio eta.
    It also sets the parameter a=255 and b that are used to express the
    interaction rate of the neutron. In addition, it initalizes an instance
    of the PhysicalConstants class.

    Args:
        N: number of effective relativistic degrees of freedom N before
           e-e+ annihilation
        N_eff: number of effective relativistic degrees of freedom N after
               e-e+ annihilation
        eta: baryon-to-photon ratio
    """
    def __init__(self, N=43./4, N_eff=13, eta=5e-10):
        self.phys_const = helper.PhysicalConstants()

        self.N = N
        self.N_eff = N_eff
        self.eta = eta
        self.a = 255
        self.b = helper.calculate_b(self.phys_const, self.a, self.N)
        self.t_c = 180

    def set_t_c(self, X_no_decay):
        """Setter function for neutron capture time.

        Sets the neutron capture time in helium which is calculated using
        the CalculateCaptureTime class and its method calculate_capture_time().
        """
        calc_t_c = capturetime.CalculateCaptureTime(X_no_decay, self.N_eff,
                                                    self.eta)
        self.t_c = calc_t_c.calculate_capture_time()

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
        def X_integrand(x, y):
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
            self.set_t_c(self.X_no_n_decay())
        return np.exp(-self.t_c/self.phys_const.tau_n) * self.X_no_n_decay()

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
