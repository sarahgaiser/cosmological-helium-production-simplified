import scipy.integrate as integrate
import numpy as np
import helper_functions as helper

from neutron_baryon_ratio import NeutronBaryonRatio


class VariationsOfNeutronBaryonRatio(NeutronBaryonRatio):
    """Class including effects of variation of parameters on neutron abundance.

    VariationsOfNeutronBaryonRatio inherits from NeutronBaryonRatio.
    It considers the effects of the variation of the coeffient b,
    describing the interaction rate of the neutron, the chemical potential
    of the electron, called alpha, ...

    In the __init__ function, NeutronBaryonRatio is initialized with the
    effective number of relativistic degrees of freedom N
    before e+e- annihilation, and the effective number of relativistic degrees
    of freedom N_eff after e+e- annihilation is set.

    Args:
        N: effective number of relativistic degrees of freedom
           before e+e- annihilation
        N_eff: effective number of relativistic degrees of freedom
               after e+e- annihilation
        eta: baryon-to-photon ratio
    """
    def __init__(self, N=43./4, N_eff=13, eta=5e-10):
        # the __init__ function is kept for now in case something
        # should be added, it's not really of any use
        # as it is at the moment
        NeutronBaryonRatio.__init__(self, N, N_eff, eta)

    def C_var_with_b_coeff(self):
        """Calculates the factor C needed in variation of b.

        To calculate the relative variation of the neutron-to-baryon fraction
        (without neutron decay) with change of the parameter b,
        the factor C needs to be calculated which is defined as follows:

        delta_X/X = -C * delta_b/b

        Returns:
            C = 1/X * integrate(0,infinity)[exp(y - K(y)) * X_eq(y)^2 * K(y)]
        """
        def C_integrand(y):
            """Defines integrand needed to calculate C as above.

            Integrand = exp(y - K(y)) * X_eq(y)^2 * K(y)

            Args:
                y: dimensionless temperature variable, y=delta_m/T
            """
            return (np.exp(y - helper.K(y, self.b)) * self.X_eq(y)**2
                    * helper.K(y, self.b))

        return 1./self.X_no_n_decay() * integrate.quad(C_integrand, 0, 100)[0]

    def X_1_var_with_alpha(self):
        """Calculates factor X_1 needed in variation with alpha.

        To account for possible chemical potential of electron neutrino,
        itroduce alpha=mu/T. The neutron-to-baryon ratio then becomes:

        X_alpha = exp(-alpha) * X - alpha * X_1

        NOTE: X_1 is very small and can therefore be neglected. This
        function is included only for the sake of completeness.

        Returns:
            X_1 = integrate(0,infinity)[exp(y - K(y)) * X_eq(y)^2 * G(y)]
        """
        def X_1_integrand(y):
            """Defines integrand needed to calculate X_1."""
            return (np.exp(y - helper.K(y, self-b)) * self.X_eq(y)**2
                    * helper.G(y, self.b))

        return integrate.quad(X_1_integrand, 0, 100)[0]

    def variation_with_alpha(self, alpha):
        """Variation of He mass fraction with alpha.

        This function calculates the variation of the primordial helium
        mass fraction with the chemical potential of electron neutrinos.

        Args:
            alpha: alpha = mu/T, with mu chem potential of electron neutrino
                   and T temperature
        Returns:
            variation of He mass fraction with alpha
        """
        return -self.He_mass_fraction() * alpha

    def variation_with_neutrino_number(self, delta_N_nu):
        """Variation of He mass fraction with neutrino number.

        This function calculates the variation of the primordial helium
        mass fraction with the number of neutrinos.

        Args:
            delta_N_nu: delta_N_nu = N_nu - 3 with N_nu number of neutrinos
        Returns:
            variation of He mass fraction with neutrino number
        """
        self.set_t_c(self.X_no_n_decay())
        C = self.C_var_with_b_coeff()
        prefactor = ((1/(2*self.N_eff) * self.t_c/self.phys_const.tau_n
                      + C/(2*self.N))
                     * 7/4 * self.He_mass_fraction())

        return prefactor * delta_N_nu

    def variation_with_tau_n(self, delta_tau_n):
        """Variation of He mass fraction with neutron mean life.

        This function calculates the variation of the primordial helium
        mass fraction with the mean life of neutrons.

        Args:
            delta_tau_n: delta_tau_n = (tau_n - 879.4) in s
                         with tau_n neutron mean life
        Returns:
            variation of He mass fraction with neutron mean life
        """
        self.set_t_c(self.X_no_n_decay())
        tau = self.phys_const.tau_n
        C = self.C_var_with_b_coeff()
        return self.He_mass_fraction()/tau * (C + self.t_c/tau) * delta_tau_n

    def variation_with_eta(self, eta):
        """Variation of He mass fraction with eta.

        This function calculates the variation of the primordial helium
        mass fraction with the baryon-to-photon ratio eta.

        Args:
            eta: new value of baryon-to-photon ratio
        Returns:
            variation of He mass fraction with baryon-to-photon ratio
        """
        self.set_t_c(self.X_no_n_decay())
        tau = self.phys_const.tau_n
        prefactor = 0.18 * self.He_mass_fraction() * self.t_c/tau
        return prefactor * np.log(eta/self.eta)
