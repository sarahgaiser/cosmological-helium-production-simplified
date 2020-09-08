import neutron_baryon_ratio as nbratio
import variations_neutron_baryon_ratio as variations
import calculate_capture_time as capturetime
import helper_functions as helper
import important_functions_in_BBN as BBN_fctns

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib

matplotlib.rcParams['text.usetex'] = True


def plot_neutron_baryon_ratio_variation():
    """Plot neutron-to-baryon ratio with varying n_x*m_x/n_gamma.

    This function uses a set of already calculated data which is written
    to CompareNeutronFractionUsingnx.txt to plot the neutron-to-baryon
    ratio with varying n_x*m_x/n_gamma.
    """

    compare_neutron_data = np.loadtxt('CompareNeutronFractionUsingnx.txt')

    fig, ax = plt.subplots()
    ax.hlines(y=0.15, xmin=-1, xmax=11, color='grey', linestyle='--')
    ax.plot(compare_neutron_data[:, 0], compare_neutron_data[:, 1],
            color='green', label=r'$\frac{m_x n_{x,0}}{n_\gamma}$ = 0 MeV')
    ax.plot(compare_neutron_data[:, 0], compare_neutron_data[:, 2],
            linestyle='--', color='#8FBC8F',
            label=r'$\frac{m_x n_{x,0}}{n_\gamma}$ = 0.01 MeV')
    ax.plot(compare_neutron_data[:, 0], compare_neutron_data[:, 3],
            linestyle='-.', color='#8FBC8F',
            label=r'$\frac{m_x n_{x,0}}{n_\gamma}$ = 0.1 MeV')
    ax.plot(compare_neutron_data[:, 0], compare_neutron_data[:, 4],
            linestyle='-', color='#8FBC8F',
            label=r'$\frac{m_x n_x}{n_\gamma}$ = 1 MeV')

    # create legend for plot
    legend = ax.legend(loc='upper right', ncol=2)
    # set labels, limits on axes and title of plot
    ax.set_ylabel('X(y)')
    ax.set_xlabel('y')
    ax.set_xlim(-0.1, 10.1)
    ax.set_title(r'Neutron-to-total-baryon ratio X(y) for several values of '
                 r'$\frac{n_x m_x}{n_\gamma}$')

    # create plot of zoomed in version which is included in main plot
    axins = inset_axes(ax, width='50%', height='40%', loc=5, borderpad=1)
    axins.plot(compare_neutron_data[:, 0], compare_neutron_data[:, 1],
               color='green')
    axins.plot(compare_neutron_data[:, 0], compare_neutron_data[:, 2],
               linestyle='--', color='#8FBC8F')
    axins.plot(compare_neutron_data[:, 0], compare_neutron_data[:, 3],
               linestyle='-.', color='#8FBC8F')
    axins.plot(compare_neutron_data[:, 0], compare_neutron_data[:, 4],
               linestyle='-', color='#8FBC8F')
    # set axes limits for zoomed version
    axins.set_xlim(6, 10)
    axins.set_ylim(0.15, 0.165)

    plt.show()


def plot_neutron_baryon_ratio_difference():
    """Plot difference of neutron-to-baryon ratio with nx*mx/ngamma > 0
       to value without additional massive particle.

    This function uses a set of already calculated data which is written
    to CompareNeutronFractionUsingnx.txt.
    """
    compare_neutron_data = np.loadtxt('CompareNeutronFractionUsingnx.txt')

    fig, ax = plt.subplots()
    ax.plot(compare_neutron_data[:, 0],
            compare_neutron_data[:, 2] - compare_neutron_data[:, 1],
            linestyle='--', color='#8FBC8F',
            label=r'$\mathrm{X}(y, \frac{n_x m_x}{n_\gamma} = 0.01)$')
    ax.plot(compare_neutron_data[:, 0],
            compare_neutron_data[:, 3] - compare_neutron_data[:, 1],
            linestyle='-.', color='#8FBC8F',
            label=r'$\mathrm{X}(y, \frac{n_x m_x}{n_\gamma} = 0.1)$')
    ax.plot(compare_neutron_data[:, 0],
            compare_neutron_data[:, 4] - compare_neutron_data[:, 1],
            linestyle='-', color='#8FBC8F',
            label=r'$\mathrm{X}(y, \frac{n_x m_x}{n_\gamma} = 1)$')

    # create legend for plot
    legend = ax.legend(loc='upper left')
    # set labels, limits on axes and title of plot
    ax.set_ylabel(r'$\mathrm{X}(y, \frac{n_x m_x}{n_\gamma})'
                  r' - \mathrm{X}(y, 0)$')
    ax.set_xlabel('y')
    ax.set_title(r'Difference of neutron-to-total-baryon ratio '
                 r'$\mathrm{X}(y, \frac{n_x m_x}{n_\gamma})$'
                 r' to $\mathrm{X}(y, \frac{n_x m_x}{n_\gamma} = 0)$')

    plt.show()


def plot_total_reaction_rate():
    """Plot total reaction rate as function of y."""
    tau_n = helper.PhysicalConstants().tau_n

    y_values = np.arange(0.01, 10, 0.01)
    results_total = []
    results_np = []
    results_pn = []
    for y in y_values:
        results_total.append(BBN_fctns.total_reaction_rate(y, tau_n))
        results_np.append(BBN_fctns.lambda_np(y, tau_n))
        pn = (BBN_fctns.total_reaction_rate(y, tau_n)
              - BBN_fctns.lambda_np(y, tau_n))
        results_pn.append(pn)

    fig, ax = plt.subplots()
    ax.plot(y_values, results_total, color='green',
            label=r'$\Lambda(y)=(1+e^{-y})\lambda_{\mathrm{np}}$')
    ax.plot(y_values, results_np, color='#8FBC8F', linestyle='--',
            label=r'reaction n to p: $\lambda_{\mathrm{np}}$')
    ax.plot(y_values, results_pn,
            color='#8FBC8F', linestyle='-.',
            label=r'reaction p to n: $e^{-y}\lambda_{\mathrm{np}}$')

    ax.set_ylim(-0.1, 1)
    ax.set_xlabel(r'$y$')
    ax.set_ylabel(r'$\Lambda(y)$')
    ax.set_title(r'Total reaction rate $\Lambda(y)$ of neutrons and protons')
    legend = ax.legend(loc='upper right')

    plt.show()


def plot_I():
    """Plot integrating factor I(y, y') as function of y and y'."""
    y_values = np.arange(0.001, 10, 0.001)

    results_y_1 = []
    results_y_2 = []
    results_y_10 = []
    results_y_inf = []

    for y in y_values:
        k_y = helper.K(y, nbratio.NeutronBaryonRatio().b)
        k_y_1 = helper.K(1, nbratio.NeutronBaryonRatio().b)
        k_y_2 = helper.K(2, nbratio.NeutronBaryonRatio().b)
        k_y_10 = helper.K(10, nbratio.NeutronBaryonRatio().b)

        results_y_1.append(np.exp(k_y_1-k_y))
        results_y_2.append(np.exp(k_y_2-k_y))
        results_y_10.append(np.exp(k_y_10-k_y))
        results_y_inf.append(np.exp(-k_y))

    fig, ax = plt.subplots()
    ax.plot(y_values, results_y_1, color='#8FBC8F', linestyle='-',
            label=r'$y = 1$')
    ax.plot(y_values, results_y_2, color='#8FBC8F', linestyle='--',
            label=r'$y = 2$')
    ax.plot(y_values, results_y_10, color='#8FBC8F', linestyle='-.',
            label=r'$y = 10$')
    ax.plot(y_values, results_y_inf, color='green',
            label=r'$y = \infty$')

    legend = ax.legend(loc='upper left')
    ax.set_title(r'$I(y, y^\prime) = \exp(K(y) - K(y^\prime))$')
    ax.set_xlabel(r'$y^\prime$')
    ax.set_ylabel(r'$I(y, y^\prime)$')

    # created zoomed in plot
    axins = inset_axes(ax, width='50%', height='40%', loc=5, borderpad=1)
    axins.plot(y_values, results_y_1, linestyle='-', color='#8FBC8F')
    axins.plot(y_values, results_y_2, linestyle='--', color='#8FBC8F')
    axins.plot(y_values, results_y_10, linestyle='-.', color='#8FBC8F')
    axins.plot(y_values, results_y_inf, color='green')

    axins.set_xlim(0, 6)
    axins.set_ylim(0, 2)

    plt.show()


def plot_neutron_baryon_ratio():
    """Plot neutron-to-baryon ratio as function of y."""
    y_values = np.arange(0.01, 10, 0.01)
    results_X_eq, results_correction, results_X = [], [], []

    for y in y_values:
        X_eq, correction, X_y = BBN_fctns.neutron_to_baryon_ratio(y)
        results_X_eq.append(X_eq)
        results_correction.append(correction)
        results_X.append(X_y)

    fig, ax = plt.subplots()

    ax.plot(y_values, results_X, linestyle='-', color='green',
            label=r'$X(y)$')
    ax.plot(y_values, results_X_eq, linestyle='--', color='#8FBC8F',
            label=r'thermal equil. abundance $X_{\mathrm{eq}}(y)$')
    ax.plot(y_values, results_correction, linestyle='-.', color='#8FBC8F',
            label=r'freeze-out correction')

    legend = ax.legend(loc='upper right')
    ax.set_xlim(0, 10)
    ax.set_xlabel(r'$y$')
    ax.set_ylabel(r'$X(y)$')
    ax.set_title(r'Neutron-to-baryon ratio $X(y)$'
                 r' as function of $y=\frac{\Delta m}{T}$')
    plt.show()


def plot_variation_with_eta():
    """Plot variation of helium abundance with eta."""
    He_fraction_variation = variations.VariationsOfNeutronBaryonRatio()
    eta_values = np.arange(1, 10, 0.1)*1e-10

    He_fraction = He_fraction_variation.He_mass_fraction()
    results = []
    for eta in eta_values:
        results.append(He_fraction
                       + He_fraction_variation.variation_with_eta(eta))

    fig, ax = plt.subplots()

    ax.plot(eta_values, results, color='green')
    ax.hlines(y=He_fraction, xmin=0, xmax=11e-10, color='grey',
              linestyle='--')
    ax.vlines(x=5e-10, ymin=0, ymax=1, color='grey', linestyle='--')

    ax.set_ylim(0.22, 0.25)
    ax.set_xlim(0.8e-10, 10.2e-10)

    ax.set_title(r'Variation of helium abundance $Y_4$ with $\eta$')
    ax.set_ylabel(r'$Y_4(\eta)$')
    ax.set_xlabel(r'$\eta$')
    plt.show()


def plot_variation_with_tau():
    """Plot variation of helium abundance with tau (neutron mean life)."""
    He_fraction_variation = variations.VariationsOfNeutronBaryonRatio()
    delta_tau_values = np.arange(-30, 30, 0.1)

    He_fraction = He_fraction_variation.He_mass_fraction()
    tau, results = [], []
    for delta_tau in delta_tau_values:
        variation = He_fraction_variation.variation_with_tau_n(delta_tau)
        results.append(He_fraction + variation)
        tau.append(helper.PhysicalConstants().tau_n + delta_tau)

    fig, ax = plt.subplots()

    ax.plot(tau, results, color='green')
    ax.hlines(y=He_fraction, xmin=840, xmax=920, color='grey',
              linestyle='--')
    ax.vlines(x=879.4, ymin=0, ymax=1, color='grey', linestyle='--')

    ax.set_ylim(0.232, 0.246)
    ax.set_xlim(848, 912)

    ax.set_title(r'Variation of helium abundance $Y_4$ with $\tau_\mathrm{n}$')
    ax.set_ylabel(r'$Y_4(\tau_\mathrm{n})$')
    ax.set_xlabel(r'$\tau_\mathrm{n}$')
    plt.show()


def plot_variation_with_N_neutrinos():
    """Plot variation of helium abundance with nuber of neutrinos."""
    He_fraction_variation = variations.VariationsOfNeutronBaryonRatio()
    delta_N_nu_values = np.arange(-3, 3, 0.1)

    He_fraction = He_fraction_variation.He_mass_fraction()
    N_nu, results = [], []
    for delta_N_nu in delta_N_nu_values:
        vari = He_fraction_variation.variation_with_neutrino_number(delta_N_nu)
        results.append(He_fraction + vari)
        N_nu.append(3 + delta_N_nu)

    fig, ax = plt.subplots()

    ax.plot(N_nu, results, color='green')
    ax.hlines(y=He_fraction, xmin=-1, xmax=7, color='grey',
              linestyle='--')
    ax.vlines(x=3, ymin=0, ymax=1, color='grey', linestyle='--')

    ax.set_ylim(0.15, 0.28)
    ax.set_xlim(-0.1, 6)

    ax.set_title(r'Variation of helium abundance $Y_4$ with $N_\nu$')
    ax.set_ylabel(r'$Y_4(N_\nu)$')
    ax.set_xlabel(r'$N_\nu$')
    plt.show()


def plot_variation_with_alpha():
    """Plot variation of helium abundance with alpha
       (scaled chemical potential electron neutrino)."""
    He_fraction_variation = variations.VariationsOfNeutronBaryonRatio()
    alpha_values = np.arange(0, 0.1, 0.001)

    He_fraction = He_fraction_variation.He_mass_fraction()
    results = []
    for alpha in alpha_values:
        variation = He_fraction_variation.variation_with_alpha(alpha)
        results.append(He_fraction + variation)

    fig, ax = plt.subplots()

    ax.plot(alpha_values, results, color='green')
    ax.hlines(y=He_fraction, xmin=-1, xmax=11, color='grey',
              linestyle='--')
    ax.vlines(x=0, ymin=0, ymax=1, color='grey', linestyle='--')

    ax.set_ylim(0.21, 0.245)
    ax.set_xlim(-0.001, 0.101)

    ax.set_title(r'Variation of helium abundance $Y_4$ with $\alpha$')
    ax.set_ylabel(r'$Y_4(\alpha)$')
    ax.set_xlabel(r'$\alpha$')
    plt.show()


def plot_capture_time_with_nx_fraction():
    """Plot capture time of neutrons as function of (n_x m_x)/n_gamma.

    This function uses a set of already calculated data which is written
    to CaptureTimeUsingnx.txt to plot the capture time of neutrons as
    function of (n_x m_x)/n_gamma.
    """
    tc_data = np.loadtxt('CaptureTimeUsingnx.txt')
    plt.title(r'Change of capture time $t_c$ with '
              r'$\frac{m_X n_{X,0}}{n_{\gamma, 0}}$')
    plt.xlabel(r'$\frac{m_X n_{X,0}}{n_{\gamma, 0}}$/MeV')
    plt.ylabel(r'$t_c$/s')
    plt.plot(tc_data[:, 0], tc_data[:, 1], color='green')
    plt.xlim(0, 1)
    plt.ylim(100, 190)
    plt.show()


def plot_He_mass_fraction_with_nx_fraction():
    """Plot mass fraction of helium as function of (n_x m_x)/n_gamma.

    This function uses a set of already calculated data which is written
    to HeMassFractionUsingnx_correct.txt to plot the helium abundance
    as function of (n_x m_x)/n_gamma.
    """
    He_data = np.loadtxt('HeMassFractionUsingnx_correct.txt')
    plt.title(r'Change of helium mass fraction $Y_4$ with '
              r'$\frac{m_X n_{X,0}}{n_{\gamma, 0}}$')
    plt.xlabel(r'$\frac{m_X n_{X,0}}{n_{\gamma, 0}}$/MeV')
    plt.ylabel(r'$Y_4$')
    plt.hlines(y=0.239, xmin=-1, xmax=101, color='grey',
               linestyle='--')
    plt.plot(He_data[:, 0], He_data[:, 1], color='green')
    plt.xlim(0, 1)
    plt.ylim(0.22, 0.28)
    plt.show()


# plot_neutron_baryon_ratio_variation()
# plot_total_reaction_rate()
# plot_I()
# plot_neutron_baryon_ratio()
# plot_variation_with_alpha()
# plot_variation_with_N_neutrinos()
# plot_variation_with_tau()
# plot_variation_with_eta()
# plot_capture_time_with_nx_fraction()
# plot_He_mass_fraction_with_nx_fraction()
