import neutron_baryon_ratio as nbratio
import variations_neutron_baryon_ratio as variations
import calculate_capture_time as capturetime
import helper_functions as helper

import numpy as np


def lambda_np(y, tau_n):
    """Interaction rate of reactions of neutrons to protons.

    Args:
        y: dimensionless temperature variable, y=delta_m/T
        tau_n: neutron mean life
    Returns:
        interaction rate at given y
     """
    return 255/tau_n * (12/y**5 + 6/y**4 + 1/y**3)


def total_reaction_rate(y, tau_n):
    """Total interaction rate of protons and neutrons.

    Total interaction rate = lambda_np + lambda_pn
    with lambda_pn = exp(-y) * lambda_np

    Args:
        y: dimensionless temperature variable, y=delta_m/T
        tau_n: neutron mean life
    Returns:
        total interaction rate at y
    """
    return (1 + np.exp(-y)) * lambda_np(y, tau_n)


def neutron_to_baryon_ratio(y):
    """Neutron-to-baryon ratio X(y) without including neutron decay.

    Args:
        y: dimensionless temperature variable, y=delta_m/T
    Returns:
        X_eq: equilibrium neutron-to-baryon ratio
        freeze_out_correction: correction to X_eq due to neutron freeze-out
        X_no_decay: neutron-to-baryon ratio without neutron decay
                    X_no_decay = X_eq + freeze_out_correction
    """
    X_no_decay = nbratio.NeutronBaryonRatio().X_no_n_decay(y)
    X_eq = nbratio.NeutronBaryonRatio().X_eq(y)
    freeze_out_correction = X_no_decay - X_eq

    return X_eq, freeze_out_correction, X_no_decay
