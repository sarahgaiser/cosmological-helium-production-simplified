# Cosmological helium production simplified

The work in this project is based on the paper "Cosmological helium production simplified" by Bernstein, Braun and Feinberg (1989).
It implements all important integrals and functions to determine the helium abundance after BBN which were mentioned in the paper.
Furthermore, an additional section includes the change of the helium abundance in case an additional stable, massive and non-relativistic
particle is preset during BBN.

# Structure
- **neutron_baryon_ratio.py** Class containing important equations for neutron-to-baryon ratio.
- **calculate_capture_time.py** Class including all the functions needed to calculate t_c.
- **variations_neutron_baryon_ratio.py** Class including effects of variation of parameters on neutron abundance.
- **include_massive_particle.py** Contains all equations needed to include influence of massive particle on BBN.
- **helper_functions.py**, **important_functions_in_BBN.py** Two sets of helper functions.
- **analysis.py** Contains functions used to plot several functions of interest.
