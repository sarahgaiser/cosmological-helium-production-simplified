import neutron_baryon_ratio as nbratio
import variations_neutron_baryon_ratio as variations
import calculate_capture_time as capturetime

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib

matplotlib.rcParams['text.usetex'] = True


# plot neutron-to-baryon ratio with varying nx*mx/ngamma
compare_neutron_data = np.loadtxt('CompareNeutronFractionUsingnx.txt')

fig, ax = plt.subplots()
ax.hlines(y=0.15, xmin=-1, xmax=11, color='grey', linestyle='--')
ax.plot(compare_neutron_data[:, 0], compare_neutron_data[:, 1],
        color='green', label=r'$\frac{n_x m_x}{n_\gamma}$ = 0')
ax.plot(compare_neutron_data[:, 0], compare_neutron_data[:, 2], linestyle='--',
        color='#8FBC8F', label=r'$\frac{n_x m_x}{n_\gamma}$ = 0.01')
ax.plot(compare_neutron_data[:, 0], compare_neutron_data[:, 3], linestyle='-.',
        color='#8FBC8F', label=r'$\frac{n_x m_x}{n_\gamma}$ = 0.1')
ax.plot(compare_neutron_data[:, 0], compare_neutron_data[:, 4], linestyle='-',
        color='#8FBC8F', label=r'$\frac{n_x m_x}{n_\gamma}$ = 1')
legend = ax.legend(loc='upper right', ncol=2)
ax.set_ylabel('X(y)')
ax.set_xlabel('y')
ax.set_xlim(-0.1, 10.1)
ax.set_title(r'Neutron-to-total-baryon ratio X(y) for several values of '
             r'$\frac{n_x m_x}{n_\gamma}$')

axins = inset_axes(ax, width='50%', height='40%', loc=5, borderpad=1)
axins.plot(compare_neutron_data[:, 0], compare_neutron_data[:, 1],
           color='green')
axins.plot(compare_neutron_data[:, 0], compare_neutron_data[:, 2],
           linestyle='--', color='#8FBC8F')
axins.plot(compare_neutron_data[:, 0], compare_neutron_data[:, 3],
           linestyle='-.', color='#8FBC8F')
axins.plot(compare_neutron_data[:, 0], compare_neutron_data[:, 4],
           linestyle='-', color='#8FBC8F')

x1, x2, y1, y2 = 6, 10, 0.15, 0.165  # specify the limits
axins.set_xlim(x1, x2)  # apply the x-limits
axins.set_ylim(y1, y2)  # apply the y-limits

# plot difference of neutron-to-baryon ratio with nx*mx/ngamma > 0
# to value without additional massive particle

fig2, ax2 = plt.subplots()
ax2.plot(compare_neutron_data[:, 0],
         compare_neutron_data[:, 2] - compare_neutron_data[:, 1],
         'r--', label='nx = 0.01')
ax2.plot(compare_neutron_data[:, 0],
         compare_neutron_data[:, 3] - compare_neutron_data[:, 1],
         'r-.', label='nx = 0.1')
ax2.plot(compare_neutron_data[:, 0],
         compare_neutron_data[:, 4] - compare_neutron_data[:, 1],
         'r-', label='nx = 1')
legend = ax2.legend(loc='upper left')

plt.show()


# tc_data = np.loadtxt('CaptureTimeUsingnx.txt')
# plt.plot(tc_data[:, 0], tc_data[:, 1])
# plt.show()

# He_data = np.loadtxt('HeMassFractionUsingnx.txt')
# plt.plot(He_data[:, 0], He_data[:, 1])
# plt.xlim(0, 10)
# plt.show()
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
