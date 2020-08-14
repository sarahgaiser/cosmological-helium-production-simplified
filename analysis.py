import neutron_baryon_ratio as nbratio
import variations_neutron_baryon_ratio as variations
import calculate_capture_time as capturetime

var_nbratio = variations.VariationsOfNeutronBaryonRatio()

print(var_nbratio.variation_with_alpha(1))
print(var_nbratio.variation_with_neutrino_number(1))
print(var_nbratio.variation_with_tau_n(90))
print(var_nbratio.variation_with_eta(6e-10))

print(nbratio.NeutronBaryonRatio().He_mass_fraction())

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
