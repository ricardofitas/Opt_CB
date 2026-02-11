from Optimization_CB_2_liners_MC_patched_fast import calculate_ez_mc

X = [0.000525, 0.000525, (5.65 * 10 / 2.65)*1e-3, 10e-3, 0.000525] 

stats_uncoated = calculate_ez_mc(X, material="uncoated", n_samples=2000, seed=0)

X = [0.000517, 0.000517, (5.65 * 10 / 2.65)*1e-3, 10e-3, 0.000517]

stats_coated   = calculate_ez_mc(X, material="coated",   n_samples=2000, seed=0)

print("uncoated [meanEz,stdEz,meanEy,stdEy] =", stats_uncoated)
print("coated   [meanEz,stdEz,meanEy,stdEy] =", stats_coated)

from Optimization_CB_v2_prod_MC_patched import opt_calc_prod_mc

X1 = [1.791214375, 0.334639907, 8.487148517, 1.173761631, 10, 0, 0]

stats_uncoated = opt_calc_prod_mc(X1, material="uncoated", n_samples=20, seed=0)
stats_coated   = opt_calc_prod_mc(X1, material="coated",   n_samples=20, seed=0)

print("uncoated [meanObj,stdObj,meanEzeff,stdEzeff] =", stats_uncoated)
print("coated   [meanObj,stdObj,meanEzeff,stdEzeff] =", stats_coated)


X2 = [0.1, 9.645485035, 0.1, 10, 0.1, 1, 1]

stats_uncoated = opt_calc_prod_mc(X2, material="uncoated", n_samples=20, seed=0)
stats_coated   = opt_calc_prod_mc(X2, material="coated",   n_samples=20, seed=0)

print("uncoated [meanObj,stdObj,meanEzeff,stdEzeff] =", stats_uncoated)
print("coated   [meanObj,stdObj,meanEzeff,stdEzeff] =", stats_coated)

X3 = [1.383607899, 2.549796399, 5.818867374, 6.456406202, 5.142503464, 0.373052427, 0.471046705]

stats_uncoated = opt_calc_prod_mc(X3, material="uncoated", n_samples=20, seed=0)
stats_coated   = opt_calc_prod_mc(X3, material="coated",   n_samples=20, seed=0)

print("uncoated [meanObj,stdObj,meanEzeff,stdEzeff] =", stats_uncoated)
print("coated   [meanObj,stdObj,meanEzeff,stdEzeff] =", stats_coated)
