# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 21:27:15 2025

@author: ricar
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

# Parameters for the curve
lambda_ = 5.65e-3  # wavelength
Amp = 2.65e-3      # amplitude
num_waves = 5

# Generate x values for 5 waves
x = np.linspace(0, num_waves * lambda_, 1000)
# NURBS-like curve shape (approximation using a scaled sine wave for comparison)
nurbs_like_y = Amp * np.sin(2 * np.pi * x / lambda_)
# Actual sine wave for comparison
sine_y = Amp * np.sin(2 * np.pi * x / lambda_)

# Plotting
fig = plt.figure(figsize=(10, 3))
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])

# Top: NURBS-like curve (replacing the original custom shape with sine-like periodicity)
ax0 = plt.subplot(gs[0])
ax0.plot(x, nurbs_like_y, color='black', linewidth=2)
ax0.set_title("Extended Curve (5 Waves) – NURBS-like Approximation")
ax0.set_xticks([])
ax0.set_yticks([])
ax0.set_xlim([0, num_waves * lambda_])
ax0.set_aspect('equal')

# Bottom: Pure sine wave
ax1 = plt.subplot(gs[1])
ax1.plot(x, sine_y, color='black', linewidth=2)
ax1.set_title("Extended Curve (5 Waves) – Sine Function")
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_xlim([0, num_waves * lambda_])
ax1.set_aspect('equal')

plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from geomdl import NURBS
from sympy import symbols

# Mocking full_opt_process with a dummy function that returns high values to bypass the constraint
def full_opt_process(params):
    return 1.0, 1.0, params[-2], params[-1]

def opt_calc_prod(X, num_waves=5):
    lambda_ = 5.65e-3
    Amp = 2.65e-3

    suma = X[0] + X[1] + X[2] + X[3] + X[4]
    d1 = X[0] / suma
    d2 = X[1] / suma
    d3 = X[2] / suma
    d4 = X[3] / suma
    d5 = X[4] / suma
    r1 = 0.0103 * np.exp(9.17 * X[5]) + 0.1
    r2 = 0.0103 * np.exp(9.17 * X[6]) + 0.1

    r1_opt, r2_opt, X5_opt, X6_opt = full_opt_process([d1, d2, d3, d4, d5, X[5], X[6]])

    AAA = [0, d1, d1 + d2, d1 + d2 + d3, d1 + d2 + d3 + d4,
           1 + d1, 1 + d1 + d2, 1 + d1 + d2 + d3, 1 + d1 + d2 + d3 + d4, 2]
    BBB = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0]
    weights = np.array([1, r1, r1, r2, r2, r1, r1, r2, r2, 1])
    control_points1 = np.column_stack((AAA, BBB))

    curve = NURBS.Curve()
    curve.degree = 2
    curve.ctrlpts = control_points1.tolist()
    curve.weights = weights.tolist()

    num_ctrlpts = len(control_points1)
    degree = curve.degree
    num_knots = num_ctrlpts + degree + 1
    interior_knots = np.linspace(0, 1, num_knots - 2 * (degree + 1) + 2)[1:-1]
    curve.knotvector = np.concatenate([
        np.zeros(degree + 1),
        interior_knots,
        np.ones(degree + 1)
    ]).tolist()

    curve.sample_size = 2000
    curve.evaluate()
    curve_points = np.array(curve.evalpts)

    n_points = len(curve_points[:, 0])
    t1 = int(((d1 + d2 + d3 / 2) / 2) * n_points)
    t2 = int(((1 + d1 + d2 + d3 / 2) / 2) * n_points)

    l = abs(curve_points[t2, 0] - curve_points[t1, 0])
    scale1 = lambda_ / l
    height = np.max(curve_points[t1:t2, 1]) - np.min(curve_points[t1:t2, 1])
    scale2 = Amp / height

    local_curve0 = scale1 * curve_points[t1:t2, 0]
    local_curve1 = scale2 * curve_points[t1:t2, 1]
    local_curve = np.vstack([local_curve0, local_curve1])

    # Replicate the base curve to form the full 5-wave pattern
    full_curve = np.hstack([
        np.vstack([local_curve[0] + i * lambda_, local_curve[1]])
        for i in range(num_waves)
    ])

    
    # Plot the result and save with transparent background
    plt.figure(figsize=(10, 2))
    plt.plot(full_curve[0], full_curve[1], color='black', linewidth=12)
    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{case_name.replace(' ', '_').lower()}_nurbs.png", dpi=300, transparent=True)
    plt.close()

# Running the function for all 3 cases
cases = {
    "Caso 1": [0.1, 0.1, 10, 0.1, 10, 0.536972299, 0.561127743],
    "Caso 2": [3.522855914, 6.166853059, 3.650503401, 10, 1.923928021, 0.254534082, 0.99051799],
    "Caso 3": [5.545192772, 0.1, 9.380469802, 0.625598768, 0.974769034, 0, 0]
}

for case_name, X in cases.items():
    print(f"Saving: {case_name}")
    opt_calc_prod(X)

import numpy as np
import matplotlib.pyplot as plt

# Parâmetros da função senoide
lambda_ = 5.65e-3  # comprimento de onda [m]
Amp = 2.65e-3      # amplitude [m]
num_waves = 5

# Geração dos dados
x = np.linspace(0, num_waves * lambda_, 1000)
y = Amp * np.sin(2 * np.pi * x / lambda_)

# Criação da figura
fig, ax = plt.subplots(figsize=(10, 2))
ax.plot(x, y, color='black', linewidth=12)

# Estilo
ax.axis('equal')
ax.axis('off')
plt.tight_layout()

# Salvar em PNG com fundo transparente
fig.savefig("senoide_5ondas.png", dpi=300, transparent=True)
plt.close(fig)
