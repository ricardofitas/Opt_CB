import numpy as np
import matplotlib.pyplot as plt
from geomdl import NURBS
from scipy.optimize import minimize

# Define target shape points
d1, d2, d3, d4, d5 = 0.2, 0.1, 0.4, 0.1, 0.2
segments = [0, d1, d1 + d2/2, d1 + d2, d1 + d2 + d3/2, d1 + d2 + d3, d1 + d2 + d3 + d4/2, d1 + d2 + d3 + d4, 1]
heights = [0, 0, 0.5, 1, 1, 1, 0.5, 0, 0]
control_points = np.column_stack((segments, heights))

# NURBS evaluation function
def evaluate_nurbs(r1, r2, num_points=300):
    weights = [1, r1, 1, r1, 1, r2, 1, r2, 1]
    curve = NURBS.Curve()
    curve.degree = 2
    curve.ctrlpts = control_points.tolist()
    curve.weights = weights

    # Construct knot vector
    num_ctrlpts = len(control_points)
    num_knots = num_ctrlpts + curve.degree + 1
    interior_knots = np.linspace(0, 1, num_knots - 2 * (curve.degree + 1) + 2)[1:-1]
    knotvector = np.concatenate([
        np.zeros(curve.degree + 1),
        interior_knots,
        np.ones(curve.degree + 1)
    ])
    curve.knotvector = knotvector.tolist()

    # Sample the curve
    curve.sample_size = num_points
    curve.evaluate()
    return np.array(curve.evalpts)

# Objective to minimize (MSE)
def objective(x):
    r1, r2 = x
    nurbs_pts = evaluate_nurbs(r1, r2)
    x_target = np.linspace(0, 1, len(nurbs_pts))
    y_target = np.interp(x_target, segments, heights)
    return np.mean((nurbs_pts[:, 1] - y_target) ** 2)

# Optimize r1 and r2
x0 = [2, 2]
result = minimize(objective, x0, bounds=[(1, 10), (1, 10)])
opt_r1, opt_r2 = result.x

# Final curve
curve_pts = evaluate_nurbs(opt_r1, opt_r2)

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(segments, heights, 'r--o', label='Target shape')
plt.plot(curve_pts[:, 0], curve_pts[:, 1], 'b-', label=f'NURBS fit (r1={opt_r1:.2f}, r2={opt_r2:.2f})')
plt.legend()
plt.title('Fitting NURBS to Desired Geometry')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.tight_layout()
plt.show()
