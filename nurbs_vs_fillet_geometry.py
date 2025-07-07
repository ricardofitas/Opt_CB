import cadquery as cq
import numpy as np
import matplotlib.pyplot as plt
from geomdl import NURBS

# Input design vector and radius estimates
X = [0, 1, 0, 1, 0, 0, 0]
lambda_ = 5.65e-3
Amp = 2.65e-3

# Generate normalized control points
d1, d2, d3, d4, d5 = X[:5]
total = sum([d1, d2, d3, d4, d5])
norm = lambda x: x / total
AAA = np.array([
    0,
    norm(d1),
    norm(d1 + d2 / 2),
    norm(d1 + d2),
    norm(d1 + d2 + d3 / 2),
    norm(d1 + d2 + d3),
    norm(d1 + d2 + d3 + d4 / 2),
    norm(d1 + d2 + d3 + d4),
    1
])
BBB = np.array([0, 0, 0.5, 1, 1, 1, 0.5, 0, 0])
pts = [(x * lambda_ * 1000, y * Amp * 1000) for x, y in zip(AAA, BBB)]

# Build wire from points
w = cq.Workplane("XY").moveTo(*pts[0])
for i in range(1, len(pts)):
    w = w.lineTo(*pts[i])

# Explicitly construct a wire (CadQuery object) and extract raw OCC shape
cq_wire = w.wire().val()

# Use cq.occ_impl.shapes.Shape (import explicitly) to access .discretize()
from cadquery.occ_impl.shapes import Shape

wire_pts = np.array(pts)

# NURBS curve generation
def evaluate_nurbs(X, r1_weight, r2_weight, sample_size=2000):
    suma = sum(X[:5])
    d1 = X[0] / suma
    d2 = X[1] / suma
    d3 = X[2] / suma
    d4 = X[3] / suma
    d5 = X[4] / suma
    AAA = [0, d1, d1 + d2 / 2, d1 + d2, d1 + d2 + d3 / 2,
           d1 + d2 + d3, d1 + d2 + d3 + d4 / 2, d1 + d2 + d3 + d4, 1]
    BBB = [0, 0, 0.5, 1, 1, 1, 0.5, 0, 0]
    ctrlpts = list(zip(AAA, BBB))
    weights = [1, r1_weight, 1, r1_weight, 1, r2_weight, 1, r2_weight, 1]
    curve = NURBS.Curve()
    curve.degree = 2
    curve.ctrlpts = ctrlpts
    curve.weights = weights
    curve.sample_size = sample_size
    num_knots = len(ctrlpts) + curve.degree + 1
    interior_knots = np.linspace(0, 1, num_knots - 2 * (curve.degree + 1) + 2)[1:-1]
    curve.knotvector = np.concatenate([
        np.zeros(curve.degree + 1),
        interior_knots,
        np.ones(curve.degree + 1)
    ]).tolist()
    curve.evaluate()
    points = np.array(curve.evalpts)
    scale_x = lambda_ / (points[-1, 0] - points[0, 0])
    scale_y = Amp / (np.max(points[:, 1]) - np.min(points[:, 1]))
    points[:, 0] *= scale_x * 1000
    points[:, 1] *= scale_y * 1000
    return points

r1_weight = 0.0103 * np.exp(9.17 * X[5]) + 0.9897
r2_weight = 0.0103 * np.exp(9.17 * X[6]) + 0.9897
nurbs_pts = evaluate_nurbs(X, r1_weight, r2_weight)

# Plot
plt.figure(figsize=(12, 5))
plt.plot(nurbs_pts[:, 0], nurbs_pts[:, 1], 'r-', label='NURBS Curve')
plt.plot(wire_pts[:, 0], wire_pts[:, 1], 'b--', label='Piecewise Geometry')
plt.title("NURBS vs. Constructed Geometry")
plt.xlabel("X [mm]")
plt.ylabel("Y [mm]")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
