import csv
import ast
import pandas as pd
import re
import numpy as np
from scipy.optimize import minimize
from geomdl import NURBS


def build_control_points(X):
    # Normalize distances
    suma = sum(X[:5])
    d1 = X[0] / suma
    d2 = X[1] / suma
    d3 = X[2] / suma
    d4 = X[3] / suma
    d5 = X[4] / suma

    # Target envelope for comparison (piecewise shape)
    AAA = [0, d1, d1+d2, d1+d2+d3, d1+d2+d3+d4, 1 + d1, 1+d1+d2, 1+d1+d2+d3, 1+d1+d2+d3+d4, 2]  
    # AAA = [0, lambda_/2 - ep, lambda_/2, lambda_ - ep, lambda_]
    BBB = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0]  
    # BBB = [1, 1, 0, 0, 1]
    
    
    return np.column_stack((AAA, BBB)), AAA, BBB


def evaluate_scaled_curve(X, r1_weight, r2_weight, sample_size=100):
    control_points1, _, _ = build_control_points(X)

    # Assign weights
    weights = np.array([1, r1_weight, r1_weight, r2_weight, r2_weight, r1_weight, r1_weight, r2_weight, r2_weight, 1])
    
    curve = NURBS.Curve()
    curve.degree = 2
    curve.ctrlpts = control_points1.tolist()
    curve.weights = weights.tolist()

    num_ctrlpts = len(control_points1)
    num_knots = num_ctrlpts + curve.degree + 1
    interior_knots = np.linspace(0, 1, num_knots - 2 * (curve.degree + 1) + 2)[1:-1]
    curve.knotvector = np.concatenate([
        np.zeros(curve.degree + 1),
        interior_knots,
        np.ones(curve.degree + 1)
    ]).tolist()

    curve.sample_size = sample_size
    curve.evaluate()
    curve_points = np.array(curve.evalpts)

    lambda_ = 5.65
    Amp = 2.65
    
    suma = X[0] + X[1] + X[2] + X[3] +  X[4]
    d1 = X[0]/ suma #0.5
    d2 = X[1]/ suma #0.1
    d3 = X[2]/ suma #0.4
    
    n_points = len(curve_points[:,0])
    t1 = int(( (d1+d2+d3/2 )/ 2) * n_points)  # end of r1 zone
    t2 = int(( 1/ 2) * n_points)            # end of full profile (r2 zone ends here)
    t3 = int(((1+d1+d2+d3/2) / 2) * n_points) 
    
    # Compute scaling factors
    l = abs(curve_points[t2, 0] - curve_points[t1, 0])
    scale1 = lambda_ / l
    height = np.max(curve_points[t1:t2,1]) - np.min(curve_points[t1:t2,1])
    scale2 = Amp / height
    
    
    # Scale curve points
    local_curve0 = scale1 * curve_points[:,0]
    local_curve1 = scale2 * curve_points[:,1]
    local_curve = np.vstack([local_curve0, local_curve1])

    return local_curve, t1, t2, t3

def compute_curvature_radius(x, y):
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
    radius = 1 / (curvature + 1e-8)  # small epsilon to avoid division by zero
    return radius, curvature


def plot_nurbs_vs_fillet_geometry_true_arcs_fixed(X, r1_opt, r2_opt):
    d1, d2, d3, d4, d5 = X[:5]
    total_length = sum([d1, d2, d3, d4, d5])
    norm = lambda x: x / total_length

    AAA = np.array([0,
                    norm(d1),
                    norm(d1 + d2),
                    norm(d1 + d2 + d3),
                    norm(d1 + d2 + d3 + d4),
                    1])
    BBB = np.array([0, 0, 1, 1, 0, 0])

    lambda_ = 5.65  # m
    Amp = 2.65      # m

    points = np.column_stack((AAA * lambda_, BBB * Amp))

    fig, ax = plt.subplots(figsize=(12, 5))

    fillet_indices = [2, 4, 6, 8]
    i = 0
    while i < len(points) - 1:
        if (i + 1 in fillet_indices) and (i > 0 and i + 2 < len(points)):
            p_start = points[i]
            p_corner = points[i + 1]
            p_end = points[i + 2]
            radius = r1_opt / 1000 if i + 1 in [2, 8] else r2_opt / 1000
            arc_x, arc_y = compute_fillet_arc_safe(p_start, p_corner, p_end, radius)

            if arc_x.size > 0:
                arc_start = np.array([arc_x[0], arc_y[0]])
                arc_end = np.array([arc_x[-1], arc_y[-1]])
                ax.plot([p_start[0], arc_start[0]], [p_start[1], arc_start[1]], 'b--')
                ax.plot(arc_x, arc_y, color='orange')
                ax.plot([arc_end[0], p_end[0]], [arc_end[1], p_end[1]], 'b--')
                i += 2
                continue

        ax.plot([points[i, 0], points[i + 1, 0]], [points[i, 1], points[i + 1, 1]], 'b--')
        i += 1

    ax.set_title("Optimized Geometry with True Fillet Arcs (Corrected)")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.grid(True)
    plt.tight_layout()
    plt.show()
    
def robust_percentile_radius(curvature, percentile=5):
    # Smooth to avoid local spikes
    curvature_smooth = gaussian_filter1d(curvature, sigma=10)
    
    # Clip small curvature (i.e., flat areas) if they dominate
    nonzero_curv = curvature_smooth[curvature_smooth > 1e-6]
    if len(nonzero_curv) < 10:
        return np.inf  # Too flat to evaluate meaningfully

    radius = 1 / nonzero_curv
    return np.percentile(radius, percentile)

def optimize_r1_r2_given_X(X):
    # Step 1: Get full NURBS curve
    r1_init = 0.0103 * np.exp(9.17 * X[5]) + 0.1
    r2_init = 0.0103 * np.exp(9.17 * X[6]) + 0.1
    curve, t1, t2, t3 = evaluate_scaled_curve(X, r1_init, r2_init)
    x, y = curve[0], curve[1]
    x = list(x) + list(x[:2])
    y = list(y) + list(y[:2])

    # Step 2: Compute radius of curvature at each point
    radius,  _ = compute_curvature_radius(x, y)
    
    # Extract radii zones
    r1_zone = radius[t1:t2+2]
    r2_zone = radius[t2:t3+2]
    

    # Step 5: Take minimum radius in each region (maximum curvature)
    r1_opt = np.min(r1_zone)
    r2_opt = np.min(r2_zone)

    return r1_opt, r2_opt

def inverse_weight_to_design_param(r):
    return (np.log((r - 0.9897) / 0.0103)) / 9.17


def full_opt_process(X):
    # Optimize r1 and r2 weights
    r1_opt_weight, r2_opt_weight = optimize_r1_r2_given_X(X)
    return r1_opt_weight, r2_opt_weight, X[5], X[6]


# This code is now ready to be integrated into your CSV processing function
# to compute r1_opt, r2_opt and corresponding X values for every entry.


# Classification table based on the provided image
classification_table = [
    ("G", (None, 1.8), (None, 0.6)),
    ("F", (1.8, 2.6), (0.6, 1.0)),
    ("E", (2.6, 3.5), (1.0, 1.9)),
    ("D", (3.5, 4.8), (1.9, 2.2)),
    ("B", (4.8, 6.5), (2.2, 3.1)),
    ("C", (6.5, 7.9), (3.1, 4.0)),
    ("A", (7.9, 10.0), (4.0, 5.0)),
    ("K", (10.0, None), (5.0, None))
]


def classify(value, ranges):
    """Classify a value based on given ranges."""
    for label, (low, high) in ranges:
        if (low is None or value >= low) and (high is None or value < high):
            return label
    return "Unknown"

def clean_np_float_strings(s):
    """Replace 'np.float64(x)' or 'numpy.float64(x)' with 'x' in a string."""
    return re.sub(r'(np\.float64|numpy\.float64)\(([^)]+)\)', r'\2', s)

import matplotlib.pyplot as plt
from matplotlib.patches import Arc

def plot_nurbs_vs_fillet_geometry_true_arcs_fixed(X, r1_opt, r2_opt):
    d1, d2, d3, d4, d5 = X[:5]
    total_length = sum([d1, d2, d3, d4, d5])
    norm = lambda x: x / total_length

    AAA = np.array([0,
                    norm(d1),
                    norm(d1 + d2 / 2),
                    norm(d1 + d2),
                    norm(d1 + d2 + d3 / 2),
                    norm(d1 + d2 + d3),
                    norm(d1 + d2 + d3 + d4 / 2),
                    norm(d1 + d2 + d3 + d4),
                    1])
    BBB = np.array([0, 0, 0.5, 1, 1, 1, 0.5, 0, 0])

    lambda_ = 5.65e-3  # m
    Amp = 2.65e-3      # m

    points = np.column_stack((AAA * lambda_, BBB * Amp))

    fig, ax = plt.subplots(figsize=(12, 5))

    fillet_indices = [2, 4, 6, 8]
    i = 0
    while i < len(points) - 1:
        if (i + 1 in fillet_indices) and (i > 0 and i + 2 < len(points)):
            p_start = points[i]
            p_corner = points[i + 1]
            p_end = points[i + 2]
            radius = r1_opt / 1000 if i + 1 in [2, 8] else r2_opt / 1000
            arc_x, arc_y = compute_fillet_arc_safe(p_start, p_corner, p_end, radius)

            if arc_x.size > 0:
                arc_start = np.array([arc_x[0], arc_y[0]])
                arc_end = np.array([arc_x[-1], arc_y[-1]])
                ax.plot([p_start[0], arc_start[0]], [p_start[1], arc_start[1]], 'b--')
                ax.plot(arc_x, arc_y, color='orange')
                ax.plot([arc_end[0], p_end[0]], [arc_end[1], p_end[1]], 'b--')
                i += 2
                continue

        ax.plot([points[i, 0], points[i + 1, 0]], [points[i, 1], points[i + 1, 1]], 'b--')
        i += 1

    ax.set_title("Optimized Geometry with True Fillet Arcs (Corrected)")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.grid(True)
    plt.tight_layout()
    plt.show()
    
    
def compute_fillet_arc_safe(p1, p2, p3, radius, num_points=50):
    """
    Safe version of fillet arc computation with fallback in colinear cases.
    """
    # Vectors
    v1 = p1 - p2
    v2 = p3 - p2
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 == 0 or norm_v2 == 0:
        return np.array([]), np.array([])  # Degenerate case

    v1 /= norm_v1
    v2 /= norm_v2

    dot_product = np.clip(np.dot(v1, v2), -1.0, 1.0)
    angle = np.arccos(dot_product) / 2

    if angle == 0:
        return np.array([]), np.array([])  # No angle to fillet

    dist_to_center = radius / np.sin(angle)

    bisector = v1 + v2
    norm_bisector = np.linalg.norm(bisector)
    if norm_bisector == 0:
        return np.array([]), np.array([])

    bisector /= norm_bisector
    center = p2 + bisector * dist_to_center

    start_vector = p1 - center
    end_vector = p3 - center
    start_angle = np.arctan2(start_vector[1], start_vector[0])
    end_angle = np.arctan2(end_vector[1], end_vector[0])

    # Determine correct sweep
    theta = np.linspace(start_angle, end_angle, num_points)
    if np.cross(v1, v2) < 0:
        theta = np.linspace(start_angle, end_angle, num_points)
    else:
        theta = np.linspace(end_angle, start_angle, num_points)

    arc_x = center[0] + radius * np.cos(theta)
    arc_y = center[1] + radius * np.sin(theta)

    return arc_x, arc_y

def plot_nurbs_vs_fillet_geometry_true_arcs_fixed(X, r1_opt, r2_opt):
    d1, d2, d3, d4, d5 = X[:5]
    total_length = sum([d1, d2, d3, d4, d5])
    norm = lambda x: x / total_length

    AAA = np.array([0,
                    norm(d1),
                    norm(d1 + d2 / 2),
                    norm(d1 + d2),
                    norm(d1 + d2 + d3 / 2),
                    norm(d1 + d2 + d3),
                    norm(d1 + d2 + d3 + d4 / 2),
                    norm(d1 + d2 + d3 + d4),
                    1])
    BBB = np.array([0, 0, 0.5, 1, 1, 1, 0.5, 0, 0])

    lambda_ = 5.65e-3  # m
    Amp = 2.65e-3      # m

    points = np.column_stack((AAA * lambda_, BBB * Amp))

    fig, ax = plt.subplots(figsize=(12, 5))

    fillet_indices = [2, 4, 6, 8]
    i = 0
    while i < len(points) - 1:
        if (i + 1 in fillet_indices) and (i > 0 and i + 2 < len(points)):
            p_start = points[i]
            p_corner = points[i + 1]
            p_end = points[i + 2]
            radius = r1_opt / 1000 if i + 1 in [2, 8] else r2_opt / 1000
            arc_x, arc_y = compute_fillet_arc_safe(p_start, p_corner, p_end, radius)

            if arc_x.size > 0:
                arc_start = np.array([arc_x[0], arc_y[0]])
                arc_end = np.array([arc_x[-1], arc_y[-1]])
                ax.plot([p_start[0], arc_start[0]], [p_start[1], arc_start[1]], 'b--')
                ax.plot(arc_x, arc_y, color='orange')
                ax.plot([arc_end[0], p_end[0]], [arc_end[1], p_end[1]], 'b--')
                i += 2
                continue

        ax.plot([points[i, 0], points[i + 1, 0]], [points[i, 1], points[i + 1, 1]], 'b--')
        i += 1

    ax.set_title("Optimized Geometry with True Fillet Arcs (Corrected)")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.grid(True)
    plt.tight_layout()
    plt.show()

def process_csv(input_file, output_csv, output_xlsx):
    with open(input_file, 'r') as infile:
        reader = csv.reader(infile)
        data = []

        # Skip the first row
        next(reader, None)

        # Extended headers
        headers = [
            "Mass/ECT [m^-1]", "Mass/FCT [mm^2/Pa]",
            "d1 [mm]", "d2 [mm]", "d3 [mm]", "d4 [mm]", "d5 [mm]", "X5", "X6",
            "w1", "w2", "r1 [mm]", "r2 [mm]",
            "ECT [m^3 x 10^-6]", "Mass [mm^2]", "FCT [Pa]", "Inc_bin"
        ]

        for row in reader:
            try:
                design_vector_str = clean_np_float_strings(row[2])
                design_vector = ast.literal_eval(design_vector_str)

                if isinstance(design_vector, list) and len(design_vector) >= 7:
                    d1, d2, d3, d4, d5, X5, X6 = design_vector[:7]
                    r1 = 0.0103 * np.exp(9.17 * X5) + 0.9897
                    r2 = 0.0103 * np.exp(9.17 * X6) + 0.9897

                    # Optimize and get the optimal values
                    r1_opt, r2_opt, X5_opt, X6_opt = full_opt_process([d1, d2, d3, d4, d5, X5, X6])

                    new_row = (
                        row[0:2] +                # First two columns
                        [d1, d2, d3, d4, d5, X5, X6] +
                        [r1, r2, r1_opt, r2_opt] +
                        row[3:]                   # Remaining columns
                    )
                    data.append(new_row)
                else:
                    row.extend(["Error"] * 6)
                    data.append(row)

            except Exception as e:
                row.extend(["Error"] * 6)
                data.append(row)

    # Save the output
    df = pd.DataFrame(data, columns=headers)
    df.to_csv(output_csv, index=False)
    df.to_excel(output_xlsx, index=False, sheet_name="Processed Data")
    print(f"Processed CSV saved as {output_csv}")
    print(f"Processed XLSX saved as {output_xlsx}")

# Example usage
roots = ["Prob_prod_fix/EPSO_v5_WC"]

for rooti in roots:
    input_csv = rooti + "/Iteration_99.csv"  # Replace with your input CSV file
    output_csv = rooti + "/output.csv"  # Desired output file name
    output_xlsx = rooti + "/output.xlsx"  # Desired output file name
    process_csv(input_csv, output_csv, output_xlsx)

