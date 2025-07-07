import numpy as np
import sympy as sp
from scipy.linalg import solve
from geomdl import NURBS
from sympy import symbols, atan, lambdify, Matrix
from scipy.interpolate import interp1d
from sympy import symbols, Matrix, cos, sin, lambdify, atan
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt
from numpy import trapz
from scipy.ndimage import gaussian_filter1d

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

def full_opt_process(X):
    # Optimize r1 and r2 weights
    r1_opt_weight, r2_opt_weight = optimize_r1_r2_given_X(X)
    return r1_opt_weight, r2_opt_weight, X[5], X[6]

def opt_calc_prod(X):
    """X = [thickness liner, thickness flute, amplitude, wavelenght, ...
            d1, d2, d3, d4, d5, r1, r2]
    
    """
    
    # Define starting points
    start_points = np.array([[0, 0], [1, 0]])
    decide_point = np.array([1, 1])
    
    # Total number of points to generate
    num_points = 9
    # num_points = 5
    distance = 1
    
    # Periodic length and amplitude
    lambda_ = 5.65e-3
    Amp = 2.65e-3
    
    suma = X[0] + X[1] + X[2] + X[3] +  X[4]
    d1 = X[0]/ suma #0.5
    d2 = X[1]/ suma #0.1
    d3 = X[2]/ suma #0.4
    d4 = X[3]/ suma #0.1
    d5 = X[4]/ suma #0.1
    r1 = 0.0103*np.exp(9.17*X[5])+0.1
    r2 = 0.0103*np.exp(9.17*X[6])+0.1
    
    r1_opt, r2_opt, X5_opt, X6_opt = full_opt_process([d1, d2, d3, d4, d5, X[5], X[6]])
    

    constr = True if r1_opt > 0.9 and r2_opt > 0.9 else True  
    
    # Material properties
    E1 = 1e9  # Young's modulus MD [Pa]
    E2 = E1 / 2  # Young's modulus CD [Pa]
    t = 0.0002  # Paper thickness [m]
    epsilon = 1e-6
    E3 = E1 / 190  # (Mann et al., 1979)
    
    # Calculated properties
    G12 = 0.387 * np.sqrt(E1 * E2)  # (Baum, 1981)
    nu12 = 0.293 * np.sqrt(E1 / E2)
    G13 = E1 / 55  # (Baum et al., 1981)
    G23 = E2 / 35  # (Baum et al., 1981)
    nu13 = 0.001  # (Nordstrand, 1995)
    nu23 = 0.001  # (Nordstrand, 1995)
    
    # Compliance matrix
    C123 = np.array([
        [1 / E1, -nu12 / E1, -nu13 / E1, 0, 0, 0],
        [-nu12 / E1, 1 / E2, -nu23 / E2, 0, 0, 0],
        [-nu13 / E1, -nu23 / E2, 1 / E3, 0, 0, 0],
        [0, 0, 0, 1 / G12, 0, 0],
        [0, 0, 0, 0, 1 / G13, 0],
        [0, 0, 0, 0, 0, 1 / G23]
    ])
    
    i = 1
    N = 1
    results = np.full((N, 5), np.nan)
    results1 = np.full((N, num_points * 3 + 2), np.nan)
    ep = 10**-6
    AAA = [0, d1, d1+d2, d1+d2+d3, d1+d2+d3+d4, 1 + d1, 1+d1+d2, 1+d1+d2+d3, 1+d1+d2+d3+d4, 2]  
    # AAA = [0, lambda_/2 - ep, lambda_/2, lambda_ - ep, lambda_]
    BBB = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0]  
    # BBB = [1, 1, 0, 0, 1]
    
    # Assign weights
    weights = np.array([1, r1, r1, r2, r2, r1, r1, r2, r2, 1])
    
    # Combine into control points array
    control_points1 = np.column_stack((AAA, BBB))
    
    # Create NURBS curve
    curve = NURBS.Curve()
    
    degree = curve.degree = 2
    curve.ctrlpts = control_points1.tolist()
    curve.weights = weights.tolist()
    num_ctrlpts = len(control_points1)
    degree = curve.degree
    num_knots = num_ctrlpts + degree + 1
    
    # Corrected knot vector with uniform spacing
    interior_knots = np.linspace(0, 1, num_knots - 2 * (degree + 1) + 2)[1:-1]  # Remove first & last values to avoid duplicates
    
    curve.knotvector = np.concatenate([
        np.zeros(degree + 1),  # Leading zeros
        interior_knots,  # Uniform internal knots
        np.ones(degree + 1)  # Ending ones
    ]).tolist()
    
    curve.sample_size = 2000
    # Evaluate NURBS curve
    curve.evaluate()
    curve_points = np.array(curve.evalpts)
    
    n_points = len(curve_points[:,0])
    t1 = int(( (d1+d2+d3/2 )/ 2) * n_points)  # end of r1 zone
    t2 = int(((1+d1+d2+d3/2) / 2) * n_points)            # end of full profile (r2 zone ends here)
    
    
    # Compute scaling factors
    l = abs(curve_points[t2, 0] - curve_points[t1, 0])
    scale1 = lambda_ / l
    height = np.max(curve_points[t1:t2,1]) - np.min(curve_points[t1:t2,1])
    scale2 = Amp / height
    
    
    # Scale curve points
    local_curve0 = scale1 * curve_points[t1:t2,0]
    local_curve1 = scale2 * curve_points[t1:t2,1]
    local_curve = np.vstack([local_curve0, local_curve1])
    
    # Find segments (mock function, replace with actual logic)
    def find_segment_points(curve):
        return np.array([0, len(curve[0]) - 1]), curve
    
    indice, segments = find_segment_points(local_curve)
    
    # Polynomial fitting and material property calculation
    A = np.zeros((3, 3))
    D = np.zeros((3, 3))
    F = np.zeros((2, 2))
    
    x_sym = symbols('x')
    
    for j in range(len(indice) - 1):
        
        # Extract local segment
        start_idx = indice[j]
        end_idx = indice[j + 1]
        x1 = local_curve[0, start_idx:end_idx]
        y1 = local_curve[1, start_idx:end_idx]
        
        plt.plot(local_curve[0, :], local_curve[1, :], color='black', linewidth=10)
        
        # Compute segment length
        l1 = abs(x1[-1] - x1[0])  # Not needed explicitly if we use x1 directly
        
        H_interp = interp1d(x1, y1, kind="cubic", fill_value="extrapolate")  # Interpolates H(x)
    
        dy_dx_vals = np.gradient(y1, x1)
        
        # Create interpolation function for smooth dy/dx
        dy_dx_interp = interp1d(x1, dy_dx_vals, kind="cubic", fill_value="extrapolate")
        
        # Define symbolic variable
        x_sym = symbols('x')
        
        H_func = np.vectorize(lambda x: float(H_interp(float(x))))  # H(x), not dy/dx
        theta_func = lambda x: np.arctan(dy_dx_interp(x))  # Correct derivative
        
        theta_vals = theta_func(x1)
        
        tv_vals = t / np.cos(theta_vals)
        d_vals = H_func(x1) ** 2 * tv_vals + tv_vals ** 3 / 12
    
        # Material compliance matrix
        C123 = Matrix([[1 / E1, -nu12 / E1, -nu13 / E1, 0, 0, 0],
                       [-nu12 / E1, 1 / E2, -nu23 / E2, 0, 0, 0],
                       [-nu13 / E1, -nu23 / E2, 1 / E3, 0, 0, 0],
                       [0, 0, 0, 1 / G12, 0, 0],
                       [0, 0, 0, 0, 1 / G13, 0],
                       [0, 0, 0, 0, 0, 1 / G23]])
    
        # Transformation matrices for rotation about y-axis
        theta = symbols('theta')
        c, s = cos(theta), sin(theta)
    
        Te_sym  = Matrix([[c**2, 0, s**2, 0, -s*c, 0],
                     [0, 1, 0, 0, 0, 0],
                     [s**2, 0, c**2, 0, s*c, 0],
                     [0, 0, 0, c, 0, -s],
                     [2*s*c, 0, -2*s*c, 0, c**2 - s**2, 0],
                     [0, 0, 0, -s, 0, c]])
    
        Ts_sym  = Matrix([[c**2, 0, s**2, 0, 2*s*c, 0],
                     [0, 1, 0, 0, 0, 0],
                     [s**2, 0, c**2, 0, -2*s*c, 0],
                     [0, 0, 0, c, 0, s],
                     [-s*c, 0,  s*c, 0, c**2 - s**2, 0],
                     [0, 0, 0, -s, 0, c]])
        
        # Convert to numerical functions
        Te_func = lambdify(theta, Te_sym, "numpy")
        Ts_func = lambdify(theta, Ts_sym, "numpy")
        
        # Compute `Te`, `Ts` matrices for all `theta_vals`
        Te_vals = np.array([Te_func(th) for th in theta_vals])
        Ts_vals = np.array([Ts_func(th) for th in theta_vals])
        
        # Compute Cxyz dynamically
        Cxyz_vals = np.array([
            np.array(Te_vals[i]) @ np.array(C123.evalf()).astype(np.float64) @ np.array(Ts_vals[i])
            for i in range(len(x1))
        ])
        
        # Extract Q and G dynamically at each `x1` value
        Q_vals = np.linalg.inv(Cxyz_vals[:, :3, :3])
        G_vals = np.linalg.inv(Cxyz_vals[:, 3:, 3:])
        
        # Interpolation of `Q` and `G` (so they can be evaluated dynamically per `x`)
        Q_interp_funcs = [[interp1d(x1, Q_vals[:, i, j], kind="cubic", fill_value="extrapolate") for j in range(3)] for i in range(3)]
        G_interp_funcs = [[interp1d(x1, G_vals[:, i, j], kind="cubic", fill_value="extrapolate") for j in range(2)] for i in range(2)]
        
        tv_interp = interp1d(x1, tv_vals, kind="cubic", fill_value="extrapolate")
        d_interp = interp1d(x1, d_vals, kind="cubic", fill_value="extrapolate")
    
            
        # Define interpolation-based functions
        A_funcs = [[lambda x_val, i=i, j=j: float(tv_interp(x_val) * Q_interp_funcs[i][j](x_val)) for j in range(3)] for i in range(3)]
        D_funcs = [[lambda x_val, i=i, j=j: float(d_interp(x_val) * Q_interp_funcs[i][j](x_val)) for j in range(3)] for i in range(3)]
        F_funcs = [[lambda x_val, i=i, j=j: float(tv_interp(x_val) * G_interp_funcs[i][j](x_val)) for j in range(2)] for i in range(2)]
    
    
        for i in range(3):  
            for j in range(3):
                A[i, j] += np.trapz([A_funcs[i][j](xi) for xi in x1], x1) / lambda_
                D[i, j] += np.trapz([D_funcs[i][j](xi) for xi in x1], x1) / lambda_
        
        for i in range(2):
            for j in range(2):
                F[i, j] += np.trapz([F_funcs[i][j](xi) for xi in x1], x1) / lambda_
    
    
    # ABD Matrix Components (Corrected)
    As = np.linalg.inv(A)
    Bs = -np.linalg.inv(A) @ np.zeros((3, 3))  # Since B is zero
    Ds = D - np.zeros((3, 3)) @ Bs
    
    Ai = As - (Bs @ np.linalg.inv(Ds) @ Bs)
    Bi = Bs @ np.linalg.inv(Ds)
    Di = np.linalg.inv(Ds)
    Fi = np.linalg.inv(F)
    
    # Effective thickness (Marek & Garbowski, 2015)
    th = np.sqrt(12 * np.sum(D.diagonal()) / np.sum(A.diagonal()))
    
    Eye = 1 / (th * Ai[1, 1])
    Exe = 1 / (th * Ai[0, 0])
    Ez = 12 / (th ** 3 * Di[2, 2])
    
    Ezeff = (Ez * th + E3 * 2 * t) / (th + 2 * t)
    Eyeff = (Eye * th + E2 * 2 * t) / (th + 2 * t)
    Exeff = (Exe * th + E1 * 2 * t) / (th + 2 * t)
    
    # Extract x and y coordinates
    x = local_curve[0, :]
    y = local_curve[1, :]

    # Compute differential distances in x and y
    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.sqrt(dx**2 + dy**2)  # Arc length differential

    # Compute total curve length using Euclidean distance
    curve_length_integral = np.sum(ds)
    total_x_length = np.max(x) - np.min(x)

    # Compute expected weight per unit length (Area per unit length)
    expected_weight = (curve_length_integral * t) / total_x_length + 2*t

    # Compute centroid (center of mass per unit length) of the fluted core
    A_core = np.sum(ds * t)  # Area of fluted core
    y_bar_core = np.sum(y[:-1] * ds * t) / A_core  # Centroid of fluted core

    # Compute liner placement positions
    y_liner_top = np.max(y) + t  # Centroid of top liner
    y_liner_bottom = np.min(y) - t  # Centroid of bottom liner

    # Compute liner areas
    A_liner = total_x_length * t  # Area of one liner
    A_total = A_core + 2 * A_liner  # Total area

    # Compute overall centroid including liners
    y_bar = (A_core * y_bar_core + A_liner * y_liner_top + A_liner * y_liner_bottom) / A_total

    # Compute second moment of area for the fluted core relative to centroid
    I_core = np.sum((y[:-1] - y_bar)**2 * ds * t)

    # Compute correct liner placement distances from the overall centroid
    d_liner_top = abs(y_liner_top - y_bar)
    d_liner_bottom = abs(y_liner_bottom - y_bar)

    # Compute second moment of area for each liner using Parallel Axis Theorem
    W = total_x_length  # Assume unit width
    I_liner_top = (W * t**3) / 12 + (W * t) * d_liner_top**2
    I_liner_bottom = (W * t**3) / 12 + (W * t) * d_liner_bottom**2

    # Total inertia per unit length
    inertia = (I_core + I_liner_top + I_liner_bottom) / total_x_length

    # Compute efficiency metric: Inertia-to-Area Ratio (I/A)
    efficiency = inertia * 10**6 / A_total
    
    return 1/ inertia * 10**(-6), A_total, Ezeff, constr
