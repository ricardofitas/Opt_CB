import numpy as np
import sympy as sp
from scipy.linalg import solve
from geomdl import NURBS
from sympy import symbols, atan, lambdify, Matrix
from scipy.interpolate import interp1d
from sympy import symbols, Matrix, cos, sin, lambdify, atan
from scipy.integrate import trapezoid
from shapely.geometry import LineString
from matplotlib import pyplot as plt

def opt_calc_prod(X):
    
    """X = [thickness liner, thickness flute, amplitude, wavelenght, ...
            d1, d2, d3, d4, d5, r1, r2]
    
    """
    
    # Define starting points
    start_points = np.array([[0, 0], [1, 0]])
    decide_point = np.array([1, 1])
    
    # Total number of points to generate
    
    # num_points = 5
    distance = 1
    
    # Periodic length and amplitude
    lambda_ = 5.65e-3
    Amp = 2.65e-3
       
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
    
    
    def extract_vectors(X):
        # Extract x, y, and w values
        x_values = X[1::3]  # x2, x3, ..., xn
        y_values = X[0::3]  # y1, y2, ..., yn
        w_values = X[2::3]  # w2, w3, ..., wn
        
        # Normalize x values
        max_x = sum(x_values) + 1/len(x_values)
        # AAA = [0] + [x / max_x for x in x_values] + [1]
        AAA = [0] + [sum(x_values[0:(i+1)]) / max_x for i, k in enumerate(x_values)] + [1]
        
        # Construct BBB vector
        BBB = y_values + [y_values[0]]
        
        # Construct W vector
        W = np.array([1] + w_values + [1], dtype=float)
        
        # Convert W to Wr
        Wr = 0.0103 * np.exp(9.17 * W) + 0.9897
        Wr = Wr.tolist()
        
        num_points = len(x_values)
        
        print(AAA, BBB)
    
        return AAA, BBB, Wr, num_points

    # Example usage
    AAA, BBB, W, num_points = extract_vectors(X)

    # Assign weights
    weights = np.array(W)
    
    # Combine into control points array
    control_points1 = np.column_stack((AAA, BBB))
    
    results = np.full((N, 5), np.nan)
    results1 = np.full((N, num_points * 3 + 2), np.nan)
    
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
    
    # Compute scaling factors
    l = abs(control_points1[-1, 0] - control_points1[0, 0])
    scale1 = lambda_ / l
    height = np.max(curve_points[:, 1]) - np.min(curve_points[:, 1])
    scale2 = Amp / height
    
    # Scale curve points
    local_curve = np.zeros_like(curve_points.T)
    local_curve[0, :] = scale1 * curve_points[:, 0]
    local_curve[1, :] = scale2 * curve_points[:, 1]
    
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
        
        def remove_duplicates(x, y):
            """
            Removes duplicate x-values while preserving order, keeping the first occurrence.
            
            Parameters:
            x (array-like): x-values
            y (array-like): corresponding y-values
        
            Returns:
            tuple: (filtered_x, filtered_y)
            """
            unique_x, indices = np.unique(x, return_index=True)  # Get unique values and their indices
            sorted_indices = np.sort(indices)  # Ensure indices are in the correct order
            return x[sorted_indices], y[sorted_indices]  # Filter x and y
        
        x1, y1 = remove_duplicates(np.array(x1), np.array(y1))
        
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
        epsilon = 1e-8  # Small regularization term
        Q_vals = np.linalg.inv(Cxyz_vals[:, :3, :3] + epsilon * np.eye(3)[None, :, :])
        epsilon = 1e-8  # Small value
        G_matrices = Cxyz_vals[:, 3:, 3:]
        identity_matrix = np.eye(G_matrices.shape[1])
        identity_matrix = identity_matrix[None, :, :]
        G_vals = np.linalg.inv(G_matrices + epsilon * identity_matrix)
        
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
                A[i, j] += np.trapezoid([A_funcs[i][j](xi) for xi in x1], x1) / lambda_
                D[i, j] += np.trapezoid([D_funcs[i][j](xi) for xi in x1], x1) / lambda_
        
        for i in range(2):
            for j in range(2):
                F[i, j] += np.trapezoid([F_funcs[i][j](xi) for xi in x1], x1) / lambda_
    
    
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
    
    def check_self_intersection(curve_points):
        """
        Check if a 2D curve (set of points) intersects itself.
    
        Parameters:
        curve_points (np.ndarray): A (N, 2) array representing N points in 2D space.
    
        Returns:
        bool: True if the curve self-intersects, False otherwise.
        """
        curve_points = np.array(curve_points)  # Ensure it's a NumPy array
    
        print("Shape of curve_points:", curve_points.shape)  # Debugging step
    
        if curve_points.shape[1] != 2:
            raise ValueError(f"Expected (N,2) array for LineString, got {curve_points.shape}")
    
        line = LineString(curve_points)
        return line.is_simple  # is_simple is False if the line self-intersects
    
    is_not_intersecting = check_self_intersection(local_curve.T)
    
    if ((inertia * 10**6 < 0) or (A_total < 0) or (Ezeff < 0) or 
        (not np.isreal(inertia * 10**6)) or (not np.isreal(A_total)) or (not np.isreal(Ezeff)) or
        np.isnan(inertia * 10**6) or np.isnan(A_total) or np.isnan(Ezeff)):
        is_not_intersecting = False
     
    
    return inertia * 10**6, A_total, Ezeff, is_not_intersecting
