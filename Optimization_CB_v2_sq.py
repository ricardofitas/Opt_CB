import numpy as np
import sympy as sp
from scipy.linalg import solve
from geomdl import NURBS
from sympy import symbols, atan, lambdify, Matrix
from scipy.interpolate import interp1d
from sympy import symbols, Matrix, cos, sin, lambdify, atan
from scipy.integrate import trapezoid

# Define starting points
start_points = np.array([[0, 0], [1, 0]])
decide_point = np.array([1, 1])

# Total number of points to generate
num_points = 7
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
results = np.full((N, 5), np.nan)
results1 = np.full((N, num_points * 3 + 2), np.nan)

w_scale = 100
ep = 10**-6
AAA = [0, lambda_/2 - ep/2, lambda_/2, lambda_/2 + ep/2, lambda_ - ep, lambda_ - ep/2, lambda_]  # X-coordinates evenly spaced
# AAA = [0, lambda_/2 - ep, lambda_/2, lambda_ - ep, lambda_]
BBB = [0, 0, -1, 0, 0, 1, 0]  # Y-coordinates as sine wave
# BBB = [1, 1, 0, 0, 1]

# Assign weights
weights = np.ones(num_points) * w_scale
weights[0] = 1  # Lower weight for valleys
weights[-1] = 1  # Higher weight for peaks

# Combine into control points array
control_points1 = np.column_stack((AAA, BBB))

# Create NURBS curve
curve = NURBS.Curve()

degree = curve.degree = 1
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

curve.sample_size = 20000
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
