import numpy as np
from scipy.integrate import quad
from sympy import symbols, Matrix, cos, sin, atan

# Define Variables
X = np.array([0.2, 0.2, 5.65, 2.65]) * 0.001

E1 = 1e9      # Young's modulus MD [Pa]
E2 = E1 / 2   # Young's modulus CD [Pa]
f_start = X[0]   # Paper thickness [m]
p = X[1]      # Flute height [m]
t = X[2]      # Flute wavelength [m]
h = f_start + t  # Core height [m]
epsilon = 1e-6

E3 = E1 / 190  # (Mann et al., 1979)

# Calculated properties
G12 = 0.387 * np.sqrt(E1 * E2)  # (Baum, 1981)
nu12 = 0.293 * np.sqrt(E1 / E2)
G13 = E1 / 55  # (Baum et al., 1981)
G23 = E2 / 35  # (Baum et al., 1981)
nu13 = 0.001   # (Nordstrand, 1995)
nu23 = 0.001   # (Nordstrand, 1995)

# Flute Geometry
x = symbols('x')
a = 2 * np.pi / p
b = np.pi * f_start / p
H = (f_start / 2) * sin(a * x)  # Flute profile
theta = atan(b * cos(a * x))  # Rotation angle at x

# Material Compliance Matrix
C123 = Matrix([[1 / E1, -nu12 / E1, -nu13 / E1, 0, 0, 0],
               [-nu12 / E1, 1 / E2, -nu23 / E2, 0, 0, 0],
               [-nu13 / E1, -nu23 / E2, 1 / E3, 0, 0, 0],
               [0, 0, 0, 1 / G12, 0, 0],
               [0, 0, 0, 0, 1 / G13, 0],
               [0, 0, 0, 0, 0, 1 / G23]])

# Transformation Matrices
c = cos(theta)
s = sin(theta)

Te = Matrix([[c**2, 0, s**2, 0, -s*c, 0],
             [0, 1, 0, 0, 0, 0],
             [s**2, 0, c**2, 0, s*c, 0],
             [0, 0, 0, c, 0, -s],
             [2*s*c, 0, -2*s*c, 0, c**2 - s**2, 0],
             [0, 0, 0, -s, 0, c]])

Ts = Matrix([[c**2, 0, s**2, 0, 2*s*c, 0],
             [0, 1, 0, 0, 0, 0],
             [s**2, 0, c**2, 0, -2*s*c, 0],
             [0, 0, 0, c, 0, -s],
             [-s*c, 0, s*c, 0, c**2 - s**2, 0],
             [0, 0, 0, -s, 0, c]])

# Rotated Compliance Matrix
Cxyz = (Te @ C123 @ Ts).evalf()

Q_sym = Cxyz[:3, :3].inv() 
G_sym = Cxyz[3:, 3:].inv()

# Vertical Thickness of Fluting
tv = t / cos(theta)

# Homogenization Through Thickness
A11 = lambda x_val: float((tv * Q_sym[0, 0]).subs(x, x_val).evalf())
A22 = lambda x_val: float((tv * Q_sym[1, 1]).subs(x, x_val).evalf())
A33 = lambda x_val: float((tv * Q_sym[2, 2]).subs(x, x_val).evalf())

D11 = lambda x_val: float(((H**2 * tv + tv**3 / 12).subs(x, x_val).evalf() * Q_sym[0, 0].subs(x, x_val).evalf()))
D22 = lambda x_val: float(((H**2 * tv + tv**3 / 12).subs(x, x_val).evalf() * Q_sym[1, 1].subs(x, x_val).evalf()))
D33 = lambda x_val: float(((H**2 * tv + tv**3 / 12).subs(x, x_val).evalf() * Q_sym[2, 2].subs(x, x_val).evalf()))

F11 = lambda x_val: float((tv * G_sym[0, 0]).subs(x, x_val).evalf())
F22 = lambda x_val: float((tv * G_sym[1, 1]).subs(x, x_val).evalf())

# Homogenization Along x-Direction
A = np.zeros((3, 3))
D = np.zeros((3, 3))
F = np.zeros((2, 2))

A[0, 0] = (1 / p) * quad(A11, 0, p)[0]
A[1, 1] = (1 / p) * quad(A22, 0, p)[0]
A[2, 2] = (1 / p) * quad(A33, 0, p)[0]

D[0, 0] = (1 / p) * quad(D11, 0, p)[0]
D[1, 1] = (1 / p) * quad(D22, 0, p)[0]
D[2, 2] = (1 / p) * quad(D33, 0, p)[0]

F[0, 0] = (1 / p) * quad(F11, 0, p)[0]
F[1, 1] = (1 / p) * quad(F22, 0, p)[0]

# ABD Matrix Components
As = np.linalg.inv(A)
Bs = -np.linalg.inv(A) @ np.zeros((3, 3))  # B is zero
Ds = D - np.zeros((3, 3)) @ Bs

Ai = As - (Bs @ np.linalg.inv(Ds) @ Bs)
Bi = Bs @ np.linalg.inv(Ds)
Di = np.linalg.inv(Ds)
Fi = np.linalg.inv(F)

# Effective Thickness (Marek & Garbowski, 2015)
th = np.sqrt(12 * np.sum(D.diagonal()) / np.sum(A.diagonal()))

Ez = 12 / (th ** 3 * Di[2, 2])  # Correct component for Ez
