import numpy as np
from scipy.optimize import minimize
import math
import scipy.integrate as spi
from sympy import symbols, sin, cos, atan, pi, Matrix, simplify, integrate
from scipy.integrate import quad

def expected_arc_length(A, lambda_, t):
        omega = 2 * np.pi / lambda_

        def integrand(x):
            return np.sqrt(1 + (A * omega * np.cos(omega * x)) ** 2)

        # Calculate the integral from 0 to t
        arc_length, _ = spi.quad(integrand, 0, t)

        # Calculate the expected value per unit x
        expected_value = arc_length / t

        return expected_value

def weight(X):


    L = X[0]*2



    # Example usage
    A = X[3]/2  # Amplitude
    lambda_ = X[2]  # Wavelength
    t = 1

    expected_value = expected_arc_length(A, lambda_, t)
    total_area = X[1]*expected_value + L

    return total_area


def calculate_ez(X):

    X = [i*0.001 for i in X]
    
    t = X[1]
    p = X[2]
    f_start = X[3]
    # Constants
    #E1 = 1.709e9     # Young's modulus MD [Pa]
    #E2 = 0.918e9     # Young's modulus CD [Pa]
    #E3 = E1/190  # E1 / 190

    h = f_start + t        # Core height [m]

    E1 = 7e9
    E2 = E1 / 2
    E3 = E1 / 190


    G12 = 0.387 * np.sqrt(E1 * E2)     # Baum, 1981
    nu12 = 0.293 * np.sqrt(E1 / E2)
    G13 = E1 / 55    # Baum et al., 1981
    G23 = E2 / 35    # Baum et al., 1981
    nu13 = 0.001     # Nordstrand, 1995
    nu23 = 0.001     # Nordstrand, 1995

    # Flute geometry
    x = symbols('x')
    a = 2 * pi / p
    b = pi * f_start / p
    H = (f_start / 2) * sin(a * x)        # Flute profile
    theta = atan(b * cos(a * x))    # Rotation angle at x

    # Material compliance matrix
    C123 = Matrix([[1 / E1, -nu12 / E1, -nu13 / E1, 0, 0, 0],
                   [-nu12 / E1, 1 / E2, -nu23 / E2, 0, 0, 0],
                   [-nu13 / E1, -nu23 / E2, 1 / E3, 0, 0, 0],
                   [0, 0, 0, 1 / G12, 0, 0],
                   [0, 0, 0, 0, 1 / G13, 0],
                   [0, 0, 0, 0, 0, 1 / G23]])

    # Transformation matrices for rotation about y-axis
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
                 [0, 0, 0, c, 0, s],
                 [-s*c, 0,  s*c, 0, c**2 - s**2, 0],
                 [0, 0, 0, -s, 0, c]])

    # Rotated compliance matrix
    Cxyz = simplify(Te * C123 * Ts)

    # Reduced rotated stiffness matrices
    Cxyz_num = Cxyz.subs(x, 0).evalf()
    Q = np.linalg.inv(np.array(Cxyz_num[:3, :3]).astype(np.float64))
    G = np.linalg.inv(np.array(Cxyz_num[3:, 3:]).astype(np.float64))

    # Vertical thickness of fluting
    tv = t / cos(theta)

    z = H
    d = H**2 * tv + tv**3 / 12

    # Homogenization through the thickness (z-direction)
    A11 = lambda x_val: float(tv.subs(x, x_val) * Q[0, 0])
    A12 = lambda x_val: float(tv.subs(x, x_val) * Q[0, 1])
    A21 = lambda x_val: float(tv.subs(x, x_val) * Q[1, 0])
    A22 = lambda x_val: float(tv.subs(x, x_val) * Q[1, 1])
    A33 = lambda x_val: float(tv.subs(x, x_val) * Q[2, 2])
    D11 = lambda x_val: float(d.subs(x, x_val) * Q[0, 0])
    D12 = lambda x_val: float(d.subs(x, x_val) * Q[0, 1])
    D21 = lambda x_val: float(d.subs(x, x_val) * Q[1, 0])
    D22 = lambda x_val: float(d.subs(x, x_val) * Q[1, 1])
    D33 = lambda x_val: float(d.subs(x, x_val) * Q[2, 2])
    F11 = lambda x_val: float(tv.subs(x, x_val) * G[0, 0])
    F22 = lambda x_val: float(tv.subs(x, x_val) * G[1, 1])

    # Homogenization along the MD (x-direction)
    A = np.zeros((3, 3))
    D = np.zeros((3, 3))
    F = np.zeros((2, 2))
    A[0, 0] = (1 / p) * quad(A11, 0, p)[0]
    A[0, 1] = (1 / p) * quad(A12, 0, p)[0]
    A[1, 0] = (1 / p) * quad(A21, 0, p)[0]
    A[1, 1] = (1 / p) * quad(A22, 0, p)[0]
    A[2, 2] = (1 / p) * quad(A33, 0, p)[0]
    D[0, 0] = (1 / p) * quad(D11, 0, p)[0]
    D[0, 1] = (1 / p) * quad(D12, 0, p)[0]
    D[1, 0] = (1 / p) * quad(D21, 0, p)[0]
    D[1, 1] = (1 / p) * quad(D22, 0, p)[0]
    D[2, 2] = (1 / p) * quad(D33, 0, p)[0]
    F[0, 0] = (1 / p) * quad(F11, 0, p)[0]
    F[1, 1] = (1 / p) * quad(F22, 0, p)[0]

    # Partially inverted components of ABD (Gibson, 2012)
    As = np.linalg.inv(A)
    Bs = -np.linalg.inv(A) @ np.zeros((3, 3))  # Since B is zero
    Ds = D - np.zeros((3, 3)) @ Bs

    # Fully inverted components of ABD (Gibson, 2012)
    Ai = As - (Bs @ np.linalg.inv(Ds) @ Bs)
    Bi = Bs @ np.linalg.inv(Ds)
    Di = np.linalg.inv(Ds)
    Fi = np.linalg.inv(F)

    # Effective thickness (Marek & Garbowski, 2015)
    th = np.sqrt(12 * (D[0, 0] + D[1, 1] + D[2, 2]) / (A[0, 0] + A[1, 1] + A[2, 2]))

    Eye = 1 / (th*Ai[1, 1])
    Ez = 12 / (th**3 * Di[2, 2])  # Correct component for Ez
    Ezeff = (Ez*th + E3*2*X[0])/(th + 2*X[0])
    Eyeff = (Eye*th + E2*2*X[0])/(th + 2*X[0])
    return Ezeff, Eyeff, th

def critical_stress(X):

    X = [i*0.001 for i in X]

    cc = True

    def gihi(D, lambda_a, a, c1, c2, c3, h, mu):  # Define the gi and hi values from the coefficients you provided

        D11, D12, D22, D44, D55, D66 = D[0], D[1], D[2], D[3], D[4], D[5]

        gi_values = [
            17 * np.pi ** 6 * c3 ** 2 * D11 ** 2 * D22 * a ** 2 * h ** 7,  # g1
            17 * np.pi ** 6 * c2 * c3 ** 2 * D11 ** 2 * D44 * h ** 7,  # g2
            168 * np.pi ** 4 * c3 ** 2 * D11 ** 2 * D66 * a ** 2 * h ** 5 * lambda_a ** 2,  # g3
            -17 * np.pi ** 6 * c3 * c2 ** 2 * D11 * D12 ** 2 * a ** 2 * h ** 7,  # g4
            34 * np.pi ** 6 * c2 * c3 * D11 * D12 * D22 * a ** 4 * h ** 7,  # g5
            336 * np.pi ** 4 * c2 * c3 * D11 * D12 * D66 * a ** 4 * h ** 5 * lambda_a ** 2,  # g6
            17 * np.pi ** 6 * c3 * D11 * D22 ** 2 * a ** 6 * h ** 7,  # g7
            102 * np.pi ** 6 * c2 * c3 * D11 * D22 * D44 * a ** 4 * h ** 7,  # g8
            14280 * np.pi ** 4 * c2 * c3 * D11 * D22 * D55 * a ** 4 * h ** 5 * lambda_a ** 2,  # g9
            14280 * np.pi ** 4 * c3 * D11 * D22 * D66 * a ** 6 * h ** 5 * lambda_a ** 2,  # g10
            68 * np.pi ** 6 * c3 * c2 ** 2 * D11 * D44 ** 2 * a ** 2 * h ** 7,  # g11
            14280 * np.pi ** 4 * c3 * c2 ** 2 * D11 * D44 * D55 * a ** 2 * h ** 5 * lambda_a ** 2,  # g12
            14952 * np.pi ** 4 * c2 * c3 * D11 * D44 * D66 * a ** 4 * h ** 5 * lambda_a ** 2,  # g13
            141120 * np.pi ** 2 * c2 * c3 * D11 * D55 * D66 * a ** 4 * h ** 3 * lambda_a ** 4,  # g14
            -34 * np.pi ** 6 * c3 ** 2 * D12 ** 3 * a ** 4 * h ** 7,  # g15
            -17 * np.pi ** 6 * c2 ** 2 * D12 ** 2 * D22 * a ** 6 * h ** 7,  # g16
            -136 * np.pi ** 6 * c3 ** 2 * D12 ** 2 * D44 * a ** 4 * h ** 7,  # g17
            -14112 * np.pi ** 4 * c2 ** 3 * D12 ** 2 * D55 * a ** 4 * h ** 5 * lambda_a ** 2,  # g18
            -14112 * np.pi ** 4 * c2 ** 2 * D12 ** 2 * D66 * a ** 6 * h ** 5 * lambda_a ** 2,  # g19
            336 * np.pi ** 4 * c2 ** 2 * D12 * D22 * D55 * a ** 6 * h ** 5 * lambda_a ** 2,  # g20
            -136 * np.pi ** 6 * c2 ** 3 * D12 * D44 ** 2 * a ** 4 * h ** 7,
            # g21 (seems duplicated from g17, check for correctness)
            -27888 * np.pi ** 4 * c2 ** 3 * D12 * D44 * D55 * a ** 4 * h ** 5 * lambda_a ** 2,  # g22
            -27888 * np.pi ** 4 * c2 ** 2 * D12 * D44 * D66 * a ** 6 * h ** 5 * lambda_a ** 2,  # g23
            282240 * np.pi ** 2 * c2 ** 2 * D12 * D55 * D66 * a ** 6 * h ** 3 * lambda_a ** 4,  # g24
            17 * np.pi ** 6 * c2 * D22 ** 2 * D44 * a ** 8 * h ** 7,  # g25
            168 * np.pi ** 4 * c2 * D22 ** 2 * D55 * a ** 8 * h ** 5 * lambda_a ** 2,  # g26
            68 * np.pi ** 6 * c2 ** 2 * D22 * D44 ** 2 * a ** 6 * h ** 7,  # g27
            14952 * np.pi ** 4 * c2 ** 2 * D22 * D44 * D55 * a ** 6 * h ** 5 * lambda_a ** 2,  # g28
            14280 * np.pi ** 4 * c2 * D22 * D44 * D66 * a ** 8 * h ** 5 * lambda_a ** 2,  # g29
            141120 * np.pi ** 2 * c2 * D22 * D55 * D66 * a ** 8 * h ** 3 * lambda_a ** 2,  # g30
            672 * np.pi ** 4 * c2 ** 3 * D44 ** 2 * D55 * a ** 4 * h ** 5 * lambda_a ** 2,  # g31
            672 * np.pi ** 4 * c2 ** 2 * D44 ** 2 * D66 * a ** 6 * h ** 5 * lambda_a ** 2,  # g32
            564480 * np.pi ** 2 * c2 ** 2 * D44 * D55 * D66 * a ** 6 * h ** 3 * lambda_a ** 4  # g33
        ]

        hi_values = [
            289 * np.pi ** 4 * D11 * D22 * a ** 6 * h ** 4 * mu ** 4,  # h1
            1734 * np.pi ** 4 * D11 * D22 * a ** 4 * h ** 4 * lambda_a ** 2 * mu ** 2,  # h2
            289 * np.pi ** 4 * D11 * D22 * a ** 2 * h ** 4 * lambda_a ** 4,  # h3
            289 * np.pi ** 4 * D11 * D44 * a ** 6 * h ** 4 * mu ** 6,  # h4
            2023 * np.pi ** 4 * D11 * D44 * a ** 4 * h ** 4 * lambda_a ** 2 * mu ** 4,  # h5
            2023 * np.pi ** 4 * D11 * D44 * a ** 2 * h ** 4 * lambda_a ** 4 * mu ** 2,  # h6
            289 * np.pi ** 4 * D11 * D44 * h ** 4 * lambda_a ** 6,  # h7
            2856 * np.pi ** 2 * D11 * D66 * a ** 6 * h ** 2 * lambda_a ** 2 * mu ** 4,  # h8
            17136 * np.pi ** 2 * D11 * D66 * a ** 4 * h ** 2 * lambda_a ** 4 * mu ** 2,  # h9
            2856 * np.pi ** 2 * D11 * D66 * a ** 2 * h ** 2 * lambda_a ** 6,  # h10
            -289 * np.pi ** 4 * D12 ** 2 * a ** 6 * h ** 4 * mu ** 4,  # h11
            -578 * np.pi ** 4 * D12 ** 2 * a ** 4 * h ** 4 * lambda_a ** 2 * mu ** 2,  # h12
            -289 * np.pi ** 4 * D12 ** 2 * a ** 2 * h ** 4 * lambda_a ** 4,  # h13
            -578 * np.pi ** 4 * D12 * D44 * a ** 6 * h ** 4 * mu ** 4,  # h14
            -1156 * np.pi ** 4 * D12 * D44 * a ** 4 * h ** 4 * lambda_a ** 2 * mu ** 2,  # h15
            -578 * np.pi ** 4 * D12 * D44 * a ** 2 * h ** 4 * lambda_a ** 4,  # h16
            289 * np.pi ** 4 * D22 * D44 * a ** 6 * h ** 4 * mu ** 2,  # h17
            289 * np.pi ** 4 * D22 * D44 * a ** 4 * h ** 4 * lambda_a ** 2,  # h18
            2856 * np.pi ** 2 * D22 * D55 * a ** 6 * h ** 2 * lambda_a ** 2 * mu ** 2,  # h19
            2856 * np.pi ** 2 * D22 * D55 * a ** 4 * h ** 2 * lambda_a ** 4,  # h20
            2856 * np.pi ** 2 * D44 * D55 * a ** 6 * h ** 2 * lambda_a ** 2 * mu ** 4,  # h21
            5712 * np.pi ** 2 * D44 * D55 * a ** 4 * h ** 2 * lambda_a ** 4 * mu ** 2,  # h22
            2856 * np.pi ** 2 * D44 * D55 * a ** 2 * h ** 2 * lambda_a ** 6,  # h23
            2856 * np.pi ** 2 * D44 * D66 * a ** 6 * h ** 2 * lambda_a ** 2 * mu ** 2,  # h24
            2856 * np.pi ** 2 * D44 * D66 * a ** 4 * h ** 2 * lambda_a ** 4,  # h25
            28224 * D55 * D66 * a ** 6 * lambda_a ** 4 * mu ** 2,  # h26
            28224 * D55 * D66 * a ** 4 * lambda_a ** 6  # h27
        ]

        return sum(gi_values), sum(hi_values)

    # Your main function, N_hat
    def N_hat(x, *args):
        lambda_a, mu = x
        alpha, beta, chi, a, *D = args[:10]  # Extracts the first six parameters
        c1 = np.pi ** 2 * h ** 2
        c2 = lambda_a ** 2 + a ** 2 * mu ** 2
        c3 = lambda_a ** 4 + 6 * a ** 2 * lambda_a ** 2 * mu ** 2 + a ** 4 * mu ** 4

        # Recalculate G and H based on the current lambda_a and mu
        G, H = gihi(D, lambda_a, a, c1, c2, c3, h, mu)

        # Calculate the critical stress state function
        return G / ((60 * a ** 2 * lambda_a ** 2 * (c2 * alpha + 2 * a ** 2 * beta * mu + a ** 2 * chi)) * H)

    def exx(k, A, e_z, e_x):
        # Define the integrand as a function of x, incorporating theta's definition

        def e_fx(theta, e_z, e_x):
            return e_x * np.sin(theta) ** 2 + e_z * np.cos(theta) ** 2

        def integrand(x):
            theta = np.pi / 2 - np.arctan(A * k * np.cos(k * x))
            return e_fx(theta, e_z, e_x)

        # Perform the integration
        result, _ = quad(integrand, 0, 2 * np.pi / k)
        return k / (2 * np.pi) * result

    def ezz(k, A, e_z, e_x):
        # Define the integrand as a function of x, incorporating theta's definition
        def e_fz(theta, e_z, e_x):
            return e_z * np.sin(theta) ** 2 + e_x * np.cos(theta) ** 2

        def integrand(x):
            theta = np.pi / 2 - np.arctan(A * k * np.cos(k * x))
            return e_fz(theta, e_z, e_x)

        # Perform the integration
        result, _ = quad(integrand, 0, 2 * np.pi / k)
        return k / (2 * np.pi) * result

    def calculate_D(Exx, Eyy, v, D44):

        D11 = Exx / (1 - v ** 2)
        D12 = v * Exx / (1 - v ** 2)
        D22 = Eyy / (1 - 1 * v ** 2)
        D44 = D44
        D55 = Exx / 300
        D66 = Exx / 30

        return [D11, D12, D22, D44, D55, D66]

    # Initial guess x0 as (lambda_0 / a, mu_0) based on the document
    lambda_a0 = 1.0  # Adjust as necessary
    mu_0 = 1.0  # Adjust as necessary
    x0 = [lambda_a0, mu_0]

    sigma_cr_fin = np.inf

    for iii in range(1):
        if iii == 0:
            h = X[0]
            Exx = 7000e6
            v = 0.2
            a = X[2]
            Eyy = Exx/2

        else:
            expected_value = expected_arc_length(X[3]/2, X[2], 1)
            h = X[1]*expected_value
            Exx = 7000e6
            omega = 2 * np.pi / X[2]
            Exx = exx(omega, X[3]/2, Exx/20, Exx)
            v = 0.2
            a = X[2]/2


        # Additional arguments, replace with actual values or calculations
        alpha = 0  # Example value
        beta = 0  # Example value
        chi = 1  # Example value
        G_xy = Exx/3
        D = calculate_D(Exx, Eyy, v, G_xy)

        args = (alpha, beta, chi, a, *D)

        # Perform minimization using Nelder-Mead method
        result = minimize(N_hat, x0, args=args, method='Nelder-Mead', options={'xatol': 1e-8, 'fatol': 1e-8})

        # The result object contains the solution
        x_cr = result.x
        lambda_a, mu = x_cr
        N_cr = N_hat(x_cr, *args)

        # Calculate critical stresses based on the critical edge load
        sigma_11_cr = (N_cr * alpha) / h
        sigma_12_cr = (N_cr * beta) / h
        sigma_22_cr = (N_cr * chi) / h

        # The critical stress sigma_cr in matrix form
        sigma_cr = [sigma_11_cr, sigma_12_cr, sigma_22_cr]

        # Calculate angles phi and theta
        phi = math.atan2(sigma_22_cr, math.sqrt(sigma_11_cr ** 2 + sigma_12_cr ** 2))
        theta = math.atan2(sigma_12_cr, sigma_11_cr)

        # Calculate sigma^R
        sigma_R = sigma_22_cr / math.cos(phi)  # Using sigma_22 because it directly relates to cos(phi)

        # Convert angles from radians to degrees for easier interpretation
        phi_deg = math.degrees(phi)
        theta_deg = math.degrees(theta)

        # Compute n11, n12, and n22 based on phi and theta
        n11 = np.sin(phi) * np.cos(theta)
        n12 = np.sin(phi) * np.sin(theta)
        n22 = np.cos(phi)

        ff = -0.36
        Xt = 8.57e7
        Xc = 2.52e7
        Yc = 1.47e7
        Yt = 3.52e7
        T = 1.5e7
        #Xc = 1e7
        #Yc = 5e6
        #Xt = 1e7
        #Yt = 5e6

        F1 = 1 / Xt + 1 / Xc
        F2 = 1 / Yt + 1 / Yc
        F11 = - 1 / (Xt * Xc)
        F22 = - 1 / (Yt * Yc)
        F12 = ff * np.sqrt(F11 * F22)
        F66 = 1 / T ** 2

        # Solve for sigma_tw_R using the quadratic formula (from Tsai-Wu criterion)
        A = F11 * np.sin(phi) ** 2 * np.cos(theta) ** 2 + F22 * np.cos(phi) ** 2 + F66 * np.sin(phi) ** 2 * np.sin(
            theta) ** 2 + 2 * F12 * np.sin(phi) * np.cos(phi) * np.cos(theta)
        B = F1 * np.sin(phi) * np.cos(theta) + F2 * np.cos(phi)
        C = -1

        # Calculate discriminant
        discriminant = B ** 2 - 4 * A * C
        
        
        if discriminant >= 0:
            sigma_tw_R1 = (-B + np.sqrt(discriminant)) / (2 * A)
            sigma_tw_R2 = (-B - np.sqrt(discriminant)) / (2 * A)

            if sigma_tw_R1 > 0 and sigma_tw_R2 > 0:
                sigma_tw = min(sigma_tw_R1, sigma_tw_R2)
            elif sigma_tw_R1 > 0 and sigma_tw_R2 < 0:
                sigma_tw = sigma_tw_R1
            elif sigma_tw_R1 < 0 and sigma_tw_R2 > 0:
                sigma_tw = sigma_tw_R2
            else:
                sigma_tw = None
        else:
            sigma_tw = None

        # Solve for sigma_cr_R (from buckling condition)
        c = lambda_a ** 2 + a ** 2 * mu ** 2
        c1 = np.pi ** 2 * h ** 2
        c2 = lambda_a ** 2 + a ** 2 * mu ** 2
        c3 = lambda_a ** 4 + 6 * a ** 2 * lambda_a ** 2 * mu ** 2 + a ** 4 * mu ** 4

        G, H = gihi(D, lambda_a, a, c1, c2, c3, h, mu)
        sigma_cr_R = abs(
            G / ((-c * n11 + 2 * a ** 2 * mu * abs(n12) - a ** 2 * n22) * (60 * a ** 2 * h * lambda_a ** 2 * H)))
        if sigma_tw is None:
            sigma_cr = sigma_cr_R
        else:
            sigma_cr = min(sigma_tw, sigma_cr_R)
        if sigma_cr < sigma_R:
            cc = False
            
        
        sigma_cr_fin = min(sigma_cr, sigma_cr_fin)

    return sigma_cr_fin, cc


