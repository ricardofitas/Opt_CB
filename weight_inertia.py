import numpy as np

def compute_expected_weight(local_curve, t):
    
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

    return expected_weight * 10**3, inertia * 10**9, efficiency
