import numpy as np
import matplotlib.pyplot as plt

def compute_fillet_arc(p1, p2, p3, radius, num_points=50):
    v1 = p1 - p2
    v2 = p3 - p2
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)

    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    angle = np.arccos(dot) / 2
    if np.isclose(angle, 0) or np.allclose(v1, -v2):
        return np.array([]), np.array([]), None, None  # No arc

    # Bisector direction
    bisector = v1 + v2
    bisector /= np.linalg.norm(bisector)

    # Tangent offset (limit to avoid overshooting)
    offset = radius / np.tan(angle)
    offset_limit = 0.99 * min(np.linalg.norm(p1 - p2), np.linalg.norm(p3 - p2))
    offset = min(offset, offset_limit)

    # Arc center
    center = p2 + bisector * (radius / np.sin(angle))

    # Tangent points
    p_start = p2 + v1 * offset
    p_end = p2 + v2 * offset

    # Arc angles
    start_vec = p_start - center
    end_vec = p_end - center
    start_angle = np.arctan2(start_vec[1], start_vec[0])
    end_angle = np.arctan2(end_vec[1], end_vec[0])

    # Determine sweep direction (cross product in 2D)
    cross = v1[0]*v2[1] - v1[1]*v2[0]
    if cross < 0:
        theta = np.linspace(start_angle, end_angle, num_points)
    else:
        theta = np.linspace(start_angle, end_angle, num_points)[::-1]

    arc_x = center[0] + radius * np.cos(theta)
    arc_y = center[1] + radius * np.sin(theta)

    return arc_x, arc_y, p_start, p_end



def plot_nurbs_with_manual_fillets(X, r1, r2, n_repeats=1):
    # Reconstruct control points
    d1, d2, d3, d4, d5 = X[:5]
    lambda_ = 5.65e-3
    Amp = 2.65e-3
    norm = lambda x: x / sum([d1, d2, d3, d4, d5])
    AAA_base = np.array([0,
                         norm(d1),
                         norm(d1 + d2 / 2),
                         norm(d1 + d2),
                         norm(d1 + d2 + d3 / 2),
                         norm(d1 + d2 + d3),
                         norm(d1 + d2 + d3 + d4 / 2),
                         norm(d1 + d2 + d3 + d4),
                         1])
    BBB = np.array([0, 0, 0.5, 1, 1, 1, 0.5, 0, 0])

    fig, ax = plt.subplots(figsize=(8, 4))
    
    for rep in range(n_repeats):
        x_offset = rep * lambda_
        AAA = AAA_base * lambda_ + x_offset
        pts = np.array([(x, y * Amp) for x, y in zip(AAA, BBB)])

        # Draw fillets and segments
        i = 0
        while i < len(pts) - 1:
            if i in [1, 3, 5, 7]:
                p1, p2, p3 = pts[i - 1], pts[i], pts[i + 1]
                radius = r1 / 1000 if i in [1, 3] else r2 / 1000
                arc_x, arc_y, p_start, p_end = compute_fillet_arc(p1, p2, p3, radius)
                if arc_x.size > 0:
                    ax.plot([p1[0], p_start[0]], [p1[1], p_start[1]], 'b--')
                    ax.plot(arc_x, arc_y, 'b--')
                    ax.plot([p_end[0], p3[0]], [p_end[1], p3[1]], 'b--')
                    i += 2
                    continue
            i += 1

    # Plot NURBS just once for visual reference
    def evaluate_nurbs(X, r1_weight, r2_weight, sample_size=2000):
        from geomdl import NURBS
        suma = sum(X[:5])
        d1, d2, d3, d4, d5 = [X[i] / suma for i in range(5)]
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
        pts = np.array(curve.evalpts)
        scale_x = lambda_ / (pts[-1, 0] - pts[0, 0])
        scale_y = Amp / (np.max(pts[:, 1]) - np.min(pts[:, 1]))
        pts[:, 0] = pts[:, 0] * scale_x
        pts[:, 1] = pts[:, 1] * scale_y
        return pts

    X5, X6 = X[5], X[6]
    r1_weight = 0.0103 * np.exp(9.17 * X5) + 0.9897
    r2_weight = 0.0103 * np.exp(9.17 * X6) + 0.9897
    nurbs_base = evaluate_nurbs(X, r1_weight, r2_weight)

    # Replicate NURBS curve as background reference
    for rep in range(n_repeats):
        nurbs = nurbs_base.copy()
        nurbs[:, 0] += rep * lambda_
        ax.plot(nurbs[:, 0], nurbs[:, 1], 'r-', label='NURBS Curve' if rep == 0 else None)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(f"NURBS vs. Real Geometry")
    ax.grid(True)
    ax.legend()
    ax.grid(False)
    plt.tight_layout()
    plt.show()


# Run the final visual with test data
plot_nurbs_with_manual_fillets(
    [0.393460392,	1.802275225,	1.239343829,	7.82725377,	0.370237126,	0.532507751,	0.174536967],
    r1=2.28716137, r2=16.75557933)

