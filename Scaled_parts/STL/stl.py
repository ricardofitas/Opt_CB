import numpy as np

# ===========================
# User configuration
# ===========================
INPUT_FILE = "local_curve_3.txt"   # TXT/CSV with two columns: x y
OUTPUT_STL = "corrugator3c.stl"    # output STL file
N_REPETITIONS = 2                 # how many times to repeat the curve
FRACTION_BASE_HEIGHT = 0.25       # how far below y_min (as a fraction of total height)
EXTRUSION_HEIGHT = 10.0           # thickness in Z
PLOT_CURVE = True                 # <- set True to see the curve
SAVE_PLOT_AS = None               # e.g. "curve.png" to also save, or None


def load_curve(path):
    """
    Reads (x, y) points from a TXT/CSV file.
    - Ignores blank lines and lines starting with '#'
    - Skips a header like 'x y'
    - Accepts whitespace or comma separators
    - Uses the first two numeric columns on each valid line
    """
    points = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            s = s.replace(",", " ")
            parts = s.split()
            if len(parts) < 2:
                continue
            try:
                x = float(parts[0]); y = float(parts[1])
            except ValueError:
                continue  # header or non-numeric line
            points.append((x, y))

    if not points:
        raise ValueError("No numeric (x, y) pairs found in file.")

    data = np.array(points, dtype=float)
    idx = np.argsort(data[:, 0])          # sort by x
    return data[idx, :]


def plot_curve(curve_xy, title="Input curve", save_path=None):
    """
    Draws the curve with matplotlib (only the polyline, no fill/area).
    """
    import matplotlib.pyplot as plt  # lazy import so it's optional at runtime
    x = curve_xy[:, 0]
    y = curve_xy[:, 1]

    fig, ax = plt.subplots()
    ax.plot(x, y)                     # no explicit colors/styles
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def build_base_polygon(curve_xy, fraction_base_height):
    x = curve_xy[:, 0]
    y = curve_xy[:, 1]
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    total_height = y_max - y_min
    if total_height <= 0:
        raise ValueError("The curve must have at least two distinct y values.")
    y_base = y_min - fraction_base_height * total_height
    points = [(x[0], y_base), *zip(x, y), (x[-1], y_base)]
    return np.array(points, dtype=float), y_base


def polygon_area(poly):
    x = poly[:, 0]; y = poly[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def point_in_triangle(p, a, b, c, eps=1e-12):
    v0 = c - a; v1 = b - a; v2 = p - a
    dot00 = np.dot(v0, v0); dot01 = np.dot(v0, v1); dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1); dot12 = np.dot(v1, v2)
    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < eps:
        return False
    inv = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv
    v = (dot00 * dot12 - dot01 * dot02) * inv
    return (u >= -eps) and (v >= -eps) and (u + v <= 1.0 + eps)


def ear_clipping_triangulation(poly):
    n = len(poly)
    if n < 3: return []
    if n == 3: return [(0, 1, 2)]
    indices = list(range(n))
    triangles = []

    def is_convex(i_prev, i_cur, i_next):
        a, b, c = poly[i_prev], poly[i_cur], poly[i_next]
        return np.cross(b - a, c - b) > 0

    while len(indices) > 3:
        ear_found = False
        m = len(indices)
        for idx in range(m):
            i_prev = indices[(idx - 1) % m]
            i_cur  = indices[idx]
            i_next = indices[(idx + 1) % m]
            if not is_convex(i_prev, i_cur, i_next):
                continue
            a, b, c = poly[i_prev], poly[i_cur], poly[i_next]
            ear = True
            for j in indices:
                if j in (i_prev, i_cur, i_next): continue
                if point_in_triangle(poly[j], a, b, c):
                    ear = False; break
            if ear:
                triangles.append((i_prev, i_cur, i_next))
                indices.pop(idx)
                ear_found = True
                break
        if not ear_found:
            print("Warning: could not find more 'ears'. Polygon may be self-intersecting.")
            break
    if len(indices) == 3:
        triangles.append(tuple(indices))
    return triangles


def extrude_polygon(poly2d, height):
    poly = np.array(poly2d, dtype=float)
    if polygon_area(poly) < 0:
        poly = poly[::-1]
    n = len(poly)
    tris_idx = ear_clipping_triangulation(poly)
    triangles = []

    # top & bottom
    for i, j, k in tris_idx:
        a2, b2, c2 = poly[i], poly[j], poly[k]
        a_top = np.array([a2[0], b2[1]*0 + a2[1], height])
        b_top = np.array([b2[0], b2[1], height])
        c_top = np.array([c2[0], c2[1], height])
        triangles.append((a_top, b_top, c_top))
        a_bot = np.array([a2[0], a2[1], 0.0])
        b_bot = np.array([b2[0], b2[1], 0.0])
        c_bot = np.array([c2[0], c2[1], 0.0])
        triangles.append((c_bot, b_bot, a_bot))

    # sides
    for i in range(n):
        j = (i + 1) % n
        x0, y0 = poly[i]
        x1, y1 = poly[j]
        v0 = np.array([x0, y0, 0.0])
        v1 = np.array([x1, y1, 0.0])
        v2 = np.array([x0, y0, height])
        v3 = np.array([x1, y1, height])
        triangles.append((v0, v1, v3))
        triangles.append((v0, v3, v2))

    return np.array(triangles, dtype=float)


def repeat_mesh(triangles, dx, n_repetitions):
    all_tris = []
    for k in range(n_repetitions):
        shift = np.array([k * dx, 0.0, 0.0])
        all_tris.append(triangles + shift)
    return np.vstack(all_tris)


def write_stl_ascii(filename, triangles):
    with open(filename, "w", encoding="ascii") as f:
        f.write("solid extrusion\n")
        for tri in triangles:
            v1, v2, v3 = tri
            normal = np.cross(v2 - v1, v3 - v1)
            norm = np.linalg.norm(normal)
            normal = normal / norm if norm > 0 else np.array([0.0, 0.0, 0.0])
            f.write(f"  facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}\n")
            f.write("    outer loop\n")
            for v in (v1, v2, v3):
                f.write(f"      vertex {v[0]:.6e} {v[1]:.6e} {v[2]:.6e}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write("endsolid extrusion\n")


def main():
    # 1) Load curve
    curve = load_curve(INPUT_FILE)

    # (optional) Plot just the curve
    if PLOT_CURVE:
        plot_curve(curve, title="Input curve", save_path=SAVE_PLOT_AS)

    # 2) Build base polygon (area under the curve)
    polygon, y_base = build_base_polygon(curve, FRACTION_BASE_HEIGHT)

    # 3) Extrude to a 3D volume
    triangles = extrude_polygon(polygon, EXTRUSION_HEIGHT)

    # 4) Determine width in X for spacing repetitions
    x = curve[:, 0]
    width_x = float(np.max(x) - np.min(x))

    # 5) Repeat mesh along X
    repeated_triangles = repeat_mesh(triangles, width_x, N_REPETITIONS)

    # 6) Save STL
    write_stl_ascii(OUTPUT_STL, repeated_triangles)

    print(f"STL generated at '{OUTPUT_STL}'")
    print(f"y_base = {y_base:.3f}")


if __name__ == "__main__":
    main()
