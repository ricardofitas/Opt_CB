import numpy as np

# ===========================
# User configuration
# ===========================
INPUT_FILE = "local_curve_sine.txt"   # TXT/CSV with two columns: x y
OUTPUT_STL = "corrugator_v2_sine.stl"    # output STL file
PLOT_2D_DRAWING = True
DRAWING_FILE = "corrugator_sine_2d.pdf"  # podes usar .png, .svg, .pdf
N_REPETITIONS = 10                 # how many times to repeat the curve
FRACTION_BASE_HEIGHT = 0.25       # how far below y_min (as a fraction of total height)
EXTRUSION_HEIGHT = 25.0           # thickness in Z
PLOT_CURVE = True                 # <- set True to see the curve
SAVE_PLOT_AS = None               # e.g. "curve.png" to also save, or None
PITCH_X = None

def compute_main_dimensions(curve_xy, y_base, dx, n_rep, extrusion_width):
    x = curve_xy[:, 0]
    y = curve_xy[:, 1]
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))

    width_x = x_max - x_min                  # largura de 1 período (da curva original)
    L = width_x + (n_rep - 1) * dx           # comprimento total correto

    H = y_max - y_base
    H_curve = y_max - y_min
    base_drop = y_min - y_base

    return {
        "pitch_x": dx,
        "width_one": width_x,
        "length_x": L,
        "y_base": y_base,
        "y_max": y_max,
        "height_total": H,
        "height_curve": H_curve,
        "base_drop": base_drop,
        "extrusion_width_z": extrusion_width,
        "x0": x_min,
        "x1": x_min + L,
    }



def plot_2d_with_dimensions(curve_xy, y_base, dx, n_rep, dims, save_path=None, title="2D profile with main dimensions"):
    import matplotlib.pyplot as plt

    x0 = dims["x0"]
    x1 = dims["x1"]
    y_max = dims["y_max"]
    H = dims["height_total"]

    # repetir curva para desenho
    xs_all = []
    ys_all = []
    for k in range(n_rep):
        xs_all.append(curve_xy[:, 0] + k * dx)
        ys_all.append(curve_xy[:, 1])

    fig, ax = plt.subplots()
    for k in range(n_rep):
        ax.plot(xs_all[k], ys_all[k])

    # linha de base
    ax.plot([x0, x1], [y_base, y_base])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)

    # offsets para as cotas
    off = 0.12 * H if H > 0 else 1.0

    # cota comprimento L
    y_dim = y_base - off
    ax.annotate("", xy=(x0, y_dim), xytext=(x1, y_dim),
                arrowprops=dict(arrowstyle="<->"))
    ax.text((x0 + x1) / 2, y_dim, f"L = {dims['length_x']:.3f}",
            ha="center", va="bottom")

    # cota altura H
    x_dim = x0 - 0.08 * dims["length_x"] if dims["length_x"] > 0 else x0 - 1.0
    ax.annotate("", xy=(x_dim, y_base), xytext=(x_dim, y_max),
                arrowprops=dict(arrowstyle="<->"))
    ax.text(x_dim, (y_base + y_max) / 2, f"H = {dims['height_total']:.3f}",
            ha="right", va="center", rotation=90)

    # anotação da largura de extrusão (Z)
    ax.text(x0, y_max + off * 0.3, f"Wz (extrusão) = {dims['extrusion_width_z']:.3f}",
            ha="left", va="bottom")

    # enquadrar melhor
    ax.set_xlim(x0 - 0.15 * dims["length_x"], x1 + 0.05 * dims["length_x"])
    ax.set_ylim(y_dim - off * 0.2, y_max + off)

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    
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

    # 3) Extrude to a 3D volume (Z = "largura/espessura")
    triangles = extrude_polygon(polygon, EXTRUSION_HEIGHT)

    # 4) Determine width in X and choose pitch dx
    x = curve[:, 0]
    width_x = float(np.max(x) - np.min(x))
    dx = width_x if PITCH_X is None else float(PITCH_X)

    # 5) Repeat mesh along X (IMPORTANT: use dx, not width_x)
    repeated_triangles = repeat_mesh(triangles, dx, N_REPETITIONS)

    # 6) Save STL
    write_stl_ascii(OUTPUT_STL, repeated_triangles)

    print(f"STL generated at '{OUTPUT_STL}'")
    print(f"y_base = {y_base:.3f}")
    print(f"width_x (1 periodo) = {width_x:.3f}")
    print(f"dx (pitch) = {dx:.3f}")

    # 7) 2D drawing with dimensions
    dims = compute_main_dimensions(curve, y_base, dx, N_REPETITIONS, EXTRUSION_HEIGHT)
    if PLOT_2D_DRAWING:
        plot_2d_with_dimensions(curve, y_base, dx, N_REPETITIONS, dims, save_path=DRAWING_FILE)

    print("Dimensões principais:")
    for k, v in dims.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
