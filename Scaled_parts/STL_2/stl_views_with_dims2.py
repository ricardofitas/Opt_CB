import argparse
import numpy as np
import vtk
import svgwrite


"""Generate orthographic STL views (XY/XZ/YZ) into a single SVG with global dimensions.

Notes
-----
* The SVG text is intentionally English (titles/labels).
* All views use a *shared drawing scale* so they look consistent.
* Pitch is computed as: pitch = total_length / number_of_repetitions.
"""


# ------------------------- Robust outlines (occlusion-aware) -------------------------

def _polydata_points_to_numpy(pd: vtk.vtkPolyData) -> np.ndarray:
    """Return Nx3 numpy array of polydata points."""
    pts = pd.GetPoints()
    if pts is None:
        return np.zeros((0, 3), dtype=float)
    n = pts.GetNumberOfPoints()
    arr = np.empty((n, 3), dtype=float)
    for i in range(n):
        arr[i] = pts.GetPoint(i)
    return arr


def _convex_hull_2d(points: np.ndarray) -> list[tuple[float, float]]:
    """Monotonic chain convex hull. Returns vertices in CCW order."""
    if len(points) == 0:
        return []
    # Reduce duplicates by rounding (important for stable sorting)
    pts = np.round(points.astype(float), decimals=6)
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    # unique rows
    if len(pts) > 1:
        keep = np.ones(len(pts), dtype=bool)
        keep[1:] = np.any(np.diff(pts, axis=0) != 0, axis=1)
        pts = pts[keep]
    if len(pts) <= 2:
        return [tuple(p) for p in pts]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        p = (float(p[0]), float(p[1]))
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in pts[::-1]:
        p = (float(p[0]), float(p[1]))
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]
    return hull


def _outline_xz_envelope(points3d: np.ndarray, *, bins: int = 2000, max_points: int = 1400):
    """Occlusion-aware XZ outline via z-envelope over x bins."""
    if len(points3d) == 0:
        return []
    x = points3d[:, 0]
    z = points3d[:, 2]
    xmin, xmax = float(x.min()), float(x.max())
    if xmax <= xmin:
        return []

    bins = int(max(200, min(bins, max(200, len(points3d) // 20))))
    edges = np.linspace(xmin, xmax, bins + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, bins - 1)
    zmax = np.full(bins, np.nan, dtype=float)
    zmin = np.full(bins, np.nan, dtype=float)
    for b in range(bins):
        m = idx == b
        if np.any(m):
            zb = z[m]
            zmax[b] = float(np.max(zb))
            zmin[b] = float(np.min(zb))

    x_cent = 0.5 * (edges[:-1] + edges[1:])

    def _interp_nan(v: np.ndarray) -> np.ndarray:
        good = ~np.isnan(v)
        if not np.any(good):
            return v
        return np.interp(x_cent, x_cent[good], v[good])

    zmax_i = _interp_nan(zmax)
    zmin_i = _interp_nan(zmin)

    # light smoothing to reduce jagged tessellation in the drawing
    w = 7
    ker = np.ones(w) / w
    zmax_s = np.convolve(zmax_i, ker, mode="same")
    zmin_s = np.convolve(zmin_i, ker, mode="same")
    
    # If the STL is essentially a single surface extruded in Y, then for each x-bin
    # we get zmax ~= zmin and the "outline polygon" collapses to only the sine curve.
    # In that case, force a flat base at the global minimum of the curve (valley level).
    z_span = float(np.nanmax(zmax_s) - np.nanmin(zmax_s))
    if not np.isfinite(z_span):
        return []

    thickness_med = float(np.nanmedian(np.abs(zmax_s - zmin_s)))
    eps = max(1e-6, 1e-3 * z_span)  # robust threshold

    if thickness_med < eps:
        z_base = float(np.nanmin(zmax_s))  # valley level
        zmin_s = np.full_like(zmax_s, z_base)

    top = list(zip(x_cent, zmax_s))
    bot = list(zip(x_cent[::-1], zmin_s[::-1]))
    poly = top + bot
    if poly and (poly[0] != poly[-1]):
        poly.append(poly[0])

    # decimate to keep SVG light
    if len(poly) > max_points:
        step = int(np.ceil(len(poly) / max_points))
        poly = poly[::step]
        if poly[0] != poly[-1]:
            poly.append(poly[0])

    return [poly]


def outer_outline_2d(poly: vtk.vtkPolyData, view: str, *, unit_scale: float):
    """Return polylines_2d approximating the *outer* projected outline.

    This avoids VTK's vtkPolyDataSilhouette artifacts when the STL is composed
    of repeated chunks with internal caps (silhouette is not occlusion-aware).
    """
    pts3d = _polydata_points_to_numpy(poly)
    if view == "xz":
        # projection along Y
        polylines = _outline_xz_envelope(pts3d)
    elif view == "xy":
        hull = _convex_hull_2d(pts3d[:, [0, 1]])
        polylines = [hull + [hull[0]]] if len(hull) >= 3 else [hull]
    elif view == "yz":
        hull = _convex_hull_2d(pts3d[:, [1, 2]])
        polylines = [hull + [hull[0]]] if len(hull) >= 3 else [hull]
    else:
        raise ValueError

    # apply unit scale
    polylines_s = []
    for poly2 in polylines:
        polylines_s.append([(a * unit_scale, b * unit_scale) for (a, b) in poly2])
    return polylines_s


# ------------------------- VTK: STL + silhouette -------------------------


def load_stl_pipeline(path: str) -> vtk.vtkPolyDataNormals:
    """Return a VTK pipeline (vtkPolyDataNormals) instead of raw polydata.

    Using SetInputConnection(...) everywhere is much more robust across VTK
    Python wrappers than passing vtkPolyData objects into SetInputData(...).
    """
    r = vtk.vtkSTLReader()
    r.SetFileName(path)

    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(r.GetOutputPort())

    norms = vtk.vtkPolyDataNormals()
    norms.SetInputConnection(clean.GetOutputPort())
    norms.ConsistencyOn()
    norms.AutoOrientNormalsOn()
    norms.SplittingOff()
    norms.Update()

    return norms


def make_camera_for_view(bounds, view: str) -> vtk.vtkCamera:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    cz = (zmin + zmax) / 2.0
    dx = (xmax - xmin)
    dy = (ymax - ymin)
    dz = (zmax - zmin)
    maxdim = max(dx, dy, dz) if max(dx, dy, dz) > 0 else 1.0
    dist = maxdim * 3.0

    cam = vtk.vtkCamera()
    cam.ParallelProjectionOn()
    cam.SetFocalPoint(cx, cy, cz)

    if view == "xy":      # look along +Z
        cam.SetPosition(cx, cy, cz + dist)
        cam.SetViewUp(0, 1, 0)
    elif view == "xz":    # look along +Y
        cam.SetPosition(cx, cy + dist, cz)
        cam.SetViewUp(0, 0, 1)
    elif view == "yz":    # look along +X
        cam.SetPosition(cx + dist, cy, cz)
        cam.SetViewUp(0, 0, 1)
    else:
        raise ValueError("view must be one of: xy, xz, yz")

    return cam


def silhouette_polydata(input_port, camera: vtk.vtkCamera) -> vtk.vtkPolyData:
    sil = vtk.vtkPolyDataSilhouette()
    sil.SetInputConnection(input_port)
    sil.SetCamera(camera)
    sil.SetEnableFeatureAngle(0)  # pure silhouette
    sil.Update()
    return sil.GetOutput()



def vtk_lines_to_polylines(pd: vtk.vtkPolyData):
    pts = pd.GetPoints()
    if pts is None:
        return []
    polylines = []
    for i in range(pd.GetNumberOfCells()):
        cell = pd.GetCell(i)
        ids = cell.GetPointIds()
        n = ids.GetNumberOfIds()
        if n < 2:
            continue
        poly = []
        for j in range(n):
            x, y, z = pts.GetPoint(ids.GetId(j))
            poly.append((x, y, z))
        polylines.append(poly)
    return polylines


# ------------------------- Projection + pitch -------------------------

def project_point(p, view: str):
    x, y, z = p
    if view == "xy":
        return (x, y)
    if view == "xz":
        return (x, z)
    if view == "yz":
        return (y, z)
    raise ValueError


def estimate_pitch_from_xz(polylines_xz_2d, dx, dz, bins=2000):
    """Estimate corrugation pitch from the XZ silhouette.

    Returns
    -------
    (pitch_mm, n_repetitions) or (None, None)

    Pitch is reported as: pitch = total_length / n_repetitions.
    We estimate an initial pitch from peak-to-peak distances, infer an integer
    repetition count, then recompute pitch via L/n.
    """
    if not polylines_xz_2d:
        return None, None

    xs = []
    zs = []
    for poly in polylines_xz_2d:
        for (x, z) in poly:
            xs.append(x)
            zs.append(z)
    xs = np.asarray(xs)
    zs = np.asarray(zs)

    if len(xs) < 100:
        return None, None

    x_min, x_max = xs.min(), xs.max()
    if x_max <= x_min:
        return None, None

    bins = int(min(max(bins, 300), 5000))
    edges = np.linspace(x_min, x_max, bins + 1)
    idx = np.clip(np.digitize(xs, edges) - 1, 0, bins - 1)

    zmax = np.full(bins, -np.inf)
    np.maximum.at(zmax, idx, zs)

    ok = np.isfinite(zmax)
    if ok.sum() < bins * 0.2:
        return None, None

    x_centers = (edges[:-1] + edges[1:]) / 2.0
    zmax_interp = zmax.copy()
    good = np.where(ok)[0]
    bad = np.where(~ok)[0]
    zmax_interp[bad] = np.interp(x_centers[bad], x_centers[good], zmax_interp[good])

    w = 7
    kernel = np.ones(w) / w
    z_smooth = np.convolve(zmax_interp, kernel, mode="same")

    h = max(dz, 1e-9)
    thr = 0.03 * h
    peaks = []
    for i in range(2, len(z_smooth) - 2):
        if z_smooth[i] > z_smooth[i-1] and z_smooth[i] > z_smooth[i+1]:
            if (z_smooth[i] - min(z_smooth[i-1], z_smooth[i+1])) > thr:
                peaks.append(i)

    if len(peaks) < 3:
        return None, None

    peak_x = x_centers[peaks]
    d = np.diff(peak_x)
    d = d[(d > 0.001 * dx) & (d < 0.5 * dx)]
    if len(d) < 2:
        return None, None

    pitch_raw = float(np.median(d))
    if pitch_raw <= 0:
        return None, None

    n_round = max(1, int(round(dx / pitch_raw)))
    n_from_peaks = max(1, len(peaks) - 1)
    n = n_from_peaks if abs(n_round - n_from_peaks) <= 1 else n_round
    pitch = float(dx / n)

    return pitch, n


# ------------------------- SVG: drawing + dimensions -------------------------

# Font sizes (mm). User requested ~1/2 of previous values.
TITLE_FONT = "2mm"
LABEL_FONT = "1.05mm"
DIM_FONT   = "0.8mm"

# Dimension styling (scaled down with the fonts)
DIM_STROKE_W = 0.14
DIM_ARROW    = 0.8
HALO_STROKE_W = 0.5


def add_text_halo(dwg, text, insert, font_size, text_anchor="middle", rotate=None, rotate_center=None):
    """Draw text with a white halo (stroke) underneath for legibility."""
    t_halo = dwg.text(
        text,
        insert=insert,
        font_size=font_size,
        text_anchor=text_anchor,
        fill="black",
        stroke="white",
        stroke_width=HALO_STROKE_W,
        stroke_linejoin="round",
    )
    if rotate is not None:
        t_halo.rotate(rotate, center=rotate_center)
    dwg.add(t_halo)

    t = dwg.text(text, insert=insert, font_size=font_size, text_anchor=text_anchor, fill="black")
    if rotate is not None:
        t.rotate(rotate, center=rotate_center)
    dwg.add(t)


def add_dim_h(dwg, x1, x2, y, offset, text, stroke="black"):
    y2 = y + offset
    dwg.add(dwg.line((x1, y), (x1, y2), stroke=stroke, stroke_width=DIM_STROKE_W))
    dwg.add(dwg.line((x2, y), (x2, y2), stroke=stroke, stroke_width=DIM_STROKE_W))
    dwg.add(dwg.line((x1, y2), (x2, y2), stroke=stroke, stroke_width=DIM_STROKE_W))

    s = DIM_ARROW

    def arrow(px, py, direction):
        return dwg.polygon(
            [(px, py), (px + direction * s, py - s / 1.6), (px + direction * s, py + s / 1.6)],
            fill=stroke,
        )

    direction = 1 if x2 > x1 else -1
    dwg.add(arrow(x1, y2, direction))
    dwg.add(arrow(x2, y2, -direction))

    mx = (x1 + x2) / 2.0
    add_text_halo(dwg, text, insert=(mx, y2 - 2.2), font_size=DIM_FONT, text_anchor="middle")


def add_dim_v(dwg, y1, y2, x, offset, text, stroke="black"):
    x2 = x + offset
    dwg.add(dwg.line((x, y1), (x2, y1), stroke=stroke, stroke_width=DIM_STROKE_W))
    dwg.add(dwg.line((x, y2), (x2, y2), stroke=stroke, stroke_width=DIM_STROKE_W))
    dwg.add(dwg.line((x2, y1), (x2, y2), stroke=stroke, stroke_width=DIM_STROKE_W))

    s = DIM_ARROW

    def arrow(px, py, direction):
        return dwg.polygon(
            [(px, py), (px - s / 1.6, py + direction * s), (px + s / 1.6, py + direction * s)],
            fill=stroke,
        )

    direction = 1 if y2 > y1 else -1
    dwg.add(arrow(x2, y1, direction))
    dwg.add(arrow(x2, y2, -direction))

    my = (y1 + y2) / 2.0
    tx = x2 + 3.0
    add_text_halo(
        dwg,
        text,
        insert=(tx, my),
        font_size=DIM_FONT,
        text_anchor="middle",
        rotate=-90,
        rotate_center=(tx, my),
    )


def fit_polylines_to_box(polylines_2d, box_x, box_y, box_w, box_h, pad=6, scale=None, center=True):
    pts = np.array([p for poly in polylines_2d for p in poly], dtype=float)
    if pts.size == 0:
        return [], (box_x + pad, box_y + pad, box_x + box_w - pad, box_y + box_h - pad)

    minx, miny = pts[:, 0].min(), pts[:, 1].min()
    maxx, maxy = pts[:, 0].max(), pts[:, 1].max()
    w = maxx - minx
    h = maxy - miny
    if w <= 0:
        w = 1.0
    if h <= 0:
        h = 1.0

    avail_w = box_w - 2 * pad
    avail_h = box_h - 2 * pad
    if scale is None:
        scale = min(avail_w / w, avail_h / h)

    draw_w = w * scale
    draw_h = h * scale

    ox = box_x + pad
    oy = box_y + pad
    if center:
        ox += max(0.0, (avail_w - draw_w) / 2.0)
        oy += max(0.0, (avail_h - draw_h) / 2.0)

    def tx(p):
        x, y = p
        X = ox + (x - minx) * scale
        Y = oy + (maxy - y) * scale
        return (X, Y)

    out = [[tx(p) for p in poly] for poly in polylines_2d]
    return out, (ox, oy, ox + draw_w, oy + draw_h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stl")
    ap.add_argument("--out", default="views_dims.svg")
    ap.add_argument("--unit-scale", type=float, default=1.0,
                    help="STL unit multiplier (e.g., STL in meters -> 1000 for mm)")
    ap.add_argument("--decimals", type=int, default=2)
    ap.add_argument("--repetitions", type=int, default=10,
                    help="If pitch detection fails, assume this many corrugation repetitions along length")
    ap.add_argument("--outline", choices=["outer", "silhouette"], default="outer",
                    help="How to generate 2D outlines: 'outer' (occlusion-aware, robust) or 'silhouette' (VTK)")
    ap.add_argument("--swap-yz", action="store_true",
                help="Swap Y and Z axes before generating views (fix STL axis orientation)")
    args = ap.parse_args()

    norms = load_stl_pipeline(args.stl)

    input_port = norms.GetOutputPort()
    poly_for_bounds = norms.GetOutput()
    
    if args.swap_yz:
        # Swap Y <-> Z (x stays x)
        m = vtk.vtkMatrix4x4()
        m.Identity()
        m.SetElement(1, 1, 0); m.SetElement(1, 2, 1)
        m.SetElement(2, 1, 1); m.SetElement(2, 2, 0)
    
        tr = vtk.vtkTransform()
        tr.SetMatrix(m)
    
        tf = vtk.vtkTransformPolyDataFilter()
        tf.SetTransform(tr)
        tf.SetInputConnection(norms.GetOutputPort())
        tf.Update()
    
        input_port = tf.GetOutputPort()
        poly_for_bounds = tf.GetOutput()
    
    bounds = poly_for_bounds.GetBounds()
    xmin, xmax, ymin, ymax, zmin, zmax = bounds

    dx = (xmax - xmin) * args.unit_scale
    dy = (ymax - ymin) * args.unit_scale
    dz = (zmax - zmin) * args.unit_scale

    W, H = 297, 210
    dwg = svgwrite.Drawing(args.out, size=(f"{W}mm", f"{H}mm"), viewBox=f"0 0 {W} {H}")

    title = f"STL views + dimensions (unit scale x{args.unit_scale})"
    dwg.add(dwg.text(title, insert=(10, 12), font_size=TITLE_FONT))

    margin = 10
    gap = 8
    header_h = 14
    top_y = margin + header_h

    col_w = (W - 2 * margin - gap) / 2.0
    usable_h = H - top_y - margin
    top_h = min(60.0, usable_h * 0.35)
    bot_h = usable_h - gap - top_h

    boxes = {
        "xy": (margin, top_y, col_w, top_h),
        "xz": (margin, top_y + top_h + gap, col_w, bot_h),
        "yz": (margin + col_w + gap, top_y + top_h + gap, col_w, bot_h),
        "note": (margin + col_w + gap, top_y, col_w, top_h),
    }

    view_labels = {
        "xy": "Top view (XY)",
        "xz": "Side view (XZ)",
        "yz": "End view (YZ)",
    }

    view_dims = {
        "xy": (dx, dy),
        "xz": (dx, dz),
        "yz": (dy, dz),
    }

    pad = 10
    eps = 1e-9
    scale_candidates = [
        (boxes["xy"][2] - 2 * pad) / max(dx, eps),
        (boxes["xy"][3] - 2 * pad) / max(dy, eps),
        (boxes["xz"][2] - 2 * pad) / max(dx, eps),
        (boxes["xz"][3] - 2 * pad) / max(dz, eps),
        (boxes["yz"][2] - 2 * pad) / max(dy, eps),
        (boxes["yz"][3] - 2 * pad) / max(dz, eps),
    ]
    global_scale = 0.98 * float(min(scale_candidates))

    nbx, nby, nbw, nbh = boxes["note"]
    dwg.add(dwg.rect(insert=(nbx, nby), size=(nbw, nbh), fill="none", stroke="black", stroke_width=0.2))
    dwg.add(dwg.text("Notes", insert=(nbx + 2, nby + 7), font_size=LABEL_FONT))
    dwg.add(dwg.text(f"Drawing scale: {global_scale:.4f} mm/mm", insert=(nbx + 2, nby + 16), font_size=DIM_FONT))

    pitch = None
    nrep = None

    for v in ("xy", "xz", "yz"):
        bx, by, bw, bh = boxes[v]
        dwg.add(dwg.rect(insert=(bx, by), size=(bw, bh), fill="none", stroke="black", stroke_width=0.2))
        dwg.add(dwg.text(view_labels[v], insert=(bx + 2, by + 7), font_size=LABEL_FONT))

        if args.outline == "silhouette":
            cam = make_camera_for_view(bounds, v)
            sil_pd = silhouette_polydata(input_port, cam)
            polylines3d = vtk_lines_to_polylines(sil_pd)

            polylines2d = []
            for poly3 in polylines3d:
                poly2 = []
                for p3 in poly3:
                    a, b = project_point(p3, v)
                    poly2.append((a * args.unit_scale, b * args.unit_scale))
                polylines2d.append(poly2)
        else:
            # robust outer projection outline (recommended)
            polylines2d = outer_outline_2d(poly_for_bounds, v, unit_scale=args.unit_scale)

        pitch_detected = False
        if v == "xz":
            p_est, n_est = estimate_pitch_from_xz(polylines2d, dx=dx, dz=dz)
            if p_est is not None and n_est is not None:
                pitch, nrep = p_est, n_est
                pitch_detected = True
            else:
                nrep = max(1, int(args.repetitions))
                pitch = float(dx / nrep)

        fitted, (x1, y1, x2, y2) = fit_polylines_to_box(
            polylines2d, bx, by, bw, bh, pad=pad, scale=global_scale, center=True
        )

        for poly in fitted:
            if len(poly) >= 2:
                dwg.add(dwg.polyline(poly, fill="none", stroke="black", stroke_width=0.3))

        w_dim, h_dim = view_dims[v]
        fmt = f"{{:.{args.decimals}f}}"
        w_text = fmt.format(w_dim) + " mm"
        h_text = fmt.format(h_dim) + " mm"

        off_h = min(10.0, (by + bh) - y2 - 3.0)
        off_v = min(10.0, (bx + bw) - x2 - 3.0)
        off_h = max(5.0, off_h)
        off_v = max(5.0, off_v)

        add_dim_h(dwg, x1, x2, y2, offset=off_h, text=w_text)
        add_dim_v(dwg, y1, y2, x2, offset=off_v, text=h_text)

        if v == "xz":
            note = f"Pitch = {fmt.format(pitch)} mm (L/{nrep})"
            note += "  [detected]" if pitch_detected else f"  [assumed {nrep}]"
            add_text_halo(dwg, note, insert=(bx + 2, by + 16), font_size=DIM_FONT, text_anchor="start")

    dwg.save()
    print("Generated:", args.out)


if __name__ == "__main__":
    main()
