# -*- coding: utf-8 -*-
"""
Generate orthographic STL views (XY / XZ / YZ) in a single SVG sheet with dimensions.

Design goals for corrugator-style parts:
- English text everywhere.
- Layout: Top view above Side view; End view to the right; Notes top-right.
- No "internal seam" lines from repeated STL chunks (avoid VTK silhouette artifacts).
- Side view shows the corrugation (top envelope) AND a flat base (if present) without vertical seam lines.
- Pitch is reported as: pitch = total_length / repetitions (default 10).

Dependencies:
  pip install vtk svgwrite numpy
"""

import argparse
import numpy as np
import vtk
import svgwrite


# ------------------------- Styling (mm units) -------------------------

TITLE_FONT = "10mm"
LABEL_FONT = "3mm"
DIM_FONT   = "2mm"

DIM_STROKE_W = 0.20
DIM_ARROW    = 1.2
HALO_STROKE_W = 0.8


def add_text_halo(dwg, text, insert, font_size, text_anchor="start",
                  rotate=None, rotate_center=None):
    """Text with a white halo behind (readable on top of lines)."""
    t_bg = dwg.text(
        text, insert=insert, font_size=font_size, text_anchor=text_anchor,
        fill="black", stroke="white", stroke_width=HALO_STROKE_W,
        stroke_linejoin="round"
    )
    t_fg = dwg.text(
        text, insert=insert, font_size=font_size, text_anchor=text_anchor,
        fill="black"
    )
    if rotate is not None:
        t_bg.rotate(rotate, center=rotate_center)
        t_fg.rotate(rotate, center=rotate_center)
    dwg.add(t_bg)
    dwg.add(t_fg)


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
    add_text_halo(dwg, text, insert=(mx, y2 - 1.8), font_size=DIM_FONT, text_anchor="middle")


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
    tx = x2 + 2.2
    add_text_halo(
        dwg, text, insert=(tx, my), font_size=DIM_FONT, text_anchor="middle",
        rotate=-90, rotate_center=(tx, my)
    )


# ------------------------- STL loading + optional axis swap -------------------------

def load_stl_polydata(path: str) -> vtk.vtkPolyData:
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
    return norms.GetOutput()


def swap_yz_polydata(poly: vtk.vtkPolyData) -> vtk.vtkPolyData:
    """Swap Y <-> Z axes (x stays x)."""
    m = vtk.vtkMatrix4x4()
    m.Identity()
    m.SetElement(1, 1, 0); m.SetElement(1, 2, 1)
    m.SetElement(2, 1, 1); m.SetElement(2, 2, 0)

    tr = vtk.vtkTransform()
    tr.SetMatrix(m)

    tf = vtk.vtkTransformPolyDataFilter()
    tf.SetTransform(tr)
    tf.SetInputData(poly)
    tf.Update()
    return tf.GetOutput()


def poly_points_to_numpy(poly: vtk.vtkPolyData) -> np.ndarray:
    pts = poly.GetPoints()
    if pts is None:
        return np.zeros((0, 3), dtype=float)
    n = pts.GetNumberOfPoints()
    arr = np.empty((n, 3), dtype=float)
    for i in range(n):
        arr[i] = pts.GetPoint(i)
    return arr


# ------------------------- "Clean" outlines (no internal seams) -------------------------

def outline_rect_xy(bounds, unit_scale):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    poly = [
        (xmin * unit_scale, ymin * unit_scale),
        (xmax * unit_scale, ymin * unit_scale),
        (xmax * unit_scale, ymax * unit_scale),
        (xmin * unit_scale, ymax * unit_scale),
        (xmin * unit_scale, ymin * unit_scale),
    ]
    return [poly]


def outline_rect_yz(bounds, unit_scale):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    poly = [
        (ymin * unit_scale, zmin * unit_scale),
        (ymax * unit_scale, zmin * unit_scale),
        (ymax * unit_scale, zmax * unit_scale),
        (ymin * unit_scale, zmax * unit_scale),
        (ymin * unit_scale, zmin * unit_scale),
    ]
    return [poly]


def outline_side_xz_clean(poly: vtk.vtkPolyData, bounds, unit_scale,
                          bins=1400, smooth_w=7, q_plate=0.01):
    """
    Side view (XZ) outline (robust for chunked/extruded STLs):

    What we draw (no internal seam lines):
      1) Corrugation top envelope z_top(x) (from vertex maxima per x-bin)
      2) A flat "contact/base-top" line at the *valley* level of that envelope
         (z_contact = min(z_top_s)) so the wave always sits on the base line.
      3) Optional base thickness: a bottom line at global z_min, plus end verticals.
      4) End verticals that connect the base-top line to the envelope at x-min/x-max,
         so you see the lateral walls (what you were missing).

    This avoids vtkPolyDataSilhouette artifacts and avoids "missing base" when the
    STL has sparse bottom vertices (large triangles).
    """
    pts = poly_points_to_numpy(poly)
    if len(pts) == 0:
        return []

    x = pts[:, 0]
    z = pts[:, 2]

    xmin, xmax = float(x.min()), float(x.max())
    if xmax <= xmin:
        return []

    # bins along x
    bins = int(max(400, min(bins, max(400, len(pts) // 25))))
    edges = np.linspace(xmin, xmax, bins + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, bins - 1)
    x_cent = 0.5 * (edges[:-1] + edges[1:])

    z_top = np.full(bins, np.nan, dtype=float)

    for b in range(bins):
        m = (idx == b)
        if np.any(m):
            z_top[b] = float(np.max(z[m]))

    # interpolate missing
    good = ~np.isnan(z_top)
    if np.any(good):
        z_top = np.interp(x_cent, x_cent[good], z_top[good])
    else:
        return []

    # smooth the envelope a bit (keeps the sine smooth, reduces tessellation noise)
    if smooth_w and smooth_w > 1:
        w = int(smooth_w)
        w = max(3, min(w, 31))
        ker = np.ones(w) / w
        z_top = np.convolve(z_top, ker, mode="same")

    # Scale to drawing units (mm)
    x_left  = xmin * unit_scale
    x_right = xmax * unit_scale
    x_cent_s = x_cent * unit_scale
    z_top_s  = z_top  * unit_scale

    # Base contact level: make valleys sit on the base line
    z_contact = float(np.min(z_top_s))

    # Bottom of part (if there is actual thickness)
    z_bottom = float(np.min(z)) * unit_scale
    base_thickness = z_contact - z_bottom

    # Build polylines
    polylines = []

    # 1) top envelope (include true endpoints so end walls align with bounds)
    top_poly = [(x_left, float(z_top_s[0]))] + list(zip(x_cent_s.tolist(), z_top_s.tolist())) + [(x_right, float(z_top_s[-1]))]
    polylines.append(top_poly)

    # 2) base-top contact line (always)
    polylines.append([(x_left, z_contact), (x_right, z_contact)])

    # 3) end walls from base-top to envelope (these were missing)
    polylines.append([(x_left, z_contact), (x_left, float(z_top_s[0]))])
    polylines.append([(x_right, z_contact), (x_right, float(z_top_s[-1]))])

    # 4) optional base bottom + thickness walls (only if meaningful)
    # threshold ~0.5mm in drawing units, but scale with overall height too
    height_span = float(np.max(z_top_s) - np.min(z_top_s))
    thr = max(0.5, 0.05 * max(height_span, 1.0))
    if base_thickness > thr:
        polylines.append([(x_left, z_bottom), (x_right, z_bottom)])
        polylines.append([(x_left, z_bottom), (x_left, z_contact)])
        polylines.append([(x_right, z_bottom), (x_right, z_contact)])

    return polylines
def fit_polylines_to_box(polylines_2d, box_x, box_y, box_w, box_h, pad=10, scale=None, center=True):
    pts = np.array([p for poly in polylines_2d for p in poly], dtype=float)
    if pts.size == 0:
        return [], (box_x + pad, box_y + pad, box_x + box_w - pad, box_y + box_h - pad)

    minx, miny = float(pts[:, 0].min()), float(pts[:, 1].min())
    maxx, maxy = float(pts[:, 0].max()), float(pts[:, 1].max())
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
        Y = oy + (maxy - y) * scale  # invert Y for SVG
        return (X, Y)

    out = [[tx(p) for p in poly] for poly in polylines_2d]
    return out, (ox, oy, ox + draw_w, oy + draw_h)


# ------------------------- Main -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stl")
    ap.add_argument("--out", default="views_dims.svg")
    ap.add_argument("--unit-scale", type=float, default=1.0,
                    help="STL unit multiplier (e.g., STL in meters -> 1000 for mm)")
    ap.add_argument("--decimals", type=int, default=2)
    ap.add_argument("--swap-yz", action="store_true",
                    help="Swap Y and Z axes before generating views")
    ap.add_argument("--repetitions", type=int, default=10,
                    help="Repetitions along length for pitch = L/n (default: 10)")
    args = ap.parse_args()

    poly = load_stl_polydata(args.stl)
    if args.swap_yz:
        poly = swap_yz_polydata(poly)

    bounds = poly.GetBounds()
    xmin, xmax, ymin, ymax, zmin, zmax = bounds

    dx = (xmax - xmin) * args.unit_scale
    dy = (ymax - ymin) * args.unit_scale
    dz = (zmax - zmin) * args.unit_scale

    nrep = max(1, int(args.repetitions))
    pitch = dx / nrep

    # A4 landscape (mm)
    W, H = 297, 210
    dwg = svgwrite.Drawing(args.out, size=(f"{W}mm", f"{H}mm"), viewBox=f"0 0 {W} {H}")

    dwg.add(dwg.text(f"STL views + dimensions (unit scale x{args.unit_scale})",
                     insert=(10, 14), font_size=TITLE_FONT))

    margin = 10
    gap = 8
    header_h = 18
    top_y = margin + header_h

    col_w = (W - 2 * margin - gap) / 2.0
    usable_h = H - top_y - margin
    top_h = min(60.0, usable_h * 0.35)
    bot_h = usable_h - gap - top_h

    boxes = {
        "xy": (margin, top_y, col_w, top_h),
        "note": (margin + col_w + gap, top_y, col_w, top_h),
        "xz": (margin, top_y + top_h + gap, col_w, bot_h),
        "yz": (margin + col_w + gap, top_y + top_h + gap, col_w, bot_h),
    }

    view_labels = {"xy": "Top view (XY)", "xz": "Side view (XZ)", "yz": "End view (YZ)"}
    view_dims = {"xy": (dx, dy), "xz": (dx, dz), "yz": (dy, dz)}

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

    # Notes box
    nbx, nby, nbw, nbh = boxes["note"]
    dwg.add(dwg.rect(insert=(nbx, nby), size=(nbw, nbh), fill="none", stroke="black", stroke_width=0.2))
    dwg.add(dwg.text("Notes", insert=(nbx + 2, nby + 7), font_size=LABEL_FONT))
    dwg.add(dwg.text(f"Drawing scale: {global_scale:.4f} mm/mm",
                     insert=(nbx + 2, nby + 16), font_size=DIM_FONT))

    fmt = f"{{:.{args.decimals}f}}"

    for v in ("xy", "xz", "yz"):
        bx, by, bw, bh = boxes[v]
        dwg.add(dwg.rect(insert=(bx, by), size=(bw, bh), fill="none", stroke="black", stroke_width=0.2))
        dwg.add(dwg.text(view_labels[v], insert=(bx + 2, by + 7), font_size=LABEL_FONT))

        if v == "xy":
            polylines2d = outline_rect_xy(bounds, args.unit_scale)
        elif v == "yz":
            polylines2d = outline_rect_yz(bounds, args.unit_scale)
        else:
            polylines2d = outline_side_xz_clean(poly, bounds, args.unit_scale)

        fitted, (x1, y1, x2, y2) = fit_polylines_to_box(
            polylines2d, bx, by, bw, bh, pad=pad, scale=global_scale, center=True
        )

        for polyline in fitted:
            if len(polyline) >= 2:
                dwg.add(dwg.polyline(polyline, fill="none", stroke="black", stroke_width=0.35))

        w_dim, h_dim = view_dims[v]
        w_text = f"{fmt.format(w_dim)} mm"
        h_text = f"{fmt.format(h_dim)} mm"

        off_h = min(10.0, (by + bh) - y2 - 3.0)
        off_v = min(10.0, (bx + bw) - x2 - 3.0)
        off_h = max(5.0, off_h)
        off_v = max(5.0, off_v)

        add_dim_h(dwg, x1, x2, y2, offset=off_h, text=w_text)
        add_dim_v(dwg, y1, y2, x2, offset=off_v, text=h_text)

        if v == "xz":
            note = f"Pitch = {fmt.format(pitch)} mm (L/{nrep})"
            add_text_halo(dwg, note, insert=(bx + 2, by + 16), font_size=DIM_FONT, text_anchor="start")

    dwg.save()
    print("Generated:", args.out)


if __name__ == "__main__":
    main()