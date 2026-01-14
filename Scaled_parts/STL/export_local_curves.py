# -*- coding: utf-8 -*-
"""
Export local interpolation points of the NURBS-based profile to TXT files.

This script implements the same curve construction logic as Drawing_curves.py,
but instead of (only) plotting, it writes the *local* curve segment to files
named: local_curve_1.txt, local_curve_2.txt, local_curve_3.txt, ...

TXT format:
x y
0.000000 2.350818
0.004125 2.345915
...

Dependencies:
  pip install numpy geomdl

Usage:
  - Edit the `cases` dict at the bottom (each value is a list of 7 parameters)
  - Run: python export_local_curves.py
"""

from __future__ import annotations

import numpy as np
from geomdl import NURBS


def _default_full_opt_process(params: list[float]) -> tuple[float, float, float, float]:
    """
    Placeholder for the 'full_opt_process' used in Drawing_curves.py.

    In Drawing_curves.py it is mocked to avoid constraints. If you have the
    real optimizer, replace this function (or pass your own into
    `generate_local_curve_points`).
    """
    return 1.0, 1.0, params[-2], params[-1]


def generate_local_curve_points(
    X: list[float],
    *,
    lambda_: float,
    Amp: float,
    sample_size: int = 2000,
    make_x_start_at_zero: bool = True,
    full_opt_process=_default_full_opt_process,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the NURBS curve from the 7-parameter list X and return local (x,y) points.

    X meaning (as used in Drawing_curves.py):
      X[0:5] -> relative distances (d1..d5 after normalization by their sum)
      X[5], X[6] -> parameters used to compute weights r1, r2 (via exp mapping)

    Returns:
      x_local, y_local as 1D numpy arrays (same length).
    """
    if len(X) != 7:
        raise ValueError(f"Expected 7 parameters, got {len(X)}")

    # Normalize distances
    suma = X[0] + X[1] + X[2] + X[3] + X[4]
    if suma == 0:
        raise ValueError("Sum of the first five parameters must be non-zero.")
    d1, d2, d3, d4, d5 = (X[i] / suma for i in range(5))

    # Map last two parameters to weights (same formula as Drawing_curves.py)
    r1 = 0.0103 * np.exp(9.17 * X[5]) + 0.1
    r2 = 0.0103 * np.exp(9.17 * X[6]) + 0.1

    # Optional optimization hook (kept to mirror original structure)
    _r1_opt, _r2_opt, _X5_opt, _X6_opt = full_opt_process([d1, d2, d3, d4, d5, X[5], X[6]])

    # Control points (same structure as Drawing_curves.py)
    AAA = [
        0,
        d1,
        d1 + d2,
        d1 + d2 + d3,
        d1 + d2 + d3 + d4,
        1 + d1,
        1 + d1 + d2,
        1 + d1 + d2 + d3,
        1 + d1 + d2 + d3 + d4,
        2,
    ]
    BBB = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0]
    weights = np.array([1, r1, r1, r2, r2, r1, r1, r2, r2, 1], dtype=float)
    ctrlpts = np.column_stack((AAA, BBB))

    curve = NURBS.Curve()
    curve.degree = 2
    curve.ctrlpts = ctrlpts.tolist()
    curve.weights = weights.tolist()

    # Uniform-ish clamped knot vector (same approach as Drawing_curves.py)
    num_ctrlpts = len(ctrlpts)
    degree = curve.degree
    num_knots = num_ctrlpts + degree + 1
    interior_knots = np.linspace(0, 1, num_knots - 2 * (degree + 1) + 2)[1:-1]
    curve.knotvector = np.concatenate(
        [np.zeros(degree + 1), interior_knots, np.ones(degree + 1)]
    ).tolist()

    curve.sample_size = int(sample_size)
    curve.evaluate()
    curve_points = np.array(curve.evalpts, dtype=float)  # shape: (N, 2)

    n_points = curve_points.shape[0]

    # Segment indices (same formula as Drawing_curves.py)
    t1 = int(((d1 + d2 + d3 / 2) / 2) * n_points)
    t2 = int(((1 + d1 + d2 + d3 / 2) / 2) * n_points)

    if not (0 <= t1 < t2 <= n_points):
        raise RuntimeError(f"Bad segment indices: t1={t1}, t2={t2}, n={n_points}")

    # Scale to target wavelength/amplitude
    l = abs(curve_points[t2 - 1, 0] - curve_points[t1, 0])
    if l == 0:
        raise RuntimeError("Local segment has zero length in x; cannot scale.")
    scale1 = lambda_ / l

    yseg = curve_points[t1:t2, 1]
    height = float(np.max(yseg) - np.min(yseg))
    if height == 0:
        raise RuntimeError("Local segment has zero height in y; cannot scale.")
    scale2 = Amp / height

    x_local = scale1 * curve_points[t1:t2, 0]
    y_local = scale2 * curve_points[t1:t2, 1]

    # Many CAD/FE pipelines expect local x to start at 0
    if make_x_start_at_zero:
        x_local = x_local - x_local[0]

    return x_local, y_local


def write_xy_txt(path: str, x: np.ndarray, y: np.ndarray, *, decimals: int = 6) -> None:
    """
    Write x,y arrays to a txt file with header 'x y' and fixed decimal formatting.
    """
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")
    fmt = f"%.{decimals}f %.{decimals}f"
    with open(path, "w", encoding="utf-8") as f:
        f.write("x y\n")
        for xi, yi in zip(x, y):
            f.write((fmt % (float(xi), float(yi))) + "\n")


def export_cases_to_txt(
    cases: dict[str, list[float]],
    *,
    lambda_: float,
    Amp: float,
    out_prefix: str = "local_curve_",
    out_suffix: str = ".txt",
    sample_size: int = 2000,
) -> None:
    """
    Export each case (7 params) to local_curve_X.txt, where X is 1..N
    based on insertion order of `cases`.
    """
    for i, (name, X) in enumerate(cases.items(), start=1):
        x_local, y_local = generate_local_curve_points(
            X,
            lambda_=lambda_,
            Amp=Amp,
            sample_size=sample_size,
            make_x_start_at_zero=True,
        )
        out_path = f"{out_prefix}{i}{out_suffix}"
        write_xy_txt(out_path, x_local, y_local, decimals=6)
        print(f"[OK] {name} -> {out_path} ({len(x_local)} points)")


if __name__ == "__main__":
    # === Set your physical scaling here (same constants appear in Drawing_curves.py) ===
    lambda_ = 5.65e-3 * 10 / 2.65  # wavelength [m] (as in the later sine script block)
    Amp = 2.65e-3 * 10 / 2.65      # amplitude [m]

    # === Provide your 7-parameter cases here ===
    cases = {
        "Caso 1": [1.791214375, 0.334639907, 8.487148517, 1.173761631, 10, 0, 0],
        "Caso 2": [0.1, 9.645485035, 0.1, 10, 0.1, 1, 1],
        "Caso 3": [1.383607899, 2.549796399, 5.818867374, 6.456406202, 5.142503464, 0.373052427, 0.471046705],
    }

    export_cases_to_txt(cases, lambda_=lambda_, Amp=Amp, sample_size=2000)
