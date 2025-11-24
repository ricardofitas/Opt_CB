<h1 align="center">Opt_CB: Optimization of Corrugated Boards</h1>

<div align="center">
  <span class="author-block">
    <a href="https://github.com/ricardofitas">Ricardo Fitas</a>
  </span>
</div>

$~$

<p align="center">
  <!-- Placeholder: replace with an overview figure of the corrugated board optimization pipeline -->
  <img src="figures/opt_cb_overview_placeholder.png" alt="Placeholder for Opt_CB overview figure" width="700">
</p>

## 🎯 Overview

This repository contains research code for the **optimization of corrugated board geometries** using:

- **NURBS-based parameterizations** of the flute profile
- **Analytical and numerical mechanics formulations** for effective properties
- **Multi-objective metaheuristics** (MOPSO, NSGA-II, etc.) and post-processing tools

The goal is to explore trade-offs between **mechanical performance** (stiffness, classification, effective properties) and **material usage / geometry** for corrugated boards.

> ⚠️ **License reminder**  
> This repository is distributed under a **source-available, no-modification license**.  
> Please read [LICENSE.txt](./LICENSE.txt) carefully before using the code.

---

## ✨ Highlights

- 🌀 **NURBS-based flute geometry**  
  Parameterizes the corrugated core as a smooth NURBS curve, with distances and radii encoded in a compact design vector.

- 📐 **Effective property and classification analysis**  
  Scripts for computing effective orthotropic properties, mass, inertia, and **classification** of optimized boards into categories (A–G, etc.), suitable for engineering studies.

- 🎯 **Multi-objective optimization engine**  
  Standalone implementations of **Multi-Objective Particle Swarm Optimization (MOPSO)** and **NSGA-II**, adapted for corrugated board design spaces.

- 🧱 **Geometry export to CAD / meshing tools**  
  Utilities to export flute geometries to **STL** (and example Gmsh geometry generation) for downstream FEM and CAD workflows.

- 🔬 **Research-oriented, script-based workflow**  
  The code is structured as **explicit research scripts**, making the full pipeline transparent for academic inspection.

<p align="center">
  <!-- Placeholder: replace with a figure showing geometry parametrization or Pareto front -->
  <img src="figures/opt_cb_pareto_placeholder.png" alt="Placeholder for Pareto front or geometry figure" width="600">
</p>

---

## 📁 Project Structure

```text
.
├── Optimization_CB_v2.py             # NURBS-based corrugated board optimizer (core analytical formulation)
├── Optimization_CB_v2_prod.py        # Production-oriented optimizer with smoothing, plotting, and post-processing hooks
├── Optimization_CB_v2_sine.py        # Baseline sinusoidal-wave profile optimization
├── Optimization_CB_v2_sq.py          # Square-wave profile variant
├── Optimization_CB.py                # Early analytical formulation / reference implementation
├── Optimization_CB_2_liners.py       # Optimizations for 2-liner configurations
├── Optimization_CB_rand_nurbs.py     # Experiments with randomized NURBS initializations
│
├── OCB_analysis.py                   # Post-processing & classification of optimized designs (A–G classes, etc.)
├── OCB_analysis_prod_fix.py          # Extended/production-ready analysis with NURBS reconstruction
│
├── MOPSO_v3.py                       # Standalone multi-objective PSO engine
├── MOPSO_v3_2_liners.py              # MOPSO variant targeting 2-liner geometries
├── MOPSO_nurbs.py                    # MOPSO driver tailored to NURBS-based flute profiles
│
├── MOETPSO/                          # Multi-objective evolutionary & PSO utilities
│   ├── MOPSO.py                      # Generic MOPSO implementation
│   ├── NSGA.py                       # NSGA-II implementation (DEAP-based)
│   ├── FEM.py                        # Gmsh-based FEM geometry/meshing prototype
│   ├── EPSO_analysis.py              # EPSO/PSO result aggregation and scaling utilities
│   ├── GVRP.py                       # Additional optimization utilities / experiments
│   └── Paper_combinations_CB.py      # Paper configuration & feature scaling helper
│
├── STL/
│   ├── stl.py                        # Convert 2D flute curve to extruded 3D STL geometry
│   ├── local_curve_1.txt             # Sample local curve input
│   ├── local_curve_2.txt
│   ├── local_curve_3.txt
│   ├── 1.png                         # Placeholder renderings of sample geometries
│   ├── 2.png
│   └── 3.png
│
├── Drawing_curves.py                 # Utilities for plotting generated profiles
├── nurbs_vs_fillet_geometry.py       # Comparison between NURBS curve and filleted CAD geometry
├── profile_drawing.py                # Additional profile drawing helpers
├── test1_NURBS.py                    # Simple NURBS experiments / sanity checks
├── test_effective_calc.py            # Checks for effective property calculations
├── weight_inertia.py                 # Mass and inertia computations for board configurations
│
├── LICENSE.txt                       # Source-available, no-modification license
└── README.md                         # This file
