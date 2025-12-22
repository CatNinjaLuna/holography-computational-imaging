# Physics-Grounded Holography: From Wave Optics to Learning-Based Reconstruction

This repository explores **holographic phase recovery and image reconstruction** through a structured, physics-first pipeline:

1. Physical modeling and simulation
2. Classical inverse problem methods
3. Learning-based reconstruction with physics constraints
4. Downstream applications in 3D perception

The goal is to understand **what holography encodes physically**, how phase recovery can be solved as an inverse problem, and how modern learning-based methods compare to (and integrate with) classical optics-based approaches.

This project is intentionally organized by _scientific progression_, not by tools.

---

## Project Motivation

Holography is a canonical example of a **physics-constrained inverse problem**:

-  Sensors measure only intensity
-  Phase information is lost
-  Reconstruction requires physical assumptions or priors

Recent deep learning approaches demonstrate impressive results but often remove explicit physical constraints. This repository studies **both sides**:

-  What classical physics guarantees
-  What learning-based models can infer
-  Where each approach succeeds or fails

---

## Repository Structure

-  **Phase 1 — Physics & Simulation**

   -  Forward wave propagation (Fresnel diffraction)
   -  Numerical hologram generation
   -  Analytical and numerical validation

-  **Phase 2 — Inverse Problems**

   -  Phase retrieval under noise and defocus
   -  Classical solvers (GS, TIE, optimization-based)
   -  Failure modes and stability analysis

-  **Phase 3 — Learning-Based Reconstruction**

   -  Neural networks for phase recovery
   -  Physics-based forward models in the loss
   -  Benchmarks against classical solvers

-  **Phase 4 — Applications**
   -  3D reconstruction and depth estimation
   -  Point clouds or volumetric reconstructions
   -  Robotics or microscopy-oriented use cases

Each phase is self-contained and includes its own documentation.

---

## Design Principles

-  Physics before learning
-  Explicit forward models
-  Quantitative evaluation
-  Clear comparison between methods
-  Reproducibility over performance chasing

---

## Intended Audience

This repository is suitable for:

-  Computational imaging researchers
-  Applied optics / physics students
-  Robotics perception researchers
-  ML researchers working on inverse problems

---

## Status

Active research / exploration repository.  
Each phase can be read and used independently.
