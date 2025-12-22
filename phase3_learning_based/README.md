# Phase 3 — Learning-Based Phase Recovery

This phase introduces **learning-based methods** for holographic reconstruction while retaining explicit physical structure.

The goal is not to replace physics, but to **augment inverse problem solving** with data-driven priors.

---

## Objectives

-  Train neural networks to recover phase from holograms
-  Embed physical forward models into the training loss
-  Compare learning-based methods against classical solvers

---

## Model Design

-  UNet-style architectures
-  Input: hologram or back-propagated complex field
-  Output: phase (or complex field)
-  Loss functions include:
   -  Intensity consistency via forward propagation
   -  Phase smoothness or regularization
   -  Reconstruction fidelity

---

## Physics-Based Loss

Rather than supervising only on phase:

-  Predicted phase is forward-propagated
-  Resulting intensity is compared to the measured hologram

This enforces **wave-consistent learning**.

---

## Benchmarks

-  GS vs TIE vs optimization
-  Learning-based vs classical under noise
-  Generalization outside training distribution

This phase investigates **when learning helps — and when it does not**.
