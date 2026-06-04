# Suggested Reading Paths

## Path A: New to Computational EM

**Prerequisite check**: If you need refreshers on vector calculus, complex numbers, or linear algebra, start with [Mathematical Prerequisites](mathematical-prerequisites.md).

1. Getting Started
2. Part I — Foundations (all chapters, including fundamentals/06 Excitation Theory)
3. Part II — Package Fundamentals
4. Tutorial 01 (First PEC Plate)
5. validation/06 (Near/Total-Field Rayleigh Sphere)
6. Tutorial 04 (Sphere–Mie benchmark)
7. methods/01 (GMRES & Preconditioning)

Goal: build numerical intuition before optimization.

---

## Path B: Interested in Inverse Design

1. Part I Chapters 2–5 + fundamentals/06 (Excitation Theory)
2. Part III — Differentiable Design (all chapters)
3. methods/01 (GMRES) → methods/05 (solve_scattering Workflow)
4. Tutorial 02 (Adjoint gradient check)
5. Tutorial 03 (Beam steering design)
6. Part V — Validation

Goal: trust gradients, then optimize.

---

## Path C: CAD/Platform Scattering User

1. Part II Chapter 1 (Mesh pipeline)
2. fundamentals/06 (Excitation Theory)
3. Tutorial 06 (OBJ repair workflow)
4. Tutorial 05 (Airplane RCS)
5. methods/02 (ACA H-Matrices) → methods/05 (solve_scattering Workflow)
6. Part VI Chapters 1, 2, and 4

Goal: robust imported-mesh simulation with realistic resource limits.

---

## Path D: API-First Developer

1. API Overview
2. API Types, Mesh, RWG
3. API Assembly/Solve + Fields/RCS
4. API Adjoint/Optimization + Verification
5. API Excitation + ACA Workflow
6. fundamentals/06 (Excitation Theory)
7. methods/ (all chapters)
8. Source modules in `src/` as needed

Goal: implement custom workflows quickly.

---

## Path E: Validation-Focused Reviewer

1. Part V — Validation
2. Tutorial 04 (Sphere–Mie)
3. validation/06 (Near/Total-Field Rayleigh Sphere)
4. Tutorial 03 (Beam steering)
5. API Verification

Goal: reproduce consistency gates and benchmark evidence.
