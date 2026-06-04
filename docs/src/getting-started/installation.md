# Installation

## Purpose

Set up a reproducible local environment so you can run forward MoM solves,
adjoint gradients, and all validation examples without hidden configuration.

---

## Learning Goals

After this chapter, you should be able to:

1. Install and instantiate the package environment.
2. Verify that the package imports and tests pass.
3. Run example scripts from the repository root.

---

## 1) Prerequisites

- Julia `1.10` or newer.
- A local checkout of the repository (recommended for learning/tutorial use).

Check Julia version:

```bash
julia --version
```

---

## 2) Local-Checkout Installation (Recommended)

From your shell:

```bash
cd /path/to/DiffMoM.jl
julia --project=. -e 'import Pkg; Pkg.instantiate()'
```

This installs all dependencies pinned by `Project.toml` in the project
environment.

---

## 3) Install by URL (Alternative)

If you want to add the package from GitHub in a general Julia environment:

```julia
import Pkg
Pkg.add(url="https://github.com/jake-w-liu/DiffMoM.jl")
```

For tutorial reproducibility, the local project environment is still preferred.

---

## 4) Sanity Check

From the repository root:

```bash
julia --project=. -e 'using DiffMoM; println("DiffMoM loaded")'
```

Then run the regression suite:

```bash
julia --project=. test/runtests.jl
```

If this passes, your installation is correct.

---

## 5) Optional: Build Documentation Locally

```bash
julia --project=docs docs/make.jl
```

Documenter output is generated under `docs/build/`.

---

## 6) First Example Runs

From the repository root:

```bash
julia --project=. examples/03_beamsteering_physical_unitcell.jl
julia --project=. examples/05_solver_methods.jl
julia --project=. examples/04_pec_sphere_mie.jl
```

Generated numeric artifacts are written into `data/` and figures into `figs/`.

---

## Troubleshooting

- **`Package ... not found`**: run `Pkg.instantiate()` in the same project.
- **Slow first run**: Julia compiles methods on first execution.
- **Plot backend errors**: install a plotting-capable environment (headless CI
  may need an alternative backend).

---

## Code Mapping

- Package metadata and compat bounds: `Project.toml`
- Main module and exports: `src/DiffMoM.jl`
- Test entry point: `test/runtests.jl`
- Docs build entry point: `docs/make.jl`

---

## Exercises

- Basic: instantiate the project and run only `examples/01_pec_plate_basics.jl`.
- Challenge: run `test/runtests.jl`, then identify which generated CSV files
  correspond to each major validation gate.
