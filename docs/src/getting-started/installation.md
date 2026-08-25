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

- A Julia version allowed by the `julia` entry under `[compat]` in the
  repository-root `Project.toml`.
- A local checkout of the repository for the examples and validation scripts.

Check Julia version:

```bash
julia --version
```

---

## 2) Local-Checkout Installation

From your shell:

```bash
cd /path/to/DiffMoM.jl
julia --project=. -e 'import Pkg; Pkg.instantiate()'
```

This resolves the dependencies declared by `Project.toml` within its compatibility
bounds. The package does not commit a manifest, so a fresh checkout resolves a
compatible environment for the active Julia version.

---

## 3) Install by URL (Alternative)

If you want to add the package from GitHub in a general Julia environment:

```julia
import Pkg
Pkg.add(url="https://github.com/jake-w-liu/DiffMoM.jl")
```

For tutorial reproducibility, use the local project environment.

---

## 4) Sanity Check

From the repository root:

```bash
julia --project=. -e 'using DiffMoM; println("DiffMoM loaded")'
```

Then run the regression suite:

```bash
julia --project=. --startup-file=no -e 'using Pkg; Pkg.test()'
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

Each script prints the paths of any generated artifacts. Output locations are
owned by the script because some workflows target package data while others
target paper-specific directories.

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
