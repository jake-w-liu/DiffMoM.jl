import Pkg
Pkg.activate(@__DIR__)
Pkg.develop(Pkg.PackageSpec(path = joinpath(@__DIR__, "..")))
Pkg.instantiate()

using Documenter
using DiffMoM

makedocs(
    sitename = "DiffMoM.jl",
    authors = "Jake W. Liu",
    modules = [DiffMoM],
    checkdocs = :none,
    remotes = nothing,
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        mathengine = Documenter.MathJax3(),
        edit_link = nothing,
        repolink = nothing,
    ),
    pages = [
        "Home" => "index.md",
        "Getting Started" => [
            "Installation" => "getting-started/installation.md",
            "Quickstart" => "getting-started/quickstart.md",
            "Conventions and Units" => "getting-started/conventions.md",
        ],
        "Part I — Foundations" => [
            "Why Integral Equations" => "fundamentals/01-why-integral-equations.md",
            "EFIE and Boundary Conditions" => "fundamentals/02-efie-boundary-conditions.md",
            "MoM and RWG Discretization" => "fundamentals/03-mom-rwg-discretization.md",
            "Singular Integration" => "fundamentals/04-singular-integration.md",
            "Conditioning and Preconditioning" => "fundamentals/05-conditioning-preconditioning.md",
            "Excitation Theory and Usage" => "fundamentals/06-excitation-theory-and-usage.md",
        ],
        "Part II — Package Fundamentals" => [
            "Mesh Pipeline" => "package-basics/01-mesh-pipeline.md",
            "Forward Pipeline" => "package-basics/02-forward-pipeline.md",
            "Field Postprocessing, Q, and RCS" => "package-basics/03-farfield-q-rcs.md",
            "Visualization" => "package-basics/04-visualization.md",
        ],
        "Part III — Differentiable Design" => [
            "Adjoint Method" => "differentiable-design/01-adjoint-method.md",
            "Impedance Sensitivities" => "differentiable-design/02-impedance-sensitivities.md",
            "Ratio Objectives" => "differentiable-design/03-ratio-objectives.md",
            "Optimization Workflow" => "differentiable-design/04-optimization-workflow.md",
            "Multi-Angle RCS" => "differentiable-design/05-multiangle-rcs.md",
        ],
        "Part IV — Methods" => [
            "GMRES and Near-Field Preconditioning" => "methods/01-gmres-and-nf-preconditioning.md",
            "ACA H-Matrix Compression" => "methods/02-aca-hmatrix-compression.md",
            "Physical Optics" => "methods/03-physical-optics.md",
            "Matrix-Free Operators" => "methods/04-matrix-free-operators.md",
            "solve_scattering Workflow" => "methods/05-solve-scattering-workflow.md",
            "MLFMA" => "methods/06-mlfma.md",
            "Periodic EFIE and Floquet Metrics" => "methods/07-periodic-efie-and-floquet-metrics.md",
            "Density Topology Optimization" => "methods/08-density-topology-optimization.md",
        ],
        "Part V — Validation" => [
            "Internal Consistency" => "validation/01-internal-consistency.md",
            "Gradient Verification" => "validation/02-gradient-verification.md",
            "Bempp Cross-Validation" => "validation/03-bempp-cross-validation.md",
            "Sphere-vs-Mie Benchmark" => "validation/04-sphere-mie-benchmark.md",
            "Meep Open-Source Cross-Validation" => "validation/05-meep-cross-validation.md",
            "Near/Total-Field Rayleigh Sphere" => "validation/06-near-total-field-rayleigh-sphere.md",
        ],
        "Part VI — Advanced Workflows" => [
            "Complex OBJ Platforms" => "advanced/01-complex-obj-platforms.md",
            "Large-Problem Strategy" => "advanced/02-large-problem-strategy.md",
            "Robustness Studies" => "advanced/03-robustness-studies.md",
            "Debugging Playbook" => "advanced/04-debugging-playbook.md",
        ],
        "Tutorials" => [
            "First PEC Plate" => "tutorials/01-first-pec-plate.md",
            "Adjoint Gradient Check" => "tutorials/02-adjoint-gradient-check.md",
            "Beam Steering Design" => "tutorials/03-beam-steering-design.md",
            "Sphere-Mie RCS" => "tutorials/04-sphere-mie-rcs.md",
            "Airplane RCS" => "tutorials/05-airplane-rcs.md",
            "OBJ Repair Workflow" => "tutorials/06-obj-repair-workflow.md",
        ],
        "API Reference" => [
            "Overview" => "api/overview.md",
            "Types" => "api/types.md",
            "Mesh Utilities" => "api/mesh.md",
            "RWG Utilities" => "api/rwg.md",
            "Excitation" => "api/excitation.md",
            "Assembly and Solve" => "api/assembly-solve.md",
            "ACA Workflow" => "api/aca-workflow.md",
            "Octree" => "api/octree.md",
            "MLFMA" => "api/mlfma.md",
            "Fields and RCS" => "api/farfield-rcs.md",
            "Periodic Methods" => "api/periodic-methods.md",
            "Composite Operators" => "api/composite-operators.md",
            "Spatial Patches" => "api/spatial-patches.md",
            "Adjoint and Optimization" => "api/adjoint-optimize.md",
            "Density Topology" => "api/density-topology.md",
            "Verification" => "api/verification.md",
            "Visualization" => "api/visualization.md",
            "Physical Optics" => "api/physical-optics.md",
        ],
        "Appendices" => [
            "Units and Conventions" => "appendices/units-conventions.md",
            "Mathematical Prerequisites" => "appendices/mathematical-prerequisites.md",
            "Symbol Glossary" => "appendices/symbol-glossary.md",
            "Suggested Reading Paths" => "appendices/reading-path.md",
            "FAQ" => "appendices/faq.md",
        ],
    ],
)

# deploydocs(
#     repo = "github.com/jake-w-liu/DiffMoM.jl.git",
# )
