module DiffMoM

using LinearAlgebra: LinearAlgebra,
                     Adjoint,
                     Hermitian,
                     I,
                     SymTridiagonal,
                     Transpose,
                     axpy!,
                     cross,
                     dot,
                     eigen,
                     eigvals,
                     issuccess,
                     ldiv!,
                     lu,
                     lu!,
                     mul!,
                     norm,
                     opnorm,
                     rmul!,
                     svdvals
using SparseArrays: SparseArrays,
                    AbstractSparseMatrix,
                    SparseMatrixCSC,
                    dropzeros!,
                    nnz,
                    nonzeros,
                    nzrange,
                    rowvals,
                    sparse
using StaticArrays: @SMatrix, MVector, SMatrix, SVector
using Random: Random, randperm
using Krylov: Krylov
using SpecialFunctions: besselh,
                        besselj,
                        besseljx,
                        bessely,
                        erfc,
                        erfcx,
                        sphericalbesselj
using FFTW: FFTW
using IncompleteLU: IncompleteLU

include("Types.jl")

# Geometry
include("geometry/Mesh.jl")
include("geometry/MeshIO.jl")

# Basis functions & quadrature
include("basis/RWG.jl")
include("basis/Quadrature.jl")
include("basis/Greens.jl")
include("basis/PeriodicGreens.jl")

# Assembly
include("assembly/SingularIntegrals.jl")
include("assembly/EFIE.jl")
include("assembly/Impedance.jl")
include("assembly/Excitation.jl")
include("assembly/CompositeOperator.jl")
include("assembly/SpatialPatches.jl")
include("assembly/PeriodicEFIE.jl")
include("assembly/DensityInterpolation.jl")

# Fast methods
include("fast/ClusterTree.jl")
include("fast/ACA.jl")
include("fast/Octree.jl")
include("fast/MLFMA.jl")

# Post-processing (FarField needed by QMatrix)
include("postprocessing/FarField.jl")
include("postprocessing/NearField.jl")

# Optimization objectives
include("optimization/QMatrix.jl")

# Solvers
include("solver/Solve.jl")
include("solver/NearFieldPreconditioner.jl")
include("solver/IterativeSolve.jl")

# Adjoint & optimization
include("optimization/Adjoint.jl")
include("optimization/Verification.jl")
include("optimization/Optimize.jl")
include("optimization/MultiAngleRCS.jl")
include("optimization/DensityFiltering.jl")
include("optimization/DensityAdjoint.jl")

# 2D TM Volume Integral Equation (MoM)
include("mom2d/Types2D.jl")
include("mom2d/Greens2D.jl")
include("mom2d/Assembly2D.jl")
include("mom2d/Excitation2D.jl")
include("mom2d/Scatter2D.jl")
include("mom2d/Mie2D.jl")

# 3D vector material volume solver (DDA / VIE-style)
include("mom3d/Types3D.jl")
include("mom3d/MaterialModels3D.jl")
include("mom3d/DDA3D.jl")
include("mom3d/EMDDA3D.jl")
include("mom3d/Adjoint3D.jl")
include("mom3d/FFTDDA3D.jl")
include("mom3d/SurfaceIE3D.jl")

# Workflow
include("Workflow.jl")

# Post-processing (remaining)
include("postprocessing/Diagnostics.jl")
include("postprocessing/PhysicalOptics.jl")
include("postprocessing/PTD.jl")
include("postprocessing/Mie.jl")
include("postprocessing/Visualization.jl")
include("postprocessing/PeriodicMetrics.jl")
include("assembly/GroundedEFIE.jl")

end # module
