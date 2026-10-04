module DiffMoM

using LinearAlgebra: LinearAlgebra,
                     Adjoint,
                     Diagonal,
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
using Random: Random
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
using TOML: TOML
using SHA: SHA
import EzXML
import XML2_jll

include("Types.jl")

# Geometry
include("geometry/Mesh.jl")
include("geometry/MeshIO.jl")

# Basis functions & quadrature
include("basis/RWG.jl")
include("basis/NestedRWG.jl")
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
include("solver/RetainedSolve.jl")
include("Workflow.jl")
include("error_estimation/GalerkinError.jl")
include("error_estimation/Conditioning.jl")
include("error_estimation/Calibration.jl")

# Post-processing (remaining)
include("postprocessing/Diagnostics.jl")
include("postprocessing/PhysicalOptics.jl")
include("postprocessing/PTD.jl")
include("postprocessing/Mie.jl")
include("postprocessing/Visualization.jl")
include("postprocessing/PeriodicMetrics.jl")
include("assembly/GroundedEFIE.jl")

# Shielded planar layered-media MoM (spectral box modes + rooftop basis)
include("planar/PlanarTypes.jl")
include("planar/PlanarImmittance.jl")
include("planar/PlanarPowerWaves.jl")
include("planar/PlanarBasis.jl")
include("planar/PlanarVias.jl")
include("planar/PlanarVolumes.jl")
include("planar/PlanarGreens.jl")
include("planar/PlanarConductorLoss.jl")
include("planar/PlanarSolve.jl")
include("planar/PlanarUFFT.jl")
include("planar/PlanarFFTAssembly.jl")
include("planar/PlanarTriangleTransforms.jl")
include("planar/PlanarConformal.jl")
include("planar/PlanarConformalMeshing.jl")
include("planar/PlanarConformalUFFT.jl")
include("planar/PlanarConformalProjection.jl")
include("planar/PlanarConformalMultiProjection.jl")
include("planar/PlanarConformalDefect.jl")
include("planar/PlanarHybrid.jl")
include("planar/PlanarHybridUFFT.jl")
include("planar/PlanarCurrents.jl")
include("planar/PlanarSweep.jl")
include("planar/PlanarSurface.jl")
include("planar/PlanarMetalModel.jl")
include("planar/PlanarDeembed.jl")
include("planar/PlanarExtract.jl")
include("planar/PlanarCircuit.jl")
include("planar/PlanarSpiceIO.jl")
include("planar/PlanarSpectreIO.jl")
include("planar/PlanarCalibration.jl")
include("planar/PlanarCalibrationStandards.jl")
include("planar/PlanarTerminalReturns.jl")
include("planar/PlanarFloatingBridge.jl")
include("planar/PlanarPortContraction.jl")
include("planar/PlanarAxialRefinement.jl")
include("planar/PlanarSubdivision.jl")
include("planar/PlanarVectorFit.jl")
include("planar/PlanarNetworkIO.jl")
include("planar/PlanarOutputs.jl")
include("planar/PlanarAdjoint.jl")
include("planar/PlanarGeometry.jl")
include("planar/PlanarLibrary.jl")
include("planar/PlanarLayout.jl")
include("planar/PlanarConformalLayout.jl")
include("planar/PlanarProject.jl")
include("planar/PlanarProjectModel.jl")
include("planar/PlanarProjectCircuitNodes.jl")
include("planar/PlanarProjectPhysical.jl")
include("planar/PlanarProjectFloating.jl")
include("planar/PlanarProjectSolve.jl")
include("planar/PlanarConnectivity.jl")
include("planar/PlanarSonnetIO.jl")
include("planar/PlanarSonnetScalarFiles.jl")
include("planar/PlanarSonnetTechnology.jl")
include("planar/PlanarSonnetModelFiles.jl")
include("planar/PlanarSonnetConformal.jl")
include("planar/PlanarSonnetComponents.jl")
include("planar/PlanarSonnetFloating.jl")
include("planar/PlanarLayoutIO.jl")
include("planar/PlanarArtworkIO.jl")
include("planar/PlanarArtworkAffine.jl")
include("planar/PlanarArtworkSweep.jl")
include("planar/PlanarGerberIO.jl")
include("planar/PlanarODBCompression.jl")
include("planar/PlanarODBIO.jl")
include("planar/PlanarODBSymbolsExtra.jl")
include("planar/PlanarArtworkCertified.jl")
include("planar/PlanarArtworkExact.jl")
include("planar/PlanarArtworkLinearExact.jl")
include("planar/PlanarArtworkRegionExact.jl")
include("planar/PlanarArtworkBounds.jl")
include("planar/PlanarPlots.jl")
include("planar/PlanarRadiation.jl")

end # module
