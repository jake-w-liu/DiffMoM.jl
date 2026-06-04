# 21_meep_validation_figure.jl — Generate MEEP vs MoM comparison figure
#
# Reads results_meep_validation.csv and generates a bar-chart comparison figure
# showing reflectance agreement between the periodic MoM solver and Meep FDTD.
#
# Run: julia --project=. examples/21_meep_validation_figure.jl

using CSV, DataFrames
using PlotlySupply
using PlotlyKaleido
import PlotlySupply: savefig
PlotlyKaleido.start(mathjax=false)

const PKG_DIR = dirname(@__DIR__)
const DATA_DIR = joinpath(PKG_DIR, "..", "data")
const FIG_DIR = joinpath(PKG_DIR, "..", "figures")

# IEEE figure constants
const COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
const DASHES = ["solid", "dash", "dashdot", "dot"]
const IEEE_SINGLE_COL_W = 504
const IEEE_SINGLE_COL_H = 360

df = CSV.read(joinpath(DATA_DIR, "results_meep_validation.csv"), DataFrame)

# Scatter plot: MoM reflectance vs MEEP reflectance
# Perfect agreement → identity line

R_mom = collect(df.R_MoM)
R_meep = collect(df.R_MEEP)

fig = plot_scatter(
    [R_mom, [0.0, 1.05]],
    [R_meep, [0.0, 1.05]];
    mode=["markers", "lines"],
    legend=["Slot structures", "Identity"],
    marker_size=[10, 0],
    marker_symbol=["circle", ""],
    dash=["", "dash"],
    color=[COLORS[1], "gray"],
    xlabel="MoM reflectance",
    ylabel="MEEP reflectance",
    xrange=[0.0, 1.05],
    yrange=[0.0, 1.05],
    fontsize=14,
    width=IEEE_SINGLE_COL_W, height=IEEE_SINGLE_COL_W,  # square
)
set_legend!(fig; position=:bottomright)
savefig(fig, joinpath(FIG_DIR, "fig_results_meep_validation.pdf"))
println("✓ Saved: figures/fig_results_meep_validation.pdf")
