# 21_near_total_field_rayleigh_sphere.jl
#
# Validate scattered and total electric fields against the Rayleigh-limit
# analytical solution for a small PEC sphere under plane-wave illumination.
#
# The reference model is:
#   - incident field: plane wave
#   - scattered field: electric dipole with p = 4*pi*eps0*a^3*E0*pol
#   - total field: E_inc + E_sca
#
# This is accurate when ka << 1. The example keeps ka small enough that the
# dipole model is a meaningful benchmark for both compute_nearfield and
# compute_total_field.
#
# Run:
#   julia --project=. examples/21_near_total_field_rayleigh_sphere.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
include(joinpath(@__DIR__, "..", "src", "DiffMoM.jl"))
using .DiffMoM
using LinearAlgebra
using Printf
using CSV
using DataFrames
using PlotlySupply

function make_icosphere(radius::Float64; subdivisions::Int=2)
    phi_g = (1 + sqrt(5.0)) / 2
    verts0 = [
        (-1.0,  phi_g, 0.0), ( 1.0,  phi_g, 0.0), (-1.0, -phi_g, 0.0), ( 1.0, -phi_g, 0.0),
        ( 0.0, -1.0,  phi_g), ( 0.0,  1.0,  phi_g), ( 0.0, -1.0, -phi_g), ( 0.0,  1.0, -phi_g),
        ( phi_g, 0.0, -1.0), ( phi_g, 0.0,  1.0), (-phi_g, 0.0, -1.0), (-phi_g, 0.0,  1.0),
    ]
    faces = [
        (1, 12, 6), (1, 6, 2), (1, 2, 8), (1, 8, 11), (1, 11, 12),
        (2, 6, 10), (6, 12, 5), (12, 11, 3), (11, 8, 7), (8, 2, 9),
        (4, 10, 5), (4, 5, 3), (4, 3, 7), (4, 7, 9), (4, 9, 10),
        (5, 10, 6), (3, 5, 12), (7, 3, 11), (9, 7, 8), (10, 9, 2),
    ]
    verts = [Vec3(v...) / norm(Vec3(v...)) for v in verts0]

    for _ in 1:subdivisions
        edge_mid = Dict{Tuple{Int, Int}, Int}()
        new_faces = NTuple{3, Int}[]

        function midpoint_index(i::Int, j::Int)
            key = i < j ? (i, j) : (j, i)
            haskey(edge_mid, key) && return edge_mid[key]
            vmid = (verts[i] + verts[j]) / 2
            vmid /= norm(vmid)
            push!(verts, vmid)
            edge_mid[key] = length(verts)
            return length(verts)
        end

        for (i, j, k) in faces
            a = midpoint_index(i, j)
            b = midpoint_index(j, k)
            c = midpoint_index(k, i)
            push!(new_faces, (i, a, c))
            push!(new_faces, (j, b, a))
            push!(new_faces, (k, c, b))
            push!(new_faces, (a, b, c))
        end
        faces = new_faces
    end

    xyz = zeros(3, length(verts))
    tri = zeros(Int, 3, length(faces))
    for i in eachindex(verts)
        xyz[:, i] = radius .* verts[i]
    end
    for t in eachindex(faces)
        tri[:, t] .= faces[t]
    end
    return TriMesh(xyz, tri)
end

function rayleigh_dipole_field(r::Vec3, k::Float64, moment::CVec3)
    eps0 = 8.854187817e-12
    r_vec = r
    R = norm(r_vec)
    if R < 1e-12
        return CVec3(0.0 + 0im, 0.0 + 0im, 0.0 + 0im)
    end

    r_hat = r_vec / R
    term1 = cross(r_hat, moment)
    term1 = cross(term1, r_hat) * k^2
    term2 = (3 * r_hat * dot(r_hat, moment) - moment) * (1 / R^2 - 1im * k / R)
    return (term1 + term2) * exp(-1im * k * R) / (4 * pi * eps0 * R)
end

function stack_fields(field_func, points)
    E = zeros(ComplexF64, 3, length(points))
    for i in eachindex(points)
        E[:, i] .= field_func(points[i])
    end
    return E
end

function rel_error_columns(E_num::AbstractMatrix, E_ref::AbstractMatrix)
    errs = zeros(size(E_num, 2))
    for i in axes(E_num, 2)
        denom = max(norm(E_ref[:, i]), 1e-30)
        errs[i] = norm(E_num[:, i] - E_ref[:, i]) / denom
    end
    return errs
end

println("="^60)
println("Example 21: Near/Total Field Validation on a Rayleigh Sphere")
println("="^60)

figdir = joinpath(@__DIR__, "figs")
datadir = joinpath(@__DIR__, "..", "data")
mkpath(figdir)
mkpath(datadir)

# Problem setup: electrically small PEC sphere
a = 0.02
freq = 100e6
c0 = 299792458.0
eps0 = 8.854187817e-12
lambda0 = c0 / freq
k = 2 * pi / lambda0
eta0 = 376.730313668
ka = k * a

println(@sprintf("\nSphere radius: %.3f m", a))
println(@sprintf("Frequency: %.3f MHz", freq / 1e6))
println(@sprintf("Lambda: %.3f m", lambda0))
println(@sprintf("Size parameter ka: %.4f", ka))

mesh = make_icosphere(a; subdivisions=2)
rwg = build_rwg(mesh)
N = rwg.nedges
println("Mesh: $(nvertices(mesh)) verts, $(ntriangles(mesh)) tris, $N RWG unknowns")

k_vec = Vec3(0.0, 0.0, -k)
E0 = 1.0
pol = Vec3(1.0, 0.0, 0.0)
pw = make_plane_wave(k_vec, E0, pol)

println("\nAssembling and solving the MoM system...")
t_asm = @elapsed Z = assemble_Z_efie(mesh, rwg, k; quad_order=3, eta0=eta0)
v = assemble_excitation(mesh, rwg, pw; quad_order=3)
t_sol = @elapsed I = Z \ v
residual = norm(Z * I - v) / norm(v)
println(@sprintf("  Assembly: %.3f s", t_asm))
println(@sprintf("  Solve:    %.3f s", t_sol))
println(@sprintf("  Residual: %.3e", residual))

# Rayleigh analytical reference
alpha_pec = 4 * pi * eps0 * a^3
p_vec = alpha_pec * E0 * pol
p_rayleigh = CVec3(complex(p_vec[1]), complex(p_vec[2]), complex(p_vec[3]))

r_over_a = collect(range(1.4, stop=3.0, length=9))
obs_x = [Vec3(rf * a, 0.0, 0.0) for rf in r_over_a]
obs_z = [Vec3(0.0, 0.0, rf * a) for rf in r_over_a]
obs_all = vcat(obs_x, obs_z)
labels = vcat(["x-axis" for _ in obs_x], ["z-axis" for _ in obs_z])
dist_labels = vcat(r_over_a, r_over_a)

println("\nEvaluating scattered and total fields...")
E_sca_mom = compute_nearfield(mesh, rwg, I, obs_all, k; quad_order=7, eta0=eta0)
E_tot_mom = compute_total_field(mesh, rwg, I, pw, obs_all, k; quad_order=7, eta0=eta0)

E_sca_ref = stack_fields(obs_all) do r
    rayleigh_dipole_field(r, k, p_rayleigh)
end
E_tot_ref = stack_fields(obs_all) do r
    E_inc = pol * E0 * exp(-1im * dot(k_vec, r))
    E_inc + rayleigh_dipole_field(r, k, p_rayleigh)
end

sca_err = rel_error_columns(E_sca_mom, E_sca_ref)
tot_err = rel_error_columns(E_tot_mom, E_tot_ref)

println("\nReference model: Rayleigh electric dipole for a small PEC sphere")
println("Incident field is the exact plane wave; scattered field is the dipole approximation.")
println("Observation points are kept within 3a so the comparison stays in the near zone.")
println("\n  axis    r/a    rel_err(E_sca)    rel_err(E_tot)")
for i in eachindex(obs_all)
    println(@sprintf("  %-6s  %3.1f      %.4e        %.4e",
                     labels[i], dist_labels[i], sca_err[i], tot_err[i]))
end

max_sca_err = maximum(sca_err)
max_tot_err = maximum(tot_err)
mean_sca_err = sum(sca_err) / length(sca_err)
mean_tot_err = sum(tot_err) / length(tot_err)

sca_mag_mom = vec([norm(E_sca_mom[:, i]) for i in axes(E_sca_mom, 2)])
sca_mag_ref = vec([norm(E_sca_ref[:, i]) for i in axes(E_sca_ref, 2)])
tot_mag_mom = vec([norm(E_tot_mom[:, i]) for i in axes(E_tot_mom, 2)])
tot_mag_ref = vec([norm(E_tot_ref[:, i]) for i in axes(E_tot_ref, 2)])

println("\nSummary:")
println(@sprintf("  mean rel_err(E_sca): %.4e", mean_sca_err))
println(@sprintf("  max  rel_err(E_sca): %.4e", max_sca_err))
println(@sprintf("  mean rel_err(E_tot): %.4e", mean_tot_err))
println(@sprintf("  max  rel_err(E_tot): %.4e", max_tot_err))

df = DataFrame(
    axis = labels,
    r_over_a = dist_labels,
    x_m = [r[1] for r in obs_all],
    y_m = [r[2] for r in obs_all],
    z_m = [r[3] for r in obs_all],
    esca_mom_norm = sca_mag_mom,
    esca_ref_norm = sca_mag_ref,
    etot_mom_norm = tot_mag_mom,
    etot_ref_norm = tot_mag_ref,
    rel_err_esca = sca_err,
    rel_err_etot = tot_err,
)
csv_path = joinpath(datadir, "rayleigh_near_total_field_validation.csv")
CSV.write(csv_path, df)
println("\nCSV saved: $csv_path")

idx_x = 1:length(r_over_a)
idx_z = (length(r_over_a) + 1):length(obs_all)

function add_axis_curves!(fig, row::Int, col::Int, xvals, y_ref, y_mom, axis_label, color)
    addtraces!(fig, scatter(
        x=xvals,
        y=y_ref,
        mode="lines+markers",
        name="$axis_label reference",
        line=attr(color=color, width=2),
        marker=attr(size=7),
    ); row=row, col=col)
    addtraces!(fig, scatter(
        x=xvals,
        y=y_mom,
        mode="lines+markers",
        name="$axis_label MoM",
        line=attr(color=color, width=2, dash="dash"),
        marker=attr(size=7, symbol="circle-open"),
    ); row=row, col=col)
end

fig_mag = subplots(
    1, 2;
    sync=false,
    width=1100,
    height=460,
    subplot_titles=reshape([
        "Scattered-field Magnitude |E_sca|",
        "Total-field Magnitude |E_tot|",
    ], 1, 2),
)

add_axis_curves!(fig_mag, 1, 1, r_over_a, sca_mag_ref[idx_x], sca_mag_mom[idx_x], "x-axis", "#1f77b4")
add_axis_curves!(fig_mag, 1, 1, r_over_a, sca_mag_ref[idx_z], sca_mag_mom[idx_z], "z-axis", "#d62728")
add_axis_curves!(fig_mag, 1, 2, r_over_a, tot_mag_ref[idx_x], tot_mag_mom[idx_x], "x-axis", "#1f77b4")
add_axis_curves!(fig_mag, 1, 2, r_over_a, tot_mag_ref[idx_z], tot_mag_mom[idx_z], "z-axis", "#d62728")

p_mag = fig_mag.plot
relayout!(
    p_mag,
    xaxis=attr(title="Observation distance r/a"),
    xaxis2=attr(title="Observation distance r/a"),
    yaxis=attr(title="Field magnitude (V/m)"),
    yaxis2=attr(title="Field magnitude (V/m)"),
    legend=attr(x=0.02, y=0.98),
    margin=attr(l=70, r=40, t=70, b=60),
)
fig_mag_path = joinpath(figdir, "21_rayleigh_near_total_field_magnitude.png")
savefig(p_mag, fig_mag_path)
println("Plot saved: $fig_mag_path")

fig_err = subplots(
    1, 2;
    sync=false,
    width=1100,
    height=460,
    subplot_titles=reshape([
        "Relative Error Along x-axis",
        "Relative Error Along z-axis",
    ], 1, 2),
)

addtraces!(fig_err, scatter(
    x=r_over_a,
    y=sca_err[idx_x],
    mode="lines+markers",
    name="E_sca error (x-axis)",
    line=attr(color="#1f77b4", width=2),
    marker=attr(size=7),
); row=1, col=1)
addtraces!(fig_err, scatter(
    x=r_over_a,
    y=tot_err[idx_x],
    mode="lines+markers",
    name="E_tot error (x-axis)",
    line=attr(color="#2ca02c", width=2, dash="dash"),
    marker=attr(size=7, symbol="diamond-open"),
); row=1, col=1)
addtraces!(fig_err, scatter(
    x=r_over_a,
    y=sca_err[idx_z],
    mode="lines+markers",
    name="E_sca error (z-axis)",
    line=attr(color="#d62728", width=2),
    marker=attr(size=7),
); row=1, col=2)
addtraces!(fig_err, scatter(
    x=r_over_a,
    y=tot_err[idx_z],
    mode="lines+markers",
    name="E_tot error (z-axis)",
    line=attr(color="#9467bd", width=2, dash="dash"),
    marker=attr(size=7, symbol="diamond-open"),
); row=1, col=2)

p_err = fig_err.plot
relayout!(
    p_err,
    xaxis=attr(title="Observation distance r/a"),
    xaxis2=attr(title="Observation distance r/a"),
    yaxis=attr(title="Relative error", type="log"),
    yaxis2=attr(title="Relative error", type="log"),
    legend=attr(x=0.02, y=0.98),
    margin=attr(l=70, r=40, t=70, b=60),
)
fig_err_path = joinpath(figdir, "21_rayleigh_near_total_field_errors.png")
savefig(p_err, fig_err_path)
println("Plot saved: $fig_err_path")

@assert residual < 1e-10
@assert max_sca_err < 0.35
@assert max_tot_err < 0.10

println("\n" * "="^60)
println("Done. Near-field and total-field agree with the Rayleigh reference")
println("within the expected small-ka error envelope.")
println("="^60)
