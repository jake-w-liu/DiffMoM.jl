# validate_sonnet_stripline.jl — DiffMoM vs Sonnet stripline benchmarks
#
# Replicates the published Sonnet stripline benchmark set (s25/s50/s100,
# shipped under examples/stripline_benchmarks/ in the Sonnet install and
# copied alongside this script) and compares de-embedded S-parameters at
# 15 GHz:
#
#   box:   a = 4.996541 mm between the port walls, two 0.5 mm air layers,
#          lossless PEC covers (TMET/BMET "Lossless" SUP)
#   strip: centered in y on the mid interface, width per case below,
#          west/east 50-ohm wall ports
#   freq:  15 GHz single point (FREQ Y AN SWEEP 15.0 in the .son files)
#
# Sonnet's STD ports are co-calibrated: the reported "De-embedded
# S-Parameters" exclude the gap-port shunt discontinuity.  DiffMoM raw
# port results include it, so the comparison applies
# deembed_double_delay_calibrate/apply with thru standards of length a/2
# and a (same cross-section, same cell size) — the extracted parasitic is
# a pure shunt to ~1e-12, matching Sonnet's port model.
#
# The validator locates em.exe via the SONNET_EM environment variable or
# the standard install root; if Sonnet is not installed it prints
# UNVERIFIED and exits 2. Native outputs and metadata are retained under
# data/sonnet_validation (override with SONNET_VALIDATION_DIR).
#
# Run: julia --project=. validation/sonnet_stripline/validate_sonnet_stripline.jl

using DiffMoM
using Printf
using TOML
include("sonnet_reference.jl")
using .SonnetReference

const A = 4.996541e-3          # box length (x, port-to-port) [m]
const H = 1.0e-3               # total substrate height (two halves) [m]
const NX, NY = 64, 64          # cell grid; cal standards halve NX exactly

# name => (box width b [m], strip width w [m], Sonnet-case file)
const CASES = [
    ("s25",  13.31456e-3,   3.3286400e-3,  "s25.son"),
    ("s50",  11.539117e-3,  1.4423896e-3,  "s50.son"),
    ("s100", 8.0736925e-3,  0.5046058e-3,  "s100.son"),
]

# ---------------- Sonnet side ----------------

"""Run `em` on `sonfile` inside a retained directory; return the parsed
de-embedded S row (freq, |S11|, ang, |S21|, ang, |S12|, ang, |S22|, ang)
plus the per-port Z0 diagnostic."""
function run_sonnet(em::String, sonfile::String, evidence_dir::String)
    # Native BOX fields count half-cells. Match the actual DiffMoM grid.
    source=joinpath(evidence_dir,sonfile)
    original=replace(read(joinpath(@__DIR__,sonfile),String),"\r\n"=>"\n")
    write(source,replace(original,r"(?m)^(BOX 1 [^ ]+ [^ ]+) \d+ \d+"=>
        m->first(match(r"^(BOX 1 [^ ]+ [^ ]+)",m).captures)*" $(2NX) $(2NY)"))
    ref = reference_run(em, source;
        output_dir=joinpath(evidence_dir, splitext(sonfile)[1]))
    length(ref.rows) == 1 && ref.rows[1][1] == 15.0 ||
        error("$sonfile must report exactly one 15 GHz row")
    sz0 = get(ref.z0, (1, 15.0), complex(NaN))
    isfinite(sz0) && real(sz0) > 0 || error("missing Sonnet port Z0: $(ref.logf)")
    return ref.rows[1], real(sz0)
end

# ---------------- DiffMoM side ----------------

"""Build the strip problem: `a` is the box length (x), `b` the width,
`w` the strip width; strip spans the full a on the mid interface."""
function dmm_case(a, b, w, nx, ny)
    stack = PlanarStackup(
        [PlanarLayer(1.0, 1.0, H / 2), PlanarLayer(1.0, 1.0, H / 2)],
        TERM_GND, TERM_GND, a, b)
    grid = CellGrid(a, b, nx, ny)
    s = sheet_level(1, nx, ny)
    rasterize_rect!(s, grid, 0.0, a, b / 2 - w / 2, b / 2 + w / 2)
    prow = findall(j -> any(s.mask[:, j]), 1:ny)
    for j in prow
        s.connect_west[j] = true
        s.connect_east[j] = true
    end
    ports = [PlanarPort(1, :west, prow[1]:prow[end], 50.0),
             PlanarPort(1, :east, prow[1]:prow[end], 50.0)]
    return build_planar_problem(stack, grid, [s], ports)
end

# ---------------- driver ----------------

function main()
    em = find_em()
    em === nothing && begin
        println("UNVERIFIED: sonnet em.exe not found " *
                "(set SONNET_EM to its path)")
        return 2
    end
    evidence_dir = evidence_directory("stripline")
    println("em: $em")
    println("evidence: $evidence_dir")
    println(@sprintf("%-5s %9s %9s %9s %9s %9s %9s %9s",
                     "case", "|S11|", "d|S11|", "|S21|", "d|S21|",
                     "angS21", "dang", "dZ0%"))
    fails = String[]
    results = Dict{String,Any}[]
    for (name, b, w, son) in CASES
        (sv, sz0) = run_sonnet(em, son, evidence_dir)
        Sref = twoport_matrix(sv)
        prob = dmm_case(A, b, w, NX, NY)
        probl = dmm_case(A / 2, b, w, NX ÷ 2, NY)
        r = solve_planar(prob, 15e9; mx=2NX, my=2NY,
                         max_bytes=8_000_000_000)
        rl = solve_planar(probl, 15e9; mx=NX, my=2NY,
                          max_bytes=8_000_000_000)
        cal = deembed_double_delay_calibrate(rl.y, r.y; len=A / 2)
        cal.residual < 1e-6 ||
            push!(fails, "$name: double-delay residual $(cal.residual)")
        Yd = deembed_double_delay_apply(r.y, cal; line=false)
        S = planar_y_to_s(Yd, [50.0, 50.0])
        d_s11 = abs(S[1, 1]) - sv[2]
        d_s21 = abs(S[2, 1]) - sv[4]
        d_ang = wrapped_phase_deg(S[2, 1], Sref[2, 1])
        dz0 = (real(cal.zc) - sz0) / sz0 * 100
        max_complex_error = maximum(abs, S - Sref)
        write_touchstone(joinpath(evidence_dir, name, "diffmom.s2p"), [15e9], [S])
        push!(results, Dict("case" => name, "nx" => NX, "ny" => NY,
            "native_box_halfcell_counts" => [2NX,2NY],
            "mx" => 2NX, "my" => 2NY, "basis_count" => planar_basis_count(prob.basis),
            "max_complex_s_error" => max_complex_error, "s21_phase_error_deg" => d_ang,
            "z0_error_percent" => dz0, "calibration_residual" => cal.residual,
            "sonnet_z0_ohm" => sz0, "diffmom_z0_real_ohm" => real(cal.zc)))
        @printf("%-5s %9.6f %+9.6f %9.6f %+9.6f %9.3f %+9.3f %+8.2f\n",
                name, abs(S[1, 1]), d_s11, abs(S[2, 1]), d_s21,
                rad2deg(angle(S[2, 1])), d_ang, dz0)
        # gates: measured ~1e-4..2e-2 at 64x64; leave 2-3x margin for
        # platform BLAS differences
        abs(d_s21) <= 0.03 ||
            push!(fails, "$name |S21| off by $(abs(d_s21)) (> 0.03)")
        abs(d_ang) <= 0.5 ||
            push!(fails, "$name ang(S21) off by $(abs(d_ang)) deg")
        abs(dz0) <= 5.0 ||
            push!(fails, "$name Z0 off by $(abs(dz0))% (> 5%)")
        abs(d_s11) <= 0.06 ||
            push!(fails, "$name |S11| off by $(abs(d_s11))")
        max_complex_error <= 0.06 ||
            push!(fails, "$name max complex S error $max_complex_error (> 0.06)")
    end
    open(joinpath(evidence_dir, "comparison.toml"), "w") do io
        TOML.print(io, Dict("cases" => results, "failures" => fails,
            "status" => isempty(fails) ? "PASS" : "FAIL",
            "scope" => "64x64 lossless wall-port striplines; not a full Sonnet parity certificate"))
    end
    isempty(fails) && return 0
    foreach(f -> println("FAIL: ", f), fails)
    return 1
end

exit(main())
