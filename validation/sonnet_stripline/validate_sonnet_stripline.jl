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
# UNVERIFIED and exits 0 (a missing external tool is never a pass).
#
# Run: julia --project=. validation/sonnet_stripline/validate_sonnet_stripline.jl

using DiffMoM
using Printf

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

function find_em()
    haskey(ENV, "SONNET_EM") && isfile(ENV["SONNET_EM"]) &&
        return ENV["SONNET_EM"]
    for root in (ENV["ProgramFiles"], ENV["ProgramFiles(x86)"],
                 raw"C:\Program Files", raw"C:\Program Files (x86)")
        dir = joinpath(root, "Sonnet Software")
        isdir(dir) || continue
        for ver in sort(readdir(dir); rev=true)
            em = joinpath(dir, ver, "bin", "em.exe")
            isfile(em) && return em
        end
    end
    return nothing
end

"""Run `em` on `sonfile` inside a scratch dir; return the parsed
de-embedded S row (freq, |S11|, ang, |S21|, ang, |S12|, ang, |S22|, ang)
plus the per-port Z0 diagnostic."""
function run_sonnet(em::String, sonfile::String)
    mktempdir() do dir
        src = joinpath(@__DIR__, sonfile)
        dst = joinpath(dir, sonfile)
        cp(src, dst)
        run(`$em $dst`)
        logf = joinpath(dir, "sondata", replace(sonfile, ".son" => ""),
                        "log_response.log")
        isfile(logf) || error("sonnet produced no response log for $sonfile")
        lines = split(read(logf, String), '\n')
        svals = Float64[]
        z0 = NaN
        for (i, line) in enumerate(lines)
            if occursin("De-embedded S-Parameters", line)
                # scan forward: f |S11| ang |S21| ang |S12| ang |S22| ang
                for l2 in @view lines[(i + 1):end]
                    tok = split(l2)
                    length(tok) == 9 || continue
                    vals = tryparse.(Float64, tok)
                    all(!isnothing, vals) || continue
                    svals = Float64.(vals)
                    break
                end
            elseif occursin("Z0=", line)
                m = match(r"Z0=\(([-+0-9.eE]+)", line)
                m !== nothing && (z0 = parse(Float64, m.captures[1]))
            end
        end
        isempty(svals) &&
            error("no de-embedded S row in sonnet response for $sonfile")
        return svals, z0
    end
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
        return 0
    end
    println("em: $em")
    println(@sprintf("%-5s %9s %9s %9s %9s %9s %9s %9s",
                     "case", "|S11|", "d|S11|", "|S21|", "d|S21|",
                     "angS21", "dang", "dZ0%"))
    fails = String[]
    for (name, b, w, son) in CASES
        (sv, sz0) = run_sonnet(em, son)
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
        d_ang = rad2deg(angle(S[2, 1])) - sv[5]
        dz0 = (real(cal.zc) - sz0) / sz0 * 100
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
    end
    isempty(fails) && return 0
    foreach(f -> println("FAIL: ", f), fails)
    return 1
end

exit(main())
