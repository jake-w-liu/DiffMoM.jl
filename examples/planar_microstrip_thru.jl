# Shielded microstrip through line: a strip on the substrate/air interface
# of a covered microstrip, port-fed at the west and east sidewalls.
#
#   stackup:  alumina substrate (epsr = 9.8, h = 0.635 mm) over PEC ground,
#             1.27 mm air cover to a PEC lid
#   strip:    w = 0.625 mm on 0.125 mm cells -> about 50 Ohm, eps_eff ~ 6.5
#   analysis: ABS adaptive rational sweep over 1-20 GHz, then Z0/eps_eff
#             extraction from the thru's short-circuit admittance
#
# The adaptive sweep interpolates a rational model through adaptively placed
# EM analyses, so the dense frequency grid costs only ~10 direct solves.

using DiffMoM, Printf, LinearAlgebra

const W_LINE = 0.625e-3       # strip width
const H_SUB  = 0.635e-3       # substrate thickness (alumina)
const H_AIR  = 1.27e-3        # air cover height
const L_LINE = 6.0e-3         # line/box length (x)
const BW     = 6.0e-3         # box width (y)
const FREQS  = range(1e9, 20e9; length=40)

stack = PlanarStackup(
    [PlanarLayer(9.8, 1.0, H_SUB), PlanarLayer(1.0, 1.0, H_AIR)],
    TERM_GND, PlanarTerminator(TERM_PEC), L_LINE, BW)

nx, ny = 48, 48
grid = CellGrid(L_LINE, BW, nx, ny)
s = sheet_level(1, nx, ny)                       # substrate/air interface
rasterize_rect!(s, grid, 0.0, L_LINE,
                BW / 2 - W_LINE / 2, BW / 2 + W_LINE / 2)
prow = findall(j -> any(s.mask[:, j]), 1:ny)
for j in prow
    s.connect_west[j] = true
    s.connect_east[j] = true
end
ports = [PlanarPort(1, :west, prow[1]:prow[end], 50.0),
         PlanarPort(1, :east, prow[1]:prow[end], 50.0)]
prob = build_planar_problem(stack, grid, [s], ports)
println("basis functions: ", planar_basis_count(prob.basis))

# --- adaptive band synthesis: rational interpolation on adaptively placed
#     analysis points (Sonnet ABS algorithm) ---
sw = planar_sweep_abs(prob, 1e9, 20e9; n_eval=40, rel_tol=1e-3,
                      max_points=24, mx=2nx, my=2ny)
println("analyzed frequencies: ", round.(sw.freqs ./ 1e9; digits=2), " GHz")
println("converged: ", sw.converged,
        "  (worst estimated error ", maximum(sw.est_err), ")")

println("   f [GHz]   |S11|      |S21|     ang S21 [deg]")
for j in eachindex(sw.dense_freqs)
    S = sw.dense_s[j]
    @printf("%9.2f   %.4f    %.4f    %9.1f\n",
            sw.dense_freqs[j] / 1e9, abs(S[1, 1]), abs(S[2, 1]),
            rad2deg(angle(S[2, 1])))
end

# --- characteristic impedance and effective permittivity from the thru ---
# extract mid-band: near half-wave multiples the ABCD branch constants
# shrink and the extraction loses conditioning
r = solve_planar(prob, 5e9; mx=2nx, my=2ny)
zc, gl = planar_line_params(r.y)
c0 = 299792458.0
k0 = 2pi * 5e9 / c0
beta = abs(imag(gl)) / L_LINE            # gl is the whole-line gamma*l
eps_eff = (beta / k0)^2
@printf("\n5 GHz:   Z0 = %.2f + %.2fi Ohm\n", real(zc), imag(zc))
@printf("         eps_eff = %.3f   (alpha = %.2f dB/m)\n", eps_eff,
        20 * log10(MathConstants.e) * real(gl) / L_LINE)
