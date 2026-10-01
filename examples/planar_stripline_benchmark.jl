# Shielded stripline benchmark (canonical standard problem):
# lossless air-filled stripline, ground-plane spacing h = 1 mm,
# strip width w = 1.4423896 mm -> characteristic impedance exactly 50 Ohm.
# Line length = quarter wavelength at 15 GHz (4.99654097 mm), so a perfect
# analysis returns |S11| = 0 and |S21| = 1 with -90 deg phase.
#
# The gap-port model and cell subsectioning leave a small systematic error;
# the published error model predicts a few-percent impedance error at this
# subsection count (Nw ~ 6 cells across the strip).

using DiffMoM, Printf

const W_LINE = 1.4423896e-3   # 50-Ohm strip width for h=1mm, epsr=1
const H_GND = 1.0e-3          # ground-plane spacing
const L_LINE = 4.99654097e-3  # quarter wavelength at 15 GHz
const BW = 10.0e-3            # box width across the strip
const FREQ = 15e9

stack = PlanarStackup(
    [PlanarLayer(1.0, 1.0, H_GND / 2), PlanarLayer(1.0, 1.0, H_GND / 2)],
    TERM_GND, TERM_GND, L_LINE, BW)

nx, ny = 32, 42
grid = CellGrid(L_LINE, BW, nx, ny)
s = sheet_level(1, nx, ny)
rasterize_rect!(s, grid, 0.0, L_LINE, BW / 2 - W_LINE / 2, BW / 2 + W_LINE / 2)
prow = findall(j -> any(s.mask[:, j]), 1:ny)
for j in prow
    s.connect_west[j] = true
    s.connect_east[j] = true
end
ports = [PlanarPort(1, :west, prow[1]:prow[end], 50.0),
         PlanarPort(1, :east, prow[1]:prow[end], 50.0)]
prob = build_planar_problem(stack, grid, [s], ports)
println("basis functions: ", planar_basis_count(prob.basis))

r = solve_planar(prob, FREQ; mx = 4nx, my = 4ny)
S = r.s
Zin = inv(r.y)
z0_eq = sqrt(Zin[1, 1] * Zin[2, 2] - Zin[1, 2] * Zin[2, 1])
theta = acos(ComplexF64(Zin[1, 1] / Zin[2, 1]))

@printf("S11 = %.4f  (%.1f deg)\n", abs(S[1, 1]), rad2deg(angle(S[1, 1])))
@printf("S21 = %.4f  (%.1f deg)\n", abs(S[2, 1]), rad2deg(angle(S[2, 1])))
@printf("equivalent line Z0 = %.2f Ohm (target 50)\n", real(z0_eq))
@printf("electrical length  = %.1f deg (target 90)\n", rad2deg(real(theta)))
