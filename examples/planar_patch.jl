# Edge-fed rectangular patch antenna in a shielded enclosure.
#
#   substrate:  RT/Duroid 5880 (epsr = 2.2, tand = 9e-4), h = 1.575 mm
#               over PEC ground, PMC lid (electrically open top face)
#   patch:      10 mm x 10 mm centred, fed by a 1 mm line to the west wall
#   analysis:   ABS adaptive sweep; the input impedance Zin = 50*(1+S11)/
#               (1-S11) crosses resonance where Im(Zin) -> 0, nominally
#               near c/(2 L sqrt(eps_eff)) ~ 9.5-10 GHz.  (A lossless
#               one-port has |S11| = 1 identically, so the resonance is
#               read off the input impedance, not the |S11| magnitude.)
#
# Ports must land on box walls, so the feed line bridges the patch to the
# west sidewall; the port cells are the two wall-connected feed rows.

using DiffMoM, Printf

const EPSR  = 2.2 * (1 - 9e-4im)   # Duroid 5880 with dielectric loss
const H_SUB = 1.575e-3
const BOX_A = 25.0e-3         # x extent
const BOX_B = 20.0e-3         # y extent
const PATCH = 10.0e-3         # square patch side
const FEEDW = 1.0e-3          # feed-line width
const GAPX  = 7.5e-3          # feed-in distance of the patch edge

stack = PlanarStackup([PlanarLayer(EPSR, 1.0, H_SUB)],
                      TERM_GND, PlanarTerminator(TERM_PMC), BOX_A, BOX_B)

nx, ny = 50, 40               # 0.5 mm cells
grid = CellGrid(BOX_A, BOX_B, nx, ny)
s = sheet_level(1, nx, ny)
# patch centred in y, starting GAPX from the west wall
rasterize_rect!(s, grid, GAPX, GAPX + PATCH,
                BOX_B / 2 - PATCH / 2, BOX_B / 2 + PATCH / 2)
# feed line from the west wall to the patch edge
rasterize_rect!(s, grid, 0.0, GAPX + grid.dx / 2,
                BOX_B / 2 - FEEDW / 2, BOX_B / 2 + FEEDW / 2;
                connected=true)
prow = findall(j -> s.connect_west[j], 1:ny)
ports = [PlanarPort(1, :west, prow[1]:prow[end], 50.0)]
prob = build_planar_problem(stack, grid, [s], ports)
println("basis functions: ", planar_basis_count(prob.basis))

sw = planar_sweep_abs(prob, 8e9, 12e9; n_eval=33, rel_tol=1e-3,
                      max_points=24, mx=2nx, my=2ny)
println("analyzed frequencies: ", round.(sw.freqs ./ 1e9; digits=2), " GHz")

# input impedance at the 50 Ohm reference: the edge-fed patch resonance
# is the anti-resonant peak of Re(Zin) with Im(Zin) crossing zero
zins = [50 * (1 + S[1, 1]) / (1 - S[1, 1]) for S in sw.dense_s]
jmin = argmax([real(z) for z in zins])
println("   f [GHz]    |S11| [dB]   Re Zin [Ohm]   Im Zin [Ohm]")
for j in eachindex(sw.dense_freqs)
    S = sw.dense_s[j]
    mark = j == jmin ? "  <- resonance" : ""
    @printf("%9.2f   %9.2f    %11.1f   %11.1f%s\n", sw.dense_freqs[j] / 1e9,
            20 * log10(abs(S[1, 1])), real(zins[j]), imag(zins[j]), mark)
end
