# Single-turn square loop inductor in a shielded multilayer box.
#
#   stackup:  thin insulator (epsr = 4, 0.1 mm) over PEC ground,
#             main substrate (epsr = 9.8, 0.5 mm), open air top
#   metal:    rectangular loop on the top interface (interface 2),
#             open on the west side; both terminals feed at the west
#             wall on adjacent rows
#   analysis: |S11|/|S21| sweep plus series-branch inductance and Q
#             through planar_inductor on the 2-port admittance matrix
#
# Layout (0.1 mm cells on a 4 mm box, 1-cell track): the loop enters at
# west wall row 10, wraps the perimeter, and returns to the west wall at
# row 14.  A closed multi-turn spiral cannot place both terminals on
# walls of a single metal level (Jordan separation), so real planar
# spirals drop the inner terminal through a via or airbridge; the via
# column basis is exercised in test_planar.jl instead.

using DiffMoM, Printf

const D_INS = 0.1e-3          # insulator thickness
const D_SUB = 0.5e-3          # substrate thickness
const BOX   = 4.0e-3
const NXY   = 40              # 0.1 mm cells

stack = PlanarStackup(
    [PlanarLayer(4.0, 1.0, D_INS), PlanarLayer(9.8, 1.0, D_SUB)],
    TERM_GND, TERM_SPACE, BOX, BOX)

grid = CellGrid(BOX, BOX, NXY, NXY)
loop = sheet_level(2, NXY, NXY)      # loop on the top interface

hseg!(m, i1, i2, j) = (m.mask[i1:i2, j] .= true)
vseg!(m, i, j1, j2) = (m.mask[i, j1:j2] .= true)

# loop path: west wall row 10 -> perimeter -> west wall row 14
hseg!(loop,  1, 34, 10)      # south arm (enters at the west wall)
vseg!(loop, 34, 10, 30)      # east arm
hseg!(loop,  5, 34, 30)      # north arm
vseg!(loop,  5, 14, 30)      # inner west arm
hseg!(loop,  1,  5, 14)      # return stub to the west wall
loop.connect_west[10] = true
loop.connect_west[14] = true

ports = [PlanarPort(1, :west, 10:10, 50.0),
         PlanarPort(1, :west, 14:14, 50.0)]
prob = build_planar_problem(stack, grid, [loop], ports)
println("basis functions: ", planar_basis_count(prob.basis))

println("   f [GHz]    |S11|      |S21|     L [nH]    Q")
for f in range(0.5e9, 5e9; length=10)
    r = solve_planar(prob, f; mx=2NXY, my=2NXY)
    ind = planar_inductor(r.y, f)
    @printf("%9.2f   %.4f    %.4f    %8.3f   %6.1f\n", f / 1e9,
            abs(r.s[1, 1]), abs(r.s[2, 1]), ind.l * 1e9, ind.q)
end
