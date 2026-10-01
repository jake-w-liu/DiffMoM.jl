# test_planar.jl — shielded planar layered-media MoM (box-mode spectral
# Galerkin solver with rooftop basis and gap ports).

const _PG = DiffMoM

# ---------- test helpers ----------

# independent reference cascade in tanh form (production uses the
# exp(-2*gamma*d) form; the two must agree on finite, moderate modes)
function _ref_input_impedance(zc, gd, zl)
    if !isfinite(zl)
        return zc / tanh(gd)          # open-terminated section
    end
    t = tanh(gd)
    den = zc + zl * t
    iszero(den) && return ComplexF64(Inf)   # anti-resonance
    return zc * (zl + zc * t) / den
end

function _ref_term(stack, which, pol, omega, kc2)
    term = which === :bottom ? stack.bottom : stack.top
    term.kind === TERM_PEC && return zero(ComplexF64)
    term.kind === TERM_PMC && return ComplexF64(Inf)
    term.kind === TERM_SURFACE && return term.zs
    gamma0 = sqrt(ComplexF64(kc2) -
        _PG.planar_k2_layer(omega, term.epsr, term.mur))
    return _PG._planar_zchar(pol, omega,
        term.epsr * _PG._EPS0, term.mur * _PG._MU0, gamma0)
end

function _ref_modal_voltage(stack, omega, kc2, pol, f, s)
    L = length(stack.layers)
    gamma = [sqrt(ComplexF64(kc2) -
        _PG.planar_k2_layer(omega, l.epsr, l.mur)) for l in stack.layers]
    zc = [_PG._planar_zchar(pol, omega, l.epsr * _PG._EPS0,
        l.mur * _PG._MU0, gamma[i]) for (i, l) in enumerate(stack.layers)]
    zdn = Vector{ComplexF64}(undef, L + 1)
    zup = Vector{ComplexF64}(undef, L + 1)
    zdn[1] = _ref_term(stack, :bottom, pol, omega, kc2)
    for l in 1:L
        zdn[l + 1] = _ref_input_impedance(zc[l], gamma[l] *
            stack.layers[l].thickness, zdn[l])
    end
    zup[L + 1] = _ref_term(stack, :top, pol, omega, kc2)
    for l in L:-1:1
        zup[l] = _ref_input_impedance(zc[l], gamma[l] *
            stack.layers[l].thickness, zup[l + 1])
    end
    zd, zu = zdn[s + 1], zup[s + 1]
    infd = !isfinite(zd); infu = !isfinite(zu)
    vs = if infd && infu
        ComplexF64(Inf)
    elseif infd
        zu
    elseif infu
        zd
    else
        den = zd + zu
        iszero(den) ? ComplexF64(Inf) : zd * zu / den
    end
    f == s && return vs
    v = vs
    if f > s    # propagate upward through layers s+1..f
        for l in (s + 1):f
            gd = gamma[l] * stack.layers[l].thickness
            u = zc[l] / zup[l + 1]
            v /= isfinite(u) ? cosh(gd) + u * sinh(gd) : Inf
        end
    else        # propagate downward through layers f+1..s
        for l in (f + 1):s
            gd = gamma[l] * stack.layers[l].thickness
            u = zc[l] / zdn[l]
            v /= isfinite(u) ? cosh(gd) + u * sinh(gd) : Inf
        end
    end
    return v
end

# direct (non-BLAS) impedance sum for one basis pair -- independent
# accumulation path used to validate the blocked assembly
function _direct_z(stack, grid, sheets, basis, omega, p, q; mx, my)
    mg = _PG.planar_mode_grid(grid, mx, my)
    nb = _PG.planar_basis_count(basis)
    iface = [sheets[basis.level[r]].interface for r in 1:nb]
    ab4 = grid.a * grid.b / 4
    fx_p = _PG._basis_fx!(zeros(mg.mx), basis, p, mg, grid)
    fy_p = _PG._basis_fy!(zeros(mg.my), basis, p, mg, grid)
    fx_q = _PG._basis_fx!(zeros(mg.mx), basis, q, mg, grid)
    fy_q = _PG._basis_fy!(zeros(mg.my), basis, q, mg, grid)
    acc = zero(ComplexF64)
    for n in 1:mg.my, m in 1:mg.mx
        kx, ky = mg.kx[m], mg.ky[n]
        kc2 = kx * kx + ky * ky
        nte2 = ky * ky * mg.ic[m] * mg.js[n] +
               kx * kx * mg.is[m] * mg.jc[n]
        if kc2 != 0 && nte2 != 0      # TE contribution
            wp = (_PG._is_xdir(basis.kind[p]) ? ky : -kx) *
                 fx_p[m] * fy_p[n]
            wq = (_PG._is_xdir(basis.kind[q]) ? ky : -kx) *
                 fx_q[m] * fy_q[n]
            v = _ref_modal_voltage(stack, omega, kc2, TE_POL,
                iface[p], iface[q])
            acc -= v * wp * wq / nte2
        end
        if m > 1 && n > 1             # TM contribution
            ntm2 = kc2 * ab4
            wp = (_PG._is_xdir(basis.kind[p]) ? kx : ky) *
                 fx_p[m] * fy_p[n]
            wq = (_PG._is_xdir(basis.kind[q]) ? kx : ky) *
                 fx_q[m] * fy_q[n]
            v = _ref_modal_voltage(stack, omega, kc2, TM_POL,
                iface[p], iface[q])
            acc -= v * wp * wq / ntm2
        end
    end
    return acc
end

# rooftop value at (x, y) for basis p -- independent of the production
# assembly, used to build the Gram overlap by plain quadrature
function _rooftop_at(basis, grid, p, x, y)
    dx, dy = grid.dx, grid.dy
    kind = basis.kind[p]
    j = basis.ej[p]
    if kind == _PG._BASIS_X_FULL
        (y < (j - 1) * dy || y >= j * dy) && return 0.0
        t = 1.0 - abs(x - basis.ei[p] * dx) / dx
        return max(t, 0.0)
    elseif kind == _PG._BASIS_X_LO
        (y < (j - 1) * dy || y >= j * dy) && return 0.0
        return (0 <= x < dx) ? 1.0 - x / dx : 0.0
    elseif kind == _PG._BASIS_X_HI
        (y < (j - 1) * dy || y >= j * dy) && return 0.0
        return (grid.a - dx <= x < grid.a) ? (x - (grid.a - dx)) / dx : 0.0
    end
    i = basis.ei[p]
    (x < (i - 1) * dx || x >= i * dx) && return 0.0
    if kind == _PG._BASIS_Y_FULL
        t = 1.0 - abs(y - basis.ej[p] * dy) / dy
        return max(t, 0.0)
    elseif kind == _PG._BASIS_Y_LO
        return (0 <= y < dy) ? 1.0 - y / dy : 0.0
    elseif kind == _PG._BASIS_Y_HI
        return (grid.b - dy <= y < grid.b) ? (y - (grid.b - dy)) / dy : 0.0
    end
    return 0.0
end

function _gram_quadrature(basis, grid, p, q; sub=64)
    dx, dy = grid.dx / sub, grid.dy / sub
    acc = 0.0
    for js in 1:grid.ny * sub
        y = (js - 0.5) * dy
        for is in 1:grid.nx * sub
            x = (is - 0.5) * dx
            acc += _rooftop_at(basis, grid, p, x, y) *
                   _rooftop_at(basis, grid, q, x, y)
        end
    end
    return acc * dx * dy
end

# canonical stripline problem: air dielectric, h gnd spacing, width w for
# 50 Ohm, quarter-wave line at 15 GHz, west/east ports
function _stripline_problem(nx, ny; bw=10.0e-3, two_ports=true)
    h = 1.0e-3
    w = 1.4423896e-3
    a = 4.99654097e-3
    stack = PlanarStackup(
        [PlanarLayer(1.0, 1.0, h / 2), PlanarLayer(1.0, 1.0, h / 2)],
        TERM_GND, TERM_GND, a, bw)
    grid = CellGrid(a, bw, nx, ny)
    s = sheet_level(1, nx, ny)
    rasterize_rect!(s, grid, 0.0, a, bw / 2 - w / 2, bw / 2 + w / 2)
    prow = findall(j -> any(s.mask[:, j]), 1:ny)
    for j in prow
        s.connect_west[j] = true
        s.connect_east[j] = true
    end
    ports = PlanarPort[PlanarPort(1, :west, prow[1]:prow[end], 50.0)]
    two_ports &&
        push!(ports, PlanarPort(1, :east, prow[1]:prow[end], 50.0))
    return build_planar_problem(stack, grid, [s], ports)
end

# ---------- tests ----------

@testset "planar: mode-grid pulse transforms vs quadrature" begin
    grid = CellGrid(12.0e-3, 8.0e-3, 12, 9)
    mg = planar_mode_grid(grid, 12, 9)
    dx, dy = grid.dx, grid.dy
    # unit triangle (peak 1, support 2*dx) vs cos(kx*x), centred at 0
    for m in (1, 2, 5, mg.mx)
        k = mg.kx[m]
        M = 4000
        num = sum(1:M) do s
            x = 2dx * (s - 0.5) / M - dx
            max(0.0, 1.0 - abs(x) / dx) * cos(k * x) * (2dx / M)
        end
        @test mg.tx_tri_cos[m] ≈ num rtol = 1e-5
    end
    # rect centred at 0 is even: stored transform is the cos integral
    # (the sin(k*y_c) cell-centre factor is applied separately)
    for n in (2, 5, mg.my)
        k = mg.ky[n]
        M = 2000
        num = sum(1:M) do s
            y = dy * (s - 0.5) / M - dy / 2
            cos(k * y) * (dy / M)
        end
        @test mg.ty_rect_sin[n] ≈ num rtol = 1e-5
    end
    # wall half-ramp (1 - x/dx) on [0,dx] vs cos(kx*x)
    for m in (1, 2, 5, mg.mx)
        k = mg.kx[m]
        M = 4000
        num = sum(1:M) do s
            x = dx * (s - 0.5) / M
            (1 - x / dx) * cos(k * x) * (dx / M)
        end
        @test mg.hx_cos[m] ≈ num rtol = 1e-5
    end
end

@testset "planar: cascade vs tanh-form reference" begin
    omega = 2pi * 10e9
    stack = PlanarStackup(
        [PlanarLayer(4.0, 1.0, 0.5e-3), PlanarLayer(1.0, 1.0, 0.25e-3),
         PlanarLayer(2.0, 1.0, 0.25e-3)],
        TERM_GND, TERM_GND, 10e-3, 8e-3)
    for m in 0:4, n in 0:3
        (m == 0 && n == 0) && continue
        kc2 = (m * pi / 10e-3)^2 + (n * pi / 8e-3)^2
        for pol in (TE_POL, TM_POL)
            (pol === TM_POL && (m == 0 || n == 0)) && continue
            c = planar_mode_cascade(stack, omega, kc2, pol)
            for f in 0:3, s in 0:3
                got = planar_modal_voltage(c, f, s)
                want = _ref_modal_voltage(stack, omega, kc2, pol, f, s)
                @test got ≈ want rtol = 1e-9
            end
        end
    end
    # mixed terminations (lossy dielectric, surface impedance, open top)
    t_open = PlanarTerminator{ComplexF64}(TERM_OPEN, 0.0im, 1.0, 1.0)
    t_surf = PlanarTerminator{ComplexF64}(TERM_SURFACE, 0.5 + 0.25im,
        1.0, 1.0)
    stack2 = PlanarStackup(
        [PlanarLayer(9.8 - 0.02im, 1.0, 0.635e-3),
         PlanarLayer(1.0, 1.0, 0.4e-3)],
        t_surf, t_open, 5e-3, 5e-3)
    for m in 0:3, n in 0:2
        (m == 0 && n == 0) && continue
        kc2 = (m * pi / 5e-3)^2 + (n * pi / 5e-3)^2
        for pol in (TE_POL, TM_POL)
            (pol === TM_POL && (m == 0 || n == 0)) && continue
            c = planar_mode_cascade(stack2, omega, kc2, pol)
            for f in 0:2, s in 0:2
                got = planar_modal_voltage(c, f, s)
                want = _ref_modal_voltage(stack2, omega, kc2, pol, f, s)
                @test got ≈ want rtol = 1e-9
            end
        end
    end
end

@testset "planar: assembly matches direct modal sum" begin
    omega = 2pi * 10e9
    stack = PlanarStackup(
        [PlanarLayer(4.0, 1.0, 0.5e-3), PlanarLayer(1.0, 1.0, 0.5e-3)],
        TERM_GND, TERM_GND, 8e-3, 6e-3)
    grid = CellGrid(8e-3, 6e-3, 8, 6)
    s1 = sheet_level(1, 8, 6)
    rasterize_rect!(s1, grid, 1e-3, 4e-3, 1e-3, 5e-3)
    s2 = sheet_level(2, 8, 6)
    rasterize_rect!(s2, grid, 4e-3, 7e-3, 2e-3, 5e-3)
    sheets = [s1, s2]
    basis = build_planar_basis(grid, sheets, PlanarPort[])
    # interleave basis ordering so same-interface bases are NOT contiguous
    ord = sortperm(basis.level)
    basis = _PG.PlanarBasisSet(basis.kind[ord], basis.ei[ord],
        basis.ej[ord], basis.level[ord], basis.x0[ord], basis.y0[ord],
        basis.width[ord], basis.port[ord])
    Z = assemble_planar_z(stack, grid, sheets, basis, omega;
        mx=24, my=20, block=7)
    nb = _PG.planar_basis_count(basis)
    @test size(Z) == (nb, nb)
    @test all(isfinite, Z)
    for (p, q) in ((1, 1), (1, nb), (nb, nb), (2, 3), (nb - 1, 2))
        d = _direct_z(stack, grid, sheets, basis, omega, p, q;
            mx=24, my=20)
        @test Z[p, q] ≈ d rtol = 1e-9
    end
    Z32 = assemble_planar_z(stack, grid, sheets, basis, omega;
        mx=24, my=20, block=32)
    Zall = assemble_planar_z(stack, grid, sheets, basis, omega;
        mx=24, my=20, block=100_000)
    @test Z32 ≈ Zall rtol = 1e-10
    @test Z32 == assemble_planar_z(stack, grid, sheets, basis, omega;
        mx=24, my=20, block=32)
    # Galerkin Z of a reciprocal network is symmetric
    @test Z ≈ transpose(Z) rtol = 1e-8
end

@testset "planar: surface-impedance Gram term" begin
    # strip connected to BOTH sidewalls: produces x=0 (LO) and x=a (HI)
    # half-rooftops, whose Gram neighbours are ei+1 and ei-1 respectively
    stack = PlanarStackup(
        [PlanarLayer(1.0, 1.0, 0.5e-3), PlanarLayer(1.0, 1.0, 0.5e-3)],
        TERM_GND, TERM_GND, 4e-3, 4e-3)
    grid = CellGrid(4e-3, 4e-3, 8, 6)
    s = sheet_level(1, 8, 6)
    rasterize_rect!(s, grid, 0.0, 4e-3, 1e-3, 3e-3)
    rows = findall(j -> any(s.mask[:, j]), 1:6)
    s.connect_west[rows] .= true
    s.connect_east[rows] .= true
    sheets = [s]
    basis = build_planar_basis(grid, sheets, PlanarPort[])
    omega = 2pi * 5e9
    zs = 0.4 + 0.3im
    Z0 = assemble_planar_z(stack, grid, sheets, basis, omega;
        mx=16, my=12)
    Zs = assemble_planar_z(stack, grid, sheets, basis, omega;
        mx=16, my=12, surface_zs=zs)
    dG = (Zs - Z0) / zs
    nb = _PG.planar_basis_count(basis)
    has_hi = any(basis.kind .== _PG._BASIS_X_HI)
    has_lo = any(basis.kind .== _PG._BASIS_X_LO)
    @test has_hi && has_lo
    for p in 1:nb, q in 1:nb
        want = _gram_quadrature(basis, grid, p, q; sub=48)
        if abs(want) > 1e-14 * grid.dx * grid.dy
            @test dG[p, q] ≈ want rtol = 2e-3
        else
            @test abs(dG[p, q]) <= 1e-14 * grid.dx * grid.dy
        end
    end
end

@testset "planar: low-frequency port conventions" begin
    # floating metal patch between two grounds, driven at the west edge:
    # quasi-static parallel-plate capacitor, Y11 positive imaginary.
    # Patch is inset on east/south/north so only the port edge reaches a
    # wall; residual excess over the ideal plate formula is fringing.
    h = 1.0e-3; a = 4.0e-3; b = 4.0e-3
    stack = PlanarStackup(
        [PlanarLayer(1.0, 1.0, h / 2), PlanarLayer(1.0, 1.0, h / 2)],
        TERM_GND, TERM_GND, a, b)
    grid = CellGrid(a, b, 16, 16)
    s = sheet_level(1, 16, 16)
    x1 = a - 2 * grid.dx
    y0 = 2 * grid.dy
    y1 = b - 2 * grid.dy
    rasterize_rect!(s, grid, 0.0, x1, y0, y1)
    prow = findall(j -> any(s.mask[:, j]), 1:16)
    for j in prow
        s.connect_west[j] = true
    end
    ports = [PlanarPort(1, :west, prow[1]:prow[end], 50.0)]
    prob = build_planar_problem(stack, grid, [s], ports)
    f = 0.5e9
    r = solve_planar(prob, f; mx=48, my=48)
    a_patch = x1 * (y1 - y0)
    c_expected = 4 * _PG._EPS0 * a_patch / h  # two faces x eps0*A/(h/2)
    @test imag(r.y[1, 1]) > 0                 # capacitive
    @test imag(r.y[1, 1]) ≈ 2pi * f * c_expected rtol = 0.35

    # single-port shorted stripline stub: inductive input (+j*omega*L)
    prob1 = _stripline_problem(16, 20; two_ports=false)
    r1 = solve_planar(prob1, 0.5e9; mx=48, my=60)
    zin = inv(r1.y)[1, 1]
    @test imag(zin) > 0
end

@testset "planar: stripline 50-Ohm benchmark" begin
    prob = _stripline_problem(32, 42)
    r = solve_planar(prob, 15e9; mx=4 * 32, my=4 * 42)
    @test abs(r.s[1, 1]) < 0.15
    @test 0.94 < abs(r.s[2, 1]) < 1.0
    @test -115 < rad2deg(angle(r.s[2, 1])) < -85
    zin = inv(r.y)
    z0_eq = sqrt(zin[1, 1] * zin[2, 2] - zin[1, 2] * zin[2, 1])
    @test 45 < real(z0_eq) < 58               # ~6% subsectioning error
end

@testset "planar: validation and limits" begin
    bad = PlanarStackup([PlanarLayer(-1.0, 1.0, 1e-3)], TERM_GND,
        TERM_GND, 1e-3, 1e-3)
    @test_throws ArgumentError planar_validate(bad)
    @test_throws ArgumentError CellGrid(1e-3, 1e-3, 0, 4)
    @test_throws ArgumentError planar_mode_grid(CellGrid(1e-3, 1e-3, 2, 2),
        0, 4)

    prob = _stripline_problem(16, 20)
    omega = 2pi * 1e9
    @test_throws ArgumentError assemble_planar_z(prob.stack, prob.grid,
        prob.sheets, prob.basis, -omega)
    @test_throws ArgumentError assemble_planar_z(prob.stack, prob.grid,
        prob.sheets, prob.basis, omega; block=0)
    @test_throws ArgumentError assemble_planar_z(prob.stack, prob.grid,
        prob.sheets, prob.basis, omega; max_bytes=100)
    @test_throws ArgumentError solve_planar(prob, 0.0)
    @test_throws ArgumentError solve_planar(prob, -1e9)

    # port with no connected edges -> build_planar_problem must fail
    gr = CellGrid(4.99654097e-3, 10e-3, 16, 20)
    s = sheet_level(1, 16, 20)
    rasterize_rect!(s, gr, 0.0, 4.99654097e-3, 4e-3, 6e-3)
    @test_throws ArgumentError build_planar_problem(prob.stack, gr, [s],
        [PlanarPort(1, :west, 8:10, 50.0)])
    # invalid wall symbol is caught during basis construction
    s.connect_west[8:10] .= true
    @test_throws ArgumentError build_planar_problem(prob.stack, gr, [s],
        [PlanarPort(1, :diagonal, 8:10, 50.0)])

    # malformed y_to_s inputs
    Y = Matrix{ComplexF64}(I, 2, 2) * (0.02 + 0im)
    @test_throws ArgumentError planar_y_to_s(Y, [50.0, -50.0])
    @test_throws ArgumentError planar_y_to_s(Y, [50.0, 50.0 + 10im])
    @test_throws DimensionMismatch planar_y_to_s(Y, [50.0])

    # Touchstone ordering: 2-port is column-major (S11 S21 S12 S22);
    # n>2 writes the frequency alone then rows in row-major order
    mktempdir() do dir
        s2 = ComplexF64[0.1+0.1im 0.3+0.3im; 0.2+0.2im 0.4+0.4im]
        p2 = joinpath(dir, "t.s2p")
        write_touchstone(p2, [1e9], [s2])
        fields = split.(filter(l -> !startswith(l, ('!', '#')),
            readlines(p2)))
        vals = parse.(Float64, fields[1])
        @test vals[2:end] ≈ [0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4]

        s3 = ComplexF64[10p+q + (p+q)im/10 for p in 1:3, q in 1:3]
        p3 = joinpath(dir, "t.s3p")
        write_touchstone(p3, [2e9], [s3])
        lines = filter(l -> !startswith(l, ('!', '#')), readlines(p3))
        @test parse(Float64, lines[1]) == 2e9
        @test length(lines) == 4
        row1 = parse.(Float64, split(lines[2]))
        @test row1 ≈ [11, 0.2, 12, 0.3, 13, 0.4]   # row p=1: S11 S12 S13
        row3 = parse.(Float64, split(lines[4]))
        @test row3 ≈ [31, 0.4, 32, 0.5, 33, 0.6]   # row p=3
        @test_throws DimensionMismatch write_touchstone(
            joinpath(dir, "bad.s3p"), [1e9, 2e9], [s3])
    end
end

@testset "planar: y_to_s with unequal reference impedances" begin
    znet = ComplexF64[60 10; 10 40]
    ynet = inv(znet)
    z0 = [50.0, 75.0]
    s_par = planar_y_to_s(ynet, z0)
    z0d = LinearAlgebra.Diagonal(ComplexF64.(z0))
    drt = LinearAlgebra.Diagonal(sqrt.(z0))
    s_ref = inv(drt) * ((znet - z0d) * inv(znet + z0d)) * drt
    @test s_par ≈ s_ref rtol = 1e-12
    # equal z0 reduces to the scalar form (Z - z0 I)(Z + z0 I)^-1
    s_eq = planar_y_to_s(ynet, [50.0, 50.0])
    s_eq_ref = (znet - 50I) * inv(znet + 50I)
    @test s_eq ≈ s_eq_ref rtol = 1e-12
end

@testset "planar: complex-step frequency perturbation" begin
    prob = _stripline_problem(16, 20)
    f0 = 10e9
    r0 = solve_planar(prob, f0; mx=32, my=40)
    eps_step = 1e-8 * f0
    rc = solve_planar(prob, ComplexF64(f0, eps_step); mx=32, my=40)
    @test all(isfinite, rc.s)
    # real part of S is stationary to O(eps^2) under the complex step
    @test real(rc.s[2, 1]) ≈ real(r0.s[2, 1]) rtol = 1e-6
    # the imaginary part carries eps*dS/df: cross-check via central FD
    h = 1e-4 * f0
    rp = solve_planar(prob, f0 + h; mx=32, my=40)
    rm = solve_planar(prob, f0 - h; mx=32, my=40)
    dfd = (rp.s[2, 1] - rm.s[2, 1]) / (2h)
    # second-order complex-step for complex f: dS/df ~= imag(S(f+ie))/e
    # is exact only for real-valued functions; instead verify magnitude of
    # perturbation is consistent with the FD slope
    @test abs(rc.s[2, 1] - r0.s[2, 1]) ≈ abs(dfd) * eps_step rtol = 0.01
end
