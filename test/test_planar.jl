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
function _stripline_problem(nx, ny; bw=10.0e-3, a=4.99654097e-3,
        two_ports=true)
    h = 1.0e-3
    w = 1.4423896e-3
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

@testset "planar: conductor surface impedance and roughness" begin
    sig_cu = 5.8e7
    # textbook values: delta(Cu, 1 GHz) = 2.088 um, Rs = 8.25 mOhm
    @test planar_skin_depth(1e9, sig_cu) ≈ 2.089e-6 rtol = 1e-3
    @test planar_skin_depth(4e9, sig_cu) ≈ planar_skin_depth(1e9, sig_cu) / 2 rtol = 1e-12
    zs = planar_surface_zs(1e9, sig_cu)
    @test real(zs) ≈ 8.2502e-3 rtol = 1e-3
    @test imag(zs) ≈ real(zs)          # (1+i)*Rs under e^{+i wt}

    # Hammerstad-Jensen: K = 1 + (2/pi)*atan(1.4*(rms/delta)^2)
    d1 = planar_skin_depth(1e9, sig_cu)
    @test roughness_factor(HammerstadRoughness(0.0), 1e9, sig_cu) == 1.0
    @test roughness_factor(HammerstadRoughness(d1), 1e9, sig_cu) ≈
        1 + (2 / pi) * atan(1.4) rtol = 1e-12
    @test roughness_factor(HammerstadRoughness(50 * d1), 1e9, sig_cu) ≈ 2.0 atol = 1e-3
    # modified form: rf is the asymptotic multiplier
    @test roughness_factor(HammerstadRoughness(50 * d1; rf=1.5),
        1e9, sig_cu) ≈ 1.5 atol = 1e-3
    # monotone in rms, saturating
    k1 = roughness_factor(HammerstadRoughness(0.5 * d1), 1e9, sig_cu)
    k2 = roughness_factor(HammerstadRoughness(2.0 * d1), 1e9, sig_cu)
    @test 1 < k1 < k2 < 2

    # Huray snowball: K = 1 + (3/2)*(4*pi*r^2*rho)/(1 + x + x^2/2), x = d/r
    r_n = 0.5e-6; rho = 1.4e11
    @test roughness_factor(HurayRoughness(r_n, 0.0), 1e9, sig_cu) == 1.0
    kh = roughness_factor(HurayRoughness(r_n, rho), 1e9, sig_cu)
    x = d1 / r_n
    @test kh ≈ 1 + 1.5 * 4pi * r_n^2 * rho / (1 + x + x^2 / 2) rtol = 1e-12
    # delta >> r (low f): nodules electrically small, fields bypass -> K -> 1
    @test roughness_factor(HurayRoughness(r_n, rho), 1.0, sig_cu) ≈ 1.0 atol = 1e-6
    # delta << r (high f): saturates at the nodule-area factor 1+6*pi*r^2*rho
    kmax = roughness_factor(HurayRoughness(r_n, rho), 1e16, sig_cu)
    @test kmax ≈ 1 + 1.5 * 4pi * r_n^2 * rho rtol = 1e-3
    @test 1 < kh < kmax

    # roughness application: full scaling vs loss_only (Re(Zs) only)
    zsr = planar_surface_zs(1e9, sig_cu;
        roughness=HurayRoughness(r_n, rho))
    @test zsr ≈ kh * zs rtol = 1e-12
    zsl = planar_surface_zs(1e9, sig_cu;
        roughness=HurayRoughness(r_n, rho), loss_only=true)
    @test real(zsl) ≈ kh * real(zs) rtol = 1e-12
    @test imag(zsl) ≈ imag(zs) rtol = 1e-12

    # end-to-end: the Gram loss term is linear in surface_zs, so a rough
    # surface scales the whole loss contribution by K
    stack = PlanarStackup(
        [PlanarLayer(1.0, 1.0, 0.5e-3), PlanarLayer(1.0, 1.0, 0.5e-3)],
        TERM_GND, TERM_GND, 4e-3, 4e-3)
    grid = CellGrid(4e-3, 4e-3, 8, 6)
    s = sheet_level(1, 8, 6)
    rasterize_rect!(s, grid, 0.0, 4e-3, 1e-3, 3e-3)
    rows = findall(j -> any(s.mask[:, j]), 1:6)
    s.connect_west[rows] .= true
    s.connect_east[rows] .= true
    basis = build_planar_basis(grid, [s], PlanarPort[])
    omega = 2pi * 1e9
    Z0 = assemble_planar_z(stack, grid, [s], basis, omega; mx=16, my=12)
    Zsm = assemble_planar_z(stack, grid, [s], basis, omega;
        mx=16, my=12, surface_zs=zs)
    Zro = assemble_planar_z(stack, grid, [s], basis, omega;
        mx=16, my=12, surface_zs=zsr)
    @test Zro - Z0 ≈ kh .* (Zsm - Z0) rtol = 1e-10
    @test maximum(real.(Zro - Zsm)) > 0   # extra dissipation

    # validation
    @test_throws ArgumentError HammerstadRoughness(-1e-6)
    @test_throws ArgumentError HammerstadRoughness(1e-6; rf=0.5)
    @test_throws ArgumentError HurayRoughness(0.0, 1e11)
    @test_throws ArgumentError HurayRoughness(1e-6, -1.0)
    @test_throws ArgumentError planar_skin_depth(0.0, sig_cu)
    @test_throws ArgumentError planar_skin_depth(1e9, 0.0)
    @test_throws ArgumentError planar_skin_depth(1e9, sig_cu; mur=0.0)
    @test_throws ArgumentError planar_surface_zs(NaN, sig_cu)
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

@testset "planar: port de-embedding" begin
    # synthetic calibration standards from known (yd, zc, gl)
    yd = 0.0003 + 0.004im
    zc = 53.0 + 0.7im
    gl = 0.32 + 0.05im
    Md = ComplexF64[1 0; yd 1]
    L = planar_line_abcd(zc, gl)
    L2 = planar_line_abcd(zc, 2gl)
    yl = _PG._y_of_abcd(Md * L * Md)
    y2l = _PG._y_of_abcd(Md * L2 * Md)
    cal = deembed_double_delay_calibrate(yl, y2l; len=5e-3)
    @test cal.yd ≈ yd
    @test cal.zc ≈ zc rtol = 1e-10
    @test cal.gamma_l ≈ gl rtol = 1e-10
    @test cal.residual < 1e-10

    # measured DUT = parasitic chain on each port around device
    ydev = ComplexF64[0.02+0.03im 0.005-0.001im;
                      0.005-0.001im -0.01+0.04im]
    M1 = Md * L
    M2 = Md * L
    A_ = Diagonal([M1[1, 1], M2[1, 1]])
    B_ = Diagonal([M1[1, 2], M2[1, 2]])
    C_ = Diagonal([M1[2, 1], M2[2, 1]])
    D_ = Diagonal([M1[2, 2], M2[2, 2]])
    ya = (C_ + D_ * ydev) * inv(A_ + B_ * ydev)
    @test deembed_ports(ya, [M1, M2]) ≈ ydev rtol = 1e-10
    @test deembed_double_delay_apply(ya, cal) ≈ ydev rtol = 1e-9
    # identity chains leave Y untouched
    I2 = Matrix{ComplexF64}(I, 2, 2)
    @test deembed_ports(ya, [I2, I2]) ≈ ya

    # 1-port port extension vs direct transmission-line input impedance
    zc0 = 50.0; gl0 = 0.7 + 0.02im
    yb1 = fill(0.015 + 0.002im, 1, 1)
    t = tanh(gl0)
    za = zc0 * (inv(yb1[1, 1]) + zc0 * t) / (zc0 + inv(yb1[1, 1]) * t)
    ya1 = fill(inv(za), 1, 1)
    @test deembed_port_extension(ya1, zc0, gl0)[1, 1] ≈ yb1[1, 1] rtol = 1e-10
    # removing a negative length equals adding it: round-trip identity
    yr = deembed_port_extension(
        deembed_port_extension(ya1, zc0, gl0), zc0, -gl0)
    @test yr[1, 1] ≈ ya1[1, 1] rtol = 1e-10

    # invalid inputs
    @test_throws DimensionMismatch deembed_ports(ya, [M1])
    @test_throws DimensionMismatch deembed_ports(ya, [M1, ones(2, 3)])
    @test_throws ArgumentError deembed_ports(
        fill(NaN * im, 2, 2), [M1, M2])
    @test_throws ArgumentError deembed_double_delay_calibrate(
        yl, y2l; len=0.0)
    @test_throws ArgumentError deembed_double_delay_calibrate(
        yl, y2l; len=5e-3, tol=-1.0)
    @test_throws DimensionMismatch deembed_double_delay_calibrate(
        yl, ones(3, 3); len=5e-3)
    # a standard pair that violates the pure-shunt assumption must fail:
    # discontinuity with a series impedance component gives P12 != 0
    Ms = ComplexF64[1 0.8+0.4im; 0 1] * Md   # series z + shunt yd
    ybl = _PG._y_of_abcd(Ms * L * Ms)
    yb2 = _PG._y_of_abcd(Ms * L2 * Ms)
    @test_throws ArgumentError deembed_double_delay_calibrate(
        ybl, yb2; len=5e-3)

    # solver end-to-end: calibrate on simulated thru standards
    # (4 mm and 8 mm strips) and de-embed a 12 mm strip; the residual is
    # a pure 4 mm line of the extracted (zc, gamma*l)
    f = 15e9; ell = 4.0e-3
    pl = _stripline_problem(16, 20; a=ell)
    p2l = _stripline_problem(32, 20; a=2ell)
    pdut = _stripline_problem(48, 20; a=3ell)
    y1 = solve_planar(pl, f).y
    y2 = solve_planar(p2l, f).y
    cs = deembed_double_delay_calibrate(y1, y2; len=ell)
    @test cs.residual < 1e-6         # gap ports are intrinsically shunt
    @test 40 < real(cs.zc) < 90      # ~50-70 Ohm at this discretization
    @test abs(imag(cs.gamma_l)) > 0.5   # electrically long enough
    # free-space electrical length of the 4 mm standard at 15 GHz
    @test imag(cs.gamma_l) ≈ 2pi * f * ell / _PG._C0 rtol = 0.02
    yd3 = solve_planar(pdut, f).y
    yde = deembed_double_delay_apply(yd3, cs)
    # device after removing Md*L(ell) at each port: pure ell-length line
    yline = _PG._y_of_abcd(planar_line_abcd(cs.zc, cs.gamma_l))
    @test yde ≈ yline rtol = 1e-4
    # naked-only removal leaves the full 3*ell line
    ynak = deembed_double_delay_apply(yd3, cs; line=false)
    yline3 = _PG._y_of_abcd(planar_line_abcd(cs.zc, 3 * cs.gamma_l))
    @test ynak ≈ yline3 rtol = 1e-4
end

@testset "planar: parameter adjoint gradients" begin
    # lossy stripline: nonzero imag baselines so the FD step sizes are sane
    h = 1.0e-3; w = 1.4423896e-3; a = 4.99654097e-3; bw = 10e-3
    stack = PlanarStackup(
        [PlanarLayer(2.2 - 0.04im, 1.3 + 0.01im, h / 2),
         PlanarLayer(2.2 - 0.04im, 1.3 + 0.01im, h / 2)],
        TERM_GND, TERM_GND, a, bw)
    grid = CellGrid(a, bw, 16, 20)
    s = sheet_level(1, 16, 20)
    rasterize_rect!(s, grid, 0.0, a, bw / 2 - w / 2, bw / 2 + w / 2)
    prow = findall(j -> any(s.mask[:, j]), 1:20)
    for j in prow
        s.connect_west[j] = true
        s.connect_east[j] = true
    end
    ports = PlanarPort[PlanarPort(1, :west, prow[1]:prow[end], 50.0),
                       PlanarPort(1, :east, prow[1]:prow[end], 50.0)]
    prob = build_planar_problem(stack, grid, [s], ports)
    f = Y -> abs2(planar_y_to_s(Y, [50.0, 50.0])[2, 1])
    freq = 15e9

    params = PlanarParam[PlanarParam(1, :epsr, :re),
        PlanarParam(1, :epsr, :im), PlanarParam(1, :mur, :re),
        PlanarParam(1, :mur, :im), PlanarParam(1, :thickness, :re),
        PlanarParam(2, :epsr, :re)]
    theta = planar_param_values(stack, params)
    @test theta ≈ [2.2, -0.04, 1.3, 0.01, 0.0005, 2.2]
    J, g = planar_objective_gradient(prob, freq, f; params=params)
    @test J ≈ f(solve_planar(prob, freq).y)
    for j in eachindex(params)
        hs = 1e-6 * max(abs(theta[j]), 1.0)
        tp = copy(theta); tm = copy(theta)
        tp[j] += hs; tm[j] -= hs
        Jp = f(solve_planar(PlanarProblem(
            planar_with_params(stack, params, tp), prob.grid,
            prob.sheets, prob.ports, prob.basis), freq).y)
        Jm = f(solve_planar(PlanarProblem(
            planar_with_params(stack, params, tm), prob.grid,
            prob.sheets, prob.ports, prob.basis), freq).y)
        fd = (Jp - Jm) / (2hs)
        @test g[j] ≈ fd rtol = 1e-4
    end
    # symmetric stackup: layer-1 and layer-2 epsr gradients must agree
    @test g[1] ≈ g[6] rtol = 1e-9

    # multi-interface contraction: sheets on interfaces 1 and 2
    a2 = 6e-3; bw2 = 6e-3; hh = 0.6e-3
    stack2 = PlanarStackup(
        [PlanarLayer(3.0 + 0im, 1.0, hh), PlanarLayer(1.0 + 0im, 1.0, hh),
         PlanarLayer(2.0 + 0im, 1.0, hh)],
        TERM_GND, TERM_GND, a2, bw2)
    grid2 = CellGrid(a2, bw2, 12, 12)
    s1 = sheet_level(1, 12, 12)
    rasterize_rect!(s1, grid2, 0.0, a2, bw2 / 2 - 0.4e-3, bw2 / 2 + 0.4e-3)
    s2 = sheet_level(2, 12, 12)
    rasterize_rect!(s2, grid2, 0.0, a2, bw2 / 2 - 0.8e-3, bw2 / 2 + 0.8e-3)
    pr1 = findall(j -> any(s1.mask[:, j]), 1:12)
    pr2 = findall(j -> any(s2.mask[:, j]), 1:12)
    for j in pr1
        s1.connect_west[j] = true; s1.connect_east[j] = true
    end
    for j in pr2
        s2.connect_west[j] = true; s2.connect_east[j] = true
    end
    prob2 = build_planar_problem(stack2, grid2, [s1, s2],
        PlanarPort[PlanarPort(1, :west, pr1[1]:pr1[end], 50.0),
                   PlanarPort(2, :east, pr2[1]:pr2[end], 50.0)])
    params2 = PlanarParam[PlanarParam(2, :epsr, :re),
        PlanarParam(2, :thickness, :re), PlanarParam(3, :mur, :re)]
    theta2 = planar_param_values(stack2, params2)
    J2, g2 = planar_objective_gradient(prob2, 8e9, f; params=params2)
    for j in eachindex(params2)
        hs = 1e-6 * max(abs(theta2[j]), 1.0)
        tp = copy(theta2); tm = copy(theta2)
        tp[j] += hs; tm[j] -= hs
        Jp = f(solve_planar(PlanarProblem(
            planar_with_params(stack2, params2, tp), prob2.grid,
            prob2.sheets, prob2.ports, prob2.basis), 8e9).y)
        Jm = f(solve_planar(PlanarProblem(
            planar_with_params(stack2, params2, tm), prob2.grid,
            prob2.sheets, prob2.ports, prob2.basis), 8e9).y)
        fd = (Jp - Jm) / (2hs)
        @test g2[j] ≈ fd rtol = 1e-4
    end

    # terminator parameters: surface-impedance top
    top = PlanarTerminator(TERM_SURFACE, 0.5 + 0.2im, 1.0 + 0im, 1.0 + 0im)
    stack3 = PlanarStackup(
        [PlanarLayer(2.2 + 0im, 1.0, hh), PlanarLayer(1.0 + 0im, 1.0, hh)],
        TERM_GND, top, a2, bw2)
    s3 = sheet_level(1, 12, 12)
    rasterize_rect!(s3, grid2, 0.0, a2, bw2 / 2 - 0.6e-3, bw2 / 2 + 0.6e-3)
    pr3 = findall(j -> any(s3.mask[:, j]), 1:12)
    for j in pr3
        s3.connect_west[j] = true; s3.connect_east[j] = true
    end
    prob3 = build_planar_problem(stack3, grid2, [s3],
        PlanarPort[PlanarPort(1, :west, pr3[1]:pr3[end], 50.0),
                   PlanarPort(1, :east, pr3[1]:pr3[end], 50.0)])
    params3 = PlanarParam[PlanarParam(3, :zs, :re),
        PlanarParam(3, :zs, :im), PlanarParam(1, :epsr, :re)]
    theta3 = planar_param_values(stack3, params3)
    J3, g3 = planar_objective_gradient(prob3, 8e9, f; params=params3)
    for j in eachindex(params3)
        hs = 1e-6 * max(abs(theta3[j]), 1.0)
        tp = copy(theta3); tm = copy(theta3)
        tp[j] += hs; tm[j] -= hs
        Jp = f(solve_planar(PlanarProblem(
            planar_with_params(stack3, params3, tp), prob3.grid,
            prob3.sheets, prob3.ports, prob3.basis), 8e9).y)
        Jm = f(solve_planar(PlanarProblem(
            planar_with_params(stack3, params3, tm), prob3.grid,
            prob3.sheets, prob3.ports, prob3.basis), 8e9).y)
        fd = (Jp - Jm) / (2hs)
        @test g3[j] ≈ fd rtol = 1e-4
    end

    # analytic Wirtinger gradient path: f = Re(Y11) has G[1,1] = 1
    fre = Y -> real(Y[1, 1])
    gY = Y -> ComplexF64[i == 1 && j == 1 ? 1.0 + 0im : 0.0 + 0im
                         for i in 1:2, j in 1:2]
    Jr, gr = planar_objective_gradient(prob, freq, fre;
        params=[PlanarParam(1, :epsr, :re)], gY=gY)
    _, gr_fd = planar_objective_gradient(prob, freq, fre;
        params=[PlanarParam(1, :epsr, :re)])
    @test gr[1] ≈ gr_fd[1] rtol = 1e-6

    # dual cascade: modal-voltage derivative vs central FD of the real
    # cascade on the same mode (independent of the contraction path)
    p_d = PlanarParam(1, :epsr, :re)
    ds = _PG._planar_dual_stackup(stack, p_d)
    omega = 2pi * freq
    kc2 = (pi / a)^2 + (pi / bw)^2
    h2 = 1e-7
    for pol in (TE_POL, TM_POL)
        cd = planar_mode_cascade(ds, omega, kc2, pol)
        sp = planar_with_params(stack, [p_d], [2.2 + h2])
        sm = planar_with_params(stack, [p_d], [2.2 - h2])
        cp = planar_mode_cascade(sp, omega, kc2, pol)
        cm = planar_mode_cascade(sm, omega, kc2, pol)
        for fi in 0:2, si in 0:2
            dv = planar_modal_voltage(cd, fi, si)
            fdv = (planar_modal_voltage(cp, fi, si) -
                   planar_modal_voltage(cm, fi, si)) / (2h2)
            @test dv.v ≈ planar_modal_voltage(
                planar_mode_cascade(stack, omega, kc2, pol), fi, si)
            @test dv.d ≈ fdv rtol = 1e-6
        end
    end

    # default params cover every layer scalar and run to finite values
    dp = planar_default_params(prob3.stack)
    @test any(p.field === :zs for p in dp)
    J4, g4 = planar_objective_gradient(prob3, 8e9, f; params=dp)
    @test all(isfinite, g4)

    # invalid descriptors and limits
    @test_throws ArgumentError PlanarParam(-1, :epsr, :re)
    @test_throws ArgumentError PlanarParam(1, :bogus, :re)
    @test_throws ArgumentError PlanarParam(1, :epsr, :cc)
    @test_throws ArgumentError planar_objective_gradient(prob, freq, f;
        params=[PlanarParam(9, :epsr, :re)])
    @test_throws ArgumentError planar_objective_gradient(prob, freq, f;
        params=[PlanarParam(1, :zs, :re)])
    @test_throws ArgumentError planar_with_params(stack,
        [PlanarParam(1, :epsr, :re)], [NaN])
    @test_throws DimensionMismatch planar_with_params(stack,
        [PlanarParam(1, :epsr, :re)], [2.2, 2.2])
    @test_throws ArgumentError planar_objective_gradient(prob, freq, f;
        params=[PlanarParam(1, :epsr, :re)], max_bytes=64)
    @test_throws ArgumentError planar_objective_gradient(prob, freq, f;
        params=[PlanarParam(1, :epsr, :re)], h_fd=-1.0)
    @test_throws ArgumentError planar_objective_gradient(prob, freq,
        Y -> NaN; params=[PlanarParam(1, :epsr, :re)])
end

@testset "planar: dual-scalar degenerate-mode limits" begin
    D = _PG._PlanarDual{ComplexF64}
    # sqrt at exact cutoff: zero seed must not produce NaN through 0/0
    x0 = D(0.0 + 0im, 0.0 + 0im)
    @test sqrt(x0).v == 0 && sqrt(x0).d == 0
    xs = D(0.0 + 0im, 1.0 + 0im)
    @test !isfinite(sqrt(xs).d)          # genuine singular derivative

    # inv_tau at gd = 0: limit value 1 with d = -u * e1.d
    u = D(0.3 + 0.2im, 9.0 + 9im)
    dx = 0.4 + 0.1im
    e1 = D(1.0 + 0im, -dx)              # e1 = exp(-gd), so e1.d = -dx
    e2 = D(1.0 + 0im, -2dx)
    it = _PG._planar_inv_tau(u, e2, e1)
    @test it.v == 1.0 + 0im
    @test it.d ≈ u.v * dx               # d/dgd(cosh + u*sinh) at 0 = u
    # reference via explicit dual cosh/sinh
    xdu = D(0.0 + 0im, dx)
    cosh_x = D(cosh(xdu.v), xdu.d * sinh(xdu.v))
    sinh_x = D(sinh(xdu.v), xdu.d * cosh(xdu.v))
    @test (cosh_x + u * sinh_x).d ≈ it.d

    # input impedance at e2 = 1: dZin = dzl + (Zc - zl^2/Zc)*dx
    zc = D(50.0 + 3im, 0.0im)
    zl = D(30.0 - 2im, 0.1 + 0.2im)
    zi = _PG._planar_input_impedance(zc, e2, zl)
    @test zi.v ≈ zl.v
    K = zc.v - zl.v * zl.v / zc.v
    @test zi.d ≈ zl.d + K * dx
    # tanh-form reference with duals: Zin = Zc(zl + Zc t)/(Zc + zl t)
    t = D(0.0 + 0im, dx)                # t = tanh(gd)
    ref = zc * (zl + zc * t) / (zc + zl * t)
    @test ref.d ≈ zi.d rtol = 1e-10
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
