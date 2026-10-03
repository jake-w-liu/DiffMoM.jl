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

# axial decay of a (possibly uniaxial) layer, written in kz form:
#   kz_TE^2 = k_t^2 - (mur/mur_z)*kc2,  kz_TM^2 = k_t^2 - (epsr/epsr_z)*kc2
# and gamma = sqrt(-kz^2) (exp(-gamma*z) convention, principal root)
function _ref_gamma(l, omega, kc2, pol)
    k2 = _PG.planar_k2_layer(omega, l.epsr, l.mur)
    a = pol === TE_POL ? l.mur / l.mur_z : l.epsr / l.epsr_z
    return sqrt(ComplexF64(-(k2 - a * kc2)))
end

function _ref_modal_voltage(stack, omega, kc2, pol, f, s)
    L = length(stack.layers)
    gamma = [_ref_gamma(l, omega, kc2, pol) for l in stack.layers]
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
    fx_p = _PG._basis_fx!(zeros(mg.mx), basis, p, mg, grid)
    fy_p = _PG._basis_fy!(zeros(mg.my), basis, p, mg, grid)
    fx_q = _PG._basis_fx!(zeros(mg.mx), basis, q, mg, grid)
    fy_q = _PG._basis_fy!(zeros(mg.my), basis, q, mg, grid)
    acc = zero(ComplexF64)
    pec = mg.walls == WALL_PEC
    for n in 1:mg.my, m in 1:mg.mx
        kx, ky = mg.kx[m], mg.ky[n]
        kc2 = kx * kx + ky * ky
        # modal norms swap sin/cos parity under PMC sidewalls; a zero norm
        # marks a mode that does not exist under the wall boundary
        nte2, ntm2 = if pec
            (ky * ky * mg.ic[m] * mg.js[n] +
             kx * kx * mg.is[m] * mg.jc[n],
             kx * kx * mg.ic[m] * mg.js[n] +
             ky * ky * mg.is[m] * mg.jc[n])
        else
            (ky * ky * mg.is[m] * mg.jc[n] +
             kx * kx * mg.ic[m] * mg.js[n],
             kx * kx * mg.is[m] * mg.jc[n] +
             ky * ky * mg.ic[m] * mg.js[n])
        end
        if kc2 != 0 && nte2 != 0      # TE contribution
            wp = (_PG._is_xdir(basis.kind[p]) ? ky : -kx) *
                 fx_p[m] * fy_p[n]
            wq = (_PG._is_xdir(basis.kind[q]) ? ky : -kx) *
                 fx_q[m] * fy_q[n]
            v = _ref_modal_voltage(stack, omega, kc2, TE_POL,
                iface[p], iface[q])
            acc -= v * wp * wq / nte2
        end
        if ntm2 != 0                  # TM contribution
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

@testset "planar: uniaxial layers" begin
    omega = 2pi * 10e9
    # constructor forms agree; isotropic default is bitwise-identical
    li = PlanarLayer(4.0 - 0.1im, 1.2 + 0.01im, 0.3e-3)
    lk = PlanarLayer(3.0, 1.0, 0.5e-3; epsr_z=5.0, mur_z=2.0)
    lp = PlanarLayer(3.0, 1.0, 0.5e-3, 5.0, 2.0)
    @test (lk.epsr_z, lk.mur_z) == (5.0, 2.0) == (lp.epsr_z, lp.mur_z)
    @test li.epsr_z == li.epsr && li.mur_z == li.mur

    # mixed isotropic/uniaxial stackup vs the independent kz-form cascade
    stack = PlanarStackup(
        [PlanarLayer(4.0, 1.0, 0.5e-3; epsr_z=6.0),
         PlanarLayer(2.0, 1.5, 0.25e-3; mur_z=3.0),
         PlanarLayer(9.0, 1.0, 0.25e-3)],
        TERM_GND, TERM_GND, 10e-3, 8e-3)
    for m in 0:4, n in 0:3
        (m == 0 && n == 0) && continue
        kc2 = (m * pi / 10e-3)^2 + (n * pi / 8e-3)^2
        for pol in (TE_POL, TM_POL)
            (pol === TM_POL && (m == 0 || n == 0)) && continue
            c = planar_mode_cascade(stack, omega, kc2, pol)
            for f in 0:3, s in 0:3
                @test planar_modal_voltage(c, f, s) ≈
                    _ref_modal_voltage(stack, omega, kc2, pol, f, s) rtol=1e-9
            end
        end
    end

    # isotropic reduction must be bitwise identical to the scalar formula
    kc2 = 1.5e5
    for pol in (TE_POL, TM_POL)
        @test _PG._planar_gamma_layer(pol, kc2, omega, li) ==
            sqrt(ComplexF64(kc2) -
                 _PG.planar_k2_layer(omega, li.epsr, li.mur))
    end

    # physical resonance: a PEC-PEC uniaxial cavity resonates when the
    # standing wave closes, gamma*h = i*n*pi (n=1 here).  For TM that is
    # k_t^2 = (pi/h)^2 + (epsr/epsr_z)*kc^2; for TE, (mur/mur_z)*kc^2.
    h = 2.0e-3; ab = 10e-3
    kc2 = (pi / ab)^2 + (pi / ab)^2     # (1,1) box mode
    et, ez = 3.0, 6.0
    cav_tm = PlanarStackup(
        [PlanarLayer(et, 1.0, h; epsr_z=ez)], TERM_GND, TERM_GND, ab, ab)
    k2_tm = (pi / h)^2 + (et / ez) * kc2
    f_tm = sqrt(k2_tm / et) * _PG._C0 / (2pi)
    mt, mz = 2.0, 4.0
    cav_te = PlanarStackup(
        [PlanarLayer(et, mt, h; mur_z=mz)], TERM_GND, TERM_GND, ab, ab)
    k2_te = (pi / h)^2 + (mt / mz) * kc2
    f_te = sqrt(k2_te / (et * mt)) * _PG._C0 / (2pi)
    for (cav, fres, pol) in ((cav_tm, f_tm, TM_POL),
                             (cav_te, f_te, TE_POL))
        c = planar_mode_cascade(cav, 2pi * fres, kc2, pol)
        res = abs(c.zdn[2] + c.zup[2])
        cd = planar_mode_cascade(cav, 2pi * fres * 1.02, kc2, pol)
        off = abs(cd.zdn[2] + cd.zup[2])
        @test isfinite(res) && res < 1e-2 * off
    end

    # assembly level: direct modal sum on a uniaxial stackup
    st_u = PlanarStackup(
        [PlanarLayer(4.0, 1.0, 0.5e-3; epsr_z=6.5),
         PlanarLayer(1.0, 1.0, 0.5e-3)],
        TERM_GND, TERM_GND, 8e-3, 6e-3)
    grid = CellGrid(8e-3, 6e-3, 8, 6)
    su = sheet_level(1, 8, 6)
    rasterize_rect!(su, grid, 1e-3, 4e-3, 1e-3, 5e-3)
    bu = build_planar_basis(grid, [su], PlanarPort[])
    Zu = assemble_planar_z(st_u, grid, [su], bu, omega; mx=20, my=16)
    @test Zu[1, 1] ≈ _direct_z(st_u, grid, [su], bu, omega, 1, 1;
        mx=20, my=16) rtol = 1e-9

    # axial parameter gradients through the dual-number cascade
    stack_g = PlanarStackup(
        [PlanarLayer(3.0 - 0.02im, 1.1, 0.5e-3; epsr_z=4.5, mur_z=0.8),
         PlanarLayer(1.0, 1.0, 0.5e-3)],
        TERM_GND, TERM_GND, ab, ab)
    grid_g = CellGrid(ab, ab, 12, 12)
    sg = sheet_level(1, 12, 12)
    rasterize_rect!(sg, grid_g, 0.0, ab, ab / 2 - 0.5e-3, ab / 2 + 0.5e-3)
    prg = findall(j -> any(sg.mask[:, j]), 1:12)
    for j in prg
        sg.connect_west[j] = true; sg.connect_east[j] = true
    end
    prob_g = build_planar_problem(stack_g, grid_g, [sg],
        PlanarPort[PlanarPort(1, :west, prg[1]:prg[end], 50.0),
                   PlanarPort(1, :east, prg[1]:prg[end], 50.0)])
    fg = Y -> abs2(planar_y_to_s(Y, [50.0, 50.0])[2, 1])
    params = PlanarParam[PlanarParam(1, :epsr_z, :re),
        PlanarParam(1, :epsr_z, :im), PlanarParam(1, :mur_z, :re),
        PlanarParam(1, :mur_z, :im), PlanarParam(1, :epsr, :re)]
    theta = planar_param_values(stack_g, params)
    @test theta ≈ [4.5, 0.0, 0.8, 0.0, 3.0]
    J, g = planar_objective_gradient(prob_g, 12e9, fg; params=params)
    for j in eachindex(params)
        hs = 1e-6 * max(abs(theta[j]), 1.0)
        tp = copy(theta); tm = copy(theta)
        tp[j] += hs; tm[j] -= hs
        Jp = fg(solve_planar(PlanarProblem(
            planar_with_params(stack_g, params, tp), prob_g.grid,
            prob_g.sheets, prob_g.ports, prob_g.basis), 12e9).y)
        Jm = fg(solve_planar(PlanarProblem(
            planar_with_params(stack_g, params, tm), prob_g.grid,
            prob_g.sheets, prob_g.ports, prob_g.basis), 12e9).y)
        @test g[j] ≈ (Jp - Jm) / (2hs) rtol = 1e-4
    end
    # default params: axial fields appear only for anisotropic layers
    dp_iso = planar_default_params(PlanarStackup([PlanarLayer(2.0, 1.0, 1e-3)],
        TERM_GND, TERM_GND, ab, ab))
    @test !any(p -> p.field in (:epsr_z, :mur_z), dp_iso)
    dp_an = planar_default_params(stack_g)
    @test any(p -> p.field === :epsr_z && p.index == 1, dp_an)
    @test any(p -> p.field === :mur_z && p.index == 1, dp_an)
    @test !any(p -> p.field in (:epsr_z, :mur_z) && p.index == 2, dp_an)

    # validation: axial constants must be finite with Re > 0
    @test_throws ArgumentError planar_validate(PlanarStackup(
        [PlanarLayer(2.0, 1.0, 1e-3; epsr_z=-1.0)],
        TERM_GND, TERM_GND, ab, ab))
    @test_throws ArgumentError planar_validate(PlanarStackup(
        [PlanarLayer(2.0, 1.0, 1e-3; mur_z=0.0)],
        TERM_GND, TERM_GND, ab, ab))
    @test_throws ArgumentError planar_validate(PlanarStackup(
        [PlanarLayer(2.0, 1.0, 1e-3; epsr_z=NaN)],
        TERM_GND, TERM_GND, ab, ab))
    @test_throws ArgumentError planar_param_values(stack_g,
        [PlanarParam(1, :epsr_y, :re)])
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

# quadrature of rooftop p against the transverse mode parity factors:
# x-directed bases couple through e_x (PEC cos*sin, PMC sin*cos),
# y-directed bases through e_y (PEC sin*cos, PMC cos*sin).  The sum is
# restricted to the basis support (one or two cells) so `sub` can be fine.
function _mode_overlap_quadrature(basis, grid, p, m, n, mg; sub=256)
    kx, ky = mg.kx[m], mg.ky[n]
    pec = mg.walls == WALL_PEC
    xdir = _PG._is_xdir(basis.kind[p])
    tx(x) = xdir == pec ? cos(kx * x) : sin(kx * x)
    ty(y) = xdir == pec ? sin(ky * y) : cos(ky * y)
    kind = basis.kind[p]
    i, j = basis.ei[p], basis.ej[p]
    dx, dy = grid.dx, grid.dy
    # flow-direction support spans two cells for a full rooftop, one for
    # a wall half; the transverse direction always spans one cell
    if kind == _PG._BASIS_X_LO || kind == _PG._BASIS_Y_LO
        xr = xdir ? (0.0, dx) : ((i - 1) * dx, i * dx)
        yr = xdir ? ((j - 1) * dy, j * dy) : (0.0, dy)
    elseif kind == _PG._BASIS_X_HI || kind == _PG._BASIS_Y_HI
        xr = xdir ? (grid.a - dx, grid.a) : ((i - 1) * dx, i * dx)
        yr = xdir ? ((j - 1) * dy, j * dy) : (grid.b - dy, grid.b)
    else
        xr = xdir ? ((i - 1) * dx, (i + 1) * dx) : ((i - 1) * dx, i * dx)
        yr = xdir ? ((j - 1) * dy, j * dy) : ((j - 1) * dy, (j + 1) * dy)
    end
    xs = range(xr[1], xr[2]; length=sub + 1)
    ys = range(yr[1], yr[2]; length=sub + 1)
    ddx = step(xs); ddy = step(ys)
    acc = 0.0
    for yy in Iterators.drop(ys, 1) .- ddy / 2, xx in Iterators.drop(xs, 1) .- ddx / 2
        acc += _rooftop_at(basis, grid, p, xx, yy) * tx(xx) * ty(yy)
    end
    return acc * ddx * ddy
end

@testset "planar: PMC sidewalls" begin
    omega = 2pi * 10e9
    stack = PlanarStackup(
        [PlanarLayer(4.0, 1.0, 0.5e-3), PlanarLayer(1.0, 1.0, 0.5e-3)],
        TERM_GND, TERM_GND, 8e-3, 6e-3)
    g_pec = CellGrid(8e-3, 6e-3, 8, 6)
    g_pmc = CellGrid(8e-3, 6e-3, 8, 6; walls=WALL_PMC)
    @test g_pec.walls == WALL_PEC
    @test g_pmc.walls == WALL_PMC
    @test_throws TypeError CellGrid(8e-3, 6e-3, 8, 6; walls=:pmc)

    mg_p = planar_mode_grid(g_pmc, 10, 8)
    @test mg_p.walls == WALL_PMC

    # half-ramp sin transform (kh - sin kh)/(k^2 h) vs quadrature
    @test mg_p.hx_sin[1] == 0.0
    @test mg_p.hy_sin[1] == 0.0
    for m in (2, 4, mg_p.mx)
        k = mg_p.kx[m]
        M = 8000
        num = sum(1:M) do s
            x = g_pmc.dx * (s - 0.5) / M
            (1 - x / g_pmc.dx) * sin(k * x) * (g_pmc.dx / M)
        end
        @test mg_p.hx_sin[m] ≈ num rtol = 1e-5
    end

    # modal norm^2 factors vs box quadrature of the modal |e_t|^2:
    # PMC TE e ∝ (ky sinx cosy, -kx cosx siny); TM e ∝ (kx sinx cosy, ky cosx siny)
    # (midpoint rule integrates the sin^2/cos^2 products exactly)
    K = 240
    ddx, ddy = g_pmc.a / K, g_pmc.b / K
    for m in 1:mg_p.mx, n in 1:mg_p.my
        kx, ky = mg_p.kx[m], mg_p.ky[n]
        te_num = tm_num = 0.0
        for jj in 1:K, ii in 1:K
            x, y = (ii - 0.5) * ddx, (jj - 0.5) * ddy
            sx, cx = sin(kx * x), cos(kx * x)
            sy, cy = sin(ky * y), cos(ky * y)
            te_num += ky^2 * sx^2 * cy^2 + kx^2 * cx^2 * sy^2
            tm_num += kx^2 * sx^2 * cy^2 + ky^2 * cx^2 * sy^2
        end
        te_num *= ddx * ddy
        tm_num *= ddx * ddy
        @test te_num ≈ ky^2 * mg_p.is[m] * mg_p.jc[n] +
                      kx^2 * mg_p.ic[m] * mg_p.js[n] rtol = 1e-10
        @test tm_num ≈ kx^2 * mg_p.is[m] * mg_p.jc[n] +
                      ky^2 * mg_p.ic[m] * mg_p.js[n] rtol = 1e-10
    end

    # mode existence masks: PMC drops TE on the m=0/n=0 axes but keeps
    # TM there (except the transverse-free (0,0)); PEC does the converse
    function _mode_volt(walls, m, n)
        L = length(stack.layers)
        mkc() = _PG.PlanarCascade(zeros(ComplexF64, L + 1),
            zeros(ComplexF64, L + 1), zeros(ComplexF64, L),
            zeros(ComplexF64, L))
        scr = (zeros(ComplexF64, L), zeros(ComplexF64, L),
            zeros(ComplexF64, L))
        grid = CellGrid(stack.a, stack.b, 4, 4; walls=walls)
        mg = planar_mode_grid(grid, m + 1, n + 1)
        vte = zeros(ComplexF64, 1, 1)
        vtm = zeros(ComplexF64, 1, 1)
        _PG._planar_mode_voltages!(vte, vtm, mkc(), mkc(), scr,
            stack, omega, mg, [m], [n], [(1, 1)])
        return vte[1, 1], vtm[1, 1]
    end
    te01, tm01 = _mode_volt(WALL_PMC, 1, 2)
    @test iszero(te01) && !iszero(tm01)   # PMC: TE(0,n) absent, TM(0,n) lives
    te10, tm10 = _mode_volt(WALL_PMC, 2, 1)
    @test iszero(te10) && !iszero(tm10)   # PMC: TE(m,0) absent, TM(m,0) lives
    @test all(iszero, _mode_volt(WALL_PMC, 1, 1))  # no uniform (0,0) mode
    te01e, tm01e = _mode_volt(WALL_PEC, 1, 2)
    @test !iszero(te01e) && iszero(tm01e) # PEC: TE(0,n) lives, TM absent
    te10e, tm10e = _mode_volt(WALL_PEC, 2, 1)
    @test !iszero(te10e) && iszero(tm10e)
    @test all(!iszero, _mode_volt(WALL_PMC, 2, 2))
    @test all(!iszero, _mode_volt(WALL_PEC, 2, 2))

    # separable basis transforms vs 2-D quadrature of the rooftop against
    # the wall-specific mode parity factors (all six basis kinds, both
    # wall kinds); PMC walls get wall-connected half rooftops too
    for walls in (WALL_PEC, WALL_PMC)
        gr = CellGrid(8e-3, 6e-3, 8, 6; walls=walls)
        mg = planar_mode_grid(gr, 8, 6)
        s = sheet_level(1, 8, 6)
        rasterize_rect!(s, gr, 0.0, gr.a, 0.0, gr.b)
        for j in 1:gr.ny
            s.connect_west[j] = true
            s.connect_east[j] = true
        end
        for i in 1:gr.nx
            s.connect_south[i] = true
            s.connect_north[i] = true
        end
        basis = build_planar_basis(gr, [s], PlanarPort[])
        fxb = zeros(mg.mx)
        fyb = zeros(mg.my)
        kinds = [_PG._BASIS_X_FULL, _PG._BASIS_X_LO, _PG._BASIS_X_HI,
            _PG._BASIS_Y_FULL, _PG._BASIS_Y_LO, _PG._BASIS_Y_HI]
        for kd in kinds
            p = findfirst(==(kd), basis.kind)
            @test !isnothing(p)   # full-metal sheet must spawn every kind
            _PG._basis_fx!(fxb, basis, p, mg, gr)
            _PG._basis_fy!(fyb, basis, p, mg, gr)
            for (m, n) in ((1, 2), (2, 1), (3, 3), (mg.mx, mg.my))
                num = _mode_overlap_quadrature(basis, gr, p, m, n, mg)
                @test fxb[m] * fyb[n] ≈ num rtol = 2e-4 atol = 1e-12
            end
        end
    end

    # blocked assembly vs the independent direct modal sum on PMC walls
    s1p = sheet_level(1, 8, 6)
    rasterize_rect!(s1p, g_pmc, 0.0, 5e-3, 1e-3, 5e-3)
    for j in findall(jj -> any(s1p.mask[:, jj]), 1:6)
        s1p.connect_west[j] = true
    end
    s2p = sheet_level(2, 8, 6)
    rasterize_rect!(s2p, g_pmc, 3e-3, 8e-3, 2e-3, 6e-3)
    for i in findall(ii -> any(s2p.mask[ii, :]), 1:8)
        s2p.connect_north[i] = true
    end
    sheets_p = [s1p, s2p]
    basis_p = build_planar_basis(g_pmc, sheets_p, PlanarPort[])
    Zp = assemble_planar_z(stack, g_pmc, sheets_p, basis_p, omega;
        mx=20, my=16, block=9)
    nb_p = _PG.planar_basis_count(basis_p)
    @test size(Zp) == (nb_p, nb_p)
    @test all(isfinite, Zp)
    for (p, q) in ((1, 1), (1, nb_p), (nb_p, nb_p), (2, 3), (nb_p - 1, 2))
        d = _direct_z(stack, g_pmc, sheets_p, basis_p, omega, p, q;
            mx=20, my=16)
        @test Zp[p, q] ≈ d rtol = 1e-9
    end
    @test Zp ≈ transpose(Zp) rtol = 1e-8   # reciprocity
    @test Zp == assemble_planar_z(stack, g_pmc, sheets_p, basis_p, omega;
        mx=20, my=16, block=9)             # deterministic assembly

    # end-to-end port solve on PMC walls: a gap port on a magnetic wall
    # drives the sheet edge against a non-conducting boundary, so the
    # line end is electrically open and the transmission collapses
    sline = sheet_level(1, 8, 6)
    rasterize_rect!(sline, g_pmc, 0.0, 8e-3, 2e-3, 4e-3)
    for j in findall(jj -> any(sline.mask[:, jj]), 1:6)
        sline.connect_west[j] = true
        sline.connect_east[j] = true
    end
    prow = findall(jj -> any(sline.mask[:, jj]), 1:6)
    prob_pmc = build_planar_problem(stack, g_pmc, [sline],
        PlanarPort[PlanarPort(1, :west, prow[1]:prow[end], 50.0),
                   PlanarPort(1, :east, prow[1]:prow[end], 50.0)])
    r_pmc = solve_planar(prob_pmc, omega)
    @test all(isfinite, r_pmc.y)
    @test r_pmc.y ≈ transpose(r_pmc.y) rtol = 1e-8
    @test abs(r_pmc.s[1, 1]) > 0.9          # open-ended strip reflects
    @test abs(r_pmc.s[2, 1]) < 0.2          # ... and hardly transmits
    prob_pec = build_planar_problem(stack, g_pec, [sline],
        PlanarPort[PlanarPort(1, :west, prow[1]:prow[end], 50.0),
                   PlanarPort(1, :east, prow[1]:prow[end], 50.0)])
    r_pec = solve_planar(prob_pec, omega)
    @test !isapprox(r_pec.y[1, 1], r_pmc.y[1, 1]; rtol=1e-3)

    # adjoint gradient on a PMC-wall problem matches finite differences
    f = Y -> imag(Y[1, 1])
    params = PlanarParam[PlanarParam(1, :epsr, :re),
        PlanarParam(1, :thickness, :re), PlanarParam(2, :mur, :re)]
    theta = planar_param_values(stack, params)
    J, g = planar_objective_gradient(prob_pmc, 10e9, f; params=params)
    @test J ≈ f(solve_planar(prob_pmc, 10e9).y)
    for j in eachindex(params)
        hs = 1e-6 * max(abs(theta[j]), 1.0)
        tp = copy(theta); tm = copy(theta)
        tp[j] += hs; tm[j] -= hs
        Jp = f(solve_planar(PlanarProblem(
            planar_with_params(stack, params, tp), prob_pmc.grid,
            prob_pmc.sheets, prob_pmc.ports, prob_pmc.basis), 10e9).y)
        Jm = f(solve_planar(PlanarProblem(
            planar_with_params(stack, params, tm), prob_pmc.grid,
            prob_pmc.sheets, prob_pmc.ports, prob_pmc.basis), 10e9).y)
        @test g[j] ≈ (Jp - Jm) / (2hs) rtol = 1e-4
    end
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

@testset "planar: circuit-element extraction" begin
    f = 10e9; om = 2pi * f; len = 6e-3
    # known per-unit-length constants -> line ABCD -> extract
    zp = 12.0 + om * 250e-9im          # 12 Ohm/m, 250 nH/m
    yp = 0.05 + om * 80e-12im          # 0.05 S/m, 80 pF/m
    zc = sqrt(zp / yp); gl = len * sqrt(zp * yp)
    Yl = _PG._y_of_abcd(planar_line_abcd(zc, gl))
    rl = planar_rlgc(Yl, len, f)
    @test rl.r ≈ 12.0 rtol = 1e-10
    @test rl.l ≈ 250e-9 rtol = 1e-10
    @test rl.g ≈ 0.05 rtol = 1e-10
    @test rl.c ≈ 80e-12 rtol = 1e-10
    lp = planar_line_params(Yl)
    @test lp.zc ≈ zc rtol = 1e-12
    @test lp.gl ≈ gl rtol = 1e-12
    # from a double-delay calibration
    cal = DoubleDelayCal(0.001im, zc, gl, len, 1e-12)
    rl2 = planar_rlgc(cal, f)
    @test rl2.l ≈ rl.l rtol = 1e-12
    @test rl2.c ≈ rl.c rtol = 1e-12

    # pi-model roundtrip from known branches
    ys0 = 0.01 + 0.03im; ya0 = 0.002 - 0.001im; yb0 = 0.004 + 0.0005im
    Yp = ComplexF64[ya0+ys0 -ys0; -ys0 yb0+ys0]
    pm = planar_pi_model(Yp)
    @test pm.ys ≈ ys0 && pm.ya ≈ ya0 && pm.yb ≈ yb0

    # inductor: series RL branch -> 2-port Y -> extract L, R, Q
    zI = 0.8 + om * 4e-9im
    Yi = ComplexF64[1 -1; -1 1] .* inv(zI) .+
         ComplexF64[0.001 0; 0 0.002]
    ip = planar_inductor(Yi, f)
    @test ip.r ≈ 0.8 rtol = 1e-10
    @test ip.l ≈ 4e-9 rtol = 1e-10
    @test ip.q ≈ imag(zI) / real(zI) rtol = 1e-10
    ip1 = planar_inductor(fill(inv(zI), 1, 1), f)
    @test ip1.l ≈ 4e-9 rtol = 1e-10
    # lossless branch -> infinite Q
    ipl = planar_inductor(fill(inv(om * 4e-9im), 1, 1), f)
    @test ipl.q == Inf

    # invalid inputs
    @test_throws DimensionMismatch planar_rlgc(ones(3, 3), len, f)
    @test_throws ArgumentError planar_rlgc(Yl, 0.0, f)
    @test_throws ArgumentError planar_rlgc(Yl, len, -f)
    @test_throws ArgumentError planar_rlgc(fill(NaN * im, 2, 2), len, f)
    @test_throws DimensionMismatch planar_pi_model(ones(3, 3))
    @test_throws DimensionMismatch planar_inductor(ones(3, 3), f)
    @test_throws ArgumentError planar_inductor(zeros(2, 2), f)
    # non-line section (shunt only: B = 0) fails closed
    @test_throws ArgumentError planar_line_params(
        ComplexF64[0.01 0; 0 0.01])

    # solver end-to-end: air stripline is TEM, so L*C = mu0*eps0 = 1/c0^2
    # and the extracted p.u.l. values are frequency-consistent
    fe = 15e9; elle = 4.0e-3
    cs = deembed_double_delay_calibrate(
        solve_planar(_stripline_problem(16, 20; a=elle), fe).y,
        solve_planar(_stripline_problem(32, 20; a=2elle), fe).y;
        len=elle)
    yd3 = solve_planar(_stripline_problem(48, 20; a=3elle), fe).y
    yde = deembed_double_delay_apply(yd3, cs)   # pure elle line
    rle = planar_rlgc(yde, elle, fe)
    @test rle.l * rle.c ≈ 1 / _PG._C0^2 rtol = 0.02
    @test rle.l > 0 && rle.c > 0
    rlcal = planar_rlgc(cs, fe)
    @test rlcal.l ≈ rle.l rtol = 0.05
    @test rlcal.c ≈ rle.c rtol = 0.05
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

# ---------- via columns ----------

# axial via profiles on t = u/h in [0,1]: uniform, up-taper, and the
# internal down-taper (reversed up-taper used for Ju moments)
_via_prof(prof, t) = prof == 1 ? 1.0 : prof == 2 ? t : 1.0 - t

# quadrature reference for the 2*exp(-x)-scaled moment coefficients:
#   cA[prof] = 2 e^-x int_0^1 w(t) cosh(x t) dt,
#   cB[prof] = 2 e^-x int_0^1 w(t) sinh(x t) dt
function _via_moment_ref(x, prof, fun; n=8192)
    acc = 0.0
    for k in 1:n
        t = (k - 0.5) / n
        acc += _via_prof(prof, t) * fun(x * t)
    end
    return 2 * exp(-x) * acc / n
end

# driven-line reference for the via<->via kernel: integrate the modal TM
# transmission line V' = -gamma*Zc*I + s(u), I' = -(gamma/Zc)*V over the
# stack by shooting (linear, so one homogeneous + one particular pass),
# then reduce the field-side moment int w_a*V' du over layer ja.
function _via_line_moment(stack, omega, kc2, jb, kb, ja, ka; n=8192)
    L = length(stack.layers)
    hlay = [Float64(real(l.thickness)) for l in stack.layers]
    zb = [0.0; cumsum(hlay)]
    gam = [_PG._planar_gamma_layer(TM_POL, kc2, omega, l)
           for l in stack.layers]
    zc = [gam[i] / (im * omega * stack.layers[i].epsr * _PG._EPS0)
          for i in 1:L]
    H = zb[end]; du = H / n
    lay(u) = clamp(searchsortedlast(zb, min(u, H - du / 2)), 1, L)
    hsrc = hlay[jb]
    wb(t) = _via_prof(kb == _PG._BASIS_VIA_U ? 1 : 2, t / hsrc)
    function integrate(with_source)
        V, I = zero(ComplexF64), one(ComplexF64)   # V(0) = -zbot*I(0)
        Vs = Vector{ComplexF64}(undef, n + 1)
        Is = similar(Vs)
        Vs[1] = V; Is[1] = I
        for k in 1:n
            u = (k - 1) * du
            function rhs(u, V, I)
                s = with_source && lay(u) == jb ? wb(u - zb[jb]) : 0.0
                l = lay(u)
                return -gam[l] * zc[l] * I + s, -(gam[l] / zc[l]) * V
            end
            k1v, k1i = rhs(u, V, I)
            k2v, k2i = rhs(u + du / 2, V + du / 2 * k1v, I + du / 2 * k1i)
            k3v, k3i = rhs(u + du / 2, V + du / 2 * k2v, I + du / 2 * k2i)
            k4v, k4i = rhs(u + du, V + du * k3v, I + du * k3i)
            V += du / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
            I += du / 6 * (k1i + 2 * k2i + 2 * k3i + k4i)
            Vs[k + 1] = V; Is[k + 1] = I
        end
        return Vs, Is
    end
    Qv, Qi = integrate(false)          # homogeneous (I(0) = 1)
    Pv, Pi = integrate(true)           # particular + I(0) = 1
    res(Ve, Ie) = Ve[end]              # top face: V(H) = ztop*I = 0
    alpha = -res(Pv, Pi) / res(Qv, Qi)
    V = Pv .+ alpha .* Qv
    I = Pi .+ alpha .* Qi
    lo, hi = zb[ja], zb[ja + 1]
    hfld = hlay[ja]
    wa(t) = _via_prof(ka == _PG._BASIS_VIA_U ? 1 : 2, t / hfld)
    acc = zero(ComplexF64)
    for k in 1:n
        u0 = (k - 1) * du
        (u0 >= lo && u0 < hi) || continue
        um = u0 + du / 2
        # V' = -gamma*Zc*(I - i_z); the impressed source adds back inside
        # its own layer (the filament contact handled separately below)
        s = ja == jb ? wb(um - lo) : 0.0
        vp = -gam[ja] * zc[ja] * (I[k] + I[k + 1]) / 2 + s
        acc += wa(um - lo) * vp
    end
    return acc * du
end

@testset "planar: via columns" begin
    omega = 2pi * 10e9
    # layer 2 is uniaxial so beta = kc^2 eps_t/(gamma^2 eps_z) != 1+k^2/g^2
    stack = PlanarStackup(
        [PlanarLayer(4.0, 1.0, 0.4e-3),
         PlanarLayer(2.2, 1.0, 0.6e-3; epsr_z=1.6),
         PlanarLayer(2.5, 1.0, 0.3e-3)],
        TERM_GND, TERM_GND, 8e-3, 6e-3)
    L = length(stack.layers)
    grid = CellGrid(8e-3, 6e-3, 8, 6)

    # --- axial moment coefficients: closed forms vs quadrature,
    #     small-x analytic continuations, and branch-edge agreement ---
    for x in (0.05, 0.7, 3.0, 9.0)
        e1 = exp(-x); e2 = e1 * e1
        for prof in 1:3
            @test _PG._via_ca(x, e1, e2, prof) ≈
                _via_moment_ref(x, prof, cosh) rtol = 1e-5
            @test _PG._via_cb(x, e1, e2, prof) ≈
                _via_moment_ref(x, prof, sinh) rtol = 1e-5
        end
        @test _PG._via_q1(x, e1, e2) ≈
            (_via_moment_ref(x, 3, cosh) - _via_moment_ref(x, 2, cosh)) /
            x rtol = 1e-5
        @test _PG._via_q2(x, e1, e2) ≈
            _via_moment_ref(x, 3, sinh) / x rtol = 1e-5
        @test _PG._via_q3(x, e1, e2) ≈
            _via_moment_ref(x, 2, sinh) / x rtol = 1e-5
        @test _PG._via_cbx_u(x, e1, e2) ≈
            _via_moment_ref(x, 1, sinh) / x rtol = 1e-5
        @test _PG._via_wbar(x, e1) ≈ tanh(x / 2) / x rtol = 1e-10
    end
    # x -> 0 limits of each analytic-continuation branch, normalized by
    # the 2*e1 scaling
    xs = 1e-6; e1s = exp(-xs); e2s = e1s * e1s
    @test _PG._via_ca(xs, e1s, e2s, 1) / e1s ≈ 2.0 rtol = 1e-10
    @test _PG._via_ca(xs, e1s, e2s, 2) / e1s ≈ 1.0 rtol = 1e-10
    @test _PG._via_ca(xs, e1s, e2s, 3) / e1s ≈ 1.0 rtol = 1e-10
    @test _PG._via_cb(xs, e1s, e2s, 1) / (e1s * xs) ≈ 1.0 rtol = 1e-8
    @test _PG._via_cb(xs, e1s, e2s, 2) / (e1s * xs) ≈ 2 / 3 rtol = 1e-8
    @test _PG._via_cb(xs, e1s, e2s, 3) / (e1s * xs) ≈ 1 / 3 rtol = 1e-8
    @test _PG._via_q1(xs, e1s, e2s) / (e1s * xs) ≈ -1 / 6 rtol = 1e-8
    @test _PG._via_q2(xs, e1s, e2s) / e1s ≈ 1 / 3 rtol = 1e-10
    @test _PG._via_q3(xs, e1s, e2s) / e1s ≈ 2 / 3 rtol = 1e-10
    @test _PG._via_cbx_u(xs, e1s, e2s) / e1s ≈ 1.0 rtol = 1e-10
    @test _PG._via_wbar(xs, e1s) ≈ 0.5 rtol = 1e-10
    # pin the x^2 order term of the q1 series (regression: 1/180, not 1/240)
    xb = 0.005; e1b = exp(-xb); e2b = e1b * e1b
    @test _PG._via_q1(xb, e1b, e2b) ≈
        -2 * e1b * xb * (1 / 12 + xb * xb / 180) rtol = 1e-9
    # the |x| < 0.01 series branch must reproduce the closed forms
    # evaluated inside the branch (same analytic function)
    xc = 0.0095; e1c = exp(-xc); e2c = e1c * e1c
    @test _PG._via_ca(xc, e1c, e2c, 1) ≈ (1 - e2c) / xc rtol = 1e-9
    @test _PG._via_ca(xc, e1c, e2c, 2) ≈
        (1 - e2c) / xc - (1 + e2c - 2 * e1c) / xc^2 rtol = 1e-8
    @test _PG._via_ca(xc, e1c, e2c, 3) ≈
        (1 + e2c - 2 * e1c) / xc^2 rtol = 1e-8
    @test _PG._via_cb(xc, e1c, e2c, 1) ≈ (1 + e2c - 2 * e1c) / xc rtol = 1e-8
    @test _PG._via_cb(xc, e1c, e2c, 2) ≈
        ((xc - 1) + e2c * (xc + 1)) / xc^2 rtol = 1e-8
    @test _PG._via_cb(xc, e1c, e2c, 3) ≈
        ((1 - e2c) - 2 * xc * e1c) / xc^2 rtol = 1e-7
    @test _PG._via_q1(xc, e1c, e2c) ≈
        (2 * (1 + e2c - 2 * e1c) - xc * (1 - e2c)) / xc^3 rtol = 1e-5
    @test _PG._via_q2(xc, e1c, e2c) ≈
        ((1 - e2c) - 2 * xc * e1c) / xc^3 rtol = 1e-6
    @test _PG._via_q3(xc, e1c, e2c) ≈
        ((xc - 1) + e2c * (xc + 1)) / xc^3 rtol = 1e-7
    @test _PG._via_cbx_u(xc, e1c, e2c) ≈
        (1 + e2c - 2 * e1c) / xc^2 rtol = 1e-8
    @test _PG._via_wbar(xc, e1c) ≈ (1 - e1c) / ((1 + e1c) * xc) rtol = 1e-10

    # --- via <- sheet kernel telescopes to face voltages (independent
    #     tanh-form cascade reference) ---
    mg = planar_mode_grid(grid, 10, 8)
    for n in (2, 5), m in (2, 6)
        kx, ky = mg.kx[m], mg.ky[n]
        kc2 = kx * kx + ky * ky
        ntm2 = kx^2 * mg.ic[m] * mg.js[n] + ky^2 * mg.is[m] * mg.jc[n]
        n3_2 = mg.is[m] * mg.js[n]
        casc = planar_mode_cascade(stack, omega, kc2, TM_POL)
        vsts = [_PG._via_layer_state(stack, casc, omega, kc2, j)
                for j in 1:L]
        for j in 1:L, s in 0:L
            lj = stack.layers[j]
            gj = _ref_gamma(lj, omega, kc2, TM_POL)
            k2t = _PG.planar_k2_layer(omega, lj.epsr, lj.mur)
            coef = 1 + k2t / (gj * gj)
            wbar = tanh(gj * lj.thickness / 2) / (gj * lj.thickness)
            vt = _ref_modal_voltage(stack, omega, kc2, TM_POL, j, s)
            vb = _ref_modal_voltage(stack, omega, kc2, TM_POL, j - 1, s)
            for kind in (_PG._BASIS_VIA_U, _PG._BASIS_VIA_T)
                at, ab, kk = kind == _PG._BASIS_VIA_U ?
                    (1.0, -1.0, 0.0) : (1.0, 0.0, -1.0)
                ref = coef * (at * vt + ab * vb +
                    kk * wbar * (vt + vb)) / ntm2
                got = _PG._elem_pair_pol(_PG._via_elem(j, kind), s,
                    casc, vsts, _PG._VolLayerState{ComplexF64}[],
                    ntm2, n3_2)
                @test got ≈ ref rtol = 1e-10
                # sheet <- via is the reciprocal of the same kernel
                got2 = _PG._elem_pair_pol(s, _PG._via_elem(j, kind),
                    casc, vsts, _PG._VolLayerState{ComplexF64}[],
                    ntm2, n3_2)
                @test got2 == got
            end
        end
    end

    # --- via <-> via kernel vs an independent driven-line solve ---
    combos = Tuple{Int,UInt8}[]
    for j in 1:L
        push!(combos, (j, _PG._BASIS_VIA_U), (j, _PG._BASIS_VIA_T))
    end
    for n in (1, 2, 4), m in (1, 3, 7)
        kx, ky = mg.kx[m], mg.ky[n]
        kc2 = kx * kx + ky * ky
        ntm2 = kx^2 * mg.ic[m] * mg.js[n] + ky^2 * mg.is[m] * mg.jc[n]
        n3_2 = mg.is[m] * mg.js[n]
        ntm2 == 0 && continue
        casc = planar_mode_cascade(stack, omega, kc2, TM_POL)
        vsts = [_PG._via_layer_state(stack, casc, omega, kc2, j)
                for j in 1:L]
        for (jb, kb) in combos, (ja, ka) in combos
            M = _via_line_moment(stack, omega, kc2, jb, kb, ja, ka)
            iab = ja == jb ? _PG._via_overlap(ka, kb,
                Float64(real(stack.layers[ja].thickness))) : 0.0
            st = vsts[ja]
            om_eps_z = omega * stack.layers[jb].epsr_z * _PG._EPS0
            refk = 1im * (st.coef * M + (1 - st.coef) * iab) /
                (om_eps_z * n3_2)
            got = _PG._elem_pair_pol(_PG._via_elem(ja, ka),
                _PG._via_elem(jb, kb), casc, vsts,
                _PG._VolLayerState{ComplexF64}[], ntm2, n3_2)
            @test got ≈ refk rtol = 3e-3   # shooting-rule residual
        end
    end

    # --- PMC sidewall parity: via transforms follow the g3 factor ---
    for walls in (WALL_PEC, WALL_PMC)
        gr = CellGrid(8e-3, 6e-3, 8, 6; walls=walls)
        mgw = planar_mode_grid(gr, 10, 8)
        sv = sheet_level(1, 8, 6)
        vpm = via_level(1, 8, 6)
        vpm.uni[3, 4] = true
        bw = build_planar_basis(gr, [sv], PlanarPort[]; vias=[vpm])
        pv = _PG.planar_basis_count(bw)
        @test bw.kind[pv] == _PG._BASIS_VIA_U
        fxv = _PG._basis_fx!(zeros(mgw.mx), bw, pv, mgw, gr)
        fyv = _PG._basis_fy!(zeros(mgw.my), bw, pv, mgw, gr)
        # via lateral profile is a rectangle pulse on cell (3,4):
        # PEC transforms vs sin, PMC vs cos
        dx, dy = gr.dx, gr.dy
        for m in (2, 4, mgw.mx)
            k = mgw.kx[m]
            num = sum(0:511) do q
                x = (3 - 1) * dx + (q + 0.5) * dx / 512
                (walls == WALL_PEC ? sin(k * x) : cos(k * x)) * dx / 512
            end
            @test fxv[m] ≈ num rtol = 1e-5
        end
        for n in (2, 5, mgw.my)
            k = mgw.ky[n]
            num = sum(0:511) do q
                y = (4 - 1) * dy + (q + 0.5) * dy / 512
                (walls == WALL_PEC ? sin(k * y) : cos(k * y)) * dy / 512
            end
            @test fyv[n] ≈ num rtol = 1e-5
        end
    end

    # --- validation boundaries ---
    @test_throws ArgumentError via_level(0, 8, 6)
    s0 = sheet_level(1, 8, 6)
    rasterize_rect!(s0, grid, 0.0, 3e-3, 1e-3, 5e-3)
    # via layer outside 1:L rejected at problem build
    badlay = ViaLevel(L + 1, falses(8, 6), falses(8, 6))
    badlay.uni[1, 1] = true
    @test_throws ArgumentError build_planar_problem(stack, grid, [s0],
        PlanarPort[PlanarPort(1, :west, 1:2, 50.0)]; vias=[badlay])
    # mask shape must match the cell grid
    badsz = ViaLevel(1, falses(4, 4), falses(4, 4))
    @test_throws DimensionMismatch build_planar_basis(grid, [s0],
        PlanarPort[]; vias=[badsz])
    badsz2 = ViaLevel(1, falses(8, 6), falses(8, 7))
    @test_throws DimensionMismatch build_planar_basis(grid, [s0],
        PlanarPort[]; vias=[badsz2])
    # a basis that references a missing via level must fail, not guess
    vone = via_level(1, 8, 6)
    vone.uni[2, 2] = true
    bv = build_planar_basis(grid, [s0], PlanarPort[]; vias=[vone])
    @test_throws ArgumentError assemble_planar_z(stack, grid, [s0],
        bv, omega; vias=ViaLevel[], mx=8, my=8)
    # exact axial cutoff (gamma == 0) is a loud DomainError
    kc2_cut = real(_PG.planar_k2_layer(omega,
        stack.layers[1].epsr, stack.layers[1].mur))
    casc0 = _PG.PlanarCascade(zeros(ComplexF64, L + 1),
        zeros(ComplexF64, L + 1), zeros(ComplexF64, L),
        zeros(ComplexF64, L))
    @test_throws DomainError _PG._via_layer_state(stack, casc0, omega,
        kc2_cut, 1)

    # --- basis enumeration: one basis per marked cell per kind ---
    vl = via_level(1, 8, 6)
    vl.uni[2, 3] = true; vl.tap[2, 3] = true; vl.tap[4, 1] = true
    pr1 = findall(j -> any(s0.mask[:, j]), 1:6)
    s0.connect_west[pr1[1]] = true
    probv = build_planar_problem(stack, grid, [s0],
        PlanarPort[PlanarPort(1, :west, pr1[1]:pr1[1], 50.0)];
        vias=[vl])
    nb = _PG.planar_basis_count(probv.basis)
    @test count(>=(_PG._BASIS_VIA_U), probv.basis.kind) == 3
    # pushes run j-outer, i-inner, uni before tap within a cell:
    # (4,1) tap first, then (2,3) uni then (2,3) tap
    @test probv.basis.kind[nb - 2] == _PG._BASIS_VIA_T
    @test probv.basis.kind[nb - 1] == _PG._BASIS_VIA_U
    @test probv.basis.kind[nb] == _PG._BASIS_VIA_T
    @test probv.basis.width[nb] == grid.dx * grid.dy

    # --- assembled Z with vias: finite, symmetric, block-size invariant
    Z = assemble_planar_z(stack, grid, [s0], probv.basis, omega;
        vias=[vl], mx=12, my=10)
    @test size(Z) == (nb, nb) && all(isfinite, Z)
    @test Z ≈ transpose(Z) rtol = 1e-10
    Zb = assemble_planar_z(stack, grid, [s0], probv.basis, omega;
        vias=[vl], mx=12, my=10, block=5)
    @test maximum(abs.(Z .- Zb)) / maximum(abs.(Z)) < 1e-12
    # a via level with empty masks contributes no bases and no coupling
    vempty = via_level(2, 8, 6)
    Ze = assemble_planar_z(stack, grid, [s0], probv.basis, omega;
        vias=[vl, vempty], mx=12, my=10)
    @test Ze == Z

    # direct per-mode sum for one via-sheet and one via-via entry: the
    # via<-sheet reference uses the independent tanh-form cascade
    function _direct_pair(p, q)
        kp, kq = probv.basis.kind[p], probv.basis.kind[q]
        ep = kp >= _PG._BASIS_VIA_U ?
            _PG._via_elem(vl.layer, kp) : s0.interface
        eq = kq >= _PG._BASIS_VIA_U ?
            _PG._via_elem(vl.layer, kq) : s0.interface
        mgd = planar_mode_grid(grid, 12, 10)
        fxp = _PG._basis_fx!(zeros(mgd.mx), probv.basis, p, mgd, grid)
        fyp = _PG._basis_fy!(zeros(mgd.my), probv.basis, p, mgd, grid)
        fxq = _PG._basis_fx!(zeros(mgd.mx), probv.basis, q, mgd, grid)
        fyq = _PG._basis_fy!(zeros(mgd.my), probv.basis, q, mgd, grid)
        acc = zero(ComplexF64)
        for n in 1:mgd.my, m in 1:mgd.mx
            kx, ky = mgd.kx[m], mgd.ky[n]
            kc2 = kx^2 + ky^2
            nte2 = ky^2 * mgd.ic[m] * mgd.js[n] +
                kx^2 * mgd.is[m] * mgd.jc[n]
            ntm2 = kx^2 * mgd.ic[m] * mgd.js[n] +
                ky^2 * mgd.is[m] * mgd.jc[n]
            n3_2 = mgd.is[m] * mgd.js[n]
            if ep >= 0 && eq >= 0
                xdir_p = _PG._is_xdir(kp); xdir_q = _PG._is_xdir(kq)
                if kc2 != 0 && nte2 != 0
                    wp = (xdir_p ? ky : -kx) * fxp[m] * fyp[n]
                    wq = (xdir_q ? ky : -kx) * fxq[m] * fyq[n]
                    acc -= _ref_modal_voltage(stack, omega, kc2, TE_POL,
                        ep, eq) * wp * wq / nte2
                end
                if ntm2 != 0
                    wp = (xdir_p ? kx : ky) * fxp[m] * fyp[n]
                    wq = (xdir_q ? kx : ky) * fxq[m] * fyq[n]
                    acc -= _ref_modal_voltage(stack, omega, kc2, TM_POL,
                        ep, eq) * wp * wq / ntm2
                end
            elseif ntm2 != 0
                # via pair: via transforms are raw fx*fy; a sheet keeps
                # its TM kx/ky factor
                wp = kp >= _PG._BASIS_VIA_U ? fxp[m] * fyp[n] :
                    (_PG._is_xdir(kp) ? kx : ky) * fxp[m] * fyp[n]
                wq = kq >= _PG._BASIS_VIA_U ? fxq[m] * fyq[n] :
                    (_PG._is_xdir(kq) ? kx : ky) * fxq[m] * fyq[n]
                if ep < 0 && eq < 0
                    casc = planar_mode_cascade(stack, omega, kc2, TM_POL)
                    vsts = [_PG._via_layer_state(stack, casc, omega,
                        kc2, j) for j in 1:L]
                    kern = _PG._elem_pair_pol(ep, eq, casc, vsts,
                        _PG._VolLayerState{ComplexF64}[], ntm2, n3_2)
                else
                    # telescope the via field element onto the sheet's
                    # face voltages from the independent tanh cascade
                    fe, se = ep < 0 ? (ep, eq) : (eq, ep)
                    jv = _PG._via_elem_layer(fe)
                    kv = _PG._via_elem_kind(fe)
                    lj = stack.layers[jv]
                    gj = _ref_gamma(lj, omega, kc2, TM_POL)
                    k2t = _PG.planar_k2_layer(omega, lj.epsr, lj.mur)
                    coef = 1 + k2t / (gj * gj)
                    wbar = tanh(gj * lj.thickness / 2) /
                        (gj * lj.thickness)
                    vt = _ref_modal_voltage(stack, omega, kc2, TM_POL,
                        jv, se)
                    vb = _ref_modal_voltage(stack, omega, kc2, TM_POL,
                        jv - 1, se)
                    at, ab, kk = kv == _PG._BASIS_VIA_U ?
                        (1.0, -1.0, 0.0) : (1.0, 0.0, -1.0)
                    kern = coef * (at * vt + ab * vb +
                        kk * wbar * (vt + vb)) / ntm2
                end
                acc -= kern * wp * wq
            end
        end
        return acc
    end
    psheet = findfirst(<(_PG._BASIS_VIA_U), probv.basis.kind)
    @test Z[nb - 2, psheet] ≈ _direct_pair(nb - 2, psheet) rtol = 1e-9
    @test Z[nb - 2, nb - 1] ≈ _direct_pair(nb - 2, nb - 1) rtol = 1e-9
    @test Z[psheet, nb] ≈ _direct_pair(psheet, nb) rtol = 1e-9

    # --- end to end: via wall shunts the stripline to ground ---
    probw = _stripline_problem(16, 20)
    s_w = probw.sheets[1]
    prow = findall(j -> any(s_w.mask[:, j]), 1:20)
    istrip = findall(i -> any(s_w.mask[i, :]), 1:16)
    vwall = via_level(1, 16, 20)
    for i in istrip, j in prow
        vwall.uni[i, j] = true
    end
    probw2 = build_planar_problem(probw.stack, probw.grid,
        probw.sheets, probw.ports; vias=[vwall])
    rv = solve_planar(probw2, 1.0e9; mx=48, my=60)
    r0 = solve_planar(probw, 1.0e9; mx=48, my=60)
    @test all(isfinite, rv.s)
    @test rv.s[2, 1] ≈ rv.s[1, 2] rtol = 1e-10   # reciprocity
    # the via wall shifts the port susceptance in the inductive direction
    # (imag(Y) decreases); the volume-current column is a partial clamp,
    # not an ideal short — verified independently at kernel level
    yv = rv.y[1, 1]; y0 = r0.y[1, 1]
    @test isfinite(yv)
    @test imag(yv) < imag(y0)
    @test 1e-4 < abs(yv - y0) / abs(y0) < 0.5

    # --- gradient through the via kernels matches finite difference ---
    s_g = sheet_level(1, 8, 6)
    rasterize_rect!(s_g, grid, 0.0, 8e-3, 2e-3, 4e-3)
    prg = findall(j -> any(s_g.mask[:, j]), 1:6)
    for j in prg
        s_g.connect_west[j] = true; s_g.connect_east[j] = true
    end
    vg = via_level(1, 8, 6)
    vg.uni[4, 3] = true; vg.tap[4, 3] = true
    ports_g = PlanarPort[PlanarPort(1, :west, prg[1]:prg[end], 50.0),
                         PlanarPort(1, :east, prg[1]:prg[end], 50.0)]
    probg = build_planar_problem(stack, grid, [s_g], ports_g;
        vias=[vg])
    fg = Y -> abs2(planar_y_to_s(Y, [50.0, 50.0])[2, 1])
    params = PlanarParam[PlanarParam(1, :epsr, :re),
        PlanarParam(1, :thickness, :re)]
    theta = planar_param_values(stack, params)
    J, g = planar_objective_gradient(probg, 8e9, fg; params=params)
    @test J ≈ fg(solve_planar(probg, 8e9).y)
    for j in eachindex(params)
        hs = 1e-6 * max(abs(theta[j]), 1.0)
        tp = copy(theta); tm = copy(theta)
        tp[j] += hs; tm[j] -= hs
        Jp = fg(solve_planar(PlanarProblem(
            planar_with_params(stack, params, tp), probg.grid,
            probg.sheets, probg.ports, probg.vias, probg.basis), 8e9).y)
        Jm = fg(solve_planar(PlanarProblem(
            planar_with_params(stack, params, tm), probg.grid,
            probg.sheets, probg.ports, probg.vias, probg.basis), 8e9).y)
        @test g[j] ≈ (Jp - Jm) / (2hs) rtol = 1e-4
    end
end

# single-layer modal-voltage Green's function by direct 4x4 BVP solve:
# V(0+) satisfies the zdn load, V(h-) the zup load, V continuous at the
# source plane u with the normalized derivative jump  zc .
function _vol_bvp_green(gam, zc, zdn, zup, h, u, z)
    e_u, e_h = exp(-gam * u), exp(-gam * h)
    M = ComplexF64[
        (zc + zdn) (zc - zdn) 0 0;
        0 0 e_h * (zc - zup) inv(e_h) * (zc + zup);
        e_u inv(e_u) -e_u -inv(e_u);
        -e_u inv(e_u) e_u -inv(e_u)]
    x = M \ ComplexF64[0, 0, 0, zc]
    a1, b1, a2, b2 = x
    return z <= u ? a1 * exp(-gam * z) + b1 * exp(gam * z) :
                    a2 * exp(-gam * z) + b2 * exp(gam * z)
end

_vol_trapz(f, xs) = sum((xs[i + 1] - xs[i]) *
    (f(xs[i]) + f(xs[i + 1])) / 2 for i in 1:length(xs) - 1)

@testset "planar: volume rooftops" begin
    omega = 2pi * 10e9
    stack = PlanarStackup(
        [PlanarLayer(4.0, 1.0, 0.4e-3),
         PlanarLayer(2.2, 1.0, 0.6e-3; epsr_z=1.6),
         PlanarLayer(2.5, 1.0, 0.3e-3)],
        TERM_GND, TERM_GND, 8e-3, 6e-3)
    L = length(stack.layers)
    grid = CellGrid(8e-3, 6e-3, 8, 6)

    # --- endpoint and self kernels vs the independent BVP Green's
    #     function, single layer with prescribed endpoint loads ---
    eps0 = _PG._EPS0
    for (gam, zc, zdn, zup) in (
            (2.0 + 0im, 3.0, 1.0, 5.0),
            (0.8 - 0.05im, 1.2 + 0.3im, 0.4, 2.0 + 1.0im),
            (5.0 + 0im, 0.6, 3.0, 0.2))
        # fabricate a TM layer that realizes (gamma, Zc) at omega=1:
        #   Zc = gamma/(i*eps) -> eps = gamma/(i*Zc)
        w1 = 1.0
        epsr = gam / (1im * zc * eps0)
        lay = PlanarLayer(epsr, 1.0 + 0im, 1.0)
        stk = PlanarStackup([lay], TERM_GND, TERM_GND, 1.0, 1.0)
        # gamma^2 = (epsr/epsr_z)*kc2 - w^2*epsr*mur/c0^2
        kc2 = real(gam^2 + w1^2 * epsr * 1.0 / _PG._C0^2)
        casc = _PG.PlanarCascade(ComplexF64[zdn, 0.0],
            ComplexF64[0.0, zup], ComplexF64[1.0], ComplexF64[1.0])
        st = _PG._vol_layer_state(stk, casc, w1, kc2, 1, TM_POL)
        h = 1.0
        # the state runs on the layer's effective gamma/Zc -- feed the
        # same effective values to the independent BVP
        geff = _ref_gamma(lay, w1, kc2, TM_POL)
        zceff = _PG._planar_zchar(TM_POL, w1, lay.epsr * eps0,
            lay.mur * _PG._MU0, geff)
        nz, nu = 800, 800
        zs = range(0, h; length=nz); us = range(0, h; length=nu)
        vt_ref = _vol_trapz(
            u -> _vol_bvp_green(geff, zceff, zdn, zup, h, u, h) / h, us)
        vb_ref = _vol_trapz(
            u -> _vol_bvp_green(geff, zceff, zdn, zup, h, u, 0.0) / h, us)
        ms_ref = _vol_trapz(u -> _vol_trapz(
            z -> _vol_bvp_green(geff, zceff, zdn, zup, h, u, z) / h,
            zs) / h, us)
        @test st.vt ≈ vt_ref rtol = 1e-3
        @test st.vb ≈ vb_ref rtol = 1e-3
        @test st.mself ≈ ms_ref rtol = 1e-3
        # wbar = tanh(x/2)/x on the effective (complex) layer gamma
        @test st.wbar ≈ tanh(geff * h / 2) / (geff * h) rtol = 1e-10
    end

    # --- the unscaled closed forms agree with the scaled state on a
    #     real stackup (independent formula organization) ---
    mg = planar_mode_grid(grid, 10, 8)
    for n in (2, 5), m in (2, 6), pol in (TE_POL, TM_POL)
        kx, ky = mg.kx[m], mg.ky[n]
        kc2 = kx^2 + ky^2
        casc = planar_mode_cascade(stack, omega, kc2, pol)
        for j in 1:L
            lay = stack.layers[j]
            h = Float64(real(lay.thickness))
            gam = _ref_gamma(lay, omega, kc2, pol)
            zc = _PG._planar_zchar(pol, omega, lay.epsr * eps0,
                lay.mur * _PG._MU0, gam)
            x = gam * h
            zdn, zup = casc.zdn[j], casc.zup[j + 1]
            Sh = sinh(x) / gam; Ch = (cosh(x) - 1) / gam
            idu = zdn * Sh + zc * Ch; iuu = zup * Sh + zc * Ch
            dz = zc * (zdn + zup) * cosh(x) +
                (zc * zc + zdn * zup) * sinh(x)
            vt_ref = zc * zup * idu / (h * dz)
            vb_ref = zc * zdn * iuu / (h * dz)
            phi_d(u) = zdn * cosh(gam * u) + zc * sinh(gam * u)
            phi_u(u) = zup * cosh(gam * (h - u)) +
                zc * sinh(gam * (h - u))
            G(z, u) = zc * (z <= u ? phi_d(z) * phi_u(u) :
                                     phi_d(u) * phi_u(z)) / dz
            ms_ref = _vol_trapz(u -> _vol_trapz(
                z -> G(z, u), range(0, h; length=400)) / h,
                range(0, h; length=2000)) / h
            st = _PG._vol_layer_state(stack, casc, omega, kc2, j, pol)
            @test st.vt ≈ vt_ref rtol = 1e-9
            @test st.vb ≈ vb_ref rtol = 1e-9
            @test st.mself ≈ ms_ref rtol = 1e-4
        end
    end

    # --- thin layer -> interface sheet (mid-stack, V ~ const) ---
    # the residual is first-order in the layer thickness
    for (h, tol) in ((1e-6, 3e-3), (1e-8, 5e-5))
        stk = PlanarStackup(
            [PlanarLayer(1.0, 1.0, 1e-3), PlanarLayer(1.0, 1.0, h),
             PlanarLayer(1.0, 1.0, 1e-3)],
            TERM_GND, TERM_GND, 1.0, 1.0)
        s1 = sheet_level(2, 8, 6)
        rasterize_rect!(s1, grid, 1e-3, 6e-3, 2e-3, 4e-3)
        bs = build_planar_basis(grid, [s1], PlanarPort[])
        Zs = assemble_planar_z(stk, grid, [s1], bs, omega; mx=12, my=10)
        vl = vol_level(2, 8, 6)
        rasterize_rect!(vl, grid, 1e-3, 6e-3, 2e-3, 4e-3)
        bv = build_planar_basis(grid, SheetLevel[], PlanarPort[];
            vols=[vl])
        Zv = assemble_planar_z(stk, grid, SheetLevel[], bv, omega;
            mx=12, my=10, vols=[vl])
        @test maximum(abs.(Zv - Zs)) / maximum(abs.(Zs)) < tol
    end

    # --- assembled Z: finite, symmetric through every pair family ---
    sv = sheet_level(1, 8, 6)
    rasterize_rect!(sv, grid, 0.0, 5e-3, 1e-3, 5e-3; connected=true)
    vv = vol_level(2, 8, 6)
    rasterize_rect!(vv, grid, 1e-3, 6e-3, 1e-3, 5e-3)
    vvia = via_level(3, 8, 6)
    vvia.uni[4, 3] = true; vvia.tap[4, 3] = true
    ports_v = PlanarPort[PlanarPort(1, :west, 3:4, 50.0)]
    probv = build_planar_problem(stack, grid, [sv], ports_v;
        vias=[vvia], vols=[vv])
    nb = _PG.planar_basis_count(probv.basis)
    @test count(k -> _PG._is_vol_kind(k), probv.basis.kind) > 0
    # vol bases carry no port
    @test all(p -> !_PG._is_vol_kind(probv.basis.kind[p]) ||
        probv.basis.port[p] == 0, 1:nb)
    Z = assemble_planar_z(stack, grid, [sv], probv.basis, omega;
        vias=[vvia], vols=[vv], mx=12, my=10)
    @test size(Z) == (nb, nb) && all(isfinite, Z)
    @test Z ≈ transpose(Z) rtol = 1e-10
    Zb = assemble_planar_z(stack, grid, [sv], probv.basis, omega;
        vias=[vvia], vols=[vv], mx=12, my=10, block=5)
    @test maximum(abs.(Z .- Zb)) / maximum(abs.(Z)) < 1e-12

    # PMC sidewalls: same kernel path with swapped modal parity
    gridp = CellGrid(8e-3, 6e-3, 8, 6; walls=WALL_PMC)
    svp = sheet_level(1, 8, 6)
    rasterize_rect!(svp, gridp, 0.0, 5e-3, 1e-3, 5e-3; connected=true)
    vvp = vol_level(2, 8, 6)
    rasterize_rect!(vvp, gridp, 1e-3, 6e-3, 1e-3, 5e-3)
    probvp = build_planar_problem(stack, gridp, [svp],
        PlanarPort[PlanarPort(1, :west, 3:4, 50.0)]; vols=[vvp])
    Zp = assemble_planar_z(stack, gridp, [svp], probvp.basis, omega;
        vols=[vvp], mx=12, my=10)
    @test all(isfinite, Zp)
    @test Zp ≈ transpose(Zp) rtol = 1e-10

    # --- vol <-> sheet / via pair reciprocity at kernel level ---
    for n in (2, 5), m in (2, 6)
        kx, ky = mg.kx[m], mg.ky[n]
        kc2 = kx^2 + ky^2
        ntm2 = kx^2 * mg.ic[m] * mg.js[n] + ky^2 * mg.is[m] * mg.jc[n]
        n3_2 = mg.is[m] * mg.js[n]
        casc = planar_mode_cascade(stack, omega, kc2, TM_POL)
        vsts = [_PG._via_layer_state(stack, casc, omega, kc2, j)
                for j in 1:L]
        vsts_tm = [_PG._vol_layer_state(stack, casc, omega, kc2, j,
            TM_POL) for j in 1:L]
        for jv in 1:L, js in 1:L
            ev = _PG._vol_elem(js)
            # uniform via sees Vt - Vb; taper sees Vt - <V> (same layer)
            for kind in (_PG._BASIS_VIA_U, _PG._BASIS_VIA_T)
                evia = _PG._via_elem(jv, kind)
                got = _PG._elem_pair_pol(evia, ev, casc, vsts,
                    vsts_tm, ntm2, n3_2)
                got2 = _PG._elem_pair_pol(ev, evia, casc, vsts,
                    vsts_tm, ntm2, n3_2)
                @test got ≈ got2
                if jv == js
                    stv = vsts_tm[js]
                    stvia = vsts[jv]
                    m = kind == _PG._BASIS_VIA_U ? stv.vt - stv.vb :
                        stv.vt - stv.mself
                    @test got ≈ stvia.coef * m / ntm2
                end
            end
            # vol <-> vol symmetry
            got = _PG._elem_pair_pol(_PG._vol_elem(jv), ev, casc,
                vsts, vsts_tm, ntm2, n3_2)
            got2 = _PG._elem_pair_pol(ev, _PG._vol_elem(jv), casc,
                vsts, vsts_tm, ntm2, n3_2)
            @test got ≈ got2
            jv == js && @test got ≈ vsts_tm[jv].mself / ntm2
            # vol <-> sheet reciprocity vs direct telescoping
            s = 1
            gotv = _PG._elem_pair_pol(_PG._vol_elem(jv), s, casc,
                vsts, vsts_tm, ntm2, n3_2)
            gots = _PG._elem_pair_pol(s, _PG._vol_elem(jv), casc,
                vsts, vsts_tm, ntm2, n3_2)
            @test gotv ≈ gots
            refv = vsts_tm[jv].wbar *
                (planar_modal_voltage(casc, jv, s) +
                 planar_modal_voltage(casc, jv - 1, s)) / ntm2
            @test gotv ≈ refv
        end
    end

    # --- TE kernel for volume elements: same formulas on the TE
    #     cascade; via-involving pairs are identically zero ---
    for n in (2, 4), m in (2, 5)
        kx, ky = mg.kx[m], mg.ky[n]
        kc2 = kx^2 + ky^2
        nte2 = ky^2 * mg.ic[m] * mg.js[n] + kx^2 * mg.is[m] * mg.jc[n]
        nte2 == 0 && continue
        casc = planar_mode_cascade(stack, omega, kc2, TE_POL)
        vsts_te = [_PG._vol_layer_state(stack, casc, omega, kc2, j,
            TE_POL) for j in 1:L]
        j1, j2 = 2, 3   # j1 < j2: field faces take the source's vb
        st = _PG._vol_layer_state(stack, casc, omega, kc2, j1, TE_POL)
        stb = vsts_te[j2]
        denom = planar_modal_voltage(casc, j2 - 1, j2 - 1)
        vt = stb.vb * planar_modal_voltage(casc, j1, j2 - 1) / denom
        vb = stb.vb * planar_modal_voltage(casc, j1 - 1, j2 - 1) / denom
        ref = st.wbar * (vt + vb) / nte2
        got = _PG._elem_pair_pol(_PG._vol_elem(j1), _PG._vol_elem(j2),
            casc, _PG._ViaLayerState{ComplexF64}[], vsts_te, nte2, 0.0)
        @test got ≈ ref
    end

    # --- validation boundaries ---
    @test_throws ArgumentError vol_level(0, 8, 6)
    badlay = VolLevel(L + 1, falses(8, 6),
        falses(6), falses(6), falses(8), falses(8))
    badlay.mask[1, 1] = true
    @test_throws ArgumentError build_planar_problem(stack, grid, [sv],
        ports_v; vols=[badlay])
    badsz = VolLevel(1, falses(4, 4),
        falses(4), falses(4), falses(4), falses(4))
    @test_throws DimensionMismatch build_planar_basis(grid, [sv],
        PlanarPort[]; vols=[badsz])
    vone = vol_level(1, 8, 6)
    rasterize_rect!(vone, grid, 1e-3, 3e-3, 1e-3, 3e-3)
    bv2 = build_planar_basis(grid, [sv], PlanarPort[]; vols=[vone])
    @test_throws ArgumentError assemble_planar_z(stack, grid, [sv],
        bv2, omega; vols=VolLevel[], mx=8, my=8)

    # --- deeply evanescent mode: all state values finite ---
    casc_ev = planar_mode_cascade(stack, omega, 1e8, TM_POL)
    for j in 1:L
        st = _PG._vol_layer_state(stack, casc_ev, omega, 1e8, j, TM_POL)
        @test isfinite(st.wbar) && isfinite(st.vt) &&
              isfinite(st.vb) && isfinite(st.mself)
    end

    # --- end to end: thick strip with a co-located port sheet ---
    sv2 = sheet_level(1, 8, 6)
    rasterize_rect!(sv2, grid, 0.0, 5e-3, 1e-3, 5e-3; connected=true)
    vv2 = vol_level(1, 8, 6)
    rasterize_rect!(vv2, grid, 0.0, 5e-3, 1e-3, 5e-3; connected=true)
    ports2 = PlanarPort[PlanarPort(1, :west, 2:5, 50.0)]
    prob2 = build_planar_problem(stack, grid, [sv2], ports2;
        vols=[vv2])
    r2 = solve_planar(prob2, 8e9; mx=16, my=12)
    @test all(isfinite, r2.s)
    r2b = solve_planar(build_planar_problem(stack, grid, [sv2],
        ports2), 8e9; mx=16, my=12)
    # the volume bases change the answer (physics: thick conductor)
    @test abs(r2.y[1, 1] - r2b.y[1, 1]) > 1e-4 * abs(r2b.y[1, 1])

    # --- gradient through the volume kernels vs finite difference ---
    s_g = sheet_level(1, 8, 6)
    rasterize_rect!(s_g, grid, 0.0, 5e-3, 1e-3, 5e-3; connected=true)
    v_g = vol_level(2, 8, 6)
    rasterize_rect!(v_g, grid, 0.0, 5e-3, 1e-3, 5e-3; connected=true)
    ports_g = PlanarPort[PlanarPort(1, :west, 2:5, 50.0)]
    probg = build_planar_problem(stack, grid, [s_g], ports_g;
        vols=[v_g])
    fg = Y -> abs2(Y[1, 1])
    params = PlanarParam[PlanarParam(2, :epsr, :re),
        PlanarParam(2, :thickness, :re)]
    theta = planar_param_values(stack, params)
    J, g = planar_objective_gradient(probg, 8e9, fg; params=params)
    @test J ≈ fg(solve_planar(probg, 8e9).y)
    for j in eachindex(params)
        hs = 1e-6 * max(abs(theta[j]), 1.0)
        tp = copy(theta); tm = copy(theta)
        tp[j] += hs; tm[j] -= hs
        Jp = fg(solve_planar(PlanarProblem(
            planar_with_params(stack, params, tp), probg.grid,
            probg.sheets, probg.ports, probg.vias, probg.basis,
            probg.vols), 8e9).y)
        Jm = fg(solve_planar(PlanarProblem(
            planar_with_params(stack, params, tm), probg.grid,
            probg.sheets, probg.ports, probg.vias, probg.basis,
            probg.vols), 8e9).y)
        @test g[j] ≈ (Jp - Jm) / (2hs) rtol = 1e-4
    end
end

@testset "planar: adaptive sweep and box resonances" begin
    # Thiele rational interpolant: a model through N samples of a
    # rational function R(f) = num(f)/den(f) of degree <= (N-2)/2
    # reproduces it to roundoff at off-grid points.
    R = f -> (2.0 - 0.5im + (1.0 + 0.2im) * f) / (1.0 - (0.3 - 0.1im) * f)
    xs = collect(range(0.0, 4.0; length=5))
    cf = _PG._thiele_build(xs, R.(xs))
    for f in (0.3, 1.7, 2.9, 3.99)
        @test _PG._thiele_eval(cf, f) ≈ R(f) rtol = 1e-12
    end
    # two samples -> linear interpolant
    c2 = _PG._thiele_build([0.0, 1.0], [1.0 + 2im, 3.0 - 1im])
    @test _PG._thiele_eval(c2, 0.4) ≈ (1 + 2im) + 0.4 * ((3 - 1im) - (1 + 2im))
    # degenerate ordinates truncate and flag the model
    cd = _PG._thiele_build([0.0, 1.0, 2.0], [1.0, 1.0, 2.0])
    @test cd.ndeg < 3
    @test_throws ArgumentError _PG._thiele_build(Float64[], ComplexF64[])

    # ABS sweep on the stripline benchmark problem: interpolated S must
    # reproduce direct solves away from the analysis points.
    prob = _stripline_problem(16, 20)
    sw = planar_sweep_abs(prob, 1e9, 8e9; n_eval=33, rel_tol=1e-3,
                          max_points=20, mx=64, my=80)
    @test sw isa PlanarSweep
    @test length(sw.freqs) >= 3
    @test length(sw.s) == length(sw.freqs)
    @test length(sw.dense_s) == length(sw.dense_freqs) == 33
    @test length(sw.est_err) == 33
    # every dense S-matrix is fully populated and finite
    for S in sw.dense_s
        @test size(S) == (2, 2)
        @test all(isfinite, S)
    end
    # analyzed frequencies are reproduced by direct solves
    @test all(sw.s[j] ≈ solve_planar(prob, sw.freqs[j]; mx=64, my=80).s
              for j in eachindex(sw.freqs))
    if sw.converged
        # interpolated S tracks a direct solve at an off-analysis point
        jw = argmax(sw.est_err)
        fcheck = sw.dense_freqs[jw]
        if !any(fk -> abs(fcheck - fk) < 0.5 * 7e9 / 33, sw.freqs)
            @test sw.dense_s[jw] ≈
                solve_planar(prob, fcheck; mx=64, my=80).s rtol = 2e-2
        end
        @test all(e <= 1e-3 for (j, e) in enumerate(sw.est_err)
                  if !any(fk -> abs(sw.dense_freqs[j] - fk) <=
                          0.5 * 7e9 / 33, sw.freqs))
    end
    # capped sweep reports non-convergence instead of a trusted result
    sw3 = planar_sweep_abs(prob, 1e9, 8e9; n_eval=33, rel_tol=1e-9,
                           max_points=3, mx=64, my=80)
    @test !sw3.converged
    @test all(all(isfinite, S) for S in sw3.dense_s)
    @test_throws ArgumentError planar_sweep_abs(prob, 8e9, 1e9)
    @test_throws ArgumentError planar_sweep_abs(prob, 1e9, 8e9; rel_tol=0.0)
    @test_throws ArgumentError planar_sweep_abs(prob, 1e9, 8e9; n_eval=4)
    @test_throws ArgumentError planar_sweep_abs(prob, 1e9, 8e9; max_points=2)

    # box resonances: PEC-PEC air cavity, TM_mn0 poles at the transverse
    # eigenfrequencies c/(2)*sqrt((m/a)^2 + (n/b)^2)
    c0 = _PG._C0
    a, b, h = 15e-3, 10e-3, 1.0e-3
    cav = PlanarStackup([PlanarLayer(1.0, 1.0, h)],
                        TERM_GND, TERM_GND, a, b)
    res = planar_box_resonances(cav, 10e9, 40e9;
                                walls=WALL_PEC, mmax=4, nmax=4)
    @test all(r -> r isa PlanarResonance, res)
    for (m, n) in ((1, 1), (2, 1), (1, 2), (3, 1))
        fana = c0 / 2 * sqrt((m / a)^2 + (n / b)^2)
        10e9 < fana < 40e9 || continue
        hits = findall(r -> r.m == m && r.n == n && r.pol == TM_POL, res)
        @test !isempty(hits)
        @test minimum(abs(res[k].freq - fana) / fana for k in hits) < 2e-3
    end
    # no spurious TE poles in the low band of a thin cavity
    @test all(r -> r.pol == TM_POL || r.freq > 30e9, res)
    # sorted, deduplicated
    @test issorted([r.freq for r in res])
    # sidewall parity: PEC TM modes need m,n >= 1, so no (m,0) TM pole;
    # PMC walls allow the cos-parity TM_(1,0) mode at c/(2a) = 10 GHz
    @test all(r -> r.pol != TM_POL || (r.m >= 1 && r.n >= 1), res)
    res_pmc = planar_box_resonances(cav, 5e9, 40e9;
                                    walls=WALL_PMC, mmax=4, nmax=4)
    @test any(r -> r.pol == TM_POL && r.m == 1 && r.n == 0 &&
                   abs(r.freq - c0 / 2 / a) / (c0 / 2 / a) < 2e-3,
              res_pmc)
    @test_throws ArgumentError planar_box_resonances(cav, 40e9, 10e9)
    @test_throws ArgumentError planar_box_resonances(cav, 10e9, 40e9;
                                                     nsamp=4)
end
