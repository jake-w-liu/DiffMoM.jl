using DiffMoM, Test

function _solver_audit_conformal_fixture(walls)
    a, b = .003, .002
    stack = PlanarStackup([PlanarLayer(1., 1., .0005),
        PlanarLayer(1., 1., .0005)], TERM_GND, TERM_GND, a, b)
    mesh = PlanarConformalMesh([0. a a 0.; 0. 0. b b], [1 1; 2 3; 3 4])
    PlanarConformalProblem(stack, mesh, [PlanarConformalPort(1, :west, (0., b)),
        PlanarConformalPort(1, :east, (0., b))]; sidewalls=walls)
end

function _solver_audit_conformal_weight_allocations(prob, te, tm, kx, ky)
    DiffMoM._planar_conformal_weights!(te, tm, prob, kx, ky)
    @allocated DiffMoM._planar_conformal_weights!(te, tm, prob, kx, ky)
end

@testset "conformal modal weights: reuse buffers without edge slices" begin
    for walls in (WALL_PEC, WALL_PMC)
        prob = _solver_audit_conformal_fixture(walls)
        te = zeros(length(prob.basis.width)); tm = similar(te)
        for (m, n) in ((0, 1), (1, 0), (2, 3), (97, 89))
            kx, ky = m*pi/prob.stack.a, n*pi/prob.stack.b
            @test _solver_audit_conformal_weight_allocations(prob, te, tm, kx, ky) == 0
            @test all(isfinite, te) && all(isfinite, tm)
        end
    end
end

function _solver_audit_cached_weight_allocations(prob,te,tm,kx,ky,cache)
    DiffMoM._planar_conformal_cached_weights!(te,tm,prob,kx,ky,cache)
    @allocated DiffMoM._planar_conformal_cached_weights!(te,tm,prob,kx,ky,cache)
end

@testset "conformal triangle phase cache preserves scalar weights" begin
    for walls in (WALL_PEC,WALL_PMC)
        original=_solver_audit_conformal_fixture(walls)
        mesh=original.mesh
        multiple=PlanarConformalMesh(mesh.vertices,hcat(mesh.triangles,mesh.triangles);
            interfaces=[mesh.interfaces;fill(2,length(mesh.interfaces))])
        for prob in (original,PlanarConformalProblem(original.stack,multiple,original.ports;sidewalls=walls))
            nb=length(prob.basis.width);te=zeros(nb);tm=similar(te);scalar_te=similar(te);scalar_tm=similar(te)
            cache=DiffMoM._planar_conformal_triangle_cache(prob)
            @test sizeof(cache.factors)+sizeof(cache.weights)==DiffMoM._planar_conformal_triangle_cache_bytes(prob)
            for (m,n) in ((0,0),(0,1),(1,0),(2,3),(97,89))
                kx,ky=m*pi/prob.stack.a,n*pi/prob.stack.b
                DiffMoM._planar_conformal_weights!(scalar_te,scalar_tm,prob,kx,ky)
                @test _solver_audit_cached_weight_allocations(prob,te,tm,kx,ky,cache)==0
                @test reinterpret(UInt64,te)==reinterpret(UInt64,scalar_te)
                @test reinterpret(UInt64,tm)==reinterpret(UInt64,scalar_tm)
            end
        end
    end
end

function _solver_audit_conformal_assembly_budget(prob,mx,my)
    nb=length(prob.basis.width);layers=length(prob.stack.layers)
    16nb^2+56(mx+my)+16nb+240(layers+1)+192length(prob.mesh.interfaces)+16nb+24nb
end

@testset "conformal modal batches preserve tight budgets and reactions" begin
    for walls in (WALL_PEC,WALL_PMC)
        original=_solver_audit_conformal_fixture(walls);mesh=original.mesh
        multiple=PlanarConformalMesh(mesh.vertices,hcat(mesh.triangles,mesh.triangles);
            interfaces=[mesh.interfaces;fill(2,length(mesh.interfaces))])
        for prob in (original,PlanarConformalProblem(original.stack,multiple,original.ports;sidewalls=walls))
            mx,my=5,7;budget=_solver_audit_conformal_assembly_budget(prob,mx,my)
            zs=[.2+.01im*t for t in eachindex(prob.mesh.interfaces)]
            reference=assemble_planar_conformal_z(prob,3e9;mx,my,surface_zs=zs,max_bytes=budget)
            @test_throws ArgumentError assemble_planar_conformal_z(prob,3e9;mx,my,max_bytes=budget-1)
            levels=unique(prob.basis.interfaces)
            groups=[(level,findfirst(==(level),prob.basis.interfaces):findlast(==(level),prob.basis.interfaces)) for level in levels]
            columnbytes=16(2length(prob.basis.width)+length(groups)^2)
            # The threshold below two columns retains scalar accumulation.
            below=assemble_planar_conformal_z(prob,3e9;mx,my,surface_zs=zs,max_bytes=budget+2columnbytes-1)
            @test reinterpret(UInt64,vec(below))==reinterpret(UInt64,vec(reference))
            for columns in (2,3,7,64)
                available=columnbytes*columns
                batch=DiffMoM._planar_conformal_modal_batch(prob,groups,available)
                @test sizeof(batch.weights)+sizeof(batch.voltages)+sizeof(batch.scaled)<=available
                actual=assemble_planar_conformal_z(prob,3e9;mx,my,surface_zs=zs,max_bytes=budget+available)
                @test actual≈reference rtol=2e-14
                @test actual≈transpose(actual) rtol=2e-14
                @test all(isfinite,actual)
            end
        end
    end
end
