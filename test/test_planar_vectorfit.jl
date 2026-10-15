using Test, LinearAlgebra, DiffMoM

include(joinpath(@__DIR__, "helpers", "workflow_spice_y.jl"))

@testset "planar vector fit: real stable N-port and SPICE synthesis" begin
    fs = collect(range(1e8,1e10;length=51))
    D = [0.02 -0.005;-0.005 0.03]
    R = [1e8 -2e7;-2e7 8e7]
    E = [1e-12 -2e-13;-2e-13 2e-12]
    Ys = [D+2pi*1im*f*E+R/(2pi*1im*f+2e9) for f in fs]
    model = planar_fit_rational(Ys,fs;order=1)
    @test model.relative_rms_error < 1e-10
    @test model.poles[1] ≈ -2e9 rtol=1e-8
    @test model.d ≈ D rtol=1e-9
    @test model.e ≈ E rtol=1e-9
    @test model.sampled_passive
    @test model.globally_passive
    @test planar_rational_certificate(model).certified
    @test planar_rational_eval(model,-2e9) ≈ conj.(planar_rational_eval(model,2e9))
    mktempdir() do dir
        path = planar_write_spice(model,joinpath(dir,"model.cir"))
        for f in (0.0,1e8,2e9,1e10,3e10)
            @test _workflow_spice_y(path,f,2) ≈ planar_rational_eval(model,f) rtol=1e-9
            @test _workflow_spice_y(path,f,2;common_mode=1+.7im) ≈ planar_rational_eval(model,f) rtol=1e-9
            @test maximum(abs,_workflow_spice_y(path,f,2;common_mode=1+.7im,drive_voltage=0.))<1e-12
        end
    end
    # Complex conjugate poles/residues must realize a real time response.
    p = -1e9+3e9im
    residue = ComplexF64[1e7+2e6im -2e6+1e6im;-2e6+1e6im 8e6+1e6im]
    mixed = [D+residue/(2pi*1im*f-p)+conj.(residue)/(2pi*1im*f-conj(p)) for f in fs]
    pair = planar_fit_rational(mixed,fs;order=2)
    @test pair.relative_rms_error < 1e-8
    @test all(real.(pair.poles) .< 0)
    @test pair.poles[2] ≈ conj(pair.poles[1])
    mktempdir() do dir
        path = planar_write_spice(pair,joinpath(dir,"pair.cir"))
        for f in (0.0,1e8,2e9,1e10)
            @test _workflow_spice_y(path,f,2) ≈ planar_rational_eval(pair,f) rtol=1e-9
            @test _workflow_spice_y(path,f,2;common_mode=1+.7im) ≈ planar_rational_eval(pair,f) rtol=1e-9
            @test maximum(abs,_workflow_spice_y(path,f,2;common_mode=1+.7im,drive_voltage=0.))<1e-12
        end
    end
    # Unequal-reference S fitting shares the established power-wave basis.
    Ss = [planar_y_to_s(Y,[50,75]) for Y in Ys]
    smodel = planar_fit_rational(Ss,fs;order=1,format=:s,z0=[50,75])
    @test smodel.relative_rms_error < 1e-9
    @test planar_rational_eval(smodel,2e9) ≈ planar_rational_eval(model,2e9) rtol=1e-8
    # Repair records its conductance cost and recomputes the fit error.
    active = [fill(-0.01,1,1) for _ in fs]
    repaired = planar_fit_rational(active,fs;order=1)
    @test repaired.sampled_passive
    @test repaired.globally_passive
    @test repaired.passivity_shift > 0.009
    @test repaired.relative_rms_error > 0.9
    raw = planar_fit_rational(active,fs;order=1,enforce_passivity=false)
    @test !raw.sampled_passive
    @test raw.relative_rms_error < 1e-10
    @test_throws ArgumentError planar_fit_rational(Ys,fs;max_bytes=1)
    @test_throws ArgumentError planar_fit_rational(Ys,fill(1e9,length(fs)))
    @test_throws ArgumentError planar_write_spice(model,"unused.cir";subckt_name="bad name")
    mktempdir() do directory
        path=joinpath(directory,"sentinel.cir");write(path,"retain original file")
        @test_throws ArgumentError planar_write_spice(model,path;port_names=["input","INPUT"])
        @test read(path,String)=="retain original file"
        @test_throws ArgumentError planar_write_spice(model,path;port_names=["port","PoRt"])
        @test read(path,String)=="retain original file"
        @test_throws ArgumentError planar_write_spice(model,path;port_names=["gNd","port"])
        @test read(path,String)=="retain original file"
        @test_throws ArgumentError planar_write_spice(model,path;port_names=["DM_State","port"])
        @test read(path,String)=="retain original file"
        planar_write_spice(model,path;port_names=["Input","Output"])
        @test occursin(".subckt diffmom_nport Input Output dm_ref",read(path,String))
    end
end

@testset "planar vector fit: global positive-real certification" begin
    function planted(poles,residues,d,e)
        return PlanarRationalModel(poles,residues,d,e,[0.1,10.0],
            NaN,NaN,false,false,:none,NaN,0.0,0.0)
    end
    # A narrow active band falls between all three sampling frequencies.
    poles = ComplexF64[-0.001+10im,-0.001-10im]
    residues = [fill(-0.01+0im,1,1),fill(-0.01+0im,1,1)]
    narrow = planted(poles,residues,fill(1.0,1,1),zeros(1,1))
    @test planar_rational_passivity(narrow,[0.0,1.0,2.0]).passive
    certificate = planar_rational_certificate(narrow)
    @test !certificate.certified
    @test certificate.method == :hamiltonian
    @test length(certificate.crossings_hz) >= 2
    @test !planar_rational_passivity(narrow,[10/(2pi)]).passive
    # Stable pole locations alone do not make a negative capacitor passive.
    negative_c = planted(ComplexF64[-1], [fill(0.0+0im,1,1)],
        fill(1.0,1,1),fill(-1.0,1,1))
    @test planar_rational_passivity(negative_c,[0.0,1.0,2.0]).passive
    @test !planar_rational_certificate(negative_c).certified
    # Tiny negative energy and affine skew cannot be hidden by a relative
    # eigenvalue/symmetry tolerance against a large positive eigenvalue.
    tiny_negative = planted(ComplexF64[-1],[zeros(ComplexF64,2,2)],zeros(2,2),
        [-1e-12 0.0;0.0 1e-3])
    @test !planar_rational_certificate(tiny_negative).certified
    skew_cap = planted(ComplexF64[-1],[zeros(ComplexF64,2,2)],Matrix{Float64}(I,2,2),
        [1e-3 1e-15;-1e-15 1e-3])
    @test !planar_rational_certificate(skew_cap).certified
    # Semidefinite feedthrough has a direct residue certificate.
    passive = planted(ComplexF64[-2],[ComplexF64[1 -1;-1 1]],zeros(2,2),
        [1.0 -1.0;-1.0 1.0])
    positive = planar_rational_certificate(passive)
    @test positive.certified
    @test positive.method == :positive_residues
end
