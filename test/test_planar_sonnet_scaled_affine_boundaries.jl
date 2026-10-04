module ScaledSonnetAffineBoundaryTests
using Test,DiffMoM,SHA,TOML,LinearAlgebra
const fixture=joinpath(@__DIR__,"fixtures/scaled_geovar_affine_boundaries")
module LegacyAffineScaling
include("fixtures/scaled_geovar_affine_boundaries/helper_before.jl")
end

decimal_center(values)=DiffMoM._sonnet_geovar_scaled_coordinate(values[1],values[2],values[3],.2,.3,true)

@testset "Independent affine oracle for scaled geometry range failures" begin
    hashes=TOML.parsefile(joinpath(fixture,"sha256.toml"))["sha256"]
    actual=Set(replace(relpath(joinpath(dir,name),fixture),'\\'=>'/')
        for (dir,_,names) in walkdir(fixture) for name in names if name!="sha256.toml")
    @test Set(keys(hashes))==actual
    for (path,digest) in hashes
        @test bytes2hex(sha256(read(joinpath(fixture,path))))==digest
    end
    proof=TOML.parsefile(joinpath(fixture,"scaled_affine_failure_probe_20261004.toml"))
    @test length(proof["cases"])==4
    for row in proof["cases"]
        values=(row["value"],row["first"],row["second"],row["nominal"],row["target"],row["symmetric"])
        expected=setprecision(BigFloat,4608) do
            value,first,second,nominal,target,symmetric=values
            anchor=symmetric ? (BigFloat(first)+BigFloat(second))/2 : BigFloat(first)
            Float64(anchor+(BigFloat(value)-anchor)*BigFloat(target)/BigFloat(nominal))
        end
        @test expected==row["affine_oracle"]
        @test LegacyAffineScaling._sonnet_geovar_scaled_coordinate(values...)==row["observed_before"]
        @test row["observed_before"]!=expected
        @test DiffMoM._sonnet_geovar_scaled_coordinate(values...)≈expected rtol=8eps(Float64) atol=0
    end
    fixed=DiffMoM._sonnet_geovar_fixed_coordinate
    tiny=nextfloat(0.);large=floatmax(Float64)
    @test !fixed(1.,1.,nextfloat(1.),true)
    @test fixed(3tiny,3tiny,3tiny,true)
    @test fixed(2tiny,tiny,3tiny,true)
    @test fixed(0.,-large,large,true)
    @test !fixed(1.,-large,large,true)
    @test fixed(1.,1.,nextfloat(1.),false)
    # A rounded ratio can erase a tiny, representable result at a rounded
    # midpoint. Returning every rounded midpoint immediately would lose it.
    values=(1.,2.0^-54,2.,2.0^-55,1.,true)
    expected=setprecision(BigFloat,4608) do
        value,first,second,nominal,target,symmetric=values
        anchor=(BigFloat(first)+BigFloat(second))/2
        Float64(anchor+(BigFloat(value)-anchor)*BigFloat(target)/BigFloat(nominal))
    end
    @test expected==2.0^-55
    @test LegacyAffineScaling._sonnet_geovar_scaled_coordinate(values...)==1.
    @test DiffMoM._sonnet_geovar_scaled_coordinate(values...)==expected
    decimal=[.5*.001,.4*.001,.6*.001]
    decimal_center(decimal)
    @test (@allocated decimal_center(decimal))==0
end

@testset "Actual native decimal midpoint compatibility" begin
    native=joinpath(fixture,"decimal_native")
    proof=TOML.parsefile(joinpath(native,"comparison.toml"))
    @test proof["source_unchanged"] && proof["source_before"]==proof["source_after"]
    @test length(proof["cases"])==2
    for row in proof["cases"]
        @test row["reader_before"]=="REJECTED"
        @test row["native_status"]=="PASS" && row["bit_identical_s"] && row["native_full_s_error"]==0
        @test row["literal_public_full_s_error"]<=.005 && row["literal_original_voltage_residual"]<=1e-9
        tag=row["case"]
        p=read_sonnet_project(joinpath(native,tag*"_parameter.son"))
        q=read_sonnet_project(joinpath(native,tag*"_literal.son"))
        saved=deepcopy(p.polygons);savedports=deepcopy(p.ports)
        result=DiffMoM._sonnet_geometry_project(p,1e9)
        @test only(result.polygons).vertices≈only(q.polygons).vertices rtol=8eps(Float64)
        @test all(a.vertices==b.vertices for (a,b) in zip(p.polygons,saved))
        @test all(a.values==b.values for (a,b) in zip(p.ports,savedports))
        @test only(sonnet_planar_problem(p;freq=1e9).sheets).mask==only(sonnet_planar_problem(q;freq=1e9).sheets).mask
        reference=planar_read_touchstone(joinpath(native,tag*"_parameter/native_raw.s2p"))
        @test reference.s==planar_read_touchstone(joinpath(native,tag*"_literal/native_raw.s2p")).s
        solved=solve_sonnet_project(p,1e9;raw=true,method=:dense_fft,mx=128,my=128,retain_matrix=true)
        @test maximum(abs,solved.s-only(reference.s))<=.005
        raw=solved.raw;rhs=zeros(ComplexF64,size(raw.currents))
        for b in eachindex(raw.problem.basis.port)
            port=raw.problem.basis.port[b];iszero(port) && continue
            rhs[b,port]=(port==1 ? -1. : 1.)*raw.problem.basis.width[b]
        end
        @test norm(raw.z_mom*raw.currents-rhs)/norm(rhs)<=1e-9
    end
end
end
