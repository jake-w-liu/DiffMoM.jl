module LibraryRemainingProofTests
using DiffMoM,Test,LinearAlgebra,SHA,TOML
const directory=joinpath(@__DIR__,"fixtures","library_remaining_domain")
const before_hash="b69db254f721e9c0b57637b36849f1184619407859439ab559977f6f914cdab3"
const after_hash="54d3328bb16cf70bb4c4597bebc7e9172e52ea7ad3619dbcd2ee8196c747ae53"
const sha_file=file->bytes2hex(sha256(read(joinpath(directory,file))))

@testset "Remaining Library proof integrity and source correspondence" begin
    hashes=TOML.parsefile(joinpath(directory,"sha256.toml"))["sha256"]
    @test length(hashes)==29
    @test Set(keys(hashes))==Set(filter(!=("sha256.toml"),readdir(directory)))
    for (file,digest) in hashes
        @test sha_file(file)==digest
    end
    metadata=TOML.parsefile(joinpath(directory,"proofs.toml"))
    @test metadata["before_sha256"]==sha_file(metadata["before_source"])==before_hash
    @test metadata["after_sha256"]==sha_file(metadata["after_source"])==after_hash
    @test read(joinpath(directory,"source_before.jl"))==
        read(joinpath(@__DIR__,"fixtures","library_stored_domain","source_after.jl"))
    @test metadata["current_public_assertions"]==1109
    @test metadata["current_source_unchanged_assertions"]==1
    @test length(metadata["reports"])==6
    @test count(r->r["source_snapshot_present"],metadata["reports"])==4
    for artifact in metadata["reports"]
        report=TOML.parsefile(joinpath(directory,artifact["file"]))
        if artifact["source_snapshot_present"]
            @test !isempty(artifact["source"])
            @test report["source_sha256"]==sha_file(artifact["source"])
        else
            @test isempty(artifact["source"])
            @test report["source_sha256"]==artifact["source_sha256"]
            @test report["source_sha256"]!=before_hash && report["source_sha256"]!=after_hash
        end
    end
    before=TOML.parsefile(joinpath(directory,"library_remaining_domain_before.toml"))
    @test length(before["rows"])==8
    @test count(r->r["case"]=="nonzero_broadside_offset_stored_zero",before["rows"])==2
    @test all(r->r["status"]=="accepted",before["rows"])
    pins=TOML.parsefile(joinpath(directory,"library_pin_geometry_archived_before.toml"))["rows"]
    @test length(pins)==28
    @test any(r->r["direction_tangent_component"]>.05,pins)
    @test any(r->r["relative_midpoint_error"]>.05,pins)
    cap=TOML.parsefile(joinpath(directory,"library_air_bridge_capacitance_archived_before.toml"))["rows"]
    @test length(cap)==3
    @test all(r->isfinite(r["independent_expected"]) && r["independent_expected"]>0,cap)
    final=TOML.parsefile(joinpath(directory,"library_current_independent_review.toml"))["rows"]
    @test length(final)==140
    @test all(r->isfinite(r["actual"]) && r["actual"]>0 &&
        isapprox(r["actual"],r["expected"];rtol=3e-15,atol=nextfloat(0.)),final)
    midpoint=TOML.parsefile(joinpath(directory,"library_pin_midpoint_subnormal_before.toml"))["rows"]
    @test length(midpoint)==2
    @test all(r->r["offset"]!=0 && r["offset"]==r["edge_x"]==r["exact_midpoint_x"] &&
        r["stored_midpoint_x"]==0,midpoint)
    @test metadata["reports"][5]["source_snapshot_present"]==false
    @test metadata["reports"][5]["source_sha256"]==
        "be566fa5838091fad9a009ecab6a31924131352abfb70cb1bc216436be31d96b"
    @test Set(d["status"] for d in metadata["dispositions"])==
        Set(["harness_race","harness_missing_import","frequency_keyword_candidate_not_classified","norm_candidate_unconfirmed"])
    @test occursin("Some tests did not pass",read(joinpath(directory,"library_pin_geometry_before.log"),String))
    @test occursin("UndefVarError",read(joinpath(directory,"library_pin_midpoint_subnormal_probe.log"),String))
    @test occursin("not classified",read(joinpath(directory,"README.md"),String))
end

# Byte verification precedes including a historical source; no intermediate
# source is reconstructed from patches or attributed to an absent snapshot.
sha_file("source_before.jl")==before_hash || error("Historical Library source bytes differ")
module ArchivedLibrary
using DiffMoM
import DiffMoM: _circuit_stored_real,_P2,_p2_seg_len,_p2_signed_area,
    _planar_regular_ngon,_EPS0,_MU0
include(joinpath(@__DIR__,"fixtures","library_remaining_domain","source_before.jl"))
end

@testset "Captured original broadside and stale pin failures" begin
    for value in (big"1e-1000",-big"1e-1000")
        shape=ArchivedLibrary.planar_broadside_coupled_lines(length=.001,width=.0001,
            upper_level=2,lower_level=1,offset=value,metal="pec")
        @test value!=0
        @test shape.meta["offset"]==0.
        @test shape.polygons[1].vertices==shape.polygons[2].vertices
    end
    original=ArchivedLibrary.planar_line(length=.001,width=.0001,level=1,metal="pec")
    for offset in (1e9,1e10,1e11)
        placed=ArchivedLibrary.planar_transform(original;offset=(0.,offset))
        vertices=only(placed.polygons).vertices
        for pin in placed.pins
            a=vertices[pin.edge];b=vertices[mod1(pin.edge+1,length(vertices))]
            width=hypot((b-a)...)
            @test isfinite(width) && width>0
            @test pin.width!=width
            @test abs(pin.width-width)/pin.width>=.001
        end
    end
    offedge=false;nonnormal=false
    for angle in (.1,.3,pi/4,1.),offset in ((0.,1e9),(0.,1e10),(0.,1e11),(1e11,0.),(1e11,1e11))
        placed=ArchivedLibrary.planar_transform(original;angle,offset)
        vertices=only(placed.polygons).vertices
        for pin in placed.pins
            a=vertices[pin.edge];b=vertices[mod1(pin.edge+1,length(vertices))]
            width=hypot((b-a)...)
            nonnormal|=abs(dot(pin.direction,(b-a)/width))>.05
            offedge|=hypot((pin.point-(a+(b-a)/2))...)/width>.05
        end
    end
    @test nonnormal
    @test offedge
end

@testset "Captured original collapsed via and capacitance failures" begin
    for span in (.001,1.)
        shape=ArchivedLibrary.planar_air_bridge(;span,width=.0001,landing=.0001,
            bridge_level=2,base_level=1,via_type="v",metal="pec",via_margin=prevfloat(.00005))
        for via in shape.vias
            @test length(unique(via.vertices))==2
            @test_throws ArgumentError planar_normalize_polygon(via.vertices)
        end
    end
    for (height,epsr,area) in ((1e-300,1e100,1e-300),(1e300,1e-100,1e300),(1e-300,1e100,1e-200))
        stack=PlanarStackup([PlanarLayer(epsr,1.,height)],TERM_GND,TERM_GND,.001,.001)
        expected=setprecision(BigFloat,4096) do
            Float64(BigFloat(DiffMoM._EPS0)*BigFloat(area)*BigFloat(epsr)/BigFloat(height))
        end
        @test isfinite(expected)&&expected>0
        @test_throws ArgumentError ArchivedLibrary.planar_parallel_plate_capacitance(stack,1,0,area)
    end
end

@testset "Corrected broadside nonzero storage domain" begin
    for value in (big"1e-1000",-big"1e-1000",big"1e1000",-big"1e1000")
        @test_throws ArgumentError planar_broadside_coupled_lines(length=.001,width=.0001,
            upper_level=2,lower_level=1,offset=value,metal="pec")
    end
    for value in (BigFloat(0),BigFloat(.0002),BigFloat(nextfloat(0.)))
        shape=planar_broadside_coupled_lines(length=.001,width=.0001,
            upper_level=2,lower_level=1,offset=value,metal="pec")
        @test shape.meta["offset"]==Float64(value)
    end
end

@testset "Corrected pins match exact stored polygon edges" begin
    shape=planar_line(length=.001,width=.0001,level=1,metal="pec")
    for mirror in (false,true),angle in (0.,.1,.3,pi/4,1.),offset in
            ((0.,1e9),(0.,1e10),(0.,1e11),(1e11,0.),(1e11,1e11),
                (nextfloat(0.),0.),(-nextfloat(0.),0.),(.002,.003))
        placed=planar_transform(shape;angle,offset,mirror)
        vertices=only(placed.polygons).vertices
        area=sum(Rational{BigInt}(vertices[i][1])*Rational{BigInt}(vertices[mod1(i+1,length(vertices))][2])-
            Rational{BigInt}(vertices[i][2])*Rational{BigInt}(vertices[mod1(i+1,length(vertices))][1]) for i in eachindex(vertices))
        for pin in placed.pins
            a=vertices[pin.edge];b=vertices[mod1(pin.edge+1,length(vertices))]
            delta=b-a;width=hypot(delta...)
            middle=Float64.((Rational{BigInt}.(a)+Rational{BigInt}.(b))/2)
            normal=sign(area)*DiffMoM._P2(delta[2]/width,-delta[1]/width)
            @test pin.width==width
            @test pin.point==middle
            @test pin.direction==normal
            @test norm(pin.direction)≈1 rtol=3e-16
            @test abs(dot(pin.direction,delta/width))<=2e-16
        end
    end
end

@testset "Corrected air bridge has valid via footprints" begin
    for span in (.001,1.)
        @test_throws ArgumentError planar_air_bridge(;span,width=.0001,landing=.0001,
            bridge_level=2,base_level=1,via_type="v",metal="pec",via_margin=prevfloat(.00005))
    end
    for margin in (0.,.00001,.00004)
        shape=planar_air_bridge(span=.001,width=.0001,landing=.0001,
            bridge_level=2,base_level=1,via_type="v",metal="pec",via_margin=margin)
        for via in shape.vias
            @test length(unique(via.vertices))==4
            normalized,_=planar_normalize_polygon(via.vertices)
            @test length(normalized)==4
        end
    end
end

function check_capacitance(ratios,area)
    stack=PlanarStackup([PlanarLayer(epsr,1.,height) for (height,epsr) in ratios],
        TERM_GND,TERM_GND,.001,.001)
    expected=setprecision(BigFloat,4096) do
        Float64(BigFloat(DiffMoM._EPS0)*BigFloat(area)/
            sum(BigFloat(height)/BigFloat(epsr) for (height,epsr) in ratios))
    end
    if isfinite(expected)&&expected>0
        @test planar_parallel_plate_capacitance(stack,length(ratios),0,area)≈expected rtol=3e-15 atol=nextfloat(0.)
    else
        @test_throws ArgumentError planar_parallel_plate_capacitance(stack,length(ratios),0,area)
    end
end
@testset "Corrected capacitance denominator vs 4096-bit series oracle" begin
    combinations=[(1e-300,1e100),(1e300,1e-100),(1e-300,1e-100),
        (1e300,1e100),(1e-100,1e300),(1e100,1e-300),(1e-6,4.),
        (nextfloat(0.),1e100),(1e100,nextfloat(0.))]
    for ratios in ([v] for v in combinations),area in (1e-300,1e-200,1e-8,1e200,1e300)
        check_capacitance(ratios,area)
    end
    for left in combinations,right in combinations,area in (1e-300,1e-8,1e300)
        check_capacitance([left,right],area)
    end
end
end
