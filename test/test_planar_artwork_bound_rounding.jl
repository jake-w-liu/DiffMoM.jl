module ArtworkRationalBoundStateTests
using DiffMoM, Test, SHA, LinearAlgebra
function run_tests()
    root=dirname(@__DIR__)
    file=joinpath(root,"src/planar/PlanarArtworkBounds.jl")
    before=bytes2hex(sha256(read(file)))
    Q=Rational{BigInt}
    subnormal=Q(nextfloat(0.))
    maxfinite=Q(floatmax(Float64))
    one_neighbor=Q(nextfloat(1.))
    q=(big(2)^(precision(Float32)+1)+1)//big(2)^(precision(Float32)+1)
    values=Q[0,1,-1,q,-q,one_neighbor,-one_neighbor,
        (Q(1)+one_neighbor)/2,-(Q(1)+one_neighbor)/2,
        subnormal,subnormal/2,subnormal/3,2subnormal/3,-subnormal/3,
        Q(floatmin(Float64)),Q(prevfloat(floatmin(Float64))),
        maxfinite,maxfinite*2,-maxfinite*2,Q(1)/3,-Q(1)/3]
    original=(precision(BigFloat),rounding(BigFloat))
    rows=Dict{String,Any}[]
    enclose(v,target,lower)=isfinite(v) ? (lower ? Q(v)<=target : Q(v)>=target) : (lower ? v==-Inf : v==Inf)
    tight(v,target,lower)=isfinite(v) ? !enclose(lower ? nextfloat(v) : prevfloat(v),target,lower) : true
    radius=Float64(q)
    rect=DiffMoM._artwork_rectangle(2radius,2radius)
    ordinary=DiffMoM._artwork_transform(rect)
    matrix=typeof(ordinary.inverse)(1.,1.,1.,nextfloat(1.))
    shape=DiffMoM._ArtworkTransform(rect,ordinary.matrix,matrix,(0.,0.))
    corners=[inv(Q.(matrix))*Q.([x,y]) for x in (-radius,radius),y in (-radius,radius)]
    @testset "Exact artwork bounds retain enclosure under caller precision and rounding" begin
        @test unsafe_string(ccall((:mpfr_print_rnd_mode,DiffMoM._planar_mpfr_library),Cstring,(Cint,),DiffMoM._artwork_mpfr_up))=="MPFR_RNDU"
        @test unsafe_string(ccall((:mpfr_print_rnd_mode,DiffMoM._planar_mpfr_library),Cstring,(Cint,),DiffMoM._artwork_mpfr_down))=="MPFR_RNDD"
        for precision_bits in (precision(Float16),precision(Float32),precision(Float64)),mode in (RoundNearest,RoundDown,RoundUp)
            setprecision(BigFloat,precision_bits) do
                setrounding(BigFloat,mode) do
                    settings=(precision(BigFloat),rounding(BigFloat))
                    for target in values,lower in (true,false)
                        value=DiffMoM._artwork_exact_bound_float(target,lower)
                        @test enclose(value,target,lower)
                        @test tight(value,target,lower)
                        @test (precision(BigFloat),rounding(BigFloat))==settings
                    end
                    bounds=DiffMoM._artwork_bounds(shape)
                    @test all(p->Q(bounds[1])<=p[1]<=Q(bounds[2]) && Q(bounds[3])<=p[2]<=Q(bounds[4]),corners)
                    mktempdir() do directory
                        path=joinpath(directory,"features")
                        write(path,"UNITS=MM\n\$0 custom\nP 0 0 0 P 0 0\n")
                        artwork=read_odb_features(path;symbol_resolver=name->shape)
                        bounds=only(artwork.objects).bounds
                        @test all(p->Q(bounds[1])<=p[1]<=Q(bounds[2]) && Q(bounds[3])<=p[2]<=Q(bounds[4]),corners)
                        grid=CellGrid(2radius,2radius,1,1)
                        masks=artwork_cell_masks(artwork,grid;offset=(radius,radius))
                        @test only(Base.values(masks))[1,1]
                    end
                    @test (precision(BigFloat),rounding(BigFloat))==settings
                    push!(rows,Dict("precision"=>precision_bits,"rounding"=>string(mode),"bounds"=>collect(bounds)))
                end
            end
        end
        tasks=[Threads.@spawn (DiffMoM._artwork_exact_bound_float(q,true),DiffMoM._artwork_exact_bound_float(q,false)) for _ in 1:Threads.nthreads()]
        for task in tasks
            lower,upper=fetch(task)
            @test Q(lower)<=q<=Q(upper)
        end
        @test (precision(BigFloat),rounding(BigFloat))==original
        @test bytes2hex(sha256(read(file)))==before
    end
end
run_tests()
end
