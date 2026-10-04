module ODBSurfaceBudgetTests
using DiffMoM,Test
const DM=DiffMoM

function comb(n)
    points=Tuple{Int,Int}[(0,0),(n,0),(n,2)]
    for k in n-1:-1:0
        push!(points,(k,iseven(k) ? 2 : 1))
        k>0 && push!(points,(k,iseven(k-1) ? 2 : 1))
    end
    push!(points,(0,0))
    "UNITS=MM\nS P 0\nOB 0 0 I\n"*join(("OS $x $y\n" for (x,y) in points[2:end]))*"OE\nSE\n"
end

function readbody(body;kwargs...)
    mktempdir() do directory
        path=joinpath(directory,"features");write(path,body)
        read_odb_features(path;kwargs...)
    end
end

@testset "ODB contour storage is reserved before allocation" begin
    for n in (100,1000,10000)
        body=comb(n);limit=3ncodeunits(body)+2048
        mktempdir() do directory
            path=joinpath(directory,"features");write(path,body)
            call()=try
                read_odb_features(path;max_bytes=limit)
                nothing
            catch e
                e isa ArgumentError || rethrow()
                e
            end
            error=call()
            @test error isa ArgumentError
            @test occursin("ODB contour segment storage",sprint(showerror,error))
            # This includes the owned input snapshot. No proportional
            # contour geometry or per-segment parsing has occurred yet.
            @test (@allocated call())<ncodeunits(body)+50000
            observed=Int[]
            document=read_odb_features(path;max_bytes=64*1024^2,
                _payload_observer=p->push!(observed,p))
            shape=only(document.objects).shape
            segments=only(shape.parts)[2].segments
            @test length(segments)==2n+2
            @test Base.summarysize(shape)>limit
            @test DM._artwork_contains(shape,.5e-3,.5e-3)
            @test !DM._artwork_contains(shape,-.5e-3,.5e-3)
            @test issorted(observed)
            @test length(read_odb_features(path;max_bytes=last(observed)).objects)==1
            @test_throws ArgumentError read_odb_features(path;max_bytes=last(observed)-1)
        end
    end
end

@testset "Contour slot preflight matches parser whitespace and ownership" begin
    for white in (" ","\t","\u00a0","\u2003","\u202f"),newline in ("\n","\r\n")
        body=join(("UNITS=MM","S P 0","OB 0 0 I",white*"OS"*white*"1 0",
            "# OS 999 999",white*"OC"*white*"0 1 0 0 N",white*"OS"*white*"0 0",
            white*"OE","SE"),newline)*newline
        shape=only(readbody(body).objects).shape
        segments=only(shape.parts)[2].segments
        @test length(segments)==3
        @test DM._artwork_contains(shape,.25e-3,.25e-3)
        @test !DM._artwork_contains(shape,1.1e-3,.25e-3)
        @test_throws ArgumentError readbody(body;max_objects=2)
    end
    # A later contour must not overwrite the first contour's owned buffer.
    body="UNITS=MM\nS P 0\nOB 0 0 I\nOS 1 0\nOS 1 1\nOS 0 1\nOS 0 0\nOE\n"*
        "OB 2 0 I\nOS 3 0\nOS 3 1\nOS 2 1\nOS 2 0\nOE\nSE\n"
    shape=only(readbody(body).objects).shape
    @test length(shape.parts)==2
    @test shape.parts[1][2].segments!==shape.parts[2][2].segments
    for x in (.25,.75,2.25,2.75)
        @test DM._artwork_contains(shape,x*1e-3,.5e-3)
    end
    @test !DM._artwork_contains(shape,1.5e-3,.5e-3)
    for suffix in ("", "OS 1 0\n", "OS 1 0\nOE\nSE\n", "OE\nSE\n",
            "OSinvalid 1 0\nOE\nSE\n")
        @test_throws ArgumentError readbody("UNITS=MM\nS P 0\nOB 0 0 I\n"*suffix)
    end
end
end
