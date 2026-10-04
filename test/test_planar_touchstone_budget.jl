using DiffMoM,Test,LinearAlgebra

function _touchstone_budget_file(path,kind,n)
    open(path,"w") do io
        if kind in (:data,:reference)
            println(io,"[Version] 2.1\n# HZ S RI R 50\n[Number of Ports] ",n,"\n[Number of Frequencies] 1")
            kind==:reference && println(io,"[Reference] ",repeat("50 ",n))
            println(io,"[Network Data]\n1 ",repeat("0 0 ",n*n),"\n[End]")
        else
            println(io,kind==:options ? "# HZ S RI R "*repeat("50 ",n) :
                "# HZ S RI R 50\n! TERM "*repeat("50 0 ",n))
            print(io,"1 ")
            for p in 1:n
                p>1 && println(io)
                for q in 1:n
                    q>1 && (q-1)%4==0 && println(io)
                    print(io,"0 0 ")
                end
            end
            println(io)
        end
    end
end

@testset "Explicit empty Touchstone Reference rejects rather than using defaults" begin
    mktempdir() do directory
        for version in ("2.0","2.1"),n in (1,2),reference in ("[Reference]","[reference]   ","[Reference]\n! no values\n\t")
            path=joinpath(directory,"empty.s$(n)p")
            header="[Version] $version\n# HZ S RI R 50\n[Number of Ports] $n\n[Number of Frequencies] 1\n"*
                (n==2 ? "[Two-Port Data Order] 21_12\n" : "")
            response="[Network Data]\n1 "*repeat("0 0 ",n*n)*"\n[End]\n"
            original=header*reference*"\n"*response;write(path,original)
            error=try planar_read_touchstone(path);nothing catch e;e end
            @test error isa ArgumentError && occursin("[Reference]",sprint(showerror,error))
            @test read(path,String)==original
            # The optional keyword may be absent, or it may start a wrapped
            # set of values on the next line. Both retain valid defaults/data.
            write(path,header*response)
            @test planar_read_touchstone(path).z0==fill(50.,n)
            write(path,header*"[Reference]\n"*join(fill("75",n),"\n")*"\n"*response)
            @test planar_read_touchstone(path).z0==fill(75.,n)
        end
    end
end

@testset "Touchstone valid long records reject before numeric allocation" begin
    mktempdir() do directory
        for kind in (:data,:reference,:term,:options)
            path=joinpath(directory,"$(kind).s224p")
            _touchstone_budget_file(path,kind,224)
            accepted=planar_read_touchstone(path;max_bytes=32*1024^2)
            @test size(accepted.s[1])==(224,224) && iszero(norm(accepted.s[1]))
            @test_throws ArgumentError planar_read_touchstone(path;max_bytes=512)
            bytes=@allocated try planar_read_touchstone(path;max_bytes=512) catch;end
            @test bytes<50_000
            if kind in (:data,:reference)
                anonymous=joinpath(directory,"$(kind).ts");cp(path,anonymous)
                @test_throws ArgumentError planar_read_touchstone(anonymous;max_bytes=512)
                bytes=@allocated try planar_read_touchstone(anonymous;max_bytes=512) catch;end
                @test bytes<50_000
            end
        end
        # Triangular input still produces a dense output; its lower-bound
        # workspace check does not assume 2*n*n input fields.
        path=joinpath(directory,"triangle.s3p")
        write(path,"[Version] 2.1\n# HZ S RI R 50\n[Number of Ports] 3\n[Number of Frequencies] 1\n[Matrix Format] Lower\n[Network Data]\n1 0 0 .1 0 0 0 .2 0 .3 0 0 0\n[End]\n")
        @test planar_read_touchstone(path;max_bytes=4000).s[1]≈[0 .1 .2;.1 0 .3;.2 .3 0]
        # Long numerical precision is valid independently of the line length.
        path=joinpath(directory,"precision.s1p")
        write(path,"# HZ S RI R 50\n1 0."*repeat("0",100_000)*" 0\n")
        @test planar_read_touchstone(path;max_bytes=4000).s[1]==zeros(ComplexF64,1,1)
    end
end

@testset "Touchstone streaming numeric/options/termination preflight" begin
    line="1 "*repeat("0 0 ",224^2)
    target=Float64[]
    @test_throws ArgumentError DiffMoM._touchstone_append_numbers!(target,line,0,512)
    @test isempty(target)
    bytes=@allocated try DiffMoM._touchstone_append_numbers!(target,line,0,512) catch;end
    @test bytes<10_000
    DiffMoM._touchstone_append_numbers!(target,line,0,1_000_000)
    @test length(target)==100353 && target[1]==1 && all(iszero,view(target,2:length(target)))
    for (helper,text) in ((DiffMoM._touchstone_options,"HZ S RI R "*repeat("50 ",224)),
            (DiffMoM._touchstone_termination_values,repeat("50 0 ",224)))
        @test_throws ArgumentError helper(text;max_bytes=512)
        bytes=@allocated try helper(text;max_bytes=512) catch;end
        @test bytes<20_000
    end
    refs=Float64[]
    DiffMoM._touchstone_append_numbers!(refs,"50 75",0,16)
    @test_throws ArgumentError DiffMoM._touchstone_append_numbers!(refs,"100",0,16)
    @test refs==[50,75]
    @test DiffMoM._touchstone_options("HZ S RI R 50D0")[4]==[50.]
    @test DiffMoM._touchstone_termination_values("50 10 &")==([50.,10.],true)
    @test_throws ArgumentError DiffMoM._touchstone_termination_values("50 & 10")
end
