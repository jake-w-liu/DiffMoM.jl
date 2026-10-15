using DiffMoM,Test

@testset "W-line RLGC positivity uses each matrix's physical scale" begin
    mktempdir() do directory
        path=joinpath(directory,"line.lib")
        function model(key,entries)
            fields=Dict("lo"=>"4e-7 1e-7 4e-7","co"=>"1e-10 -2e-11 1e-10",
                "ro"=>"0 0 0","go"=>"0 0 0")
            fields[key]=join(entries," ")
            parameters=join((name*"="*fields[name] for name in ("lo","co","ro","go"))," ")
            write(path,".subckt line a b c d\nw1 a b 0 c d 0 n=2 l=.01 rlgcmodel=m\n"*
                ".model m w modeltype=rlgc n=2 "*parameters*"\n.ends line\n")
            return planar_spice_model(planar_read_spice(path),"line")
        end
        # The invalid matrix has eigenvalues -scale and 3scale. The
        # valid singular matrix has eigenvalues zero and 2scale.
        for (key,scale,field) in (("lo",1e-9,:l),("co",1e-12,:c),
                ("ro",1e-6,:r),("go",1e-12,:g))
            @test_throws ArgumentError model(key,[scale,-2scale,scale])
            spec=only(values(model(key,[scale,-scale,scale]).coupled))
            @test getproperty(spec,field)==[scale -scale;-scale scale]
        end
        @test iszero(only(values(model("ro",[0,0,0]).coupled)).r)
    end
end
