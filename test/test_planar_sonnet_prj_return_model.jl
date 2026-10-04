using Test, DiffMoM, LinearAlgebra, SHA

let native=joinpath(@__DIR__, "fixtures", "native_prj_common_return", "native",
        "native_prj_common_return_77wGmW")
    geometry=joinpath(@__DIR__, "fixtures", "native_sparam_model_files", "native", "sparameter.son")
    child_bytes=read(joinpath(native, "child.son"))
    child_y(f)=ComplexF64[.2 -.2; -.2 .2+im*(2pi*f*10e-12-1/(2pi*f*25e-9))]
    function return_y(f, resistance)
        y=child_y(f)
        iszero(resistance) && return y
        incidence=Float64[1 0; 0 1; -1 -1]
        nodal=incidence*y*transpose(incidence)
        nodal[3,3]+=1/resistance
        return nodal[1:2,1:2]-nodal[1:2,3:3]*(nodal[3:3,3:3]\nodal[3:3,1:2])
    end
    # An independent nodal oracle uses pin-to-return voltage differences.
    # The public staged path must retain the same return without adding a pin.
    @testset "Recursive PRJ return through public model-file intake" begin
        mktempdir() do directory
            parent=joinpath(directory,"parent.son")
            device=joinpath(directory,"device.son")
            primitive=joinpath(directory,"primitive.son")
            write(parent,replace(read(geometry,String),
                "TYPE SPARAM 1"=>"TYPE SPROJ device.son\nINHSWP N"))
            write(primitive,child_bytes)
            for (case,resistance) in (("implicit_ground",0.),("explicit_ground",0.),
                    ("common_return_resistor",13.)), flag in (0,1)
                source=replace(read(joinpath(native,case*".son"),String),
                    "child.son 2 1"=>"primitive.son 2 $flag")
                write(device,source)
                for frequency in (250e6,300e6,350e6)
                    files=sonnet_component_files(parent,frequency)
                    binding=only(files.bindings)
                    y=return_y(frequency,resistance)
                    expected=Matrix{ComplexF64}((I+50y)\(I-50y))
                    @test binding.response≈expected rtol=2e-12 atol=2e-13
                    @test size(binding.response)==(2,2) && binding.pin_indices==[1,2]
                    @test Set(keys(files.sources))==Set(realpath.([parent,device,primitive]))
                    @test length(files.circuits)==2
                    for snapshot in values(files.sources)
                        @test snapshot.sha256==bytes2hex(sha256(snapshot.bytes))
                    end
                    @test solve_planar_circuit(files.circuits[realpath(device)],frequency;
                        floating_gauge=:auto).s≈expected rtol=2e-12 atol=2e-13
                end
            end

            # A large literal node label must compact to one additional node.
            sparse=string(typemax(Int)-1)
            source=replace(read(joinpath(native,"common_return_resistor.son"),String),
                "child.son"=>"primitive.son", "PRJ 1 2 7"=>"PRJ 1 2 $sparse",
                "RES 7 0"=>"RES $sparse 0")
            write(device,source)
            frequency=300e6
            files=sonnet_component_files(parent,frequency;max_project_nodes=3)
            @test files.circuits[realpath(device)].nnodes==3
            @test only(files.bindings).response≈planar_y_to_s(return_y(frequency,13.),[50.,50.]) rtol=2e-12

            # Dependencies are copied before callback evaluation and stay owned.
            response=copy(only(files.bindings).response)
            snapshot=copy(files.sources[realpath(primitive)].bytes)
            write(primitive,replace(String(copy(child_bytes)),"R=5"=>"R=9"))
            @test files.sources[realpath(primitive)].bytes==snapshot
            @test only(files.bindings).response==response
            @test solve_planar_circuit(files.circuits[realpath(device)],frequency;
                floating_gauge=:auto).s≈response rtol=1e-13
            @test only(sonnet_component_files(parent,frequency).bindings).response!=response
            write(primitive,child_bytes)

            @testset "Return nodes retain all intake bounds before callbacks" begin
                calls=Ref(0)
                reference=f->(calls[]+=1;50.)
                for options in ((max_bytes=1,),(max_bytes=1000,),
                        (max_project_nodes=2,),(max_project_elements=2,),
                        (max_dependencies=2,),(max_project_depth=1,))
                    @test_throws ArgumentError sonnet_component_files(parent,frequency;
                        requested_reference=reference,options...)
                    @test calls[]==0
                end
                for statement in ("PRJ 1 2 7 9 primitive.son 2 1",
                        "PRJ 1 2 -7 primitive.son 2 1",
                        "PRJ 1 2 1 primitive.son 2 1",
                        "PRJ 1 2 7 primitive.son 0 1",
                        "PRJ 1 2 7 primitive.son $(typemax(Int)) 1",
                        "PRJ 1 2 7 primitive.son 999999999999999999999 1",
                        "PRJ 1 2 7 primitive.son 2 INVALID",
                        "PRJ 1 2 7 primitive.son 2 1 VARIABLE x=3",
                        "PRJ 1 2 7 device.son 2 1")
                    malformed=replace(source,r"(?m)^PRJ[^\r\n]*"=>statement)
                    write(device,malformed)
                    @test_throws ArgumentError sonnet_component_files(parent,frequency;
                        requested_reference=reference)
                    @test calls[]==0
                end
                mktempdir() do outside
                    escaped=joinpath(outside,"outside.son")
                    write(escaped,child_bytes)
                    malformed=replace(source,"primitive.son"=>replace(escaped,'\\'=>'/'))
                    write(device,malformed)
                    @test_throws ArgumentError sonnet_component_files(parent,frequency;
                        requested_reference=reference)
                    @test calls[]==0
                end
            end
        end
    end
end
