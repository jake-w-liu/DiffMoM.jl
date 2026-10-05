using DiffMoM,Test

@testset "Native filenames reject invalid budgets before input-sized work" begin
    base=read(joinpath(@__DIR__,"fixtures","native_box_port_attachment",
        "cases","attachment__baseline","project.son"),String)
    lines=split(replace(base,"\r\n"=>"\n"),'\n')
    box=findfirst(x->startswith(x,"BOX "),lines)
    mktempdir() do dir
        allocations=Int[]
        for count in (2,128,1024)
            tokens=String.(split(lines[box]));tokens[2]=string(count-1)
            text=count==2 ? base : join(vcat(lines[1:box-1],[join(tokens,' ')],
                fill(lines[box+1],count-1),[lines[box+2]],lines[box+3:end]),'\n')
            path=joinpath(dir,"layers_$count.son");write(path,text)
            for limit in (0,-1,BigInt(typemax(Int))+1,typemax(UInt128),1.5)
                message=_native_raster_budget_error(()->solve_sonnet_project(path,1e9;
                    raw=true,max_bytes=limit,mx=8,my=8))
                @test occursin("max_bytes",message)
            end
            reject=()->_native_raster_budget_error(()->solve_sonnet_project(path,1e9;
                raw=true,max_bytes=0,mx=8,my=8))
            reject()
            bytes=minimum(@allocated(reject()) for _ in 1:3)
            @test bytes<25000
            push!(allocations,bytes)
        end
        @test maximum(allocations)-minimum(allocations)<4096
        @test occursin("max_bytes",_native_raster_budget_error(()->
            solve_sonnet_project(joinpath(dir,"missing.son"),1e9;max_bytes=0)))
        small=joinpath(dir,"layers_2.son")
        direct=solve_sonnet_project(read_sonnet_project(small),1e9;
            raw=true,grid=(8,8),mx=8,my=8)
        result=solve_sonnet_project(small,1e9;raw=true,grid=(8,8),mx=8,my=8)
        @test result.s==direct.s && result.y==direct.y
        @test result.raw.currents==direct.raw.currents
        circuit=joinpath(dir,"dc_resistor.son")
        write(circuit,"FTYP SONNETPRJ 19\nDIM\nFREQ HZ\nRES OH\nCAP PF\nIND NH\nLNG MM\nEND DIM\nCKT\nRES 1 0 R=50\nDEF1P 1 main R 50\nEND CKT\n")
        cp=read_sonnet_project(circuit)
        @test cp isa SonnetNetlistProject
        for f in (0.,1e9)
            direct=solve_sonnet_project(cp,f;max_bytes=624)
            result=solve_sonnet_project(circuit,f;max_bytes=624)
            @test result.s==direct.s && result.y==direct.y
            @test abs(only(result.s))<1e-12
        end
        @test occursin("max_bytes",_native_raster_budget_error(()->
            solve_sonnet_project(circuit,0.;max_bytes=0)))
    end
end
