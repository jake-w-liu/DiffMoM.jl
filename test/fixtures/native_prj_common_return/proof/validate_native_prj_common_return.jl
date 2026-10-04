using DiffMoM, LinearAlgebra, SHA, TOML, Test
include("../sonnet_stripline/sonnet_reference.jl")
using .SonnetReference

function project_text(body,n=2)
    ports=join(1:n," ")
    """FTYP SONNETPRJ 19
VER "18.53"
DIM
ANG DEG
CAP PF
CON /OH
FREQ GHZ
IND NH
LNG MM
RES OH
END DIM
CKT
$body
DEF$(n)P $ports PARENT R 50
END CKT
CONTROL
VARSWP
OPTIONS
SPEED 0
SUBSPLAM N 20
CACHE_ABS 1
Q_ACC N
END CONTROL
VARSWP
ENABLED Y
FREQ Y AN SWEEP 0.25 0.35 0.05
END
END VARSWP
FILEOUT
TOUCH ND Y native_parent.s$(n)p IC 15 S RI R 50
FOLDER .
END FILEOUT
"""
end
child_y(f)=ComplexF64[.2 -.2;-.2 .2+im*(2pi*f*10e-12-1/(2pi*f*25e-9))]
child_s(f)=planar_y_to_s(child_y(f),[50.,50.])
function independent_y(f,n,blocks;return_shunt=0.)
    nn=maximum(vcat(collect(1:n),[collect(b) for b in blocks]...))
    y=zeros(ComplexF64,nn,nn)
    for (a,b,r) in blocks
        d=zeros(Float64,nn,2)
        d[a,1]=1;d[b,2]=1
        if r!=0;d[r,:].-=1;end
        y .+= d*child_y(f)*transpose(d)
    end
    return_shunt>0 && (y[end,end]+=inv(return_shunt))
    internal=setdiff(1:nn,1:n)
    # Test sources use sparse native labels; omit unused coordinates in this
    # independent physical nodal oracle instead of allocating a zero KCL row.
    internal=filter(i->any(!iszero,y[i,:]),internal)
    isempty(internal) ? y[1:n,1:n] :
        y[1:n,1:n]-y[1:n,internal]*(y[internal,internal]\y[internal,1:n])
end

function main()
    directory=evidence_directory("native_prj_common_return")
    report=Dict{String,Any}("runner_sha256"=>bytes2hex(sha256(read(@__FILE__))),"cases"=>Any[])
    child=joinpath(directory,"child.son")
    write(child,replace(project_text("RES 1 2 R=5\nCAP 2 0 C=10\nIND 2 0 L=25"),
        "native_parent.s2p"=>"native_child.s2p"))
    cases=[
        (;name="implicit_ground",n=2,body="PRJ 1 2 child.son 2 1",blocks=[(1,2,0)],shunt=0.),
        (;name="explicit_ground",n=2,body="PRJ 1 2 0 child.son 2 1",blocks=[(1,2,0)],shunt=0.),
        (;name="common_return_resistor",n=2,body="PRJ 1 2 7 child.son 2 1\nRES 7 0 R=13",blocks=[(1,2,7)],shunt=13.),
        (;name="common_return_external",n=3,body="PRJ 1 2 3 child.son 2 1",blocks=[(1,2,3)],shunt=0.),
        (;name="shared_return_resistor",n=3,body="PRJ 1 2 7 child.son 2 1\nPRJ 2 3 7 child.son 2 1\nRES 7 0 R=13",blocks=[(1,2,7),(2,3,7)],shunt=13.),
        (;name="shared_return_external",n=4,body="PRJ 1 2 4 child.son 2 1\nPRJ 2 3 4 child.son 2 1",blocks=[(1,2,4),(2,3,4)],shunt=0.)]
    @testset "Actual native PRJ implicit/explicit/floating/shared common return" begin
        for case in cases
            source=joinpath(directory,case.name*".son");write(source,project_text(case.body,case.n))
            run=reference_run(find_em(),source;output_dir=joinpath(directory,case.name),dependencies=[child],deembedded=false,nports=case.n)
            data=checked_native_touchstone(run,joinpath(run.output_dir,"native_parent.s$(case.n)p");deembedded=false)
            @test data.frequencies≈[250e6,300e6,350e6]
            rawsid=Dict{Float64,Matrix{ComplexF64}}()
            sidpath=joinpath(run.output_dir,"sondata",case.name,"response_nd.sid")
            numeric=Float64[]
            for line in eachline(sidpath)
                values=tryparse.(Float64,split(line));isempty(values) && continue
                all(!isnothing,values) && append!(numeric,values)
            end
            width=1+2case.n^2
            length(numeric)==3width || error("native SID count disagrees with sweep")
            for k in 1:width:length(numeric)
                rawsid[numeric[k]]=reshape(ComplexF64[complex(numeric[j],numeric[j+1]) for j in k+1:2:k+width-1],case.n,case.n)
            end
            errors=Float64[]
            for (f,s) in zip(data.frequencies,data.s)
                y=independent_y(f,case.n,case.blocks;return_shunt=case.shunt)
                oracle=Matrix{ComplexF64}((I+50y)\(I-50y))
                push!(errors,maximum(abs,s-oracle))
                @test last(errors)<1e-10
                @test norm(s-transpose(s))<1e-12
                @test opnorm(s)<=1+1e-12
                if occursin("external",case.name)
                    # Native external RI output rounds individual entries to
                    # about 12 decimals; its validated full-precision SID is
                    # the appropriate oracle for cancellation of a null mode.
                    @test rawsid[f]*ones(case.n)≈ones(case.n) atol=1e-12
                end
            end
            push!(report["cases"],Dict("case"=>case.name,"ports"=>case.n,
                "source_sha256"=>bytes2hex(sha256(read(source))),"full_s_max_error"=>maximum(errors),
                "blocks"=>[collect(b) for b in case.blocks],"return_shunt"=>case.shunt))
        end
    end
    report["scope"]="Actual native circuit PRJ common-return node semantics; geometry calibration and SMD project licensing are distinct and unverified"
    open(joinpath(directory,"comparison.toml"),"w") do io;TOML.print(io,report);end
    println("Evidence: ",directory)
end
main()
