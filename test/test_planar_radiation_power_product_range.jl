module RadiationPowerProductRangeTests
using DiffMoM,LinearAlgebra,Test,TOML
function reference_power(v,i)
    total=zero(Rational{BigInt})
    for p in eachindex(v,i)
        total+=Rational{BigInt}(real(v[p]))*Rational{BigInt}(real(i[p]))+
            Rational{BigInt}(imag(v[p]))*Rational{BigInt}(imag(i[p]))
    end
    Float64(total/2)
end
function coherent_half_power()
    project=load_planar_project(joinpath(@__DIR__,"..","examples","planar_project_line.toml"))
    freq=1e9;model=planar_project_layout(project;freq)
    prob=model.layout.problem;ports=prob.ports
    @test length(ports)==2 && model.layout.contraction===nothing
    a,b=prob.stack.a,prob.stack.b;stack=prob.stack
    nb=planar_basis_count(prob.basis);pbs=DiffMoM._port_basis_indices(prob.basis,length(ports))
    @test all(!isempty(p) for p in pbs)
    selected=vcat(first.(pbs),findall(iszero,prob.basis.port)[1:length(ports)])
    @test length(unique(selected))==2length(ports)
    rhs=zeros(ComplexF64,nb,length(ports));DiffMoM._planar_dense_source_rhs!(rhs,prob)
    edge=exponent(floatmax(Float64))-precision(Float64)
    target=Matrix{ComplexF64}(I,length(ports),length(ports))
    Z=-Matrix{ComplexF64}(I,nb,nb);Z[selected,selected].=0
    for p in eachindex(ports)
        Z[selected[p],selected[p+length(ports)]]=rhs[selected[p],p]
        Z[selected[p+length(ports)],selected[p]]=rhs[selected[p],p]
    end
    Z[selected[length(ports)+1:end],selected[length(ports)+1:end]]=target
    span=exponent(floatmax(Float64))+1-exponent(nextfloat(0.0))
    bits=2span+ndigits(max(2BigInt(nb),1);base=2)
    record=setprecision(BigFloat,bits) do
        factor=lu(Complex{BigFloat}.(Z));X=factor\Complex{BigFloat}.(rhs)
        Y=DiffMoM._planar_dense_port_y(prob,pbs,DiffMoM._planar_port_sign.(ports),X)
        residual=norm(Complex{BigFloat}.(Z)*X-Complex{BigFloat}.(rhs))/norm(Complex{BigFloat}.(rhs))
        (factor=factor,X=X,Y=Y,residual=Float64(residual))
    end
    @test issuccess(record.factor) && all(isfinite,record.X) && all(isfinite,record.Y)
    @test record.residual<=1e-9
    refs=model.z0
    S=planar_y_to_s(record.Y,refs)
    freq=1e9;result=PlanarResult(prob,complex(2pi*freq),complex(freq),Z,record.factor,record.X,record.Y,S,refs)
    voltage=zeros(ComplexF64,length(ports));voltage[1]=ldexp(1.0,cld(exponent(floatmax(Float64)),2))
    project_result=PlanarProjectResult(project,model,result,nothing,freq,model.port_names,refs,record.Y,S)
    exact_current=setprecision(BigFloat,bits) do
        Complex{BigFloat}.(record.Y)*Complex{BigFloat}.(voltage)
    end
    exact_power=setprecision(BigFloat,bits) do
        Float64(real(dot(Complex{BigFloat}.(voltage),exact_current))/2)
    end
    @test all(isfinite,ComplexF64.(exact_current)) && isfinite(exact_power) && exact_power>0
    source_stack=PlanarStackup(stack.layers,TERM_SPACE,TERM_SPACE,a,b)
    coeff=DiffMoM._planar_current_product(record.X,voltage)
    direct=planar_farfield(prob,coeff,freq;theta=[.4,.8],phi=[.2],radiation_stack=source_stack,accepted_power=exact_power)
    current=record.Y*voltage
    @test all(isfinite,current) && !isfinite(real(dot(voltage,current)))
    @test exact_power==reference_power(voltage,current)
    @test all(isfinite,coeff) && all(isfinite,direct.etheta) && all(isfinite,direct.ephi)
    saved=copy(voltage),copy(record.Y),copy(record.X)
    for solved in (result,project_result)
        actual=planar_farfield(solved;voltages=voltage,theta=[.4,.8],phi=[.2],radiation_stack=source_stack)
        @test actual.etheta==direct.etheta && actual.ephi==direct.ephi
        @test actual.accepted_power==exact_power
        @test (voltage,record.Y,record.X)==saved
    end
end
function exercise(root,output)
    @assert realpath(dirname(dirname(pathof(DiffMoM))))==realpath(root)
    @assert output===nothing || !isfile(output)
    result=@testset "Radiation retains finite physical half-power beyond dot storage range" begin
        coherent_half_power()
    end
    counts=Test.get_test_counts(result)
    output===nothing || open(output,"w") do io
        TOML.print(io,Dict("version"=>string(VERSION),"threads"=>Threads.nthreads(),"passes"=>counts.passes+counts.cumulative_passes,"fails"=>counts.fails+counts.cumulative_fails,"errors"=>counts.errors+counts.cumulative_errors))
    end
end
length(ARGS)==2 ? exercise(ARGS...) : exercise(normpath(joinpath(@__DIR__,"..")),nothing)
end
