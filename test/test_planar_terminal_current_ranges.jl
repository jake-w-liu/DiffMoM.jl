module TerminalCurrentRangeTests
using DiffMoM,LinearAlgebra,SparseArrays,Test,TOML
function exact_current(Y,v)
    out=Vector{ComplexF64}(undef,size(Y,1))
    for p in axes(Y,1)
        re,im=zero(Rational{BigInt}),zero(Rational{BigInt})
        for q in axes(Y,2)
            ar,ai=Rational{BigInt}(real(Y[p,q])),Rational{BigInt}(imag(Y[p,q]))
            br,bi=Rational{BigInt}(real(v[q])),Rational{BigInt}(imag(v[q]))
            re+=ar*br-ai*bi;im+=ar*bi+ai*br
        end
        out[p]=complex(Float64(re),Float64(im))
    end
    out
end
# Supported caller-owned solved coefficients are coherent with their
# matrix, LU, source equation and retained port admittance. This is a
# numerical API regression, not an assembled device reference.
function public_coherent_terminal_cancellation()
    a=b=.002;grid=CellGrid(a,b,4,4)
    stack=PlanarStackup([PlanarLayer(1.,1.,.001),PlanarLayer(1.,1.,.001)],TERM_GND,TERM_GND,a,b)
    sheet=sheet_level(1,4,4);rasterize_rect!(sheet,grid,0,a,0,b)
    ports=[PlanarPort(1,:x,2,2:2,50.),PlanarPort(1,:x,3,2:2,50.)]
    prob=build_planar_problem(stack,grid,[sheet],ports)
    nb=planar_basis_count(prob.basis);pbs=DiffMoM._port_basis_indices(prob.basis,length(ports))
    @test all(length(p)==1 for p in pbs)
    selected=vcat(only.(pbs),findall(iszero,prob.basis.port)[1:length(ports)])
    @test length(unique(selected))==2length(ports)
    rhs=zeros(ComplexF64,nb,length(ports));DiffMoM._planar_dense_source_rhs!(rhs,prob)
    edge=exponent(floatmax(Float64))-precision(Float64)
    target=im*ldexp(1.0,edge)*ComplexF64[1 -1;-1 1]
    Z=Matrix{ComplexF64}(I,nb,nb);Z[selected,selected].=0
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
    @test issuccess(record.factor) && all(isfinite,record.X) && record.Y==target
    @test record.residual<=1e-9
    refs=ones(ComplexF64,length(ports))
    S=planar_y_to_s(record.Y,refs)
    freq=1e9;result=PlanarResult(prob,complex(2pi*freq),complex(freq),Z,record.factor,record.X,record.Y,S,refs)
    voltage=fill(complex(ldexp(1.0,exponent(floatmax(Float64))÷2-precision(Float64))),length(ports))
    source_stack=PlanarStackup(stack.layers,TERM_SPACE,TERM_SPACE,a,b)
    coeff=DiffMoM._planar_current_product(record.X,voltage)
    direct=planar_farfield(prob,coeff,freq;theta=[.4,.8],phi=[.2],radiation_stack=source_stack)
    @test !all(isfinite,record.Y*voltage)
    @test all(isfinite,coeff) && all(isfinite,direct.etheta) && all(isfinite,direct.ephi)
    original=copy(voltage),copy(record.Y),copy(record.X)
    actual=planar_farfield(result;voltages=voltage,theta=[.4,.8],phi=[.2],radiation_stack=source_stack)
    @test actual.etheta==direct.etheta && actual.ephi==direct.ephi
    @test actual.accepted_power===nothing
    @test (voltage,record.Y,record.X)==original
end

# Use the actual project lowering, source incidence and port geometry.
# Only the caller-owned constitutive matrix/LU is controlled here.
function public_project_terminal_cancellation()
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
    target=im*ldexp(1.0,edge)*ComplexF64[1 -1;-1 1]
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
    @test issuccess(record.factor) && all(isfinite,record.X) && imag.(record.Y)==imag.(target)
    @test record.residual<=1e-9
    refs=model.z0
    S=planar_y_to_s(record.Y,refs)
    freq=1e9;result=PlanarResult(prob,complex(2pi*freq),complex(freq),Z,record.factor,record.X,record.Y,S,refs)
    voltage=fill(complex(ldexp(1.0,exponent(floatmax(Float64))÷2-precision(Float64))),length(ports))
    project_result=PlanarProjectResult(project,model,result,nothing,freq,model.port_names,refs,record.Y,S)
    reference_current=setprecision(BigFloat,bits) do
        Complex{BigFloat}.(record.Y)*Complex{BigFloat}.(voltage)
    end
    exact_power=setprecision(BigFloat,bits) do
        Float64(real(dot(Complex{BigFloat}.(voltage),reference_current))/2)
    end
    @test all(isfinite,ComplexF64.(reference_current)) && isfinite(exact_power) && exact_power>0
    source_stack=PlanarStackup(stack.layers,TERM_SPACE,TERM_SPACE,a,b)
    coeff=DiffMoM._planar_current_product(record.X,voltage)
    direct=planar_farfield(prob,coeff,freq;theta=[.4,.8],phi=[.2],radiation_stack=source_stack,accepted_power=exact_power)
    @test !all(isfinite,record.Y*voltage)
    @test all(isfinite,coeff) && all(isfinite,direct.etheta) && all(isfinite,direct.ephi)
    @test ComplexF64.(reference_current)==exact_current(record.Y,voltage)
    original=copy(voltage),copy(record.Y),copy(record.X)
    actual=planar_farfield(project_result;voltages=voltage,theta=[.4,.8],phi=[.2],radiation_stack=source_stack)
    @test actual.etheta==direct.etheta && actual.ephi==direct.ephi
    @test actual.accepted_power==exact_power
    @test (voltage,record.Y,record.X)==original
end

function exercise(root,output)
    @assert realpath(dirname(dirname(pathof(DiffMoM))))==realpath(root)
    @assert output===nothing || !isfile(output)
    result=@testset "Retained inverse waves recover finite terminal product cancellation" begin
        for T in (Float16,Float32,Float64),kind in (:real,:reactive)
            edge=exponent(floatmax(T));gain=ldexp(T(1),edge)
            drive=ldexp(1.0,exponent(floatmax(Float64))-edge+1)
            Y=kind===:real ? gain*T[1 -1;-1 1] : complex(zero(T),gain)*T[1 -1;-1 1]
            v=fill(complex(drive),2);refs=ones(ComplexF64,2)
            reference=exact_current(Y,v);expected=v/2
            saved=copy(Y),copy(v),copy(refs)
            current=@inferred DiffMoM._planar_terminal_current(Y,v;max_bytes=DiffMoM._default_max_dense_payload_bytes(),retained_bytes=0)
            @test current==reference && all(iszero,reference)
            @test !all(isfinite,Y*v)
            @test (@inferred DiffMoM._planar_incident_admittance(Y,v,refs))==expected
            @test DiffMoM._planar_incident_admittance(sparse(Y),view(v,:),view(refs,:))==expected
            @test (Y,v,refs)==saved
            bits=DiffMoM._planar_terminal_product_precision(size(Y,2))
            # Charge an actually retained voltage/current/root workspace.
            held=DiffMoM._checked_array_payload_bytes(ComplexF64,2,length(v))+
                DiffMoM._checked_array_payload_bytes(Float64,length(v))
            required=held+5DiffMoM._planar_wide_scalar_payload(bits)
            @test_throws ArgumentError DiffMoM._planar_incident_admittance(Y,v,refs;max_bytes=required-1,retained_bytes=held)
            @test DiffMoM._planar_incident_admittance(Y,v,refs;max_bytes=required,retained_bytes=held)==expected
            @test DiffMoM._planar_incident_admittance(Y,v,refs)!==v
        end
        edge=exponent(floatmax(Float64))-precision(Float64)
        gain=ldexp(1.0,edge);drive=ldexp(1.0,exponent(floatmax(Float64))-precision(Float64)÷2)
        Y=im*gain*ComplexF64[1 -1;-1 1];v=fill(complex(ldexp(1.0,exponent(floatmax(Float64))÷2)),2)
        refs=fill(complex(ldexp(1.0,-edge)),2)
        expected=v./(2sqrt.(real.(refs)))
        @test all(isfinite,expected)
        @test (@inferred DiffMoM._planar_incident_admittance(Y,v,refs))==expected
        tiny=nextfloat(0.0)
        matrix=ComplexF64[im*gain -im*gain tiny;-im*gain im*gain -tiny;0 0 0]
        voltage=ComplexF64[v[1],v[1],1+im]
        reference=exact_current(matrix,voltage)
        current=DiffMoM._planar_terminal_current(matrix,voltage;max_bytes=DiffMoM._default_max_dense_payload_bytes(),retained_bytes=0)
        @test current==reference
        @test current[1]==complex(tiny,tiny) && current[2]==complex(-tiny,-tiny)
        @test isapprox(DiffMoM._planar_incident_admittance(matrix,voltage,ones(ComplexF64,3)),
            (voltage+reference)/2;rtol=3e-14,atol=0)
        # A truly unrepresentable terminal current remains an explicit error.
        @test_throws ArgumentError DiffMoM._planar_incident_admittance(Y,ComplexF64[v[1],-v[1]],refs)
        ordinary=Matrix{ComplexF64}(I,2,2);ordinary_v=ComplexF64[1,im]
        @test DiffMoM._planar_incident_admittance(ordinary,ordinary_v,ones(ComplexF64,2))==ordinary_v
        # Wider ordinary input arithmetic and caller MPFR state stay intact.
        for bits in (precision(Float64),2precision(Float64)),mode in (RoundNearest,RoundDown,RoundUp)
            setprecision(BigFloat,bits) do
                setrounding(BigFloat,mode) do
                    @test DiffMoM._planar_incident_admittance(Y,v,refs)==expected
                    @test precision(BigFloat)==bits && rounding(BigFloat)===mode
                    wide=Complex{BigFloat}.(ordinary)
                    @test DiffMoM._planar_incident_admittance(wide,ordinary_v,ones(ComplexF64,2))==ordinary_v
                end
            end
        end
        tasks=[Threads.@spawn DiffMoM._planar_incident_admittance(Y,v,refs) for _ in eachindex(v)]
        @test all(fetch(task)==expected for task in tasks)
        public_coherent_terminal_cancellation()
        public_project_terminal_cancellation()
    end
    counts=Test.get_test_counts(result)
    output===nothing || open(output,"w") do io
        TOML.print(io,Dict("version"=>string(VERSION),"threads"=>Threads.nthreads(),
            "passes"=>counts.passes+counts.cumulative_passes,"fails"=>counts.fails+counts.cumulative_fails,
            "errors"=>counts.errors+counts.cumulative_errors))
    end
end
length(ARGS)==2 ? exercise(ARGS...) : exercise(normpath(joinpath(@__DIR__,"..")),nothing)
end
