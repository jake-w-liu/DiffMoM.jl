using DiffMoM,Test,LinearAlgebra

# Independent constitutive equations with physical Thevenin sources.
function _legacy_reference_source(Z,refs)
    n=length(refs);roots=sqrt.(refs)
    matrix=vcat(hcat(Matrix{ComplexF64}(I,n,n),-Z),
        hcat(Matrix{ComplexF64}(I,n,n),Diagonal(refs)))
    vi=matrix\vcat(zeros(ComplexF64,n,n),Matrix(Diagonal(2roots)))
    return Diagonal(1 ./ (2roots))*(vi[1:n,:]-Diagonal(refs)*vi[n+1:2n,:])
end

function _legacy_normalized_parameters(Z)
    result=Dict("Z"=>Z,"Y"=>inv(Z))
    if size(Z,1)==2
        H=ComplexF64[Z[1,1]-Z[1,2]*Z[2,1]/Z[2,2] Z[1,2]/Z[2,2];
            -Z[2,1]/Z[2,2] inv(Z[2,2])]
        result["H"]=H;result["G"]=inv(H)
    end
    result
end

function _legacy_parameters_from_external_s(S,kind)
    v=Matrix{ComplexF64}(I,2,2)+S
    current=Matrix{ComplexF64}(I,2,2)-S
    kind=="Y" && return current/v
    kind=="Z" && return v/current
    inputs=vcat(current[1:1,:],v[2:2,:])
    outputs=vcat(v[1:1,:],current[2:2,:])
    return kind=="H" ? outputs/inputs : inputs/outputs
end

# Author literal standards-compliant legacy records independently of the
# production exporter: column order at two ports, row order otherwise.
function _legacy_reference_file(path,freqs,matrices,kind,format,refs)
    n=length(refs)
    open(path,"w") do io
        println(io,"# Hz ",kind," ",format," R ",join(refs," "))
        for (f,M) in zip(freqs,matrices)
            print(io,f)
            for major in 1:n,minor in 1:n
                p,q=n==2 ? (minor,major) : (major,minor)
                n>=3 && minor>1 && (minor-1)%4==0 && println(io)
                z=M[p,q]
                a,b=format=="RI" ? (real(z),imag(z)) :
                    (format=="MA" ? abs(z) : 20log10(abs(z)),180angle(z)/pi)
                print(io," ",a," ",b)
                n>=3 && minor==n && println(io)
            end
            n<3 && println(io)
        end
    end
end

# Reconstruct exactly the stored decimal data at 256 bits. Normalized
# terminal V/I equations provide the reference without production helpers.
function _legacy_stored_reference(path,n)
    setprecision(BigFloat,256) do
        lines=readlines(path);options=split(first(lines));kind=options[3];format=options[4]
        values=parse.(BigFloat,split(join(lines[2:end]," ")));stride=1+2n^2
        result=Matrix{ComplexF64}[]
        for offset in 0:stride:length(values)-stride
            M=zeros(Complex{BigFloat},n,n);index=offset+2
            positions=n==2 ? ((p,q) for q in 1:n for p in 1:n) :
                ((p,q) for p in 1:n for q in 1:n)
            for (p,q) in positions
                a,b=values[index],values[index+1];index+=2
                M[p,q]=format=="RI" ? complex(a,b) :
                    (format=="MA" ? a : big(10)^(a/20))*exp(im*big(pi)*b/180)
            end
            identity=Matrix{Complex{BigFloat}}(I,n,n)
            E,F=if kind=="Y"
                M,-identity
            elseif kind=="Z"
                identity,-M
            elseif kind=="H"
                Complex{BigFloat}[1 -M[1,2];0 -M[2,2]],
                    Complex{BigFloat}[-M[1,1] 0;-M[2,1] 1]
            else
                Complex{BigFloat}[-M[1,1] 0;-M[2,1] 1],
                    Complex{BigFloat}[1 -M[1,2];0 -M[2,2]]
            end
            push!(result,ComplexF64.(-(E-F)\(E+F)))
        end
        result
    end
end

@testset "Touchstone legacy per-port normalization vs independent physical sources" begin
    mktempdir() do directory
        for n in 1:8
            refs=[.1+7p^2 for p in 1:n];D=Diagonal(sqrt.(refs))
            Z=ComplexF64[(p==q ? 2+.1p+.2im : .03p/(p+q)+im*.04q/(p+q)) for p in 1:n,q in 1:n]
            expected=_legacy_reference_source(D*Z*D,refs)
            for (kind,M) in _legacy_normalized_parameters(Z),format in ("RI","MA","DB")
                path=joinpath(directory,"analytic_$(n)_$(kind)_$(format).s$(n)p")
                _legacy_reference_file(path,[1e9,2e9],[M,M],kind,format,refs)
                data=planar_read_touchstone(path);stored=_legacy_stored_reference(path,n)
                @test data.frequencies==[1e9,2e9]
                @test data.z0==refs
                for k in 1:2
                    @test maximum(abs,data.s[k]-expected)<1e-12
                    @test maximum(abs,data.s[k]-stored[k])<1e-12
                end
            end
        end
    end
end

@testset "Touchstone legacy normalization vs archived actual ngspice voltages" begin
    fixture=joinpath(@__DIR__,"fixtures","spice_linear_ngspice47","ladder_grounded")
    rows=readlines(joinpath(fixture,"drive1.dat"))
    freqs=[parse(Float64,first(split(line))) for line in rows[2:end]]
    refs=[50.,75.];external=[zeros(ComplexF64,2,2) for _ in freqs]
    for drive in 1:2
        lines=readlines(joinpath(fixture,"drive$drive.dat"))
        for k in eachindex(freqs)
            v=parse.(Float64,split(lines[k+1]))
            @test v[1]==freqs[k]
            for p in 1:2
                external[k][p,drive]=complex(v[2p],v[2p+1])/sqrt(refs[p])-(p==drive)
            end
        end
    end
    mktempdir() do directory
        for kind in ("Y","Z","H","G"),format in ("RI","MA","DB")
            matrices=[_legacy_parameters_from_external_s(S,kind) for S in external]
            path=joinpath(directory,"ngspice_$(kind)_$(format).s2p")
            _legacy_reference_file(path,freqs,matrices,kind,format,refs)
            data=planar_read_touchstone(path);stored=_legacy_stored_reference(path,2)
            @test data.frequencies==freqs
            @test data.z0==refs
            for k in eachindex(freqs)
                @test maximum(abs,data.s[k]-stored[k])<1e-10
                # Preserve the original external-engine gate on RI data.
                format=="RI" && @test maximum(abs,data.s[k]-external[k])<1e-10
            end
        end
    end
end
