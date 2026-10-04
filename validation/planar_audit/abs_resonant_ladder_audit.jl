using DiffMoM,Test,LinearAlgebra,TOML,Dates,SHA
const audit_directory = normpath(joinpath(@__DIR__, "..", "..", "data", "planar_audit"))
const audit_output = isempty(ARGS) ? joinpath(audit_directory, "abs_resonant_ladder_audit.toml") : abspath(only(ARGS))
mkpath(dirname(audit_output))

const DM=DiffMoM

# Independent ABCD cascade. Each shunt series RLC branch has an analytic
# resonance; no production circuit stamp or interpolation produces truth.
function ladder(f,centers,resistance;delay=0.)
    om=2pi*f;T=Matrix{ComplexF64}(I,2,2)
    for center in centers
        L=1e-9;C=1/((2pi*center)^2*L)
        Y=1/(resistance+1im*(om*L-1/(om*C)))
        phase=om*delay
        T=T*ComplexF64[cos(phase) 50im*sin(phase);im*sin(phase)/50 cos(phase)]*
            ComplexF64[1 0;Y 1]
    end
    A,B,C,D=T[1,1],T[1,2],T[2,1],T[2,2]
    den=A+B/50+C*50+D
    return ComplexF64[(A+B/50-C*50-D)/den 2*(A*D-B*C)/den;
        2/den (-A+B/50-C*50+D)/den]
end

function main()
    rows=Dict{String,Any}()
    for (name,centers,R,delay) in (("single",[3.7e9],1.,0.),
            ("three_with_delay",[2.2e9,3.7e9,5.1e9],.2,2e-11),
            ("deep_notch",[3.7e9],1e-4,0.))
        response=f->ladder(f,centers,R;delay)
        sweep=planar_sweep_abs(response,1e9,6e9;nports=2,n_eval=1001,
            max_points=64,rel_tol=1e-5,max_bytes=8*1024^2)
        error=maximum(maximum(abs,sweep.dense_s[k]-response(sweep.dense_freqs[k]))
            for k in eachindex(sweep.dense_freqs))
        defect=maximum(maximum(abs,S-transpose(S)) for S in sweep.dense_s)
        peak=max(0.,maximum(opnorm(S)-1 for S in sweep.dense_s))
        at_resonance=argmin(abs.(sweep.dense_freqs.-3.7e9))
        rows[name]=Dict("converged"=>sweep.converged,"analyses"=>length(sweep.freqs),
            "max_absolute_full_s"=>error,"reciprocity_defect"=>defect,"passivity_excess"=>peak,
            "notch_db"=>20log10(abs(sweep.dense_s[at_resonance][2,1])),
            "analytic_notch_db"=>20log10(abs(response(3.7e9)[2,1])),
            "source_resistance_ohm"=>R,"centers_hz"=>centers,"delay_s"=>delay)
        println(name," => ",rows[name])
    end
    rows["utc"]=string(now(UTC));rows["scope"]="validation-only independent RLC ABCD ladder versus current adaptive sweep; declared 1-6GHz candidate grid, no arbitrary hidden-resonance guarantee"
    rows["source_sha256"]=open(io->bytes2hex(sha256(io)),joinpath(@__DIR__,"..","..","src","planar","PlanarSweep.jl"))
    open(audit_output,"w") do io;TOML.print(io,rows);end
end
main()
