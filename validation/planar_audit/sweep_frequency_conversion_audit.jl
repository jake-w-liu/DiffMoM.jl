using DiffMoM,TOML,Dates
const audit_directory = normpath(joinpath(@__DIR__, "..", "..", "data", "planar_audit"))
const audit_output = isempty(ARGS) ? joinpath(audit_directory, "sweep_frequency_conversion_audit.toml") : abspath(only(ARGS))
mkpath(dirname(audit_output))


function main()
    rows=Dict{String,Any}()
    cases=("representable_bigfloat"=>(big"1e9",big"2e9"),
        "overflow"=>(big"1e500",big"2e500"),
        "underflow"=>(big"1e-500",big"2e-500"),
        "unresolvable_midpoint"=>(1e9,nextfloat(1e9)))
    for (name,(lo,hi)) in cases
        calls=Ref(0)
        response=f->begin calls[]+=1;fill(.2+0im,1,1) end
        try
            result=planar_sweep_abs(response,lo,hi;nports=1,n_eval=17)
            rows[name]=Dict("status"=>"RETURNED","calls"=>calls[],
                "converged"=>result.converged,"finite_positive_frequencies"=>all(f->isfinite(f)&&f>0,result.freqs),
                "distinct_analysis_frequencies"=>length(unique(result.freqs)),
                "analysis_count"=>length(result.freqs),
                "finite_positive_dense"=>all(f->isfinite(f)&&f>0,result.dense_freqs))
        catch e
            rows[name]=Dict("status"=>"REJECTED","calls"=>calls[],"reason"=>sprint(showerror,e))
        end
        println(name," => ",rows[name])
    end
    calls=Ref(0)
    try
        result=planar_sweep_abs(f->begin calls[]+=1;fill(Complex{BigFloat}(big"1e500",0),1,1) end,
            1e9,2e9;nports=1,n_eval=17,max_points=4)
        rows["response_overflow"]=Dict("status"=>"RETURNED","calls"=>calls[],
            "converged"=>result.converged,"finite_samples"=>all(S->all(isfinite,S),result.s))
    catch e
        rows["response_overflow"]=Dict("status"=>"REJECTED","calls"=>calls[],"reason"=>sprint(showerror,e))
    end
    println("response_overflow => ",rows["response_overflow"])
    rows["utc"]=string(now(UTC))
    open(audit_output,"w") do io;TOML.print(io,rows);end
end
main()
