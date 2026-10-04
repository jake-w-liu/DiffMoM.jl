using DiffMoM,LinearAlgebra,SHA,TOML
const repo=normpath(joinpath(@__DIR__,".."))
function main()
    original=joinpath(repo,"data/sonnet_validation/geovar_zero_nominal_axes_GAt2Wu")
    output=mktempdir(joinpath(repo,"data/planar_audit");prefix="zero_nominal_literal_support_causal_",cleanup=false)
    cp(@__FILE__,joinpath(output,"producer.jl"))
    tag="XDIR_dir_-1_target_0.0625_literal"
    source=joinpath(original,tag*".son");reference_path=joinpath(original,tag,"native_raw.s2p")
    paths=vcat([@__FILE__,source,reference_path],[joinpath(dir,file) for (dir,_,files) in walkdir(joinpath(repo,"src")) for file in files])
    hashes()=Dict(path=>bytes2hex(sha256(read(path))) for path in paths)
    report=Dict{String,Any}("scope"=>"Independent causal control: original public mask versus two native-observed diagonal boundary cells, with/without their actual undriven PEC wall contact. Unchanged .005 full-S and1e-9 original residual gates.","source_before"=>hashes(),"cases"=>Any[])
    base=sonnet_planar_problem(read_sonnet_project(source);freq=1e9)
    native=only(planar_read_touchstone(reference_path).s)
    try
        for option in ("original","native_cells","native_cells_and_wall_contact")
            sheets=deepcopy(base.sheets)
            if option!="original"
                sheets[1].mask[31,19]=true;sheets[1].mask[32,20]=true
                option=="native_cells_and_wall_contact" && (sheets[1].connect_east[20]=true)
            end
            prob=build_planar_problem(base.stack,base.grid,sheets,base.ports)
            result=solve_planar(prob,1e9;method=:dense_fft,mx=128,my=128,retain_matrix=true)
            rhs=zeros(ComplexF64,size(result.currents))
            for b in eachindex(prob.basis.port)
                port=prob.basis.port[b];iszero(port) && continue
                rhs[b,port]=(port==1 ? -1. : 1.)*prob.basis.width[b]
            end
            row=Dict("case"=>option,"full_s_error"=>maximum(abs,result.s-native),
                "original_voltage_residual"=>norm(result.z_mom*result.currents-rhs)/norm(rhs))
            push!(report["cases"],row);println(row);flush(stdout)
        end
    finally
        report["source_after"]=hashes();report["source_unchanged"]=report["source_before"]==report["source_after"]
        open(joinpath(output,"comparison.toml"),"w") do io;TOML.print(io,report);end
        println("Retained causal support evidence: ",output)
    end
    @assert report["source_unchanged"]
end
main()
