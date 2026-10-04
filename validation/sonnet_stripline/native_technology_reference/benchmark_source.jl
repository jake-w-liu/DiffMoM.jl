# Manufactured linked-STF versus inline-stack native control. No production
# STF reader or table/etch/rpv lowering is credited by this validation probe.
using DiffMoM, SHA, TOML, LinearAlgebra
include("sonnet_reference.jl")
using .SonnetReference

function main()
    em=find_em();em===nothing && error("installed engine required")
    evidence=evidence_directory("native_stf_scalar")
    write(joinpath(evidence,"benchmark_source.jl"),read(@__FILE__))
    stf=joinpath(evidence,"mini.stf")
    write(stf,"""
        <?xml version="1.0" ?>
        <technology_file has_private="false" version="1700">
          <units cunit="SM" lunit="UM" runit="OHUM" srunit="OHSQ" tempunit="C"/>
          <public>
            <variables><var name="TopH" units="LENG" value="100"/><var name="BottomH" units="LENG" value="100"/></variables>
            <materials>
              <dielectric name="Air"><params erel="1"/></dielectric>
              <dielectric name="Substrate"><params erel="4"/></dielectric>
            </materials>
            <metal_model_defs><metal_model model_type="Normal" name="Thin_def"/></metal_model_defs>
            <stackup>
              <TOP material="Lossless" model="Thin_def"/>
              <diel dielectric="Air" name="Upper" thickness="TopH"/>
              <diel dielectric="Substrate" name="Lower" thickness="BottomH"/>
              <BOTTOM material="Lossless" model="Thin_def"/>
            </stackup>
          </public>
        </technology_file>
        """)
    rows=Dict{String,Any}[];baseline=nothing
    report=Dict{String,Any}("scope"=>"manufactured scalar linked-STF/inline-stack engine controls; no table, etch, encrypted or production-STF acceptance",
        "native_linked_inline_identity_gate"=>1e-9,"independent_handlayout_complex_s_gate"=>.005,
        "stf_sha256"=>bytes2hex(sha256(read(stf))),"runs"=>rows)
    for (tag,linked,serialized_cover) in (("inline",false,0.),("linked",true,0.),("linked_serialized_cover10",true,10.))
        stack=linked ? "STF mini.stf\n" : ""
        box=linked ? "BOX 0 1 1 40 40 100 0" : "BOX 1 1 1 40 40 100 0\n.1 1 1 0 0 0 2 \"Air\"\n.1 4 1 0 0 0 2 \"Substrate\""
        source=joinpath(evidence,tag*".son")
        write(source,"""
            FTYP SONPROJ 19
            DIM
            ANG DEG
            CAP PF
            CON /OH
            FREQ GHZ
            IND NH
            LNG MM
            RES OH
            END DIM
            CONTROL
            OPTIONS
            SPEED 0
            SUBSPLAM N 100
            END CONTROL
            GEO
            $(stack)TMET "SerializedCover" 0 SUP $serialized_cover 0 0 0
            BMET "SerializedCover" 0 SUP $serialized_cover 0 0 0
            $box
            POR1 BOX
            POLY 1 1
            3
            1 50 0 0 0 0 .5
            POR1 BOX
            POLY 1 1
            1
            2 50 0 0 0 1 .5
            NUM 1
            0 5 -1 N 1 1 1 100 100 0 0 0 Y
            0 .4
            1 .4
            1 .6
            0 .6
            0 .4
            END
            END GEO
            VARSWP
            ENABLED Y
            FREQ Y AN SWEEP 1 10 9
            END
            END VARSWP
            FILEOUT
            TOUCH ND Y native.s2p IC 15 S RI R 50
            FOLDER .
            END FILEOUT
            """)
        row=Dict{String,Any}("case"=>tag,"linked_stf"=>linked,"serialized_cover_rdc"=>serialized_cover,
            "source_sha256"=>bytes2hex(sha256(read(source))))
        try
            native=reference_run(em,source;output_dir=joinpath(evidence,tag),deembedded=false,
                dependencies=linked ? [stf] : String[])
            output=joinpath(native.output_dir,"native.s2p")
            data=checked_native_touchstone(native,output;deembedded=false,expected_z0=50.)
            row["frequencies_hz"]=data.frequencies
            row["s_real"]=[vec(real.(s)) for s in data.s];row["s_imag"]=[vec(imag.(s)) for s in data.s]
            row["output_sha256"]=bytes2hex(sha256(read(output)));row["status"]="SIMULATED"
            if baseline===nothing && !linked
                baseline=data
                # Independent hand geometry/stack, not a read of the STF.
                grid=CellGrid(.001,.001,20,20)
                sheet=sheet_level(1,20,20);sheet.mask[:,9:12].=true
                sheet.connect_west[9:12].=true;sheet.connect_east[9:12].=true
                stackup=PlanarStackup([PlanarLayer(4.,1.,.0001),PlanarLayer(1.,1.,.0001)],TERM_GND,TERM_GND,.001,.001)
                problem=build_planar_problem(stackup,grid,[sheet],[PlanarPort(1,:west,9:12,50.),PlanarPort(1,:east,9:12,50.)])
                errors=Float64[];residuals=Float64[]
                for (f,s) in zip(data.frequencies,data.s)
                    solved=solve_planar_contracted(problem,f,Matrix{Float64}(I,2,2);method=:dense_fft,
                        mx=80,my=80,max_bytes=100_000_000)
                    push!(errors,maximum(abs.(solved.s-s)));push!(residuals,maximum(solved.raw.relative_residuals))
                end
                row["handlayout_native_full_s_errors"]=errors;row["handlayout_original_residuals"]=residuals
                row["handlayout_status"]=maximum(errors)<=report["independent_handlayout_complex_s_gate"] &&
                    maximum(residuals)<=1e-9 ? "PASS" : "FAIL"
            elseif baseline!==nothing
                data.frequencies==baseline.frequencies || error("native identity control frequencies differ")
                difference=max([maximum(abs.(s-b)) for (s,b) in zip(data.s,baseline.s)]...)
                row["linked_inline_full_s_error"]=difference
                row["identity_status"]=difference<=report["native_linked_inline_identity_gate"] ? "PASS" : "FAIL"
            end
        catch err
            row["status"]="UNVERIFIED";row["error"]=sprint(showerror,err)
            stderr=joinpath(evidence,tag,"engine_stderr.log")
            isfile(stderr) && (row["native_stderr"]=read(stderr,String))
            if occursin("Sonnet Lite does not allow use of linked STF files.",get(row,"native_stderr",""))
                row["status"]="UNVERIFIED_LICENSE"
                row["native_stderr_sha256"]=bytes2hex(sha256(read(stderr)))
                row["native_stdout"]=read(joinpath(evidence,tag,"engine_stdout.log"),String)
            end
        end
        push!(rows,row)
        open(joinpath(evidence,"comparison.toml"),"w") do io;TOML.print(io,report);end
        println(tag," ",row["status"]," ",get(row,"identity_status","")," ",get(row,"error",""))
    end
    println("evidence: ",evidence)
end
main()
