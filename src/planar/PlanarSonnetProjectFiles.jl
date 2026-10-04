# Bounded snapshot traversal for linear native project components.
mutable struct _SonnetLinearProjects
    frequency::Float64
    root::String
    budget::_SpiceBudget
    max_dependencies::Int
    max_depth::Int
    max_nodes::Int
    max_elements::Int
    sources::Dict{String,SonnetModelSource}
    networks::Dict{String,PlanarNetworkData}
    circuits::Dict{String,PlanarCircuit}
    responses::Dict{String,Tuple{Matrix{ComplexF64},Vector{ComplexF64}}}
    active::Set{String}
end
_sonnet_project_path_key(path)=Sys.iswindows() ? lowercase(path) : path

function _sonnet_files_project_source!(projects,path)
    haskey(projects.sources,path) && return projects.sources[path]
    length(projects.sources)<projects.max_dependencies ||
        throw(ArgumentError("native dependency count exceeds max_dependencies"))
    snapshot=_sonnet_files_snapshot(path,projects.budget)
    projects.sources[path]=snapshot
    return snapshot
end

function _sonnet_files_project_network!(projects,p,row,np)
    path=_sonnet_files_path(projects.root,p,row.tokens[end],row)
    if !haskey(projects.networks,path)
        suffix=match(r"\.([syz])(\d+)p$"i,path)
        suffix===nothing && lowercase(splitext(path)[2])!=".ts" &&
            _sonnet_error(p.source,row.line,"unrepresented circuit model file format")
        suffix===nothing || tryparse(Int,suffix[2])==np ||
            _sonnet_error(p.source,row.line,"circuit data port count disagrees with terminals")
        snapshot=_sonnet_files_project_source!(projects,path)
        data=mktemp() do file,io
            write(io,snapshot.bytes);close(io)
            planar_read_touchstone(file;nports=np,max_bytes=projects.budget.limit-projects.budget.used)
        end
        _sonnet_files_reserve!(projects.budget,_sonnet_files_data_payload(data))
        projects.networks[path]=data
    end
    data=projects.networks[path]
    length(data.z0)==np || _sonnet_error(p.source,row.line,"shared circuit data has inconsistent port count")
    first(data.frequencies)<=projects.frequency<=last(data.frequencies) ||
        _sonnet_error(p.source,row.line,"frequency outside circuit data coverage; extrapolation is forbidden")
    return path,data
end

function _sonnet_files_linear_project!(projects::_SonnetLinearProjects,path,np,depth=1)
    key=_sonnet_project_path_key(path)
    key in projects.active && throw(ArgumentError("recursive native project dependency"))
    depth<=projects.max_depth || throw(ArgumentError("native project depth exceeds max_project_depth"))
    if haskey(projects.circuits,path)
        length(projects.circuits[path].ports)==np || throw(ArgumentError("shared project has inconsistent pin count"))
        return projects.circuits[path]
    end
    push!(projects.active,key)
    try
        snapshot=_sonnet_files_project_source!(projects,path)
        _sonnet_files_reserve!(projects.budget,BigInt(32)*length(snapshot.bytes)+4096)
        records=SonnetRecord[]
        for (line,text) in enumerate(eachsplit(String(copy(snapshot.bytes)),'\n'))
            tokens=_sonnet_tokens(text);isempty(tokens) || push!(records,SonnetRecord(line,tokens))
        end
        p=_sonnet_read_records(path,records)
        p isa SonnetNetlistProject || throw(ArgumentError(
            "geometry SPROJ children require their explicit calibrated project adapter"))
        scales=Dict("RES"=>get(Dict("OH"=>1.,"OHMS"=>1.,"KOH"=>1e3,"MOH"=>1e6),get(p.units,"RES","OH"),NaN),
            "CAP"=>get(Dict("F"=>1.,"PF"=>1e-12,"FF"=>1e-15,"NF"=>1e-9,"UF"=>1e-6),get(p.units,"CAP",""),NaN),
            "IND"=>get(Dict("H"=>1.,"NH"=>1e-9,"PH"=>1e-12,"UH"=>1e-6,"MH"=>1e-3),get(p.units,"IND",""),NaN))
        all(isfinite,values(scales)) || throw(ArgumentError("unsupported linear project RLC units"))
        definitions=Dict{String,Int}();nodes=Set{Int}();elements=0;ports=0;currents=0
        totalnodes=0;totalelements=0
        compiled=BigInt(4096);workspace=BigInt(0);finalports=0
        project_responses=Dict{String,Matrix{ComplexF64}}();scaledrecords=SonnetRecord[]
        networks=Dict{String,PlanarNetworkData}()
        function terminal(token,row)
            value=tryparse(Int,token)
            value!==nothing && value>=0 || _sonnet_error(p.source,row.line,"linear project requires literal nonnegative node labels")
            value==0 || push!(nodes,value)
            length(nodes)<=projects.max_nodes || throw(ArgumentError("native project nodes exceed max_project_nodes"))
        end
        for row in p.circuit
            t=row.tokens;kind=first(t);definition=match(r"^DEF(\d+)P$",kind)
            push!(scaledrecords,row)
            if definition!==nothing
                count=tryparse(Int,definition[1])
                count!==nothing && count>0 && count<=projects.max_nodes && length(t)==count+4 && t[end-1]=="R" ||
                    _sonnet_error(p.source,row.line,"linear project DEF requires literal ports, name and R reference")
                for token in t[2:count+1];terminal(token,row);end
                external=parse.(Int,t[2:count+1])
                all(>(0),external) && length(unique(external))==count ||
                    _sonnet_error(p.source,row.line,"linear project DEF ports must be distinct positive nodes")
                reference=_sonnet_circuit_literal(p,row,t[end]);reference>0 ||
                    _sonnet_error(p.source,row.line,"linear project reference must be positive")
                name=t[count+2];haskey(definitions,name) && _sonnet_error(p.source,row.line,"duplicate linear project definition")
                definitions[name]=count;finalports=count
                totalnodes+=length(nodes)
                totalnodes<=projects.max_nodes || throw(ArgumentError("native project nodes exceed max_project_nodes"))
                n=BigInt(length(nodes))+currents+count
                # Sum all DEF workspaces bounds simultaneous nested solves.
                workspace+=BigInt(16)*(n*n+3n*count+4BigInt(count)^2+2(count+currents))+
                    BigInt(32)*(length(nodes)+1)+BigInt(16)*elements+BigInt(256)*ports^2
                compiled+=2048+BigInt(32)*length(nodes)+BigInt(64)*(count+BigInt(count)^2)+BigInt(1280)*elements
                empty!(nodes);currents=0;ports=0;elements=0
                continue
            end
            elements+=1
            totalelements+=1
            totalelements<=projects.max_elements || throw(ArgumentError("native project elements exceed max_project_elements"))
            localports=if kind in ("RES","CAP","IND")
                length(t)==4 || _sonnet_error(p.source,row.line,"invalid linear project lumped branch")
                for token in t[2:3];terminal(token,row);end
                parameter=split(t[4],'=';limit=2);expected=kind=="RES" ? "R" : kind=="CAP" ? "C" : "L"
                length(parameter)==2 && first(parameter)==expected || _sonnet_error(p.source,row.line,"invalid linear project lumped value")
                stored=_sonnet_circuit_literal(p,row,parameter[2],scales[kind])
                tokens=copy(t);tokens[4]=expected*"="*string(stored)
                scaledrecords[end]=SonnetRecord(row.line,tokens)
                1
            elseif occursin(r"^S\d+P$",kind)
                count=tryparse(Int,match(r"^S(\d+)P$",kind)[1])
                count!==nothing && 0<count<=projects.max_nodes && length(t)==count+2 ||
                    _sonnet_error(p.source,row.line,"invalid linear project data block")
                for token in t[2:count+1];terminal(token,row);end
                _,data=_sonnet_files_project_network!(projects,p,row,count)
                networks[abspath(joinpath(dirname(p.source),t[end]))]=data
                count
            elseif kind=="PRJ"
                fileindex=findfirst(token->endswith(lowercase(token),".son"),t)
                fileindex!==nothing && fileindex>=3 && length(t) in (fileindex+2,fileindex+5) ||
                    _sonnet_error(p.source,row.line,"linear PRJ requires literal path, port count and inheritance flag")
                count=tryparse(Int,t[fileindex+1])
                count!==nothing && count==fileindex-2 && count>0 && t[fileindex+2] in ("0","1") ||
                    _sonnet_error(p.source,row.line,"invalid linear PRJ port count or inheritance flag")
                length(t)==fileindex+2 || t[fileindex+3]=="DATE" ||
                    _sonnet_error(p.source,row.line,"unrepresented linear PRJ parameter bindings")
                for token in t[2:fileindex-1];terminal(token,row);end
                dependency=_sonnet_files_path(projects.root,p,t[fileindex],row)
                _sonnet_files_linear_project!(projects,dependency,count,depth+1)
                value,refs=projects.responses[dependency]
                project_responses[abspath(joinpath(dirname(p.source),t[fileindex]))]=planar_renormalize_s(value,refs,50.)
                count
            elseif haskey(definitions,kind)
                count=definitions[kind]
                length(t)==count+1 || _sonnet_error(p.source,row.line,"linear DEF invocation port count mismatch")
                for token in t[2:end];terminal(token,row);end
                count
            else
                _sonnet_error(p.source,row.line,"unsupported linear project statement $kind")
            end
            currents+=localports;ports=max(ports,localports)
        end
        isempty(nodes) && elements==0 && finalports>0 || throw(ArgumentError("linear project requires a final DEF network"))
        finalports==np || throw(ArgumentError("project port count disagrees with ordered model pins"))
        _sonnet_files_reserve!(projects.budget,compiled+BigInt(64)*np*np+
            sum(BigInt(32)*length(s) for s in values(project_responses);init=BigInt(0)))
        _enforce_payload_limit(_checked_payload_sum("linear project nested solve",projects.budget.used,workspace),
            projects.budget.limit,"linear project nested solve","max_bytes")
        frequency=projects.frequency;remaining=projects.budget.limit-projects.budget.used
        definition_cache=IdDict{PlanarCircuit,Matrix{ComplexF64}}()
        definition_response=(child,f)->begin
            f==frequency || throw(ArgumentError("staged linear project frequency mismatch"))
            get!(definition_cache,child) do
                solve_planar_circuit(child,f;max_bytes=remaining,floating_gauge=:auto).s
            end
        end
        project_response=(child,f)->begin
            f==frequency || throw(ArgumentError("staged linear project frequency mismatch"))
            project_responses[child]
        end
        network_response=(child,count)->begin
            data=networks[child]
            response=f->begin
                f==frequency || throw(ArgumentError("staged linear project frequency mismatch"))
                planar_network_response(data,f;z0=50.)
            end
            response,50.
        end
        # Compile the same validated SI values, avoiding a second unit rounding.
        units=copy(p.units);merge!(units,Dict("RES"=>"OH","CAP"=>"F","IND"=>"H"))
        effective=SonnetNetlistProject(p.source,units,p.frequency_scale,scaledrecords,p.records)
        circuit=sonnet_planar_circuit(effective;project_response,_network_response=network_response,
            _definition_response=definition_response)
        result=solve_planar_circuit(circuit,frequency;max_bytes=remaining,floating_gauge=:auto)
        projects.circuits[path]=circuit
        projects.responses[path]=(result.s,result.z0)
        return circuit
    finally
        delete!(projects.active,key)
    end
end
