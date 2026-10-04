function sonnet_planar_circuit(p::SonnetNetlistProject;project_response=nothing,
        _network_response=nothing,_definition_response=nothing)
    resscale=get(Dict("OH"=>1.,"OHMS"=>1.,"KOH"=>1e3,"MOH"=>1e6),
        get(p.units,"RES","OH"),NaN)
    capscale=get(Dict("F"=>1.,"PF"=>1e-12,"FF"=>1e-15,"NF"=>1e-9,
        "UF"=>1e-6),get(p.units,"CAP",""),NaN)
    indscale=get(Dict("H"=>1.,"NH"=>1e-9,"PH"=>1e-12,"UH"=>1e-6,
        "MH"=>1e-3),get(p.units,"IND",""),NaN)
    all(isfinite,(resscale,capscale,indscale)) || throw(ArgumentError("unsupported circuit RLC units"))
    definitions=Dict{String,PlanarCircuit}(); pending=SonnetRecord[]; final=nothing
    for record in p.circuit
        t=record.tokens
        m=match(r"^DEF(\d+)P$",t[1])
        if m===nothing
            push!(pending,record); continue
        end
        np=parse(Int,m[1]); external=parse.(Int,t[2:np+1]); name=t[np+2]
        haskey(definitions,name) && throw(ArgumentError("duplicate native network definition $name"))
        # All element node numbers precede the first noninteger token.
        nodes=copy(external)
        for row in pending
            for token in row.tokens[2:end]
                value=tryparse(Int,token)
                value===nothing && break
                push!(nodes,value)
            end
        end
        all(>=(0),nodes) || _sonnet_error(p.source,record.line,"negative native circuit node label")
        labels=sort!(unique!(filter!(!=(0),nodes)))
        nodemap=Dict(label=>index for (index,label) in enumerate(labels))
        nodemap[0]=0
        circuit=PlanarCircuit(length(labels),[nodemap[label] for label in external];z0=parse(Float64,t[end]))
        for row in pending
            q=row.tokens; kind=q[1]
            if kind in ("RES","CAP","IND")
                length(q)==4 || _sonnet_error(p.source,row.line,"invalid native lumped branch")
                a,b=(nodemap[parse(Int,token)] for token in q[2:3])
                parameter=split(q[4],'=';limit=2)
                expected=kind=="RES" ? "R" : kind=="CAP" ? "C" : "L"
                length(parameter)==2 && parameter[1]==expected ||
                    _sonnet_error(p.source,row.line,"invalid native lumped value")
                # Convert the source literal and its units before rounding to
                # storage. A nonzero native C/L must never become an exact
                # open/short solely because its SI value underflows Float64.
                scale=kind=="RES" ? resscale : kind=="CAP" ? capscale : indscale
                stored=_sonnet_circuit_literal(p,row,parameter[2],scale)
                kind=="RES" ? circuit_add_rlc!(circuit,a,b;r=stored) :
                    kind=="CAP" ? circuit_add_rlc!(circuit,a,b;c=stored) :
                    circuit_add_rlc!(circuit,a,b;l=stored)
            elseif occursin(r"^S\d+P$",kind)
                localports=parse(Int,match(r"^S(\d+)P$",kind)[1])
                length(q)==localports+2 || _sonnet_error(p.source,row.line,"invalid native data block")
                terminals=[nodemap[parse(Int,token)] for token in q[2:localports+1]]
                path=abspath(joinpath(dirname(p.source),q[end]))
                response,z0=_network_response===nothing ? _sonnet_touchstone_response(path,localports) :
                    _network_response(path,localports)
                circuit_add_network!(circuit,terminals,response;z0=z0)
            elseif kind=="PRJ"
                project_response===nothing && throw(ArgumentError("native PRJ requires an explicit calibrated project_response callback"))
                fileindex=findfirst(token->endswith(lowercase(token),".son"),q)
                fileindex===nothing && _sonnet_error(p.source,row.line,"PRJ lacks a .son file")
                fileindex+2<=length(q) || _sonnet_error(p.source,row.line,"PRJ lacks port count/reference data")
                count=parse(Int,q[fileindex+1])
                count>0 && fileindex-2 in (count,count+1) ||
                    _sonnet_error(p.source,row.line,"PRJ requires its declared pins and an optional common-return node")
                reference=fileindex-2==count+1 ? nodemap[parse(Int,q[fileindex-1])] : 0
                terminals=[(nodemap[parse(Int,token)],reference) for token in q[2:count+1]]
                path=abspath(joinpath(dirname(p.source),q[fileindex]))
                callback=let path=path,provider=project_response
                    f->provider(path,f)
                end
                circuit_add_network!(circuit,terminals,callback;z0=50.)
            elseif haskey(definitions,kind)
                child=definitions[kind]
                length(q)==length(child.ports)+1 || _sonnet_error(p.source,row.line,"subnetwork terminal count mismatch")
                terminals=[nodemap[parse(Int,token)] for token in q[2:end]]
                callback=let child=child,provider=_definition_response
                    f->provider===nothing ? solve_planar_circuit(child,f).s : provider(child,f)
                end
                circuit_add_network!(circuit,terminals,callback;z0=child.z0)
            else
                _sonnet_error(p.source,row.line,"unsupported circuit statement $kind")
            end
        end
        definitions[name]=circuit; final=circuit; empty!(pending)
    end
    isempty(pending) || throw(ArgumentError("native circuit statements follow final network definition"))
    final===nothing && throw(ArgumentError("native circuit has no network definition"))
    return final
end
