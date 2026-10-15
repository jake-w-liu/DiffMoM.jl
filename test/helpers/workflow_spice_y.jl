# Shared independent SPICE oracle; including this file never runs tests.
if !isdefined(@__MODULE__, :_workflow_spice_y)
# Re-stamp emitted SPICE independently, including controlled current and
# voltage sources and ideal sensing sources for affine capacitance.
function _workflow_spice_y(path,f,nports;common_mode=0.,drive_voltage=1.)
    reference=last(split(only(filter(line->startswith(line,".subckt "),readlines(path)))))
    isground(node)=lowercase(node) in ("gnd","0")
    lines = [split(strip(line)) for line in eachline(path)
        if !isempty(strip(line)) && !(first(strip(line)) in ('*','.'))]
    names = Set{String}()
    for words in lines
        for node in words[2:3]
            isground(node) || push!(names,node)
        end
        if first(words[1]) in ('G','E')
            for node in words[4:5]
                isground(node) || push!(names,node)
            end
        end
    end
    nodes = sort!(collect(names))
    ids = Dict(node=>k for (k,node) in enumerate(nodes))
    nodeid(node) = get(ids,node,0)
    sources = [words for words in lines if first(words[1]) in ('V','E')]
    sense = Dict(words[1]=>length(nodes)+k for (k,words) in enumerate(sources))
    N = length(nodes)+length(sources)+nports+1
    M = zeros(ComplexF64,N,N)
    rhs = zeros(ComplexF64,N,nports)
    function entry(a,b,value)
        a>0 && b>0 && (M[a,b]+=value)
    end
    for words in lines
        kind = first(words[1])
        p,m = nodeid(words[2]),nodeid(words[3])
        if kind in ('R','C')
            y = kind=='R' ? inv(parse(Float64,words[4])) :
                2pi*1im*f*parse(Float64,words[4])
            entry(p,p,y);entry(p,m,-y);entry(m,p,-y);entry(m,m,y)
        elseif kind=='G'
            cp,cm,g = nodeid(words[4]),nodeid(words[5]),parse(Float64,words[6])
            entry(p,cp,g);entry(p,cm,-g);entry(m,cp,-g);entry(m,cm,g)
        elseif kind=='F'
            id,g = sense[words[4]],parse(Float64,words[5])
            entry(p,id,g);entry(m,id,-g)
        else
            id = sense[words[1]]
            entry(p,id,1);entry(m,id,-1);entry(id,p,1);entry(id,m,-1)
            if kind=='E'
                cp,cm,g = nodeid(words[4]),nodeid(words[5]),parse(Float64,words[6])
                entry(id,cp,-g);entry(id,cm,g)
            end
        end
    end
    drive = (N-nports):(N-1)
    ref=nodeid(reference);ref>0 || throw(ArgumentError("SPICE reference formal node is global ground"))
    for p in 1:nports
        node = nodeid("p$p")
        entry(node,drive[p],1);entry(ref,drive[p],-1)
        entry(drive[p],node,1);entry(drive[p],ref,-1)
        rhs[drive[p],p]=drive_voltage
    end
    entry(ref,N,1);entry(N,ref,1);rhs[N,:].=common_mode
    return -(M \ rhs)[drive,:]
end
end
