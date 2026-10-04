# Exact finite-dyadic correlation PSD and bounded coupling compilation.
function _spice_dyadic_denbits(x::Float64)
    u=reinterpret(UInt64,abs(x));e=Int((u>>52)&0x7ff)
    m=u&0x000fffffffffffff
    iszero(m) && iszero(e) && return 0
    e==0 || (m|=UInt64(1)<<52)
    max(0,-(e==0 ? -1074 : e-1023-52)-trailing_zeros(m))
end
function _spice_psd_group!(members,edges,b,max_coupled_size,origin)
    n=length(members);n<=max_coupled_size || _spice_failure(origin,"coupled-inductor group exceeds max_coupled_size")
    denbits=maximum((_spice_dyadic_denbits(k) for (_,_,k) in edges);init=0)
    # Hadamard bounds every minor's bit length. Four full arrays cover old/new
    # GMP limbs, in-place elimination temporaries and garbage collection.
    bits=BigInt(n)*(denbits+ndigits(max(n,1);base=2)+3)+2
    bytes=_checked_payload_sum("coupled-inductor exact PSD workspace",
        _checked_array_payload_bytes(UInt8,4,BigInt(n)^2,64+cld(bits,8)),
        _checked_array_payload_bytes(Int,4,n))
    _spice_reserve!(b,bytes)
    try
        position=Dict(v=>i for (i,v) in enumerate(members));D=BigInt(1)<<denbits
        A=Matrix{BigInt}(undef,n,n)
        for i in 1:n,j in 1:n;A[i,j]=i==j ? D : BigInt(0);end
        for (a,c,k) in edges
            q=Rational{BigInt}(k);v=numerator(q)*(D÷denominator(q))
            i,j=position[a],position[c];A[i,j]=v;A[j,i]=v
        end
        previous=BigInt(1)
        for k in 1:n
            pivot=A[k,k]
            pivot>=0 || _spice_failure(origin,"joint coupled-inductor correlation matrix is not PSD")
            if iszero(pivot)
                all(i->iszero(A[k,i]),k+1:n) || _spice_failure(origin,"joint coupled-inductor correlation matrix is not PSD")
                continue
            end
            for i in k+1:n,j in i:n
                value,remainder=divrem(pivot*A[i,j]-A[i,k]*A[j,k],previous)
                iszero(remainder) || error("coupled-inductor exact PSD elimination lost divisibility")
                A[i,j]=value;A[j,i]=value
            end
            previous=pivot
        end
    finally
        b.used-=bytes
    end
end

function _spice_compile_couplings(pending,elements,b,max_coupled_size,max_coupling_pairs)
    result=_SpiceCoupling[];rows=Dict{Int,Vector{Tuple{Int,Float64}}}()
    isempty(pending) && return result,rows
    _spice_reserve!(b,256length(elements)+512length(pending))
    lookup=Dict(e.name=>i for (i,e) in enumerate(elements));pairs=Set{Tuple{Int,Int}}()
    parent=collect(eachindex(elements));edges=Tuple{Int,Int,Float64}[];origins=Dict{Int,_SpiceCard}()
    function root(i)
        while parent[i]!=i;parent[i]=parent[parent[i]];i=parent[i];end
        i
    end
    for (name,names,coefficient,card) in pending
        all(s->haskey(lookup,s)&&elements[lookup[s]].kind=='L',names) && length(unique(names))==length(names) ||
            _spice_failure(card,"K must name distinct local inductors")
        _spice_reserve!(b,256length(names)+512*binomial(BigInt(length(names)),2))
        ids=[lookup[s] for s in names]
        push!(result,_SpiceCoupling(name,ids,coefficient,card))
        for p in eachindex(ids),q in p+1:length(ids)
            i,j=minmax(ids[p],ids[q]);(i,j) in pairs && _spice_failure(card,"duplicate inductor coupling pair")
            length(pairs)<max_coupling_pairs || _spice_failure(card,"coupling pair count exceeds max_coupling_pairs")
            push!(pairs,(i,j));li,lj=elements[i].value,elements[j].value
            li>=0 && lj>=0 || _spice_failure(card,"coupled inductors require finite nonnegative L")
            iszero(coefficient) || iszero(li) || iszero(lj) || begin
                si,sj=sqrt(li),sqrt(lj)
                all(isfinite,(inv(si),inv(sj),coefficient*si,coefficient*sj,coefficient*si*sj)) &&
                    !iszero(coefficient*si) && !iszero(coefficient*sj) ||
                    _spice_failure(card,"coupled-inductor coefficients are not representable")
                push!(get!(rows,i,Tuple{Int,Float64}[]),(j,coefficient*sj))
                push!(get!(rows,j,Tuple{Int,Float64}[]),(i,coefficient*si))
                get!(origins,i,card);get!(origins,j,card)
                push!(edges,(i,j,coefficient));parent[root(i)]=root(j)
            end
        end
    end
    groups=Dict{Int,Vector{Int}}()
    for i in keys(rows);push!(get!(groups,root(i),Int[]),i);end
    grouped_edges=Dict{Int,Vector{Tuple{Int,Int,Float64}}}()
    for edge in edges;push!(get!(grouped_edges,root(edge[1]),Tuple{Int,Int,Float64}[]),edge);end
    for (group,members) in groups
        _spice_psd_group!(members,grouped_edges[group],b,max_coupled_size,origins[first(members)])
    end
    result,rows
end
