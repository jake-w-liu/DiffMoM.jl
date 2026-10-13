# Linear frequency-domain subset of ngspice47 manual sections6.1/6.2.
# These records retain two independent return coordinates and physical currents.
struct _SpiceLineSpec
    kind::Symbol
    z0::Float64
    td::Float64
    r::Float64
    l::Float64
    g::Float64
    c::Float64
    length::Float64
    model_name::String
    model_card::Union{Nothing,_SpiceCard}
    options::Vector{Pair{String,Float64}}
end

# Normalize each factor independently: neither omega nor intermediate products
# need be representable when the final constitutive coefficient is finite.
function _spice_line_product(factors...)
    value=1.0+0im;exponent=0
    for factor in factors
        factor=_circuit_line_value(factor,"transmission-line factor")
        scale=max(abs(real(factor)),abs(imag(factor)))
        iszero(scale) && return 0.0+0im
        _,shift=frexp(scale)
        value*=complex(ldexp(real(factor),-shift),ldexp(imag(factor),-shift))
        _,normal=frexp(max(abs(real(value)),abs(imag(value))))
        value=complex(ldexp(real(value),-normal),ldexp(imag(value),-normal))
        exponent+=shift+normal
    end
    result=complex(ldexp(real(value),exponent),ldexp(imag(value),exponent))
    # Derived components below the Float64 range round normally. Reject loss
    # of the entire nonzero coefficient, rather than a negligible cross term
    # (e.g. the O(f^2) real part of a near-DC shunt admittance).
    isfinite(result) && !iszero(result) ||
        throw(ArgumentError("transmission-line coefficient must remain finite and preserve nonzero values in ComplexF64"))
    result
end

function _spice_line_fields(fields,values,budget,card;model=false)
    numeric=model ? ("r","l","g","c","len","rel","abs","compactrel","compactabs") : ("z0","td","f","nl")
    flags=model ? ("nocontrol","steplimit","nosteplimit","lininterp","quadinterp","mixedinterp","truncnr","truncdontcut") : ()
    result=Dict{String,Float64}();index=1
    while index<=length(fields)
        key=_spice_identifier(fields[index])
        haskey(result,key) && _spice_failure(card,"duplicate transmission-line parameter $key")
        if index<length(fields) && fields[index+1]=="="
            index+2<=length(fields) || _spice_failure(card,"missing transmission-line parameter value")
            key in numeric || key in flags || _spice_failure(card,"unsupported transmission-line parameter $key")
            result[key]=_spice_expression(fields[index+2],values,budget);index+=3
        else
            key in flags || _spice_failure(card,"transmission-line numeric parameters require name=value")
            result[key]=1.;index+=1
        end
        key in flags && !(result[key] in (0.,1.)) && _spice_failure(card,"transient flag $key must be zero or one")
    end
    result
end

function _spice_ltra_spec(card,values,budget)
    tokens=card.tokens
    length(tokens)>=3 && lowercase(tokens[1])==".model" && lowercase(tokens[3])=="ltra" ||
        _spice_failure(card,"only linear .model LTRA is supported")
    _spice_reserve!(budget,1024+128length(tokens)+2ncodeunits(tokens[2]))
    fields=copy(tokens[4:end])
    if !isempty(fields) && startswith(fields[1],"(")
        endswith(fields[end],")") || _spice_failure(card,"unclosed LTRA parameter list")
        fields[1]=fields[1][2:end];fields[end]=fields[end][1:end-1]
        filter!(!isempty,fields)
    end
    parameters=_spice_line_fields(fields,values,budget,card;model=true)
    haskey(parameters,"len") || _spice_failure(card,"LTRA requires LEN")
    r,l,g,c=(get(parameters,key,0.) for key in ("r","l","g","c"))
    len=parameters["len"]
    all(x->isfinite(x) && x>=0,(r,l,g,c)) && len>0 ||
        _spice_failure(card,"LTRA requires nonnegative finite R/L/G/C and positive LEN")
    ((g==0 && c>0 && (r>0 || l>0)) || (r>0 && g>0 && l==0 && c==0)) ||
        _spice_failure(card,"LTRA represents only RLC/RC/LC/RG; general RLCG and degenerate models are unsupported")
    options=Pair{String,Float64}[]
    for (key,value) in parameters
        key in ("r","l","g","c","len") && continue
        value>=0 || _spice_failure(card,"negative transient LTRA control $key")
        key=="compactrel" && value>1 && _spice_failure(card,"COMPACTREL must be in [0,1]")
        push!(options,key=>value)
    end
    sort!(options;by=first)
    _spice_reserve!(budget,512+128length(options))
    _SpiceLineSpec(:LTRA,0.,0.,r,l,g,c,len,_spice_identifier(tokens[2]),card,options)
end

function _spice_t_spec(card,values,budget)
    length(card.tokens)>=8 || _spice_failure(card,"T requires four terminals, Z0 and TD or F")
    parameters=_spice_line_fields(card.tokens[6:end],values,budget,card)
    haskey(parameters,"z0") && parameters["z0"]>0 || _spice_failure(card,"T requires positive Z0")
    haskey(parameters,"td")==haskey(parameters,"f") && _spice_failure(card,"T requires exactly one TD or F")
    haskey(parameters,"nl") && !haskey(parameters,"f") && _spice_failure(card,"NL requires F")
    td=if haskey(parameters,"td")
        parameters["td"]
    else
        parameters["f"]>0 || _spice_failure(card,"T frequency must be positive")
        nl=get(parameters,"nl",.25)
        nl>=0 || _spice_failure(card,"T normalized length must be nonnegative")
        result=nl/parameters["f"]
        isfinite(result) && (iszero(nl) || !iszero(result)) || _spice_failure(card,"T delay is not representable in Float64")
        result
    end
    isfinite(td) && td>=0 || _spice_failure(card,"T delay must be finite and nonnegative")
    _spice_reserve!(budget,512)
    _SpiceLineSpec(:T,parameters["z0"],td,0.,0.,0.,0.,0.,"",nothing,Pair{String,Float64}[])
end

# N-conductor RLGC multiconductor line (.model x W MODELTYPE=RLGC).
# Matrices are per-unit-length, symmetric, and stored in full.
struct _SpiceCoupledSpec
    n::Int
    length::Float64
    r::Matrix{Float64}
    l::Matrix{Float64}
    g::Matrix{Float64}
    c::Matrix{Float64}
    model_name::String
    model_card::Union{Nothing,_SpiceCard}
end

function _spice_w_matrix(values,key,n,card)
    length(values)==n*(n+1)÷2 || _spice_failure(card,"W $key requires $(n*(n+1)÷2) lower-triangular entries")
    result=Matrix{Float64}(undef,n,n);index=1
    for row in 1:n, col in 1:row
        result[row,col]=result[col,row]=values[index];index+=1
    end
    all(isfinite,result) || _spice_failure(card,"W $key entries must be finite")
    scale=max(1.0,maximum(abs,result))
    minimum(eigvals(Hermitian(result)))>=-1e-9*scale ||
        _spice_failure(card,"W $key matrix must be positive semidefinite")
    result
end

function _spice_w_model_spec(card,values,budget)
    tokens=card.tokens
    length(tokens)>=3 && lowercase(tokens[1])==".model" && lowercase(tokens[3])=="w" ||
        _spice_failure(card,"W model card expected")
    _spice_reserve!(budget,1024+128length(tokens)+2ncodeunits(tokens[2]))
    fields=copy(tokens[4:end])
    if !isempty(fields) && startswith(fields[1],"(")
        endswith(fields[end],")") || _spice_failure(card,"unclosed W parameter list")
        fields[1]=fields[1][2:end];fields[end]=fields[end][1:end-1]
        filter!(!isempty,fields)
    end
    # Each parameter begins as key=value and consumes numeric entries until
    # the next key boundary; matrix sizes are checked after N is known.
    parameters=Dict{String,Vector{Float64}}();texts=Dict{String,String}()
    index=1
    while index<=length(fields)
        key=_spice_identifier(fields[index])
        index+1<=length(fields) && fields[index+1]=="=" ||
            _spice_failure(card,"W model parameters require name=value")
        haskey(parameters,key) && _spice_failure(card,"duplicate W model parameter $key")
        stop=index+2
        while stop<=length(fields) && !(stop+1<=length(fields) && fields[stop+1]=="=")
            stop+=1
        end
        raw=fields[index+2:stop-1]
        index=stop
        isempty(raw) && _spice_failure(card,"W model parameter $key requires a value")
        if key in ("n","fdie")
            length(raw)==1 || _spice_failure(card,"W $key requires one value")
            parameters[key]=[_spice_expression(raw[1],values,budget)]
        elseif key=="modeltype"
            length(raw)==1 || _spice_failure(card,"W MODELTYPE requires one value")
            texts[key]=lowercase(_spice_identifier(raw[1]))
        else
            key in ("lo","co","ro","go","rs","gd") || _spice_failure(card,"unsupported W model parameter $key")
            parameters[key]=[_spice_expression(token,values,budget) for token in raw]
        end
    end
    get(texts,"modeltype","rlgc")=="rlgc" ||
        _spice_failure(card,"only W MODELTYPE=RLGC is supported")
    for key in ("rs","gd")
        haskey(parameters,key) && (any(!iszero,parameters[key]) &&
            _spice_failure(card,"frequency-dependent W $key terms are unsupported"))
    end
    haskey(parameters,"n") || _spice_failure(card,"W model requires N")
    nraw=parameters["n"][1]
    isfinite(nraw) && nraw>=1 && nraw==round(nraw) || _spice_failure(card,"W N must be a positive integer")
    n=Int(nraw);n<=64 || _spice_failure(card,"W conductor count exceeds limit")
    haskey(parameters,"lo") && haskey(parameters,"co") ||
        _spice_failure(card,"W model requires Lo and Co")
    l=_spice_w_matrix(get(parameters,"lo",Float64[]),"Lo",n,card)
    c=_spice_w_matrix(get(parameters,"co",Float64[]),"Co",n,card)
    r=_spice_w_matrix(get(parameters,"ro",zeros(n*(n+1)÷2)),"Ro",n,card)
    g=_spice_w_matrix(get(parameters,"go",zeros(n*(n+1)÷2)),"Go",n,card)
    haskey(parameters,"fdie") && parameters["fdie"][1]<=0 &&
        _spice_failure(card,"W FDIE must be positive")
    _spice_reserve!(budget,512+128n*n)
    _SpiceCoupledSpec(n,0.,r,l,g,c,_spice_identifier(tokens[2]),card)
end

function _spice_line_scope(cards,values,inherited,budget)
    any(card->lowercase(card.tokens[1])==".model",cards) || return inherited
    _spice_reserve!(budget,512+128length(inherited)+128length(values))
    models=Dict{String,Any}(k=>v for (k,v) in inherited);context=copy(values);seen=Set{String}()
    for card in cards
        tag=lowercase(card.tokens[1])
        tag==".param" && _spice_setparams!(context,_spice_assignments(card.tokens[2:end]),budget)
        if tag==".model"
            name=_spice_identifier(card.tokens[2])
            name in seen && _spice_failure(card,"duplicate case-insensitive local model $name")
            push!(seen,name)
            models[name]=lowercase(card.tokens[3])=="w" ?
                _spice_w_model_spec(card,context,budget) : _spice_ltra_spec(card,context,budget)
        end
    end
    models
end
function _spice_global_models(lib,values,budget)
    lib.dialect===:spice || return Dict{String,Any}()
    count=0;inside=false
    for card in lib.records
        tag=lowercase(card.tokens[1])
        tag==".subckt" && (inside=true)
        tag==".ends" && (inside=false)
        !inside && tag==".model" && (count+=1)
    end
    iszero(count) && return Dict{String,Any}()
    _spice_reserve!(budget,_checked_array_payload_bytes(UInt8,16,count))
    cards=_SpiceCard[];inside=false
    sizehint!(cards,count)
    for card in lib.records
        tag=lowercase(card.tokens[1])
        tag==".subckt" && (inside=true)
        tag==".ends" && (inside=false)
        !inside && tag==".model" && push!(cards,card)
    end
    _spice_line_scope(cards,values,Dict{String,Any}(),budget)
end

function _spice_ltra_coefficients(spec,f)
    series=complex(spec.r,real(_spice_line_product(2pi,f,spec.l)))
    shunt=complex(spec.g,real(_spice_line_product(2pi,f,spec.c)))
    left=sqrt(series);right=sqrt(shunt)
    gl=_spice_line_product(left,right,spec.length)
    if abs(gl)<=.125
        # Finite ABCD weak equations avoid loss of total series resistance as
        # Zc tends to infinity near DC. No frequency snapping is performed.
        gl2=gl*gl
        sinhc=abs(gl)<sqrt(eps(Float64)) ? 1+gl2/6 : sinh(gl)/gl
        return (;small=true,a=cosh(gl),b=_spice_line_product(series,spec.length,sinhc),
            c=_spice_line_product(shunt,spec.length,sinhc),z=0.0+0im,gl)
    end
    z=_circuit_line_value(left/right,"LTRA characteristic impedance";nonzero=true)
    (;small=false,a=0.0+0im,b=0.0+0im,c=0.0+0im,z,gl)
end

# x*coth(x) and x*csch(x) are analytic at x=0 (both limit to one); stable
# exponential forms apply for Re(x)>=0, which sqrt-branch gamma guarantees.
function _spice_coupled_hyperbolic(x)
    ax=abs(x)
    if ax<=1e-3
        x2=x*x
        return 1+x2/3-x2*x2/45,1-x2/6+7x2*x2/360
    end
    e2=exp(-2x)
    return x*(1+e2)/(1-e2),2x*exp(-x)/(1-e2)
end

# Frequency response of a uniform N-conductor RLGC section. With Z=R+jwL,
# Y=G+jwC and ZY=V*diag(gamma^2)*V^-1 the modal decomposition reduces each
# conductor mode to a scalar line; the physical 2N-port admittance is
# Z^-1*[A -B; -B A] where A=V*diag(gamma*coth) V^-1, B likewise for csch.
# Both gamma*coth(gamma*len) and gamma*csch(gamma*len) are analytic in
# gamma^2, so the formula stays regular for electrically short sections.
function _spice_coupled_line_y(spec,f)
    n=spec.n;omega=2pi*f
    Z=spec.r.+1im.*(omega.*spec.l)
    Y=spec.g.+1im.*(omega.*spec.c)
    product=Matrix{ComplexF64}(Z*Matrix{ComplexF64}(Y))
    factor=eigen(product)
    gamma=sqrt.(factor.values);V=factor.vectors
    inverse=try
        inv(V)
    catch
        throw(ArgumentError("coupled W line modal decomposition is defective at $f Hz"))
    end
    # A defective or numerically singular eigenvector matrix must not
    # silently produce a wrong admittance; verify the factorization.
    scale=max(opnorm(product),eps(Float64))
    opnorm(V*Diagonal(factor.values)*inverse-product)<=1e-7*scale ||
        throw(ArgumentError("coupled W line modal decomposition is ill-conditioned at $f Hz"))
    a=Vector{ComplexF64}(undef,n);b=Vector{ComplexF64}(undef,n)
    for k in 1:n
        a[k],b[k]=_spice_coupled_hyperbolic(gamma[k]*spec.length)./spec.length
    end
    Zi=try
        inv(Z)
    catch
        throw(ArgumentError("coupled W line has a singular series-impedance matrix at $f Hz"))
    end
    left=Zi*V*Diagonal(a)*inverse
    right=Zi*V*Diagonal(b)*inverse
    result=[left -right;-right left]
    all(isfinite,result) ||
        throw(ArgumentError("coupled W line response must remain finite in ComplexF64 at $f Hz"))
    result
end

# W in1..inN refin out1..outN refout N=k L=len RLGCMODEL=name
function _spice_w_instance(card,values,models,budget)
    t=card.tokens
    split=findfirst(==("="),t)
    split===nothing && _spice_failure(card,"W requires N, L and RLGCMODEL parameters")
    nodes=t[2:split-2];parameters=t[split-1:end]
    length(parameters)%3==0 || _spice_failure(card,"W parameters require name=value assignments")
    entries=Dict{String,String}()
    for k in 1:3:length(parameters)
        parameters[k+1]=="=" || _spice_failure(card,"W parameters require name=value")
        key=lowercase(_spice_identifier(parameters[k]))
        haskey(entries,key) && _spice_failure(card,"duplicate W parameter $key")
        key in ("n","l","rlgcmodel","tablemodel","fsmodel","smodel","rlgcfile") ||
            _spice_failure(card,"unsupported W parameter $key")
        entries[key]=parameters[k+2]
    end
    for key in ("tablemodel","fsmodel","smodel","rlgcfile")
        haskey(entries,key) && _spice_failure(card,"unsupported W model source $key; only RLGCMODEL is supported")
    end
    haskey(entries,"n") && haskey(entries,"l") && haskey(entries,"rlgcmodel") ||
        _spice_failure(card,"W requires N, L and RLGCMODEL")
    nraw=_spice_expression(entries["n"],values,budget)
    isfinite(nraw) && nraw>=1 && nraw==round(nraw) || _spice_failure(card,"W N must be a positive integer")
    n=Int(nraw);n<=64 || _spice_failure(card,"W conductor count exceeds limit")
    length(nodes)==2n+2 || _spice_failure(card,"W requires $(2n+2) terminal nodes")
    len=_spice_expression(entries["l"],values,budget)
    isfinite(len) && len>0 || _spice_failure(card,"W length must be finite and positive")
    key=_spice_identifier(entries["rlgcmodel"])
    haskey(models,key) && models[key] isa _SpiceCoupledSpec ||
        _spice_failure(card,"undefined W RLGC model $key")
    spec=models[key]
    spec.n==n || _spice_failure(card,"W instance N does not match model $key")
    _spice_reserve!(budget,512+128n*n)
    (nodes=nodes,spec=_SpiceCoupledSpec(n,len,spec.r,spec.l,spec.g,spec.c,key,spec.model_card))
end

function _spice_line_stamp!(matrix,row,terminals,spec,f)
    if spec.kind===:T
        gl=_spice_line_product(1im,2pi,f,spec.td)
        _circuit_line_stamp!(matrix,row,terminals,spec.z0+0im,gl)
    else
        coefficients=_spice_ltra_coefficients(spec,f)
        if coefficients.small
            _circuit_voltage_stamp!(matrix,row,terminals[1],1.)
            _circuit_voltage_stamp!(matrix,row,terminals[2],-coefficients.a)
            matrix[row,row+1]=coefficients.b
            matrix[row+1,row]=1.;matrix[row+1,row+1]=coefficients.a
            _circuit_voltage_stamp!(matrix,row+1,terminals[2],-coefficients.c)
        else
            _circuit_line_stamp!(matrix,row,terminals,coefficients.z,coefficients.gl)
        end
    end
    nothing
end
