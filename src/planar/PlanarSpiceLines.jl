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

function _spice_line_scope(cards,values,inherited,budget)
    any(card->lowercase(card.tokens[1])==".model",cards) || return inherited
    _spice_reserve!(budget,512+128length(inherited)+128length(values))
    models=copy(inherited);context=copy(values);seen=Set{String}()
    for card in cards
        tag=lowercase(card.tokens[1])
        tag==".param" && _spice_setparams!(context,_spice_assignments(card.tokens[2:end]),budget)
        if tag==".model"
            name=_spice_identifier(card.tokens[2])
            name in seen && _spice_failure(card,"duplicate case-insensitive local model $name")
            push!(seen,name);models[name]=_spice_ltra_spec(card,context,budget)
        end
    end
    models
end
function _spice_global_models(lib,values,budget)
    lib.dialect===:spice || return Dict{String,_SpiceLineSpec}()
    count=0;inside=false
    for card in lib.records
        tag=lowercase(card.tokens[1])
        tag==".subckt" && (inside=true)
        tag==".ends" && (inside=false)
        !inside && tag==".model" && (count+=1)
    end
    iszero(count) && return Dict{String,_SpiceLineSpec}()
    _spice_reserve!(budget,_checked_array_payload_bytes(UInt8,16,count))
    cards=_SpiceCard[];inside=false
    sizehint!(cards,count)
    for card in lib.records
        tag=lowercase(card.tokens[1])
        tag==".subckt" && (inside=true)
        tag==".ends" && (inside=false)
        !inside && tag==".model" && push!(cards,card)
    end
    _spice_line_scope(cards,values,Dict{String,_SpiceLineSpec}(),budget)
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
