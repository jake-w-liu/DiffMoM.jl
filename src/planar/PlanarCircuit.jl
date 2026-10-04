# Circuit/EM coupling using modified nodal analysis. Network blocks keep
# their native Y, Z, or power-wave S equations, so exact thru/short/open
# S blocks do not require a singular conversion to admittance.

export PlanarCircuit, PlanarCircuitResult, circuit_add_rlc!, circuit_add_network!
export circuit_add_line!, circuit_add_transformer!, solve_planar_circuit
export planar_circuit_sparams
export planar_port_incidence,circuit_add_em!

abstract type _PlanarCircuitElement end
struct _CircuitRLC <: _PlanarCircuitElement
    terminals::Vector{Tuple{Int,Int}}
    r::Union{Nothing,Float64}
    l::Union{Nothing,Float64}
    c::Union{Nothing,Float64}
    topology::Symbol
end
struct _CircuitNetwork{F} <: _PlanarCircuitElement
    terminals::Vector{Tuple{Int,Int}}
    response::F
    format::Symbol
    z0::Any
end
struct _CircuitLine{Z,G} <: _PlanarCircuitElement
    terminals::Vector{Tuple{Int,Int}}
    zc::Z
    gl::G
end
struct _CircuitTransformer <: _PlanarCircuitElement
    terminals::Vector{Tuple{Int,Int}}
    ratio::Float64
end

"""Circuit with nodes `1:nnodes` and ground node `0`. `ports` lists
external `(positive,negative)` terminal pairs, or positive nodes referenced
to ground. `z0` is a scalar, per-port vector or frequency provider with
finite positive real parts. Complex references use Kurokawa power waves.
Add RLC branches, line sections, ideal transformers and measured or
EM network blocks, then call [`solve_planar_circuit`](@ref)."""
mutable struct PlanarCircuit
    nnodes::Int
    ports::Vector{Tuple{Int,Int}}
    z0::Any
    elements::Vector{_PlanarCircuitElement}
end

function _circuit_terminals(nnodes::Int, terminals)
    result = Tuple{Int,Int}[]
    for pair in terminals
        p, m = pair isa Integer ? (Int(pair),0) : (Int(pair[1]),Int(pair[2]))
        0 <= p <= nnodes && 0 <= m <= nnodes && p != m ||
            throw(ArgumentError("circuit terminals must be distinct nodes in 0:$nnodes"))
        push!(result,(p,m))
    end
    isempty(result) && throw(ArgumentError("at least one circuit terminal pair is required"))
    return result
end

_circuit_z0(z0,n::Int;freq=nothing)=_planar_reference_values(z0,n;freq)

"""Signed node/port incidence matrix D for `(positive,negative)` terminal
pairs. Node zero is omitted. Physical port voltages are `transpose(D)*v`,
and nodal current injection is `D*Iport`; an admittance stamps as
`D*Y*transpose(D)`. This retains every represented source mode and does
not create independent common modes absent from the EM source set."""
function planar_port_incidence(nnodes::Integer,terminals;
        max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    nnodes>=1 && nnodes<=typemax(Int) || throw(ArgumentError("invalid circuit node count"))
    _enforce_payload_limit(_checked_payload_sum("port incidence",
        _checked_array_payload_bytes(Float64,nnodes,length(terminals)),
        _checked_array_payload_bytes(Int,2,length(terminals))),max_bytes,"port incidence","max_bytes")
    pairs=_circuit_terminals(Int(nnodes),terminals);D=zeros(Float64,nnodes,length(pairs))
    for (k,(p,m)) in enumerate(pairs)
        p>0 && (D[p,k]=1);m>0 && (D[m,k]=-1)
    end
    return D
end

"""Attach a solved EM network to explicit positive/negative circuit
nodes, preserving its native power-wave reference impedances. Common-mode
information is retained exactly when supplied by the solved source set.
The terminal pairs specify wiring; node-zero coordinate choices do not
infer additional physical source modes."""
function circuit_add_em!(circuit::PlanarCircuit,terminals,result;z0=nothing)
    hasproperty(result,:s) || throw(ArgumentError("EM result must provide its scattering matrix"))
    refs=z0===nothing ? (hasproperty(result,:z0) ? result.z0 :
        hasproperty(result,:problem) ? [p.z0 for p in result.problem.ports] :
        throw(ArgumentError("EM reference impedances must be supplied"))) : z0
    return circuit_add_network!(circuit,terminals,result.s;format=:s,z0=refs)
end

function PlanarCircuit(nnodes::Integer, ports; z0=50.0)
    nnodes >= 1 || throw(ArgumentError("nnodes must be positive"))
    terminals = _circuit_terminals(Int(nnodes),ports)
    return PlanarCircuit(Int(nnodes),terminals,_planar_store_references(z0,length(terminals)),
        _PlanarCircuitElement[])
end

function _circuit_stored_real(value::Real,label::AbstractString)
    stored=Float64(value)
    isfinite(stored) && (iszero(value) || !iszero(stored)) ||
        throw(ArgumentError("$label must remain finite and preserve nonzero values in Float64"))
    return stored
end

"""Add a series or parallel R/L/C branch between nodes `p` and `m`.
Omit unused values (`nothing`). Units are Ω, H and F. Explicit zero R/L
models an ideal short, zero C an open; the DC limits are supported. Values
must remain finite and preserve nonzero values in stored Float64 units."""
function circuit_add_rlc!(circuit::PlanarCircuit, p::Integer, m::Integer;
        r::Union{Nothing,Real}=nothing, l::Union{Nothing,Real}=nothing,
        c::Union{Nothing,Real}=nothing, topology::Symbol=:series)
    topology in (:series,:parallel) || throw(ArgumentError("topology must be :series or :parallel"))
    all(x -> x === nothing || (isfinite(x) && x >= 0),(r,l,c)) ||
        throw(ArgumentError("R/L/C values must be finite and nonnegative"))
    all(isnothing,(r,l,c)) && throw(ArgumentError("RLC branch has no elements"))
    convert_value(x) = x === nothing ? nothing : _circuit_stored_real(x,"R/L/C value")
    values=map(convert_value,(r,l,c))
    push!(circuit.elements,_CircuitRLC(_circuit_terminals(circuit.nnodes,[(p,m)]),
        values...,topology))
    return circuit
end

"""Add a network block whose local ports are the given terminal pairs
(or ground-referenced node numbers). `response` is a matrix or a callable
`f_hz -> matrix`; `format` is `:s`, `:y` or `:z`. S blocks use Kurokawa
reference impedances `z0`, numeric or frequency providers with positive
real parts. Native constitutive equations retain
exact shorts/thrus/opens and accept nonreciprocal or active network blocks."""
function circuit_add_network!(circuit::PlanarCircuit, terminals,response;
        format::Symbol=:s,z0=50.0)
    format in (:s,:y,:z) || throw(ArgumentError("network format must be :s, :y or :z"))
    ports = _circuit_terminals(circuit.nnodes,terminals)
    response isa AbstractMatrix && size(response)!=(length(ports),length(ports)) &&
        throw(ArgumentError("network matrix must match its local ports before conversion"))
    stored = response isa AbstractMatrix ? Matrix{ComplexF64}(response) : response
    if stored isa AbstractMatrix
        size(stored) == (length(ports),length(ports)) && all(isfinite,stored) ||
            throw(ArgumentError("network matrix must be finite and match its local ports"))
    end
    push!(circuit.elements,_CircuitNetwork(ports,stored,format,_planar_store_references(z0,length(ports))))
    return circuit
end

"""Add a two-port transmission line. `zc` [Ω] and `gl = gamma*length`
may be numbers or frequency-dependent callables. `terminals` gives its
left and right terminal pairs, each ordered signal then return. Numeric
values must remain finite and preserve nonzero components in ComplexF64;
`zc` must be nonzero. Travelling-wave constitutive equations retain exact
zero-length/half-wave constraints and avoid growing hyperbolic coefficients
for attenuating lines. Providers receive the stored finite frequency."""
function circuit_add_line!(circuit::PlanarCircuit,terminals,zc,gl)
    stored_z=zc isa Number ? _circuit_line_value(zc,"line characteristic impedance";nonzero=true) : zc
    stored_g=gl isa Number ? _circuit_line_value(gl,"line electrical length") : gl
    ports = _circuit_terminals(circuit.nnodes,terminals)
    length(ports) == 2 || throw(ArgumentError("line needs two terminal pairs"))
    push!(circuit.elements,_CircuitLine(ports,stored_z,stored_g))
    return circuit
end

function _circuit_line_value(value,label;nonzero=false)
    value isa Number && isfinite(value) || throw(ArgumentError("$label must be a finite number"))
    stored=complex(_circuit_stored_real(real(value),label),_circuit_stored_real(imag(value),label))
    nonzero && iszero(stored) && throw(ArgumentError("$label must be nonzero"))
    return stored
end

"""Add an ideal transformer with voltage ratio `V1/V2 = ratio` and
current relation `ratio*I1 + I2 = 0`. Both ports may have floating returns.
The stored Float64 ratio must remain finite and nonzero."""
function circuit_add_transformer!(circuit::PlanarCircuit,terminals,ratio::Real)
    isfinite(ratio) && !iszero(ratio) || throw(ArgumentError("transformer ratio must be finite and nonzero"))
    stored=_circuit_stored_real(ratio,"transformer ratio")
    ports = _circuit_terminals(circuit.nnodes,terminals)
    length(ports) == 2 || throw(ArgumentError("transformer needs two terminal pairs"))
    push!(circuit.elements,_CircuitTransformer(ports,stored))
    return circuit
end

"""Circuit result: `s` is the external power-wave response, `voltages`
are node potentials, `currents` are external currents into the circuit,
and `y` is the port admittance when it exists. An ideal thru/short can
have a finite S response and no finite Y representation (`y=nothing`).
`gauge_nodes` lists coordinate anchors selected for floating circuit
components; their zero potential does not add a physical ground path."""
struct PlanarCircuitResult
    freq::Float64
    s::Matrix{ComplexF64}
    y::Union{Nothing,Matrix{ComplexF64}}
    voltages::Matrix{ComplexF64}
    currents::Matrix{ComplexF64}
    gauge_nodes::Vector{Int}
    z0::Vector{ComplexF64}
end
PlanarCircuitResult(f,s,y,v,i)=PlanarCircuitResult(f,s,y,v,i,Int[])
PlanarCircuitResult(f,s,y,v,i,g)=PlanarCircuitResult(f,s,y,v,i,g,fill(50.0+0im,size(s,1)))

_circuit_gauge_terminals(element)=element.terminals
function _circuit_uniform_y_null(H)
    n=size(H,1)
    # Exact algebraic closure is deliberate: a small finite common-mode
    # conductance is a physical path and must not be removed by tolerance.
    for p in 1:n
        iszero(sum(H[p,q] for q in 1:n)) && iszero(sum(H[q,p] for q in 1:n)) || return false
    end
    return true
end

function _circuit_gauge_nodes(circuit,uniform_y_null=falses(length(circuit.elements)))
    n=circuit.nnodes;parent=collect(0:n);used=falses(n)
    function root(node)
        while parent[node+1]!=node
            parent[node+1]=parent[parent[node+1]+1];node=parent[node+1]
        end
        return node
    end
    function join(pair)
        a,b=pair
        a>0 && (used[a]=true);b>0 && (used[b]=true)
        ra,rb=root(a),root(b);parent[ra+1]=rb
    end
    for pair in circuit.ports;join(pair);end
    for (k,element) in enumerate(circuit.elements)
        pairs=_circuit_gauge_terminals(element)
        if uniform_y_null[k] && all(p->p[2]==pairs[1][2],pairs)
            # A nodal Y block with zero row/column sums has no current
            # path to its shared reference. Its signal nodes couple, and
            # the reference keeps a separate arbitrary coordinate.
            reference=pairs[1][2];reference>0 && (used[reference]=true)
            signal=pairs[1][1]
            for pair in pairs;join((signal,pair[1]));end
        elseif uniform_y_null[k] && all(p->p[1]==pairs[1][1],pairs)
            reference=pairs[1][1];reference>0 && (used[reference]=true)
            signal=pairs[1][2]
            for pair in pairs;join((signal,pair[2]));end
        else
            for pair in pairs;join(pair);end
        end
    end
    # Unused numeric IDs are isolated components too. Their arbitrary
    # coordinates may be anchored without changing any represented mode.
    ground=root(0);anchors=Dict{Int,Int}()
    for node in 1:n
        group=root(node);group==ground && continue
        get!(anchors,group,node)
    end
    return sort!(collect(values(anchors)))
end

_circuit_stamp_extra!(M,row,element,f,branch_rows)=throw(ArgumentError("unsupported circuit element"))
_circuit_owned_model(element)=nothing
function _circuit_owned_model_payload(circuit)
    seen=Base.IdSet{Any}();payload=0
    for element in circuit.elements
        model=_circuit_owned_model(element)
        model===nothing || model in seen || begin
            push!(seen,model)
            payload=_checked_payload_sum("circuit owned models",payload,model.payload)
        end
    end
    payload
end

function _circuit_network_workspace_payload(circuit)
    largest=0
    for element in circuit.elements
        element isa _CircuitNetwork || continue
        n=length(element.terminals)
        # A provider-returned plane and the owned ComplexF64 snapshot can
        # coexist. Constitutive coefficients are stamped directly below.
        largest=max(largest,_checked_array_payload_bytes(ComplexF64,2,n,n))
    end
    return largest
end

@inline _circuit_value(value::Number,f) = value
@inline _circuit_value(value,f) = value(f)

function _circuit_voltage_stamp!(M,row,terminals,coefficient)
    p,m = terminals
    p > 0 && (M[row,p] += coefficient)
    m > 0 && (M[row,m] -= coefficient)
    return nothing
end
function _circuit_current_stamp!(M,column,terminals,coefficient)
    p,m = terminals
    p > 0 && (M[p,column] += coefficient)
    m > 0 && (M[m,column] -= coefficient)
    return nothing
end

function _circuit_line_stamp!(M,p,terminals,z,g)
    # Currents enter both terminals. For attenuation >= 0:
    # V1-z*I1 = exp(-g)*(V2+z*I2), and its reciprocal partner.
    # For gain, multiply those equations by exp(g) instead. The retained
    # exponential has magnitude <= 1 in either case; no ABCD determinant
    # subtraction or growing cosh/sinh coefficients enter the system.
    reverse=real(g)<0
    w=exp(reverse ? g : -g)
    scale=max(1.,abs(real(z)),abs(imag(z)))
    a=inv(scale);b=z/scale;q=p+1
    if reverse
        _circuit_voltage_stamp!(M,p,terminals[1],w*a)
        _circuit_voltage_stamp!(M,p,terminals[2],-a)
        M[p,p]=-w*b;M[p,q]=-b
        _circuit_voltage_stamp!(M,q,terminals[2],w*a)
        _circuit_voltage_stamp!(M,q,terminals[1],-a)
        M[q,q]=-w*b;M[q,p]=-b
    else
        _circuit_voltage_stamp!(M,p,terminals[1],a)
        _circuit_voltage_stamp!(M,p,terminals[2],-w*a)
        M[p,p]=-b;M[p,q]=-w*b
        _circuit_voltage_stamp!(M,q,terminals[2],a)
        _circuit_voltage_stamp!(M,q,terminals[1],-w*a)
        M[q,q]=-b;M[q,p]=-w*b
    end
    return nothing
end

function _circuit_scattering_stamp!(M,ids,terminals,H,refs::Vector{ComplexF64})
    roots=_planar_reference_roots(refs)
    n=length(terminals)
    for p in 1:n,q in 1:n
        h=H[p,q];root=roots[q]
        a=((p==q ? 1.0 : 0.0)-h)/root
        b=-(h*refs[q]+(p==q ? conj(refs[q]) : 0.0))/root
        _circuit_voltage_stamp!(M,ids[p],terminals[q],a)
        M[ids[p],ids[q]]+=b
    end
    return nothing
end

"""Solve the combined circuit/EM N-port at a nonnegative frequency [Hz]
using modified nodal analysis and power-wave boundary excitations. A
single LU handles every external port. `max_bytes` limits dense workspace
payload before allocating or evaluating frequency-dependent blocks.
The largest sequential internal network snapshot is reserved even when
it has more local ports than the external circuit. Native constitutive
coefficients are stamped directly without dense A/B intermediates.
Frequency must remain finite and preserve nonzero values in Float64;
response and reference providers receive that stored frequency.
`floating_gauge=:auto` chooses one voltage coordinate per ungrounded
terminal-connected component. Port differences and currents remain
physical; no reference conductor is clamped to global ground. The
same coordinate rule covers isolated allocated nodes, retaining their
unexcited potential as zero without inventing a circuit response mode.
default `:reject` retains the singular-circuit rejection contract."""
function solve_planar_circuit(circuit::PlanarCircuit,f::Real;
        max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES,floating_gauge::Symbol=:reject)
    isfinite(f) && f >= 0 || throw(ArgumentError("circuit frequency must be finite and nonnegative"))
    f=_circuit_stored_real(f,"circuit frequency")
    floating_gauge in (:reject,:auto) || throw(ArgumentError("floating_gauge must be :reject or :auto"))
    np = length(circuit.ports)
    ncurr = sum(length(element.terminals) for element in circuit.elements;init=0)
    N = circuit.nnodes + ncurr + np
    _enforce_payload_limit(_checked_payload_sum("circuit solve",
        _circuit_owned_model_payload(circuit),
        _circuit_network_workspace_payload(circuit),
        _checked_array_payload_bytes(ComplexF64,N,N),
        _checked_array_payload_bytes(ComplexF64,3,N,np),
        _checked_array_payload_bytes(ComplexF64,4,np,np),
        _checked_array_payload_bytes(ComplexF64,2,np+ncurr),
        _checked_array_payload_bytes(Int,2,length(circuit.elements)),
        floating_gauge===:auto ? _checked_array_payload_bytes(Int,4,circuit.nnodes+1) : 0),
        max_bytes,"circuit solve","max_bytes")
    refs=_circuit_z0(circuit.z0,np;freq=f)
    roots=_planar_reference_roots(refs)
    M = zeros(ComplexF64,N,N)
    RHS = zeros(ComplexF64,N,np)
    column = circuit.nnodes
    omega = 2pi * f
    branch_rows=Vector{Int}(undef,length(circuit.elements));uniform_y_null=falses(length(circuit.elements))
    nextrow=circuit.nnodes+1
    for (k,element) in enumerate(circuit.elements)
        branch_rows[k]=nextrow;nextrow+=length(element.terminals)
    end
    for (element_index,element) in enumerate(circuit.elements)
        nl = length(element.terminals)
        ids = (column+1):(column+nl)
        for (local_index,id) in enumerate(ids)
            _circuit_current_stamp!(M,id,element.terminals[local_index],1.0)
        end
        if element isa _CircuitRLC
            e = element
            row = first(ids)
            if e.topology === :series
                if e.c !== nothing && (iszero(e.c) || iszero(omega))
                    M[row,row] = 1.0 # series capacitor is open at DC
                else
                    z = (e.r === nothing ? 0.0 : e.r) +
                        (e.l === nothing ? 0.0 : 1im*omega*e.l) +
                        (e.c === nothing ? 0.0 : inv(1im*omega*e.c))
                    _circuit_voltage_stamp!(M,row,e.terminals[1],1.0)
                    M[row,row] = -z
                end
            else
                short = (e.r !== nothing && iszero(e.r)) ||
                    (e.l !== nothing && (iszero(e.l) || iszero(omega)))
                if short
                    _circuit_voltage_stamp!(M,row,e.terminals[1],1.0)
                else
                    y = (e.r === nothing ? 0.0 : inv(e.r)) +
                        (e.l === nothing ? 0.0 : inv(1im*omega*e.l)) +
                        (e.c === nothing ? 0.0 : 1im*omega*e.c)
                    _circuit_voltage_stamp!(M,row,e.terminals[1],y)
                    M[row,row] = -1.0
                end
            end
        elseif element isa _CircuitNetwork
            e = element
            value = e.response isa AbstractMatrix ? e.response : e.response(f)
            value isa AbstractMatrix && size(value) == (nl,nl) && all(isfinite,value) ||
                throw(ArgumentError("network response must be a finite $(nl)x$(nl) matrix at $f Hz"))
            H = Matrix{ComplexF64}(value)
            all(isfinite,H) || throw(ArgumentError("network response must remain finite in ComplexF64 at $f Hz"))
            floating_gauge===:auto && e.format===:y && (uniform_y_null[element_index]=_circuit_uniform_y_null(H))
            if e.format === :y
                for p in 1:nl
                    for q in 1:nl
                        _circuit_voltage_stamp!(M,ids[p],e.terminals[q],H[p,q])
                    end
                    M[ids[p],ids[p]]-=1.0
                end
            elseif e.format === :z
                for p in 1:nl
                    _circuit_voltage_stamp!(M,ids[p],e.terminals[p],1.0)
                    for q in 1:nl
                        M[ids[p],ids[q]]-=H[p,q]
                    end
                end
            else
                local_refs=_circuit_z0(e.z0,nl;freq=f)
                _circuit_scattering_stamp!(M,ids,e.terminals,H,local_refs)
            end
        elseif element isa _CircuitLine
            e = element
            z=_circuit_line_value(_circuit_value(e.zc,f),"line characteristic impedance";nonzero=true)
            g=_circuit_line_value(_circuit_value(e.gl,f),"line electrical length")
            _circuit_line_stamp!(M,first(ids),e.terminals,z,g)
        elseif element isa _CircuitTransformer
            e = element::_CircuitTransformer
            p,q = ids
            _circuit_voltage_stamp!(M,p,e.terminals[1],1.0)
            _circuit_voltage_stamp!(M,p,e.terminals[2],-e.ratio)
            M[q,p],M[q,q] = e.ratio,1.0
        else
            _circuit_stamp_extra!(M,first(ids),element,f,branch_rows)
        end
        column += nl
    end
    port_ids = (column+1):(column+np)
    for p in 1:np
        id = port_ids[p]
        _circuit_current_stamp!(M,id,circuit.ports[p],-1.0)
        _circuit_voltage_stamp!(M,id,circuit.ports[p],1.0)
        M[id,id] = refs[p]
        RHS[id,p] = 2roots[p]
    end
    all(isfinite,M) || throw(ArgumentError("circuit coefficients are non-finite"))
    gauges=floating_gauge===:auto ? _circuit_gauge_nodes(circuit,uniform_y_null) : Int[]
    # The sum of KCL rows in each ungrounded terminal component is
    # redundant. Replace one such row by a voltage-coordinate choice.
    # This removes only the arbitrary common potential, not a physical
    # conductor mode or a terminal current equation.
    for node in gauges
        M[node,:].=0;M[node,node]=1;RHS[node,:].=0
    end
    factor = lu!(M;check=false)
    issuccess(factor) || throw(ArgumentError(
        "circuit system is singular; check floating nodes and redundant ideal constraints"))
    solution = ldiv!(factor,RHS)
    all(isfinite,solution) || throw(ArgumentError("circuit solution is non-finite"))
    currents = Matrix{ComplexF64}(solution[port_ids,:])
    voltages = Matrix{ComplexF64}(solution[1:circuit.nnodes,:])
    S = Matrix{ComplexF64}(I,np,np) - Diagonal(roots)*currents
    Vports = Matrix{ComplexF64}(undef,np,np)
    for p in 1:np,q in 1:np
        positive,negative = circuit.ports[p]
        Vports[p,q] = (positive == 0 ? 0.0im : voltages[positive,q]) -
                     (negative == 0 ? 0.0im : voltages[negative,q])
    end
    F = lu(Vports;check=false)
    Y = issuccess(F) ? Matrix{ComplexF64}(currents / F) : nothing
    return PlanarCircuitResult(Float64(f),S,Y,voltages,currents,gauges,refs)
end

"""Sweep the external power-wave S-matrix of a combined circuit/EM model."""
planar_circuit_sparams(circuit::PlanarCircuit,freqs::AbstractVector{<:Real};kw...) =
    [solve_planar_circuit(circuit,f;kw...).s for f in freqs]
