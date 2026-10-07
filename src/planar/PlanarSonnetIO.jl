# Native Sonnet geometry-project import. Parsing preserves all records;
# lowering rejects physics/workflow features the current backend cannot express.

export SonnetRecord, SonnetPolygon, SonnetPortSpec, SonnetProject, SonnetNetlistProject
export read_sonnet_project, sonnet_planar_problem, sonnet_variable_value, sonnet_planar_circuit
export SonnetPlanarResult, solve_sonnet_project, sonnet_metal_zs

function _sonnet_port_reference(p,ps,freq::Real,variables=Dict{String,Float64}())
    length(ps.values)>=5 || throw(ArgumentError("incomplete native port R/X/L/C reference"))
    r,x,l,c=[sonnet_variable_value(p,token;variables,freq) for token in ps.values[2:5]]
    # Native port_equiv.png specifies the capacitor in parallel with the
    # series R+jX+jωL branch. Its zero value omits that shunt capacitor.
    # Port fields have fixed units Ω, Ω, nH, pF, independently of DIM.
    # Installed PortNormalizingImpedances.html states this contract; actual
    # DIM RES OH/KOH/MOH controls retain identical 50Ω/RPV5Ω responses.
    inductance=l*1e-9;capacitor=c*1e-12
    (iszero(l) || !iszero(inductance)) && (iszero(c) || !iszero(capacitor)) ||
        throw(ArgumentError("native port L/C must preserve nonzero values after SI conversion"))
    return PlanarPortImpedance(r=r,x=x,l=inductance,c=capacitor,topology=:parallel)(freq)
end

"""One native record with its source line and quoted tokens decoded."""
struct SonnetRecord
    line::Int
    tokens::Vector{String}
end

"""A native polygon. Levels/material indices remain zero-based as in Sonnet.
Coordinates are SI; closed endpoint and vertex order are preserved for ports.
`kind` is :sheet, :via, or :brick. Via target is GND, TOP, or a level number."""
struct SonnetPolygon
    kind::Symbol
    level::Int
    material::Int
    id::Int
    vertices::Matrix{Float64}
    target::String
    technology::String
    flags::Vector{String}
end

"""Native gap/box/co-calibrated/via port including its complete metadata.
`edge` is a zero-based native polygon-edge index; `values` retain R/X/L/C,
position, and optional reference-plane settings without losing expressions.
Port R/X/L/C fields use fixed Ω/Ω/nH/pF units, independently of DIM. The
reference capacitor is parallel to the series R+jX+jωL branch."""
struct SonnetPortSpec
    kind::Symbol
    polygon::Int
    edge::Int
    number::Int
    values::Vector{String}
    records::Vector{SonnetRecord}
end

"""Parsed native geometry project. Every input record is retained. Materials,
variables, sweeps, components, and technology declarations remain accessible
even where backend lowering rejects their unsupported semantics."""
struct SonnetProject
    source::String
    units::Dict{String,String}
    length_scale::Float64
    frequency_scale::Float64
    box::Vector{String}
    layers::Vector{Vector{String}}
    metals::Vector{Vector{String}}
    top::Vector{String}
    bottom::Vector{String}
    polygons::Vector{SonnetPolygon}
    ports::Vector{SonnetPortSpec}
    variables::Dict{String,String}
    components::Vector{Vector{SonnetRecord}}
    sweeps::Vector{SonnetRecord}
    records::Vector{SonnetRecord}
end

"""Native circuit project with complete CKT statements and unit declarations.
Subnetwork definitions remain ordered as in the native file."""
struct SonnetNetlistProject
    source::String
    units::Dict{String,String}
    frequency_scale::Float64
    circuit::Vector{SonnetRecord}
    records::Vector{SonnetRecord}
end

# Native port-edge checks allow half an actual cell plus 0.0001 cell at a
# wall. Compare in cell coordinates: SI subtraction can flip boundary ties.
function _sonnet_polygon_edge_count(vertices)
    n=size(vertices,2)
    return n>1 && vertices[1,1]==vertices[1,n] && vertices[2,1]==vertices[2,n] ? n-1 : n
end

function _sonnet_port_edge_indices(vertices,edge)
    n=_sonnet_polygon_edge_count(vertices)
    0<=edge<n || throw(ArgumentError("native port edge is invalid"))
    return edge+1,mod1(edge+2,n)
end

function _sonnet_shared_port_segment(p,poly,first,second)
    v=poly.vertices;a=(v[1,first],v[2,first]);b=(v[1,second],v[2,second])
    a!=b || throw(ArgumentError("native port has a degenerate edge"))
    axis=abs(b[1]-a[1])>=abs(b[2]-a[2]) ? 1 : 2
    lower,upper=minmax(a[axis],b[axis]);shared=nothing
    for other in p.polygons
        other.kind===:sheet && other.level==poly.level && other.id!=poly.id || continue
        w=other.vertices;n=_sonnet_polygon_edge_count(w)
        for i in 1:n
            j=mod1(i+1,n);c=(w[1,i],w[2,i]);d=(w[1,j],w[2,j])
            lo=max(lower,min(c[axis],d[axis]));hi=min(upper,max(c[axis],d[axis]))
            lo<hi || continue
            collinear=if a[1]==b[1]
                c[1]==d[1]==a[1]
            elseif a[2]==b[2]
                c[2]==d[2]==a[2]
            else
                iszero(_planar_orient2d(a...,b...,c...)) && iszero(_planar_orient2d(a...,b...,d...))
            end
            collinear || continue
            shared===nothing || throw(ArgumentError("more than two polygon edges coincide with native port edge"))
            low=a[axis]==lo ? a : b[axis]==lo ? b : c[axis]==lo ? c : d
            high=a[axis]==hi ? a : b[axis]==hi ? b : c[axis]==hi ? c : d
            shared=a[axis]<b[axis] ? (low,high) : (high,low)
        end
    end
    shared===nothing && throw(ArgumentError("native internal port has no adjacent return polygon edge"))
    return shared
end

function _sonnet_box_port_edge_inside(vertices,edge,a,b,nx,ny)
    first,second=_sonnet_port_edge_indices(vertices,edge)
    return -.5001<(vertices[1,first]/a)*nx<nx+.5001 &&
        -.5001<(vertices[1,second]/a)*nx<nx+.5001 &&
        -.5001<(vertices[2,first]/b)*ny<ny+.5001 &&
        -.5001<(vertices[2,second]/b)*ny<ny+.5001
end

# Native wall ownership uses the source BOX grid, even with a caller grid.
# The conformal importer shares this boundary normalization.
# Copy only sheets containing vertices that actually need wall projection.
function _sonnet_raster_wall_project(p,a,b,nx,ny)
    polygons=p.polygons
    for (index,poly) in enumerate(p.polygons)
        poly.kind===:sheet || continue
        vertices=poly.vertices
        for j in axes(vertices,2),axis in 1:2
            extent=axis==1 ? a : b;count=axis==1 ? nx : ny
            value=poly.vertices[axis,j];cell=(value/extent)*count
            projected=-.5001<cell<.5001 ? 0. : count-.5001<cell<count+.5001 ? extent : value
            projected==value && continue
            vertices===poly.vertices && (vertices=copy(vertices))
            vertices[axis,j]=projected
        end
        vertices===poly.vertices && continue
        polygons===p.polygons && (polygons=copy(polygons))
        polygons[index]=SonnetPolygon(poly.kind,poly.level,poly.material,poly.id,
            vertices,poly.target,poly.technology,poly.flags)
    end
    polygons===p.polygons && return p
    return SonnetProject(p.source,p.units,p.length_scale,p.frequency_scale,p.box,p.layers,p.metals,
        p.top,p.bottom,polygons,p.ports,p.variables,p.components,p.sweeps,p.records)
end

function _sonnet_tokens(line::AbstractString)
    out=String[]; buf=IOBuffer(); quoted=false; active=false
    chars=collect(line);i=1
    while i<=length(chars)
        c=chars[i]
        if quoted && c=='\\' && i<length(chars) && chars[i+1]=='"'
            print(buf,'"');active=true;i+=2;continue
        end
        if c=='"'
            quoted=!quoted; active=true
        elseif c=='!' && !quoted
            break
        elseif isspace(c) && !quoted
            if active
                push!(out,String(take!(buf))); active=false
            end
        else
            print(buf,c); active=true
        end
        i+=1
    end
    quoted && throw(ArgumentError("unterminated quoted Sonnet field"))
    active && push!(out,String(take!(buf)))
    return out
end

_sonnet_error(source,line,message)=throw(ArgumentError("$source:$line: $message"))
function _sonnet_section(records,name,source)
    first=findfirst(r->r.tokens==[name],records)
    first===nothing && _sonnet_error(source,0,"missing $name section")
    last=findnext(r->r.tokens==["END",name],records,first+1)
    last===nothing && _sonnet_error(source,records[first].line,"unterminated $name section")
    return records[first+1:last-1]
end

"""Read Sonnet v18 geometry/circuit records and validate polygon/port references.
An external STF stack is rejected explicitly. Expressions
are retained and evaluated by a restricted arithmetic interpreter at lowering.
Parsing a project does not certify that every feature can be simulated."""
function read_sonnet_project(path::AbstractString)
    # Dependency snapshots use resolved paths as keys, including aliases of
    # macOS temporary directories and Windows short filenames.
    source=realpath(path)
    records=SonnetRecord[]
    open(source,"r") do io
        for (line,text) in enumerate(eachline(io))
            tokens=_sonnet_tokens(text)
            isempty(tokens) || push!(records,SonnetRecord(line,tokens))
        end
    end
    return _sonnet_read_records(source,records)
end

# Shared by the bounded STF adapter after explicit static materialization.
# No filesystem rewrite or unsupported-feature bypass is exposed publicly.
function _sonnet_read_records(source::String,records::Vector{SonnetRecord})
    isempty(records) && _sonnet_error(source,0,"empty project")
    length(records[1].tokens)>=2 && records[1].tokens[1]=="FTYP" &&
        records[1].tokens[2] in ("SONPROJ","SONNETPRJ") ||
        _sonnet_error(source,records[1].line,"unsupported native project type")
    dims=_sonnet_section(records,"DIM",source)
    units=Dict{String,String}()
    for r in dims
        length(r.tokens)==2 || _sonnet_error(source,r.line,"invalid DIM record")
        units[r.tokens[1]]=uppercase(r.tokens[2])
    end
    ls=get(Dict("M"=>1.,"CM"=>1e-2,"MM"=>1e-3,"UM"=>1e-6,
        "NM"=>1e-9,"MIL"=>25.4e-6,"MILS"=>25.4e-6,"IN"=>.0254),get(units,"LNG",""),NaN)
    fs=get(Dict("HZ"=>1.,"KHZ"=>1e3,"MHZ"=>1e6,"GHZ"=>1e9,
        "THZ"=>1e12),get(units,"FREQ",""),NaN)
    isfinite(ls) && isfinite(fs) || _sonnet_error(source,0,"unsupported length/frequency units")
    if records[1].tokens[2]=="SONNETPRJ"
        circuit=_sonnet_section(records,"CKT",source)
        definitions=filter(r->occursin(r"^DEF\d+P$",r.tokens[1]),circuit)
        isempty(definitions) && _sonnet_error(source,0,"netlist requires a DEF<n>P definition")
        for r in definitions
            np=parse(Int,match(r"^DEF(\d+)P$",r.tokens[1])[1])
            length(r.tokens)==np+4 && r.tokens[np+3]=="R" ||
                _sonnet_error(source,r.line,"invalid network definition")
            parse.(Int,r.tokens[2:np+1])
            parse(Float64,r.tokens[end])>0 || _sonnet_error(source,r.line,"invalid network reference impedance")
        end
        return SonnetNetlistProject(source,units,fs,circuit,records)
    end
    geo=_sonnet_section(records,"GEO",source)
    any(r->r.tokens[1]=="STF",geo) && _sonnet_error(source,0,
        "external STF stack requires its technology-file importer")
    bi=findfirst(r->r.tokens[1]=="BOX",geo)
    bi===nothing && _sonnet_error(source,0,"missing BOX")
    box=geo[bi].tokens[2:end]
    length(box)>=5 || _sonnet_error(source,geo[bi].line,"invalid BOX")
    nl=parse(Int,box[1])+1
    nl>0 && bi+nl<=length(geo) || _sonnet_error(source,geo[bi].line,"invalid layer count")
    layers=[copy(geo[bi+i].tokens) for i in 1:nl]
    all(x->length(x)>=7,layers) || _sonnet_error(source,geo[bi].line,"incomplete dielectric layer record")
    top_records=filter(r->r.tokens[1]=="TMET",geo)
    bot_records=filter(r->r.tokens[1]=="BMET",geo)
    length(top_records)==length(bot_records)==1 || _sonnet_error(source,0,"TMET/BMET must occur once")
    top=top_records[1].tokens[2:end]; bottom=bot_records[1].tokens[2:end]
    metals=[r.tokens[2:end] for r in geo if r.tokens[1]=="MET"]
    vars=Dict{String,String}()
    for r in geo
        r.tokens[1]=="VALVAR" || continue
        length(r.tokens)>=4 || _sonnet_error(source,r.line,"invalid VALVAR")
        haskey(vars,r.tokens[2]) && _sonnet_error(source,r.line,"duplicate VALVAR")
        vars[r.tokens[2]]=r.tokens[4]
    end
    ni=findfirst(r->r.tokens[1]=="NUM",geo)
    ni===nothing && _sonnet_error(source,0,"missing polygon NUM")
    np=parse(Int,geo[ni].tokens[2]); np>=0 || _sonnet_error(source,geo[ni].line,"negative NUM")
    polygons=SonnetPolygon[]; i=ni+1
    for _ in 1:np
        i<=length(geo) || _sonnet_error(source,0,"NUM exceeds polygon records")
        kind=:sheet
        if geo[i].tokens==["VIA","POLYGON"]
            kind=:via; i+=1
        elseif geo[i].tokens==["BRI","POLY"]
            kind=:brick; i+=1
        end
        i<=length(geo) || _sonnet_error(source,0,"polygon marker lacks its header")
        header=geo[i]; t=header.tokens; i+=1
        length(t)>=5 || _sonnet_error(source,header.line,"incomplete polygon header")
        level,nv,mat=parse.(Int,t[1:3]); id=parse(Int,t[5])
        nv>=3 || _sonnet_error(source,header.line,"polygon requires >=3 vertices")
        endpoint=findnext(r->length(r.tokens)==1 && r.tokens[1]=="END",geo,i)
        endpoint!==nothing || _sonnet_error(source,header.line,"unterminated polygon")
        # A serialized count must agree with the captured coordinate records
        # before it controls storage. A tiny malformed source can otherwise
        # allocate gigabytes solely from its untrusted nv header.
        coordinates=count(i:endpoint-1) do index
            q=geo[index].tokens
            length(q)==2 && !(q[1] in ("TOLEVEL","TLAYNAM"))
        end
        coordinates==nv || _sonnet_error(source,header.line,
            "vertex count $coordinates differs from declared $nv")
        vertices=Matrix{Float64}(undef,2,nv); k=0; target=""; technology=""; flags=copy(t[4:end])
        while i<endpoint
            row=geo[i]; q=row.tokens
            if q[1]=="TOLEVEL"
                target=q[2]
                append!(flags,q[3:end])
            elseif q[1]=="TLAYNAM"
                technology=q[2]
            elseif length(q)==2 && all(v->tryparse(Float64,v)!==nothing,q)
                k+=1; k<=nv || _sonnet_error(source,row.line,"too many polygon vertices")
                vertices[1,k]=parse(Float64,q[1])*ls
                vertices[2,k]=parse(Float64,q[2])*ls
            else
                _sonnet_error(source,row.line,"unsupported polygon record $(q[1])")
            end
            i+=1
        end
        k==nv || _sonnet_error(source,header.line,"vertex count $k differs from declared $nv")
        all(isfinite,vertices) || _sonnet_error(source,header.line,"non-finite polygon vertex")
        kind==:via && isempty(target) && _sonnet_error(source,header.line,"via lacks TOLEVEL")
        push!(polygons,SonnetPolygon(kind,level,mat,id,vertices,target,technology,flags))
        i+=1
    end
    i>length(geo) || _sonnet_error(source,geo[i].line,"unparsed geometry after NUM polygons")
    length(unique(p.id for p in polygons))==length(polygons) || _sonnet_error(source,0,"duplicate polygon IDs")
    ids=Dict(p.id=>p for p in polygons)
    ports=SonnetPortSpec[]; components=Vector{SonnetRecord}[]
    i=1
    while i<ni
        r=geo[i]
        if r.tokens[1]=="POR1"
            j=i+1
            while j<ni && !(geo[j].tokens[1] in ("POR1","CUPGRP","SMD"))
                j+=1
            end
            pr=geo[i:j-1]
            pi=findfirst(x->x.tokens[1]=="POLY",pr)
            pi===nothing && _sonnet_error(source,r.line,"port lacks POLY")
            pi+2<=length(pr) || _sonnet_error(source,r.line,"port lacks edge/electrical data")
            polygon=parse(Int,pr[pi].tokens[2]); edge=parse(Int,pr[pi+1].tokens[1])
            values=pr[pi+2].tokens
            length(values)>=7 || _sonnet_error(source,r.line,"incomplete port values")
            number=parse(Int,values[1])
            haskey(ids,polygon) || _sonnet_error(source,r.line,"unknown port polygon $polygon")
            0<=edge<_sonnet_polygon_edge_count(ids[polygon].vertices) || _sonnet_error(source,r.line,"invalid port edge")
            push!(ports,SonnetPortSpec(Symbol(lowercase(r.tokens[2])),polygon,edge,number,copy(values),pr))
            i=j
        elseif r.tokens[1]=="SMD" && i+1<ni && geo[i+1].tokens[1]=="ID"
            j=i+1
            while j<ni && geo[j].tokens!=["END"]
                j+=1
            end
            j<ni || _sonnet_error(source,r.line,"unterminated SMD component")
            cr=geo[i:j]
            count(x->x.tokens[1]=="TYPE",cr)==1 || _sonnet_error(source,r.line,"component requires exactly one TYPE")
            push!(components,cr); i=j+1
        else
            i+=1
        end
    end
    sweeps=[r for r in records if r.tokens[1]=="FREQ" && length(r.tokens)>2]
    return SonnetProject(source,units,ls,fs,box,layers,metals,top,bottom,
        polygons,ports,vars,components,sweeps,records)
end

function _sonnet_math_real(value,quantity)
    value isa Complex && !iszero(imag(value)) &&
        throw(ArgumentError("native $quantity requires real arguments"))
    return _circuit_stored_real(real(value),"native $quantity")
end

function _sonnet_math_signed_magnitude(value,quantity)
    # The installed engine compares/remainders complex operands through
    # their magnitudes with a negative sign exactly when real(value)<0.
    # Pure imaginary operands (including a -0 real part) remain positive.
    return _circuit_stored_real(real(value)<0 ? -abs(value) : abs(value),
        "native $quantity magnitude")
end

function _sonnet_math_projection(value,quantity)
    isfinite(value) || throw(ArgumentError("native $quantity requires a finite argument"))
    return _circuit_stored_real(real(value),"native $quantity")
end

function _sonnet_math_axis(value,op)
    # Native real-axis branches ignore the input's imaginary signed zero.
    # Actual +/-0 controls select upper sqrt/acosh on the negative axis and
    # the lower (positive-x) asin/acos/atanh branch outside [-1,1].
    if iszero(imag(value))
        x=real(value)
        z=(op in (:asin,:acos,:atanh) && abs(x)>1 && x>0) ? -0. : 0.
        return ComplexF64(x,z)
    end
    return value
end

function _sonnet_expr(ex,project,overrides,freq,active,depth::Int=0)
    depth<=128 || throw(ArgumentError("native scalar evaluation depth budget exceeded"))
    ex isa Number && return _circuit_stored_real(ex,"native scalar literal")
    if ex isa Symbol
        name=String(ex)
        lowercase(name)=="pi" && return pi
        uppercase(name)=="FREQ" && return freq
        if haskey(overrides,name)
            overrides[name] isa Real || throw(ArgumentError("native scalar overrides must be real"))
            return _circuit_stored_real(overrides[name],"native scalar override $name")
        end
        haskey(project.variables,name) || throw(ArgumentError("unknown Sonnet variable $name"))
        name in active && throw(ArgumentError("cyclic Sonnet variable $name"))
        push!(active,name)
        val=_sonnet_expr(_sonnet_parse_scalar(project.variables[name]),project,overrides,freq,active,depth+1)
        delete!(active,name)
        # Native quantity variables store the real projection of their
        # finite defining expression. Actual SRES controls retain Inner=3
        # from cmplx(3,4), so abs(Inner)=3 while abs(cmplx(3,4))=5.
        # This also resets imaginary signed zero at a named variable boundary.
        isfinite(val) || throw(ArgumentError("non-finite Sonnet variable $name"))
        return _circuit_stored_real(real(val),"native variable $name")
    end
    ex isa Expr && ex.head==:call || throw(ArgumentError("unsupported Sonnet expression"))
    if ex.args[1] in (:table1,:table2)
        kind=ex.args[1]
        length(ex.args)==(kind==:table1 ? 3 : 4) || throw(ArgumentError("native $kind has invalid argument count"))
        ex.args[2] isa String || throw(ArgumentError("native table filename must be a literal quoted string"))
        keys=[_sonnet_expr(x,project,overrides,freq,active,depth+1) for x in ex.args[3:end]]
        native_keys=kind===:table1 ? [_sonnet_math_projection(keys[1],"table1 key")] :
            [_sonnet_math_signed_magnitude(keys[1],"table2 row key"),
                _sonnet_math_projection(keys[2],"table2 column key")]
        return _sonnet_scalar_table_value(overrides,kind,ex.args[2],native_keys)
    end
    # Dispatch directly instead of rebuilding a dictionary of boxed functions
    # and project-capturing closures at every node in every material/provider
    # evaluation. Fold the n-ary ASTs produced for native sums and products
    # without constructing argument vectors or dynamic splat buffers.
    op=ex.args[1];n=length(ex.args)-1
    if op in (:+,:*)
        n==0 && return op===:+ ? 0. : 1.
        value=_sonnet_expr(ex.args[2],project,overrides,freq,active,depth+1)
        for i in 3:length(ex.args)
            argument=_sonnet_expr(ex.args[i],project,overrides,freq,active,depth+1)
            value=op===:+ ? value+argument : value*argument
        end
        return value
    elseif op===:-
        n in (1,2) || throw(ArgumentError("native subtraction requires one or two scalar arguments"))
        a=_sonnet_expr(ex.args[2],project,overrides,freq,active,depth+1)
        # Native complex phase retains the imaginary signed zero introduced
        # by a unary negative real operand: deg(-1)=-180, but
        # deg(cmplx(-1,0))=180.
        n==1 && return a isa Real ? -ComplexF64(a,0.) : -a
        return a-_sonnet_expr(ex.args[3],project,overrides,freq,active,depth+1)
    elseif op in (:/,:^)
        n==2 || throw(ArgumentError("native $op requires exactly two scalar arguments"))
        a=_sonnet_expr(ex.args[2],project,overrides,freq,active,depth+1)
        b=_sonnet_expr(ex.args[3],project,overrides,freq,active,depth+1)
        return op===:/ ? a/b : (a isa Real && a<0 ? ComplexF64(a,0.)^b : a^b)
    elseif op in (:atan2,:hypot,:fmod,:max,:min,:cmplx)
        n==2 || throw(ArgumentError("native $op requires exactly two scalar arguments"))
        a=_sonnet_expr(ex.args[2],project,overrides,freq,active,depth+1)
        b=_sonnet_expr(ex.args[3],project,overrides,freq,active,depth+1)
        isfinite(a) && isfinite(b) || throw(ArgumentError("native $op requires finite arguments"))
        # The native atan2 control maps both signs of zero y to +pi at x<0.
        if op===:atan2
            if iszero(b)
                iszero(a) && throw(ArgumentError("native atan2 is undefined for two zero operands"))
                # Actual real/complex controls establish this native special
                # branch, including its retained real(y) imaginary part.
                return complex(pi/2,real(a))
            end
            if iszero(imag(a)) && iszero(imag(b))
                return atan(iszero(real(a)) ? 0. : real(a),real(b))
            end
            value=atan(_sonnet_math_axis(a/b,:atan))
            isfinite(value) || throw(ArgumentError("native atan2 has a non-finite complex result"))
            return real(b)<0 ? value+(real(a)<0 ? -pi : pi) : value
        end
        op===:hypot && return hypot(abs(a),abs(b))
        op===:cmplx && return a+im*b
        ar=_sonnet_math_signed_magnitude(a,op);br=_sonnet_math_signed_magnitude(b,op)
        if op===:fmod
            remainder=rem(ar,br)
            return iszero(remainder) ? 0. : remainder
        end
        # Preserve the selected complex operand; native ties choose the
        # second operand for both min and max.
        return op===:max ? (ar>br ? a : b) : (ar<br ? a : b)
    elseif op in (:sqrt,:sin,:cos,:tan,:asin,:acos,:atan,:sinh,:cosh,:tanh,
            :asinh,:acosh,:atanh,:exp,:ln,:log10,:db10,:db20,:abs,:mag,
            :real,:imag,:conj,:deg,:rad,:int,:h2p,:p2h,:m2p,:p2m)
        n==1 || throw(ArgumentError("native $op requires exactly one scalar argument"))
        a=_sonnet_expr(ex.args[2],project,overrides,freq,active,depth+1)
        op===:sqrt && return sqrt(_sonnet_math_axis(a,op))
        op===:sin && return sin(_sonnet_math_axis(a,op))
        op===:cos && return cos(_sonnet_math_axis(a,op))
        op===:tan && return tan(_sonnet_math_axis(a,op))
        op===:asin && return asin(_sonnet_math_axis(a,op))
        op===:acos && return acos(_sonnet_math_axis(a,op))
        op===:atan && return atan(_sonnet_math_axis(a,op))
        op===:sinh && return sinh(a)
        op===:cosh && return cosh(a)
        op===:tanh && return tanh(a)
        op===:asinh && return asinh(_sonnet_math_axis(a,op))
        op===:acosh && return acosh(_sonnet_math_axis(a,op))
        op===:atanh && return atanh(_sonnet_math_axis(a,op))
        op===:exp && return exp(a)
        op===:ln && return log(abs(a))
        op===:log10 && return log10(abs(a))
        op===:db10 && return 10log10(abs(a))
        op===:db20 && return 20log10(abs(a))
        op in (:abs,:mag) && return abs(a)
        op===:real && return real(a)
        op===:imag && return imag(a)
        op===:conj && return conj(a)
        op===:deg && return rad2deg(angle(a))
        op===:rad && return angle(a)
        if op===:int
            isfinite(a) || throw(ArgumentError("native int requires a finite argument"))
            integer=trunc(real(a))
            return iszero(integer) ? 0. : integer
        end
        a=_sonnet_math_projection(a,op)
        op===:h2p && return a/project.frequency_scale
        op===:p2h && return a*project.frequency_scale
        op===:m2p && return a/project.length_scale
        return a*project.length_scale
    end
    throw(ArgumentError("unsupported Sonnet expression function $op"))
end

"""Evaluate a native scalar/variable expression without arbitrary code execution.
Overrides use the quantity's native units. `FREQ` uses Hz independently of
`DIM FREQ`. `h2p`/`p2h` convert Hz/project frequency units and `m2p`/`p2m`
convert metres/project length units. Native powers associate left; documented
material math permits complex intermediate values and stores the real projection
of a finite final quantity. Unsupported functions and conditionals reject."""
function sonnet_variable_value(project::SonnetProject,text::AbstractString;
        variables=Dict{String,Float64}(),freq::Real=1e9,scalar_files=nothing,
        scalar_root=dirname(project.source),scalar_outside::Symbol=:reject,
        scalar_max_files::Integer=64,scalar_max_bytes::Integer=8*1024^2,
        scalar_max_nodes::Integer=100000,scalar_max_line_bytes::Integer=16384,
        scalar_max_storage::Integer=64*1024^2)
    isfinite(freq) && freq>=0 || throw(ArgumentError("native scalar frequency must be finite and nonnegative"))
    freq=_circuit_stored_real(freq,"native scalar frequency")
    if scalar_files!==nothing || (!(variables isa SonnetScalarVariables) &&
            (_sonnet_has_scalar_tables(project) || occursin(r"\btable[12]\s*\(",text)))
        variables=_sonnet_scalar_variables(project,variables;scalar_files,
            max_bytes=scalar_max_storage,root=scalar_root,outside=scalar_outside,
            max_files=scalar_max_files,max_bytes_source=scalar_max_bytes,
            max_nodes=scalar_max_nodes,max_line_bytes=scalar_max_line_bytes,expressions=[String(text)])
        project=_sonnet_scalar_project(project,variables)
    end
    val=_sonnet_expr(_sonnet_parse_scalar(text),project,variables,freq,Set{String}())
    isfinite(val) || throw(ArgumentError("non-finite Sonnet scalar: $text"))
    # Native numeric material fields accept inline complex expressions and
    # store their real part, just like named quantity variables. Actual NOR
    # controls (quoted/bare cmplx, sqrt and quotient) establish this boundary.
    return _circuit_stored_real(real(val),"native scalar $text")
end

function _sonnet_touchstone_response(path::AbstractString,np::Int)
    isfile(path) || throw(ArgumentError("missing native network data file $path"))
    data=planar_read_touchstone(path;nports=np)
    # Express each evaluated response on a fixed real basis for its circuit
    # block. This retains TERM/FTERM references and their physical network,
    # including frequency-dependent reference samples, through one importer.
    response=f->planar_network_response(data,f;z0=50.)
    return response,50.
end

function _sonnet_circuit_literal(p,row,text,scale=1.)
    ncodeunits(text)<=16384 || _sonnet_error(p.source,row.line,"native circuit literal byte budget exceeded")
    # One conversion after source parsing and SI scaling preserves subnormals
    # and distinguishes explicit zeros from nonzero values lost in storage.
    occursin(r"^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$",text) ||
        _sonnet_error(p.source,row.line,"native circuit requires a decimal literal")
    mantissa=first(split(lowercase(text),'e';limit=2))
    nonzero=any(c->'1'<=c<='9',mantissa)
    # Scoped precision/rounding is task-local on the supported Julia versions.
    # Retain enough precision for long decimals close to a Float64 midpoint.
    return setprecision(BigFloat,4ncodeunits(text)+2048) do
        setrounding(BigFloat,RoundNearest) do
            value=tryparse(BigFloat,text)
            value!==nothing && isfinite(value) && value>=0 && (!nonzero || !iszero(value)) ||
                _sonnet_error(p.source,row.line,"native circuit requires a finite nonnegative literal value")
            scaled=value*BigFloat(scale);stored=Float64(scaled)
            isfinite(stored) && (!nonzero || !iszero(stored)) ||
                _sonnet_error(p.source,row.line,"native circuit value must remain finite and preserve nonzero units in Float64")
            return stored
        end
    end
end

"""Lower native CKT R/L/C, Touchstone blocks, and nested DEF<n>P subnetworks
to modified nodal analysis. `project_response(path,f_hz)` supplies a referenced
PRJ's calibrated S matrix; requiring it explicitly prevents accidental use of
raw gap-port data in a circuit expecting native calibrated data. Native units
are applied to primitive resistors, capacitors and inductors. Resistance units
OH/OHMS, KOH and MOH scale primitive R by 1, 1e3 and 1e6; omitted RES defaults
to ohms. DEF network references remain in ohms. Touchstone data, including native complex
TERM/FTERM references, use the general importer and a real 50 Ω circuit block.
Native node labels are compacted within each DEF network, with literal node 0
retained as ground; unused numeric labels do not create floating MNA nodes.
The parsed project retains its original labels. Nonzero literals must remain
representable after SI conversion. All unknown circuit statements fail."""
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

function _sonnet_termination(p,metal,freq,vars)
    length(metal)>=3 || throw(ArgumentError("incomplete box-cover material"))
    model=metal[3]
    zs=sonnet_metal_zs(p,metal,freq;variables=vars,cover=true)
    iszero(zs) && return TERM_GND
    return PlanarTerminator{ComplexF64}(TERM_SURFACE,zs,1+0im,1+0im)
end

function _sonnet_loss_transition(rdc,rs,what="SUP")
    rf=complex(rs,rs)
    if iszero(rdc)
            !iszero(rs) || throw(ArgumentError("$what RF impedance underflows"))
            rf
        else
            ratio=rs/rdc
            if ratio<.25
                # x*coth(x), x=(1+i)*ratio, has a finite DC limit even
                # when the ratio rounds to zero. Scale its leading imaginary
                # term separately so a finite tiny reactance is retained.
                rm,re=frexp(rs);dm,de=frexp(rdc)
                leading=ldexp((2/3)*rm*rm/dm,2re-de)
                fourth=ratio^4
                # Separating the even/odd coth series retains the small
                # imaginary component without cancellation in complex tanh.
                real_factor=evalpoly(fourth,(1.,4/45,-16/4725,88448/638512875,
                    -925952/162820783125,357603328/1531329465290625,
                    -1936294633472/201919571963756521875))
                imaginary_factor=evalpoly(fourth,(1.,-8/315,32/31185,-256/6081075,
                    22459904/12993098493375,-318189568/4482618980214375))
                complex(rdc*real_factor,leading*imaginary_factor)
            elseif ratio>20
                # The relative coth correction is below 2exp(-40), already
                # below Float64 precision. This also avoids tanh(Inf+iInf).
                rf
            else
                rf/tanh(complex(ratio,ratio))
            end
        end
end

"""Native RES/SUP/SEN sheet impedance and NOR finite-thickness loss interpolation.
NOR's current ratio is I_top/I_bottom. Its zero-frequency limit is 1/(sigma*t),
and its RF limit is Zskin*(1+r^2)/(1+r)^2. Covers carry one-sided current.
NOR supports absent/CDVY conductivity in S/m, RSVY resistivity in ohm
centimetres, and SRVY DC sheet resistance in ohms/square. These selectors
do not use primitive DIM RES units. Thickness uses project length units.
Other metal models reject rather than losing roughness/thickness/plating."""
function sonnet_metal_zs(p::SonnetProject,metal::AbstractVector{<:AbstractString},
        freq::Real;variables=Dict{String,Float64}(),cover::Bool=false)
    isfinite(freq) && freq>0 || throw(ArgumentError("metal impedance requires finite positive frequency"))
    freq=_circuit_stored_real(freq,"native metal frequency")
    length(metal)>=3 || throw(ArgumentError("incomplete native metal definition"))
    model=metal[3]; val(t)=sonnet_variable_value(p,t;variables=variables,freq=freq)
    if model=="RES"
        length(metal)==4 || throw(ArgumentError("RES requires one parameter"))
        r=val(metal[4]); r>=0 || throw(ArgumentError("negative sheet resistance"))
        return complex(r)
    elseif model=="SEN"
        length(metal)==4 || throw(ArgumentError("SEN requires DC reactance in ohms per square"))
        return im*val(metal[4])
    elseif model=="SUP" || (model=="FREESPACE" && cover)
        length(metal)==7 || throw(ArgumentError("SUP requires four parameters"))
        rdc,rrf,xdc,ls=val.(metal[4:end])
        rdc>=0 && rrf>=0 || throw(ArgumentError("negative sheet resistance"))
        # Sonnet's general-loss Rdc/Rrf crossover is the conductor slab
        # reaction, with the DC and RF limits set independently.
        rs=rrf*sqrt(freq);rf=complex(rs,rs)
        zr=iszero(rrf) ? complex(rdc) : _sonnet_loss_transition(rdc,rs)
        # Evaluate kinetic pH reactance without an overflowing omega or an
        # underflowing SI inductance intermediate when the final result fits.
        lm,le=_circuit_omega_parts(freq,ls)
        lm,shift=frexp(lm*1e-12);le+=shift
        parts=(frexp(imag(zr)),frexp(xdc),(lm,le))
        exponent=maximum(m==0 ? typemin(Int) : e for (m,e) in parts)
        reactance=exponent==typemin(Int) ? 0. :
            ldexp(sum(ldexp(m,e-exponent) for (m,e) in parts if m!=0),exponent)
        zs=complex(real(zr),reactance)
        isfinite(zs) || throw(ArgumentError("SUP sheet impedance is not representable"))
        return zs
    elseif model=="NOR"
        length(metal) in (6,7) || throw(ArgumentError("NOR loss/current-ratio/thickness with optional selector required"))
        selector=length(metal)==7 ? metal[7] : "CDVY"
        selector in ("CDVY","RSVY","SRVY") || throw(ArgumentError("unsupported NOR loss selector $selector"))
        loss=selector=="CDVY" && uppercase(metal[4])=="INF" ? Inf : val(metal[4])
        ratio,thickness=val.(metal[5:6]); thickness*=p.length_scale
        selector=="CDVY" && loss==Inf && ratio>=0 && thickness>=0 && return 0.0im
        loss>=0 && ratio>=0 && thickness>0 && isfinite(thickness) || throw(ArgumentError("invalid NOR material parameters"))
        # Exact native selector controls establish fixed S/m, ohm-cm and
        # ohms/square respectively, distinct from primitive DIM RES units.
        rho=selector=="RSVY" ? loss*.01 : selector=="SRVY" ? loss*thickness : 0.
        if selector!="CDVY"
            isfinite(rho) && rho>=0 || throw(ArgumentError("unrepresentable NOR resistivity"))
            iszero(loss) && return 0.0im
            rho>0 || throw(ArgumentError("NOR resistivity underflows"))
        end
        sigma=selector=="CDVY" ? loss : inv(rho)
        sigma>0 && isfinite(sigma) || throw(ArgumentError("unrepresentable NOR conductivity"))
        skin=planar_surface_zs(freq,sigma)
        gamma=sigma*skin
        # r and 1/r have identical face-current weights. Reciprocal scaling
        # preserves the finite one-face limit without r^2 overflow.
        weight_ratio=ratio>1 ? inv(ratio) : ratio
        k=cover ? 1. : (1+weight_ratio^2)/(1+weight_ratio)^2
        # Rautio--Demir equation 11 scales Rrf by the face-current weights,
        # retaining the full-thickness DC resistance in the slab crossover.
        x=k*gamma*thickness
        zs=if abs(x)<1e-4
            k*skin*(inv(x)+x/3-x^3/45)
        else
            q=exp(-x)
            k*skin*(1+q*q)/(1-q*q)
        end
        isfinite(zs) || throw(ArgumentError("NOR sheet impedance is not representable"))
        return zs
    end
    throw(ArgumentError("native $model metal requires its dedicated material adapter"))
end

function _planar_via_mesh_mask(mask::BitMatrix,vertices,grid,mode::Symbol)
    mode==:full && return copy(mask)
    result=falses(size(mask))
    nx,ny=size(mask)
    if mode==:ring
        for j in 1:ny,i in 1:nx
            result[i,j]=mask[i,j] && (i==1 || j==1 || i==nx || j==ny ||
                !mask[i-1,j] || !mask[i+1,j] || !mask[i,j-1] || !mask[i,j+1])
        end
    elseif mode==:center
        center=(minimum(vertices;dims=2)+maximum(vertices;dims=2))/2
        i=clamp(floor(Int,center[1]/grid.dx)+1,1,nx)
        j=clamp(floor(Int,center[2]/grid.dy)+1,1,ny)
        mask[i,j] || throw(ArgumentError("via bounding-box center lies outside its raster footprint"))
        result[i,j]=true
    elseif mode==:vertices
        for v in eachcol(vertices)
            ci=floor(Int,v[1]/grid.dx);cj=floor(Int,v[2]/grid.dy)
            candidates=[CartesianIndex(i,j) for j in max(1,cj):min(ny,cj+1),
                i in max(1,ci):min(nx,ci+1) if mask[i,j]]
            isempty(candidates) && throw(ArgumentError("via vertex vanished on the selected grid"))
            distances=[((q[1]-.5)*grid.dx-v[1])^2+((q[2]-.5)*grid.dy-v[2])^2 for q in candidates]
            result[candidates[argmin(distances)]]=true
        end
    else
        throw(ArgumentError("via $mode meshing requires its dedicated current/geometry adapter"))
    end
    return result
end

# VOL SOLID uses the complete polygon area and an equivalent wall
# in its bounding rectangle. Separate native aspect/frequency/conductivity
# controls qualify this constitutive response. Rectangular hollow walls
# use exact cross-section loss; other wall shapes and horizontal via-pad
# loss require their own geometry/material adapters.
function _sonnet_volume_product(numerators,denominators;exponent::Int=0)
    any(iszero,numerators) && return 0.0
    mantissa=1.0
    for value in numerators
        m,e=frexp(value);mantissa*=m;exponent+=e
    end
    for value in denominators
        m,e=frexp(value);mantissa/=m;exponent-=e
    end
    return ldexp(mantissa,exponent)
end

# Translate to a local origin before the shoelace sum. Absolute products
# lose physical area when small vias lie far from the box origin.
function _sonnet_volume_polygon_area(vertices)
    x0,y0=vertices[1,1],vertices[2,1]
    return abs(sum((vertices[1,k]-x0)*(vertices[2,mod1(k+1,size(vertices,2))]-y0)-
        (vertices[1,mod1(k+1,size(vertices,2))]-x0)*(vertices[2,k]-y0)
        for k in axes(vertices,2)))/2
end

function _sonnet_volume_wall_depth(area,fill,width,height,rectangle::Bool=false)
    # A verified rectangular polygon fills its bounding box exactly;
    # preserve that identity before the square-root depth formula.
    fraction=rectangle ? fill : _sonnet_volume_product((area,fill),(width,height))
    fraction=min(fraction,1.0)
    shorter,longer=minmax(width,height);aspect=shorter/longer
    return shorter*(fraction/(1+aspect+sqrt((1-aspect)^2+4aspect*(1-fraction))))
end

function _sonnet_volume_sigma_wide(sigma,freq,area,percent,width,breadth,mesh_area;
        resistivity::Bool=false,sheet_resistance::Bool=false,declared_wall=nothing,rectangle::Bool=false)
    return setprecision(BigFloat,max(4096,precision(BigFloat))) do
        s,f,a,pct,w,b,am=BigFloat.((sigma,freq,area,percent,width,breadth,mesh_area))
        rectangle && (a=w*b)
        resistivity && (s=100/s)
        phi=pct/100
        shorter,longer=minmax(w,b);aspect=shorter/longer
        fraction=rectangle ? phi : min(a*phi/(w*b),one(BigFloat))
        depth=shorter*(fraction/(1+aspect+sqrt((1-aspect)^2+4aspect*(1-fraction))))
        sheet_resistance && (s=inv(s*(declared_wall===nothing ? depth : BigFloat(declared_wall))))
        rdc=one(BigFloat)
        rs=depth*sqrt(BigFloat(pi)*BigFloat(_MU0)*f*s)
        impedance=_sonnet_loss_transition(rdc,rs,"VOL")
        exact=(s*phi*a/am)/impedance;stored=ComplexF64(exact)
        isfinite(stored) && real(stored)>=0 && !iszero(stored) &&
            (iszero(real(exact)) || !iszero(real(stored))) &&
            (iszero(imag(exact)) || !iszero(imag(stored))) ||
            throw(ArgumentError("native VOL effective conductivity is unrepresentable"))
        return stored
    end
end

# The polygon adapter still needs general inward offsets for hollow shapes.
# Exact axis-aligned rectangular boundaries are handled without raster area.
function _sonnet_volume_rectangle(vertices)
    xmin,xmax=extrema(@view vertices[1,:]);ymin,ymax=extrema(@view vertices[2,:])
    for k in axes(vertices,2)
        x,y=vertices[1,k],vertices[2,k]
        (x==xmin || x==xmax || y==ymin || y==ymax) || return false
        next=mod1(k+1,size(vertices,2))
        (x==vertices[1,next] || y==vertices[2,next]) || return false
    end
    return true
end

function _sonnet_volume_hollow_percent(wall,unit,width,breadth)
    thickness=wall*unit
    isfinite(thickness) && thickness>=floatmin(Float64) || return nothing
    thickness>=min(width,breadth)/2 && return 100.0
    u=thickness/width;v=thickness/breadth
    percent=100*(2u+2v-4u*v)
    isfinite(percent) && percent>0 || return nothing
    return percent
end

function _sonnet_volume_hollow_wide(loss,freq,area,wall,unit,width,breadth,mesh_area;
        resistivity=false,sheet_resistance=false,rectangle::Bool=false)
    setprecision(BigFloat,max(4096,precision(BigFloat))) do
        w,b,t=BigFloat(width),BigFloat(breadth),BigFloat(wall)*BigFloat(unit)
        declared=t
        t=min(t,min(w,b)/2)
        percent=100*(2t*(w+b-2t)/(w*b))
        _sonnet_volume_sigma_wide(loss,freq,area,percent,width,breadth,mesh_area;
            resistivity,sheet_resistance,declared_wall=declared,rectangle)
    end
end

# General thin mitered wall. Offset edge reversals mark unresolved topology
# changes; those cases retain an explicit failure until separately modeled.
function _sonnet_volume_simple_wall(vertices,nvertices)
    for i in 1:nvertices, j in i+1:nvertices
        (j==i+1 || (i==1 && j==nvertices)) && continue
        next_i=mod1(i+1,nvertices);next_j=mod1(j+1,nvertices)
        ax,ay=vertices[1,i],vertices[2,i]
        bx,by=vertices[1,next_i],vertices[2,next_i]
        cx,cy=vertices[1,j],vertices[2,j]
        dx,dy=vertices[1,next_j],vertices[2,next_j]
        max(min(ax,bx),min(cx,dx))<=min(max(ax,bx),max(cx,dx)) &&
            max(min(ay,by),min(cy,dy))<=min(max(ay,by),max(cy,dy)) || continue
        a=(bx-ax)*(cy-ay)-(by-ay)*(cx-ax)
        b=(bx-ax)*(dy-ay)-(by-ay)*(dx-ax)
        ((a<=0 && b>=0)||(a>=0 && b<=0)) || continue
        c=(dx-cx)*(ay-cy)-(dy-cy)*(ax-cx)
        d=(dx-cx)*(by-cy)-(dy-cy)*(bx-cx)
        ((c<=0 && d>=0)||(c>=0 && d<=0)) && return false
    end
    return true
end

# Return nothing when rounded geometry cannot certify the thin-wall domain.
# The wide path then evaluates the original vertices, wall, units and loss.
function _sonnet_volume_simple_wall_fast(vertices,nvertices,tolerance)
    for i in 1:nvertices, j in i+1:nvertices
        (j==i+1 || (i==1 && j==nvertices)) && continue
        next_i=mod1(i+1,nvertices);next_j=mod1(j+1,nvertices)
        ax,ay=vertices[1,i],vertices[2,i];bx,by=vertices[1,next_i],vertices[2,next_i]
        cx,cy=vertices[1,j],vertices[2,j];dx,dy=vertices[1,next_j],vertices[2,next_j]
        (max(min(ax,bx),min(cx,dx))-min(max(ax,bx),max(cx,dx))>2tolerance ||
            max(min(ay,by),min(cy,dy))-min(max(ay,by),max(cy,dy))>2tolerance) && continue
        a=(bx-ax)*(cy-ay)-(by-ay)*(cx-ax)
        b=(bx-ax)*(dy-ay)-(by-ay)*(dx-ax)
        all(abs(x)>64tolerance for x in (a,b)) || return nothing
        signbit(a)==signbit(b) && continue
        c=(dx-cx)*(ay-cy)-(dy-cy)*(ax-cx)
        d=(dx-cx)*(by-cy)-(dy-cy)*(bx-cx)
        all(abs(x)>64tolerance for x in (c,d)) || return nothing
        signbit(c)==signbit(d) && continue
        return nothing
    end
    return true
end

function _sonnet_volume_polygon_sigma_fast(p,poly,grid,mask,freq,loss,wall,selector)
    source=poly.vertices;nvertices=size(source,2)
    source[1,1]==source[1,end] && source[2,1]==source[2,end] && (nvertices-=1)
    3<=nvertices<=64 || return nothing
    xmin,xmax=extrema(@view source[1,:]);ymin,ymax=extrema(@view source[2,:])
    scale=max(xmax-xmin,ymax-ymin)
    isfinite(scale) && scale>=floatmin(Float64) || return nothing
    thickness=wall*p.length_scale
    isfinite(thickness) && thickness>=floatmin(Float64) || return nothing
    t=thickness/scale;tolerance=512eps(Float64)*nvertices
    isfinite(t) && t>tolerance || return nothing
    conductivity=selector=="RSVY" ? _sonnet_volume_product((100.,),(loss,)) :
        selector=="SRVY" ? _sonnet_volume_product((1.,),(loss,wall,p.length_scale)) : loss
    isfinite(conductivity) && conductivity>=floatmin(Float64) || return nothing
    vertices=Matrix{Float64}(undef,2,nvertices)
    for k in 1:nvertices
        vertices[1,k]=(source[1,k]-source[1,1])/scale
        vertices[2,k]=(source[2,k]-source[2,1])/scale
    end
    _sonnet_volume_simple_wall_fast(vertices,nvertices,tolerance)===true || return nothing
    signed=0.0;magnitude=0.0
    for k in 1:nvertices
        next=mod1(k+1,nvertices)
        a=vertices[1,k]*vertices[2,next];b=vertices[1,next]*vertices[2,k]
        signed+=a-b;magnitude+=abs(a)+abs(b)
    end
    abs(signed)>64tolerance*max(magnitude,1.0) || return nothing
    orientation=sign(signed);area=abs(signed)/2
    lengths=Vector{Float64}(undef,nvertices);corners=Vector{Float64}(undef,nvertices)
    inset=Matrix{Float64}(undef,2,nvertices)
    for k in 1:nvertices
        previous=mod1(k-1,nvertices);following=mod1(k+1,nvertices)
        ax=vertices[1,k]-vertices[1,previous];ay=vertices[2,k]-vertices[2,previous]
        bx=vertices[1,following]-vertices[1,k];by=vertices[2,following]-vertices[2,k]
        la=hypot(ax,ay);lb=hypot(bx,by)
        la>sqrt(eps(Float64)) && lb>sqrt(eps(Float64)) || return nothing
        ax/=la;ay/=la;bx/=lb;by/=lb
        cosine=ax*bx+ay*by;sine=orientation*(ax*by-ay*bx)
        denominator=1+cosine
        denominator>=.125 || return nothing
        lengths[k]=lb;corners[k]=sine/denominator
        inset[1,k]=vertices[1,k]-t*orientation*(ay+by)/denominator
        inset[2,k]=vertices[2,k]+t*orientation*(ax+bx)/denominator
    end
    _sonnet_volume_simple_wall_fast(inset,nvertices,32tolerance)===true || return nothing
    for k in 1:nvertices
        remaining=lengths[k]-t*(corners[k]+corners[mod1(k+1,nvertices)])
        remaining>64tolerance || return nothing
    end
    metal_area=t*(sum(lengths)-t*sum(corners))
    64tolerance<metal_area<area-64tolerance || return nothing
    q=_sonnet_volume_product((thickness,sqrt(pi*_MU0),sqrt(freq),sqrt(conductivity)),())
    isfinite(q) && q>=floatmin(Float64) || return nothing
    transition=_sonnet_loss_transition(1.0,q,"VOL")
    isfinite(transition) && !_planar_metal_subnormal(transition) &&
        !iszero(real(transition)) && !iszero(imag(transition)) || return nothing
    rm,re=frexp(real(transition));im,ie=frexp(imag(transition));exponent=max(re,ie)
    denominator=ldexp(rm,re-exponent)^2+ldexp(im,ie-exponent)^2
    mesh_count=count(mask)
    real_sigma=_sonnet_volume_product((conductivity,metal_area,scale,scale,rm),
        (Float64(mesh_count),grid.dx,grid.dy,denominator);exponent=re-2exponent)
    imag_sigma=-_sonnet_volume_product((conductivity,metal_area,scale,scale,im),
        (Float64(mesh_count),grid.dx,grid.dy,denominator);exponent=ie-2exponent)
    stored=complex(real_sigma,imag_sigma)
    isfinite(stored) && real(stored)>0 && !iszero(imag(stored)) &&
        !_planar_metal_subnormal(stored) || return nothing
    return stored
end

function _sonnet_volume_polygon_sigma(p,poly,grid,stack,mask,freq,loss,wall,selector)
    fast=_sonnet_volume_polygon_sigma_fast(p,poly,grid,mask,freq,loss,wall,selector)
    fast===nothing || return fast
    return _sonnet_volume_polygon_sigma_wide(p,poly,grid,stack,mask,freq,loss,wall,selector)
end

function _sonnet_volume_polygon_sigma_wide(p,poly,grid,stack,mask,freq,loss,wall,selector)
    return setprecision(BigFloat,max(4096,precision(BigFloat))) do
        vertices=BigFloat.(poly.vertices)
        nvertices=size(vertices,2)
        vertices[:,1]==vertices[:,end] && (nvertices-=1)
        nvertices>=3 || throw(ArgumentError("native hollow VOL polygon needs three distinct vertices"))
        _sonnet_volume_simple_wall(vertices,nvertices) ||
            throw(ArgumentError("native hollow VOL polygon has a crossing or touching boundary"))
        signed=sum(vertices[1,k]*vertices[2,mod1(k+1,nvertices)]-vertices[1,mod1(k+1,nvertices)]*vertices[2,k] for k in 1:nvertices)
        orientation=sign(signed);area=abs(signed)/2
        area>0 || throw(ArgumentError("native hollow VOL polygon has zero area"))
        lengths=Vector{BigFloat}(undef,nvertices);corners=Vector{BigFloat}(undef,nvertices)
        inset=Matrix{BigFloat}(undef,2,nvertices)
        thickness=BigFloat(wall)*BigFloat(p.length_scale)
        for k in 1:nvertices
            previous=mod1(k-1,nvertices);following=mod1(k+1,nvertices)
            ax=vertices[1,k]-vertices[1,previous];ay=vertices[2,k]-vertices[2,previous]
            bx=vertices[1,following]-vertices[1,k];by=vertices[2,following]-vertices[2,k]
            la=hypot(ax,ay);lb=hypot(bx,by)
            la>0 && lb>0 || throw(ArgumentError("native hollow VOL contains a repeated vertex"))
            cosine=(ax*bx+ay*by)/(la*lb);sine=orientation*(ax*by-ay*bx)/(la*lb)
            denominator=1+cosine
            denominator>0 || throw(ArgumentError("native hollow VOL has a reversing boundary"))
            lengths[k]=lb;corners[k]=sine/denominator
            inset[1,k]=vertices[1,k]-thickness*orientation*(ay/la+by/lb)/denominator
            inset[2,k]=vertices[2,k]+thickness*orientation*(ax/la+bx/lb)/denominator
        end
        _sonnet_volume_simple_wall(inset,nvertices) ||
            throw(ArgumentError("native hollow VOL inset crosses or touches; its topology adapter is required"))
        all(k->lengths[k]-thickness*(corners[k]+corners[mod1(k+1,nvertices)])>0,1:nvertices) ||
            throw(ArgumentError("native hollow VOL wall changes polygon topology; its thick-wall adapter is required"))
        metal_area=thickness*(sum(lengths)-thickness*sum(corners))
        0<metal_area<area || throw(ArgumentError("native hollow VOL wall requires its saturation/topology adapter"))
        conductivity=selector=="RSVY" ? BigFloat(100)/BigFloat(loss) :
            selector=="SRVY" ? inv(BigFloat(loss)*thickness) : BigFloat(loss)
        mesh=BigFloat(count(mask))*BigFloat(grid.dx)*BigFloat(grid.dy)
        q=thickness*sqrt(BigFloat(pi)*BigFloat(_MU0)*BigFloat(freq)*conductivity)
        transition=_sonnet_loss_transition(one(BigFloat),q,"VOL")
        exact=conductivity*metal_area/(mesh*transition);stored=ComplexF64(exact)
        isfinite(stored) && real(stored)>0 && !iszero(stored) &&
            (iszero(real(exact)) || !iszero(real(stored))) &&
            (iszero(imag(exact)) || !iszero(imag(stored))) ||
            throw(ArgumentError("native hollow VOL effective conductivity is unrepresentable"))
        return stored
    end
end

# Evaluate cold wide recovery only when required; no captured hot-path closure.
function _sonnet_volume_sigma_fallback(sigma,loss,freq,area,percent,width,breadth,mesh_area,wall,unit,selector,solid,rectangle::Bool=false)::ComplexF64
    if selector=="SRVY"
        return solid ? _sonnet_volume_sigma_wide(loss,freq,area,percent,width,breadth,mesh_area;sheet_resistance=true,rectangle) :
            _sonnet_volume_hollow_wide(loss,freq,area,wall,unit,width,breadth,mesh_area;sheet_resistance=true,rectangle)
    end
    return _sonnet_volume_sigma_wide(sigma,freq,area,percent,width,breadth,mesh_area;rectangle)
end

function _sonnet_volume_sigma(p,poly,grid,stack,mask,freq,vars)
    metal=p.metals[poly.material+1]
    solid=length(metal)>=5 && metal[5]=="SOLID"
    length(metal) in (solid ? (6,7) : (5,6)) || throw(ArgumentError("unsupported native VOL conductivity fields"))
    selector=length(metal)==(solid ? 7 : 6) ? metal[end] : "CDVY"
    selector in ("CDVY","RSVY","SRVY") || throw(ArgumentError("unsupported VOL loss selector $selector"))
    val(t)=sonnet_variable_value(p,t;variables=vars,freq=freq)
    loss=selector=="CDVY" && uppercase(metal[4])=="INF" ? Inf : val(metal[4])
    # A SOLID record retains an inactive wall-thickness field. Validate
    # its domain while the complete polygon determines the physical area.
    wall=val(metal[solid ? 6 : 5]);percent=100.0
    isfinite(wall) && (solid ? wall>=0 : wall>0) && loss>=0 ||
        throw(ArgumentError("VOL requires nonnegative loss and a positive hollow wall thickness"))

    (selector=="CDVY" && loss==Inf) || (selector in ("RSVY","SRVY") && iszero(loss)) ? (return Inf) : nothing
    rectangle=_sonnet_volume_rectangle(poly.vertices)
    !solid && !rectangle &&
        return _sonnet_volume_polygon_sigma(p,poly,grid,stack,mask,freq,loss,wall,selector)
    sheet_resistance=selector=="SRVY"
    sigma=selector=="CDVY" ? loss : sheet_resistance ? 0.0 : _sonnet_volume_product((100.,),(loss,))
    (isfinite(sigma) && sigma>0) || (selector=="RSVY" && loss>0) || sheet_resistance ||
        throw(ArgumentError("native VOL conductivity is unrepresentable"))
    vertices=poly.vertices
    area=_sonnet_volume_polygon_area(vertices)
    xmin,xmax=extrema(@view vertices[1,:]);ymin,ymax=extrema(@view vertices[2,:])
    width=xmax-xmin;breadth=ymax-ymin
    layers=length(stack.layers)
    target=poly.target=="GND" ? layers-1 : poly.target=="TOP" ? -1 : parse(Int,poly.target)
    lo,hi=minmax(poly.level,target)
    height=sum(real(stack.layers[layers-lev].thickness) for lev in lo+1:hi)
    mesh_area=count(mask)*grid.dx*grid.dy
    all(x->isfinite(x) && x>0,(area,width,breadth,height,mesh_area)) ||
        throw(ArgumentError("native VOL needs representable positive physical height and cross-sectional areas"))
    if !solid
        percent=_sonnet_volume_hollow_percent(wall,p.length_scale,width,breadth)
        percent===nothing && return _sonnet_volume_hollow_wide(loss,freq,area,wall,p.length_scale,width,breadth,mesh_area;
            resistivity=selector=="RSVY",sheet_resistance,rectangle)
    end
    fill=percent/100
    if sheet_resistance
        depth=solid ? _sonnet_volume_wall_depth(area,fill,width,breadth,rectangle) : wall*p.length_scale
        sigma=solid ? _sonnet_volume_product((1.,),(loss,depth)) :
            _sonnet_volume_product((1.,),(loss,wall,p.length_scale))
    end
    # Preserve the original sheet/volume resistivity before reciprocal or SI products.
    if !(isfinite(sigma) && sigma>0)
        return sheet_resistance ? _sonnet_volume_sigma_fallback(sigma,loss,freq,area,percent,width,breadth,mesh_area,wall,p.length_scale,selector,solid,rectangle) :
            _sonnet_volume_sigma_wide(loss,freq,area,percent,width,breadth,mesh_area;resistivity=true,rectangle)
    end
    # Cancel physical height before forming the normalized slab transition.
    # This also avoids unnecessary large/small total-resistance intermediates.
    depth=_sonnet_volume_wall_depth(area,fill,width,breadth,rectangle)
    rdc=1.0
    rs=_sonnet_volume_product((depth,sqrt(pi*_MU0),sqrt(freq),sqrt(sigma)),())
    all(isfinite,(rdc,rs,depth)) && depth>0 || return _sonnet_volume_sigma_fallback(sigma,loss,freq,area,percent,width,breadth,mesh_area,wall,p.length_scale,selector,solid,rectangle)
    impedance=_sonnet_loss_transition(rdc,rs,"VOL")
    isfinite(impedance) && !iszero(real(impedance)) && !iszero(imag(impedance)) &&
        !_planar_metal_subnormal(impedance) || return _sonnet_volume_sigma_fallback(sigma,loss,freq,area,percent,width,breadth,mesh_area,wall,p.length_scale,selector,solid,rectangle)
    rm,re=frexp(real(impedance));im,ie=frexp(imag(impedance));e=max(re,ie)
    denominator=ldexp(rm,re-e)^2+ldexp(im,ie-e)^2
    real_sigma=_sonnet_volume_product((sigma,fill,area,rm),(mesh_area,denominator);exponent=re-2e)
    imag_sigma=-_sonnet_volume_product((sigma,fill,area,im),(mesh_area,denominator);exponent=ie-2e)
    stored=complex(real_sigma,imag_sigma)
    isfinite(stored) && real(stored)>0 && !iszero(imag(stored)) || return _sonnet_volume_sigma_fallback(sigma,loss,freq,area,percent,width,breadth,mesh_area,wall,p.length_scale,selector,solid,rectangle)
    return stored
end

function _sonnet_volume_polygon_workspace(p)
    peak_vertices=0;needs_wide=false
    for poly in p.polygons
        poly.kind===:via && 0<=poly.material<length(p.metals) || continue
        metal=p.metals[poly.material+1]
        length(metal)>=5 && metal[3]=="VOL" || continue
        needs_wide=true
        if !("RPV" in metal) && metal[5]!="SOLID" && !_sonnet_volume_rectangle(poly.vertices)
            peak_vertices=max(peak_vertices,size(poly.vertices,2))
        end
    end
    needs_wide || return 0
    # One material is evaluated at a time. Every non-RPV VOL path can need
    # wide scalar recovery; SOLID/rectangle need scratch without vertex copies.
    # General hollow paths also reserve six scalars per vertex and fast arrays.
    # BigFloat storage includes the MPFR limbs at the owned/caller precision;
    # the header allowance covers the supported Julia1.12/1.13 representations.
    limbs=8cld(BigInt(max(4096,precision(BigFloat))),64)
    return _checked_payload_sum("native volume material workspace",
        _checked_array_payload_bytes(Float64,6,peak_vertices),
        (6BigInt(peak_vertices)+224)*(96+limbs),256)
end

function _sonnet_has_volume_skin(p)
    return any(p.polygons) do poly
        poly.kind===:via && 0<=poly.material<length(p.metals) &&
            length(p.metals[poly.material+1])>=3 && p.metals[poly.material+1][3]=="VOL" &&
            !("RPV" in p.metals[poly.material+1])
    end
end

# RPV specifies a frequency-independent resistance for the complete axial
# polygon, independent of the subsection mesh. Convert it to the Ohmic Gram
# conductivity of the retained U/T cells, preserving the total resistance.
# Native VOL endpoints carry tangential current as well as axial current.
# COVERS fills the polygon; NOCOVERS retains the subsection Ring footprint.
function _sonnet_via_endpoints(p,poly)
    poly.kind===:via || return false
    poly.material==-1 && return true
    0<=poly.material<length(p.metals) || return false
    metal=p.metals[poly.material+1]
    return length(metal)>=3 && metal[3]=="VOL"
end

function _sonnet_via_parallel_zs(a,b)
    (iszero(a) || iszero(b)) && return 0.0im
    # Scale by the smaller impedance to avoid reciprocal overflow. Both
    # native passive films share one tangential electric field.
    scale(z)=max(abs(real(z)),abs(imag(z)))
    small,large=scale(a)<=scale(b) ? (a,b) : (b,a)
    combined=small/(1+small/large)
    isfinite(combined) && real(combined)>0 || throw(ArgumentError(
        "native parallel endpoint impedance is unrepresentable"))
    return combined
end

function _sonnet_via_endpoint_height(p,poly,stack)
    layers=length(stack.layers)
    target=poly.target=="GND" ? layers-1 : poly.target=="TOP" ? -1 : parse(Int,poly.target)
    lo,hi=minmax(poly.level,target)
    height=sum(real(stack.layers[layers-level].thickness) for level in lo+1:hi)
    isfinite(height) && height>0 || throw(ArgumentError("native via endpoint needs a positive representable physical height"))
    return height
end

function _sonnet_via_endpoint_sigma(p,poly,metal,loss,selector)
    selector=="CDVY" && return loss
    selector=="RSVY" && return _sonnet_volume_product((100.,),(loss,))
    vertices=poly.vertices
    xmin,xmax=extrema(@view vertices[1,:]);ymin,ymax=extrema(@view vertices[2,:])
    area=_sonnet_volume_polygon_area(vertices)
    depth=metal[5]=="SOLID" ? _sonnet_volume_wall_depth(area,1.0,xmax-xmin,ymax-ymin,_sonnet_volume_rectangle(vertices)) :
        sonnet_variable_value(p,metal[5])*p.length_scale
    return _sonnet_volume_product((1.,),(loss,depth))
end

function _sonnet_via_endpoint_wide(p,poly,stack,freq,loss,selector,wall)
    setprecision(BigFloat,max(4096,precision(BigFloat))) do
        height=BigFloat(_sonnet_via_endpoint_height(p,poly,stack))
        value=BigFloat(loss)
        sigma=if selector=="RPV"
            2/(value*height)
        elseif selector=="RSVY"
            100/value
        elseif selector=="SRVY"
            metal=p.metals[poly.material+1]
            depth=if metal[5]=="SOLID"
                vertices=poly.vertices
                xmin,xmax=extrema(@view vertices[1,:]);ymin,ymax=extrema(@view vertices[2,:])
                xmin,xmax,ymin,ymax=BigFloat.((xmin,xmax,ymin,ymax))
                area=zero(BigFloat)
                x0,y0=BigFloat(vertices[1,1]),BigFloat(vertices[2,1])
                for k in axes(vertices,2)
                    next=mod1(k+1,size(vertices,2))
                    ax,ay=BigFloat(vertices[1,k])-x0,BigFloat(vertices[2,k])-y0
                    bx,by=BigFloat(vertices[1,next])-x0,BigFloat(vertices[2,next])-y0
                    area+=ax*by-ay*bx
                end
                area=abs(area)/2
                width,breadth=xmax-xmin,ymax-ymin
                shorter,longer=minmax(width,breadth);aspect=shorter/longer
                fraction=min(area/(width*breadth),one(BigFloat))
                shorter*fraction/(1+aspect+sqrt((1-aspect)^2+4aspect*(1-fraction)))
            else
                BigFloat(wall)*BigFloat(p.length_scale)
            end
            inv(value*depth)
        else
            value
        end
        sigma>0 && isfinite(sigma) || throw(ArgumentError("native via endpoint conductivity is invalid"))
        exact=planar_two_sheet_zs(BigFloat(freq),sigma,height)[1,1]
        stored=ComplexF64(exact)
        isfinite(stored) && (iszero(real(exact)) || !iszero(real(stored))) &&
            (iszero(imag(exact)) || !iszero(imag(stored))) ||
            throw(ArgumentError("native via endpoint film impedance is unrepresentable"))
        return stored
    end
end

function _sonnet_via_endpoint_zs(p,poly,stack,freq,variables)
    poly.material==-1 && return 0.0im
    metal=p.metals[poly.material+1]
    metal[3]=="VOL" || throw(ArgumentError("native via endpoint material needs its own adapter"))
    solid=length(metal)>=5 && metal[5]=="SOLID"
    rpv="RPV" in metal
    (!solid && !rpv && !("COVERS" in poly.flags)) && return 0.0im
    selector=rpv ? "RPV" : length(metal)==(solid ? 7 : 6) ? metal[end] : "CDVY"
    value=selector=="CDVY" && uppercase(metal[4])=="INF" ? Inf :
        sonnet_variable_value(p,metal[4];variables,freq)
    (selector=="CDVY" && value==Inf) || (selector!="CDVY" && iszero(value)) ? (return 0.0im) : nothing
    value>0 && isfinite(value) || throw(ArgumentError("native via endpoint loss must be positive and finite"))
    height=_sonnet_via_endpoint_height(p,poly,stack)
    wall=solid ? 0.0 : sonnet_variable_value(p,metal[5];variables,freq)
    sigma=if rpv
        _sonnet_volume_product((2.,),(value,height))
    elseif selector=="SRVY" && !solid
        _sonnet_volume_product((1.,),(value,wall,p.length_scale))
    else
        _sonnet_via_endpoint_sigma(p,poly,metal,value,selector)
    end
    isfinite(sigma) && sigma>0 || return _sonnet_via_endpoint_wide(p,poly,stack,freq,value,selector,wall)
    return ComplexF64(planar_two_sheet_zs(freq,sigma,height)[1,1])
end

function _sonnet_via_sigma(p,poly,grid,stack,mask,freq,vars)
    poly.material==-1 && return Inf
    metal=p.metals[poly.material+1]
    val(t)=sonnet_variable_value(p,t;variables=vars,freq=freq)
    model=metal[3]
    if model=="VOL" && !("RPV" in metal)
        return _sonnet_volume_sigma(p,poly,grid,stack,mask,freq,vars)
    end
    "RPV" in metal || throw(ArgumentError("native $model via conductivity/skin loss requires its dedicated frequency-dependent adapter"))
    "COVERS" in poly.flags && model!="VOL" && throw(ArgumentError("lossy native via covers require the horizontal pad-loss adapter"))
    resistance=val(metal[4])
    resistance>=0 || throw(ArgumentError("negative native resistance per via"))
    if model=="VOL"
        solid=length(metal)>=5 && metal[5]=="SOLID"
        length(metal)==(solid ? 7 : 6) && metal[end]=="RPV" || throw(ArgumentError("unsupported VOL RPV fields"))
        wall=val(metal[solid ? 6 : 5])
        isfinite(wall) && wall>=0 || throw(ArgumentError("VOL RPV inactive wall thickness must be finite and nonnegative"))
        solid || iszero(wall) || throw(ArgumentError("VOL RPV hollow wall-thickness semantics require the dedicated adapter"))
    elseif model=="ARR"
        length(metal)==7 && metal[6]=="RPV" || throw(ArgumentError("unsupported ARR RPV fields"))
        density=val(metal[7]);density>0 || throw(ArgumentError("ARR RPV density must be positive vias per square micron"))
        v=poly.vertices
        area=abs(sum(v[1,i]*v[2,mod1(i+1,size(v,2))]-v[1,mod1(i+1,size(v,2))]*v[2,i] for i in axes(v,2)))/2
        area>0 || throw(ArgumentError("ARR RPV polygon has zero physical cross-sectional area"))
        resistance/=density*area*1e12
    else
        throw(ArgumentError("native $model does not define resistance per axial via"))
    end
    iszero(resistance) && return Inf
    L=length(stack.layers)
    target=poly.target=="GND" ? L-1 : poly.target=="TOP" ? -1 : parse(Int,poly.target)
    lo,hi=minmax(poly.level,target)
    height=sum(real(stack.layers[L-lev].thickness) for lev in lo+1:hi)
    area_mesh=count(mask)*grid.dx*grid.dy
    sigma=height/(resistance*area_mesh)
    isfinite(sigma) && sigma>0 || throw(ArgumentError("native via resistance cannot be represented at this floating-point/raster resolution"))
    return sigma
end

function _sonnet_thick_geometry(p::SonnetProject,freq,variables;
        max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    any(q->q.kind===:sheet && 0<=q.material<length(p.metals) &&
        p.metals[q.material+1][3]=="TMM",p.polygons) || return p
    stack_workspace=_checked_payload_sum("native TMM stack preflight",
        _checked_array_payload_bytes(PlanarLayer{ComplexF64},2,length(p.layers)),
        _checked_array_payload_bytes(Ptr{Cvoid},4,length(p.layers)))
    _enforce_payload_limit(stack_workspace,max_bytes,"native TMM stack preflight","max_bytes")
    _enforce_payload_limit(_sonnet_raster_source_workspace(p),max_bytes,
        "native TMM source workspace","max_bytes")
    thick=[q for q in p.polygons if q.kind==:sheet && q.material>=0 &&
        q.material<length(p.metals) && p.metals[q.material+1][3]=="TMM"]
    isempty(thick) && return p
    val(t)=sonnet_variable_value(p,t;variables=variables,freq=freq)
    oldL=length(p.layers)
    bottom_rows=reverse(p.layers)
    boundaries=vcat(0.,cumsum([val(row[1])*p.length_scale for row in bottom_rows]))
    all(diff(boundaries).>0) || throw(ArgumentError("TMM stack layers must have positive thickness"))
    zlevel(level)=boundaries[oldL-level]
    endpoints=Dict{Int,Float64}();face_material=Dict{Int,Int}()
    metals=deepcopy(p.metals);newz=copy(boundaries)
    for q in thick
        row=p.metals[q.material+1]
        length(row)>=7 || throw(ArgumentError("incomplete TMM definition"))
        extras=row[8:end]
        all(t->t in ("CDVY","TDWN"),extras) ||
            throw(ArgumentError("TMM resistivity/sheet-resistance or unknown flags require their material adapter"))
        nsheets=val(row[7])
        nsheets==2 || throw(ArgumentError("native TMM currently requires exactly two physical faces"))
        sigma=uppercase(row[4])=="INF" ? Inf : val(row[4])
        thickness=val(row[6])*p.length_scale
        thickness>0 && sigma>0 || throw(ArgumentError("invalid TMM conductivity or physical thickness"))
        base=zlevel(q.level);up=!("TDWN" in extras)
        face=base+(up ? thickness : -thickness)
        tolerance=64eps(maximum(boundaries))
        nearest=argmin(abs.(boundaries.-face))
        abs(boundaries[nearest]-face)<=tolerance && (face=boundaries[nearest])
        host=up ? oldL-q.level : oldL-q.level-1
        1<=host<=oldL || throw(ArgumentError("TMM extends beyond the native box"))
        boundaries[host]<=face<=boundaries[host+1] ||
            throw(ArgumentError("TMM polygon $(q.id) crosses its host dielectric layer"))
        endpoints[q.id]=face;push!(newz,face)
        if !haskey(face_material,q.material)
            zface=sigma==Inf ? 0im : planar_two_sheet_zs(freq,sigma,thickness)[1,1]
            push!(metals,[row[1]*"_physical_face",row[2],"SUP",string(real(zface)),
                "0",string(imag(zface)),"0"])
            face_material[q.material]=length(metals)-1
        end
    end
    sort!(newz);unique!(newz)
    newL=length(newz)-1
    newlevel(z)=newL-findfirst(==(z),newz)
    levelmap=Dict(level=>newlevel(zlevel(level)) for level in -1:oldL-1)
    layers=Vector{String}[]
    for k in newL:-1:1
        center=(newz[k]+newz[k+1])/2
        host=searchsortedlast(boundaries,center)
        row=copy(bottom_rows[host]);row[1]=string((newz[k+1]-newz[k])/p.length_scale)
        push!(layers,row)
    end
    nextid=maximum(q->q.id,p.polygons;init=0)+1
    polys=SonnetPolygon[];duplicate=Dict{Int,Int}()
    for q in p.polygons
        target=q.kind==:via && !(q.target in ("GND","TOP")) ?
            string(levelmap[parse(Int,q.target)]) : q.target
        material=haskey(endpoints,q.id) ? face_material[q.material] : q.material
        push!(polys,SonnetPolygon(q.kind,levelmap[q.level],material,q.id,q.vertices,
            target,q.technology,copy(q.flags)))
        if haskey(endpoints,q.id)
            otherlevel=newlevel(endpoints[q.id]);otherid=nextid;nextid+=1
            duplicate[q.id]=otherid
            push!(polys,SonnetPolygon(:sheet,otherlevel,material,otherid,q.vertices,
                "",q.technology,copy(q.flags)))
            push!(polys,SonnetPolygon(:via,levelmap[q.level],-1,nextid,q.vertices,
                string(otherlevel),"",["RING","NOCOVERS"]))
            nextid+=1
        end
    end
    ports=SonnetPortSpec[]
    for port in p.ports
        push!(ports,port)
        if haskey(duplicate,port.polygon)
            port.kind in (:box,:std) || throw(ArgumentError("thick metal interior ports require face-source calibration"))
            push!(ports,SonnetPortSpec(port.kind,duplicate[port.polygon],port.edge,port.number,
                copy(port.values),copy(port.records)))
        end
    end
    components=Vector{SonnetRecord}[]
    for component in p.components
        records=SonnetRecord[]
        for record in component
            tokens=copy(record.tokens)
            tokens[1]=="SMDP" && (tokens[2]=string(levelmap[parse(Int,tokens[2])]))
            push!(records,SonnetRecord(record.line,tokens))
        end
        push!(components,records)
    end
    box=copy(p.box);box[1]=string(newL-1)
    return SonnetProject(p.source,p.units,p.length_scale,p.frequency_scale,box,layers,
        metals,p.top,p.bottom,polys,ports,p.variables,components,p.sweeps,p.records)
end

function _sonnet_dielectric_sigma(p::SonnetProject,row,freq,variables)
    # Native GUI exports place the optional loss selection after the layer
    # name. Actual engine controls establish dielectric RSVY as ohm-cm;
    # these units are independent of project DIM LNG/RES and metal flags.
    suffix=length(row)>8 ? row[9:end] : String[]
    isempty(suffix) || suffix==["RSVY"] ||
        throw(ArgumentError("unsupported native dielectric loss/anisotropy flags "*join(suffix,", ")))
    value=sonnet_variable_value(p,row[6];variables,freq)
    isfinite(value) && value>=0 || throw(ArgumentError("native dielectric loss must be finite and nonnegative"))
    isempty(suffix) && return value
    value>0 || throw(ArgumentError("native dielectric RSVY must be positive ohm-cm"))
    rho=value*1e-2
    isfinite(rho) && rho>0 || throw(ArgumentError("native dielectric resistivity is unrepresentable"))
    sigma=inv(rho)
    isfinite(sigma) && sigma>0 || throw(ArgumentError("native dielectric conductivity is unrepresentable"))
    return sigma
end

# Form sigma/(omega*epsilon0) without overflowing omega or rounding a tiny
# denominator to zero before division. Only the final loss ratio is scaled;
# a physically absent conductive loss stays exactly zero at every frequency.
function _sonnet_dielectric_conduction(sigma::Real,freq::Real)
    iszero(sigma) && return zero(float(freq))
    sm,se=frexp(sigma);fm,fe=frexp(freq)
    loss=ldexp(sm/(fm*(2pi*_EPS0)),se-fe)
    isfinite(loss) || throw(ArgumentError("native dielectric conductive loss is unrepresentable"))
    return loss
end

function _sonnet_stack_geometry(p::SonnetProject,freq::Real,grid,variables;
        expand_thick::Bool=true,max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    isfinite(freq) && freq>0 || throw(ArgumentError("frequency must be positive and finite"))
    freq=_circuit_stored_real(freq,"native geometry frequency")
    if !(variables isa SonnetScalarVariables) && _sonnet_has_scalar_tables(p)
        variables=_sonnet_scalar_variables(p,variables)
        p=_sonnet_scalar_project(p,variables)
    end
    # Evaluate used thin-metal providers before allocating any geometry masks.
    # Retained unused definitions do not impose a physical material on a shape.
    checked=Set{Int}()
    for poly in p.polygons
        poly.kind==:sheet && poly.material>=0 || continue
        poly.material in checked && continue
        push!(checked,poly.material)
        metal=p.metals[poly.material+1]
        length(metal)>=3 && metal[3]=="NOR" && sonnet_metal_zs(p,metal,freq;variables)
    end
    for cover in (p.top,p.bottom)
        length(cover)>=3 && cover[3]=="NOR" && sonnet_metal_zs(p,cover,freq;variables,cover=true)
    end
    p=_sonnet_geometry_project(p,freq,variables;max_bytes)
    geometry=expand_thick ? _sonnet_thick_geometry(p,freq,variables;max_bytes) : p
    # Reserve both layer vectors, promotion references and push! growth.
    stack_workspace=_checked_payload_sum("native stack layer workspace",
        _checked_array_payload_bytes(PlanarLayer{ComplexF64},2,length(geometry.layers)),
        _checked_array_payload_bytes(Ptr{Cvoid},4,length(geometry.layers)))
    _enforce_payload_limit(stack_workspace,max_bytes,"native stack layer workspace","max_bytes")
    val(t)=sonnet_variable_value(geometry,t;variables=variables,freq=freq)
    ls=geometry.length_scale
    a,b=val(geometry.box[2])*ls,val(geometry.box[3])*ls
    halfcounts=(parse(Int,geometry.box[4]),parse(Int,geometry.box[5]))
    all(n->n>0 && iseven(n),halfcounts) ||
        throw(ArgumentError("native BOX half-cell counts must be positive even integers"))
    nx,ny=grid===nothing ? (halfcounts[1]÷2,halfcounts[2]÷2) : grid
    gr=CellGrid(a,b,nx,ny)
    layers=PlanarLayer[]
    for row in reverse(geometry.layers)
        d,e,m,te,tm=val.(row[1:5])
        sigma=_sonnet_dielectric_sigma(geometry,row,freq,variables)
        push!(layers,PlanarLayer(e*(1-im*te)-im*_sonnet_dielectric_conduction(sigma,freq),
            m*(1-im*tm),d*ls))
    end
    stack=PlanarStackup(layers,_sonnet_termination(geometry,geometry.bottom,freq,variables),
        _sonnet_termination(geometry,geometry.top,freq,variables),a,b)
    return (;geometry_project=geometry,stack,grid=gr,variables)
end

include("PlanarSonnetRasterResources.jl")

"""Lower supported native geometry to the raw-gap planar solver.
`grid=(nx,ny)` explicitly changes raster resolution from the native BOX grid.
Native BOX counts are half-cell counts: `BOX ... 96 64 ...` specifies an
actual 48 by 32 cell grid. Explicit `grid` counts are actual cells.
Wall/gap/axial ports, PEC/impedance/free-space covers, stratified dielectric
loss, and PEC vias are supported. Native dielectric `RSVY` selects positive
resistivity in ohm centimetres; an unmarked loss field is conductivity in
siemens per metre. Unknown layer loss/anisotropy flags reject. The internal material details used by
`solve_sonnet_project` additionally preserve supported sheet models, physical
two-face TMM geometry, and constant VOL/ARR resistance-per-via loss.
Independent anchored/symmetric unscaled X/Y geometry dimensions are supported.
Co-calibration, other geometry-variable modes, dielectric bricks, general via
skin loss and unsupported material semantics reject explicitly. Components
require the circuit wrapper. Parsed unsupported data remain in `SonnetProject`.
`max_bytes` bounds owned raster masks, material maps, source and basis workspace
before allocation. It uses the same raw-payload convention as `solve_planar`.
Public PEC lowering omits unused material maps; adapters requesting material
details retain those maps and their overlap checks."""
function sonnet_planar_problem(p::SonnetProject;freq::Real=1e9,grid=nothing,
        variables=Dict{String,Float64}(),_materials::Bool=false,_details::Bool=false,
        _allow_portless::Bool=false,scalar_files=nothing,
        scalar_root=dirname(p.source),scalar_outside::Symbol=:reject,
        scalar_max_files::Integer=64,scalar_max_bytes::Integer=8*1024^2,
        scalar_max_nodes::Integer=100000,scalar_max_line_bytes::Integer=16384,
        scalar_max_storage::Integer=64*1024^2,
        max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    isfinite(freq) && freq>0 || throw(ArgumentError("frequency must be positive and finite"))
    freq=_circuit_stored_real(freq,"native geometry frequency")
    limit=_spice_limit("max_bytes",max_bytes)
    _enforce_payload_limit(1024,limit,"native raster source workspace","max_bytes")
    grid=_sonnet_raster_grid(p,grid)
    source_workspace=_sonnet_raster_source_workspace(p)
    _enforce_payload_limit(source_workspace,limit,"native raster source workspace","max_bytes")
    scalar_workspace=0
    if scalar_files!==nothing || (!(variables isa SonnetScalarVariables) && _sonnet_has_scalar_tables(p))
        variables=_sonnet_scalar_variables(p,variables;scalar_files,
            max_bytes=min(scalar_max_storage,limit-source_workspace),
            root=scalar_root,outside=scalar_outside,max_files=scalar_max_files,
            max_bytes_source=scalar_max_bytes,max_nodes=scalar_max_nodes,max_line_bytes=scalar_max_line_bytes)
        scalar_workspace=_sonnet_scalar_payload(variables)
        p=_sonnet_scalar_project(p,variables)
    end
    source=p
    isempty(p.components) || throw(ArgumentError("native SMD components require the circuit adapter"))
    physical=_sonnet_stack_geometry(p,freq,grid,variables;max_bytes=limit-scalar_workspace)
    p=physical.geometry_project
    variables=physical.variables
    val(t)=sonnet_variable_value(p,t;variables=variables,freq=freq)
    ls=p.length_scale
    gr=physical.grid;st=physical.stack;layers=st.layers
    a,b=gr.a,gr.b;nx,ny=gr.nx,gr.ny
    native_nx=parse(Int,p.box[4])÷2;native_ny=parse(Int,p.box[5])÷2
    p=_sonnet_raster_wall_project(p,a,b,native_nx,native_ny)
    L=length(layers)
    any(poly->poly.kind==:brick,p.polygons) && throw(ArgumentError("native dielectric bricks require a volume dielectric adapter"))
    store_surfaces=_details || _materials
    geometry_workspace=_checked_payload_sum("native raster geometry workspace",
        source_workspace,scalar_workspace,_sonnet_raster_mask_workspace(p,gr;surfaces=store_surfaces))
    _enforce_payload_limit(geometry_workspace,limit,"native raster geometry workspace","max_bytes")
    sheets=SheetLevel[]; sheetidx=Dict{Int,Int}(); masks=Dict{Int,BitMatrix}()
    vias=ViaLevel[]; via_sigma=Float64[];via_group=Dict{Tuple{Int,Float64},Int}()
    via_polygon_level=Dict{Tuple{Int,Int},Int}()
    surface_by_level=Dict{Int,Matrix{ComplexF64}}()
    function native_sheet(level)
        if !haskey(sheetidx,level)
            push!(sheets,sheet_level(L-1-level,nx,ny))
            sheetidx[level]=length(sheets)
            store_surfaces && (surface_by_level[level]=zeros(ComplexF64,nx,ny))
        end
        return sheets[sheetidx[level]]
    end
    for poly in p.polygons
        if poly.material!=-1
            0<=poly.material<length(p.metals) || throw(ArgumentError("invalid metal index on polygon $(poly.id)"))
            _materials || throw(ArgumentError("polygon $(poly.id) uses material index $(poly.material); non-PEC metal loss/thickness requires the material adapter"))
            poly.kind==:sheet && sonnet_metal_zs(p,p.metals[poly.material+1],freq;variables=variables)
        end
        if poly.kind==:sheet
            0<=poly.level<L || throw(ArgumentError("sheet level $(poly.level) outside native stack"))
            sh=native_sheet(poly.level)
            tmp=sheet_level(sh.interface,nx,ny)
            rasterize_poly!(tmp,gr,poly.vertices[1,:],poly.vertices[2,:])
            any(tmp.mask) || throw(ArgumentError("polygon $(poly.id) vanished at raster resolution"))
            zs=poly.material==-1 ? 0.0im : sonnet_metal_zs(p,p.metals[poly.material+1],freq;variables=variables)
            surface=store_surfaces ? surface_by_level[poly.level] : nothing
            for cell in eachindex(tmp.mask)
                tmp.mask[cell] || continue
                store_surfaces && sh.mask[cell] && surface[cell]!=zs && throw(ArgumentError(
                    "different sheet materials overlap in cell $cell on native level $(poly.level); refine the raster or define a composite conductor"))
                store_surfaces && (surface[cell]=zs)
            end
            masks[poly.id]=copy(tmp.mask)
            sh.mask .|= tmp.mask
            # A sheet touching a box wall is galvanically connected even
            # where no driven port is present on that wall.
            xmin,xmax=extrema(@view poly.vertices[1,:]);ymin,ymax=extrema(@view poly.vertices[2,:])
            xmin<=0<=xmax && (sh.connect_west .|= tmp.mask[1,:])
            xmin<=a<=xmax && (sh.connect_east .|= tmp.mask[end,:])
            ymin<=0<=ymax && (sh.connect_south .|= tmp.mask[:,1])
            ymin<=b<=ymax && (sh.connect_north .|= tmp.mask[:,end])
        else
            target=poly.target=="GND" ? L-1 : poly.target=="TOP" ? -1 : parse(Int,poly.target)
            lo,hi=minmax(poly.level,target)
            -1<=lo<hi<=L-1 || throw(ArgumentError("via $(poly.id) target outside stack"))
            tmp=sheet_level(0,nx,ny)
            rasterize_poly!(tmp,gr,poly.vertices[1,:],poly.vertices[2,:])
            any(tmp.mask) || throw(ArgumentError("via polygon $(poly.id) vanished at raster resolution"))
            mode="SOLID" in poly.flags || "FULL" in poly.flags ? :full :
                "VERTICES" in poly.flags ? :vertices : "CENTER" in poly.flags ? :center :
                "BAR" in poly.flags ? :bar : :ring
            via_mask=_planar_via_mesh_mask(tmp.mask,poly.vertices,gr,mode)
            masks[poly.id]=via_mask
            sigma=_sonnet_via_sigma(p,poly,gr,st,via_mask,freq,variables)
            if sigma isa Complex && via_sigma isa Vector{Float64}
                via_sigma=ComplexF64.(via_sigma)
                via_group=Dict{Tuple{Int,ComplexF64},Int}((k[1],ComplexF64(k[2]))=>v for (k,v) in via_group)
            end
            for lev in lo+1:hi
                layer=L-lev
                key=(layer,sigma)
                vi=get(via_group,key,0)
                for (other,v) in enumerate(vias)
                    v.layer==layer && any(v.uni .& via_mask) &&
                        (other!=vi || isfinite(sigma)) && throw(ArgumentError(
                            "native via resistance polygons overlap in layer $layer; refine the raster or combine their physical models"))
                end
                if vi==0
                    push!(vias,via_level(layer,nx,ny));push!(via_sigma,sigma)
                    vi=length(vias);via_group[key]=vi
                end
                vias[vi].uni .|= via_mask
                vias[vi].tap .|= via_mask
                via_polygon_level[(poly.id,layer)]=vi
            end
        end
    end
    # Existing physical sheets retain their material on overlaps. Generated
    # films share one tangential field and their conductances add.
    generated_endpoints=Dict{Int,BitMatrix}()
    for poly in p.polygons
        _sonnet_via_endpoints(p,poly) || continue
        target=poly.target=="GND" ? L-1 : poly.target=="TOP" ? -1 : parse(Int,poly.target)
        endpoint=masks[poly.id]
        if "COVERS" in poly.flags
            footprint=sheet_level(0,nx,ny)
            rasterize_poly!(footprint,gr,poly.vertices[1,:],poly.vertices[2,:])
            endpoint=footprint.mask
        end
        # Fully covered endpoints already carry the original physical
        # sheet current; avoid evaluating an unused film or wide fallback.
        uncovered=false
        for level in (poly.level,target)
            level in (-1,L-1) && continue
            index=get(sheetidx,level,0)
            if iszero(index)
                uncovered=any(endpoint)
            else
                mask=sheets[index].mask
                generated=get(generated_endpoints,level,nothing)
                uncovered=any(cell->endpoint[cell] && (!mask[cell] ||
                    (generated!==nothing && generated[cell])),eachindex(endpoint))
            end
            uncovered && break
        end
        uncovered || continue
        zs=_sonnet_via_endpoint_zs(p,poly,st,freq,variables)
        for level in (poly.level,target)
            level in (-1,L-1) && continue
            pad=native_sheet(level)
            generated=get!(generated_endpoints,level) do
                falses(nx,ny)
            end
            for cell in eachindex(endpoint)
                endpoint[cell] || continue
                pad.mask[cell] && !generated[cell] && continue
                if store_surfaces
                    surface_by_level[level][cell]=generated[cell] ?
                        _sonnet_via_parallel_zs(surface_by_level[level][cell],zs) : zs
                end
                pad.mask[cell]=true;generated[cell]=true
            end
        end
    end
    polydict=Dict(q.id=>q for q in p.polygons)
    port_workspace=_sonnet_raster_port_workspace(p,gr,masks,polydict)
    _enforce_payload_limit(_checked_payload_sum("native raster source attachment",
        geometry_workspace,port_workspace),limit,"native raster source attachment","max_bytes")
    ports=PlanarPort[]
    numbers=Int[]; port_z0=ComplexF64[];port_weights=Float64[]
    for ps in p.ports
        ps.kind in (:box,:std,:gap,:via) || throw(ArgumentError("native $(ps.kind) port requires its terminal/calibration adapter"))
        ps.number==0 && !(ps.kind in (:box,:std)) &&
            throw(ArgumentError("interior native port zero requires an explicit ground return adapter"))
        poly=polydict[ps.polygon]
        q=ps.values
        z0=_sonnet_port_reference(p,ps,freq,variables)
        if ps.kind==:via || (ps.kind==:std && poly.kind==:via)
            poly.kind==:via || throw(ArgumentError("via port requires a via polygon"))
            target=poly.target=="GND" ? L-1 : poly.target=="TOP" ? -1 : parse(Int,poly.target)
            ps.kind==:std && !(poly.level in (-1,L-1) || target in (-1,L-1)) &&
                throw(ArgumentError("native standard via port must terminate on a box cover"))
            lo,hi=minmax(poly.level,target)
            path_layers=[L-lev for lev in lo+1:hi]
            height=sum(real(st.layers[layer].thickness) for layer in path_layers)
            height>0 || throw(ArgumentError("via port has nonpositive physical path length"))
            for layer in path_layers
                vi=get(via_polygon_level,(poly.id,layer),0)
                vi==0 && throw(ArgumentError("via port lacks current bases"))
                for cell in findall(vec(masks[poly.id]))
                    push!(ports,PlanarPort(vi,:via,cell:cell,z0;polarity=ps.number<0 ? -1 : 1))
                    push!(numbers,abs(ps.number));push!(port_z0,z0)
                    push!(port_weights,real(st.layers[layer].thickness)/height)
                end
            end
            continue
        end
        poly.kind==:sheet || throw(ArgumentError("sheet port must reference a sheet"))
        index,next=_sonnet_port_edge_indices(poly.vertices,ps.edge)
        v=poly.vertices
        _sonnet_box_port_edge_inside(v,ps.edge,a,b,native_nx,native_ny) ||
            throw(ArgumentError("native box-port edge is partially or entirely outside the box"))
        wall=v[1,index]==v[1,next]==0. ? :west : v[1,index]==v[1,next]==a ? :east :
            v[2,index]==v[2,next]==0. ? :south : v[2,index]==v[2,next]==b ? :north : nothing
        if wall===nothing
            ps.number!=0 || throw(ArgumentError("interior native port zero requires an explicit ground return adapter"))
            start,finish=_sonnet_shared_port_segment(p,poly,index,next)
            dx,dy=finish[1]-start[1],finish[2]-start[2]
            abs(dx)<=1e-8*max(a,b) || abs(dy)<=1e-8*max(a,b) ||
                throw(ArgumentError("diagonal native gap port requires a conformal terminal adapter"))
            direction=abs(dx)<abs(dy) ? :x : :y
            coordinate=direction==:x ? start[1]/gr.dx : start[2]/gr.dy
            edge=round(Int,coordinate)
            abs(coordinate-edge)<=1e-7 || throw(ArgumentError("gap port is not aligned to the selected raster grid"))
            sh=sheets[sheetidx[poly.level]]
            lower,upper=direction==:x ? minmax(start[2],finish[2]) : minmax(start[1],finish[1])
            spacing=direction==:x ? gr.dy : gr.dx
            candidates=findall(cell->lower<=(cell-.5)*spacing<=upper,
                1:(direction==:x ? ny : nx))
            isempty(candidates) && throw(ArgumentError("gap port edge vanished at raster resolution"))
            span=first(candidates):last(candidates)
            push!(ports,PlanarPort(sheetidx[poly.level],direction,edge,span,z0;
                polarity=ps.number<0 ? -1 : 1))
            push!(numbers,abs(ps.number)); push!(port_z0,z0)
            push!(port_weights,1.)
            continue
        end
        mask=masks[poly.id]
        occupied=wall==:west ? (@view mask[1,:]) : wall==:east ? (@view mask[end,:]) :
            wall==:south ? (@view mask[:,1]) : (@view mask[:,end])
        any(occupied) || throw(ArgumentError("port $(ps.number) disappeared at raster resolution"))
        # Select only the connected wall segment containing the native port.
        axis=wall in (:west,:east) ? 2 : 1;spacing=axis==2 ? gr.dy : gr.dx
        coordinate=v[axis,index]/2+v[axis,next]/2
        center=clamp(floor(Int,coordinate/spacing)+1,1,length(occupied))
        occupied[center] || throw(ArgumentError("port $(ps.number) does not lie on rasterized metal"))
        firstcell=lastcell=center
        while firstcell>1 && occupied[firstcell-1];firstcell-=1;end
        while lastcell<length(occupied) && occupied[lastcell+1];lastcell+=1;end
        sh=sheets[sheetidx[poly.level]]
        flags=wall==:west ? sh.connect_west : wall==:east ? sh.connect_east : wall==:south ? sh.connect_south : sh.connect_north
        flags[firstcell:lastcell].=true
        ps.number==0 && continue # The undriven wall half-rooftops impose V=0.
        push!(ports,PlanarPort(sheetidx[poly.level],wall,firstcell:lastcell,z0;
            polarity=ps.number<0 ? -1 : 1))
        push!(numbers,abs(ps.number)); push!(port_z0,z0)
        push!(port_weights,1.)
    end
    order=sortperm(numbers); ports=ports[order]; numbers=numbers[order]; port_z0=port_z0[order]
    port_weights=port_weights[order]
    _enforce_payload_limit(_checked_payload_sum("native raster basis construction",
        geometry_workspace,port_workspace,_sonnet_raster_basis_workspace(gr,sheets,vias)),
        limit,"native raster basis construction","max_bytes")
    # Terminal adapters first lower a genuinely source-free geometry clone,
    # then attach their physical source. Do not invent a temporary driven
    # terminal or change the public requirement for a nonempty port list.
    problem=if _allow_portless && isempty(ports)
        planar_validate(st)
        basis=build_planar_basis(gr,sheets,ports;vias=vias)
        planar_basis_count(basis)>0 || throw(ArgumentError("no basis functions: check masks/connections"))
        PlanarProblem(st,gr,sheets,ports,vias,basis)
    else
        build_planar_problem(st,gr,sheets,ports;vias=vias)
    end
    if _details
        labels=sort(unique(numbers)); contraction=zeros(length(ports),length(labels)); z0=ComplexF64[]
        for (j,label) in enumerate(labels)
            inds=findall(==(label),numbers)
            all(port_z0[i]==port_z0[first(inds)] for i in inds) || throw(ArgumentError("native terminals sharing a port must have equal reference impedance"))
            contraction[inds,j].=port_weights[inds]; push!(z0,port_z0[first(inds)])
        end
        sheet_zs=[surface_by_level[level]
            for level in sort(collect(keys(sheetidx));by=level->sheetidx[level])]
        terminal_maps=_sonnet_terminal_maps(contraction,[port.polarity for port in ports])
        model=(;problem,contraction=terminal_maps.contraction,
            floating_common=terminal_maps.floating_common,z0,sheet_zs,via_sigma,labels,geometry_project=p,
            scalar_files=_sonnet_scalar_files(variables))
        payload=_checked_payload_sum("native retained raster model",
            _sonnet_raster_retained_payload(model,source),scalar_workspace)
        _enforce_payload_limit(payload,limit,"native retained raster model","max_bytes")
        return merge(model,(;payload))
    end
    length(unique(numbers))==length(numbers) || throw(ArgumentError("native shared terminals require solve_sonnet_project for network contraction"))
    return problem
end

# Same positive label means a common voltage. A corresponding negative
# label supplies a floating reference with equal/opposite summed current.
# Raw coordinates already include source polarity and via height weights.
function _sonnet_terminal_maps(contraction::AbstractMatrix,polarities::AbstractVector)
    length(polarities)==size(contraction,1) || throw(DimensionMismatch("native terminal polarity count"))
    all(s->s in (-1,1),polarities) || throw(ArgumentError("invalid native terminal polarity"))
    D=Float64.(contraction);commons=Vector{Vector{Float64}}()
    for j in axes(D,2)
        ids=findall(!iszero,view(D,:,j))
        any(i->polarities[i]>0,ids) || throw(ArgumentError("native negative port requires a corresponding positive terminal"))
        if any(i->polarities[i]<0,ids)
            push!(commons,D[:,j].*polarities)
            D[:,j]./=2
        end
    end
    F=isempty(commons) ? zeros(Float64,size(D,1),0) : hcat(commons...)
    return (;contraction=D,floating_common=F)
end

function _sonnet_balanced_y(Y::AbstractMatrix,D::AbstractMatrix,F::AbstractMatrix)
    size(Y,1)==size(Y,2)==size(D,1)==size(F,1) ||
        throw(DimensionMismatch("native balanced terminal matrices"))
    all(isfinite,Y) && all(isfinite,D) && all(isfinite,F) ||
        throw(ArgumentError("native balanced terminal matrices must be finite"))
    V=ComplexF64.(D)
    if size(F,2)>0
        factor=lu(transpose(F)*Y*F;check=false)
        issuccess(factor) || throw(ArgumentError("native floating reference has a singular common-mode admittance"))
        V .-= F*(factor\(transpose(F)*Y*D))
    end
    y=Matrix{ComplexF64}(transpose(D)*Y*V)
    all(isfinite,y) && all(isfinite,V) || throw(ArgumentError("native balanced terminal solve is nonfinite"))
    return (;y,voltage_transfer=V)
end
_sonnet_balanced_y(Y::AbstractMatrix,model)=
    _sonnet_balanced_y(Y,model.contraction,model.floating_common)

"""Native geometry solve retaining both raw basis result and terminal contraction.
`y`/`s` are at raw gap planes. Native co-calibration/ref-plane metadata remain in
`project`; requesting calibrated output without calibration fails explicitly."""
struct SonnetPlanarResult{R,F}
    project::SonnetProject
    raw::R
    port_numbers::Vector{Int}
    y::Matrix{ComplexF64}
    s::Matrix{ComplexF64}
    voltage_transfer::Matrix{ComplexF64}
    z0::Vector{ComplexF64}
    scalar_files::F
end
SonnetPlanarResult(project,raw,numbers,y,s,transfer,z0)=
    SonnetPlanarResult(project,raw,numbers,y,s,transfer,z0,nothing)

"""Solve a parsed native geometry or circuit project at `freq` [Hz]. Geometry
returns [`SonnetPlanarResult`](@ref), retaining raw basis currents and native
terminal-number contraction. Sheet materials and covers are evaluated at the
requested frequency. A project requesting native de-embedding requires an
explicit calibration object or `raw=true`; saved native calibration records
alone do not supply a calibrated standard. `grid=(nx,ny)` selects a new raster;
unrepresented geometry and unsupported material/port/component models fail.
Two-sheet TMM conductors are expanded into separated physical faces with
half-thickness impedances, the existing host dielectric, PEC perimeter ties,
and a shared external voltage. Other sheet counts and unspecified material
flags reject. This published approximation does not model lateral side loss.
Native BOX fields store half-cell counts; an explicit `grid` uses actual cells.
Interior via sources spanning multiple layers retain total-voltage excitation
through layer-height weights. Wall port zero is an undriven ground connection.
Negative labels define floating references with equal/opposite summed currents.
Their common voltage is solved jointly. `voltage_transfer` maps external
terminal voltage to raw source voltage, including supplied calibration.
`z0` retains the complex references evaluated at this frequency. Native port
R/X/L/C fields use Ω/Ω/nH/pF with parallel C; S uses Kurokawa power waves.
Sonnet Lite only permits 50 Ω electromagnetic references; its Graph
postprocessor separately supports these complex references.

For [`SonnetNetlistProject`](@ref), the R/L/C and data blocks lower through
[`sonnet_planar_circuit`](@ref) and return [`PlanarCircuitResult`](@ref).
Referenced PRJ blocks require `project_response(path,f_hz)` with calibrated
S-parameters. Data blocks reject frequencies outside their source coverage.
The filename overload reads and dispatches the actual native project type."""
function solve_sonnet_project(p::SonnetProject,freq::Real;
        grid=nothing,variables=Dict{String,Float64}(),calibration=nothing,
        component_response=nothing,raw::Bool=false,scalar_files=nothing,
        scalar_root=dirname(p.source),scalar_outside::Symbol=:reject,
        scalar_max_files::Integer=64,scalar_max_bytes::Integer=8*1024^2,
        scalar_max_nodes::Integer=100000,scalar_max_line_bytes::Integer=16384,kw...)
    isfinite(freq) && freq>0 || throw(ArgumentError("frequency must be positive and finite"))
    freq=_circuit_stored_real(freq,"native solve frequency")
    limit=_spice_limit("max_bytes",get(kw,:max_bytes,_DEFAULT_MAX_DENSE_PAYLOAD_BYTES))
    _enforce_payload_limit(1024,limit,"native raster source workspace","max_bytes")
    deembed_requested=any(r->r.tokens[1]=="OPTIONS" &&
        any(occursin("d",token) for token in r.tokens[2:end]),p.records)
    !raw && deembed_requested && calibration===nothing && throw(ArgumentError(
        "native project requests de-embedding; supply calibration or explicitly request raw=true"))
    scalar_options=scalar_files===nothing && !_sonnet_has_scalar_tables(p) ? (;) :
        (;scalar_files,scalar_root,scalar_outside,scalar_max_files,scalar_max_bytes,
            scalar_max_nodes,scalar_max_line_bytes)
    _sonnet_has_floating(p) && return _solve_sonnet_floating(p,freq;
        grid=grid,variables=variables,calibration=calibration,
        component_response=component_response,scalar_options...,kw...)
    isempty(p.components) || return _solve_sonnet_components(p,freq;
        grid=grid,variables=variables,calibration=calibration,
        component_response=component_response,scalar_options...,kw...)
    if scalar_files!==nothing || _sonnet_has_scalar_tables(p) || variables isa SonnetScalarVariables
        variables=_sonnet_scalar_variables(p,variables;scalar_files,max_bytes=limit,
            root=scalar_root,outside=scalar_outside,max_files=scalar_max_files,
            max_bytes_source=scalar_max_bytes,max_nodes=scalar_max_nodes,max_line_bytes=scalar_max_line_bytes)
        p=_sonnet_scalar_project(p,variables)
    end
    scalar_reserve=_sonnet_scalar_files(variables)===nothing ? 0 : _sonnet_scalar_payload(variables)
    _enforce_payload_limit(scalar_reserve,limit,"native scalar solve snapshot","max_bytes")
    :surface_zs in keys(kw) && throw(ArgumentError("native material loss is selected by the project; surface_zs override is unsupported"))
    :via_sigma in keys(kw) && throw(ArgumentError("native axial resistance is selected by the project; via_sigma override is unsupported"))
    model=sonnet_planar_problem(p;freq=freq,grid=grid,variables=variables,
        _materials=true,_details=true,max_bytes=limit-scalar_reserve)
    reserve=_checked_payload_sum("native raster solve workspace",scalar_reserve,
        model.payload,_sonnet_raster_response_workspace(model))
    _enforce_payload_limit(reserve,limit,"native raster solve workspace","max_bytes")
    remaining=merge((;kw...),(max_bytes=limit-reserve,))
    result=solve_planar(model.problem,freq;surface_zs=model.sheet_zs,via_sigma=model.via_sigma,remaining...)
    terminal=_sonnet_balanced_y(result.y,model)
    y=terminal.y;voltage_transfer=terminal.voltage_transfer
    if calibration!==nothing
        y=deembed_ports(y,calibration)
        A,B,_,_=_calibration_chain_blocks(calibration,length(model.labels))
        voltage_transfer=voltage_transfer*(A+B*y)
    end
    s=planar_y_to_s(y,model.z0)
    return SonnetPlanarResult(p,result,model.labels,y,s,voltage_transfer,model.z0,model.scalar_files)
end

"""Reconstruct physical currents from native external terminal excitations,
including floating-reference potentials and supplied calibration."""
function planar_current_maps(result::SonnetPlanarResult;port::Integer=1,
        voltages=nothing,incident_waves=nothing,
        max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES,kw...)
    n=length(result.port_numbers)
    reserve=_checked_payload_sum("native current excitation",
        _checked_array_payload_bytes(ComplexF64,6,n),
        _checked_array_payload_bytes(ComplexF64,2,size(result.voltage_transfer,1)),
        _checked_array_payload_bytes(Float64,n))
    _enforce_payload_limit(reserve,max_bytes,"native current excitation","max_bytes")
    voltages!==nothing && incident_waves!==nothing && throw(ArgumentError("provide voltages or incident_waves"))
    v=if voltages!==nothing
        length(voltages)==n && all(isfinite,voltages) || throw(ArgumentError("native voltages must be finite and match ports"))
        _planar_stored_phasor.(voltages)
    elseif incident_waves!==nothing
        length(incident_waves)==n && all(isfinite,incident_waves) || throw(ArgumentError("native incident waves must be finite and match ports"))
        _planar_wave_voltage(result.s,incident_waves,result.z0)
    else
        1<=port<=n || throw(ArgumentError("native port index is invalid"))
        excitation=zeros(ComplexF64,n);excitation[port]=1;excitation
    end
    return planar_current_maps(result.raw;voltages=result.voltage_transfer*v,
        max_bytes=max_bytes-reserve,kw...)
end

function solve_sonnet_project(p::SonnetNetlistProject,freq::Real;
        project_response=nothing,kw...)
    return solve_planar_circuit(sonnet_planar_circuit(p;project_response=project_response),freq;kw...)
end

function solve_sonnet_project(path::AbstractString,freq::Real;kw...)
    _spice_limit("max_bytes",get(kw,:max_bytes,_DEFAULT_MAX_DENSE_PAYLOAD_BYTES))
    return solve_sonnet_project(read_sonnet_project(path),freq;kw...)
end
