# Explicit bounded subset of the readable Sonnet Spectre linear exports.
export planar_read_spectre

const _SPECTRE_SCALES=Dict('T'=>1e12,'G'=>1e9,'M'=>1e6,'K'=>1e3,'k'=>1e3,'_'=>1.,'%'=>1e-2,
    'c'=>1e-2,'m'=>1e-3,'u'=>1e-6,'n'=>1e-9,'p'=>1e-12,'f'=>1e-15,'a'=>1e-18)
function _spectre_number(text)
    m=match(r"^([+\-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+\-]?[0-9]+)?)([a-zA-Z_%]*)$",text)
    m===nothing && throw(ArgumentError("Spectre linear exports require literal finite numbers"))
    suffix=m[2];scale=1.
    if !isempty(suffix)
        occursin(r"[eE]",m[1]) && throw(ArgumentError("mixed Spectre exponent/scale tokens are unsupported"))
        haskey(_SPECTRE_SCALES,first(suffix)) || throw(ArgumentError("Spectre units require a recognized case-sensitive SI factor"))
        scale=_SPECTRE_SCALES[first(suffix)]
    end
    value=parse(Float64,m[1])*scale
    isfinite(value) || throw(ArgumentError("Spectre number must be finite"))
    iszero(value) && _spice_literal_nonzero(m[1]) && throw(ArgumentError("nonzero Spectre number underflows Float64"))
    value
end
_spectre_identifier(s)=occursin(r"^[a-zA-Z0-9_.+\-]+$",s) ? String(s) :
    throw(ArgumentError("unsupported Spectre identifier $s"))

# Literal values keep Spectre's case-sensitive SI scales; bare names stay
# raw so the downstream expression context can resolve parameter references.
function _spectre_value(token)
    try
        return string(_spectre_number(token))
    catch e
        e isa ArgumentError || rethrow()
        return token
    end
end

function _spectre_assignments(fields,card)
    result=Pair{String,String}[]
    length(fields)%3==0 || _spice_failure(card,"Spectre parameters require name=value assignments")
    for k in 1:3:length(fields)
        fields[k+1]=="=" || _spice_failure(card,"missing Spectre parameter equals")
        key=_spectre_identifier(fields[k]);occursin(r"^[a-zA-Z][a-zA-Z0-9_]*$",key) ||
            _spice_failure(card,"unsupported Spectre parameter identifier")
        push!(result,lowercase(key)=>_spectre_value(fields[k+2]))
    end
    result
end

const _SPECTRE_MASTERS=("resistor","capacitor","inductor","mutual_inductor",
    "vcvs","vccs","cccs","ccvs","vsource","isource","tline","nport","transformer")

# Vendor/nonlinear masters name themselves explicitly in readable exports;
# they reject with their own diagnostic rather than "undefined subckt".
const _SPECTRE_VENDOR=("diode","mosfet","bjt","jfet","bsim","snp",
    "behavioral_source","bsource","ahdl","relay","switch",
    "diff_amp","vsource_ac","isource_ac","pwl","sin")

# Lower one component record to the equivalent SPICE-token card. Spectre
# labels are arbitrary, so lowered names carry their SPICE class letter
# (E_/G_/F_/H_/V_/I_/T_/X_/K_/) to reach the correct element grammar.
function _spectre_component(card,fields,names,definitions)
    name=_spectre_identifier(fields[1]);name in names && _spice_failure(card,"duplicate Spectre component name")
    push!(names,name)
    eq=findfirst(==("="),fields);paramstart=eq===nothing ? length(fields)+1 : eq-1
    eq===nothing || (length(fields)-paramstart+1)%3==0 &&
        all(k->fields[paramstart+3k+1]=="=",0:(length(fields)-paramstart+1)÷3-1) ||
        _spice_failure(card,"Spectre component parameters require name=value assignments")
    masteridx=findnext(s->s in _SPECTRE_MASTERS,fields,2)
    instance=masteridx===nothing
    instance && (masteridx=eq===nothing ? length(fields) : paramstart-1)
    masteridx>=2 || _spice_failure(card,"Spectre component requires a master")
    master=fields[masteridx]
    nodes=fields[2:masteridx-1]
    trailing=fields[masteridx+1:paramstart-1]
    props=Pair{String,String}[]
    for k in paramstart:3:length(fields)
        fields[k+1]=="=" || _spice_failure(card,"Spectre component parameters require name=value")
        push!(props,lowercase(_spectre_identifier(fields[k]))=>fields[k+2])
    end
    length(unique(first.(props)))==length(props) || _spice_failure(card,"duplicate Spectre component parameter")
    value(key)=begin
        found=findfirst(p->p.first==key,props)
        found===nothing && _spice_failure(card,"$master requires $key")
        _spectre_value(props[found].second)
    end
    if instance
        !haskey(definitions,master) && lowercase(master) in _SPECTRE_VENDOR &&
            _spice_failure(card,"unsupported Spectre nonlinear/vendor master $master")
        isempty(nodes) && _spice_failure(card,"Spectre instance requires terminal nodes")
        master=_spectre_identifier(master)
        return ["X_"*name;[_spectre_identifier(n) for n in nodes];master;
            collect(Iterators.flatten((k,"=",_spectre_value(v)) for (k,v) in props))]
    elseif master in ("resistor","capacitor","inductor")
        kind=master=="resistor" ? 'R' : master=="capacitor" ? 'C' : 'L'
        uppercase(first(name))==kind || _spice_failure(card,"unsupported Spectre linear component label")
        length(nodes)==2 && isempty(trailing) ||
            _spice_failure(card,"Spectre $master requires two nodes")
        length(props)==1 && only(props).first==lowercase(string(kind)) ||
            _spice_failure(card,"Spectre $master requires $(lowercase(string(kind)))= value")
        return [name,_spectre_identifier(nodes[1]),_spectre_identifier(nodes[2]),value(first(props).first)]
    elseif master=="mutual_inductor"
        isempty(nodes) && isempty(trailing) ||
            _spice_failure(card,"Spectre mutual_inductor takes no nodes")
        Set(first.(props))==Set(("coupling","ind1","ind2")) ||
            _spice_failure(card,"mutual_inductor requires coupling,ind1,ind2")
        return ["K_"*name,_spectre_identifier(props[findfirst(p->p.first=="ind1",props)].second),
            _spectre_identifier(props[findfirst(p->p.first=="ind2",props)].second),value("coupling")]
    elseif master in ("vcvs","vccs")
        kind=master=="vcvs" ? 'E' : 'G'
        length(nodes)==2 && length(trailing)==2 ||
            _spice_failure(card,"Spectre $master requires output and control node pairs")
        length(props)==1 && only(props).first==master ||
            _spice_failure(card,"Spectre $master requires $(master)= gain")
        return [string(kind)*"_"*name,_spectre_identifier(nodes[1]),_spectre_identifier(nodes[2]),
            _spectre_identifier(trailing[1]),_spectre_identifier(trailing[2]),value(master)]
    elseif master in ("cccs","ccvs")
        kind=master=="cccs" ? 'F' : 'H'
        length(nodes)==2 && isempty(trailing) ||
            _spice_failure(card,"Spectre $master requires two output nodes and a probe")
        Set(first.(props))==Set((master,"probe")) ||
            _spice_failure(card,"Spectre $master requires probe= and $(master)= properties")
        probe=_spectre_identifier(props[findfirst(p->p.first=="probe",props)].second)
        return [string(kind)*"_"*name,_spectre_identifier(nodes[1]),_spectre_identifier(nodes[2]),
            "V_"*probe,value(master)]
    elseif master in ("vsource","isource")
        kind=master=="vsource" ? 'V' : 'I'
        length(nodes)==2 && isempty(trailing) ||
            _spice_failure(card,"Spectre $master requires two nodes")
        Set(first.(props))==Set(("type","dc")) && props[findfirst(p->p.first=="type",props)].second=="dc" ||
            _spice_failure(card,"only DC Spectre independent sources are supported")
        return [string(kind)*"_"*name,_spectre_identifier(nodes[1]),_spectre_identifier(nodes[2]),
            "dc",value("dc")]
    elseif master=="tline"
        length(nodes)==4 && isempty(trailing) ||
            _spice_failure(card,"Spectre tline requires four nodes")
        all(p->p.first in ("z0","td","f","nl"),props) ||
            _spice_failure(card,"Spectre tline supports only z0,td,f,nl")
        tokens=["T_"*name,_spectre_identifier(nodes[1]),_spectre_identifier(nodes[2]),
            _spectre_identifier(nodes[3]),_spectre_identifier(nodes[4])]
        foreach(p->append!(tokens,(p.first,"=",_spectre_value(p.second))),props)
        return tokens
    elseif master=="nport"
        !isempty(nodes) && isempty(trailing) ||
            _spice_failure(card,"Spectre nport requires only signal terminal nodes")
        length(props)==1 && only(props).first=="file" ||
            _spice_failure(card,"Spectre nport supports only file= Touchstone databanks")
        path=only(props).second
        return ["N_"*name;[_spectre_identifier(n) for n in nodes];"file";"=";path]
    elseif master=="transformer"
        length(nodes)==4 && isempty(trailing) ||
            _spice_failure(card,"Spectre transformer requires primary and secondary node pairs")
        length(props)==1 && only(props).first=="ratio" ||
            _spice_failure(card,"Spectre transformer supports only ratio=")
        return ["Z_"*name,_spectre_identifier(nodes[1]),_spectre_identifier(nodes[2]),
            _spectre_identifier(nodes[3]),_spectre_identifier(nodes[4]),value("ratio")]
    end
    _spice_failure(card,"unsupported Spectre record; only linear primitives, controlled sources, DC sources, tline, instances and parameters are ingested")
end

"""Read the bounded, explicit Spectre subset used by Sonnet's readable
linear RLC/IME exports. Returns [`PlanarSpiceLibrary`](@ref) with
`dialect=:spectre`, original operative records, source lines and SHA256.
Compile and attach with [`planar_spice_model`](@ref) and
[`circuit_add_spice!`](@ref). Names and SI factors are case sensitive;
`M` means mega, `m` milli, and literal node `0` is global ground.
`gnd` is an ordinary Spectre node name.

Supported grammar is one `simulator lang=spectre` declaration followed
by `subckt`/`ends` definitions, literal scalar `resistor r=`, `capacitor
c=`, `inductor l=` cards, named `mutual_inductor coupling= ind1= ind2=`,
`vcvs`/`vccs` node-pair and `cccs`/`ccvs` `probe=` controlled sources,
DC-only `vsource`/`isource` (`type=dc`), `tline` (`z0` plus `td` or
`f`/`nl`), file-backed `nport file=` Touchstone databanks, ideal
`transformer ratio=` sections, nested instances `name nodes... master
[param=value]...` and `parameters` declarations at file or subckt
scope. `nport` attaches its file's signal terminals to the common
reference and interpolates inside sampled coverage only. Parenthesized node
lists are optional; component labels must start with their R/L/C class,
as in the retained native exports. Full-line semicolon comments are
accepted. Unambiguous SI scales may have ignored unit suffixes; mixed
exponent/scale tokens reject. Values may be literals, parameter names
or `{...}` expressions resolved in the local parameter context. Includes,
vendor/nonlinear `model=` devices, transient source types and analysis
cards reject with source diagnostics. This subset is verified against
retained native exports and Nexxim netlist equivalence; it does not claim
arbitrary Spectre/vendor compatibility. All owned records and lexical workspace share `max_bytes`
before numeric parsing, and `root` constrains the source's resolved
location."""
function planar_read_spectre(path::AbstractString;root=dirname(abspath(path)),
        max_bytes=16*1024^2,max_records=10000,max_line_bytes=65536)
    max_bytes=_spice_limit("max_bytes",max_bytes);max_records=_spice_limit("max_records",max_records)
    max_line_bytes=_spice_limit("max_line_bytes",max_line_bytes)
    isdir(root) || throw(ArgumentError("Spectre root must be a directory"))
    directory=realpath(root);source=realpath(path)
    first(splitpath(relpath(source,directory)))!=".." || throw(ArgumentError("Spectre source resolves outside root"))
    b=_SpiceBudget(0,max_bytes,max_records,0,1)
    _spice_reserve!(b,2048+4filesize(source)+256+2ncodeunits(source))
    provenance=Dict(source=>open(io->bytes2hex(SHA.sha256(io)),source))
    records=_SpiceCard[];definitions=Dict{String,_SpiceDefinition}();current=nothing
    declared=false;names=Set{String}();spectreparams=Pair{String,String}[]
    open(source) do io
        for (line,raw) in enumerate(eachline(io))
            ncodeunits(raw)<=max_line_bytes || _spice_failure(_SpiceCard(source,line,"",String[]),"Spectre line exceeds max_line_bytes")
            text=strip(line==1 ? replace(raw,'\ufeff'=>"") : raw)
            (isempty(text)||startswith(text,';')) && continue
            (occursin(';',text)||occursin('$',text)) &&
                _spice_failure(_SpiceCard(source,line,String(text),String[]),"inline SPICE-style comments are unsupported in this Spectre subset")
            b.records+=1;b.records<=max_records || _spice_failure(_SpiceCard(source,line,"",String[]),"Spectre record count exceeds max_records")
            workspace=1024+32ncodeunits(text);_spice_reserve!(b,workspace)
            # Parentheses only group node lists in the supported grammar;
            # they carry no expression semantics here, so they normalize
            # away. Parentheses inside {…} expressions keep their meaning.
            chars=collect(text);depth=0
            for (i,ch) in pairs(chars)
                ch=='{' && (depth+=1);ch=='}' && (depth=max(depth-1,0))
                depth==0 && (ch=='(' || ch==')') && (chars[i]=' ')
            end
            bare=String(chars)
            fields=try _spice_tokens(bare) catch e
                _spice_failure(_SpiceCard(source,line,String(text),String[]),sprint(showerror,e))
            end
            isempty(fields) && continue
            _spice_reserve!(b,256+128length(fields)+2ncodeunits(bare))
            card=_SpiceCard(source,line,String(text),fields);push!(records,card)
            try
                if fields==["simulator","lang","=","spectre"]
                    !declared && current===nothing && isempty(definitions) || _spice_failure(card,"misplaced or duplicate Spectre dialect declaration")
                    declared=true;continue
                end
                declared || _spice_failure(card,"missing simulator lang=spectre declaration")
                if first(fields)=="subckt"
                    current===nothing && length(fields)>=3 || _spice_failure(card,"invalid or nested Spectre subckt")
                    eq=findfirst(==("="),fields);pinend=eq===nothing ? length(fields)+1 : eq-1
                    pinend>=3 || _spice_failure(card,"Spectre subckt requires a name before defaults")
                    eq===nothing || (length(fields)-pinend+1)%3==0 &&
                        all(k->fields[pinend+3k+1]=="=",0:(length(fields)-pinend+1)÷3-1) ||
                        _spice_failure(card,"Spectre subckt defaults require name=value assignments")
                    name=_spectre_identifier(fields[2]);pins=_spectre_identifier.(fields[3:pinend-1])
                    !("0" in pins) && length(unique(pins))==length(pins) || _spice_failure(card,"invalid Spectre formal pins")
                    haskey(definitions,name) && _spice_failure(card,"duplicate Spectre subckt name")
                    defaults=Pair{String,String}[]
                    if eq!==nothing
                        for k in pinend:3:length(fields)
                            key=_spectre_identifier(fields[k]);occursin(r"^[a-zA-Z][a-zA-Z0-9_]*$",key) ||
                                _spice_failure(card,"unsupported Spectre default parameter identifier")
                            push!(defaults,lowercase(key)=>_spectre_value(fields[k+2]))
                        end
                        length(unique(first.(defaults)))==length(defaults) ||
                            _spice_failure(card,"duplicate Spectre formal parameter")
                    end
                    _spice_reserve!(b,512+128length(pins)+128length(defaults))
                    current=_SpiceDefinition(name,pins,defaults,_SpiceCard[])
                    definitions[name]=current;empty!(names)
                elseif first(fields)=="ends"
                    current!==nothing && (length(fields)==1 ||
                        (length(fields)==2 && fields[2]==current.name)) ||
                        _spice_failure(card,"mismatched Spectre ends")
                    current=nothing
                elseif first(fields)=="parameters"
                    pairs=_spectre_assignments(fields[2:end],card)
                    if current===nothing
                        append!(spectreparams,pairs)
                    else
                        keys=Set(first.(current.defaults))
                        length(unique(first.(pairs)))==length(pairs) &&
                            !any(p->p.first in keys,pairs) ||
                            _spice_failure(card,"duplicate Spectre subckt parameter")
                        append!(current.defaults,pairs)
                    end
                else
                    current!==nothing || _spice_failure(card,"Spectre card lies outside a subckt")
                    first(fields) in ("include","ahdl_include","simulator","analysis","tran","ac","dc","model") &&
                        _spice_failure(card,"unsupported Spectre card class $(first(fields))")
                    tokens=_spectre_component(card,fields,names,definitions)
                    _spice_reserve!(b,256+128length(tokens)+2sum(ncodeunits,tokens))
                    push!(current.cards,_SpiceCard(source,line,String(text),tokens))
                end
            catch e
                e isa ArgumentError || rethrow()
                startswith(sprint(showerror,e),"ArgumentError: $source:") && rethrow()
                _spice_failure(card,sprint(showerror,e))
            finally
                b.used-=workspace
            end
        end
    end
    declared && !isempty(definitions) && current===nothing || throw(ArgumentError("$source: unfinished or missing Spectre linear definition"))
    length(unique(first.(spectreparams)))==length(spectreparams) ||
        throw(ArgumentError("$source: duplicate Spectre top-level parameter"))
    PlanarSpiceLibrary(source,directory,definitions,spectreparams,provenance,b.used,records,
        Tuple{String,Int,String,String}[],:spectre)
end
