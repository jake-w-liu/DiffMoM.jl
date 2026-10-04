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

"""Read the bounded, explicit Spectre subset used by Sonnet's readable
linear RLC/IME exports. Returns [`PlanarSpiceLibrary`](@ref) with
`dialect=:spectre`, original operative records, source lines and SHA256.
Compile and attach with [`planar_spice_model`](@ref) and
[`circuit_add_spice!`](@ref). Names and SI factors are case sensitive;
`M` means mega, `m` milli, and literal node `0` is global ground.
`gnd` is an ordinary Spectre node name.

Supported grammar is one `simulator lang=spectre` declaration followed
by `subckt`/`ends` definitions, literal scalar `resistor r=`, `capacitor
c=`, `inductor l=` cards and named `mutual_inductor coupling= ind1= ind2=`.
The component labels must start with their R/L/C class, as in the retained
native exports. Full-line semicolon comments are accepted. Unambiguous
SI scales may have ignored unit suffixes; mixed exponent/scale tokens
reject. Includes, expressions/parameters, nested Spectre instances,
models, nonlinear/behavioural and analysis cards reject with source
diagnostics. This subset is verified against retained native exports and
ngspice equations; it does not claim arbitrary Spectre/vendor compatibility.
All owned records and lexical workspace share `max_bytes` before numeric
parsing, and `root` constrains the source's resolved location."""
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
    declared=false;names=Set{String}()
    open(source) do io
        for (line,raw) in enumerate(eachline(io))
            ncodeunits(raw)<=max_line_bytes || _spice_failure(_SpiceCard(source,line,"",String[]),"Spectre line exceeds max_line_bytes")
            text=strip(line==1 ? replace(raw,'\ufeff'=>"") : raw)
            (isempty(text)||startswith(text,';')) && continue
            (occursin(';',text)||occursin('$',text)) &&
                _spice_failure(_SpiceCard(source,line,String(text),String[]),"inline SPICE-style comments are unsupported in this Spectre subset")
            b.records+=1;b.records<=max_records || _spice_failure(_SpiceCard(source,line,"",String[]),"Spectre record count exceeds max_records")
            workspace=1024+32ncodeunits(text);_spice_reserve!(b,workspace)
            fields=try _spice_tokens(text) catch e
                _spice_failure(_SpiceCard(source,line,String(text),String[]),sprint(showerror,e))
            end
            _spice_reserve!(b,256+128length(fields)+2ncodeunits(text))
            card=_SpiceCard(source,line,String(text),fields);push!(records,card)
            try
                if fields==["simulator","lang","=","spectre"]
                    !declared && current===nothing && isempty(definitions) || _spice_failure(card,"misplaced or duplicate Spectre dialect declaration")
                    declared=true;continue
                end
                declared || _spice_failure(card,"missing simulator lang=spectre declaration")
                if first(fields)=="subckt"
                    current===nothing && length(fields)>=3 || _spice_failure(card,"invalid or nested Spectre subckt")
                    name=_spectre_identifier(fields[2]);pins=_spectre_identifier.(fields[3:end])
                    !("0" in pins) && length(unique(pins))==length(pins) || _spice_failure(card,"invalid Spectre formal pins")
                    haskey(definitions,name) && _spice_failure(card,"duplicate Spectre subckt name")
                    _spice_reserve!(b,512+128length(pins))
                    current=_SpiceDefinition(name,pins,Pair{String,String}[],_SpiceCard[])
                    definitions[name]=current;empty!(names)
                elseif first(fields)=="ends"
                    current!==nothing && length(fields)==2 && fields[2]==current.name || _spice_failure(card,"mismatched Spectre ends")
                    current=nothing
                else
                    current!==nothing || _spice_failure(card,"Spectre card lies outside a subckt")
                    name=_spectre_identifier(fields[1]);name in names && _spice_failure(card,"duplicate Spectre component name")
                    push!(names,name);tokens=String[]
                    if length(fields)==7 && fields[4] in ("resistor","capacitor","inductor")
                        kind=fields[4]=="resistor" ? 'R' : fields[4]=="capacitor" ? 'C' : 'L'
                        uppercase(first(name))==kind && fields[5]==lowercase(string(kind)) && fields[6]=="=" ||
                            _spice_failure(card,"unsupported Spectre linear component label/property")
                        value=_spectre_number(fields[7])
                        tokens=[name,_spectre_identifier(fields[2]),_spectre_identifier(fields[3]),string(value)]
                    elseif length(fields)==11 && fields[2]=="mutual_inductor"
                        all(k->fields[k+1]=="=",3:3:9) || _spice_failure(card,"invalid mutual_inductor properties")
                        props=Dict(fields[k]=>fields[k+2] for k in 3:3:9)
                        length(props)==3 && Set(keys(props))==Set(("coupling","ind1","ind2")) ||
                            _spice_failure(card,"mutual_inductor requires coupling,ind1,ind2")
                        value=_spectre_number(props["coupling"])
                        tokens=["K_"*name,_spectre_identifier(props["ind1"]),_spectre_identifier(props["ind2"]),string(value)]
                    else
                        _spice_failure(card,"unsupported Spectre record; only literal scalar RLC and mutual_inductor exports are ingested")
                    end
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
    PlanarSpiceLibrary(source,directory,definitions,Pair{String,String}[],provenance,b.used,records,
        Tuple{String,Int,String,String}[],:spectre)
end
