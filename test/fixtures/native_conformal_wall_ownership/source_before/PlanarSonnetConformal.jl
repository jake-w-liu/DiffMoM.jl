export sonnet_conformal_layout,solve_sonnet_conformal

"""Lower a native Sonnet project onto genuine physical sheet triangles.
The native stack/material helper preserves SI units, variable expressions,
cover impedances, dielectric loss and physical two-face thick geometry.
Sheet polygons are unioned exactly, with material seams and shared axial
source cuts. Tiny sheet features do not pass through a raster lowerer.
`edge_size,interior_size` declare independent physical mesh bounds.

Via columns retain the native uniform-grid SOLID/RING/CENTER/VERTICES/BAR
footprint contract and exact axial kernels; their cell boundaries constrain
the triangular contacts. This is exact sheet geometry, with a declared
grid approximation for arbitrary via footprints. Bare open interior
terminals require explicit physical return/bridge geometry. Components,
bricks and unsupported native port/material semantics reject.
Returns `(layout,contraction,floating_common,z0,labels,via_sigma,geometry_project)`.
Shared native terminal numbers preserve common voltages; corresponding
negative terminals supply floating references with balanced total current."""
function sonnet_conformal_layout(project::SonnetProject;freq::Real=1e9,
        grid=nothing,variables=Dict{String,Float64}(),
        edge_size::Real,interior_size::Real,edge_band::Real=2edge_size,
        max_triangles::Integer=100_000,max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES,
        scalar_files=nothing,scalar_root=dirname(project.source),scalar_outside::Symbol=:reject,
        scalar_max_files::Integer=64,scalar_max_bytes::Integer=8*1024^2,
        scalar_max_nodes::Integer=100000,scalar_max_line_bytes::Integer=16384)
    isfinite(freq) && freq>0 || throw(ArgumentError("frequency must be positive and finite"))
    freq=_circuit_stored_real(freq,"native conformal frequency")
    if scalar_files!==nothing || _sonnet_has_scalar_tables(project) || variables isa SonnetScalarVariables
        variables=_sonnet_scalar_variables(project,variables;scalar_files,max_bytes,
            root=scalar_root,outside=scalar_outside,max_files=scalar_max_files,
            max_bytes_source=scalar_max_bytes,max_nodes=scalar_max_nodes,max_line_bytes=scalar_max_line_bytes)
        project=_sonnet_scalar_project(project,variables)
    end
    isempty(project.components) || throw(ArgumentError("native conformal components require their physical source/network lowering"))
    native=_sonnet_stack_geometry(project,freq,grid,variables);p=native.geometry_project
    variables=native.variables
    stack,gr=native.stack,native.grid;L=length(stack.layers);val(x)=sonnet_variable_value(p,x;variables,freq)
    any(poly->poly.kind==:brick,p.polygons) && throw(ArgumentError("native dielectric brick needs its physical volume dielectric model"))
    ne=sum((size(poly.vertices,2) for poly in p.polygons);init=0);cells=BigInt(gr.nx)*gr.ny
    nvia=BigInt(0);nvpoly=0
    for poly in p.polygons
        poly.kind===:via || continue
        target=poly.target=="GND" ? L-1 : poly.target=="TOP" ? -1 : parse(Int,poly.target)
        nvia+=abs(BigInt(poly.level)-target);nvpoly+=1
    end
    reserve=_checked_payload_sum("native conformal metadata",_checked_array_payload_bytes(Float64,8,ne),
        _checked_array_payload_bytes(Int,8,ne),_checked_array_payload_bytes(UInt64,4+2nvia+nvpoly,cld(cells,64)),
        _checked_array_payload_bytes(ComplexF64,length(p.metals)+1),
        _sonnet_scalar_files(variables)===nothing ? 0 : _sonnet_scalar_payload(variables))
    _enforce_payload_limit(reserve,max_bytes,"native conformal metadata","max_bytes")
    polygons=PlanarPolygon[];metals=Dict{String,Any}("pec"=>0.)
    contacts=PlanarConformalPort[];cp=PlanarConformalPort[];bp=PlanarPort[]
    vias=ViaLevel[];sigma=Float64[];via_group=Dict{Tuple{Int,Float64},Int}();via_indices=Dict{Tuple{Int,Int},Int}();masks=Dict{Int,BitMatrix}()
    numbers=Int[];refs=ComplexF64[];weights=Float64[];cnumbers=Int[];crefs=ComplexF64[]
    function add_sheet(name,level,vertices,metal)
        v,b=planar_normalize_polygon([_P2(vertices[1,i],vertices[2,i]) for i in axes(vertices,2)];label=name)
        push!(polygons,PlanarPolygon(name,level,metal,"",v,b))
        for i in eachindex(v)
            a,b=v[i],v[mod1(i+1,length(v))]
            wall=a[1]==b[1]==0. ? :west : a[1]==b[1]==stack.a ? :east :
                a[2]==b[2]==0. ? :south : a[2]==b[2]==stack.b ? :north : nothing
            wall===nothing && continue
            dim=wall in (:west,:east) ? 2 : 1;lo,hi=minmax(a[dim],b[dim])
            push!(contacts,PlanarConformalPort(level,wall,(lo,hi)))
        end
    end
    for poly in p.polygons
        if poly.kind===:sheet
            metal=poly.material==-1 ? "pec" : "native_$(poly.material)"
            if poly.material!=-1
                0<=poly.material<length(p.metals) || throw(ArgumentError("native sheet metal index is invalid"))
                record=p.metals[poly.material+1];sonnet_metal_zs(p,record,freq;variables)
                metals[metal]=f->sonnet_metal_zs(p,record,f;variables)
            end
            add_sheet("native_$(poly.id)",L-1-poly.level,poly.vertices,metal)
        else
            target=poly.target=="GND" ? L-1 : poly.target=="TOP" ? -1 : parse(Int,poly.target)
            lo,hi=minmax(poly.level,target);-1<=lo<hi<=L-1 || throw(ArgumentError("native via target lies outside the stack"))
            footprint=sheet_level(0,gr.nx,gr.ny);rasterize_poly!(footprint,gr,poly.vertices[1,:],poly.vertices[2,:])
            any(footprint.mask) || throw(ArgumentError("native via footprint vanished at its declared bulk grid"))
            mode="SOLID" in poly.flags || "FULL" in poly.flags ? :full : "VERTICES" in poly.flags ? :vertices :
                "CENTER" in poly.flags ? :center : "BAR" in poly.flags ? :bar : :ring
            mask=_planar_via_mesh_mask(footprint.mask,poly.vertices,gr,mode);masks[poly.id]=mask
            rho_sigma=_sonnet_via_sigma(p,poly,gr,stack,mask,freq,variables)
            if "COVERS" in poly.flags
                for nativelevel in (poly.level,target)
                    nativelevel in (-1,L-1) || add_sheet("native_via_pad_$(poly.id)_$nativelevel",L-1-nativelevel,poly.vertices,"pec")
                end
            end
            for layer in (L-hi):(L-lo-1)
                key=(layer,rho_sigma);index=get(via_group,key,0)
                for (other,v) in enumerate(vias)
                    v.layer==layer && any(v.uni .& mask) && (other!=index || isfinite(rho_sigma)) && throw(ArgumentError("native via resistance footprints overlap in one physical layer"))
                end
                if index==0
                    push!(vias,via_level(layer,gr.nx,gr.ny));push!(sigma,rho_sigma);index=length(vias);via_group[key]=index
                end
                vias[index].uni .|=mask;vias[index].tap .|=mask;via_indices[(poly.id,layer)]=index
            end
        end
    end
    polydict=Dict(poly.id=>poly for poly in p.polygons)
    for ps in p.ports
        ps.kind in (:box,:std,:gap,:via) || throw(ArgumentError("native conformal $(ps.kind) port requires its physical return/calibration model"))
        poly=polydict[ps.polygon];q=ps.values;z0=_sonnet_port_reference(p,ps,freq,variables)
        ps.kind===:via && poly.kind!==:via && throw(ArgumentError("native via port requires a via polygon"))
        if poly.kind===:via && ps.kind in (:std,:via)
            ps.number!=0 || throw(ArgumentError("native interior port zero needs an explicit physical ground model"))
            target=poly.target=="GND" ? L-1 : poly.target=="TOP" ? -1 : parse(Int,poly.target)
            ps.kind===:std && !(poly.level in (-1,L-1) || target in (-1,L-1)) &&
                throw(ArgumentError("native standard via port must terminate on a box cover"))
            lo,hi=minmax(poly.level,target);layers=(L-hi):(L-lo-1);height=sum(real(stack.layers[l].thickness) for l in layers)
            for l in layers,cell in findall(vec(masks[poly.id]))
                push!(bp,PlanarPort(via_indices[(poly.id,l)],:via,cell:cell,z0;polarity=ps.number<0 ? -1 : 1))
                push!(numbers,abs(ps.number));push!(refs,z0);push!(weights,real(stack.layers[l].thickness)/height)
            end
            continue
        end
        poly.kind===:sheet || throw(ArgumentError("native sheet port must reference a sheet polygon"))
        v=poly.vertices;n=_sonnet_polygon_edge_count(v);i,j=_sonnet_port_edge_indices(v,ps.edge)
        a=(v[1,i],v[2,i]);b=(v[1,j],v[2,j]);level=L-1-poly.level
        wall=a[1]==b[1]==0. ? :west : a[1]==b[1]==stack.a ? :east :
            a[2]==b[2]==0. ? :south : a[2]==b[2]==stack.b ? :north : nothing
        if wall===nothing
            ps.number!=0 || throw(ArgumentError("native interior port zero requires its explicit reference conductor"))
            (a[1]==b[1] || a[2]==b[2]) ||
                throw(ArgumentError("native internal port is diagonal"))
            a,b=_sonnet_shared_port_segment(p,poly,i,j)
            # Native internal source orientation is independent of which
            # adjacent polygon is referenced; match positive raster axes.
            dx,dy=b[1]-a[1],b[2]-a[2]
            (abs(dy)>abs(dx) ? dy>0 : dx<0) && ((a,b)=(b,a))
            push!(cp,PlanarConformalPort(level,:internal,(a,b);z0,polarity=ps.number<0 ? -1 : 1))
        else
            ps.number==0 && continue
            dim=wall in (:west,:east) ? 2 : 1;normal=3-dim;position=a[normal]
            lo,hi=minmax(a[dim],b[dim])
            # Collinear subdivisions share one physical wall excitation.
            for direction in (-1,1)
                k=direction==-1 ? i : j
                for _ in 1:n-2
                    next=mod1(k+direction,n)
                    v[normal,next]==position || break
                    lo=min(lo,v[dim,next]);hi=max(hi,v[dim,next]);k=next
                end
            end
            push!(cp,PlanarConformalPort(level,wall,(lo,hi);z0,polarity=ps.number<0 ? -1 : 1))
        end
        push!(cnumbers,abs(ps.number));push!(crefs,z0)
    end
    allnumbers=vcat(cnumbers,numbers);allrefs=vcat(crefs,refs);allweights=vcat(ones(length(cp)),weights)
    isempty(allnumbers) && throw(ArgumentError("native conformal solve requires driven ports"))
    layout=if isempty(polygons)
        bulk=build_planar_problem(stack,gr,SheetLevel[],bp;vias)
        PlanarConformalLayout(bulk,polygons,Int[],String[],Any[])
    else
        build_planar_conformal_layout(stack,polygons,cp;metals,wall_contacts=contacts,
            bulk_grid=isempty(vias) ? nothing : gr,vias,bulk_ports=bp,edge_size,interior_size,edge_band,max_triangles,max_bytes=max_bytes-reserve)
    end
    _enforce_payload_limit(_checked_payload_sum("native conformal contraction",reserve,
        _planar_conformal_layout_payload(layout),_checked_array_payload_bytes(Float64,3,length(allnumbers),length(unique(allnumbers)))),
        max_bytes,"native conformal contraction","max_bytes")
    labels=sort!(unique(allnumbers));C=zeros(Float64,length(allnumbers),length(labels));z0=ComplexF64[]
    for (j,label) in enumerate(labels)
        ids=findall(==(label),allnumbers);all(allrefs[i]==allrefs[first(ids)] for i in ids) || throw(ArgumentError("shared native terminal references must agree"))
        C[ids,j].=allweights[ids];push!(z0,allrefs[first(ids)])
    end
    terminal_maps=_sonnet_terminal_maps(C,[port.polarity for port in vcat(cp,bp)])
    return (;layout,terminal_maps...,z0,labels,via_sigma=sigma,geometry_project=p,
        scalar_files=_sonnet_scalar_files(variables))
end

"""Solve a native project with genuine sheet polygons and analytic
triangle/bulk Galerkin reactions. Native calibration requests require
`calibration` or explicit `raw=true`; gap-plane results retain actual
current coefficients. `edge_size,interior_size` control physical meshing;
no sheet grid approximation is introduced."""
function solve_sonnet_conformal(project::SonnetProject,freq::Real;raw::Bool=false,calibration=nothing,
        grid=nothing,variables=Dict{String,Float64}(),edge_size::Real,interior_size::Real,edge_band::Real=2edge_size,
        max_triangles::Integer=100_000,max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES,
        scalar_files=nothing,scalar_root=dirname(project.source),scalar_outside::Symbol=:reject,
        scalar_max_files::Integer=64,scalar_max_bytes::Integer=8*1024^2,
        scalar_max_nodes::Integer=100000,scalar_max_line_bytes::Integer=16384,kw...)
    requested=any(r->r.tokens[1]=="OPTIONS" && any(occursin("d",t) for t in r.tokens[2:end]),project.records)
    requested && !raw && calibration===nothing && throw(ArgumentError("native conformal project requests calibration; supply it or explicitly request raw=true"))
    model=sonnet_conformal_layout(project;freq,grid,variables,edge_size,interior_size,edge_band,max_triangles,max_bytes,
        scalar_files,scalar_root,scalar_outside,scalar_max_files,scalar_max_bytes,scalar_max_nodes,scalar_max_line_bytes)
    n=size(model.contraction,2);nr=size(model.contraction,1)
    reserve=_checked_payload_sum("native conformal contraction",_checked_array_payload_bytes(ComplexF64,8,nr,n),
        _checked_array_payload_bytes(ComplexF64,12,n,n),_checked_array_payload_bytes(Float64,2,nr,n),
        _planar_conformal_layout_payload(model.layout),
        sum(_checked_array_payload_bytes(Float64,size(p.vertices)...) for p in model.geometry_project.polygons),
        model.scalar_files===nothing ? 0 : _sonnet_scalar_files_payload(model.scalar_files))
    _enforce_payload_limit(reserve,max_bytes,"native conformal contraction","max_bytes")
    result=model.layout.problem isa Union{PlanarHybridProblem,PlanarProblem} ?
        solve_planar(model.layout,freq;via_sigma=model.via_sigma,max_bytes=max_bytes-reserve,kw...) :
        solve_planar(model.layout,freq;max_bytes=max_bytes-reserve,kw...)
    terminal=_sonnet_balanced_y(result.y,model);y=terminal.y;voltage_transfer=terminal.voltage_transfer
    if calibration!==nothing
        y=deembed_ports(y,calibration)
        A,B,_,_=_calibration_chain_blocks(calibration,length(model.labels))
        voltage_transfer=voltage_transfer*(A+B*y)
    end
    retained_project=model.scalar_files===nothing ? project : model.scalar_files.project
    SonnetPlanarResult(retained_project,result,model.labels,y,planar_y_to_s(y,model.z0),voltage_transfer,model.z0,model.scalar_files)
end
solve_sonnet_conformal(path::AbstractString,freq::Real;kw...)=solve_sonnet_conformal(read_sonnet_project(path),freq;kw...)
