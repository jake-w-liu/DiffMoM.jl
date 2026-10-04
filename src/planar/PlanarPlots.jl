using PlotlySupply: scatter, heatmap
export plot_planar_sparams,plot_planar_smith,plot_planar_layout,plot_planar_currents
export save_planar_plot

function _planar_plot_canvas(title,width,height)
    width>0 && height>0 || throw(ArgumentError("plot dimensions must be positive"))
    return subplots(1,1;sync=false,show=false,title=String(title),width=Int(width),height=Int(height))
end

function _planar_plot_pairs(pairs,n)
    result=Tuple{Int,Int}[]
    for pair in pairs
        length(pair)==2 || throw(ArgumentError("response traces need output/input port pairs"))
        p,q=pair
        p isa Integer && q isa Integer && 1<=p<=n && 1<=q<=n || throw(ArgumentError("plot port pair outside network"))
        push!(result,(Int(p),Int(q)))
    end
    isempty(result) && throw(ArgumentError("at least one response trace is required"))
    return result
end

# Plotly's JSON transport has no NaN/Inf literal. Nulls represent masked
# cells and true logarithmic nulls without assigning an invented floor.
_planar_plot_finite(values)=map(x->isfinite(x) ? x : nothing,values)

"""Interactive power-wave response curves. `pairs` selects `(output,input)`
indices. `quantity` is `:db`, `:magnitude`, `:phase` (wrapped degrees),
`:real` or `:imag`. An optional measured `reference` uses its own frequency
knots and must have matching port names and evaluated wave references.
`renormalize=true` converts the overlay at each of its frequencies to
the primary databank's reference (interpolated for sampled references).
Values are
not smoothed or clipped; zero magnitude has negative infinite dB."""
function plot_planar_sparams(data::PlanarNetworkData;
        pairs=[(1,1)],quantity::Symbol=:db,reference::Union{Nothing,PlanarNetworkData}=nothing,
        renormalize::Bool=false,
        title::AbstractString="Planar network response",width::Integer=900,height::Integer=520)
    n=_network_series_validate(data.frequencies,data.s)
    quantity in (:db,:magnitude,:phase,:real,:imag) || throw(ArgumentError("unknown response quantity"))
    selected=_planar_plot_pairs(pairs,n)
    overlay=nothing
    if reference!==nothing
        _network_series_validate(reference.frequencies,reference.s)==n && reference.port_names==data.port_names ||
            throw(ArgumentError("overlay port names/order must match"))
        overlay=Matrix{ComplexF64}[]
        for (k,f) in enumerate(reference.frequencies)
            target=_network_reference_at(data,f);source=_network_reference_at(reference,f)
            source==target || renormalize || throw(ArgumentError("overlay wave references differ at $f Hz"))
            push!(overlay,source==target ? reference.s[k] : planar_renormalize_s(reference.s[k],source,target))
        end
    end
    sf=_planar_plot_canvas(title,width,height)
    transform(z)=quantity==:db ? 20log10(abs(z)) : quantity==:magnitude ? abs(z) :
        quantity==:phase ? rad2deg(angle(z)) : quantity==:real ? real(z) : imag(z)
    for (p,q) in selected
        name="S[$(data.port_names[p]),$(data.port_names[q])]"
        addtraces!(sf,scatter(x=data.frequencies,y=_planar_plot_finite([transform(S[p,q]) for S in data.s]),
            mode="lines+markers",name=name);row=1,col=1)
        if reference!==nothing
            addtraces!(sf,scatter(x=reference.frequencies,y=_planar_plot_finite([transform(S[p,q]) for S in overlay]),
                mode="markers",name="$name reference",marker=attr(symbol="x"));row=1,col=1)
        end
    end
    ylabel=quantity==:db ? "Magnitude [dB]" : quantity==:phase ? "Wrapped phase [deg]" : String(quantity)
    relayout!(sf.plot,xaxis=attr(title="Frequency [Hz]"),yaxis=attr(title=ylabel),hovermode="x unified")
    return sf.plot
end

"""Interactive impedance Smith chart of selected terminated reflections.
The unit circle, constant normalized resistance and reactance loci are
drawn from `(z-1)/(z+1)`. Frequency [Hz] is retained in hover data. Each
reflection uses a fixed real display reference, defaulting to
`real.(data.z0)`; `z0` overrides it. Complex power-wave reflections are
converted through the terminated input impedance. The other port loads
retain their original physical terminations."""
function plot_planar_smith(data::PlanarNetworkData;ports=1:length(data.z0),
        z0=nothing,
        title::AbstractString="Impedance Smith chart",width::Integer=700,height::Integer=700)
    n=_network_series_validate(data.frequencies,data.s)
    all(p -> p isa Integer && 1<=p<=n,ports) && !isempty(ports) ||
        throw(ArgumentError("Smith chart ports must lie within the network"))
    display=_planar_reference_values(z0===nothing ? real.(data.z0) : z0,n)
    all(isreal,display) || throw(ArgumentError("impedance Smith charts require real display references"))
    reflections=[ComplexF64[_network_impedance_reflection(data.s[k][p,p],
        _network_reference_at(data,f)[p],real(display[p])) for (k,f) in enumerate(data.frequencies)] for p in ports]
    sf=_planar_plot_canvas(title,width,height)
    angles=range(0,2pi;length=361)
    function gridtrace(points,color)
        addtraces!(sf,scatter(x=real.(points),y=imag.(points),mode="lines",
            line=attr(color=color,width=1),showlegend=false,hoverinfo="skip");row=1,col=1)
    end
    gridtrace(cis.(angles),"#777777")
    for r in (.2,.5,1.,2.,5.)
        gridtrace(r/(r+1) .+ cis.(angles)./(r+1),"#dddddd")
    end
    rs=vcat(0.,10.0.^range(-4,4;length=220))
    for x in (-5.,-2.,-1.,-.5,-.2,.2,.5,1.,2.,5.)
        zs=rs .+ im*x;gridtrace((zs.-1)./(zs.+1),"#dddddd")
    end
    gridtrace(complex.([-1.,1.]),"#dddddd")
    for (p,trace) in zip(ports,reflections)
        addtraces!(sf,scatter(x=real.(trace),y=imag.(trace),
            mode="lines+markers",name="$(data.port_names[p]) ($(real(display[p])) Ω)",
            customdata=data.frequencies,hovertemplate="Γ=%{x}+j%{y}<br>f=%{customdata} Hz<extra>%{fullData.name}</extra>");row=1,col=1)
    end
    extent=max(1.1,1.05maximum(max(abs(real(s)),abs(imag(s))) for trace in reflections for s in trace))
    relayout!(sf.plot,xaxis=attr(title="Re Γ",range=[-extent,extent]),
        yaxis=attr(title="Im Γ",range=[-extent,extent],scaleanchor="x",scaleratio=1))
    return sf.plot
end

function _planar_plot_unit(unit)
    scale=get(Dict(:m=>1.,:mm=>1e-3,:um=>1e-6),unit,NaN)
    isfinite(scale) || throw(ArgumentError("plot unit must be :m, :mm or :um"))
    return scale,String(unit)
end

"""Interactive three-dimensional physical layout outlines and named pins.
Polygon heights follow bottom-to-top stack interfaces. `unit` is :m/:mm/:um.
The source geometry is displayed without changing or smoothing its mesh."""
function plot_planar_layout(layout::PlanarLayout;unit::Symbol=:um,
        title::AbstractString="Planar physical layout",width::Integer=900,height::Integer=650)
    scale,label=_planar_plot_unit(unit)
    width>0 && height>0 || throw(ArgumentError("plot dimensions must be positive"))
    sf=subplots(1,1;sync=false,show=false,title=String(title),width=Int(width),height=Int(height),
        specs=reshape([Spec(kind="scene")],1,1))
    z=real.(planar_interfaces(layout.problem.stack))
    for shape in layout.shapes
        for p in shape.polygons
            ring=vcat(p.vertices,[first(p.vertices)])
            addtraces!(sf,scatter3d(x=[v[1]/scale for v in ring],y=[v[2]/scale for v in ring],
                z=fill(z[p.level+1]/scale,length(ring)),mode="lines",name=p.name,
                line=attr(width=4),customdata=fill(p.metal,length(ring)),
                hovertemplate="%{fullData.name}<br>metal=%{customdata}<extra></extra>");row=1,col=1)
        end
        for v in shape.vias
            low,high=minmax(v.from_level,v.to_level==-1 ? 0 : v.to_level)
            x=sum(p[1] for p in v.vertices)/length(v.vertices)/scale
            y=sum(p[2] for p in v.vertices)/length(v.vertices)/scale
            addtraces!(sf,scatter3d(x=[x,x],y=[y,y],z=[z[low+1],z[high+1]]./scale,
                mode="lines+markers",name=v.name,line=attr(width=4));row=1,col=1)
        end
        for p in shape.pins
            addtraces!(sf,scatter3d(x=[p.point[1]/scale],y=[p.point[2]/scale],z=[z[p.level+1]/scale],
                mode="markers+text",text=[p.name],textposition="top center",
                name="$(shape.name).$(p.name)",showlegend=false);row=1,col=1)
        end
    end
    relayout!(sf.plot,scene=attr(aspectmode="data",xaxis=attr(title="x [$label]"),
        yaxis=attr(title="y [$label]"),zaxis=attr(title="z [$label]")))
    return sf.plot
end

"""Interactive cell-center current heatmap. `component` is :jx/:jy/:jz
or vector :magnitude. Conductor-free cells are masked, and actual complex
samples can be displayed as :magnitude/:real/:imag. Units are A/m for sheet
maps and A/m² for volume/via maps; no interpolation is applied."""
function plot_planar_currents(map::PlanarCurrentMap;component::Symbol=:magnitude,
        quantity::Symbol=:magnitude,unit::Symbol=:um,title::AbstractString="Current density",
        width::Integer=850,height::Integer=600)
    component in (:jx,:jy,:jz,:magnitude) && quantity in (:magnitude,:real,:imag) ||
        throw(ArgumentError("unknown current component or quantity"))
    component==:magnitude && quantity!=:magnitude && throw(ArgumentError("vector magnitude is real and nonnegative"))
    scale,label=_planar_plot_unit(unit)
    values=component==:magnitude ? hypot.(abs.(map.jx),abs.(map.jy),abs.(map.jz)) :
        quantity==:magnitude ? abs.(getproperty(map,component)) :
        quantity==:real ? real.(getproperty(map,component)) : imag.(getproperty(map,component))
    values[.!map.mask].=NaN
    sf=_planar_plot_canvas(title,width,height)
    density=map.kind==:sheet ? "A/m" : "A/m²"
    addtraces!(sf,heatmap(x=map.x./scale,y=map.y./scale,z=_planar_plot_finite(permutedims(values)),
        zsmooth=false,colorscale=quantity==:magnitude ? "Viridis" : "RdBu",
        zmid=quantity==:magnitude ? nothing : 0.,colorbar=attr(title=density));row=1,col=1)
    relayout!(sf.plot,xaxis=attr(title="x [$label]"),
        yaxis=attr(title="y [$label]",scaleanchor="x",scaleratio=1))
    return sf.plot
end

"""Save an interactive HTML response plot or a static image supported by
the installed Plotly renderer. HTML export does not need a desktop window."""
save_planar_plot(path::AbstractString,plot;kw...)=savefig(path,plot;kw...)
