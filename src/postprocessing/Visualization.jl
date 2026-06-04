# Visualization.jl — package-level mesh plotting utilities

export plot_mesh_wireframe, plot_mesh_comparison, save_mesh_preview

using PlotlySupply

function _realistic_axis_limits(meshes::Vector{TriMesh}; pad_frac::Float64=0.04)
    mins = [Inf, Inf, Inf]
    maxs = [-Inf, -Inf, -Inf]
    for mesh in meshes
        for i in 1:3
            mins[i] = min(mins[i], minimum(@view mesh.xyz[i, :]))
            maxs[i] = max(maxs[i], maximum(@view mesh.xyz[i, :]))
        end
    end

    spans = maxs .- mins
    max_span = max(maximum(spans), 1e-9)
    centers = (mins .+ maxs) ./ 2
    half = 0.5 * max_span * (1 + pad_frac)

    xlims = (centers[1] - half, centers[1] + half)
    ylims = (centers[2] - half, centers[2] + half)
    zlims = (centers[3] - half, centers[3] + half)
    return xlims, ylims, zlims
end

_color_str(c) = c isa Symbol ? String(c) : String(c)

function _camera_eye(camera::Tuple{Real, Real}; radius::Float64=1.85)
    az, el = camera
    x = radius * cosd(el) * cosd(az)
    y = radius * cosd(el) * sind(az)
    z = radius * sind(el)
    return attr(x = x, y = y, z = z)
end

_axis_range(lims) = lims === nothing ? nothing : [Float64(lims[1]), Float64(lims[2])]

function _scene_axis(title::AbstractString, lims; guidefontsize::Int=12, tickfontsize::Int=10)
    ax = attr(
        title = attr(text = title, font = attr(size = guidefontsize)),
        showgrid = true,
        zeroline = false,
        ticks = "outside",
        tickfont = attr(size = tickfontsize),
    )
    r = _axis_range(lims)
    r === nothing || (ax[:range] = r)
    return ax
end

function _scene_attr(; camera::Tuple{Real, Real},
                      xlims=nothing,
                      ylims=nothing,
                      zlims=nothing,
                      guidefontsize::Int=12,
                      tickfontsize::Int=10)
    return attr(
        xaxis = _scene_axis("x (m)", xlims; guidefontsize = guidefontsize, tickfontsize = tickfontsize),
        yaxis = _scene_axis("y (m)", ylims; guidefontsize = guidefontsize, tickfontsize = tickfontsize),
        zaxis = _scene_axis("z (m)", zlims; guidefontsize = guidefontsize, tickfontsize = tickfontsize),
        aspectmode = "cube",
        camera = attr(eye = _camera_eye(camera)),
        bgcolor = "rgba(0,0,0,0)",
    )
end

_layout_int(layout, key::Symbol, default::Int) = begin
    v = default
    try
        v = layout[key]
    catch
        layout isa AbstractDict && (v = get(layout, key, default))
    end
    return v isa Number ? Int(round(v)) : default
end

"""
    plot_mesh_wireframe(mesh; kwargs...)

Create a 3D wireframe plot of a mesh using package-native mesh segments.
"""
function plot_mesh_wireframe(mesh::TriMesh;
                             color=:steelblue,
                             title::AbstractString="Mesh",
                             camera::Tuple{Real,Real}=(30, 30),
                             linewidth::Real=0.7,
                             xlims=nothing,
                             ylims=nothing,
                             zlims=nothing,
                             size::Tuple{Int,Int}=(700, 500),
                             guidefontsize::Int=12,
                             tickfontsize::Int=10,
                             titlefontsize::Int=12,
                             kwargs...)
    seg = mesh_wireframe_segments(mesh)
    specs = reshape([Spec(kind = "scene")], 1, 1)
    titles = reshape([title], 1, 1)
    sf = subplots(1, 1; sync = false, width = size[1], height = size[2], specs = specs, subplot_titles = titles, per_subplot_legends = false)
    addtraces!(
        sf,
        scatter3d(
            x = seg.x,
            y = seg.y,
            z = seg.z,
            mode = "lines",
            line = attr(color = _color_str(color), width = linewidth),
            showlegend = false,
            hoverinfo = "skip",
        );
        row = 1,
        col = 1,
    )
    p = sf.plot
    relayout!(
        p,
        scene = _scene_attr(
            camera = camera,
            xlims = xlims,
            ylims = ylims,
            zlims = zlims,
            guidefontsize = guidefontsize,
            tickfontsize = tickfontsize,
        ),
        margin = attr(l = 30, r = 30, t = max(60, titlefontsize + 46), b = 30),
        font = attr(size = titlefontsize),
    )
    isempty(kwargs) || relayout!(p; kwargs...)
    return p
end

"""
    plot_mesh_comparison(mesh_a, mesh_b; kwargs...)

Create side-by-side 3D wireframe plots with shared axis limits and equal aspect.
"""
function plot_mesh_comparison(mesh_a::TriMesh, mesh_b::TriMesh;
                              title_a::AbstractString="Mesh A",
                              title_b::AbstractString="Mesh B",
                              color_a=:steelblue,
                              color_b=:darkorange,
                              camera::Tuple{Real,Real}=(30, 30),
                              size::Tuple{Int,Int}=(1200, 520),
                              pad_frac::Float64=0.04,
                              linewidth::Real=0.7,
                              guidefontsize::Int=10,
                              tickfontsize::Int=8,
                              titlefontsize::Int=10,
                              kwargs...)
    xlims, ylims, zlims = _realistic_axis_limits([mesh_a, mesh_b]; pad_frac=pad_frac)
    seg_a = mesh_wireframe_segments(mesh_a)
    seg_b = mesh_wireframe_segments(mesh_b)

    specs = reshape([Spec(kind = "scene"), Spec(kind = "scene")], 1, 2)
    titles = reshape([title_a, title_b], 1, 2)
    sf = subplots(1, 2; sync = false, width = size[1], height = size[2], specs = specs, subplot_titles = titles, per_subplot_legends = false)

    addtraces!(
        sf,
        scatter3d(
            x = seg_a.x,
            y = seg_a.y,
            z = seg_a.z,
            mode = "lines",
            line = attr(color = _color_str(color_a), width = linewidth),
            showlegend = false,
            hoverinfo = "skip",
        );
        row = 1,
        col = 1,
    )
    addtraces!(
        sf,
        scatter3d(
            x = seg_b.x,
            y = seg_b.y,
            z = seg_b.z,
            mode = "lines",
            line = attr(color = _color_str(color_b), width = linewidth),
            showlegend = false,
            hoverinfo = "skip",
        );
        row = 1,
        col = 2,
    )
    p = sf.plot
    relayout!(
        p,
        scene = _scene_attr(
            camera = camera,
            xlims = xlims,
            ylims = ylims,
            zlims = zlims,
            guidefontsize = guidefontsize,
            tickfontsize = tickfontsize,
        ),
        scene2 = _scene_attr(
            camera = camera,
            xlims = xlims,
            ylims = ylims,
            zlims = zlims,
            guidefontsize = guidefontsize,
            tickfontsize = tickfontsize,
        ),
        margin = attr(l = 30, r = 30, t = max(72, titlefontsize + 54), b = 30),
        font = attr(size = titlefontsize),
    )
    isempty(kwargs) || relayout!(p; kwargs...)
    return p
end

"""
    save_mesh_preview(mesh_a, mesh_b, out_prefix; kwargs...)

Generate and save side-by-side mesh preview plots as PNG and PDF.
Returns a named tuple with plot object and file paths.
"""
function save_mesh_preview(mesh_a::TriMesh, mesh_b::TriMesh, out_prefix::AbstractString; kwargs...)
    mkpath(dirname(out_prefix))
    p = plot_mesh_comparison(mesh_a, mesh_b; kwargs...)
    png_path = out_prefix * ".png"
    pdf_path = out_prefix * ".pdf"
    width = _layout_int(p.layout, :width, 700)
    height = _layout_int(p.layout, :height, 500)
    savefig(p, png_path; width = width, height = height)
    savefig(p, pdf_path; width = width, height = height)
    return (plot=p, png_path=png_path, pdf_path=pdf_path)
end
