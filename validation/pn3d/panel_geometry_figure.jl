include("pilot.jl")
using CSV
using DataFrames
using PlotlySupply

# evidence_min_rows: 1490
# evidence_density_justification: All 298 parent and 1192 midpoint-child triangles
# are shown to identify the slot, connected flange, and fixed-facet hierarchy.
function panel_geometry_figure(input_file, paper_dir, trial_dir)
    case = JSON.parsefile(input_file)
    case["case_id"] == "slotted_panel" || error("expected the representative panel")
    mesh = TriMesh(hcat(case["vertices_m"]...), Int.(hcat(case["triangles_zero_based"]...)) .+ 1)
    pair = build_nested_rwg_pair(mesh; max_work_bytes=WORK_BYTES)
    pair.coarse_rwg.nedges == case["dof_count"] || error("RWG count differs from the stored case")
    ntriangles(pair.fine_mesh) == 4ntriangles(mesh) || error("midpoint triangle count mismatch")
    sf = subplots(1, 1; sync=false, width=504, height=400,
        specs=reshape([Spec(kind="scene")], 1, 1),
        subplot_titles=reshape([""], 1, 1), per_subplot_legends=false)
    for (geometry, name, color, width) in (
            (pair.fine_mesh, "Midpoint refinement", "#66A9AE", 1.0),
            (mesh, "Coarse mesh", "#222222", 2.2))
        segments = mesh_wireframe_segments(geometry)
        # Centimetres keep endpoint labels compact at journal print size.
        addtraces!(sf, scatter3d(x=100 .* segments.x, y=100 .* segments.y, z=100 .* segments.z,
            mode="lines", name=name, showlegend=true,
            line=attr(color=color, width=width), hoverinfo="skip"); row=1, col=1)
    end
    figure = sf.plot
    # At 504 pixels across a 252-point journal column, 24/20-pixel text prints
    # at 12/10 points. Native-size previews alone hid the former undersizing.
    function axis(label, coordinate)
        bounds = 100 .* collect(extrema(view(mesh.xyz, coordinate, :)))
        # Label both x endpoints, but only the upper y/z endpoints: lower
        # endpoints share corners with x labels in this orthographic view.
        values = coordinate == 1 ? bounds : bounds[2:2]
        return attr(title=attr(text=label, font=attr(size=24)),
            tickfont=attr(size=20), tickmode="array", tickvals=values,
            showbackground=false, zeroline=false, gridcolor="#D5D5D5", ticks="outside")
    end
    relayout!(figure; title="", font=attr(family="Times New Roman", size=12, color="#111111"),
        legend=attr(font=attr(size=20)),
        paper_bgcolor="white", margin=attr(l=12, r=12, b=10, t=10),
        scene=attr(xaxis=axis("x (cm)", 1), yaxis=axis("y (cm)", 2), zaxis=axis("z (cm)", 3),
            aspectmode="data", bgcolor="white",
            camera=attr(eye=attr(x=1.15, y=-1.65, z=1.3),
                        projection=attr(type="orthographic"))))
    data_dir, figure_dir = joinpath(paper_dir, "data"), joinpath(paper_dir, "figs")
    for directory in (data_dir, figure_dir, trial_dir)
        mkpath(directory)
    end
    data_hashes = Dict{String,String}()
    for (name, geometry) in (("coarse", mesh), ("enriched", pair.fine_mesh))
        vertices = DataFrame(x_m=geometry.xyz[1, :], y_m=geometry.xyz[2, :], z_m=geometry.xyz[3, :])
        triangles = DataFrame(vertex_1=geometry.tri[1, :], vertex_2=geometry.tri[2, :],
                              vertex_3=geometry.tri[3, :])
        for (suffix, data) in (("vertices", vertices), ("triangles", triangles))
            filename = "panel_$(name)_$(suffix).csv"
            path = joinpath(data_dir, filename)
            CSV.write(path, data)
            data_hashes[filename] = bytes2hex(sha256(read(path)))
        end
    end
    positions = (:topright, :topleft, :bottomright, :bottomleft, :top, :bottom, :right, :left)
    set_legend!(figure; position=:topright)
    savefig(figure, joinpath(trial_dir, "placeholder.pdf"); width=504, height=400)
    for position in positions
        set_legend!(figure; position=position, bgcolor="rgba(255,255,255,0.92)")
        savefig(figure, joinpath(trial_dir, "panel_$(position).pdf"); width=504, height=400)
        savefig(figure, joinpath(trial_dir, "panel_$(position).png"); width=504, height=400)
    end
    # Top right has zero mesh overlap; it wins the prescribed tie over other empty corners.
    chosen = Symbol(get(ENV, "PN_PANEL_LEGEND", "topright"))
    chosen in positions || error("unsupported inside legend position")
    set_legend!(figure; position=chosen, bgcolor="rgba(255,255,255,0.92)")
    final_pdf = joinpath(figure_dir, "panel_geometry.pdf")
    final_png = joinpath(figure_dir, "panel_geometry.png")
    savefig(figure, final_pdf; width=504, height=400)
    savefig(figure, final_png; width=504, height=400)
    manifest = (; schema_version=1, backend="Julia + PlotlySupply",
        renderer_version=string(Base.pkgversion(PlotlySupply)),
        source_sha256=source_digest(), input_sha256=bytes2hex(sha256(read(input_file))),
        driver_sha256=bytes2hex(sha256(read(@__FILE__))),
        geometry_data=data_hashes, triangle_rows=ntriangles(mesh)+ntriangles(pair.fine_mesh),
        coarse_rwg=pair.coarse_rwg.nedges, enriched_rwg=pair.fine_rwg.nedges,
        legend_position=string(chosen), aspectmode="data", display_coordinate_unit="cm",
        pdf_sha256=bytes2hex(sha256(read(final_pdf))),
        png_sha256=bytes2hex(sha256(read(final_png))))
    open(joinpath(data_dir, "panel_geometry_provenance.json"), "w") do io
        JSON.print(io, manifest, 2)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    length(ARGS) == 3 || error("usage: panel_geometry_figure.jl CASE_JSON PAPER_DIR TRIAL_DIR")
    panel_geometry_figure(abspath(ARGS[1]), abspath(ARGS[2]), abspath(ARGS[3]))
end
