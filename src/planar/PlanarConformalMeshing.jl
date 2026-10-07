export planar_conformal_mesh, planar_refine_conformal
export planar_refine_conformal_uniform

"""Refine every genuine triangle into four congruent children for each
of `levels` rounds. Shared edge midpoints are interned, physical boundaries
and interfaces are preserved, and no hanging edges are created. This mesh
keeps a bounded set of congruent Fourier families for exact lattice FFT
solves. Adaptive independent edge/interior sizing is provided separately
by [`planar_refine_conformal`](@ref). Array payload and `max_triangles` are
checked for the final refinement before allocation."""
function planar_refine_conformal_uniform(mesh::PlanarConformalMesh,levels::Integer;
        max_triangles::Integer=100_000,max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    0<=levels<=32 || throw(ArgumentError("uniform refinement triangle count must fit integer indexing"))
    count=BigInt(length(mesh.interfaces))*BigInt(4)^Int(levels)
    count<=max_triangles || throw(ArgumentError("uniform conformal refinement exceeds max_triangles=$max_triangles"))
    # At most three new vertices per old face in each round; two vertices
    # per final triangle is a conservative bound including the input mesh.
    _planar_conformal_mesher_budget(BigInt(size(mesh.vertices,2))+2count,count,max_bytes)
    levels==0 && return mesh
    vertices=[(mesh.vertices[1,i],mesh.vertices[2,i]) for i in axes(mesh.vertices,2)]
    faces=[(mesh.triangles[1,t],mesh.triangles[2,t],mesh.triangles[3,t]) for t in axes(mesh.triangles,2)];interfaces=copy(mesh.interfaces)
    for _ in 1:levels
        midpoints=Dict{Tuple{Int,Int},Int}();children=NTuple{3,Int}[];childlevels=Int[]
        sizehint!(children,4length(faces));sizehint!(childlevels,4length(faces))
        function midpoint(a,b)
            get!(midpoints,minmax(a,b)) do
                p,q=vertices[a],vertices[b];push!(vertices,((p[1]+q[1])/2,(p[2]+q[2])/2));length(vertices)
            end
        end
        for (t,(a,b,c)) in enumerate(faces)
            ab,bc,ca=midpoint(a,b),midpoint(b,c),midpoint(c,a)
            append!(children,((a,ab,ca),(ab,b,bc),(ca,bc,c),(ab,bc,ca)));append!(childlevels,ntuple(_->interfaces[t],4))
        end
        faces=children;interfaces=childlevels
    end
    PlanarConformalMesh(hcat((collect(p) for p in vertices)...),hcat((collect(t) for t in faces)...);interfaces,max_bytes)
end

@inline function _planar_conformal_segment_distance(x,y,a,b)
    dx,dy=b[1]-a[1],b[2]-a[2];length2=dx*dx+dy*dy
    u=clamp(((x-a[1])*dx+(y-a[2])*dy)/length2,0.,1.)
    hypot(x-a[1]-u*dx,y-a[2]-u*dy)
end

# Keep boundary coordinates local to this scalar scan. Capturing p/q in
# nested generators while reassigning them in the outer refinement loop
# boxes every segment coordinate in the hot path.
function _planar_conformal_on_boundary(coordinates,level,segments,tolerance)
    for j in 1:3
        start,finish=coordinates[j],coordinates[mod1(j+1,3)]
        for (segment_level,a,b) in segments
            segment_level==level || continue
            _planar_conformal_segment_distance(start[1],start[2],a,b)<=tolerance &&
                _planar_conformal_segment_distance(finish[1],finish[2],a,b)<=tolerance && return true
        end
    end
    return false
end

function _planar_conformal_mesher_budget(nv,nt,max_bytes)
    bytes=_checked_payload_sum("conformal meshing",
        _checked_array_payload_bytes(Float64,4,nv),
        _checked_array_payload_bytes(Int,24,nt),
        _checked_array_payload_bytes(Float64,8,nt))
    _enforce_payload_limit(bytes,max_bytes,"conformal meshing","max_bytes")
end

"""Refine genuine triangles by conforming shared-edge bisection. Physical
polygon boundaries and stack interfaces are preserved. Boundary triangles
and triangles within `edge_band` of the boundary use `edge_size`; interior
triangles use `interior_size`. Both sizes bound the longest edge. Holes and
multiple sheet interfaces already represented by `mesh` are preserved.
`max_triangles` and `max_bytes` bound growth before each bisection."""
function planar_refine_conformal(mesh::PlanarConformalMesh;
        edge_size::Real,interior_size::Real,edge_band::Real=2edge_size,
        max_triangles::Integer=100_000,max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    he,hi,band=Float64(edge_size),Float64(interior_size),Float64(edge_band)
    all(isfinite,(he,hi,band)) && 0<he<=hi && band>=0 && max_triangles>=length(mesh.interfaces) ||
        throw(ArgumentError("mesh sizes must be finite with 0<edge_size<=interior_size and edge_band>=0"))
    nv,nt=size(mesh.vertices,2),size(mesh.triangles,2)
    _planar_conformal_mesher_budget(nv,nt,max_bytes)
    vertices=[(mesh.vertices[1,i],mesh.vertices[2,i]) for i in 1:nv]
    faces=[(mesh.triangles[1,t],mesh.triangles[2,t],mesh.triangles[3,t]) for t in 1:nt];levels=copy(mesh.interfaces)
    boundary=Dict{NTuple{3,Int},Int}()
    for t in 1:nt,j in 1:3
        a,b=faces[t][j],faces[t][mod1(j+1,3)]
        key=(levels[t],min(a,b),max(a,b));boundary[key]=get(boundary,key,0)+1
    end
    segments=[(key[1],vertices[key[2]],vertices[key[3]]) for (key,count) in boundary if count==1]
    # A boundary edge remains recognizable after splitting by the exact
    # physical segment distance, with a tolerance for Float64 midpoints.
    span=maximum(hypot(a[1]-b[1],a[2]-b[2]) for (_,a,b) in segments;init=he)
    tolerance=64eps(Float64)*span
    while true
        selected=nothing;selected_ratio=1+64eps(Float64)
        for t in eachindex(faces)
            ids=faces[t];triangle_coordinates=map(i->vertices[i],ids)
            cx=sum(p[1] for p in triangle_coordinates)/3;cy=sum(p[2] for p in triangle_coordinates)/3
            distance=minimum(_planar_conformal_segment_distance(cx,cy,a,b) for (level,a,b) in segments if level==levels[t];init=Inf)
            on_boundary=_planar_conformal_on_boundary(triangle_coordinates,levels[t],segments,tolerance)
            target=on_boundary || distance<=band ? he : hi
            for j in 1:3
                a,b=ids[j],ids[mod1(j+1,3)];p,q=vertices[a],vertices[b]
                ratio=hypot(p[1]-q[1],p[2]-q[2])/target
                if ratio>selected_ratio
                    selected=(levels[t],min(a,b),max(a,b));selected_ratio=ratio
                end
            end
        end
        selected===nothing && break
        level,a,b=selected;touching=Int[]
        for t in eachindex(faces)
            levels[t]==level && a in faces[t] && b in faces[t] && push!(touching,t)
        end
        length(faces)+length(touching)<=max_triangles ||
            throw(ArgumentError("conformal refinement exceeds max_triangles=$max_triangles"))
        _planar_conformal_mesher_budget(length(vertices)+1,length(faces)+length(touching),max_bytes)
        p,q=vertices[a],vertices[b];push!(vertices,((p[1]+q[1])/2,(p[2]+q[2])/2));mid=length(vertices)
        for t in touching
            face=faces[t];j=only(j for j in 1:3 if (face[j]==a && face[mod1(j+1,3)]==b)||(face[j]==b && face[mod1(j+1,3)]==a))
            first,last,opposite=face[j],face[mod1(j+1,3)],face[mod1(j+2,3)]
            faces[t]=(first,mid,opposite);push!(faces,(mid,last,opposite));push!(levels,level)
        end
    end
    coords=Matrix{Float64}(undef,2,length(vertices));triangles=Matrix{Int}(undef,3,length(faces))
    for i in eachindex(vertices)
        coords[:,i].=vertices[i]
    end
    for t in eachindex(faces)
        triangles[:,t].=faces[t]
    end
    return PlanarConformalMesh(coords,triangles;interfaces=levels,max_bytes)
end

"""Triangulate a simple physical polygon and refine it with independently
specified edge and interior sizes. The polygon is `vertices[2,N]`; an optional
repeated closing vertex is accepted. Concave polygons are supported. No
rasterization, staircase replacement or Green quadrature is performed.
For polygons with holes, supply a constrained triangle mesh directly to
[`planar_refine_conformal`](@ref)."""
function planar_conformal_mesh(polygon::AbstractMatrix{<:Real};interface::Integer=1,
        edge_size::Real,interior_size::Real,edge_band::Real=2edge_size,
        max_triangles::Integer=100_000,max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    size(polygon,1)==2 && size(polygon,2)>=3 && interface>=0 && all(isfinite,polygon) ||
        throw(ArgumentError("conformal polygon must be a finite 2×N matrix"))
    n=size(polygon,2)
    polygon[:,1]==polygon[:,end] && (n-=1)
    n>=3 || throw(ArgumentError("conformal polygon needs three distinct vertices"))
    _planar_conformal_mesher_budget(n,n-2,max_bytes)
    vertices=Matrix{Float64}(polygon[:,1:n]);all(isfinite,vertices) || throw(ArgumentError("polygon must fit finite Float64 coordinates"))
    length(Set((vertices[1,i],vertices[2,i]) for i in 1:n))==n || throw(ArgumentError("polygon has repeated vertices"))
    for j in 1:n,k in j+1:n
        a,b=j,mod1(j+1,n);c,d=k,mod1(k+1,n)
        (a==c || a==d || b==c || b==d) && continue
        o1,o2=_planar_tri_orientation(vertices,a,b,c),_planar_tri_orientation(vertices,a,b,d)
        o3,o4=_planar_tri_orientation(vertices,c,d,a),_planar_tri_orientation(vertices,c,d,b)
        crossed=((o1>0 && o2<0)||(o1<0 && o2>0)) && ((o3>0 && o4<0)||(o3<0 && o4>0))
        crossed || _planar_on_segment(vertices,a,b,c) || _planar_on_segment(vertices,a,b,d) ||
            _planar_on_segment(vertices,c,d,a) || _planar_on_segment(vertices,c,d,b) ?
            throw(ArgumentError("conformal polygon self-intersects")) : nothing
    end
    area=sum(vertices[1,j]*vertices[2,mod1(j+1,n)]-vertices[2,j]*vertices[1,mod1(j+1,n)] for j in 1:n)
    isfinite(area) && area!=0 || throw(ArgumentError("conformal polygon has zero or nonfinite area"))
    ids=area>0 ? collect(1:n) : collect(n:-1:1);faces=NTuple{3,Int}[]
    while length(ids)>3
        ear=0
        for j in eachindex(ids)
            a,b,c=ids[mod1(j-1,length(ids))],ids[j],ids[mod1(j+1,length(ids))]
            _planar_tri_orientation(vertices,a,b,c)>0 || continue
            any(i!=a && i!=b && i!=c && _planar_tri_orientation(vertices,a,b,i)>=0 &&
                _planar_tri_orientation(vertices,b,c,i)>=0 && _planar_tri_orientation(vertices,c,a,i)>=0 for i in ids) && continue
            ear=j;push!(faces,(a,b,c));break
        end
        ear>0 || throw(ArgumentError("polygon could not be triangulated; check collinear constraints"))
        deleteat!(ids,ear)
    end
    push!(faces,Tuple(ids));triangles=hcat((collect(face) for face in faces)...)
    mesh=PlanarConformalMesh(vertices,triangles;interfaces=interface,max_bytes)
    return planar_refine_conformal(mesh;edge_size,interior_size,edge_band,max_triangles,max_bytes)
end
