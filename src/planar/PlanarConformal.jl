# Genuine conformal triangle currents: normal-continuous RWG functions,
# analytic affine Fourier moments and exact polynomial conductor Gram.

export PlanarConformalMesh, PlanarConformalPort, PlanarConformalProblem
export PlanarConformalResult, assemble_planar_conformal_z, solve_planar_conformal
export planar_conformal_current_maps

"""Physical triangle mesh with coordinates `vertices[2,Nv]`, vertex indices
`triangles[3,Nt]` and one stack interface per triangle. Triangles are genuine
physical subsections; their coordinates and sizes are preserved. Coincident
vertex coordinates are interned, orientations become counterclockwise, and
overlapping, nonmanifold or nonconforming triangles are rejected."""
struct PlanarConformalMesh
    vertices::Matrix{Float64}
    triangles::Matrix{Int}
    interfaces::Vector{Int}
    areas::Vector{Float64}
end

@inline function _planar_orient2d(ax,ay,bx,by,cx,cy)
    x1,y1,x2,y2=bx-ax,by-ay,cx-ax,cy-ay
    value=x1*y2-y1*x2
    bound=8eps(Float64)*(abs(x1*y2)+abs(y1*x2))
    abs(value)>bound && return value
    # Geometry predicates retain the exact sign of the input Float64
    # coordinates when an edge is collinear or nearly collinear.
    return Float64((BigFloat(bx)-BigFloat(ax))*(BigFloat(cy)-BigFloat(ay))-
        (BigFloat(by)-BigFloat(ay))*(BigFloat(cx)-BigFloat(ax)))
end
@inline _planar_tri_orientation(v,a,b,c)=_planar_orient2d(v[1,a],v[2,a],v[1,b],v[2,b],v[1,c],v[2,c])
@inline function _planar_on_segment(v,a,b,c)
    _planar_tri_orientation(v,a,b,c)==0 || return false
    return min(v[1,a],v[1,b])<=v[1,c]<=max(v[1,a],v[1,b]) &&
        min(v[2,a],v[2,b])<=v[2,c]<=max(v[2,a],v[2,b])
end
function _planar_check_triangle_pair(v,t,u)
    for d in 1:2
        tlo,thi=min(v[d,t[1]],v[d,t[2]],v[d,t[3]]),max(v[d,t[1]],v[d,t[2]],v[d,t[3]])
        ulo,uhi=min(v[d,u[1]],v[d,u[2]],v[d,u[3]]),max(v[d,u[1]],v[d,u[2]],v[d,u[3]])
        (thi<ulo || uhi<tlo) && return
    end
    for c in t
        c in u && continue
        all(_planar_tri_orientation(v,u[j],u[mod1(j+1,3)],c)>0 for j in 1:3) &&
            throw(ArgumentError("conformal triangles overlap"))
        any(_planar_on_segment(v,u[j],u[mod1(j+1,3)],c) for j in 1:3) &&
            throw(ArgumentError("triangle mesh has a hanging vertex; split the adjacent edge"))
    end
    for c in u
        c in t && continue
        all(_planar_tri_orientation(v,t[j],t[mod1(j+1,3)],c)>0 for j in 1:3) &&
            throw(ArgumentError("conformal triangles overlap"))
        any(_planar_on_segment(v,t[j],t[mod1(j+1,3)],c) for j in 1:3) &&
            throw(ArgumentError("triangle mesh has a hanging vertex; split the adjacent edge"))
    end
    for j in 1:3,k in 1:3
        a,b=t[j],t[mod1(j+1,3)];c,d=u[k],u[mod1(k+1,3)]
        o1,o2=_planar_tri_orientation(v,a,b,c),_planar_tri_orientation(v,a,b,d)
        o3,o4=_planar_tri_orientation(v,c,d,a),_planar_tri_orientation(v,c,d,b)
        ((o1>0 && o2<0)||(o1<0 && o2>0)) &&
            ((o3>0 && o4<0)||(o3<0 && o4>0)) &&
            throw(ArgumentError("conformal triangle edges intersect"))
    end
end

function PlanarConformalMesh(vertices::AbstractMatrix{<:Real},triangles::AbstractMatrix{<:Integer};
        interfaces=1,max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    size(vertices,1)==2 && size(triangles,1)==3 && size(vertices,2)>=3 && size(triangles,2)>=1 ||
        throw(ArgumentError("vertices must be 2×Nv and triangles 3×Nt"))
    nv,nt=size(vertices,2),size(triangles,2)
    est=_checked_payload_sum("conformal mesh",_checked_array_payload_bytes(Float64,2,nv),
        _checked_array_payload_bytes(Int,7,nt),_checked_array_payload_bytes(Float64,nt),
        _checked_array_payload_bytes(Int,2,nv))
    _enforce_payload_limit(est,max_bytes,"conformal mesh","max_bytes")
    all(isfinite,vertices) && all(i->1<=i<=nv,triangles) ||
        throw(ArgumentError("triangle coordinates must be finite and vertex indices valid"))
    levels=interfaces isa Integer ? fill(Int(interfaces),nt) :
        interfaces isa AbstractVector{<:Integer} && length(interfaces)==nt ? Int.(interfaces) :
        throw(ArgumentError("interfaces must be an integer or one integer per triangle"))
    all(>=(0),levels) || throw(ArgumentError("triangle interfaces must be nonnegative"))
    v=Matrix{Float64}(vertices)
    all(isfinite,v) || throw(ArgumentError("triangle coordinates must fit finite Float64 values"))
    canonical=Dict{Tuple{Float64,Float64},Int}();mapids=Vector{Int}(undef,nv)
    for i in 1:nv
        mapids[i]=get!(canonical,(v[1,i],v[2,i]),i)
    end
    tris=map(i->mapids[i],triangles);areas=Vector{Float64}(undef,nt)
    seen=Set{NTuple{4,Int}}();edges=Dict{NTuple{3,Int},Int}()
    for t in 1:nt
        a,b,c=tris[:,t];signed=_planar_tri_orientation(v,a,b,c)
        isfinite(signed) && signed!=0 || throw(ArgumentError("triangle $t has zero or nonfinite area"))
        signed<0 && ((tris[2,t],tris[3,t])=(tris[3,t],tris[2,t]))
        areas[t]=abs(signed)/2
        ids=sort([a,b,c]);key=(levels[t],ids...)
        key in seen && throw(ArgumentError("duplicate conformal triangle"));push!(seen,key)
        for j in 1:3
            p,q=tris[j,t],tris[mod1(j+1,3),t];edge=(levels[t],min(p,q),max(p,q))
            edges[edge]=get(edges,edge,0)+1
            edges[edge]<=2 || throw(ArgumentError("nonmanifold triangle edge"))
        end
    end
    for t in 1:nt,u in t+1:nt
        levels[t]==levels[u] && _planar_check_triangle_pair(v,view(tris,:,t),view(tris,:,u))
    end
    return PlanarConformalMesh(v,Matrix{Int}(tris),levels,areas)
end

"""Wall or shared interior-edge port on a genuine triangle mesh. For box
walls, `span=(lo,hi)` is the transverse coordinate interval and current
points into metal. `PlanarConformalPort(interface,:internal,((ax,ay),(bx,by)))`
selects a continuous shared-edge path; positive current flows toward the
left of its oriented endpoints. Every internal edge requires metal on both
sides. `polarity=-1` reverses both excitation and current extraction.
`z0` accepts a positive-real-part complex reference or frequency provider;
references do not alter the physical current basis."""
struct PlanarConformalPort
    interface::Int
    wall::Symbol
    span::Tuple{Float64,Float64}
    z0::Any
    endpoints::Union{Nothing,NTuple{4,Float64}}
    polarity::Int
    function PlanarConformalPort(interface::Integer,wall::Symbol,span::Tuple{<:Real,<:Real};z0=50.,polarity::Integer=1)
        interface>=0 && wall in (:west,:east,:south,:north) ||
            throw(ArgumentError("conformal port requires a nonnegative interface and box wall"))
        lo,hi=Float64.(span);ref=_planar_store_reference(z0)
        isfinite(lo) && isfinite(hi) && lo<hi && polarity in (-1,1) ||
            throw(ArgumentError("port span must increase and polarity must be ±1"))
        new(Int(interface),wall,(lo,hi),ref,nothing,Int(polarity))
    end
    function PlanarConformalPort(interface::Integer,kind::Symbol,
            endpoints::Tuple{Tuple{<:Real,<:Real},Tuple{<:Real,<:Real}};z0=50.,polarity::Integer=1)
        interface>=0 && kind===:internal && polarity in (-1,1) || throw(ArgumentError("shared-edge ports require :internal and polarity ±1"))
        a,b=Float64.(endpoints[1]),Float64.(endpoints[2]);ref=_planar_store_reference(z0);width=hypot(b[1]-a[1],b[2]-a[2])
        all(isfinite,(a...,b...,width)) && width>0 || throw(ArgumentError("internal port endpoints must differ and be finite"))
        new(Int(interface),kind,(0.,width),ref,(a...,b...),Int(polarity))
    end
end

struct _PlanarConformalBasis
    edges::Matrix{Int}
    triangles::Matrix{Int}
    interfaces::Vector{Int}
    width::Vector{Float64}
    port::Vector{Int}
    port_sign::Vector{Float64}
end

"""Shielded multilayer sheet problem on genuine conformal triangles.
Construct with `(stack,mesh,ports;sidewalls=WALL_PEC,max_bytes=...)`.
`wall_contacts` declares undriven PEC box-wall spans; source spans override
their zero-voltage contact. Open sheet boundaries otherwise carry no normal
current. Shared triangle edges carry normal-continuous affine currents. Wall sources
claim boundary half functions; internal sources claim complete shared-edge
functions, with power-conjugate voltage/current orientation."""
struct PlanarConformalProblem{T<:Number}
    stack::PlanarStackup{T}
    mesh::PlanarConformalMesh
    ports::Vector{PlanarConformalPort}
    sidewalls::SidewallKind
    basis::_PlanarConformalBasis
    wall_contacts::Vector{PlanarConformalPort}
end

function _planar_conformal_edge_port(stack,mesh,edge,ports)
    level,a,b=edge;v=mesh.vertices;found=0
    for (p,port) in enumerate(ports)
        port.interface==level || continue
        if port.wall===:internal
            ax,ay,bx,by=port.endpoints;ux,uy=bx-ax,by-ay;length=port.span[2]
            tol=64eps(Float64)*max(stack.a,stack.b)
            abs(_planar_orient2d(ax,ay,bx,by,v[1,a],v[2,a]))<=tol*length &&
                abs(_planar_orient2d(ax,ay,bx,by,v[1,b],v[2,b]))<=tol*length || continue
            ca=((v[1,a]-ax)*ux+(v[2,a]-ay)*uy)/length
            cb=((v[1,b]-ax)*ux+(v[2,b]-ay)*uy)/length
            lo,hi=minmax(ca,cb)
            min(hi,length)>max(lo,0.)+tol || continue
            lo>=-tol && hi<=length+tol || throw(ArgumentError("internal port cuts a conformal edge"))
            found==0 || throw(ArgumentError("conformal ports overlap"));found=p
            continue
        end
        wall=port.wall;dim=wall in (:west,:east) ? 1 : 2;tdim=3-dim
        coordinate=wall in (:west,:south) ? 0. : wall===:east ? stack.a : stack.b
        tol=64eps(Float64)*max(stack.a,stack.b)
        abs(v[dim,a]-coordinate)<=tol && abs(v[dim,b]-coordinate)<=tol || continue
        lo,hi=minmax(v[tdim,a],v[tdim,b]);pl,ph=port.span
        min(hi,ph)>max(lo,pl)+tol || continue
        lo>=pl-tol && hi<=ph+tol || throw(ArgumentError("port span cuts a conformal boundary edge"))
        found==0 || throw(ArgumentError("conformal wall ports overlap"));found=p
    end
    found
end

function PlanarConformalProblem(stack::PlanarStackup,mesh::PlanarConformalMesh,
        ports::AbstractVector{PlanarConformalPort};sidewalls::SidewallKind=WALL_PEC,
        wall_contacts::AbstractVector{PlanarConformalPort}=PlanarConformalPort[],
        max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    planar_validate(stack);L=length(stack.layers);nt=length(mesh.interfaces)
    all(i->0<=i<=L,mesh.interfaces) &&
        all(i->0<=i<=L,(p.interface for p in ports)) || throw(ArgumentError("conformal interface is outside the stack"))
    all(p->p.wall!==:internal && 0<=p.interface<=L,wall_contacts) || throw(ArgumentError("conformal ground contacts require box-wall spans"))
    isempty(wall_contacts) || sidewalls===WALL_PEC || throw(ArgumentError("galvanic box-wall contacts require PEC sidewalls"))
    all(x->0<=x<=stack.a,view(mesh.vertices,1,:)) && all(y->0<=y<=stack.b,view(mesh.vertices,2,:)) ||
        throw(ArgumentError("conformal triangles must lie inside the box"))
    _enforce_payload_limit(_checked_payload_sum("conformal basis",
        _checked_array_payload_bytes(Int,21,nt),_checked_array_payload_bytes(Float64,6,nt)),
        max_bytes,"conformal basis","max_bytes")
    edges=Dict{NTuple{3,Int},Vector{Int}}()
    for t in 1:nt,j in 1:3
        a,b=mesh.triangles[j,t],mesh.triangles[mod1(j+1,3),t]
        push!(get!(edges,(mesh.interfaces[t],min(a,b),max(a,b)),Int[]),t)
    end
    be=Tuple{Int,Int}[];bt=Tuple{Int,Int}[];levels=Int[];widths=Float64[];assignment=Int[];senses=Float64[]
    for edge in sort!(collect(keys(edges)))
        faces=edges[edge];p=_planar_conformal_edge_port(stack,mesh,edge,ports)
        p!=0 && ports[p].wall===:internal && length(faces)!=2 && throw(ArgumentError("internal conformal port requires metal on both sides of every edge"))
        ground=length(faces)==1 && p==0 && any(contact->_planar_conformal_edge_port(stack,mesh,edge,(contact,))!=0,wall_contacts)
        length(faces)==1 && p==0 && !ground && continue
        level,a,b=edge;w=hypot(mesh.vertices[1,a]-mesh.vertices[1,b],mesh.vertices[2,a]-mesh.vertices[2,b])
        isfinite(w) && w>0 || throw(ArgumentError("nonfinite conformal edge length"))
        push!(be,(a,b));push!(bt,(faces[1],length(faces)==2 ? faces[2] : 0));push!(levels,level)
        push!(widths,w);push!(assignment,p)
        sense=1.
        if p!=0
            port=ports[p];sense=Float64(port.polarity)
            if port.wall===:internal
                ax,ay,bx,by=port.endpoints
                free=only(i for i in mesh.triangles[:,faces[1]] if i!=a && i!=b)
                sense*=_planar_orient2d(ax,ay,bx,by,mesh.vertices[1,free],mesh.vertices[2,free])>0 ? -1. : 1.
            end
        end
        push!(senses,sense)
    end
    for (p,port) in enumerate(ports)
        selected=findall(==(p),assignment)
        isempty(selected) && throw(ArgumentError("conformal port $p has no boundary edges"))
        abs(sum(widths[selected])-(port.span[2]-port.span[1]))<=
            64eps(Float64)*length(selected)*max(stack.a,stack.b) ||
            throw(ArgumentError("conformal port $p span is not completely covered by conductor"))
    end
    nb=length(be);em=Matrix{Int}(undef,2,nb);tm=similar(em)
    for b in 1:nb
        em[:,b].=be[b];tm[:,b].=bt[b]
    end
    PlanarConformalProblem(stack,mesh,collect(ports),sidewalls,_PlanarConformalBasis(em,tm,levels,widths,assignment,senses),collect(wall_contacts))
end

@inline function _planar_conformal_half_values(prob,b,half)
    t=prob.basis.triangles[half,b];ids=view(prob.mesh.triangles,:,t)
    a,c=prob.basis.edges[:,b];free=only(i for i in ids if i!=a && i!=c)
    sign=half==2 || prob.basis.triangles[2,b]==0 ? -1. : 1.
    factor=sign*prob.basis.width[b]/(2prob.mesh.areas[t]);v=prob.mesh.vertices
    x=ntuple(j->factor*(v[1,ids[j]]-v[1,free]),Val(3))
    y=ntuple(j->factor*(v[2,ids[j]]-v[2,free]),Val(3))
    return t,x,y
end

"""Exact vector plane-wave integral of a physical conformal basis function:
`(∫fx exp(+i k⋅r)dA, ∫fy exp(+i k⋅r)dA)`. The wave numbers are real."""
function _planar_conformal_fourier(prob::PlanarConformalProblem,b::Integer,kx::Real,ky::Real)
    1<=b<=length(prob.basis.width) && isfinite(kx) && isfinite(ky) ||
        throw(ArgumentError("invalid conformal basis or wave number"))
    fx,fy=0.0im,0.0im
    for half in 1:2
        prob.basis.triangles[half,b]==0 && continue
        t,x,y=_planar_conformal_half_values(prob,b,half)
        vertices=view(prob.mesh.vertices,:,view(prob.mesh.triangles,:,t))
        fx+=_planar_triangle_affine_fourier(vertices,x,kx,ky)
        fy+=_planar_triangle_affine_fourier(vertices,y,kx,ky)
    end
    return fx,fy
end

function _planar_conformal_weights!(te,tm,prob,kx,ky)
    nb=length(prob.basis.width)
    for b in 1:nb
        px,py=0.,0.
        for half in 1:2
            prob.basis.triangles[half,b]==0 && continue
            t,x,y=_planar_conformal_half_values(prob,b,half)
            vertices=view(prob.mesh.vertices,:,view(prob.mesh.triangles,:,t))
            xp=_planar_triangle_affine_fourier(vertices,x,kx,ky)
            xm=_planar_triangle_affine_fourier(vertices,x,kx,-ky)
            yp=_planar_triangle_affine_fourier(vertices,y,kx,ky)
            ym=_planar_triangle_affine_fourier(vertices,y,kx,-ky)
            if prob.sidewalls===WALL_PEC
                px+=(imag(xp)-imag(xm))/2;py+=(imag(yp)+imag(ym))/2
            else
                px+=(imag(xp)+imag(xm))/2;py+=(imag(yp)-imag(ym))/2
            end
        end
        te[b]=ky*px-kx*py;tm[b]=kx*px+ky*py
    end
    return te,tm
end
function _planar_conformal_weights(prob,kx,ky)
    nb=length(prob.basis.width)
    _planar_conformal_weights!(Vector{Float64}(undef,nb),Vector{Float64}(undef,nb),prob,kx,ky)
end

@inline _planar_affine_triangle_gram(a,b,area)=area/12*(sum(a)*sum(b)+sum(a[j]*b[j] for j in 1:3))
function _planar_conformal_loss_entries(emit,prob,surface_zs)
    nt=length(prob.mesh.interfaces)
    surface_zs isa Number || surface_zs isa AbstractVector && length(surface_zs)==nt ||
        throw(ArgumentError("conformal surface_zs must be scalar or one value per triangle"))
    all(z->z isa Number && isfinite(ComplexF64(z)),surface_zs isa Number ? (surface_zs,) : surface_zs) ||
        throw(ArgumentError("conformal surface impedances must be finite numbers"))
    parts=[Tuple{Int,NTuple{3,Float64},NTuple{3,Float64}}[] for _ in 1:nt]
    for b in eachindex(prob.basis.width),half in 1:2
        prob.basis.triangles[half,b]==0 && continue
        t,x,y=_planar_conformal_half_values(prob,b,half);push!(parts[t],(b,x,y))
    end
    for t in 1:nt
        zs=ComplexF64(surface_zs isa Number ? surface_zs : surface_zs[t]);iszero(zs) && continue
        for (p,x,y) in parts[t],(q,u,v) in parts[t]
            emit(p,q,-zs*(_planar_affine_triangle_gram(x,u,prob.mesh.areas[t])+
                _planar_affine_triangle_gram(y,v,prob.mesh.areas[t])))
        end
    end
end

"""Exact finite TE/TM Galerkin modal sum for genuine triangular currents.
`surface_zs` is a scalar or one scalar per physical triangle. No Green or
surface quadrature is used. Arbitrary triangle sizes are supported."""
function assemble_planar_conformal_z(prob::PlanarConformalProblem,freq::Number;
        mx::Integer=64,my::Integer=64,surface_zs=0.,
        max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    omega=2pi*ComplexF64(freq);nb=length(prob.basis.width);L=length(prob.stack.layers)
    isfinite(omega) && real(omega)>0 && mx>=1 && my>=1 || throw(ArgumentError("frequency and mode counts must be positive and finite"))
    _checked_array_payload_bytes(UInt8,mx,my;label="conformal mode count")
    est=_checked_payload_sum("conformal assembly",_checked_array_payload_bytes(ComplexF64,nb,nb),
        _checked_array_payload_bytes(Float64,7,mx),_checked_array_payload_bytes(Float64,7,my),
        _checked_array_payload_bytes(Float64,2,nb),_checked_array_payload_bytes(ComplexF64,15,L+1),
        _checked_array_payload_bytes(Float64,24,length(prob.mesh.interfaces)),
        _checked_array_payload_bytes(ComplexF64,nb),_checked_array_payload_bytes(Int,3,nb))
    _enforce_payload_limit(est,max_bytes,"conformal assembly","max_bytes")
    Z=zeros(ComplexF64,nb,nb)
    _planar_conformal_loss_entries((p,q,value)->(Z[p,q]+=value),prob,surface_zs)
    mg=planar_mode_grid(CellGrid(prob.stack.a,prob.stack.b,1,1;walls=prob.sidewalls),mx,my)
    Wte=Vector{Float64}(undef,nb);Wtm=similar(Wte);Wcomplex=Vector{ComplexF64}(undef,nb)
    groups=[(level,findfirst(==(level),prob.basis.interfaces):findlast(==(level),prob.basis.interfaces)) for level in unique(prob.basis.interfaces)]
    cte,ctm,scratch,_,_=_planar_mode_workspace(L,false,false)
    for n in 1:mg.my,m in 1:mg.mx
        kx,ky=mg.kx[m],mg.ky[n];kc2=kx*kx+ky*ky
        _planar_conformal_weights!(Wte,Wtm,prob,kx,ky)
        pec=prob.sidewalls===WALL_PEC
        nte2=pec ? ky^2*mg.ic[m]*mg.js[n]+kx^2*mg.is[m]*mg.jc[n] : ky^2*mg.is[m]*mg.jc[n]+kx^2*mg.ic[m]*mg.js[n]
        ntm2=pec ? kx^2*mg.ic[m]*mg.js[n]+ky^2*mg.is[m]*mg.jc[n] : kx^2*mg.is[m]*mg.jc[n]+ky^2*mg.ic[m]*mg.js[n]
        for (pol,W,norm2) in ((TE_POL,Wte,nte2),(TM_POL,Wtm,ntm2))
            norm2==0 && continue
            cascade=pol===TE_POL ? cte : ctm
            planar_mode_cascade!(cascade,prob.stack,omega,kc2,pol,scratch)
            Wcomplex.=W
            for (f,rows) in groups,(s,cols) in groups
                voltage=-planar_modal_voltage(cascade,f,s)/norm2
                LinearAlgebra.BLAS.geru!(voltage,view(Wcomplex,rows),view(Wcomplex,cols),view(Z,rows,cols))
            end
        end
    end
    return Z
end

"""Conformal solve result; the retained LU factors the two-sided edge-length
equilibrated matrix. `basis_scale` maps its unknowns to physical coefficients."""
struct PlanarConformalResult{F}
    problem::PlanarConformalProblem
    freq::ComplexF64
    omega::ComplexF64
    z_mom::Union{Nothing,Matrix{ComplexF64}}
    lu_fact::F
    basis_scale::Vector{Float64}
    currents::Matrix{ComplexF64}
    y::Matrix{ComplexF64}
    s::Matrix{ComplexF64}
    relative_residuals::Vector{Float64}
    z0::Vector{ComplexF64}
end
PlanarConformalResult(prob::PlanarConformalProblem,freq,omega,Z,F,scale,X,Y,S,residuals)=
    PlanarConformalResult(prob,freq,omega,Z,F,scale,X,Y,S,residuals,
        _planar_reference_values([p.z0 for p in prob.ports],length(prob.ports);freq=real(freq)))

"""Solve exact triangular Galerkin wall excitations and extract Y/S.
The coefficients have the RWG convention A/m; their normal edge current is
coefficient×edge length. Every port voltage drives all its boundary edges."""
function solve_planar_conformal(prob::PlanarConformalProblem,freq::Number;
        method::Symbol=:dense,retain_matrix::Bool=true,max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES,kw...)
    method in (:dense,:ufft) || throw(ArgumentError("conformal method must be :dense or :ufft"))
    method===:ufft && return solve_planar_conformal_ufft(prob,freq;max_bytes,kw...)
    omega=2pi*ComplexF64(freq)
    isfinite(omega) && real(omega)>0 || throw(ArgumentError("conformal solve frequency must be finite with positive real part"))
    nb=length(prob.basis.width);np=length(prob.ports)
    nb>0 && np>0 || throw(ArgumentError("conformal solve needs basis functions and ports"))
    payload=_checked_payload_sum("conformal solve",_checked_array_payload_bytes(ComplexF64,nb,np),
        _checked_array_payload_bytes(ComplexF64,2,nb),_checked_array_payload_bytes(Float64,nb),
        _checked_array_payload_bytes(ComplexF64,5,np,np),_checked_array_payload_bytes(Float64,np),
        _checked_array_payload_bytes(ComplexF64,2,np))
    est=_checked_dense_lu_work_bytes(ComplexF64,nb,retain_matrix ? 2 : 1,payload;label="conformal solve")
    _enforce_payload_limit(est,max_bytes,"conformal solve","max_bytes")
    refs=_planar_reference_values([p.z0 for p in prob.ports],np;freq=real(freq))
    reserve=_checked_payload_sum("conformal assembly reservation",payload,
        _checked_array_payload_bytes(ComplexF64,nb,nb,retain_matrix ? 1 : 0))
    Z=assemble_planar_conformal_z(prob,freq;max_bytes=max_bytes-reserve,kw...)
    all(isfinite,Z) || throw(ArgumentError("conformal matrix is nonfinite; check box resonances"))
    scale=inv.(prob.basis.width);scaled=retain_matrix ? copy(Z) : Z
    for q in 1:nb,p in 1:nb
        scaled[p,q]*=scale[p]*scale[q]
    end
    F=lu!(scaled);X=zeros(ComplexF64,nb,np)
    for b in 1:nb
        p=prob.basis.port[b];p==0 || (X[b,p]=-prob.basis.port_sign[b]*prob.basis.width[b]*scale[b])
    end
    ldiv!(F,X);X .*= scale
    Y=zeros(ComplexF64,np,np)
    for b in 1:nb
        p=prob.basis.port[b];p==0 && continue
        for q in 1:np
            Y[p,q]+=prob.basis.port_sign[b]*prob.basis.width[b]*X[b,q]
        end
    end
    residuals=Float64[]
    if retain_matrix
        rhs=zeros(ComplexF64,nb);residual=similar(rhs)
        for p in 1:np
            for b in 1:nb
                rhs[b]=prob.basis.port[b]==p ? -prob.basis.port_sign[b]*prob.basis.width[b] : 0.0im
            end
            mul!(residual,Z,view(X,:,p))
            for b in 1:nb
                residual[b]=(residual[b]-rhs[b])*scale[b];rhs[b]*=scale[b]
            end
            push!(residuals,norm(residual)/norm(rhs))
        end
    end
    PlanarConformalResult(prob,ComplexF64(freq),2pi*ComplexF64(freq),retain_matrix ? Z : nothing,F,scale,X,Y,planar_y_to_s(Y,refs),residuals,refs)
end

"""Piecewise affine physical currents on each genuine triangle. Each record
contains its three physical vertices, current values there, exact integrated
current, and interface. Excitation is supplied as terminal voltages or waves."""
function _planar_conformal_map_payload(prob)
    nt=length(prob.mesh.interfaces)
    _checked_payload_sum("conformal currents",_checked_array_payload_bytes(ComplexF64,14,nt),
        _checked_array_payload_bytes(Float64,6,nt))
end
function _planar_conformal_maps_from_coefficients(prob,coefficients;max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    length(coefficients)==length(prob.basis.width) && all(isfinite,coefficients) ||
        throw(ArgumentError("conformal coefficients must be finite and match the physical basis"))
    _enforce_payload_limit(_planar_conformal_map_payload(prob),max_bytes,"conformal currents","max_bytes")
    nt=length(prob.mesh.interfaces);currents=zeros(ComplexF64,2,3,nt)
    for b in eachindex(coefficients),half in 1:2
        prob.basis.triangles[half,b]==0 && continue
        t,x,y=_planar_conformal_half_values(prob,b,half)
        for j in 1:3
            currents[1,j,t]+=coefficients[b]*x[j];currents[2,j,t]+=coefficients[b]*y[j]
        end
    end
    [(interface=prob.mesh.interfaces[t],vertices=prob.mesh.vertices[:,prob.mesh.triangles[:,t]],
        current=currents[:,:,t],integrated_current=ComplexF64[
            prob.mesh.areas[t]/3*(currents[1,1,t]+currents[1,2,t]+currents[1,3,t]),
            prob.mesh.areas[t]/3*(currents[2,1,t]+currents[2,2,t]+currents[2,3,t])]) for t in 1:nt]
end
function _planar_conformal_current_maps(result;port::Integer=1,voltages=nothing,incident_waves=nothing,
        max_bytes::Integer=_DEFAULT_MAX_DENSE_PAYLOAD_BYTES)
    n=length(result.problem.ports)
    reserve=_checked_payload_sum("conformal current excitation",
        _checked_array_payload_bytes(ComplexF64,length(result.problem.basis.width)),
        _checked_array_payload_bytes(ComplexF64,10,n))
    _enforce_payload_limit(_checked_payload_sum("conformal current reconstruction",reserve,
        _planar_conformal_map_payload(result.problem)),max_bytes,"conformal currents","max_bytes")
    voltages!==nothing && incident_waves!==nothing && throw(ArgumentError("provide voltages or incident_waves"))
    supplied=voltages===nothing ? incident_waves : voltages
    v=if supplied===nothing
        1<=port<=n || throw(ArgumentError("invalid conformal port"))
        x=zeros(ComplexF64,n);x[port]=1;x
    else
        supplied isa AbstractVector && length(supplied)==n && all(isfinite,supplied) || throw(ArgumentError("conformal excitation must be finite and match ports"))
        ComplexF64.(supplied)
    end
    incident_waves===nothing || (v=_planar_wave_voltage(result.s,v,result.z0))
    _planar_conformal_maps_from_coefficients(result.problem,result.currents*v;max_bytes=max_bytes-reserve)
end
"""Reconstruct exact piecewise affine physical triangle currents from a
dense or FFT conformal result. Supply terminal `voltages` or
`incident_waves`, or choose the unit-voltage `port`. Each triangle record
includes physical vertices, current at those vertices, stack interface and
its exact integrated vector current."""
planar_conformal_current_maps(result::PlanarConformalResult;kw...)=_planar_conformal_current_maps(result;kw...)
planar_current_maps(result::PlanarConformalResult;kw...)=_planar_conformal_current_maps(result;kw...)
