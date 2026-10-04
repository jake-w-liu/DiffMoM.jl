using DiffMoM, LinearAlgebra
const audit_directory = normpath(joinpath(@__DIR__, "..", "..", "data", "planar_audit"))
const audit_output = isempty(ARGS) ? joinpath(audit_directory, "layout_modal_derivative.log") : abspath(only(ARGS))
mkpath(dirname(audit_output))

const DM = DiffMoM

function derivative_layout(stack=nothing)
    grid=CellGrid(4e-3,3e-3,8,6)
    stack===nothing && (stack=PlanarStackup([PlanarLayer(1.,1.,.2e-3),
        PlanarLayer(2.,1.,.4e-3),PlanarLayer(1.,1.,1e-3)],
        TERM_GND,TERM_GND,grid.a,grid.b))
    shape=planar_transform(planar_line(length=2e-3,width=1e-3,
        level=2,metal="resistive",net="wire");offset=(1e-3,1.5e-3))
    build_planar_layout(stack,grid,[shape],
        [planar_pin(shape,"p2"),planar_pin(shape,"p1")];metals=Dict("resistive"=>.1))
end

function modal_data(prob, stack, ::Type{T}) where T
    mg=planar_mode_grid(prob.grid,20,18); nb=planar_basis_count(prob.basis)
    iface=[DM._basis_elem(prob.basis,b,prob.sheets,prob.vias,prob.vols) for b in 1:nb]
    pairs=DM._level_pairs(iface);L=length(stack.layers);nmode=mg.mx*mg.my
    vlay=sort!(unique(DM._via_elem_layer(e) for e in iface if DM._is_via_elem(e)))
    volay=sort!(unique(DM._vol_elem_layer(e) for e in iface if DM._is_vol_elem(e)))
    cascade()=PlanarCascade(Vector{T}(undef,L+1),Vector{T}(undef,L+1),Vector{T}(undef,L),Vector{T}(undef,L))
    scratch=ntuple(_->Vector{T}(undef,L),3)
    vsts=Vector{DM._ViaLayerState{T}}(undef,isempty(vlay) ? 0 : L)
    volsts=(Vector{DM._VolLayerState{T}}(undef,isempty(volay) ? 0 : L),Vector{DM._VolLayerState{T}}(undef,isempty(volay) ? 0 : L))
    ml=[m for n in 1:mg.my for m in 1:mg.mx];nl=[n for n in 1:mg.my for m in 1:mg.mx]
    vte=Matrix{T}(undef,nmode,length(pairs));vtm=similar(vte)
    DM._planar_mode_voltages!(vte,vtm,cascade(),cascade(),scratch,stack,2pi*1e9,mg,ml,nl,pairs,vsts,vlay,volsts,volay)
    fxb=zeros(mg.mx,nb);fyb=zeros(mg.my,nb)
    for b in 1:nb
        DM._basis_fx!(view(fxb,:,b),prob.basis,b,mg,prob.grid)
        DM._basis_fy!(view(fyb,:,b),prob.basis,b,mg,prob.grid)
    end
    Wte=zeros(nmode,nb);Wtm=similar(Wte)
    DM._planar_weight_block!(Wte,Wtm,ml,nl,mg,fxb,fyb,prob.basis)
    return (;vte,vtm,Wte,Wtm,pairs,iface,ml,nl)
end

function modal_matrix(data, select, ::Type{T}) where T
    nb=length(data.iface);Z=zeros(T,nb,nb)
    for (i,(f,s)) in enumerate(data.pairs),p in findall(==(f),data.iface),q in findall(==(s),data.iface)
        for c in eachindex(data.ml)
            Z[p,q]-=select(data.vte[c,i])*T(data.Wte[c,p])*T(data.Wte[c,q])+
                select(data.vtm[c,i])*T(data.Wtm[c,p])*T(data.Wtm[c,q])
        end
    end
    Z
end

function precision_stack(stack,::Type{T};dual=false) where T
    layers=PlanarLayer[]
    for (i,l) in enumerate(stack.layers)
        value(x)=dual ? DM._PlanarDual{T}(T(x),zero(T)) : T(x)
        e=dual && i==1 ? DM._PlanarDual{T}(T(l.epsr),one(T)) : value(l.epsr)
        push!(layers,PlanarLayer(e,value(l.mur),value(l.thickness),value(l.epsr_z),value(l.mur_z)))
    end
    term(t)=dual ? PlanarTerminator(t.kind,DM._PlanarDual{T}(T(t.zs)),DM._PlanarDual{T}(T(t.epsr)),DM._PlanarDual{T}(T(t.mur))) : PlanarTerminator(t.kind,T(t.zs),T(t.epsr),T(t.mur))
    PlanarStackup(layers,term(stack.bottom),term(stack.top),stack.a,stack.b)
end

function main()
    layout=derivative_layout();prob=layout.source_problem;p=PlanarParam(1,:epsr,:re)
    data=modal_data(prob,prob.stack,ComplexF64)
    ddata=modal_data(prob,DM._planar_dual_stackup(prob.stack,p),DM._PlanarDual{ComplexF64})
    Z=assemble_planar_z(prob.stack,prob.grid,prob.sheets,prob.basis,2pi*1e9;vias=prob.vias,vols=prob.vols,mx=20,my=18,surface_zs=.1)
    dZ=modal_matrix(ddata,x->x.d,ComplexF64)
    nb=length(prob.basis.kind);nr=length(prob.ports);B=zeros(ComplexF64,nb,nr)
    for b in 1:nb
        port=prob.basis.port[b];port==0 && continue
        B[b,port]=-DM._planar_port_sign(prob.ports[port])*DM._planar_port_weight(prob.basis,b)
    end
    Bc=B*layout.contraction;X=Z\Bc;Y=-transpose(Bc)*X;G=2conj.(Y)
    println("Nb=",nb," dZ matrix adjoint=",real(sum(G.*(transpose(X)*dZ*X))))
    for h in (1e-2,1e-3,1e-4,1e-5,1e-6)
        sp=planar_with_params(prob.stack,[p],[1+h]);sm=planar_with_params(prob.stack,[p],[1-h])
        dp=modal_data(prob,sp,ComplexF64);dm=modal_data(prob,sm,ComplexF64)
        dv=(dp.vtm-dm.vtm)/(2h)
        errors=abs.(dv-[v.d for v in ddata.vtm]);mxerr,arg=findmax(errors)
        dZfd=(modal_matrix(dp,identity,ComplexF64)-modal_matrix(dm,identity,ComplexF64))/(2h)
        println("h=",h," max modal diff=",mxerr," at=",(data.ml[arg[1]],data.nl[arg[1]],data.pairs[arg[2]]),
            " modal relative norm=",norm(errors)/norm(dv)," dZ rel=",norm(dZfd-dZ)/norm(dZ),
            " FD dZ contraction=",real(sum(G.*(transpose(X)*dZfd*X))))
    end
    setprecision(192) do
        T=Complex{BigFloat};sp=precision_stack(prob.stack,T);sdp=precision_stack(prob.stack,T;dual=true)
        bd=modal_data(prob,sp,T);bdd=modal_data(prob,sdp,DM._PlanarDual{T})
        Zb=modal_matrix(bd,identity,T);dZb=modal_matrix(bdd,x->x.d,T)
        loss=zeros(ComplexF64,nb,nb);DM._add_gram!(loss,prob.basis,prob.grid,.1);Zb.+=T.(loss)
        Xb=Zb\T.(Bc);Yb=-transpose(T.(Bc))*Xb;Gb=2conj.(Yb)
        println("BigFloat dZ contraction=",real(sum(Gb.*(transpose(Xb)*dZb*Xb))),
            " objective=",sum(abs2,Yb)," float objective=",sum(abs2,Y),
            " dZ rel float=",norm(T.(dZ)-dZb)/norm(dZb)," Z rel float=",norm(T.(Z)-Zb)/norm(Zb))
        diff=abs.([v.d for v in bdd.vtm]-T.([v.d for v in ddata.vtm]));mxerr,arg=findmax(diff)
        println("big vs float max modal deriv diff=",mxerr," at=",(data.ml[arg[1]],data.nl[arg[1]],data.pairs[arg[2]]))
        for h in (big"0.01",big"0.001",big"0.0001")
            shifted(sign)=planar_with_params(sp,[p],[big"1"+sign*h])
            function value(sign)
                matrix=modal_matrix(modal_data(prob,shifted(sign),T),identity,T)
                matrix.+=T.(loss)
                response=-transpose(T.(Bc))*(matrix\T.(Bc))
                return sum(abs2,response)
            end
            fd=(value(1)-value(-1))/(2h)
            analytic=real(sum(Gb.*(transpose(Xb)*dZb*Xb)))
            println("BigFloat FD h=",h," derivative=",fd," relative error=",abs((fd-analytic)/analytic))
        end
    end
end
open(audit_output,"w") do io
    redirect_stdout(io) do
        main()
    end
end
println("Modal derivative report: ", audit_output)
