using DiffMoM, LinearAlgebra, SHA, TOML, Krylov
BLAS.set_num_threads(1)

function fused_update!(out,delta,scale)
    for b in eachindex(out,delta)
        out[b]=complex(fma(scale,real(delta[b]),real(out[b])),
            fma(scale,imag(delta[b]),imag(out[b])))
    end
    out
end

function compensated_update!(out,delta,scale,carry)
    for b in eachindex(out,delta,carry)
        xr=real(out[b]);xi=imag(out[b]);dr=real(delta[b]);di=imag(delta[b])
        pr=scale*dr;pi=scale*di;tr=xr+pr;ti=xi+pi
        er=fma(scale,dr,-pr)+(abs(xr)>=abs(pr) ? (xr-tr)+pr : (pr-tr)+xr)
        ei=fma(scale,di,-pi)+(abs(xi)>=abs(pi) ? (xi-ti)+pi : (pi-ti)+xi)
        cr=real(carry[b])+er;ci=imag(carry[b])+ei;ur=tr+cr;ui=ti+ci
        carry[b]=complex((tr-ur)+cr,(ti-ui)+ci);out[b]=complex(ur,ui)
    end
    out
end

function main(root,output)
    root=realpath(root)
    files=["src/planar/PlanarConformalDefect.jl","src/planar/PlanarConformalProjection.jl",
        "src/planar/PlanarConformalMultiProjection.jl","test/test_planar_conformal_defect.jl",
        "test/test_planar_conformal_defect_refinement.jl"]
    hashes()=Dict(name=>bytes2hex(sha256(read(joinpath(root,name)))) for name in files)
    before=hashes();original=replace(read(joinpath(root,files[1]),String),"\r\n"=>"\n")
    start=first(findfirst("function solve_planar_conformal_defect(prob::PlanarConformalProblem,freq::Number;",original))
    stop=first(findfirst("planar_conformal_current_maps(result::PlanarConformalDefectResult",original))-1
    solver=original[start:stop]
    update="iterations[p]+=stats.niter;X[:,p].+=scale.*delta"
    @assert count(update,solver)==1
    fused=replace(solver,update=>"iterations[p]+=stats.niter;Main.fused_update!(view(X,:,p),delta,scale)")
    compensated=replace(solver,update=>"iterations[p]+=stats.niter;Main.compensated_update!(view(X,:,p),delta,scale,compensation)")
    marker="        # Arnoldi's recurrence can underestimate"
    @assert count(marker,compensated)==1
    compensated=replace(compensated,marker=>"        compensation=x;fill!(compensation,0)\n"*marker)
    a,b=1e-3,.5e-3
    stack=PlanarStackup([PlanarLayer(1.,1.,.5e-3),PlanarLayer(1.,1.,.5e-3)],TERM_GND,TERM_GND,a,b)
    mesh=PlanarConformalMesh([0. a a 0.;0. 0. b b],[1 1;2 3;3 4])
    prob=PlanarConformalProblem(stack,mesh,[PlanarConformalPort(1,:west,(0.,b)),PlanarConformalPort(1,:east,(0.,b))])
    cases=Dict{String,Any}[]
    for (method,patch) in (("unchanged_f9",nothing),("fused_update",fused),("compensated_update",compensated))
        patch===nothing || include_string(DiffMoM,patch,files[1]*"_"*method)
        for offset in -32:32
            frequency=offset<0 ? prevfloat(1e6,-offset) : nextfloat(1e6,offset)
            row=Dict{String,Any}("method"=>method,"ulp_offset"=>offset,"frequency"=>frequency,"status"=>"PASS")
            try
                r=Base.invokelatest(solve_planar_conformal_defect,prob,frequency;modes=32,nx=16,ny=8,surface_zs=2.)
                row["initial_projected_residuals"]=r.diagnostics.initial_projected_relative_residuals
                row["original_residuals"]=r.relative_residuals;row["iterations"]=r.iterations
                resistance=real(2/(r.y[1,1]-r.y[1,2]));row["resistance"]=resistance
                @assert maximum(row["initial_projected_residuals"])<=1e-10
                @assert maximum(row["original_residuals"])<=1e-10
                @assert isapprox(resistance,2a/b;rtol=2e-6)
                @assert all(i->i<=10000,r.iterations)
            catch err
                row["status"]="FAIL";row["error"]=sprint(showerror,err,catch_backtrace())
            end
            push!(cases,row)
            println("CAPTURE ",method," ",offset," ",row["status"],get(row,"error",""))
        end
    end
    after=hashes();@assert before==after;@assert !isfile(output)
    report=Dict("status"=>"CAPTURE_COMPLETE","version"=>string(VERSION),"arch"=>string(Sys.ARCH),
        "cpu"=>Sys.CPU_NAME,"cpu_target"=>unsafe_string(Base.JLOptions().cpu_target),
        "krylov_version"=>string(pkgversion(Krylov)),"head"=>readchomp(`git -C $root rev-parse HEAD`),
        "source_before"=>before,"source_after"=>after,"source_unchanged"=>true,
        "fused_patch_sha256"=>bytes2hex(sha256(fused)),"compensated_patch_sha256"=>bytes2hex(sha256(compensated)),
        "scope"=>"Diagnostic producer only: records failures, does not turn captured failures into passing regression tests. All four acceptance gates unchanged.","cases"=>cases)
    open(output,"w") do io;TOML.print(io,report);end
    println("BEGIN_CONFORMAL_ROUNDOFF_CAPTURE_TOML")
    TOML.print(stdout,report)
    println("END_CONFORMAL_ROUNDOFF_CAPTURE_TOML")
end
main(ARGS...)
