using DiffMoM
include(joinpath(pwd(),"validation/sonnet_stripline/sonnet_reference.jl"))
using .SonnetReference
out=open("data/hfss_probe/diffmom_sweep.csv","w")
println(out,"f_ghz,re_s11,im_s11,re_s21,im_s21,re_s22,im_s22,elapsed_s")
p=read_sonnet_project("data/hfss_probe/dxf_nodrp.son")
for f in (4.95e9,5.05e9)
    t0=time()
    prob=sonnet_planar_problem(p;freq=f,_details=true,_materials=true).problem
    r=solve_planar_ufft(prob,f;mx=2048,my=2048,memory=1200,maxiter=5000,rtol=1e-9,max_bytes=4_000_000_000)
    s=r.s
    println(out,join([f/1e9,real(s[1,1]),imag(s[1,1]),real(s[2,1]),imag(s[2,1]),real(s[2,2]),imag(s[2,2]),round(time()-t0,digits=1)],","))
    flush(out)
    println("DONE f=",f)
    flush(stdout)
end
close(out)
