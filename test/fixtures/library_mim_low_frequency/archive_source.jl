using SHA,TOML
root=abspath(@__DIR__,"..","..")
target=joinpath(root,"test","fixtures","library_mim_low_frequency")
ispath(target) && error("immutable fixture already exists")
report=TOML.parsefile(joinpath(@__DIR__,"library_mim_low_frequency_reference_final.toml"))
report["source_unchanged"] && report["source_before"]==report["source_after"] ||
    error("invalid source guard")
all(r->r["status"]=="PASS",report["cases"]) || error("failed case")
paths=[joinpath(root,"src","planar",name) for name in
    ("PlanarLibrary.jl","PlanarLayout.jl","PlanarGreens.jl","PlanarSolve.jl",
     "PlanarUFFT.jl","PlanarFFTAssembly.jl")]
append!(paths,[joinpath(root,"validation","sonnet_stripline","library_mim_fixture.jl"),
    joinpath(@__DIR__,"library_mim_low_frequency_reference.jl")])
mkpath(joinpath(target,"source_snapshot"))
for path in paths
    bytes=read(path)
    bytes2hex(sha256(bytes))==report["source_after"][basename(path)] ||
        error("source changed before archive: "*path)
    write(joinpath(target,"source_snapshot",basename(path)),bytes)
end
for (source,destination) in (
        ("library_mim_low_frequency_reference.toml","initial.toml"),
        ("library_mim_low_frequency_reference.log","initial.log"),
        ("library_mim_low_frequency_reference_final.toml","final.toml"),
        ("library_mim_low_frequency_reference_final.log","final.log"))
    cp(joinpath(@__DIR__,source),joinpath(target,destination))
end
cp(@__FILE__,joinpath(target,"archive_source.jl"))
write(joinpath(target,"README.md"),"""
# Public MIM low-frequency overlap reference

Six current public solves cover 16/32/64 cells at 100/500 MHz. Literal plate
plus lead overlap is 7.03125e-8 m2; epsilon0*7.5*area/1e-6 gives
4.6692006066210936 pF independently of production capacitance estimators.
The unchanged approximation gate is 5%; original voltage equations use 1e-9.
Observed errors are 1.14-4.19%, residuals below 1.55e-12. Lead and fringing
effects are part of the solved response. This is a bounded finite-grid
electrostatic approximation check, not measured-device or continuum proof.

The initial successful run guards six named production files. The final
successful run additionally guards both fixture and validation helpers;
all eight before/after hashes match the retained source snapshots. These
guards do not certify every package source. The full immutable package
checkpoint is separate. No native engine is invoked by this analytic check;
the native MIM matrix archive is a distinct acceptance reference.

The registered test replays the complete current public solves, original
voltage residuals, reciprocity/passivity and immutable archive hashes.
""")
hashes=Dict{String,String}()
for (folder,_,files) in walkdir(target),file in files
    path=joinpath(folder,file)
    hashes[replace(relpath(path,target),'\\'=>'/')]=bytes2hex(sha256(read(path)))
end
open(joinpath(target,"sha256.toml"),"w") do io;TOML.print(io,Dict("sha256"=>hashes));end
println("Archived ",length(hashes)," files: ",target)
