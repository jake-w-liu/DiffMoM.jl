# An independent electrostatic overlap reference for the public MIM coupon.
using DiffMoM,LinearAlgebra,SHA,TOML
BLAS.set_num_threads(1)
include(joinpath(@__DIR__,"../sonnet_stripline/library_mim_fixture.jl"))
paths=[joinpath(@__DIR__,"../../src/planar/"*name) for name in
    ("PlanarLibrary.jl","PlanarLayout.jl","PlanarGreens.jl","PlanarSolve.jl",
     "PlanarUFFT.jl","PlanarFFTAssembly.jl")]
append!(paths,[joinpath(@__DIR__,"../sonnet_stripline/library_mim_fixture.jl"),@__FILE__])
label=isempty(ARGS) ? "library_mim_low_frequency_reference_final" : only(ARGS)
occursin(r"^[a-z0-9_]+$",label) || error("invalid report label")
hashes()=Dict(basename(p)=>bytes2hex(sha256(read(p))) for p in paths)
# Literal plate/lead overlap: .25*.25 + .0625*.125 mm^2. The second
# term belongs to the upper lead over the extended lower plate.
area=(.25*.25+.0625*.125)*1e-6
reference=8.854187817e-12*7.5*area/1e-6
rows=Dict{String,Any}[]
report=Dict{String,Any}("scope"=>"low-frequency mutual capacitance of the complete public wall-fed MIM coupon versus independent thin-gap overlap formula; finite grid/modes, no measured-device or general continuum acceptance",
    "literal_overlap_area_m2"=>area,"overlap_capacitance_reference_f"=>reference,
    "relative_capacitance_gate"=>.05,"original_voltage_residual_gate"=>1e-9,
    "source_before"=>hashes(),"cases"=>rows)
for cells in (16,32,64),frequency in (1e8,5e8)
    layout=mim_library_layout(cells)
    result=solve_planar(layout,frequency;method=:dense_fft,mx=4cells,my=4cells,
        retain_matrix=true,max_bytes=2_000_000_000)
    capacitance=-imag(result.y[1,2])/(2pi*frequency)
    residual=mim_original_residual(layout.problem,result)
    error=abs(capacitance-reference)/reference
    row=Dict{String,Any}("cells"=>cells,"modes"=>[4cells,4cells],
        "frequency_hz"=>frequency,"mutual_capacitance_f"=>capacitance,
        "relative_capacitance_error"=>error,"original_voltage_residual"=>residual,
        "status"=>error<=.05 && residual<=1e-9 ? "PASS" : "FAIL")
    push!(rows,row);println(row)
end
report["source_after"]=hashes()
report["source_unchanged"]=report["source_before"]==report["source_after"]
open(joinpath(@__DIR__,label*".toml"),"w") do io
    TOML.print(io,report)
end
println(report)
all(r->r["status"]=="PASS",rows) && report["source_unchanged"] || exit(1)
