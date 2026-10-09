module ExactEvaluationResourceTests
using DiffMoM,Test,TOML
root,output=length(ARGS)==2 ? (realpath(ARGS[1]),abspath(ARGS[2])) : (realpath(joinpath(@__DIR__,"..")),nothing)
@assert realpath(dirname(dirname(pathof(DiffMoM))))==root && (output===nothing || !isfile(output))
function model()
    M=floatmax(Float64)
    PlanarRationalModel(ComplexF64[-1.,-1.,-1.],
        [fill(ComplexF64(M),2,2),fill(ComplexF64(M),2,2),fill(ComplexF64(-M),2,2)],
        zeros(2,2),zeros(2,2),[0.,1.],0.,0.,false,false,:none,NaN,0.,0.)
end
m=model();f=0.;s=2pi*1im*f
den,bits,small,payload=DiffMoM._vf_exact_eval_integer_workspace(m,f,s)
result=@testset "Exact rational evaluation checks derived raw buffers before allocation" begin
    @test (@inferred Union{Nothing} DiffMoM._vf_exact_eval_integer(m,f,s;max_bytes=payload))==fill(ComplexF64(floatmax(Float64)),2,2)
    @test_throws ArgumentError DiffMoM._vf_exact_eval_integer(m,f,s;max_bytes=payload-1)
    @test_throws ArgumentError DiffMoM._vf_exact_eval_integer(m,f,s;max_bytes=0)
    @test_throws ArgumentError DiffMoM._vf_exact_eval_integer(m,f,s;max_bytes=-1)
    @test small<bits && den>=0
    scratch=DiffMoM._vf_ratio_scratch();x=BigInt(1);d=BigInt(3)
    DiffMoM._vf_float_ratio(x,d,scratch);saved=map(objectid,scratch.buffers);capacity=scratch.bits[]
    @test DiffMoM._vf_float_ratio(x,d,scratch)==1/3
    @test saved==map(objectid,scratch.buffers) && scratch.bits[]==capacity
    @test x==1 && d==3
end
c=Test.get_test_counts(result)
output===nothing || open(output,"w") do io;TOML.print(io,Dict("version"=>string(VERSION),"passes"=>c.passes+c.cumulative_passes,"fails"=>c.fails+c.cumulative_fails,"errors"=>c.errors+c.cumulative_errors,"payload"=>payload,"bits"=>bits,"small_bits"=>small),sorted=true);end

end
