module DarwinAvailableMemoryTests
using DiffMoM,Test
# ABI fields and flavor are fixed by Apple's public Mach headers.
# https://github.com/apple-oss-distributions/xnu/blob/main/osfmk/mach/vm_statistics.h
# https://github.com/apple-oss-distributions/xnu/blob/main/osfmk/mach/host_info.h
function run_tests()
 S=DiffMoM._PlanarDarwinVMStatistics;names=fieldnames(S)
 @testset "Darwin reclaimable page arithmetic and original Mach ABI" begin
  @test names==(:free_count,:active_count,:inactive_count,:wire_count,:zero_fill_count,:reactivations,:pageins,:pageouts,:faults,:cow_faults,:lookups,:hits,:purgeable_count,:purges,:speculative_count,)
  @test sizeof(S)==fieldcount(S)*sizeof(Cuint)
  @test all(fieldtype(S,i)==Cuint && fieldoffset(S,i)==(i-1)*sizeof(Cuint) for i in 1:fieldcount(S))
  eligible=(:free_count,:inactive_count,:purgeable_count)
  @test DiffMoM._PLANAR_DARWIN_HOST_VM_INFO==2
  for free in (zero(Cuint),one(Cuint),typemax(Cuint)),inactive in (zero(Cuint),one(Cuint),typemax(Cuint)),purgeable in (zero(Cuint),one(Cuint),typemax(Cuint))
   values=(free,inactive,purgeable)
   stats=S(ntuple(i->names[i] in eligible ? values[findfirst(==(names[i]),eligible)] : typemax(Cuint),fieldcount(S))...)
   expected=sum(UInt128.(values))
   @test DiffMoM._planar_darwin_reclaimable_bytes(stats,one(Cint))==expected
   @test DiffMoM._planar_darwin_reclaimable_bytes(stats,sizeof(Cuint))==expected*sizeof(Cuint)
  end
  stats=S(ntuple(_->one(Cuint),fieldcount(S))...)
  @test_throws ArgumentError DiffMoM._planar_darwin_reclaimable_bytes(stats,zero(Cint))
  @test_throws ArgumentError DiffMoM._planar_darwin_reclaimable_bytes(stats,-one(Cint))
  @test_throws OverflowError DiffMoM._planar_darwin_reclaimable_bytes(stats,typemax(UInt128))
  DiffMoM._planar_darwin_reclaimable_bytes(stats,one(Cint))
  @test (@allocated DiffMoM._planar_darwin_reclaimable_bytes(stats,one(Cint)))==0
 end
end
run_tests()
end
