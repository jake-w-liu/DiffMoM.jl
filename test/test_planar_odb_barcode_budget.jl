module ODBBarcodeBudgetTests
using DiffMoM,Test

function low_budget_shape(record)
    try
        DiffMoM._odb_barcode_shape(nothing,record,.0254,Dict{String,String}();
            max_strokes=100000,max_bytes=1)
    catch error
        error
    end
end

@testset "Barcode geometry rejects before rectangle expansion" begin
    records=[DiffMoM._odb_barcode_record("B 0 0 UPC39 standard P 0 E 0.01 0.1 N N N N B '"*
        repeat("A",n)*"'") for n in (1,1024)]
    foreach(low_budget_shape,records)
    for record in records
        error=low_budget_shape(record)
        @test error isa ArgumentError && occursin("ODB barcode geometry",sprint(showerror,error))
    end
    @test @allocated(low_budget_shape(records[2]))<16384
    mktempdir() do dir
        path=joinpath(dir,"features")
        source="UNITS=INCH\nF 1\nB 0 0 UPC39 standard P 0 E 0.01 0.1 N N N N B 'AAAA'\n"
        write(path,source)
        error=try read_odb_features(path;max_bytes=8192) catch error;error end
        @test error isa ArgumentError && occursin("ODB barcode geometry",sprint(showerror,error))
        document=read_odb_features(path;max_bytes=65536)
        @test length(document.objects)==1
        # Six symbols, each fifteen narrow units, separated by five gaps.
        @test DiffMoM._artwork_bounds(only(document.objects).shape)[2]≈95*.01*.0254
    end
end
end
