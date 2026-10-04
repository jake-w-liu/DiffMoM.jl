module NativeSonnetScalarAliasIntegrityTests
using Test, DiffMoM, SHA

@testset "Every retained scalar alias source is checked and counted" begin
    fixture=joinpath(@__DIR__,"fixtures/native_sonnet_scalar_tables/table1_nodes/escaped.son")
    mktempdir() do dir
        source=joinpath(dir,"coupon.son");cp(fixture,source)
        write(joinpath(dir,"loss.csv"),"1,1\n2,2\n")
        p=read_sonnet_project(source)
        p.variables["Loss"]="table1(\"loss.csv\",1)+table1(\"./loss.csv\",1)"
        files=sonnet_scalar_files(p;max_files=1)
        names=collect(keys(files.tables));old=files.tables[last(names)]
        @test length(names)==2
        @test files.tables[first(names)]===old
        @test length(DiffMoM._sonnet_scalar_dependency_sources(files))==1
        @test sonnet_variable_value(files,"Loss")==2.
        initial=DiffMoM._sonnet_scalar_files_payload(files)
        numeric=DiffMoM._sonnet_scalar_numeric_payload(Dict{String,Float64}())
        @test DiffMoM._sonnet_scalar_variables(p,Dict{String,Float64}();
            scalar_files=files,max_bytes=initial+numeric) isa DiffMoM.SonnetScalarVariables

        # An equivalent alias can retain a separately owned source array.
        # Its contents remain valid, but its storage must fit before copying.
        copied=copy(old.source.bytes)
        files.tables[last(names)]=SonnetScalarTable(old.kind,old.row_keys,old.column_keys,
            old.values,SonnetScalarSource(old.source.path,copied,old.source.sha256))
        retained=DiffMoM._sonnet_scalar_files_payload(files)
        @test retained==initial+256+ncodeunits(old.source.path)+length(copied)
        @test sonnet_variable_value(files,"Loss")==2.
        @test_throws ArgumentError DiffMoM._sonnet_scalar_variables(p,Dict{String,Float64}();
            scalar_files=files,max_bytes=initial+numeric)
        @test DiffMoM._sonnet_scalar_variables(p,Dict{String,Float64}();
            scalar_files=files,max_bytes=retained+numeric) isa DiffMoM.SonnetScalarVariables

        copied[1]=UInt8('9')
        @test bytes2hex(sha256(copied))!=old.source.sha256
        @test_throws ArgumentError sonnet_variable_value(files,"Loss")
        @test_throws ArgumentError DiffMoM._sonnet_check_scalar_files(files)

        # Even shared bytes must match every source's own claimed digest.
        files.tables[last(names)]=SonnetScalarTable(old.kind,old.row_keys,old.column_keys,
            old.values,SonnetScalarSource(old.source.path,old.source.bytes,repeat("0",64)))
        err=try DiffMoM._sonnet_check_scalar_files(files);nothing catch e;e end
        @test err isa ArgumentError
        @test occursin("source snapshot",sprint(showerror,err))
        files.tables[last(names)]=old
        @test DiffMoM._sonnet_scalar_files_payload(files)==initial
        @test sonnet_variable_value(files,"Loss")==2.

        # Sharing values does not imply sharing the numeric axes. Equivalent
        # separately owned arrays still consume storage before context copy.
        rows=copy(old.row_keys);columns=copy(old.column_keys)
        files.tables[last(names)]=SonnetScalarTable(old.kind,rows,columns,old.values,old.source)
        extra=sizeof(rows)+sizeof(columns)
        @test extra>0
        @test DiffMoM._sonnet_scalar_files_payload(files)==initial+extra
        @test sonnet_variable_value(files,"Loss")==2.
        @test_throws ArgumentError DiffMoM._sonnet_scalar_variables(p,Dict{String,Float64}();
            scalar_files=files,max_bytes=initial+numeric)
        @test DiffMoM._sonnet_scalar_variables(p,Dict{String,Float64}();
            scalar_files=files,max_bytes=initial+extra+numeric) isa DiffMoM.SonnetScalarVariables
    end
end
end
