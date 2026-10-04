using Test, DiffMoM

function _odb_legacy_units_features(text;kwargs...)
    mktempdir() do dir
        path=joinpath(dir,"features")
        write(path,text)
        read_odb_features(path;kwargs...)
    end
end

@testset "Legacy ODB U units retain strict declaration contract" begin
    # Literal legacy syntax occurs in the unchanged installed Sonnet
    # odb++_trans.tgz. No native translator/rendering acceptance is implied.
    for (unit,scale) in (("INCH",.0254),("MM",1e-3)),spacing in (" ","\t")
        a=_odb_legacy_units_features("F 1\nU$(spacing)$unit\n\$0 r1000\nP 2 3 0 P 0 0")
        b=_odb_legacy_units_features("F 1\nUNITS=$unit\n\$0 r1000\nP 2 3 0 P 0 0")
        @test a.coordinate_unit_m==scale
        @test only(a.objects).bounds==only(b.objects).bounds
        @test a.attributes["ODB.units.syntax"]==["U"]
        @test b.attributes["ODB.units.syntax"]==["UNITS="]
        @test a.attributes["ODB.units.value"]==[unit]
        @test DiffMoM._artwork_contains(only(a.objects).shape,2scale,3scale)
    end
    for records in ("U\n","U INCH extra\n","U CM\n","U inch\n","U INCH;ID=1\n",
            "U INCH\nU MM\n","UNITS=INCH\nU MM\n","U INCH\nUNITS=MM\n",
            "\$0 r1000\nU MM\n","U MM\n\$0 r1000\nP 0 0 0 P 0 0\nU INCH\n")
        @test_throws ArgumentError _odb_legacy_units_features(records)
    end
    @test_throws ArgumentError _odb_legacy_units_features("U INCH\n";max_bytes=1)
    # The syntax/value metadata remain live after input parsing; a limit
    # covering only the input must reject before a later symbol callback.
    @test_throws ArgumentError _odb_legacy_units_features("U INCH\n";max_bytes=21)
    called=Ref(false)
    @test_throws ArgumentError _odb_legacy_units_features("U MM\n\$0 custom\n";
        max_bytes=3ncodeunits("U MM\n\$0 custom\n")+1023,
        symbol_resolver=name->(called[]=true;DiffMoM._ArtworkEmpty()))
    @test !called[]
    observed=Int[]
    _odb_legacy_units_features("U INCH\n";_payload_observer=x->push!(observed,x))
    @test last(observed)>=21+1024
end
