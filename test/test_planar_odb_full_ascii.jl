using DiffMoM, Test

@testset "ODB Full ASCII barcode shift literals" begin
    # ISO 16388 Table A.2 mapping, independently retained in Zint's EC39Ctrl:
    # https://github.com/zint/zint/blob/master/backend/code.c
    for (literal,encoded) in (('$',"/D"),('%',"/E"),('+',"/K"),('/',"/O"))
        @test DiffMoM._odb_code39_encode(string(literal),true)==encoded
    end
    @test DiffMoM._odb_code39_encode(raw"$%+/",false)==raw"$%+/"
    @test DiffMoM._odb_code39_encode("AZ09-. ",true)=="AZ09-. "
    @test DiffMoM._odb_code39_encode("+Aa*",true)=="/KA+A/J"
    @test_throws ArgumentError DiffMoM._odb_code39_encode("*",false)
    @test_throws ArgumentError DiffMoM._odb_code39_encode("é",true)

    mktempdir() do dir
        path=joinpath(dir,"features")
        write(path,"UNITS=INCH\nF 1\nB 0 0 UPC39 standard P 0 E 0.01 0.1 Y N N N B '+'\n")
        shape=only(read_odb_features(path).objects).shape
        narrow=.01*.0254
        # Literal '+' expands to /K, hence four 15-unit symbols with three
        # one-unit gaps. Check interior witnesses of '/' and 'K' directly.
        @test DiffMoM._artwork_bounds(shape)[2]≈63narrow
        @test !DiffMoM._artwork_contains(shape,22.5narrow,.05*.0254)
        @test DiffMoM._artwork_contains(shape,33.5narrow,.05*.0254)
    end
end
