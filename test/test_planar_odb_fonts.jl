using DiffMoM, Test
import CodecZlib: GzipCompressorStream
import Tar: Tar

const _ODB_TEST_FONT = """
XSIZE 2
YSIZE 2
OFFSET 1
CHARI
LINE 1 0 1 2 P R .2
ECHAR
CHAR M
LINE 0 0 0 2 P S .2
LINE 0 2 1 0 P S .2
LINE 1 0 2 2 P S .2
LINE 2 2 2 0 P S .2
ECHAR
CHAR ;
LINE 1 .5 1 .5 P R .2
LINE 1 1.5 1 1.5 P R .2
ECHAR
"""

function _odb_font_features(text; font=_ODB_TEST_FONT, kwargs...)
    mktempdir() do directory
        write(joinpath(directory,"literal"),font)
        feature=joinpath(directory,"features");write(feature,text)
        read_odb_features(feature;font_directory=directory,kwargs...)
    end
end
_odb_font_ink(doc,x,y)=any(o->o.dark && DiffMoM._artwork_contains(o.shape,x,y),doc.objects)

@testset "ODB bounded vector font literal geometry and text insertion" begin
    doubled=replace(_ODB_TEST_FONT,"XSIZE 2"=>"XSIZE 4","YSIZE 2"=>"YSIZE 4","OFFSET 1"=>"OFFSET 2",
        "1 0 1 2"=>"2 0 2 4","0 0 0 2"=>"0 0 0 4","0 2 1 0"=>"0 4 2 0",
        "1 0 2 2"=>"2 0 4 4","2 2 2 0"=>"4 4 4 0",
        "1 .5 1 .5"=>"2 1 2 1","1 1.5 1 1.5"=>"2 3 2 3",".2"=>".4")
    for version in (0,1)
        record="T 0 0 literal P 0 3 2 .5 'I I' $version"
        a=_odb_font_features("UNITS=MM\nF 1\n"*record)
        b=_odb_font_features("UNITS=MM\n"*record;font=doubled)
        @test length(a.objects)==1
        @test DiffMoM._artwork_bounds(a.objects[1].shape)==DiffMoM._artwork_bounds(b.objects[1].shape)
        r=.0000762;xfirst=version==0 ? r : .001+r
        for x in (-.0001,.00001,.0000762,.0010762,.002,.0060762,.0070762,.008),
                y in (-.0001,.00001,.0000762,.001,.00206,.0023)
            expected=min(hypot(x-xfirst,y-clamp(y,r,.002+r)),
                hypot(x-xfirst-.006,y-clamp(y,r,.002+r)))<=r
            @test _odb_font_ink(a,x,y)==expected
            @test _odb_font_ink(b,x,y)==expected
        end
        @test a.objects[1].attributes["ODB.text.stroke_width_m"]==[string(.0001524)]
    end
    for orientation in 0:9
        angle=orientation<8 ? 90mod(orientation,4) : 37.
        field=orientation>=8 ? "$orientation 37" : string(orientation)
        doc=_odb_font_features("UNITS=MM\nT 10 20 literal P $field 3 2 .5 'I' 1")
        c,s=cosd(angle),sind(angle);matrix=[c s;-s c]
        orientation in (4,5,6,7,9) && (matrix[1,:].*=-1)
        for q in ((.0010762,.001),(.0009,.001),(.0010762,.0023))
            x,y=matrix*collect(q)+[.01,.02]
            expected=abs(q[1]-.0010762)<=.0000762 && .0000762<=q[2]<=.0020762
            @test _odb_font_ink(doc,x,y)==expected
        end
    end
    # Absolute 12mil stroke units do not depend on file/font UNITS.
    a=_odb_font_features("UNITS=MM\nT 0 0 literal P 0 3 2 .5 'I' 0")
    b=_odb_font_features("UNITS=INCH\nT 0 0 literal P 0 $(3/25.4) $(2/25.4) .5 'I' 0";
        font="UNITS=INCH\n"*_ODB_TEST_FONT)
    @test collect(DiffMoM._artwork_bounds(a.objects[1].shape))≈collect(DiffMoM._artwork_bounds(b.objects[1].shape))
    quoted=DiffMoM._odb_text_record("T 1 2 literal P 9 37 3 2 .5 'I; I'I' 1;0=2;ID=7")
    @test quoted.text=="I; I'I" && quoted.suffix==";0=2;ID=7"
    doc=_odb_font_features("UNITS=MM\n@0 .string\n&2 retained\nT 1 2 literal N 0 3 2 .5 'I; I' 1;0=2;ID=7")
    @test !only(doc.objects).dark
    @test only(doc.objects).attributes[".string"]==["2"]
    @test only(doc.objects).attributes[".string.text_lookup"]==["retained"]
    @test only(doc.objects).attributes["ID"]==["7"]
    @test only(doc.objects).attributes["ODB.text.source"]==["I; I"]
    # Clear glyph segments remain local to their glyph, hence transparent
    # to an earlier positive feature underneath the text.
    clearfont="XSIZE 2\nYSIZE 2\nOFFSET 0\nCHAR O\nLINE 1 1 1 1 P R 1\nLINE 1 1 1 1 N R 1\nECHAR\n"
    doc=_odb_font_features("UNITS=MM\n\$0 r1000\nP 0 0 0 P 0 0\nT 0 0 literal P 0 2 2 1 'O' 0";font=clearfont)
    @test DiffMoM._artwork_contains(doc.objects[1].shape,0.,0.)
    @test !DiffMoM._artwork_contains(doc.objects[2].shape,.0001524,.0001524)
end

@testset "ODB font/text validation and pre-allocation resource guards" begin
    for font in (replace(_ODB_TEST_FONT,"XSIZE 2"=>"XSIZE 0"),
            replace(_ODB_TEST_FONT,"ECHAR\nCHAR M"=>"CHAR M"),
            _ODB_TEST_FONT*"CHAR I\nECHAR\n", "XSIZE 1\nYSIZE 1\nOFFSET 0\nCHAR\n")
        @test_throws ArgumentError _odb_font_features("T 0 0 literal P 0 3 2 .5 'I' 0";font)
    end
    for record in ("T Inf 0 literal P 0 3 2 .5 'I' 0", "T 0 0 literal P 9 3 2 .5 'I' 0",
            "T 0 0 literal P 0 0 2 .5 'I' 0", "T 0 0 literal P 0 3 2 -1 'I' 0",
            "T 0 0 literal P 0 3 2 .5 'I 0", "T 0 0 ../escape P 0 3 2 .5 'I' 0",
            "T 0 0 literal P 0 3 2 .5 'X' 0")
        @test_throws ArgumentError _odb_font_features(record)
    end
    @test_throws ArgumentError _odb_font_features("T 0 0 literal P 0 3 2 .5 'I' 0";max_bytes=1)
    @test_throws ArgumentError _odb_font_features("T 0 0 literal P 0 3 2 .5 'MMM' 0";max_objects=1)
    @test_throws ArgumentError _odb_font_features(raw"T 0 0 literal P 0 3 2 .5 '$$DATE' 0")
    @test_throws ArgumentError _odb_font_features(raw"T 0 0 literal P 0 3 2 .5 '$$STEP' 0";
        text_context=Dict("STEP"=>raw"$$LAYER"))
    @test_throws ArgumentError _odb_font_features(raw"T 0 0 literal P 0 3 2 .5 '$$STEP' 0";
        text_context=Dict("STEP"=>repeat("I",10000)),max_bytes=20000)
    doc=_odb_font_features(raw"T 0 0 literal P 0 3 2 .5 '$$STEP; $$DATE-MMDDYY' 0";
        text_context=Dict("STEP"=>"I","DATE-MMDDYY"=>"M"))
    @test length(doc.objects)==1
    mktempdir() do directory
        feature=joinpath(directory,"features");write(feature,"T 0 0 literal P 0 3 2 .5 'I' 0")
        @test_throws ArgumentError read_odb_features(feature)
        calls=Ref(0)
        @test_throws ArgumentError read_odb_features(feature;max_bytes=3filesize(feature)+100,
            _font_resolver=((name,payload)->(calls[]+=1;nothing)))
        @test calls[]==0
    end
end

@testset "ODB product fonts and placement-dependent reused symbol text" begin
    mktempdir() do directory
        function entity(parts,text)
            path=joinpath(directory,parts...);mkpath(dirname(path));write(path,text)
        end
        entity(["matrix","matrix"],"STEP {\nCOL=1\nNAME=i\n}\nLAYER {\nROW=1\nNAME=i\nTYPE=SIGNAL\nPOLARITY=POSITIVE\n}\nLAYER {\nROW=2\nNAME=m\nTYPE=SIGNAL\nPOLARITY=POSITIVE\n}")
        entity(["misc","info"],"UNITS=MM\nPRODUCT_MODEL_NAME=I")
        entity(["steps","i","stephdr"],"UNITS=MM")
        font=_ODB_TEST_FONT*replace(_ODB_TEST_FONT[first(findfirst("CHARI",_ODB_TEST_FONT)):end],"CHARI"=>"CHARi","CHAR M"=>"CHAR m","CHAR ;"=>"CHAR!")
        entity(["fonts","literal"],font)
        entity(["symbols","label","features"],raw"T 0 0 literal P 0 3 2 .5 '$$LAYER' 0")
        for layer in ("i","m")
            entity(["steps","i","layers",layer,"features"],"UNITS=MM\n\$0 label\nP 10 20 0 P 0 0")
        end
        doc=read_odb(directory)
        @test length(doc.objects)==2
        ibox=DiffMoM._artwork_bounds(only(filter(o->o.layer=="i",doc.objects)).shape)
        mbox=DiffMoM._artwork_bounds(only(filter(o->o.layer=="m",doc.objects)).shape)
        @test ibox[1]≈.01 && ibox[3]≈.02
        @test mbox[1]≈.01 && mbox[3]≈.02
        @test mbox[2]-mbox[1]>10*(ibox[2]-ibox[1])
        @test all(o->!DiffMoM._odb_has_deferred_text(o.shape),doc.objects)
        entity(["steps","i","layers","i","features"],raw"T 0 0 literal P 0 3 2 .5 '$$JOB $$STEP' 0")
        @test length(read_odb(directory;layers=["i"]).objects)==1
        # Nested user symbols resolve text at the actual final placement,
        # rather than at the cached child symbol's definition origin.
        numeric="XSIZE 2\nYSIZE 2\nOFFSET 1\n"*join(("CHAR $c\nLINE 1 0 1 2 P R .2\nECHAR\n" for c in "0123456789."))
        entity(["fonts","numeric"],numeric)
        entity(["symbols","position","features"],raw"T 1 0 numeric P 0 3 2 .5 '$$X_MM' 0")
        entity(["symbols","nested","features"],"UNITS=MM\n\$0 position\nP 2 0 0 P 0 0")
        entity(["steps","i","layers","i","features"],"UNITS=MM\n\$0 nested\nP 8 0 0 P 0 0\nP 98 0 0 P 0 0")
        positions=read_odb(directory;layers=["i"])
        @test length(positions.objects)==2
        b1,b2=map(o->o.bounds,positions.objects)
        @test b1[1]≈.011 && b2[1]≈.101
        @test b2[2]-b2[1]>b1[2]-b1[1]
        # Product-owned fonts remain available while compressed layer
        # entities and scoped archive extraction are in use.
        feature=joinpath(directory,"steps","i","layers","i","features")
        open(feature*".gz","w") do io
            stream=GzipCompressorStream(io)
            try;write(stream,read(feature));finally;close(stream);end
        end
        rm(feature)
        compressed=read_odb(directory;layers=["i"])
        @test [o.bounds for o in compressed.objects]==[o.bounds for o in positions.objects]
        mktempdir() do transport
            archive=joinpath(transport,"font_product.tar");Tar.create(directory,archive)
            fromarchive=read_odb(archive;layers=["i"])
            @test [o.bounds for o in fromarchive.objects]==[o.bounds for o in positions.objects]
        end
        rm(feature*".gz")
        # Both input files fit separately; their concurrent payload does
        # not. The parent reader prefix must be reserved before font I/O.
        entity(["steps","i","layers","i","features"],repeat("# retained parent prefix\n",2000)*"T 0 0 literal P 0 3 2 .5 'I' 0")
        entity(["fonts","literal"],repeat("# retained font prefix\n",2000)*font)
        @test 3filesize(joinpath(directory,"fonts","literal"))<200000
        @test 3filesize(joinpath(directory,"steps","i","layers","i","features"))<200000
        err=try read_odb(directory;layers=["i"],max_bytes=200000);nothing catch e;e end
        @test err isa ArgumentError
        @test occursin("max_bytes",sprint(showerror,err))
    end
end
