using DiffMoM,Test
import JSON: JSON
import Tar: Tar

function _odb_flip_record(row,name,type;kwargs...)
    r=Dict("ROW"=>string(row),"NAME"=>name,"TYPE"=>type,"CONTEXT"=>"BOARD","POLARITY"=>"POSITIVE")
    for (key,value) in kwargs;r[String(key)]=value;end
    r
end
function _odb_flip_matrix(records;steps=("board","panel","outer"))
    join(("STEP {\nCOL=$i\nNAME=$name\n}\n" for (i,name) in enumerate(steps))) *
        join(("LAYER {\n"*join(("$k=$v\n" for (k,v) in record))*"}\n" for record in records))
end

@testset "ODB full buildup FLIP layer and axial-span contracts" begin
    records=[_odb_flip_record(80,"bottom","SIGNAL"),_odb_flip_record(7,"g1","POWER_GROUND"),
        _odb_flip_record(2,"top","SIGNAL"),_odb_flip_record(35,"g2","POWER_GROUND"),
        _odb_flip_record(91,"blindtop","DRILL";START_NAME="top",END_NAME="g1"),
        _odb_flip_record(105,"blindbottom","DRILL";START_NAME="g2",END_NAME="bottom"),
        _odb_flip_record(131,"through","DRILL"),_odb_flip_record(142,"outline","ROUT"),
        _odb_flip_record(151,"notes","DOCUMENT"),_odb_flip_record(190,"diagnostic","SIGNAL";CONTEXT="MISC")]
    expected=Dict("top"=>"bottom","g1"=>"g2","g2"=>"g1","bottom"=>"top",
        "blindtop"=>"blindbottom","blindbottom"=>"blindtop","through"=>"through",
        "outline"=>"outline","notes"=>"notes","diagnostic"=>"diagnostic")
    @test DiffMoM._odb_flip_layer_map(records;max_bytes=1000000)==expected
    @test DiffMoM._odb_flip_layer_map(reverse(records);max_bytes=1000000)==expected
    @test all(n->expected[expected[n]]==n,keys(expected))
    @test_throws ArgumentError DiffMoM._odb_flip_layer_map(records;max_bytes=1)
    for (field,value) in (("TYPE","MIXED"),("POLARITY","NEGATIVE"),("ROW","2"),("ROW","0"))
        invalid=deepcopy(records);invalid[1][field]=value
        @test_throws ArgumentError DiffMoM._odb_flip_layer_map(invalid;max_bytes=1000000)
    end
    invalid=deepcopy(records);invalid[5]["START_NAME"]="missing"
    @test_throws ArgumentError DiffMoM._odb_flip_layer_map(invalid;max_bytes=1000000)
    invalid=deepcopy(records);invalid[5]["POLARITY"]="NEGATIVE"
    @test_throws ArgumentError DiffMoM._odb_flip_layer_map(invalid;max_bytes=1000000)
    invalid=deepcopy(records);invalid[6]["END_NAME"]="g1"
    @test_throws ArgumentError DiffMoM._odb_flip_layer_map(invalid;max_bytes=1000000)
    duplicate=[records;_odb_flip_record(95,"blindtop2","DRILL";START_NAME="top",END_NAME="g1");
        _odb_flip_record(115,"blindbottom2","DRILL";START_NAME="g2",END_NAME="bottom")]
    @test_throws ArgumentError DiffMoM._odb_flip_layer_map(duplicate;max_bytes=1000000)
    explicit=Dict("blindtop"=>"blindbottom","blindbottom"=>"blindtop",
        "blindtop2"=>"blindbottom2","blindbottom2"=>"blindtop2")
    @test DiffMoM._odb_flip_layer_map(duplicate;override=explicit,max_bytes=1000000)["blindtop2"]=="blindbottom2"
    for override in (Dict("top"=>"g2"),Dict("notes"=>"diagnostic"),Dict("absent"=>"top"),Dict("notes"=>1))
        @test_throws ArgumentError DiffMoM._odb_flip_layer_map(records;override,max_bytes=1000000)
    end
    # Same spans with distinct technology subtypes map without ambiguity.
    typed=deepcopy(duplicate)
    for r in typed
        endswith(r["NAME"],"2") && r["TYPE"]=="DRILL" && (r["ADD_TYPE"]="BACKDRILL")
    end
    @test DiffMoM._odb_flip_layer_map(typed;max_bytes=1000000)["blindtop2"]=="blindbottom2"
end

@testset "ODB public FLIP selected layers, transforms and nested repetitions" begin
    mktempdir() do root
        function entity(parts,text)
            path=joinpath(root,parts...);mkpath(dirname(path));write(path,text)
        end
        records=[_odb_flip_record(1,"top","SIGNAL"),_odb_flip_record(3,"inner1","SIGNAL"),
            _odb_flip_record(7,"inner2","SIGNAL"),_odb_flip_record(15,"bottom","SIGNAL")]
        centers=Dict("top"=>(4,5),"inner1"=>(5,6),"inner2"=>(6,7),"bottom"=>(7,3))
        inverse=Dict("top"=>"bottom","inner1"=>"inner2","inner2"=>"inner1","bottom"=>"top")
        entity(["matrix","matrix"],_odb_flip_matrix(records))
        entity(["misc","info"],"UNITS=MM")
        entity(["steps","board","stephdr"],"UNITS=MM\nX_DATUM=1\nY_DATUM=2")
        for (layer,center) in centers
            entity(["steps","board","layers",layer,"features"],"UNITS=MM\n\$0 r1000\nP $(center[1]) $(center[2]) 0 P 0 0")
        end
        for angle in (0.,37.,90.,180.,270.),mirror in (false,true)
            entity(["steps","panel","stephdr"],"UNITS=MM\nSTEP-REPEAT {\nNAME=board\nX=30\nY=40\nDX=10\nDY=20\nNX=2\nNY=2\nANGLE=$angle\nMIRROR=$(mirror ? "YES" : "NO")\nFLIP=YES\n}")
            for layer in ("top","inner1","inner2","bottom")
                doc=read_odb(root;step="panel",layers=[layer])
                @test length(doc.objects)==4 && all(o->o.layer==layer,doc.objects)
                @test JSON.parse(only(doc.attributes["ODB.flip_layer_map"]))==inverse
                c,s=cosd(angle),sind(angle);matrix=[c s;-s c]
                !mirror && (matrix[1,:].*=-1)
                for j in 0:1,i in 0:1
                    q=matrix*([centers[inverse[layer]]...]+[10i,20j]-[1,2])*1e-3+[.03,.04]
                    @test any(o->DiffMoM._artwork_contains(o.shape,q...),doc.objects)
                    @test !any(o->DiffMoM._artwork_contains(o.shape,q[1]+.002,q[2]),doc.objects)
                end
            end
        end
        # Destination TOP still loads BOTTOM when source TOP is absent.
        rm(joinpath(root,"steps","board","layers","top","features"))
        @test length(read_odb(root;step="panel",layers=["top"]).objects)==4
        entity(["steps","board","stephdr"],"UNITS=MM")
        entity(["steps","panel","stephdr"],"UNITS=MM\nSTEP-REPEAT {\nNAME=board\nFLIP=YES\nMIRROR=NO\n}")
        entity(["steps","outer","stephdr"],"UNITS=MM\nSTEP-REPEAT {\nNAME=panel\nFLIP=YES\nMIRROR=NO\n}")
        doc=read_odb(root;step="outer",layers=["bottom"])
        @test length(doc.objects)==1 && only(doc.objects).layer=="bottom"
        @test DiffMoM._artwork_contains(only(doc.objects).shape,.007,.003)
        entity(["steps","outer","stephdr"],"UNITS=MM\nSTEP-REPEAT {\nNAME=panel\nFLIP=YES\nMIRROR=YES\n}")
        doc=read_odb(root;step="outer",layers=["bottom"])
        @test DiffMoM._artwork_contains(only(doc.objects).shape,-.007,.003)
        mktempdir() do transport
            archive=joinpath(transport,"flipped.tar");Tar.create(root,archive)
            archived=read_odb(archive;step="outer",layers=["bottom"])
            @test [o.bounds for o in archived.objects]==[o.bounds for o in doc.objects]
        end
        for flag in ("FLIP=MAYBE","MIRROR=MAYBE","NX=0","ANGLE=NaN","X=Inf")
            entity(["steps","outer","stephdr"],"STEP-REPEAT {\nNAME=board\n$flag\n}")
            @test_throws ArgumentError read_odb(root;step="outer",layers=["bottom"])
        end
    end
end

@testset "ODB flipped negative profiles, drill layers and full symmetry validation" begin
    mktempdir() do root
        function entity(parts,text)
            p=joinpath(root,parts...);mkpath(dirname(p));write(p,text)
        end
        records=[_odb_flip_record(1,"top","SIGNAL"),_odb_flip_record(3,"g1","POWER_GROUND";POLARITY="NEGATIVE"),
            _odb_flip_record(7,"g2","POWER_GROUND";POLARITY="NEGATIVE"),_odb_flip_record(15,"bottom","SIGNAL"),
            _odb_flip_record(31,"blindtop","DRILL";START_NAME="top",END_NAME="g1"),
            _odb_flip_record(34,"blindbottom","DRILL";START_NAME="g2",END_NAME="bottom")]
        entity(["matrix","matrix"],_odb_flip_matrix(records;steps=("board","panel")))
        entity(["misc","info"],"UNITS=MM")
        entity(["steps","board","stephdr"],"UNITS=MM")
        entity(["steps","panel","stephdr"],"UNITS=MM\nSTEP-REPEAT {\nNAME=board\nX=20\nFLIP=YES\n}")
        entity(["steps","board","layers","g2","features"],"UNITS=MM\n\$0 r1000\nP 7 2 0 P 0 0")
        entity(["steps","board","layers","blindbottom","features"],"UNITS=MM\n\$0 hole1000xpx0x0\nP 6 3 0 P 0 0")
        entity(["steps","board","profile"],"UNITS=MM\nS P 0\nOB 0 0 I\nOS 10 0\nOS 10 10\nOS 0 10\nOS 0 0\nOE\nSE")
        doc=read_odb(root;step="panel",layers=["g1","blindtop"])
        ground=only(filter(o->o.layer=="g1",doc.objects));drill=only(filter(o->o.layer=="blindtop",doc.objects))
        @test !DiffMoM._artwork_contains(ground.shape,.013,.002)
        @test DiffMoM._artwork_contains(ground.shape,.015,.005)
        @test !DiffMoM._artwork_contains(ground.shape,.009,.005)
        @test DiffMoM._artwork_contains(drill.shape,.014,.003)
        @test drill.attributes["ODB.drill.plating"]==["plated"]
        append!(records,[_odb_flip_record(38,"blindtop2","DRILL";START_NAME="top",END_NAME="g1"),
            _odb_flip_record(40,"blindbottom2","DRILL";START_NAME="g2",END_NAME="bottom")])
        entity(["matrix","matrix"],_odb_flip_matrix(records;steps=("board","panel")))
        entity(["steps","board","layers","blindbottom2","features"],"UNITS=MM\n\$0 r1000\nP 4 6 0 P 0 0")
        @test_throws ArgumentError read_odb(root;step="panel",layers=["blindtop2"])
        explicit=Dict("blindtop"=>"blindbottom","blindbottom"=>"blindtop",
            "blindtop2"=>"blindbottom2","blindbottom2"=>"blindtop2")
        resolved=read_odb(root;step="panel",layers=["blindtop2"],flip_layer_map=explicit)
        @test only(resolved.objects).layer=="blindtop2"
        @test DiffMoM._artwork_contains(only(resolved.objects).shape,.016,.006)
        # A selected drill alone cannot evade symmetry validation elsewhere.
        records[4]["TYPE"]="MIXED"
        entity(["matrix","matrix"],_odb_flip_matrix(records;steps=("board","panel")))
        @test_throws ArgumentError read_odb(root;step="panel",layers=["blindtop"])
    end
end

@testset "ODB matrix index and bounded metadata regression contracts" begin
    mktempdir() do root
        function entity(parts,text)
            p=joinpath(root,parts...);mkpath(dirname(p));write(p,text)
        end
        records=[_odb_flip_record(1,"top","SIGNAL")]
        base=_odb_flip_matrix(records;steps=("board",))
        entity(["misc","info"],"UNITS=MM")
        entity(["steps","board","stephdr"],"UNITS=MM")
        entity(["steps","board","layers","top","features"],"UNITS=MM\n\$0 r1000\nP 1 1 0 P 0 0")
        for matrix in (replace(base,"ROW=1"=>"ROW=0"),replace(base,"COL=1"=>"COL=0"),
                base*"STEP {\nCOL=1\nNAME=other\n}",base*"LAYER {\nROW=1\nNAME=bottom\nTYPE=SIGNAL\n}",
                base*"STEP {\nCOL=2\nNAME=board\n}")
            entity(["matrix","matrix"],matrix)
            @test_throws ArgumentError read_odb(root;step="board",layers=["top"])
        end
        entity(["matrix","matrix"],base)
        @test length(read_odb(root;step="board").objects)==1
        @test_throws ArgumentError read_odb(root;step="board",max_bytes=1)
        entity(["steps","board","stephdr"],"UNITS=MM\nX_DATUM=NaN")
        @test_throws ArgumentError read_odb(root;step="board")
    end
    data=Dict("unicode"=>"Ω","escaped"=>"\"\\\t\n", "empty"=>String[])
    expected=JSON.json(data);n=ncodeunits(expected)
    @test JSON.parse(DiffMoM._odb_bounded_json(data,256+3n))==data
    @test_throws ArgumentError DiffMoM._odb_bounded_json(data,256+3n-1)
    @test_throws ArgumentError DiffMoM._odb_bounded_json(data,1)
    controls=String(UInt8[0:31;0x22;0x5c;0x7f])*"Ω🙂"
    complete=Dict("strings"=>[controls,"", "slash/"],"boolean"=>true,"false"=>false,"null"=>nothing)
    bytes=ncodeunits(JSON.json(complete))
    rendered=DiffMoM._odb_bounded_json(complete,256+3bytes)
    @test JSON.parse(rendered)==complete
    actualbytes=ncodeunits(rendered)
    @test_throws ArgumentError DiffMoM._odb_bounded_json(complete,256+3actualbytes-1)
    # JSON1's IO writer previously built a megabyte buffer before the
    # bounded IO rejected this tiny budget. Reject before output allocation.
    large=Dict("payload"=>repeat("\\\t\"",100000))
    rejected()=try;DiffMoM._odb_bounded_json(large,257);false catch e;e isa ArgumentError end
    @test rejected()
    @test @allocated(rejected())<20000
    # Separately valid matrix/info inputs cannot independently consume
    # the complete budget while the first parsed entity remains owned.
    mktempdir() do root
        function entity(parts,text)
            p=joinpath(root,parts...);mkpath(dirname(p));write(p,text);p
        end
        matrix=entity(["matrix","matrix"],"COMMENT="*repeat("v",10000)*"\n"*
            _odb_flip_matrix([_odb_flip_record(1,"top","SIGNAL")];steps=("board",)))
        info=entity(["misc","info"],"UNITS=MM\nCOMMENT="*repeat("v",10000))
        entity(["steps","board","stephdr"],"UNITS=MM")
        entity(["steps","board","layers","top","features"],"UNITS=MM\nF 0")
        @test haskey(DiffMoM._odb_structured(matrix;max_bytes=60000).parameters,"COMMENT")
        @test haskey(DiffMoM._odb_structured(info;max_bytes=60000).parameters,"COMMENT")
        @test_throws ArgumentError read_odb(root;step="board",max_bytes=60000)
    end
end

@testset "ODB local matrix reference canonicalization preserves raw metadata" begin
    mktempdir() do root
        function entity(parts,text)
            p=joinpath(root,parts...);mkpath(dirname(p));write(p,text)
        end
        records=[_odb_flip_record(1,"TOP","SIGNAL"),_odb_flip_record(7,"BOTTOM","SIGNAL"),
            _odb_flip_record(20,"BLIND_TOP","DRILL";START_NAME="TOP",END_NAME="TOP"),
            _odb_flip_record(31,"BLIND_BOTTOM","DRILL";START_NAME="BOTTOM",END_NAME="BOTTOM")]
        matrix=_odb_flip_matrix(records;steps=("PCB","PANEL"))
        entity(["matrix","matrix"],matrix);entity(["misc","info"],"UNITS=MM")
        entity(["steps","pcb","stephdr"],"UNITS=MM")
        entity(["steps","panel","stephdr"],"UNITS=MM\nSTEP-REPEAT {\nNAME=PCB\nX=10\nFLIP=YES\n}")
        entity(["steps","pcb","layers","bottom","features"],"UNITS=MM\n\$0 r1000\nP 3 4 0 P 0 0")
        entity(["steps","pcb","layers","blind_bottom","features"],"UNITS=MM\n\$0 r1000\nP 2 5 0 P 0 0")
        doc=read_odb(root;step="PANEL",layers=["TOP","BLIND_TOP"])
        @test Set(o.layer for o in doc.objects)==Set(["top","blind_top"])
        @test DiffMoM._artwork_contains(only(filter(o->o.layer=="top",doc.objects)).shape,.007,.004)
        @test DiffMoM._artwork_contains(only(filter(o->o.layer=="blind_top",doc.objects)).shape,.008,.005)
        raw=JSON.parse(only(doc.attributes["ODB.matrix"]))
        @test raw["records"]["STEP"][1]["NAME"]=="PCB"
        @test raw["records"]["LAYER"][1]["NAME"]=="TOP"
        @test raw["records"]["LAYER"][3]["START_NAME"]=="TOP"
        @test JSON.parse(only(doc.attributes["ODB.flip_layer_map"]))["blind_top"]=="blind_bottom"
        @test DiffMoM._odb_reference_name("COMP_+_TOP")=="comp_+_top"
        @test_throws ArgumentError DiffMoM._odb_name("TOP") # global entity/path rule remains strict
        duplicate=matrix*"LAYER {\nROW=100\nNAME=top\nTYPE=SIGNAL\n}\n"
        entity(["matrix","matrix"],duplicate)
        @test_throws ArgumentError read_odb(root;step="PANEL",layers=["TOP"])
        entity(["matrix","matrix"],matrix*"STEP {\nCOL=100\nNAME=pcb\n}\n")
        @test_throws ArgumentError read_odb(root;step="PANEL",layers=["TOP"])
        entity(["matrix","matrix"],matrix)
        @test_throws ArgumentError read_odb(root;step="../PCB",layers=["TOP"])
        @test_throws ArgumentError read_odb(root;step="PANEL",layers=["../TOP"])
        @test_throws ArgumentError read_odb(root;step="PANEL",layers=["TOP"],
            flip_layer_map=Dict("TOP"=>"BOTTOM","top"=>"bottom"))
        # Canonicalization stays within resolved entity paths and does not
        # permit an actual uppercase entity directory/name to bypass _odb_name.
        @test_throws ArgumentError DiffMoM._odb_reference_name("../COMP_+_TOP")
    end
end
