using Test, DiffMoM, SHA, JSON
using LinearAlgebra: I

function _technology_test_xml(;extra="",units="",materials="",variables="",bias="",stack="",rootattrs="")
    return """<?xml version="1.0"?>
    <technology_file version="1700" $rootattrs>
      <units $units/>
      <public>
        <variables><var name="H" units="LENG" value="100"/>$variables</variables>
        <materials>
          <dielectric name="Air"><params/></dielectric>
          <dielectric name="Sub" cond_res="rsvy"><params erel="11.9" rsvy="70000"/></dielectric>
          <conductor name="Copper" cond="40000000"/>
          <conductor name="Film" condspec="rsvy" rsvy=".025"/>
          <conductor name="Sheet" condspec="shres" shres="2"/>
          $materials
        </materials>
        <metal_model_defs><metal_model name="Normal" model_type="Normal"/></metal_model_defs>
        <bias_defs>
          <bias name="Metal6_Bias">
            <lookup_table row_name="width" col_name="thickness" value_name="rho">
              <column_keys>.95</column_keys><row key=".5">.025</row>
              <row key="1">.0221</row><row key="10">.0217</row>
            </lookup_table>
            <lookup_table row_name="width" col_name="space" value_name="etch" etch_type="CAP">
              <column_keys>.5</column_keys><row key=".5">-.04</row>
            </lookup_table>
            <lookup_table row_name="width" col_name="space" value_name="etch" etch_type="RES">
              <column_keys>.5</column_keys><row key=".5">-.02</row>
            </lookup_table>
          </bias>
          <bias name="Via1_Bias"><lookup_vector row_name="area" value_name="rpv" size="1">
            <vector key=".0025">20</vector><vector key=".00375">10</vector><vector key=".005">5</vector>
          </lookup_vector></bias>
          $bias
        </bias_defs>
        <stackup><TOP material="Lossless" model="Normal"/>
          <diel name="Upper" dielectric="Air" thickness="H"/>
          <diel name="Lower" dielectric="Sub" thickness="200"/>
          $stack<BOTTOM material="Lossless" model="Normal"/>
        </stackup>
      </public>$extra
    </technology_file>"""
end

@testset "STF schema defaults, static materials and literal nodes" begin
    mktempdir() do dir
        path=joinpath(dir,"test.stf");xml=_technology_test_xml();write(path,xml)
        t=read_sonnet_technology(path)
        @test t.raw==xml
        @test t.sha256==bytes2hex(sha256(xml))
        @test t.schema_status==:not_validated && t.schema_sha256===nothing
        @test t.units==Dict("lunit"=>"UM","cunit"=>"SM","runit"=>"OHUM","srunit"=>"OHSQ","tempunit"=>"C")
        @test sonnet_technology_value(t,"H";quantity=:length)≈1e-4
        @test sonnet_technology_value(t,".025";quantity=:resistivity)==2.5e-8
        @test sonnet_technology_value(t,"H";variables=Dict("H"=>200),quantity=:length)≈2e-4
        air=sonnet_technology_material(t,"Air")
        @test (air.er,air.mur,air.tane,air.tanm,air.sigma)==(1.,1.,0.,0.,0.)
        substrate=sonnet_technology_material(t,"Sub")
        @test substrate.sigma≈100/7 && substrate.er==11.9
        @test sonnet_technology_material(t,"Copper";kind=:conductor).sigma==4e7
        @test sonnet_technology_material(t,"Film";kind=:conductor).sigma==4e7
        @test sonnet_technology_material(t,"Sheet";kind=:conductor).resistance==2
        table=sonnet_technology_table(t,"Metal6_Bias";value_name="rho")
        for (width,expected) in ((.5,.025),(1.,.0221),(10.,.0217))
            @test only(sonnet_technology_lookup(table,width,.95))==expected
        end
        for (etype,expected) in (("CAP",-.04),("RES",-.02))
            @test only(sonnet_technology_lookup(sonnet_technology_table(t,"Metal6_Bias";value_name="etch",etch_type=etype),.5,.5))==expected
        end
        vector=sonnet_technology_table(t,"Via1_Bias";value_name="rpv")
        for (area,expected) in ((.0025,20.),(.00375,10.),(.005,5.))
            @test only(sonnet_technology_lookup(vector,area))==expected
        end
        values=sonnet_technology_lookup(table,.5,.95);values[1]=999
        @test only(sonnet_technology_lookup(table,.5,.95))==.025
        for coords in ((.6,.95),(.499,.95),(11.,.95),(.5,),(.5,NaN))
            @test_throws ArgumentError sonnet_technology_lookup(table,coords...)
        end
        @test_throws ArgumentError sonnet_technology_table(t,"Metal6_Bias";value_name="etch")
        @test_throws ArgumentError sonnet_technology_table(t,"absent";value_name="rho")
        for text in ("H*2","unknown","NaN","Inf","1e500","1e-500")
            @test_throws ArgumentError sonnet_technology_value(t,text)
        end
        @test_throws ArgumentError sonnet_technology_value(t,"H";variables=Dict("H"=>Inf))
        @test_throws ArgumentError sonnet_technology_value(t,"1";quantity=:rpv)
        stack=sonnet_technology_stack(t,1e9,.001,.002)
        @test length(stack.layers)==2
        @test stack.layers[1].thickness≈.0002
        @test stack.layers[2].thickness≈.0001
        @test stack.layers[1].epsr≈11.9-im*(100/7)/(2pi*1e9*8.8541878128e-12) rtol=1e-9
        @test stack.layers[2].epsr==1
        @test stack.top.kind==TERM_GND.kind && stack.bottom.kind==TERM_GND.kind
        @test_throws ArgumentError sonnet_technology_stack(t,0.,.001,.002)
        for (unit,q,raw,expected) in (("lunit=\"MIL\"",:length,"1",25.4e-6),
            ("cunit=\"SCM\"",:conductivity,"1",100.),("cunit=\"MSCM\"",:conductivity,"1",.1),
            ("cunit=\"USCM\"",:conductivity,"1",1e-4),("runit=\"OHCM\"",:resistivity,"1",.01),
            ("runit=\"OHMM\"",:resistivity,"1",1.),("srunit=\"MOSQ\"",:sheet_resistance,"1",.001),
            ("srunit=\"MOHSQ\"",:sheet_resistance,"1",.001))
            write(path,_technology_test_xml(units=unit))
            @test sonnet_technology_value(read_sonnet_technology(path),raw;quantity=q)==expected
        end
        write(path,_technology_test_xml(units="srunit=\"MEGAOHSQ\""))
        @test_throws ArgumentError sonnet_technology_value(read_sonnet_technology(path),"1";quantity=:sheet_resistance)
    end
end

@testset "Actual native STF maintain-physical unit exports" begin
    fixture=joinpath(@__DIR__,"fixtures","native_stf_unit_gui")
    for file in JSON.parsefile(joinpath(fixture,"manifest.json"))["files"]
        bytes=read(joinpath(fixture,file["path"]))
        @test length(bytes)==file["bytes"]
        @test bytes2hex(sha256(bytes))==file["sha256"]
    end
    schema=joinpath(fixture,"matl-1.4.xsd")
    input=read_sonnet_technology(joinpath(fixture,"input_resistivity.stf");schema_path=schema)
    exported=read_sonnet_technology(joinpath(fixture,"native_resistivity_export.stf"))
    sheetinput=read_sonnet_technology(joinpath(fixture,"input_sheet_variable.stf");schema_path=schema)
    sheetexport=read_sonnet_technology(joinpath(fixture,"native_sheet_variable_export.stf"))
    aliasinput=read_sonnet_technology(joinpath(fixture,"input_schema_sheet_alias.stf");schema_path=schema)
    aliasoutput=read_sonnet_technology(joinpath(fixture,"native_schema_sheet_alias_roundtrip.stf"))
    @test input.schema_status==:validated && sheetinput.schema_status==:validated
    @test exported.schema_status==:not_validated && sheetexport.schema_status==:not_validated
    # The actual editor writes MOSQ while its stated official1.4 XSD only
    # lists OHSQ|MOHSQ. Preserve exact exports and strict optional validation.
    for native in ("native_resistivity_export.stf","native_sheet_variable_export.stf")
        @test_throws ArgumentError read_sonnet_technology(joinpath(fixture,native);schema_path=schema)
    end
    @test input.units["runit"]=="OHUM" && exported.units["runit"]=="OHMM"
    @test sheetinput.units["srunit"]=="OHSQ" && sheetexport.units["srunit"]=="MOSQ"
    @test sonnet_technology_value(input,"Rho")==70000.
    @test sonnet_technology_value(exported,"Rho")==.07
    @test sonnet_technology_value(input,"Rho";quantity=:resistivity)≈.07
    @test sonnet_technology_value(exported,"Rho";quantity=:resistivity)==.07
    @test sonnet_technology_value(exported,"1";quantity=:resistivity)==1.
    @test sonnet_technology_material(input,"Substrate").sigma≈100/7
    @test sonnet_technology_material(exported,"Lower").sigma≈100/7
    @test sonnet_technology_material(exported,"Lower";variables=Dict("Rho"=>.14)).sigma≈50/7
    @test sonnet_technology_value(sheetinput,"Rs")==2.
    @test sonnet_technology_value(sheetexport,"Rs")==2000.
    @test sonnet_technology_value(sheetinput,"Rs";quantity=:sheet_resistance)==2.
    @test sonnet_technology_value(sheetexport,"Rs";quantity=:sheet_resistance)==2.
    @test sonnet_technology_value(sheetexport,"Rs";quantity=:sheet_resistance,variables=Dict("Rs"=>500.))==.5
    @test aliasinput.units["srunit"]=="MOHSQ" && aliasinput.schema_status==:validated
    @test aliasoutput.units["srunit"]=="MOSQ" && aliasoutput.schema_status==:not_validated
    @test sonnet_technology_value(aliasinput,"Rs")==sonnet_technology_value(aliasoutput,"Rs")==2000.
    @test sonnet_technology_value(aliasinput,"Rs";quantity=:sheet_resistance)==2.
    @test sonnet_technology_value(aliasoutput,"Rs";quantity=:sheet_resistance)==2.
    a=sonnet_technology_stack(input,1e9,.001,.001)
    b=sonnet_technology_stack(exported,1e9,.001,.001)
    @test all(k->a.layers[k].epsr≈b.layers[k].epsr,1:2)
    @test all(k->a.layers[k].thickness==b.layers[k].thickness,1:2)
    @test exported.root.attributes["writeable"]=="true"
    # Public scalar conductor providers use the independently proven units.
    mktempdir() do dir
        path=joinpath(dir,"providers.stf")
        write(path,_technology_test_xml(units="runit=\"OHMM\" srunit=\"MOSQ\""))
        t=read_sonnet_technology(path)
        @test sonnet_technology_material(t,"Film";kind=:conductor).sigma==40.
        @test sonnet_technology_material(t,"Sheet";kind=:conductor).resistance==.002
        @test sonnet_technology_material(t,"Sub").sigma≈1/70000
        for lexical in ("true","false","1","0")
            write(path,_technology_test_xml(rootattrs="writeable=\"$lexical\""))
            t=read_sonnet_technology(path)
            @test t.root.attributes["writeable"]==lexical
            @test sonnet_technology_stack(t,1e9,.001,.001).layers[1].epsr≈a.layers[1].epsr
        end
        for lexical in ("TRUE","False","yes","2","")
            write(path,_technology_test_xml(rootattrs="writeable=\"$lexical\""))
            @test_throws ArgumentError read_sonnet_technology(path)
        end
    end
    oldfailure=read(joinpath(fixture,"before_reader_failure.toml"),String)
    @test occursin("gate_pass = false",oldfailure) && occursin("rho_ratio = 0.001",oldfailure)
end

@testset "STF syntax, security and resource guards" begin
    mktempdir() do dir
        path=joinpath(dir,"test.stf");xml=_technology_test_xml();write(path,xml)
        for kw in ((;max_bytes=1),(;max_elements=1),(;max_depth=2),(;max_storage_bytes=1),(;max_table_nodes=2))
            @test_throws ArgumentError read_sonnet_technology(path;kw...)
        end
        for name in (:max_bytes,:max_elements,:max_depth,:max_storage_bytes,:max_table_nodes),value in (0,-1,true,1.5)
            @test_throws ArgumentError read_sonnet_technology(path;Dict(name=>value)...)
        end
        for bad in ("<!DOCTYPE technology_file SYSTEM \"https://invalid.example/a.dtd\">"*xml,
            "<!ENTITY x SYSTEM \"file:///secret\">"*xml,replace(xml,"<public>"=>"<?fetch https://invalid.example?><public>"),
            replace(xml,"<public>"=>"<public xmlns=\"urn:unknown\">"))
            write(path,bad);@test_throws ArgumentError read_sonnet_technology(path)
        end
        write(path,replace(xml,"</technology_file>"=>""))
        @test_throws ArgumentError read_sonnet_technology(path)
        write(path,UInt8[0xff,0xfe,0x00]);@test_throws ArgumentError read_sonnet_technology(path)
        for replacement in ("<column_keys>.95 .95</column_keys>","<column_keys></column_keys>")
            write(path,replace(xml,"<column_keys>.95</column_keys>"=>replacement))
            @test_throws ArgumentError read_sonnet_technology(path)
        end
        write(path,replace(xml,"<row key=\"1\">.0221</row>"=>"<row key=\".5\">.0221</row>"))
        @test_throws ArgumentError read_sonnet_technology(path)
        write(path,replace(xml,"<vector key=\".005\">5</vector>"=>"<vector key=\".005\">5 6</vector>"))
        @test_throws ArgumentError read_sonnet_technology(path)
        write(path,_technology_test_xml(variables="<var name=\"Cycle\" value=\"Cycle\"/>"))
        @test_throws ArgumentError sonnet_technology_value(read_sonnet_technology(path),"Cycle")
        write(path,_technology_test_xml(variables="<var name=\"H\" value=\"100\"/>"))
        @test_throws ArgumentError read_sonnet_technology(path)
        write(path,_technology_test_xml(extra="<privenc><enc>abcd</enc></privenc>",rootattrs="has_private=\"true\""))
        private=read_sonnet_technology(path)
        @test last(private.root.children).name=="privenc"
        @test_throws ArgumentError sonnet_technology_stack(private,1e9,.001,.002)
        for bad in (_technology_test_xml(rootattrs="unknownphysics=\"yes\""),
            _technology_test_xml(materials="<dielectric name=\"Bad\" anisotropic=\"yes\"><params axes=\"x\"/></dielectric>"),
            _technology_test_xml(stack="<diel name=\"Extra\" dielectric=\"Air\" thickness=\"10\"><via_techlayer name=\"V\" layer_name=\"V\" material=\"Copper\" from=\"0\" to=\"1\"/></diel>"))
            write(path,bad);retained=read_sonnet_technology(path)
            if haskey(retained.materials,(:dielectric,"Bad"))
                @test_throws ArgumentError sonnet_technology_material(retained,"Bad")
            else
                @test_throws ArgumentError sonnet_technology_stack(retained,1e9,.001,.002)
            end
        end
        # A metal technology layer inside a dielectric is valid grammar; it
        # does not change the dielectric stack the provider builds.
        write(path,_technology_test_xml(stack="<diel name=\"Extra\" dielectric=\"Air\" thickness=\"10\"><metal_techlayer name=\"M\" layer_name=\"M\" material=\"Copper\" thickness=\"1\"/></diel>"))
        @test sonnet_technology_stack(read_sonnet_technology(path),1e9,.001,.002) isa PlanarStackup
        # A separately supplied local schema is the sole validation source;
        # schemaLocation never triggers a network fetch.
        schema=joinpath(dir,"test.xsd")
        write(schema,"""<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
        <xs:element name="technology_file"><xs:complexType><xs:sequence><xs:any minOccurs="0" maxOccurs="unbounded" processContents="skip"/></xs:sequence><xs:anyAttribute processContents="skip"/></xs:complexType></xs:element></xs:schema>""")
        write(path,xml)
        @test read_sonnet_technology(path;schema_path=schema).schema_status==:validated
        @test read_sonnet_technology(path;schema_path=schema).schema_sha256==bytes2hex(sha256(read(schema)))
        write(schema,replace(read(schema,String),"<xs:element name=\"technology_file\">"=>"<xs:element name=\"different\">"))
        @test_throws ArgumentError read_sonnet_technology(path;schema_path=schema)
        write(path,xml)
        @test read_sonnet_technology(path).schema_status==:not_validated
        write(path,"<technology_file/>")
        # A large ceiling is a guard, not a requested allocation. In
        # particular maxInt must not wrap in ceiling+1 arithmetic.
        for ceiling in (1024,1024^2,8*1024^2,typemax(Int))
            DiffMoM._stf_read(path,ceiling)
            @test (@allocated DiffMoM._stf_read(path,ceiling))<65536
            @test last(DiffMoM._stf_read(path,ceiling))=="<technology_file/>"
        end
        write(path,xml)
        @test read_sonnet_technology(path;max_bytes=typemax(Int)).sha256==bytes2hex(sha256(xml))
        write(path,replace(xml,"version=\"1.0\""=>"version=\"1.0\" encoding=\"ISO-8859-1\"";count=1))
        @test_throws ArgumentError read_sonnet_technology(path)
        write(schema,"<xs:schema xmlns:xs=\"http://www.w3.org/2001/XMLSchema\"><xs:include schemaLocation=\"https://invalid.example/schema\"/></xs:schema>")
        @test_throws ArgumentError read_sonnet_technology(path;schema_path=schema)
    end
end

@testset "Native STF snapshots, static materialization and Lite restriction" begin
    dir=joinpath(@__DIR__,"..","validation","sonnet_stripline","native_technology_reference")
    manifest=JSON.parsefile(joinpath(dir,"provenance.json"))
    for row in manifest["files"]
        @test bytes2hex(sha256(read(joinpath(dir,row["path"]))))==row["sha256"]
    end
    linked=read_sonnet_linked_project(joinpath(dir,"linked.son"))
    @test linked.declared_technology=="mini.stf"
    @test linked.technology.raw==read(joinpath(dir,"mini.stf"),String)
    @test linked.sha256==bytes2hex(sha256(read(joinpath(dir,"linked.son"))))
    inline=read_sonnet_project(joinpath(dir,"inline.son"))
    materialized=sonnet_materialize_project(linked)
    @test parse.(Float64,materialized.layers[1][1:6])≈parse.(Float64,inline.layers[1][1:6])
    @test parse.(Float64,materialized.layers[2][1:6])≈parse.(Float64,inline.layers[2][1:6])
    @test materialized.layers[1][end]=="Upper"
    @test materialized.records===linked.records
    @test length(materialized.polygons)==length(inline.polygons)==1
    @test materialized.polygons[1].vertices==inline.polygons[1].vertices
    @test materialized.ports[1].records[1].tokens==inline.ports[1].records[1].tokens
    overriding=sonnet_materialize_project(read_sonnet_linked_project(joinpath(dir,"linked_serialized_cover10.son")))
    @test overriding.top==materialized.top && overriding.bottom==materialized.bottom
    native=planar_read_touchstone(joinpath(dir,"inline","native.s2p"))
    for (f,s) in zip(native.frequencies,native.s)
        prob=sonnet_planar_problem(materialized;freq=f)
        reference=sonnet_planar_problem(inline;freq=f)
        @test prob.sheets[1].mask==reference.sheets[1].mask
        solved=solve_sonnet_project(materialized,f;raw=true,mx=80,my=80,method=:dense_fft,max_bytes=100_000_000)
        independent=solve_sonnet_project(inline,f;raw=true,mx=80,my=80,method=:dense_fft,max_bytes=100_000_000)
        @test maximum(abs.(solved.s-independent.s))<1e-12
        @test maximum(abs.(solved.s-s))<.005
        strict=solve_planar_contracted(prob,f,Matrix{Float64}(I,2,2);
            method=:dense_fft,mx=80,my=80,max_bytes=100_000_000)
        @test maximum(strict.raw.relative_residuals)<=1e-9
        @test maximum(abs.(strict.s-s))<.005
    end
    for case in ("linked","linked_serialized_cover10")
        @test occursin("Sonnet Lite does not allow use of linked STF files.",read(joinpath(dir,case,"engine_stderr.log"),String))
    end
    mktempdir() do temp
        source=read(joinpath(dir,"linked.son"),String);path=joinpath(temp,"linked.son")
        write(joinpath(temp,"mini.stf"),read(joinpath(dir,"mini.stf")))
        for replacement in ("STF ../mini.stf","STF C:/elsewhere/mini.stf","STF absent.stf","STF mini.stf\nSTF mini.stf")
            write(path,replace(source,"STF mini.stf"=>replacement))
            @test_throws ArgumentError read_sonnet_linked_project(path)
        end
        write(path,source)
        @test_throws ArgumentError read_sonnet_linked_project(path;max_project_bytes=1)
        @test read_sonnet_linked_project(path;max_project_bytes=typemax(Int)).sha256==linked.sha256
        # Serialized literal materials are retained for polygons without a
        # resolved technology layer, so a literal material index is valid.
        write(path,replace(source,"0 5 -1 N"=>"0 5 0 N"))
        literal=sonnet_materialize_project(read_sonnet_linked_project(path))
        @test only(literal.polygons).material==0
    end
end

function _techlayer_test_stf(;metal_model="Normal",metal_extra="",via_model="Volume",
        via_extra="",metal_direction="",mesh_fill="Default",via_fill="Vertices",
        via_pads="",bias="",m1_spec="cond",m1_attrs="",via_spec="cond",via_attrs="",
        stack_extra="")
    return """<?xml version="1.0"?>
    <technology_file version="1700">
      <units cunit="SM" lunit="UM" runit="OHUM" srunit="OHSQ"/>
      <public>
        <variables/>
        <materials>
          <dielectric name="Air"><params erel="1"/></dielectric>
          <dielectric name="Ox"><params erel="4"/></dielectric>
          <dielectric name="Sub"><params erel="11.9"/></dielectric>
          <conductor name="M1Mat" condspec="$m1_spec" cond="4e7" rsvy="2e-5" shres="1.5" rpv="1.2" rdc="2" rrf="3e-7" xdc="0.01" ls="5" $m1_attrs/>
          <conductor name="Copper" condspec="cond" cond="4e7"/>
          <conductor name="ViaMat" condspec="$via_spec" cond="2.36e6" rsvy="2e-5" shres="1.5" rpv="1.2" rdc="0.03" rrf="2e-7" xdc="0.04" ls="5" $via_attrs/>
        </materials>
        <mesh_defs>
          <mesh fill_type="$mesh_fill" name="Default_metal" type="metal"/>
          <mesh fill_type="Default" name="Default_via" type="via"/>
          <mesh fill_type="$via_fill" name="Vertices_def" type="via" $via_pads/>
        </mesh_defs>
        <metal_model_defs>
          <metal_model model_type="Normal" name="Cover_def"/>
          <metal_model model_type="$metal_model" name="Thin_def" $metal_extra/>
          <metal_model model_type="ThickMetal" name="Thick_def" num_sheets="2"/>
          <via_model model_type="$via_model" name="Volume_def" $via_extra/>
        </metal_model_defs>
        <stackup>
          <TOP material="Lossless" model="Cover_def"/>
          <diel dielectric="Air" name="L1" thickness="30">
            <metal_techlayer name="M1" layer_name="M1" material="M1Mat" model="Thin_def" thickness="1" mesh="Default_metal" $metal_direction $bias/>
          </diel>
          <diel dielectric="Ox" name="L2" thickness="5">
            <metal_techlayer name="M2" layer_name="M2" material="Copper" model="Thick_def" thickness="4" mesh="Default_metal"/>
          </diel>
          <diel dielectric="Sub" name="L3" thickness="100"/>
          <BOTTOM material="Lossless" model="Cover_def"/>
          <vias>
            <via_techlayer name="V12" layer_name="V12" material="ViaMat" model="Volume_def" from="M1" to="M2" mesh="Vertices_def" $bias/>
          </vias>
        </stackup>
        $stack_extra
      </public>
    </technology_file>"""
end

function _techlayer_test_son(;via=true)
    viapoly=via ? """VIA POLYGON
1 5 -1 N 3 1 1 100 100 0 0 0 Y
TOLEVEL 0 RING NOCOVERS
TLAYNAM V12 INH
190 190
210 190
210 210
190 210
190 190
END
""" : ""
    npoly=via ? 5 : 4
    return """FTYP SONPROJ 19
DIM
ANG DEG
CAP PF
CON /OH
FREQ GHZ
IND NH
LNG UM
RES OH
END DIM
CONTROL
OPTIONS
SPEED 0
END CONTROL
GEO
STF rich.stf
TMET "SerializedCover" 0 SUP 0.0 0 0 0
BMET "SerializedCover" 0 SUP 0.0 0 0 0
MET "Legacy" 3 NOR 5e7 0.5 1
BOX 0 400 400 20 20 100 0
POR1 STD
POLY 1 1
1
2 50 0 0 0 400 200
POR1 STD
POLY 1 1
3
1 50 0 0 0 0 200
NUM $npoly
1 5 -1 N 1 1 1 100 100 0 0 0 Y
TLAYNAM M1 INH
0 180
400 180
400 220
0 220
0 180
END
0 5 -1 N 2 1 1 100 100 0 0 0 Y
TLAYNAM M2 INH
150 190
250 190
250 210
150 210
150 190
END
$(viapoly)0 5 0 N 4 1 1 100 100 0 0 0 Y
TLAYNAM M2 NOH
60 320
120 320
120 360
60 360
60 320
END
0 5 0 N 5 1 1 100 100 0 0 0 Y
TLAYNAM Bogus INH
280 320
340 320
340 360
280 360
280 320
END
END GEO
VARSWP
ENABLED Y
FREQ Y AN SWEEP 1 2 1
END
END VARSWP
FILEOUT
TOUCH ND Y rich.s2p IC 15 S RI R 50
FOLDER .
END FILEOUT
"""
end

@testset "Linked STF techlayer INH/NOH materialization and solve" begin
    mktempdir() do dir
        stf=_techlayer_test_stf();son=_techlayer_test_son()
        write(joinpath(dir,"rich.stf"),stf)
        path=joinpath(dir,"rich.son");write(path,son)
        linked=read_sonnet_linked_project(path)
        p=sonnet_materialize_project(linked)
        @test length(p.polygons)==5
        # INH inherits the technology layer's stack level, conductor and
        # model; the literal polygon level/material are overridden.
        @test p.polygons[1].level==0 && p.polygons[1].material==1
        @test p.polygons[1].technology=="M1" && p.polygons[1].technology_mode=="INH"
        @test p.polygons[2].level==1 && p.polygons[2].material==2
        # The via resolves from/to metal names to stack levels, overriding
        # the literal TOLEVEL, and inherits the via model's VOL material.
        @test p.polygons[3].kind===:via && p.polygons[3].level==0
        @test p.polygons[3].target=="1" && p.polygons[3].material==3
        # Under INH the techlayer mesh overrides the literal RING fill.
        @test "VERTICES" in p.polygons[3].flags && !("RING" in p.polygons[3].flags)
        @test "NOCOVERS" in p.polygons[3].flags
        # NOH inherits only the metal level; the literal material stays.
        @test p.polygons[4].level==1 && p.polygons[4].material==0
        @test p.polygons[4].technology_mode=="NOH"
        # An unresolved technology name keeps literal polygon data.
        @test p.polygons[5].level==0 && p.polygons[5].material==0
        @test p.polygons[5].technology=="Bogus"
        # Generated MET rows follow the inherited Normal/ThickMetal/Volume
        # model mapping with the STF conductor conductivity.
        @test p.metals[2][1:3]==["\"M1\"","1","NOR"] && p.metals[2][4]=="4.0e7"
        @test p.metals[3][1:3]==["\"M2\"","2","TMM"] && p.metals[3][4]=="4.0e7"
        @test p.metals[3][6]=="4.0" && p.metals[3][7]=="2"
        @test p.metals[4][1:3]==["\"V12\"","4","VOL"] && p.metals[4][4]=="2.36e6"
        @test p.metals[4][5:6]==["SOLID","0"]
        @test length(p.layers)==3
        @test parse(Float64,p.layers[1][1])≈30 && parse(Float64,p.layers[2][2])≈4
        @test p.records===linked.records
        # The inherited ThickMetal pad grows a physical face into the via's
        # span; the via lands on the grown face rather than colliding with
        # the generated face-wall connector.
        solved=solve_sonnet_project(p,2e9;raw=true,grid=(40,40),
            method=:dense_fft,max_bytes=200_000_000)
        @test all(isfinite,solved.s)
        @test abs(solved.s[2,1])>.9
        # Malformed or unsupported TLAYNAM tails reject when the polygon
        # section is parsed during materialization.
        for replacement in ("TLAYNAM M1","TLAYNAM M1 XXX","TLAYNAM M1 INH EXTRA")
            write(path,replace(son,"TLAYNAM M1 INH"=>replacement;count=1))
            @test_throws ArgumentError sonnet_materialize_project(read_sonnet_linked_project(path))
        end
        write(path,son)
        # Unsupported model and mesh semantics reject on inheritance.
        for kw in ((;metal_model="TrueVolume"),(;metal_model="RoughMetal"))
            write(joinpath(dir,"rich.stf"),_techlayer_test_stf(;kw...))
            err=try sonnet_materialize_project(read_sonnet_linked_project(path));nothing catch e;e end
            @test err isa ArgumentError && occursin("physical adapter",sprint(showerror,err))
        end
        # A Down direction is inert outside ThickMetal (native accepts and
        # ignores it), so the Normal M1 row materializes unchanged.
        write(joinpath(dir,"rich.stf"),_techlayer_test_stf(metal_direction="direction=\"Down\""))
        down=sonnet_materialize_project(read_sonnet_linked_project(path))
        @test down.metals[2][1:3]==["\"M1\"","1","NOR"]
        write(joinpath(dir,"rich.stf"),_techlayer_test_stf(mesh_fill="Ring"))
        @test_throws ArgumentError sonnet_materialize_project(read_sonnet_linked_project(path))
        write(joinpath(dir,"rich.stf"),_techlayer_test_stf(bias="bias=\"B\""))
        @test_throws ArgumentError sonnet_materialize_project(read_sonnet_linked_project(path))
        write(joinpath(dir,"rich.stf"),stf)
        # NOH on a via techlayer inherits only the endpoint pair: the literal
        # material (-1 here, the perfect-via sentinel) and flags stay.
        write(path,replace(son,"TLAYNAM V12 INH"=>"TLAYNAM V12 NOH"))
        nohvia=sonnet_materialize_project(read_sonnet_linked_project(path))
        @test nohvia.polygons[3].kind===:via && nohvia.polygons[3].level==0
        @test nohvia.polygons[3].target=="1" && nohvia.polygons[3].material==-1
        @test "RING" in nohvia.polygons[3].flags
        # Kind mismatches reject: metal layer on a via polygon and a via
        # layer on a sheet polygon.
        write(path,replace(son,"TLAYNAM M1 INH"=>"TLAYNAM V12 INH";count=1))
        @test_throws ArgumentError sonnet_materialize_project(read_sonnet_linked_project(path))
        write(path,replace(son,"TLAYNAM V12 INH"=>"TLAYNAM M1 INH"))
        @test_throws ArgumentError sonnet_materialize_project(read_sonnet_linked_project(path))
        write(path,son)
        # An ignored techlayer leaves literal polygon data untouched. A via
        # endpoint on an ignored layer still rejects, so this uses a
        # via-free project.
        write(joinpath(dir,"rich.stf"),_techlayer_test_stf(
            stack_extra=" ",metal_direction="ignore=\"true\""))
        write(path,_techlayer_test_son(via=false))
        ignored=sonnet_materialize_project(read_sonnet_linked_project(path))
        @test ignored.polygons[1].level==1 && ignored.polygons[1].material==-1
        write(joinpath(dir,"rich.stf"),stf)
        # A via polygon without a literal fill still receives the techlayer
        # mesh's fill and cover tokens under INH.
        write(path,replace(son,"TOLEVEL 0 RING NOCOVERS"=>"TOLEVEL 0 NOCOVERS"))
        meshed=sonnet_materialize_project(read_sonnet_linked_project(path))
        @test "VERTICES" in meshed.polygons[3].flags
        @test "NOCOVERS" in meshed.polygons[3].flags
        write(path,son)
    end
end

@testset "Linked STF via mesh flag precedence and pads" begin
    # `soncmd -LoadTechnology` on Lite 18.53: under INH the techlayer mesh
    # replaces the polygon's literal fill token (Full→SOLID, Default→RING)
    # and its pad control selects COVERS/NOCOVERS, for every literal flag
    # combination. Diagonal/Conformal fills are native via-mesh rejections.
    mktempdir() do dir
        son=_techlayer_test_son();path=joinpath(dir,"rich.son")
        fills=(("Ring","RING"),("Vertices","VERTICES"),("Center","CENTER"),
            ("Full","SOLID"),("Solid","SOLID"),("Bar","BAR"),("Default","RING"))
        literals=("TOLEVEL 0 BAR NOCOVERS","TOLEVEL 0 RING NOCOVERS",
            "TOLEVEL 0 VERTICES NOCOVERS","TOLEVEL 0 CENTER NOCOVERS",
            "TOLEVEL 0 SOLID NOCOVERS","TOLEVEL 0 NOCOVERS",
            "TOLEVEL 0 RING COVERS","TOLEVEL 0")
        for (fill,token) in fills,literal in literals
            write(joinpath(dir,"rich.stf"),_techlayer_test_stf(via_fill=fill))
            write(path,replace(son,"TOLEVEL 0 RING NOCOVERS"=>literal))
            flags=sonnet_materialize_project(read_sonnet_linked_project(path)).polygons[3].flags
            @test token in flags
            @test !any(f->f in ("SOLID","FULL","VERTICES","CENTER","BAR","RING") && f!=token,flags)
            @test "NOCOVERS" in flags && !("COVERS" in flags)
        end
        # pads="on" emits COVERS over any literal cover flag; without it the
        # materialized flag is NOCOVERS.
        write(joinpath(dir,"rich.stf"),_techlayer_test_stf(via_pads="pads=\"on\""))
        for literal in ("TOLEVEL 0 RING NOCOVERS","TOLEVEL 0 RING COVERS",
                "TOLEVEL 0 RING","TOLEVEL 0 NOCOVERS")
            write(path,replace(son,"TOLEVEL 0 RING NOCOVERS"=>literal))
            flags=sonnet_materialize_project(read_sonnet_linked_project(path)).polygons[3].flags
            @test "COVERS" in flags && !("NOCOVERS" in flags)
            @test "VERTICES" in flags
        end
        # Native rejects non-via fill types on via meshes.
        for fill in ("Diagonal","Conformal","Staircase")
            write(joinpath(dir,"rich.stf"),_techlayer_test_stf(via_fill=fill))
            write(path,son)
            @test_throws ArgumentError sonnet_materialize_project(read_sonnet_linked_project(path))
        end
        write(joinpath(dir,"rich.stf"),_techlayer_test_stf())
        write(path,son)
    end
end

@testset "Linked STF model/spec pairing materialization" begin
    # Pairing matrix decoded from `soncmd -LoadTechnology` on Lite 18.53:
    # each model reads only the conductor attributes its condspec exposes,
    # and pairs outside the accepted set are native parse rejections.
    mktempdir() do dir
        son=_techlayer_test_son();path=joinpath(dir,"rich.son")
        write(path,son)
        function emit(;kw...)
            write(joinpath(dir,"rich.stf"),_techlayer_test_stf(;kw...))
            return sonnet_materialize_project(read_sonnet_linked_project(path))
        end
        # Resistor+res emits the DC sheet resistance; under shres the rdc
        # field is not exposed and native emits a literal RES 0.
        @test emit(metal_model="Resistor",m1_spec="res").metals[2][3:4]==["RES","2.0"]
        @test emit(metal_model="Resistor",m1_spec="shres").metals[2][3:4]==["RES","0.0"]
        # Rdc/Rrf requires the rdcrrf spec and emits NAT.
        @test emit(metal_model="Rdc/Rrf",m1_spec="rdcrrf").metals[2][3:5]==["NAT","2.0","3.0e-7"]
        # General accepts general and rdcrrf (xdc/ls read as 0 under rdcrrf).
        @test emit(metal_model="General",m1_spec="general").metals[2][3:7]==
            ["SUP","2.0","3.0e-7","0.01","5.0"]
        @test emit(metal_model="General",m1_spec="rdcrrf").metals[2][3:7]==
            ["SUP","2.0","3.0e-7","0.0","0.0"]
        # Sense requires the sense spec and emits SEN xdc.
        @test emit(metal_model="Sense",m1_spec="sense").metals[2][3:4]==["SEN","0.01"]
        # Normal/ThickMetal carry the RSVY (ohm-cm) and SRVY (ohm/sq)
        # selectors; TMM always emits its selector slot.
        @test emit(m1_spec="rsvy").metals[2][3:end]==["NOR","2.0e-9","0.5","1.0","RSVY"]
        @test emit(m1_spec="shres").metals[2][3:end]==["NOR","1.5","0.5","1.0","SRVY"]
        @test emit(metal_model="ThickMetal",m1_spec="rsvy").metals[2][3:end]==
            ["TMM","2.0e-9","0.5","1.0","2","RSVY"]
        @test emit(metal_model="ThickMetal",m1_spec="shres").metals[2][3:end]==
            ["TMM","1.5","0.5","1.0","2","SRVY"]
        @test emit(metal_model="ThickMetal",metal_extra="num_sheets=\"1\"").metals[2][7]=="2"
        @test emit(metal_model="ThickMetal",metal_direction="direction=\"Down\"").metals[2][end]=="TDWN"
        # Native-invalid metal pairings reject like Lite's parser.
        for kw in ((;metal_model="Resistor"),(;metal_model="Rdc/Rrf",m1_spec="general"),
                (;metal_model="Sense",m1_spec="general"),(;m1_spec="res"),(;m1_spec="rpv"),
                (;metal_model="Resistor",m1_spec="rdcrrf"),(;metal_model="Rdc/Rrf"),
                (;metal_model="General"),(;metal_model="Sense"))
            write(joinpath(dir,"rich.stf"),_techlayer_test_stf(;kw...))
            @test_throws ArgumentError sonnet_materialize_project(read_sonnet_linked_project(path))
        end
        # Volume accepts cond/rsvy/shres/rpv on both Solid and Wall forms.
        wall="volume_type=\"Wall\" wall_thickness=\"0.3\""
        @test emit(via_spec="rsvy",via_extra=wall).metals[4][3:end]==
            ["VOL","2.0e-9","0.3","RSVY"]
        @test emit(via_spec="shres",via_extra=wall).metals[4][3:end]==
            ["VOL","1.5","0.3","SRVY"]
        @test emit(via_spec="rpv").metals[4][3:end]==["VOL","1.2","SOLID","0","RPV"]
        @test emit(via_spec="rpv",via_extra=wall).metals[4][3:end]==
            ["VOL","1.2","0.3","RPV"]
        # Array serializes fill_factor and the RPV density tail; shres is a
        # native rejection and via_density is inert without RPV.
        @test emit(via_model="Array",via_extra="fill_factor=\"40\"").metals[4][3:end]==
            ["ARR","2.36e6","40.0"]
        @test emit(via_model="Array",via_spec="rsvy",via_extra="fill_factor=\"40\"").metals[4][3:end]==
            ["ARR","2.0e-9","40.0","RSVY"]
        @test emit(via_model="Array",via_spec="rpv",via_extra="fill_factor=\"40\" via_density=\"2\"").metals[4][3:end]==
            ["ARR","1.2","100","RPV","2.0"]
        write(joinpath(dir,"rich.stf"),_techlayer_test_stf(via_model="Array",via_spec="shres"))
        @test_throws ArgumentError sonnet_materialize_project(read_sonnet_linked_project(path))
        write(joinpath(dir,"rich.stf"),_techlayer_test_stf(via_spec="res"))
        @test_throws ArgumentError sonnet_materialize_project(read_sonnet_linked_project(path))
        # Surface accepts every spec and reads only exposed fields:
        # cond produces SFC 0 0 0, sense only xdc, rdcrrf rdc+rrf, and
        # general/surface all three (native field order rdc/xdc/rrf).
        @test emit(via_model="Surface").metals[4][3:end]==["SFC","0.0","0.0","0.0"]
        @test emit(via_model="Surface",via_spec="sense").metals[4][3:end]==["SFC","0.0","0.04","0.0"]
        @test emit(via_model="Surface",via_spec="rdcrrf").metals[4][3:end]==["SFC","0.03","0.0","2.0e-7"]
        @test emit(via_model="Surface",via_spec="general").metals[4][3:end]==["SFC","0.03","0.04","2.0e-7"]
        @test emit(via_model="Surface",via_spec="surface").metals[4][3:end]==["SFC","0.03","0.04","2.0e-7"]
        @test emit(via_model="Surface",via_spec="res").metals[4][3:end]==["SFC","0.03","0.0","0.0"]
    end
end

@testset "Native SFC via conductivity and sheet impedance" begin
    p=DiffMoM.SonnetProject("t.son",Dict{String,String}(),1e-6,1e9,String[],
        Vector{String}[],Vector{String}[],String[],String[],DiffMoM.SonnetPolygon[],
        DiffMoM.SonnetPortSpec[],Dict{String,String}(),Vector{DiffMoM.SonnetRecord}[],
        DiffMoM.SonnetRecord[],DiffMoM.SonnetRecord[])
    f=2e9
    # Sheet-side SFC is the planar surface impedance Zs=transition+jXdc.
    zs=DiffMoM.sonnet_metal_zs(p,["\"F\"","0","SFC","0.03","0.04","2e-7"],f)
    ref=DiffMoM._sonnet_loss_transition(0.03,2e-7*sqrt(f))
    @test zs≈complex(real(ref),imag(ref)+0.04)
    @test DiffMoM.sonnet_metal_zs(p,["\"F\"","0","SFC","0","0","0"],f)==0.0im
    @test DiffMoM.sonnet_metal_zs(p,["\"F\"","0","SFC","0.03","0","0"],f)==complex(0.03)
    # Trailing tokens are inert, as on Lite 18.53.
    @test DiffMoM.sonnet_metal_zs(p,["\"F\"","0","SFC","0.03","0.04","2e-7","1"],f)==zs
    @test_throws ArgumentError DiffMoM.sonnet_metal_zs(p,["\"F\"","0","SFC","0.03"],f)
    @test_throws ArgumentError DiffMoM.sonnet_metal_zs(p,["\"F\"","0","SFC","-1","0","0"],f)
    # Via-side SFC: sigma spreads the tube conductance P/(Zs*h) over the
    # mask so that sigma*A_mesh/h = P/(Zs*h). A 50x50um tube on a 25um grid.
    verts=Matrix{Float64}(undef,2,4)
    verts[:,1].=[0,0];verts[:,2].=[50e-6,0];verts[:,3].=[50e-6,50e-6];verts[:,4].=[0,50e-6]
    poly=DiffMoM.SonnetPolygon(:via,0,0,1,verts,"1","","",String[])
    val(t)=DiffMoM.sonnet_variable_value(p,t;variables=Dict{String,Float64}(),freq=f)
    mask=falses(4,4);mask[2:3,2:3].=true
    grid=(dx=25e-6,dy=25e-6)
    sigma=DiffMoM._sonnet_surface_via_sigma(p,poly,grid,
        ["\"F\"","0","SFC","0.03","0.04","2e-7"],val,f,mask)
    mesh_area=4*grid.dx*grid.dy
    @test sigma*mesh_area≈200e-6/complex(real(ref),imag(ref)+0.04)
    # The all-zero record is a perfect (infinite-conductivity) tube.
    @test DiffMoM._sonnet_surface_via_sigma(p,poly,grid,
        ["\"F\"","0","SFC","0","0","0"],val,f,mask)==Inf
    # A purely reactive wall is a purely reactive column.
    sre=DiffMoM._sonnet_surface_via_sigma(p,poly,grid,
        ["\"F\"","0","SFC","0","0.04","0"],val,f,mask)
    @test iszero(real(sre)) && imag(sre)<0
    # SFC via is equivalent to a VOL RPV via at DC for the resistive part:
    # RPV resistance rdc*h/P gives the identical mask conductivity. The
    # project's metals[1] is the RPV record (poly.material=0).
    h=5e-6
    p3=DiffMoM.SonnetProject("t.son",Dict{String,String}(),1e-6,1e9,String[],Vector{String}[],
        [["\"V\"","0","VOL",string(0.03*h/200e-6),"SOLID","0","RPV"]],String[],String[],
        DiffMoM.SonnetPolygon[],DiffMoM.SonnetPortSpec[],Dict{String,String}(),
        Vector{DiffMoM.SonnetRecord}[],DiffMoM.SonnetRecord[],DiffMoM.SonnetRecord[])
    # The level-0 -> level-1 via spans stack.layers[L-1]; h is that layer's
    # thickness and the second layer pads the stack to L=2.
    stack=PlanarStackup([PlanarLayer(1.0+0im,1.0+0im,h),PlanarLayer(1.0+0im,1.0+0im,h)],
        PlanarTerminator(TERM_PEC),PlanarTerminator(TERM_PEC),1e-3,1e-3)
    σr=DiffMoM._sonnet_via_sigma(p3,poly,grid,stack,mask,f,Dict{String,Float64}())
    σd=DiffMoM._sonnet_surface_via_sigma(p,poly,grid,
        ["\"F\"","0","SFC","0.03","0","0"],val,f,mask)
    @test σr≈σd
end

@testset "Native via lands on grown thick-metal face" begin
    # A two-sheet TMM conductor expands its base plane plus a grown face.
    # A via ending on that level lands on the face nearest its column rather
    # than tunneling through the slab into the generated PEC connector.
    mktempdir() do dir
        son=replace(_techlayer_test_son(),"STF rich.stf\n"=>"";count=1)
        son=replace(son,"BOX 0 "=>"BOX 2 ";count=1)
        layers="""30.0 1.0 1.0 0.0 0.0 0.0 2 "L1"
5.0 4.0 1.0 0.0 0.0 0.0 2 "L2"
100.0 11.9 1.0 0.0 0.0 0.0 2 "L3"
"""
        son=replace(son,"BOX 2 400 400 20 20 100 0\n"=>
            "BOX 2 400 400 20 20 100 0\n"*layers;count=1)
        for replacement in ("TLAYNAM M1 INH\n","TLAYNAM M2 INH\n","TLAYNAM M2 NOH\n","TLAYNAM Bogus INH\n","TLAYNAM V12 INH\n")
            son=replace(son,replacement=>"";count=1)
        end
        son=replace(son,"TOLEVEL 0 RING NOCOVERS"=>"TOLEVEL 1 RING NOCOVERS")
        son=replace(son,"1 5 -1 N 1 "=>"0 5 1 N 1 ";count=1)
        son=replace(son,"0 5 -1 N 2 "=>"1 5 2 N 2 ";count=1)
        son=replace(son,"1 5 -1 N 3 "=>"0 5 3 N 3 ";count=1)
        son=replace(son,"""MET "Legacy" 3 NOR 5e7 0.5 1"""=>
            """MET "Legacy" 3 NOR 5e7 0.5 1
MET "M1" 1 NOR 4.0e7 0.5 1.0
MET "M2" 2 TMM 4.0e7 0.5 4.0 2
MET "V12" 4 VOL 2.36e6 SOLID 0""")
        path=joinpath(dir,"literal.son");write(path,son)
        p=read_sonnet_project(path)
        solved=solve_sonnet_project(p,2e9;raw=true,grid=(40,40),
            method=:dense_fft,max_bytes=200_000_000)
        @test all(isfinite,solved.s)
        @test abs(solved.s[2,1])>.9
        expanded=DiffMoM._sonnet_thick_geometry(p,2e9,Dict{String,Float64}())
        via=only(filter(q->q.kind===:via && q.id==3,expanded.polygons))
        @test via.level==0 && via.target=="1"
        # The face-wall connector spans the slab band below the via's stop.
        connector=only(filter(q->q.kind===:via && q.id!=3,expanded.polygons))
        @test connector.level==2 && connector.target=="1" && connector.material==-1
    end
end

@testset "STF exact lookup rejects nonzero coordinate underflow" begin
    tiny=(BigFloat("1e-400"),-BigFloat("1e-400"),BigInt(1)//big(10)^400,-BigInt(1)//big(10)^400)
    smallest=nextfloat(0.)
    one=SonnetTechnologyTable(Dict("value_name"=>"rho"),["width"],
        Dict{Tuple,Vector{Float64}}((0.,)=>[1.],(-0.,)=>[2.],(1.,)=>[3.],(smallest,)=>[4.]))
    two=SonnetTechnologyTable(Dict("value_name"=>"etch"),["width","density"],
        Dict{Tuple,Vector{Float64}}((0.,1.)=>[1.],(-0.,1.)=>[2.],(1.,1.)=>[3.],
            (smallest,1.)=>[4.],(1.,0.)=>[5.],(1.,-0.)=>[6.],(1.,smallest)=>[7.]))
    for (table,coordinates,expected) in ((one,x->(x,),[1.,2.,3.,4.]),
        (two,x->(x,1.),[1.,2.,3.,4.]),(two,x->(1.,x),[5.,6.,3.,7.]))
        original=deepcopy(table.nodes)
        for (coordinate,value) in zip((0.,-0.,1.,smallest),expected)
            @test sonnet_technology_lookup(table,coordinates(coordinate)...)==[value]
        end
        for coordinate in tiny
            @test isfinite(coordinate) && !iszero(coordinate)
            @test iszero(Float64(coordinate))
            @test_throws ArgumentError sonnet_technology_lookup(table,coordinates(coordinate)...)
        end
        for coordinate in (Inf,NaN,BigFloat("1e400"))
            @test_throws ArgumentError sonnet_technology_lookup(table,coordinates(coordinate)...)
        end
        @test sonnet_technology_lookup(table,coordinates(BigFloat(1))...)==[3.]
        @test table.nodes==original
    end
    @test_throws ArgumentError sonnet_technology_lookup(one,.5)
    @test_throws ArgumentError sonnet_technology_lookup(two,0.)
    copyvalue=sonnet_technology_lookup(one,0.);copyvalue[1]=99.
    @test sonnet_technology_lookup(one,0.)==[1.]
end
