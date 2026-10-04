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
            _technology_test_xml(stack="<diel name=\"Extra\" dielectric=\"Air\" thickness=\"10\"><metal_techlayer name=\"M\" layer_name=\"M\" material=\"Copper\" thickness=\"1\"/></diel>"))
            write(path,bad);retained=read_sonnet_technology(path)
            if haskey(retained.materials,(:dielectric,"Bad"))
                @test_throws ArgumentError sonnet_technology_material(retained,"Bad")
            else
                @test_throws ArgumentError sonnet_technology_stack(retained,1e9,.001,.002)
            end
        end
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
        write(path,replace(source,"0 5 -1 N"=>"0 5 0 N"))
        @test_throws ArgumentError sonnet_materialize_project(read_sonnet_linked_project(path))
    end
end
