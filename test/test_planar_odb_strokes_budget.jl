using DiffMoM,Test

function _odb_stroke_budget_read(body;kwargs...)
    mktempdir() do directory
        path=joinpath(directory,"features");write(path,body)
        read_odb_features(path;layer="metal",kwargs...)
    end
end

# Independently clip the translated query path against each Cartesian
# rectangle of the original U. This oracle has no polygon crossing test.
function _odb_rectangle_line_member(rect,a,b,x,y)
    lower,upper=0.,1.
    for (q,d,lo,hi) in ((x-a[1],b[1]-a[1],rect[1],rect[2]),
                       (y-a[2],b[2]-a[2],rect[3],rect[4]))
        if d==0
            lo<=q<=hi || return false
        else
            t1,t2=(q-lo)/d,(q-hi)/d
            lower=max(lower,min(t1,t2));upper=min(upper,max(t1,t2))
            lower<=upper || return false
        end
    end
    return true
end

@testset "ODB exact concave polygon strokes retain aperture notches" begin
    DM=DiffMoM
    vertices=[(-1.,-1.),(1.,-1.),(1.,1.),(.5,1.),(.5,0.),(-.5,0.),(-.5,1.),(-1.,1.)]
    rectangles=((-1.,1.,-1.,0.),(-1.,-.5,0.,1.),(.5,1.,0.,1.))
    for stop in ((0.,0.),(.1,0.),(0.,.1),(2.,0.),(0.,2.),(.2,.3),(-.2,.3),(3.,-2.)),
            rev in (false,true),cw in (false,true)
        points=cw ? reverse(vertices) : vertices
        polygon=DM._artwork_polygon([(x*1e-3,y*1e-3) for (x,y) in points])
        a=(.125,-.375)
        start,finish=rev ? (stop,a) : (a,stop)
        body="UNITS=MM\n\$0 custom\nL $(start[1]) $(start[2]) $(finish[1]) $(finish[2]) 0 P 0\n"
        shape=only(_odb_stroke_budget_read(body;symbol_resolver=name->polygon).objects).shape
        @test DM._artwork_bounds(shape)==(-.001+min(start[1],finish[1])*1e-3,
            .001+max(start[1],finish[1])*1e-3,-.001+min(start[2],finish[2])*1e-3,
            .001+max(start[2],finish[2])*1e-3)
        for x in range(-2.013,4.013;length=79),y in range(-3.017,3.017;length=77)
            expected=any(r->_odb_rectangle_line_member(r,start,finish,x,y),rectangles)
            @test DM._artwork_contains(shape,x*1e-3,y*1e-3)==expected
        end
    end
    polygon=DM._artwork_polygon([(x*1e-3,y*1e-3) for (x,y) in vertices])
    shape=only(_odb_stroke_budget_read("UNITS=MM\n\$0 custom\nL 0 0 .1 0 0 P 0\n";
        symbol_resolver=name->polygon).objects).shape
    @test !DM._artwork_contains(shape,0.,.5e-3)
    @test DM._artwork_contains(shape,.6e-3,.5e-3)
    @test DM._artwork_contains(shape,0.,0.)
    empty_path=only(_odb_stroke_budget_read("UNITS=MM\n\$0 custom\nL 0 0 0 0 0 P 0\n";
        symbol_resolver=name->polygon).objects).shape
    for (x,y) in ((0.,.5),(.5,.5),(1.,1.),(0.,0.),(-1.,-1.),(2.,2.))
        @test DM._artwork_contains(empty_path,x*1e-3,y*1e-3)==DM._artwork_contains(polygon,x*1e-3,y*1e-3)
    end
    member()=DM._artwork_contains(shape,.131e-3,.227e-3)
    member()
    @test (@allocated member())==0
end

@testset "ODB feature lookup and attribute payload is reserved" begin
    n=2000
    definitions="UNITS=MM\n"*join(("\$$(i-1) hplate4000x2000x1000x100x100\n" for i in 1:n))
    limit=3ncodeunits(definitions)+1024
    # Even numeric segment storage alone exceeds this valid input budget.
    stencil=DiffMoM._odb_standard_symbol("hplate4000x2000x1000x100x100",1e-6)
    @test n*length(stencil.segments)*Base.elsize(stencil.segments)>limit
    @test_throws ArgumentError _odb_stroke_budget_read(definitions;max_bytes=limit)
    @test isempty(_odb_stroke_budget_read(definitions;max_bytes=8*1024^2).objects)
    for marker in ('@','&')
        text="UNITS=MM\n"*join(("$(marker)$(i-1) value$(i)\n" for i in 1:n))
        @test_throws ArgumentError _odb_stroke_budget_read(text;max_bytes=3ncodeunits(text)+1024)
        @test isempty(_odb_stroke_budget_read(text;max_bytes=1024^2).objects)
    end
    calls=Ref(0)
    body="UNITS=MM\n\$0 custom\n"
    resolver=name->(calls[]+=1;DiffMoM._ArtworkEmpty())
    @test_throws ArgumentError _odb_stroke_budget_read(body;max_bytes=3ncodeunits(body)+1024+255,
        symbol_resolver=resolver)
    @test calls[]==0
    @test isempty(_odb_stroke_budget_read(body;max_bytes=4096,symbol_resolver=resolver).objects)
    @test calls[]==1
    @test_throws ArgumentError _odb_stroke_budget_read("UNITS=MM\nID=42\n";max_bytes=3ncodeunits("UNITS=MM\nID=42\n")+1024)
    @test _odb_stroke_budget_read("UNITS=MM\nID=42\n").attributes["ID"]==["42"]
    text="UNITS=MM\n\$0 r1000\n@0 .net_name\n&0 signal\nP 0 0 0 P 0 0;0=0;ID=42\n"
    observed=Int[]
    doc=_odb_stroke_budget_read(text;_payload_observer=p->push!(observed,p))
    @test issorted(observed)
    @test last(observed)>=3ncodeunits(text)+1024+256+Base.summarysize(only(doc.objects).attributes)
    @test only(doc.objects).attributes[".net_name.text_lookup"]==["signal"]
    @test_throws ArgumentError _odb_stroke_budget_read(text;max_bytes=last(observed)-1)
    @test length(_odb_stroke_budget_read(text;max_bytes=last(observed)).objects)==1
    # External recursive readers receive each reservation before returning
    # more retained data, through the existing active-payload observer.
    current=Ref(0)
    @test_throws ArgumentError _odb_stroke_budget_read("UNITS=MM\n\$0 r1000\n";
        _payload_observer=p->(current[]=p),_remaining_budget=()->current[]<1000 ? 10000 : current[])
end

function _odb_lookup_rejection_allocations(path,limit)
    try
        read_odb_features(path;max_bytes=limit)
    catch error
        error isa ArgumentError || rethrow()
        return true
    end
    return false
end
@testset "ODB rejected unused symbols do not allocate their geometry" begin
    mktempdir() do directory
        source="UNITS=MM\n"*join(("\$$(i-1) hplate4000x2000x1000x100x100\n" for i in 1:2000))
        path=joinpath(directory,"features");write(path,source)
        limit=3ncodeunits(source)+1024
        @test _odb_lookup_rejection_allocations(path,limit)
        @test (@allocated _odb_lookup_rejection_allocations(path,limit))<100_000
    end
end

@testset "ODB source snapshot is bounded and stable across callbacks" begin
    mktempdir() do directory
        path=joinpath(directory,"features")
        source="UNITS=MM\n\$0 custom\nP 1 2 0 P 0 0\n"
        for change in (:append,:truncate,:replace,:remove)
            write(path,source)
            calls=Ref(0)
            resolver=function(name)
                calls[]+=1
                if change==:append
                    open(path,"a") do io;write(io,"P 3 4 0 P 0 0\n");end
                elseif change==:truncate
                    write(path,"")
                elseif change==:replace
                    write(path,"invalid replacement data\n")
                else
                    rm(path)
                end
                DiffMoM._ArtworkCircle((0.,0.),.001)
            end
            doc=read_odb_features(path;symbol_resolver=resolver)
            @test calls[]==1
            @test length(doc.objects)==1
            @test DiffMoM._artwork_contains(only(doc.objects).shape,.001,.002)
        end
        write(path,source)
        raw,payload=DiffMoM._odb_feature_source(path,3ncodeunits(source))
        @test raw==collect(codeunits(source))
        @test payload==3ncodeunits(source)
        @test_throws ArgumentError DiffMoM._odb_feature_source(path,3ncodeunits(source)-1)
        @test DiffMoM._odb_feature_source(path,typemax(Int))[1]==collect(codeunits(source))
        write(path,UInt8[0xff,0xfe])
        @test_throws ArgumentError read_odb_features(path)
        write(path,"")
        @test isempty(read_odb_features(path;max_bytes=1).objects)
        # Appended bytes are caller-owned. The measured operation includes
        # callback IO and the reader, excluding source/comment construction.
        comment="#"*repeat("x",2*1024^2)*"\nF 0\n"
        body="UNITS=MM\n\$0 custom\n"
        resolver=name->begin
            open(path,"a") do io;write(io,comment);end
            DiffMoM._ArtworkEmpty()
        end
        write(path,body);read_odb_features(path;max_bytes=4096,symbol_resolver=resolver)
        write(path,body)
        @test (@allocated read_odb_features(path;max_bytes=4096,symbol_resolver=resolver))<100_000
    end
end
