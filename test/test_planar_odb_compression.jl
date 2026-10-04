using DiffMoM,Test,SHA,TOML
import CodecZlib: GzipCompressorStream
import Tar: Tar

# Independent literal-only LZW encoder. Every emitted code is a byte, so
# the dictionary contents do not enter this packing oracle.
function _test_unix_compress_literals(bytes;maxbits=16,block=true,clear_every=0)
    output=UInt8[0x1f,0x9d,UInt8(maxbits)|(block ? 0x80 : 0x00)]
    width=9;nextcode=block ? 257 : 256;previous=false;bits=UInt128(0);used=0;sinceclear=0
    function flushgroup(full)
        for i in 0:(full ? width : cld(used,8))-1
            push!(output,UInt8((bits>>(8i)) & 0xff))
        end
        bits=UInt128(0);used=0
    end
    function code(value)
        bits|=UInt128(value)<<used;used+=width
        used==8width && flushgroup(true)
    end
    for byte in bytes
        if width<maxbits && nextcode>(1<<width)-1
            used>0 && flushgroup(true);width+=1
        end
        if block && clear_every>0 && sinceclear==clear_every
            code(256);used>0 && flushgroup(true)
            width=9;nextcode=257;previous=false;sinceclear=0
        end
        code(byte)
        previous && nextcode<(1<<maxbits) && (nextcode+=1)
        previous=true;sinceclear+=1
    end
    used>0 && flushgroup(false)
    return output
end

@testset "ODB bounded UNIX compress packing and dictionaries" begin
    for maxbits in (9,10,12,16),block in (false,true),clear in (0,301)
        source=UInt8[mod(i*73+19,256) for i in 0:70000]
        compressed=_test_unix_compress_literals(source;maxbits,block,clear_every=clear)
        destination=IOBuffer()
        @test DiffMoM._odb_uncompress(IOBuffer(compressed),destination,2*1024^2)==length(source)
        @test take!(destination)==source
    end
    # KwKwK: literal A, dictionary entry 257 expands to AA.
    stream=UInt8[0x1f,0x9d,0x90,0x41,0x02,0x02]
    destination=IOBuffer();DiffMoM._odb_uncompress(IOBuffer(stream),destination,2*1024^2)
    @test String(take!(destination))=="AAA"
    for source in (UInt8[],UInt8[0x1f,0x9d],UInt8[0x1f,0x9d,0x88],
            UInt8[0x1f,0x9d,0xf0],UInt8[0x1f,0x9d,0x90,0xff,0x01])
        @test_throws ArgumentError DiffMoM._odb_uncompress(IOBuffer(source),IOBuffer(),2*1024^2)
    end
    @test_throws ArgumentError DiffMoM._odb_uncompress(IOBuffer(stream),IOBuffer(),10)
    repeated=_test_unix_compress_literals(fill(UInt8('A'),700000))
    @test_throws ArgumentError DiffMoM._odb_uncompress(IOBuffer(repeated),IOBuffer(),400000)
end

function _test_odb_gzip(path,text)
    open(path,"w") do file
        compressor=GzipCompressorStream(file)
        try;write(compressor,text);finally;close(compressor);end
    end
end

@testset "ODB compressed entities and archive geometry identity" begin
    mktempdir() do directory
        root=joinpath(directory,"product")
        function entity(parts,text;suffix=".gz")
            path=joinpath(root,parts...);mkpath(dirname(path))
            if suffix==".gz";_test_odb_gzip(path*suffix,text)
            else;write(path*suffix,_test_unix_compress_literals(codeunits(text)));end
            path
        end
        entity(["matrix","matrix"],"STEP {\nCOL=1\nNAME=board\n}\nLAYER {\nROW=1\nNAME=top\nTYPE=SIGNAL\nPOLARITY=POSITIVE\n}\n")
        entity(["misc","info"],"UNITS=MM\n";suffix=".Z")
        entity(["steps","board","stephdr"],"UNITS=MM\nX_DATUM=0\nY_DATUM=0\n")
        features=entity(["steps","board","layers","top","features"],"UNITS=MM\n\$0 r1000\nP 1 1 0 P 0 0\n";suffix=".Z")
        direct=read_odb_features(features)
        @test length(direct.objects)==1
        @test direct.source==features
        doc=read_odb(root)
        @test length(doc.objects)==1
        @test DiffMoM._artwork_contains(only(doc.objects).shape,.001,.001)
        tarball=joinpath(directory,"product.tar");Tar.create(root,tarball)
        archived=read_odb(tarball)
        @test archived.source==tarball
        @test only(archived.objects).bounds==only(doc.objects).bounds
        gzip=tarball*".gz";_test_odb_gzip(gzip,read(tarball))
        zipped=read_odb(gzip)
        @test only(zipped.objects).bounds==only(doc.objects).bounds
        for suffix in (".tgz",".TGZ")
            tgz=joinpath(directory,"product"*suffix);cp(gzip,tgz;force=true)
            aliased=read_odb(tgz)
            @test aliased.source==tgz
            @test only(aliased.objects).bounds==only(doc.objects).bounds
            @test_throws ArgumentError read_odb(tgz;max_entities=2)
            @test_throws ArgumentError read_odb(tgz;max_bytes=100)
        end
        @test_throws ArgumentError read_odb(gzip;max_bytes=100)
        @test_throws ArgumentError read_odb(gzip;max_entities=2)
        write(features,"UNITS=MM")
        @test_throws ArgumentError read_odb_features(features) # ambiguous variants
        rm(features)
        broken=joinpath(directory,"broken.gz");write(broken,UInt8[0x1f,0x8b,0x08])
        @test_throws Exception read_odb_features(broken)
        bomb=joinpath(directory,"bomb.gz");_test_odb_gzip(bomb,repeat("# comment\n",100000))
        @test_throws ArgumentError read_odb_features(bomb;max_bytes=100000)
    end
    fixture=joinpath(@__DIR__,"fixtures","odb_compress","product.tar.Z")
    reference=TOML.parsefile(joinpath(dirname(fixture),"reference.toml"))
    @test bytes2hex(sha256(read(fixture)))==reference["compressed_sha256"]
    DiffMoM._with_odb_entity(fixture,32*1024^2) do expanded
        @test bytes2hex(sha256(read(expanded)))==reference["raw_tar_sha256"]
    end
    compressed=read(fixture)
    decode_to_null()=DiffMoM._odb_uncompress(IOBuffer(compressed),devnull,32*1024^2)
    decode_to_null()
    @test (@allocated decode_to_null())<500000 # fixed dictionary/buffer payload, no per-code BigInts
    nativepacked=read_odb(fixture;step="board",layers=["top"])
    @test length(nativepacked.objects)==1
    @test collect(only(nativepacked.objects).bounds)≈[.0005,.0015,.0005,.0015]
end

function _test_odb_tar(path,entries)
    open(path,"w") do io
        for (name,type,body) in entries
            header=zeros(UInt8,512)
            field(offset,value)=copyto!(header,offset+1,codeunits(value),1,ncodeunits(value))
            field(0,name);field(100,"0000644\0");field(124,string(length(body);base=8,pad=11)*"\0")
            field(148,"        ");header[157]=UInt8(type)
            field(257,"ustar\0");field(263,"00")
            field(148,string(sum(Int,header);base=8,pad=6)*"\0 ")
            write(io,header);write(io,body);write(io,zeros(UInt8,mod(-length(body),512)))
        end
        write(io,zeros(UInt8,1024))
    end
end
@testset "ODB archive path/link/duplicate and root contracts" begin
    mktempdir() do directory
        path=joinpath(directory,"bad.tar")
        for name in ("../escape","/absolute","C:/escape","..\\escape")
            _test_odb_tar(path,[(name,'0',"bad")])
            @test_throws Exception read_odb(path)
        end
        for type in ('1','2','3','6')
            _test_odb_tar(path,[("link",type,"")])
            @test_throws Exception read_odb(path)
        end
        _test_odb_tar(path,[("x",'0',"a"),("x",'0',"b")])
        @test_throws ArgumentError read_odb(path)
        if Sys.iswindows()
            _test_odb_tar(path,[("A",'0',"a"),("a",'0',"b")])
            @test_throws ArgumentError read_odb(path)
        end
        _test_odb_tar(path,[("unrelated",'0',"a")])
        @test_throws ArgumentError read_odb(path)
        _test_odb_tar(path,[("a/matrix/matrix",'0',""),("b/matrix/matrix",'0',"")])
        @test_throws ArgumentError read_odb(path)
    end
end
