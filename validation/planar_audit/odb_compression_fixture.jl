# Independent UNIX-compress transport fixture producer: Windows bsdtar/libarchive.
# The reader is tested against an independently produced uncompressed tar hash.
using Random,SHA,TOML
using DiffMoM
output=joinpath(@__DIR__,"..","..","test","fixtures","odb_compress")
mkpath(output)
mktempdir() do directory
    root=joinpath(directory,"product")
    function entity(parts,text)
        path=joinpath(root,parts...);mkpath(dirname(path));write(path,text)
    end
    entity(["matrix","matrix"],"STEP {\nCOL=1\nNAME=board\n}\nLAYER {\nROW=1\nNAME=top\nTYPE=SIGNAL\nPOLARITY=POSITIVE\n}\n")
    noise=rand(MersenneTwister(918),UInt8(33):UInt8(126),350000)
    comments=IOBuffer()
    for i in 1:80:length(noise)
        write(comments,'#');write(comments,view(noise,i:min(i+79,length(noise))));write(comments,'\n')
    end
    info="UNITS=MM\n"*String(take!(comments))*repeat("# repeated compressible manufacturing metadata\n",10000)
    entity(["misc","info"],info)
    entity(["steps","board","stephdr"],"UNITS=MM\nX_DATUM=0\nY_DATUM=0\n")
    entity(["steps","board","layers","top","features"],"UNITS=MM\nF 1\n\$0 r1000\nP 1 1 0 P 0 0\n")
    plain=joinpath(directory,"product.tar");compressed=joinpath(output,"product.tar.Z")
    run(`tar.exe --format=ustar -cf $plain -C $directory product`)
    run(`tar.exe --format=ustar -cZf $compressed -C $directory product`)
    version=read(`tar.exe --version`,String)
    expected=bytes2hex(sha256(read(plain)))
    actual=DiffMoM._with_odb_entity(compressed,32*1024^2) do expanded
        bytes2hex(sha256(read(expanded)))
    end
    expected==actual || error("independent raw tar and decoded compress tar differ")
    open(joinpath(output,"reference.toml"),"w") do io
        TOML.print(io,Dict("producer"=>strip(version),"julia_version"=>string(VERSION),
            "raw_tar_sha256"=>expected,"compressed_sha256"=>bytes2hex(sha256(read(compressed))),
            "info_sha256"=>bytes2hex(sha256(info)),"uncompressed_tar_bytes"=>filesize(plain),
            "compressed_bytes"=>filesize(compressed),"seed"=>918,
            "scope"=>"independent libarchive LZW transport, not a vendor manufacturing corpus"))
    end
    doc=read_odb(compressed;step="board",layers=["top"])
    length(doc.objects)==1 || error("compressed product lowering lost pad")
end
