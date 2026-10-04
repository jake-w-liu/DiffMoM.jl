# Bounded ODB++ transport readers. UNIX compress uses LSB-first LZW codes
# in groups of eight; width changes and CLEAR finish the current group.
# Packing reference: https://github.com/vapier/ncompress (public domain).
import CodecZlib: GzipDecompressorStream
import Tar: Tar

_odb_gzip_entity(path)=endswith(path,".gz") || endswith(lowercase(path),".tgz")
_odb_compressed_entity(path)=_odb_gzip_entity(path) || endswith(path,".Z")

function _odb_entity(path)
    suffix=_odb_compressed_entity(path)
    candidates=suffix ? [String(path)] : [String(path),String(path)*".gz",String(path)*".Z"]
    present=filter(isfile,candidates)
    length(present)==1 || throw(ArgumentError(isempty(present) ? "missing ODB entity $path" :
        "ambiguous ODB entity: both plain/compressed variants exist for $path"))
    return only(present)
end
_odb_has_entity(path)=any(isfile,(path,path*".gz",path*".Z"))

function _odb_copy_bounded(input,output,limit;chunk_bytes=65536)
    buffer=Vector{UInt8}(undef,min(chunk_bytes,limit))
    isempty(buffer) && throw(ArgumentError("max_bytes cannot hold decompression buffer"))
    total=0
    while !eof(input)
        count=readbytes!(input,buffer,length(buffer))
        count==0 && break
        count<=limit-total || throw(ArgumentError("ODB decompressed entity exceeds max_bytes=$limit"))
        total+=count
        write(output,view(buffer,1:count))
    end
    return total
end

function _odb_uncompress(input,output,limit)
    header=read(input,3)
    length(header)==3 && header[1]==0x1f && header[2]==0x9d ||
        throw(ArgumentError("invalid UNIX compress header"))
    flags=header[3];maxbits=Int(flags & 0x1f)
    9<=maxbits<=16 && flags & 0x60==0 || throw(ArgumentError("invalid UNIX compress code width/flags"))
    capacity=1<<maxbits
    scratch=_checked_payload_sum("UNIX compress decoder",4capacity,65536,16)
    _enforce_payload_limit(scratch,limit,"UNIX compress decoder scratch","max_bytes")
    prefix=Vector{UInt16}(undef,capacity);suffix=Vector{UInt8}(undef,capacity)
    stack=Vector{UInt8}(undef,capacity);buffer=Vector{UInt8}(undef,65536)
    group=Vector{UInt8}(undef,16);groupbits=0;bitoffset=0
    block=flags & 0x80!=0;width=9;nextcode=block ? 257 : 256
    previous=-1;firstbyte=UInt8(0);bufferused=0;total=0
    for i in 0:255;suffix[i+1]=UInt8(i);end
    while true
        if width<maxbits && nextcode>(1<<width)-1
            width+=1;bitoffset=0;groupbits=0
        end
        if bitoffset+width>groupbits
            count=readbytes!(input,group,width)
            count==0 && break
            groupbits=8count;bitoffset=0
            groupbits>=width || throw(ArgumentError("truncated UNIX compress code"))
        end
        # Up to three bytes contain a code. The last group can be short.
        byteindex=bitoffset÷8+1;shift=bitoffset%8;bits=UInt32(0)
        for j in 0:2
            byteindex+j<=groupbits÷8 && (bits |= UInt32(group[byteindex+j])<<(8j))
        end
        code=Int((bits>>shift)&UInt32((1<<width)-1));bitoffset+=width
        if previous<0
            code<256 || throw(ArgumentError("UNIX compress stream must begin with a literal"))
            stack[1]=UInt8(code);used=1;firstbyte=UInt8(code)
        elseif block && code==256
            width=9;nextcode=257;previous=-1;bitoffset=0;groupbits=0
            continue
        else
            code<=nextcode && code<capacity || throw(ArgumentError("invalid UNIX compress dictionary code"))
            current=code;used=0
            if current==nextcode
                used=1;stack[used]=firstbyte;current=previous
            end
            while current>=256
                current<nextcode && used<capacity-1 || throw(ArgumentError("invalid UNIX compress dictionary chain"))
                used+=1;stack[used]=suffix[current+1];parent=Int(prefix[current+1])
                parent<current || throw(ArgumentError("cyclic UNIX compress dictionary chain"))
                current=parent
            end
            used+=1;firstbyte=UInt8(current);stack[used]=firstbyte
            if nextcode<capacity
                prefix[nextcode+1]=UInt16(previous);suffix[nextcode+1]=firstbyte;nextcode+=1
            end
        end
        # total is already <= limit. Subtraction checks expansion before
        # adding, avoiding overflow and BigInt allocation per decoded code.
        used<=limit-total || throw(ArgumentError("ODB decompressed entity exceeds max_bytes=$limit"))
        total+=used
        for i in used:-1:1
            bufferused+=1;buffer[bufferused]=stack[i]
            if bufferused==length(buffer)
                write(output,buffer);bufferused=0
            end
        end
        previous=code
        # An EOF group's unused bits are padding, not another partial code.
        if bitoffset+width>groupbits && eof(input);break;end
    end
    bufferused>0 && write(output,view(buffer,1:bufferused))
    return total
end

function _with_odb_entity(callback,path,limit)
    entity=_odb_entity(path)
    if !_odb_compressed_entity(entity)
        _enforce_payload_limit(filesize(entity),limit,"ODB entity input","max_bytes")
        return callback(entity)
    end
    _enforce_payload_limit(filesize(entity),limit,"ODB compressed input","max_bytes")
    return mktempdir() do directory
        expanded=joinpath(directory,"entity")
        open(entity,"r") do input
            open(expanded,"w") do output
                if endswith(entity,".Z")
                    _odb_uncompress(input,output,limit)
                else
                    _enforce_payload_limit(131072,limit,"ODB gzip stream/copy scratch","max_bytes")
                    stream=GzipDecompressorStream(input)
                    try
                        _odb_copy_bounded(stream,output,limit)
                    finally
                        close(stream)
                    end
                end
            end
        end
        callback(expanded)
    end
end

function _with_odb_archive(callback,path,limit,max_entities)
    max_entities>0 || throw(ArgumentError("max_entities must be positive"))
    # Tar owns a 2 MiB copying buffer in the supported Julia stdlib.
    _enforce_payload_limit(2*1024^2,limit,"ODB archive extraction scratch","max_bytes")
    return _with_odb_entity(path,limit) do tarball
        total=0;count=0;seen=Set{String}();headerpayload=2*1024^2
        Tar.list(tarball) do header
            count+=1;count<=max_entities || throw(ArgumentError("ODB archive exceeds max_entities"))
            header.type in (:file,:directory) || throw(ArgumentError("ODB archives cannot contain links or special files"))
            name=replace(header.path,'\\'=>'/')
            !startswith(name,'/') && !occursin(':',name) && all(p->p!="..",split(name,'/')) ||
                throw(ArgumentError("ODB archive member is outside product directory"))
            normalized=join(filter(p->p!="." && !isempty(p),split(name,'/')),'/')
            key=Sys.iswindows() ? lowercase(normalized) : normalized
            key in seen && throw(ArgumentError("duplicate ODB archive member"))
            headerpayload=_checked_payload_sum("ODB archive metadata",headerpayload,512,2ncodeunits(normalized))
            _enforce_payload_limit(headerpayload,limit,"ODB archive extraction and metadata","max_bytes")
            push!(seen,key)
            header.size>=0 || throw(ArgumentError("negative ODB archive size"))
            total=_checked_payload_sum("ODB extracted archive",total,header.size)
            _enforce_payload_limit(total,limit,"ODB extracted archive","max_bytes")
        end
        mktempdir() do directory
            Tar.extract(tarball,directory;copy_symlinks=false,set_permissions=false)
            roots=String[]
            for (parent,_,files) in walkdir(directory)
                basename(parent)=="matrix" && any(f->f in ("matrix","matrix.gz","matrix.Z"),files) &&
                    push!(roots,dirname(parent))
            end
            length(roots)==1 || throw(ArgumentError("ODB archive must contain exactly one product matrix"))
            callback(only(roots))
        end
    end
end
