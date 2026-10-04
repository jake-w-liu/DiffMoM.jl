using DiffMoM,StaticArrays,Test,TOML
const audit_directory = normpath(joinpath(@__DIR__, "..", "..", "data", "planar_audit"))
const audit_output = isempty(ARGS) ? joinpath(audit_directory, "odb_extra_symbol_allocation_equivalence.toml") : abspath(only(ARGS))
mkpath(dirname(audit_output))

const DM=DiffMoM
@eval DM begin
    struct _ArtworkValidationAbstractTransform <: _ArtworkShape
        shape::_ArtworkShape
        matrix::SMatrix{2,2,Float64,4}
        inverse::SMatrix{2,2,Float64,4}
        origin::NTuple{2,Float64}
    end
    struct _ArtworkValidationAbstractThermal <: _ArtworkShape
        outside::_ArtworkShape
        inside::_ArtworkShape
        angle::Float64
        count::Int
        gap::Float64
    end
    _artwork_bounds(s::_ArtworkValidationAbstractTransform)=
        _artwork_bounds(_ArtworkTransform(s.shape,s.matrix,s.inverse,s.origin))
    _artwork_bounds(s::_ArtworkValidationAbstractThermal)=_artwork_bounds(s.outside)
    function _artwork_contains(s::_ArtworkValidationAbstractTransform,x,y)
        q=s.inverse*SVector(x-s.origin[1],y-s.origin[2])
        return _artwork_contains(s.shape,q[1],q[2])
    end
    function _artwork_contains(s::_ArtworkValidationAbstractThermal,x,y)
        _artwork_contains(s.outside,x,y) && !_artwork_contains(s.inside,x,y) || return false
        iszero(s.gap) && return true
        step=2Float64(pi)/s.count
        delta=mod(atan(y,x)-deg2rad(s.angle)+step/2,step)-step/2
        radius=hypot(x,y)
        return radius*cos(delta)<0 || abs(radius*sin(delta))>=s.gap/2
    end
end
function oldshape(s)
    s isa DM._ArtworkTransform && return DM._ArtworkValidationAbstractTransform(
        oldshape(s.shape),s.matrix,s.inverse,s.origin)
    s isa DM._ArtworkODBSquaredThermal && return DM._ArtworkValidationAbstractThermal(
        s.outside,s.inside,s.angle,s.count,s.gap)
    return s
end
function main()
    rows=Dict{String,Any}()
    mktempdir() do directory
        file=joinpath(directory,"features")
        for name in ("ths6000x4000x13x5x500","s_ths6000x4000x13x5x500",
                "hplate6000x4000x1000xra200xro100","oval_h6000x3000")
            write(file,"UNITS=MM\nF 1\n\$0 $(name)\nP 0 0 0 P 0 0\n")
            doc=read_odb_features(file;layer="metal");object=only(doc.objects)
            oldobject=PlanarArtworkObject(object.layer,oldshape(object.shape),object.dark,object.bounds,object.attributes)
            olddoc=PlanarArtwork(doc.source,[oldobject],doc.attributes,doc.negative_layers,doc.coordinate_unit_m)
            grid=CellGrid(.007,.005,400,400);offset=(.0035,.0025)
            a=artwork_cell_masks(doc,grid;offset);b=artwork_cell_masks(olddoc,grid;offset)
            @test a==b
            after=@allocated artwork_cell_masks(doc,grid;offset)
            before=@allocated artwork_cell_masks(olddoc,grid;offset)
            @test after<100_000
            @test before>1_000_000
            rows[name]=Dict("complete_mask_equal"=>a==b,"before_bytes"=>before,"after_bytes"=>after,
                "occupied_cells"=>count(only(values(a))))
            println(name," => ",rows[name])
        end
    end
    rows["scope"]="warmed Julia cumulative raster allocation; returned masks included; not peakRSS/native runtime"
    open(audit_output,"w") do io;TOML.print(io,rows);end
end
main()
