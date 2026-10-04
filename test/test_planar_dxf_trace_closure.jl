using DiffMoM, Test

function _dxf_closed_trace_fixture(path; legacy=false)
    header="0\nSECTION\n2\nENTITIES\n"
    if legacy
        entity="0\nPOLYLINE\n8\nCu\n70\n1\n40\n0.5\n41\n0.5\n"
        for (x,y) in ((1,1),(3,1),(3,3),(1,3))
            entity*="0\nVERTEX\n10\n$x\n20\n$y\n"
        end
        entity*="0\nSEQEND\n"
    else
        entity="0\nLWPOLYLINE\n8\nCu\n90\n4\n70\n1\n43\n0.5\n"
        for (x,y) in ((1,1),(3,1),(3,3),(1,3))
            entity*="10\n$x\n20\n$y\n"
        end
    end
    write(path,header*entity*"0\nENDSEC\n0\nEOF\n")
end

@testset "DXF closed trace width and closure" begin
    mktempdir() do dir
        for legacy in (false,true)
            path=joinpath(dir,legacy ? "legacy.dxf" : "lightweight.dxf")
            _dxf_closed_trace_fixture(path;legacy)
            doc=read_dxf(path;unit_m=1.)
            contains(x,y)=any(doc.polygons) do p
                DiffMoM._artwork_contains(DiffMoM._ArtworkPolygon(p.vertices),x,y)
            end
            # Independent square-ring membership: all four sides conduct,
            # the inner square is empty, and each outer miter reaches 0.75.
            for point in ((1.,2.),(2.,1.),(3.,2.),(2.,3.),(.8,.8),(3.2,3.2))
                @test contains(point...)
            end
            @test !contains(2.,2.)
            @test !contains(.7,.7)
        end
    end
end
