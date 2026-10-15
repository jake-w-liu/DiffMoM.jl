# Shared fixture for independent full-validation processes.
function _dense_workspace_fixture(walls=WALL_PEC)
    grid=CellGrid(.008,.006,8,6;walls)
    stack=PlanarStackup([PlanarLayer(2.2-.02im,1.,.0004),
        PlanarLayer(3.,1.1,.0006;epsr_z=4.),PlanarLayer(1.,1.,.0003)],
        TERM_GND,TERM_GND,grid.a,grid.b)
    sheets=[sheet_level(1,8,6),sheet_level(2,8,6)]
    for sh in sheets
        sh.mask.=true; sh.connect_west.=true; sh.connect_east.=true
    end
    vl=via_level(2,8,6);vl.uni[2,2]=vl.tap[2,2]=true;vl.uni[6,4]=true
    vol=vol_level(3,8,6);vol.mask.=true
    build_planar_problem(stack,grid,sheets,
        [PlanarPort(1,:west,2:4,50.),PlanarPort(2,:east,2:4,75.)];
        vias=[vl],vols=[vol])
end

# Preserve the former retained-assembly route as a regression oracle. Its
# original complete source is archived under validation/planar_audit.
function _legacy_fft_dense_workspace_assembly(prob,freq;kw...)
    operator=planar_ufft_operator(prob,freq;kw...,_fold_iterative=false)
    matrix=Matrix{ComplexF64}(undef,size(operator))
    DiffMoM._planar_fft_dense_fill!(matrix,operator)
end

