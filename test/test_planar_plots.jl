using DiffMoM,Test

@testset "planar interactive response/layout/current plots" begin
    freqs=[1e9,2e9];series=[ComplexF64[.1 .8;.8 .2],ComplexF64[.2 .7im;.7im .3]]
    data=PlanarNetworkData(freqs,series;z0=[50.,75.])
    response=plot_planar_sparams(data;pairs=[(1,1),(2,1)])
    @test length(response.data)==2
    @test response.data[1][:x]==freqs
    @test response.data[2][:y]≈20log10.(abs.([.8,.7im]))
    @test length(plot_planar_sparams(data;reference=data).data)==2
    @test_throws ArgumentError plot_planar_sparams(data;pairs=[(3,1)])
    @test_throws ArgumentError plot_planar_sparams(data;quantity=:unknown)
    @test_throws ArgumentError plot_planar_sparams(data;reference=PlanarNetworkData(freqs,series))
    smith=plot_planar_smith(data)
    @test length(smith.data)==19
    @test smith.data[end][:x]==[.2,.3]
    @test smith.data[end][:customdata]==freqs
    @test_throws ArgumentError plot_planar_smith(data;ports=[0])
    active=plot_planar_smith(PlanarNetworkData([1.],[fill(1.5+0im,1,1)]))
    @test last(active.layout[:xaxis][:range])>1.5
    grid=CellGrid(1e-3,1e-3,4,4)
    stack=PlanarStackup([PlanarLayer(1.,1.,.1e-3),PlanarLayer(1.,1.,.1e-3)],
        TERM_GND,TERM_GND,grid.a,grid.b)
    shape=planar_transform(planar_line(length=grid.a,width=.5e-3,level=1,metal="pec");
        offset=(0.,grid.b/2))
    layout=build_planar_layout(stack,grid,[shape],shape.pins)
    geometry=plot_planar_layout(layout)
    @test length(geometry.data)==3
    @test all(z -> isapprox(z,100.;rtol=1e-14),geometry.data[1][:z])
    @test_throws ArgumentError plot_planar_layout(layout;unit=:inch)
    mask=falses(3,2);mask[2,1]=true
    jx=zeros(ComplexF64,3,2);jx[2,1]=1+2im
    map=PlanarCurrentMap(:sheet,1,[.5,1.5,2.5].*1e-6,[.5,1.5].*1e-6,
        .1e-3,.1e-3,.1e-3,mask,jx,zeros(3,2),zeros(3,2))
    current=plot_planar_currents(map)
    @test current.data[1][:x]==[.5,1.5,2.5]
    @test current.data[1][:z][1,2]≈sqrt(5.)
    @test current.data[1][:z][1,1]===nothing
    @test current.data[1][:zsmooth]==false
    @test_throws ArgumentError plot_planar_currents(map;component=:magnitude,quantity=:imag)
    mktempdir() do dir
        path=joinpath(dir,"response.html")
        save_planar_plot(path,response)
        @test occursin("Plotly.newPlot",read(path,String))
        save_planar_plot(joinpath(dir,"current.html"),current)
        @test isfile(joinpath(dir,"current.html"))
        nulls=plot_planar_sparams(PlanarNetworkData(freqs,[zeros(ComplexF64,1,1),fill(.1+0im,1,1)]))
        @test nulls.data[1][:y][1]===nothing
        save_planar_plot(joinpath(dir,"null.html"),nulls)
        @test isfile(joinpath(dir,"null.html"))
    end
end
