# Shared fixture only; including this file never runs tests.
if !isdefined(@__MODULE__, :_native_floating_pad_project)
function _native_floating_pad_project(;cup=false)
    r(t)=SonnetRecord(1,String.(t))
    left=SonnetPolygon(:sheet,0,-1,1,
        [.00025 .0004375 .0004375 .00025 .00025;.0004375 .0004375 .0005625 .0005625 .0004375],"","","",String[])
    right=SonnetPolygon(:sheet,0,-1,2,
        [.0005625 .00075 .00075 .0005625 .0005625;.0004375 .0004375 .0005625 .0005625 .0004375],"","","",String[])
    pins=[r(["TYPE","NONE"]),r(["GNDREF","FLOAT"]),r(["TERMW","FEED"]),
        r(["SMDP","0",".4375",".5","L","1","1"]),r(["SMDP","0",".5625",".5","R","-1","2"])]
    ports=SonnetPortSpec[]
    if cup
        for (poly,edge,num,x) in ((1,1,1,".4375"),(2,3,-1,".5625"))
            records=[r(["POR1","CUP","Auto"]),r(["GNDREF","FLOAT"]),r(["TERMW","FEED"])]
            push!(ports,SonnetPortSpec(:cup,poly,edge,num,[string(num),"50","0","0","0",x,".5"],records))
        end
    end
    return SonnetProject("floating_contract.son",Dict("LNG"=>"MM","CAP"=>"PF","IND"=>"NH"),.001,1e9,
        ["1","1","1","32","32","20","0"],[[".5","1","1","0","0","0"],[".5","1","1","0","0","0"]],
        Vector{String}[],["PEC","0","RES","0"],["PEC","0","RES","0"],[left,right],ports,
        Dict{String,String}(),cup ? Vector{SonnetRecord}[] : [pins],SonnetRecord[],SonnetRecord[])
end
end
