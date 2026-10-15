# Shared fixture only; including this file never runs tests.
if !isdefined(@__MODULE__, :_synthetic_smd_project)
function _synthetic_smd_project(kind="RES",value="10")
    records(tokens)=SonnetRecord(1,String.(tokens))
    left=SonnetPolygon(:sheet,0,-1,1,[0.0 0.0 .0025 .0025;.0015 .0025 .0025 .0015],"","","",String[])
    right=SonnetPolygon(:sheet,0,-1,2,[.0035 .006 .006 .0035;.0015 .0015 .0025 .0025],"","","",String[])
    ports=[SonnetPortSpec(:box,1,0,1,["1","50","0","0","0","0","2"],SonnetRecord[]),
        SonnetPortSpec(:box,2,1,2,["2","50","0","0","0","6","2"],SonnetRecord[])]
    component=[records(["TYPE","IDEAL",kind,value]),records(["GNDREF","BOX","AUTO"]),
        records(["TERMW","FEED"]),records(["SMDP","0","2.5","2","L","3","1"]),
        records(["SMDP","0","3.5","2","R","4","2"])]
    return SonnetProject("synthetic_smd.son",Dict("LNG"=>"MM","CAP"=>"PF","IND"=>"NH"),.001,1e9,
        ["1","6","4","24","16","20","0"],[[".5","1","1","0","0","0"],
        [".5","1","1","0","0","0"]],Vector{String}[],["PEC","0","RES","0"],
        ["PEC","0","RES","0"],[left,right],ports,Dict{String,String}(),[component],SonnetRecord[],SonnetRecord[])
end
end
