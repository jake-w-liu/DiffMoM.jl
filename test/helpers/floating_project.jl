# Shared fixture for independent full-validation processes.
function _floating_project(;resistance=.2,reverse=false)
    return planar_project_from_dict(Dict("project"=>Dict("unit"=>"mm"),
        "box"=>Dict("width"=>6.,"length"=>4.),"mesh"=>Dict("nx"=>12,"ny"=>8),
        "stackup"=>Dict("layers"=>[Dict("thickness"=>.5),Dict("thickness"=>.5)]),
        "metals"=>Dict("film"=>Dict("type"=>"surface_impedance","rs"=>resistance)),
        "polygons"=>[Dict("name"=>"reference","level"=>1,"metal"=>"film","net"=>"local_G",
                "vertices"=>[[.5,1.5],[2.5,1.5],[2.5,2.5],[.5,2.5]]),
            Dict("name"=>"signal","level"=>1,"metal"=>"film","net"=>"S",
                "vertices"=>[[3.5,1.5],[5.5,1.5],[5.5,2.5],[3.5,2.5]])],
        "ports"=>[Dict("type"=>"floating","name"=>"diff","number"=>1,"polygon"=>"signal","edge"=>4,
            "ref_polygon"=>"reference","ref_edge"=>2,"source_edge"=>6,"polarity"=>reverse ? -1 : 1)]))
end

