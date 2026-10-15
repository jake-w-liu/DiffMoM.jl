# Shared fixture only; including this file never runs tests.
if !isdefined(@__MODULE__, :_retained_spice_wave)
function _retained_spice_wave(path,np,z0)
    lines=readlines(path);frequencies=Float64[];columns=Vector{ComplexF64}[]
    for line in lines[2:end]
        t=parse.(Float64,split(line));@test length(t)==1+4np
        push!(frequencies,t[1]);wave=ComplexF64[]
        for p in 1:np
            voltage=complex(t[4p-2],t[4p-1]);current=-complex(t[4p],t[4p+1])
            push!(wave,(voltage-z0[p]*current)/(2sqrt(z0[p])))
        end
        push!(columns,wave)
    end
    frequencies,columns
end
end
