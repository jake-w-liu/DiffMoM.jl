# Analytic simplex exponential moments for genuine affine triangle currents.
# No surface quadrature is used in modal transforms.

@inline function _planar_exp_divdiff_series(nodes::NTuple{N,ComplexF64}) where N
    center=sum(nodes)/N
    z=map(x->x-center,nodes)
    e1=sum(z)
    e2=0.0im;e3=0.0im;e4=0.0im
    for i in 1:N,j in i+1:N
        e2+=z[i]*z[j]
        for k in j+1:N
            e3+=z[i]*z[j]*z[k]
            for l in k+1:N
                e4+=z[i]*z[j]*z[k]*z[l]
            end
        end
    end
    h0=1.0+0im;h1=0.0im;h2=0.0im;h3=0.0im
    factorial=Float64(prod(1:N-1;init=1))
    value=h0/factorial
    # Complete homogeneous symmetric polynomials give the exact Taylor
    # series of the exponential divided difference, including repeated
    # nodes and zero transverse phase. Radius<=1 gives ample margin.
    for m in 1:32
        h=e1*h0-e2*h1+e3*h2-e4*h3
        factorial*=m+N-1
        value+=h/factorial
        h3=h2;h2=h1;h1=h0;h0=h
    end
    return exp(center)*value
end

@inline _planar_exp_divdiff(nodes::NTuple{1,ComplexF64})=exp(nodes[1])
@inline function _planar_exp_divdiff(nodes::NTuple{N,ComplexF64}) where N
    # Nodes are imaginary phases sorted along the imaginary axis. Only
    # local spans need a series; far-separated nodes use divided values.
    abs(nodes[end]-nodes[1])<=1.0 && return _planar_exp_divdiff_series(nodes)
    left=ntuple(i->nodes[i],Val(N-1))
    right=ntuple(i->nodes[i+1],Val(N-1))
    return (_planar_exp_divdiff(right)-_planar_exp_divdiff(left))/(nodes[end]-nodes[1])
end

@inline function _planar_sorted_phases3(a::ComplexF64,b::ComplexF64,c::ComplexF64)
    imag(a)>imag(b) && ((a,b)=(b,a))
    imag(b)>imag(c) && ((b,c)=(c,b))
    imag(a)>imag(b) && ((a,b)=(b,a))
    return (a,b,c)
end
@inline function _planar_insert_phase3(z::NTuple{3,ComplexF64},v::ComplexF64)
    if imag(v)<=imag(z[1])
        return (v,z[1],z[2],z[3])
    elseif imag(v)<=imag(z[2])
        return (z[1],v,z[2],z[3])
    elseif imag(v)<=imag(z[3])
        return (z[1],z[2],v,z[3])
    end
    return (z[1],z[2],z[3],v)
end

"""Analytic integral of an affine scalar over a physical triangle.
`vertices` is 2x3 and `values` gives its values at the three vertices.
The integrand is affine(values)*exp(i*(kx*x+ky*y))."""
function _planar_triangle_affine_fourier(vertices::AbstractMatrix{<:Real},
        values::NTuple{3,Float64},kx::Real,ky::Real)
    a=vertices[1,2]-vertices[1,1];b=vertices[2,2]-vertices[2,1]
    c=vertices[1,3]-vertices[1,1];d=vertices[2,3]-vertices[2,1]
    twicearea=abs(a*d-b*c)
    phases=ntuple(i->ComplexF64(1im*(kx*vertices[1,i]+ky*vertices[2,i])),Val(3))
    shift=phases[1]
    shifted=map(x->x-shift,phases)
    sorted=_planar_sorted_phases3(shifted...)
    total=0.0im
    for j in 1:3
        total+=values[j]*_planar_exp_divdiff(_planar_insert_phase3(sorted,shifted[j]))
    end
    return twicearea*exp(shift)*total
end
