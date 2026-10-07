# Analytic simplex exponential moments for genuine affine triangle currents.
# No surface quadrature is used in modal transforms.

# For imaginary centered nodes of radius r<=1, the normalized simplex
# integral has magnitude at least cos(r)>=1-r^2/2. The omitted Taylor tail
# after m terms is bounded by r^(m+1)/((m+1)!*(1-r/(m+2))). Select its
# order at Float64 unit roundoff with outward bounds on every operation.
@inline function _planar_exp_series_order(radius::Float64)
    isfinite(radius) && 0<=radius<=1 || throw(ArgumentError("Taylor radius outside bounded imaginary-phase cluster"))
    iszero(radius) && return 0
    square=nextfloat(radius*radius)
    lower=prevfloat(1-nextfloat(square/2))
    target=prevfloat((eps(Float64)/2)*lower)
    term=radius;m=0
    while true
        ratio=nextfloat(radius/(m+2))
        denominator=prevfloat(1-ratio)
        tail=nextfloat(term/denominator)
        tail<=target && return m
        m+=1
        term=nextfloat(term*nextfloat(radius/(m+1)))
    end
end

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
    # nodes and zero transverse phase. Bound its tail using the actual radius.
    # Preserve even/odd coefficient-space degrees before truncating; separately
    # representable low-phase moments can survive affine cancellation.
    for m in 1:(_planar_exp_series_order(maximum(abs,z))+2N-1)
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

# Each affine simplex weight is DD(z1,z2,z3,zj). Reuse the common
# three-node table instead of rebuilding its divided differences three times.
# Clustered phases share h_n, the complete homogeneous symmetric polynomial;
# duplicating node j gives g_j,n = h_n + w_j*g_j,n-1. Mixed/coincident gaps
# retain the bounded original divided-difference evaluation.
@inline function _planar_triangle_exp_weights3(z::NTuple{3,ComplexF64})
    sorted=_planar_sorted_phases3(z...)
    a,b,c=sorted
    if abs(c-a)<=1.0
        center=sum(z)/3;w=map(v->v-center,z)
        e1=sum(w);e2=w[1]*w[2]+w[1]*w[3]+w[2]*w[3];e3=prod(w)
        h0=1.0+0im;h1=0.0im;h2=0.0im
        g1=g2=g3=1.0+0im
        factorial=6.0;v1=v2=v3=1/6+0im
        # Three affine vertex values can cancel the constant and linear
        # moments. Retain both even/odd coefficient-space degrees before the tail.
        for n in 1:(_planar_exp_series_order(maximum(abs,w))+2length(z)-1)
            h=e1*h0-e2*h1+e3*h2
            g1=h+w[1]*g1;g2=h+w[2]*g2;g3=h+w[3]*g3
            factorial*=n+3
            v1+=g1/factorial;v2+=g2/factorial;v3+=g3/factorial
            h2=h1;h1=h0;h0=h
        end
        factor=exp(center)
        return (factor*v1,factor*v2,factor*v3)
    elseif abs(b-a)>1.0 && abs(c-b)>1.0
        ea,eb,ec=exp(a),exp(b),exp(c)
        ab=(eb-ea)/(b-a);bc=(ec-eb)/(c-b)
        abc=(bc-ab)/(c-a)
        aab=(ab-ea)/(b-a);abb=(eb-ab)/(b-a)
        bbc=(bc-eb)/(c-b);bcc=(ec-bc)/(c-b)
        weights=((abc-aab)/(c-a),(bbc-abb)/(c-a),(bcc-abc)/(c-a))
        return ntuple(j->z[j]==a ? weights[1] : z[j]==b ? weights[2] : weights[3],Val(3))
    end
    return ntuple(j->_planar_exp_divdiff(_planar_insert_phase3(sorted,z[j])),Val(3))
end

@inline function _planar_triangle_fourier_weights(vertices,kx,ky)
    twicearea=abs(_planar_orient2d(vertices[1,1],vertices[2,1],
        vertices[1,2],vertices[2,2],vertices[1,3],vertices[2,3]))
    phases=ntuple(i->ComplexF64(1im*(kx*vertices[1,i]+ky*vertices[2,i])),Val(3))
    shift=phases[1]
    return twicearea*exp(shift),_planar_triangle_exp_weights3(map(z->z-shift,phases))
end

@inline function _planar_triangle_sum3(a,b,c)
    first=a+b
    first_error=_sonnet_geovar_sum_error(a,b,first)
    total=first+c
    second_error=_sonnet_geovar_sum_error(first,c,total)
    return total+(first_error+second_error)
end
@inline _planar_triangle_sum3(values)=_planar_triangle_sum3(values...)

# Preserve the ordinary finite error-free sum. If only an intermediate sum
# overflows, divide the three coefficients before adding: the simplex
# factorial makes their integral representable even when their sum is not.
@inline function _planar_triangle_coefficient(values,divisor)
    total=_planar_triangle_sum3(values)
    isfinite(total) && return total/divisor
    return _planar_triangle_sum3(map(v->v/divisor,values))
end
@inline function _planar_triangle_coefficient(values,g1,g2,g3,divisor)
    products=(values[1]*g1,values[2]*g2,values[3]*g3)
    all(isfinite,products) && return _planar_triangle_coefficient(products,divisor)
    return _planar_triangle_sum3((values[1]/divisor)*g1,
        (values[2]/divisor)*g2,(values[3]/divisor)*g3)
end

# An affine field can retain a quadratic Fourier moment smaller
# than the rounding step of any individual vertex weight. Contract each
# Taylor coefficient before adding its constant contribution.
@inline function _planar_triangle_series_pair(vertices,x,y,kx,ky)
    phases=ntuple(j->ComplexF64(1im*(kx*vertices[1,j]+ky*vertices[2,j])),Val(3))
    shift=phases[1];z=map(v->v-shift,phases)
    sorted=_planar_sorted_phases3(z...)
    abs(sorted[3]-sorted[1])<=1 || return nothing
    center=sum(z)/length(z);w=map(v->v-center,z)
    e1=sum(w);e2=w[1]*w[2]+w[1]*w[3]+w[2]*w[3];e3=prod(w)
    h0=1.0+0im;h1=h2=0.0im;g1=g2=g3=1.0+0im
    factorial=6.0;fx=_planar_triangle_coefficient(x,factorial)+0im;fy=_planar_triangle_coefficient(y,factorial)+0im
    for n in 1:(_planar_exp_series_order(maximum(abs,w))+2length(z)-1)
        h=e1*h0-e2*h1+e3*h2
        g1=h+w[1]*g1;g2=h+w[2]*g2;g3=h+w[3]*g3
        factorial*=n+length(z)
        fx+=_planar_triangle_coefficient(x,g1,g2,g3,factorial)
        fy+=_planar_triangle_coefficient(y,g1,g2,g3,factorial)
        h2=h1;h1=h0;h0=h
    end
    factor=abs(_planar_orient2d(vertices[1,1],vertices[2,1],
        vertices[1,2],vertices[2,2],vertices[1,3],vertices[2,3]))*exp(shift)*exp(center)
    return factor*fx,factor*fy
end

"""Analytic integral of an affine scalar over a physical triangle.
`vertices` is 2x3 and `values` gives its values at the three vertices.
The integrand is affine(values)*exp(i*(kx*x+ky*y))."""
function _planar_triangle_affine_fourier(vertices::AbstractMatrix{<:Real},
        values::NTuple{3,Float64},kx::Real,ky::Real)
    series_pair=_planar_triangle_series_pair(vertices,values,values,kx,ky)
    series_pair===nothing || return series_pair[1]
    # Match the mesh's robust area predicate. Direct rounded products
    # can cancel for a valid near-collinear triangle and erase its entire
    # Fourier reaction, including the constant moment at zero phase.
    factor,weights=_planar_triangle_fourier_weights(vertices,kx,ky)
    total=0.0im
    for j in 1:3
        total+=values[j]*weights[j]
    end
    return factor*total
end

# The vector components have the same phases and triangle area. Share their
# analytic weights; the pair has the same scalar integral convention.
function _planar_triangle_affine_fourier_pair(vertices::AbstractMatrix{<:Real},
        x::NTuple{3,Float64},y::NTuple{3,Float64},kx::Real,ky::Real)
    series_pair=_planar_triangle_series_pair(vertices,x,y,kx,ky)
    series_pair===nothing || return series_pair
    factor,weights=_planar_triangle_fourier_weights(vertices,kx,ky)
    fx=0.0im;fy=0.0im
    for j in 1:3
        fx+=x[j]*weights[j];fy+=y[j]*weights[j]
    end
    fx*=factor;fy*=factor

    return fx,fy
end
