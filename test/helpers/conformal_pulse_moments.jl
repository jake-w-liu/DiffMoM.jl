# Shared independent simplex oracle; including this file never runs tests.
if !isdefined(@__MODULE__, :series_moments)
const Q=Rational{BigInt}

function exact_area(v)
    abs((Q(v[1,2])-Q(v[1,1]))*(Q(v[2,3])-Q(v[2,1]))-
        (Q(v[1,3])-Q(v[1,1]))*(Q(v[2,2])-Q(v[2,1])))/2
end

# Integrate the exponential's Taylor series term by term with the independent
# simplex monomial identity ∫λ₁ᵖλ₂ᑫλ₃ʳ dA = 2A p!q!r!/(p+q+r+2)!.
# The phases here are bounded by 2.3; 48 terms make the truncation error far
# smaller than double precision, including the nearly collinear witnesses.
function series_moments(v,kx,ky)
    setprecision(BigFloat,512) do
        area=BigFloat(exact_area(v));vv=BigFloat.(v)
        z=ntuple(j->im*(BigFloat(kx)*vv[1,j]+BigFloat(ky)*vv[2,j]),3)
        powers=ntuple(j->[z[j]^p for p in 0:48],3)
        weights=zeros(Complex{BigFloat},3)
        for n in 0:48
            denominator=BigFloat(factorial(big(n+3)))
            for p in 0:n,q in 0:n-p
                r=n-p-q
                term=2area*powers[1][p+1]*powers[2][q+1]*powers[3][r+1]/denominator
                weights[1]+=(p+1)*term
                weights[2]+=(q+1)*term
                weights[3]+=(r+1)*term
            end
        end
        constant=sum(weights)
        firstx=sum((vv[1,j]-vv[1,1])*weights[j] for j in 1:3)
        firsty=sum((vv[2,j]-vv[2,1])*weights[j] for j in 1:3)
        ComplexF64.((constant,firstx,firsty))
    end
end
end
