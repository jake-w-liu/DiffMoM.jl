# Work with the relative quantity change rather than a rounded ratio near one.
# FMA avoids an intermediate product overflow and preserves small represented
# changes. Rare range/cancellation cases use bounded, task-scoped precision.
@inline function _sonnet_geovar_scaled_coordinate(value,first,second,nominal,target,symmetric)
    anchor=symmetric ? first/2+second/2 : first
    change=(target-nominal)/nominal;offset=value-anchor
    half_lost=symmetric && ((!iszero(first) && iszero(first/2)) ||
        (!iszero(second) && iszero(second/2)))
    if isfinite(change) && isfinite(offset) && !half_lost
        after=fma(offset,change,value)
        isfinite(after) && (!iszero(after) && after!=value || value==anchor || target==nominal) && return after
    end
    return setprecision(BigFloat,4352) do
        setrounding(BigFloat,RoundNearest) do
            a=symmetric ? (BigFloat(first)+BigFloat(second))/2 : BigFloat(first)
            v=BigFloat(value);n=BigFloat(nominal);t=BigFloat(target)
            Float64(v+(v-a)*(t-n)/n)
        end
    end
end

