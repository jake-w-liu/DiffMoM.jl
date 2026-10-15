# Shared by the core and periodic validation processes.
function _float_vector_output_allocation(n::Int)
    zeros(Float64, n)
    return @allocated zeros(Float64, n)
end

function _filter_allocation(W, w_sum, rho)
    apply_filter(W, w_sum, rho)
    return @allocated apply_filter(W, w_sum, rho)
end

function _filter_transpose_allocation(W, w_sum, gradient)
    apply_filter_transpose(W, w_sum, gradient)
    return @allocated apply_filter_transpose(W, w_sum, gradient)
end
