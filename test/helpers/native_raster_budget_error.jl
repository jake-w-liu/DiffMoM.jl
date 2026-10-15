# Shared fixture for independent full-validation processes.
function _native_raster_budget_error(f)
    try
        f()
    catch err
        err isa ArgumentError || rethrow()
        return sprint(showerror,err)
    end
    error("expected native raster resource rejection")
end

