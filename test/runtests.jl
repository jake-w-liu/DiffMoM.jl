# Pkg.test() runs the focused suite; test_args=["full"] runs reference validation.
if isempty(ARGS)
    include("focused_suite.jl")
elseif ARGS == ["full"]
    include("full_validation.jl")
else
    throw(ArgumentError("use Pkg.test() or Pkg.test(test_args=[\"full\"])"))
end
