using Documenter
using ParametricNLPModels

makedocs(
    modules = [ParametricNLPModels],
    sitename = "ParametricNLPModels.jl",
    format = Documenter.HTML(),
    pages = [
        "Home" => "index.md",
        "Reference" => "public.md",
    ],
)

deploydocs(
    repo = "github.com/klamike/ParametricNLPModels.jl.git",
    devbranch = "main",
)
