module ParametricNLPModels

import NLPModels
import NLPModels: AbstractNLPModel, @lencheck
import SparseArrays: sparse

include("parametric_meta.jl")
include("parametric_api.jl")

end # module ParametricNLPModels
