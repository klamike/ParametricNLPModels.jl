using NLPModels
using ParametricNLPModels

mutable struct PlainNLPModel{T, S} <: AbstractNLPModel{T, S}
  meta::NLPModelMeta{T, S}
  counters::Counters
end

function PlainNLPModel(::Type{T}) where {T}
  meta = NLPModelMeta(1, x0 = zeros(T, 1), nnzo = 1, nnzj = 0, nnzh = 1)
  return PlainNLPModel(meta, Counters())
end

PlainNLPModel() = PlainNLPModel(Float64)

mutable struct SimpleParamNLPModel{T, S} <: AbstractNLPModel{T, S}
  meta::NLPModelMeta{T, S}
  param_meta::ParametricNLPModelMeta
  counters::Counters
  ps::S
end

function SimpleParamNLPModel(::Type{T}; ps = T[2, 3]) where {T}
  p1, p2 = ps[1], ps[2]
  meta = NLPModelMeta(
    2,
    ncon = 1,
    x0 = zeros(T, 2),
    lvar = T[-p1, -p2],
    uvar = T[p1, p2],
    lcon = T[-p2],
    ucon = T[p2],
    nnzh = 2,
    nnzj = 2,
    name = "Simple Parametric NLP Model",
  )
  param_meta = ParametricNLPModelMeta(
    nparam = 2,
    nnzjp = 1,
    nnzhp = 3,
    nnzgp = 2,
    nnzjplcon = 1,
    nnzjpucon = 1,
    nnzjplvar = 2,
    nnzjpuvar = 2,
    grad_param_available = true,
    jac_param_available = true,
    hess_param_available = true,
    jpprod_available = true,
    jptprod_available = true,
    hpprod_available = true,
    hptprod_available = true,
    lcon_jac_available = true,
    ucon_jac_available = true,
    lvar_jac_available = true,
    uvar_jac_available = true,
    lcon_jpprod_available = true,
    ucon_jpprod_available = true,
    lvar_jpprod_available = true,
    uvar_jpprod_available = true,
    lcon_jptprod_available = true,
    ucon_jptprod_available = true,
    lvar_jptprod_available = true,
    uvar_jptprod_available = true,
  )
  return SimpleParamNLPModel(meta, param_meta, Counters(), T.(ps))
end

SimpleParamNLPModel(; ps = [2.0, 3.0]) = SimpleParamNLPModel(Float64; ps = ps)

ParametricNLPModels.get_param_values(nlp::SimpleParamNLPModel) = nlp.ps

function ParametricNLPModels.set_param_values!(nlp::SimpleParamNLPModel, p::AbstractVector)
  @lencheck get_nparam(nlp) p
  nlp.ps .= p
  return nlp
end

function NLPModels.obj(nlp::SimpleParamNLPModel, x::AbstractVector)
  @lencheck 2 x
  increment!(nlp, :neval_obj)
  p1, p2 = nlp.ps
  return (x[1] - p1)^2 + (x[2] - p2)^2
end

function NLPModels.grad!(nlp::SimpleParamNLPModel, x::AbstractVector, g::AbstractVector)
  @lencheck 2 x g
  increment!(nlp, :neval_grad)
  p1, p2 = nlp.ps
  g .= [2 * (x[1] - p1); 2 * (x[2] - p2)]
  return g
end

function NLPModels.cons!(nlp::SimpleParamNLPModel, x::AbstractVector, c::AbstractVector)
  @lencheck 2 x
  @lencheck 1 c
  increment!(nlp, :neval_cons)
  p1 = nlp.ps[1]
  c[1] = x[1] + p1 * x[2]
  return c
end

function NLPModels.jac_structure!(nlp::SimpleParamNLPModel, rows::AbstractVector{Int}, cols::AbstractVector{Int})
  @lencheck 2 rows cols
  rows .= [1, 1]
  cols .= [1, 2]
  return rows, cols
end

function NLPModels.jac_coord!(nlp::SimpleParamNLPModel, x::AbstractVector, vals::AbstractVector)
  @lencheck 2 x vals
  increment!(nlp, :neval_jac)
  p1 = nlp.ps[1]
  vals .= [1, p1]
  return vals
end

function NLPModels.hess_structure!(nlp::SimpleParamNLPModel, rows::AbstractVector{Int}, cols::AbstractVector{Int})
  @lencheck 2 rows cols
  rows .= [1, 2]
  cols .= [1, 2]
  return rows, cols
end

function NLPModels.hess_coord!(
  nlp::SimpleParamNLPModel,
  x::AbstractVector{T},
  y::AbstractVector{T},
  vals::AbstractVector{T};
  obj_weight = one(T),
) where {T}
  @lencheck 2 x y vals
  increment!(nlp, :neval_hess)
  vals .= 2 * obj_weight
  return vals
end

function ParametricNLPModels.grad_param!(nlp::SimpleParamNLPModel, x::AbstractVector, g::AbstractVector)
  @lencheck 2 x g
  p1, p2 = nlp.ps
  g .= [-2 * (x[1] - p1); -2 * (x[2] - p2)]
  return g
end

function ParametricNLPModels.jac_param_structure!(nlp::SimpleParamNLPModel, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer})
  @lencheck 1 rows cols
  rows[1] = 1
  cols[1] = 1
  return rows, cols
end

function ParametricNLPModels.jac_param_coord!(nlp::SimpleParamNLPModel, x::AbstractVector, vals::AbstractVector)
  @lencheck 2 x
  @lencheck 1 vals
  vals[1] = x[2]
  return vals
end

function ParametricNLPModels.jpprod!(nlp::SimpleParamNLPModel, x::AbstractVector, v::AbstractVector, Jv::AbstractVector)
  @lencheck 2 x v
  @lencheck 1 Jv
  Jv[1] = x[2] * v[1]
  return Jv
end

function ParametricNLPModels.jptprod!(nlp::SimpleParamNLPModel, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector)
  @lencheck 2 x Jtv
  @lencheck 1 v
  Jtv .= [x[2] * v[1]; 0]
  return Jtv
end

function ParametricNLPModels.hess_param_structure!(nlp::SimpleParamNLPModel, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer})
  @lencheck 3 rows cols
  rows .= [1, 2, 2]
  cols .= [1, 1, 2]
  return rows, cols
end

function ParametricNLPModels.hess_param_coord!(
  nlp::SimpleParamNLPModel,
  x::AbstractVector{T},
  y::AbstractVector{T},
  vals::AbstractVector{T};
  obj_weight = one(T),
) where {T}
  @lencheck 2 x
  @lencheck 1 y
  @lencheck 3 vals
  σ = T(obj_weight)
  vals .= [-2σ, y[1], -2σ]
  return vals
end

function ParametricNLPModels.hpprod!(
  nlp::SimpleParamNLPModel,
  x::AbstractVector{T},
  y::AbstractVector{T},
  v::AbstractVector{T},
  Hv::AbstractVector{T};
  obj_weight = one(T),
) where {T}
  @lencheck 2 x v Hv
  @lencheck 1 y
  σ = T(obj_weight)
  Hv .= [-2σ * v[1]; y[1] * v[1] - 2σ * v[2]]
  return Hv
end

function ParametricNLPModels.hptprod!(
  nlp::SimpleParamNLPModel,
  x::AbstractVector{T},
  y::AbstractVector{T},
  v::AbstractVector{T},
  Htv::AbstractVector{T};
  obj_weight = one(T),
) where {T}
  @lencheck 2 x v Htv
  @lencheck 1 y
  σ = T(obj_weight)
  Htv .= [-2σ * v[1] + y[1] * v[2]; -2σ * v[2]]
  return Htv
end

function ParametricNLPModels.lcon_jac_param_structure!(nlp::SimpleParamNLPModel, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer})
  @lencheck 1 rows cols
  rows[1] = 1
  cols[1] = 2
  return rows, cols
end

function ParametricNLPModels.lcon_jac_param_coord!(nlp::SimpleParamNLPModel, vals::AbstractVector)
  @lencheck 1 vals
  vals[1] = -1
  return vals
end

function ParametricNLPModels.lcon_jpprod!(nlp::SimpleParamNLPModel, v::AbstractVector, Jv::AbstractVector)
  @lencheck 2 v
  @lencheck 1 Jv
  Jv[1] = -v[2]
  return Jv
end

function ParametricNLPModels.lcon_jptprod!(nlp::SimpleParamNLPModel, v::AbstractVector, Jtv::AbstractVector)
  @lencheck 1 v
  @lencheck 2 Jtv
  Jtv .= [0; -v[1]]
  return Jtv
end

function ParametricNLPModels.ucon_jac_param_structure!(nlp::SimpleParamNLPModel, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer})
  @lencheck 1 rows cols
  rows[1] = 1
  cols[1] = 2
  return rows, cols
end

function ParametricNLPModels.ucon_jac_param_coord!(nlp::SimpleParamNLPModel, vals::AbstractVector)
  @lencheck 1 vals
  vals[1] = 1
  return vals
end

function ParametricNLPModels.ucon_jpprod!(nlp::SimpleParamNLPModel, v::AbstractVector, Jv::AbstractVector)
  @lencheck 2 v
  @lencheck 1 Jv
  Jv[1] = v[2]
  return Jv
end

function ParametricNLPModels.ucon_jptprod!(nlp::SimpleParamNLPModel, v::AbstractVector, Jtv::AbstractVector)
  @lencheck 1 v
  @lencheck 2 Jtv
  Jtv .= [0; v[1]]
  return Jtv
end

function ParametricNLPModels.lvar_jac_param_structure!(nlp::SimpleParamNLPModel, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer})
  @lencheck 2 rows cols
  rows .= [1, 2]
  cols .= [1, 2]
  return rows, cols
end

function ParametricNLPModels.lvar_jac_param_coord!(nlp::SimpleParamNLPModel, vals::AbstractVector)
  @lencheck 2 vals
  vals .= [-1, -1]
  return vals
end

function ParametricNLPModels.lvar_jpprod!(nlp::SimpleParamNLPModel, v::AbstractVector, Jv::AbstractVector)
  @lencheck 2 v Jv
  Jv .= -v
  return Jv
end

function ParametricNLPModels.lvar_jptprod!(nlp::SimpleParamNLPModel, v::AbstractVector, Jtv::AbstractVector)
  @lencheck 2 v Jtv
  Jtv .= -v
  return Jtv
end

function ParametricNLPModels.uvar_jac_param_structure!(nlp::SimpleParamNLPModel, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer})
  @lencheck 2 rows cols
  rows .= [1, 2]
  cols .= [1, 2]
  return rows, cols
end

function ParametricNLPModels.uvar_jac_param_coord!(nlp::SimpleParamNLPModel, vals::AbstractVector)
  @lencheck 2 vals
  vals .= [1, 1]
  return vals
end

function ParametricNLPModels.uvar_jpprod!(nlp::SimpleParamNLPModel, v::AbstractVector, Jv::AbstractVector)
  @lencheck 2 v Jv
  Jv .= v
  return Jv
end

function ParametricNLPModels.uvar_jptprod!(nlp::SimpleParamNLPModel, v::AbstractVector, Jtv::AbstractVector)
  @lencheck 2 v Jtv
  Jtv .= v
  return Jtv
end
