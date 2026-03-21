export AbstractParametricNLPModelMeta, ParametricNLPModelMeta

abstract type AbstractParametricNLPModelMeta end

"""
    ParametricNLPModelMeta(; kwargs...)

Metadata related to parameters including number of parameters, nonzero counts, and API availability.
"""
struct ParametricNLPModelMeta <: AbstractParametricNLPModelMeta
  nparam::Int
  nnzjp::Int
  nnzhp::Int
  nnzgp::Int
  nnzjplcon::Int
  nnzjpucon::Int
  nnzjplvar::Int
  nnzjpuvar::Int
  grad_param_available::Bool
  jac_param_available::Bool
  hess_param_available::Bool
  jpprod_available::Bool
  jptprod_available::Bool
  hpprod_available::Bool
  hptprod_available::Bool
  lcon_jac_available::Bool
  ucon_jac_available::Bool
  lvar_jac_available::Bool
  uvar_jac_available::Bool
  lcon_jpprod_available::Bool
  ucon_jpprod_available::Bool
  lvar_jpprod_available::Bool
  uvar_jpprod_available::Bool
  lcon_jptprod_available::Bool
  ucon_jptprod_available::Bool
  lvar_jptprod_available::Bool
  uvar_jptprod_available::Bool
end

function ParametricNLPModelMeta(;
  nparam::Int = 0,
  nnzjp::Int = 0,
  nnzhp::Int = 0,
  nnzgp::Int = 0,
  nnzjplcon::Int = 0,
  nnzjpucon::Int = 0,
  nnzjplvar::Int = 0,
  nnzjpuvar::Int = 0,
  grad_param_available::Bool = false,
  jac_param_available::Bool = false,
  hess_param_available::Bool = false,
  jpprod_available::Bool = false,
  jptprod_available::Bool = false,
  hpprod_available::Bool = false,
  hptprod_available::Bool = false,
  lcon_jac_available::Bool = false,
  ucon_jac_available::Bool = false,
  lvar_jac_available::Bool = false,
  uvar_jac_available::Bool = false,
  lcon_jpprod_available::Bool = false,
  ucon_jpprod_available::Bool = false,
  lvar_jpprod_available::Bool = false,
  uvar_jpprod_available::Bool = false,
  lcon_jptprod_available::Bool = false,
  ucon_jptprod_available::Bool = false,
  lvar_jptprod_available::Bool = false,
  uvar_jptprod_available::Bool = false,
)
  dims = (
    nparam,
    nnzjp,
    nnzhp,
    nnzgp,
    nnzjplcon,
    nnzjpucon,
    nnzjplvar,
    nnzjpuvar,
  )
  any(<(0), dims) && error("Nonsensical dimensions")
  return ParametricNLPModelMeta(
    nparam,
    nnzjp,
    nnzhp,
    nnzgp,
    nnzjplcon,
    nnzjpucon,
    nnzjplvar,
    nnzjpuvar,
    grad_param_available,
    jac_param_available,
    hess_param_available,
    jpprod_available,
    jptprod_available,
    hpprod_available,
    hptprod_available,
    lcon_jac_available,
    ucon_jac_available,
    lvar_jac_available,
    uvar_jac_available,
    lcon_jpprod_available,
    ucon_jpprod_available,
    lvar_jpprod_available,
    uvar_jpprod_available,
    lcon_jptprod_available,
    ucon_jptprod_available,
    lvar_jptprod_available,
    uvar_jptprod_available,
  )
end

function ParametricNLPModelMeta(meta::AbstractParametricNLPModelMeta; kwargs...)
  return ParametricNLPModelMeta(
    nparam = get_nparam(meta),
    nnzjp = get_nnzjp(meta),
    nnzhp = get_nnzhp(meta),
    nnzgp = get_nnzgp(meta),
    nnzjplcon = get_nnzjplcon(meta),
    nnzjpucon = get_nnzjpucon(meta),
    nnzjplvar = get_nnzjplvar(meta),
    nnzjpuvar = get_nnzjpuvar(meta),
    grad_param_available = get_grad_param_available(meta),
    jac_param_available = get_jac_param_available(meta),
    hess_param_available = get_hess_param_available(meta),
    jpprod_available = get_jpprod_available(meta),
    jptprod_available = get_jptprod_available(meta),
    hpprod_available = get_hpprod_available(meta),
    hptprod_available = get_hptprod_available(meta),
    lcon_jac_available = get_lcon_jac_available(meta),
    ucon_jac_available = get_ucon_jac_available(meta),
    lvar_jac_available = get_lvar_jac_available(meta),
    uvar_jac_available = get_uvar_jac_available(meta),
    lcon_jpprod_available = get_lcon_jpprod_available(meta),
    ucon_jpprod_available = get_ucon_jpprod_available(meta),
    lvar_jpprod_available = get_lvar_jpprod_available(meta),
    uvar_jpprod_available = get_uvar_jpprod_available(meta),
    lcon_jptprod_available = get_lcon_jptprod_available(meta),
    ucon_jptprod_available = get_ucon_jptprod_available(meta),
    lvar_jptprod_available = get_lvar_jptprod_available(meta),
    uvar_jptprod_available = get_uvar_jptprod_available(meta),
    kwargs...,
  )
end

const EMPTY_PARAMETRIC_META = ParametricNLPModelMeta()

@generated function _get_param_meta_field(nlp::T, ::Val{field}) where {T <: NLPModels.AbstractNLPModel, field}
  if :param_meta in fieldnames(T)
    return :(getproperty(getfield(nlp, :param_meta), $(QuoteNode(field))))
  else
    default = getproperty(EMPTY_PARAMETRIC_META, field)
    return :($default)
  end
end

for field in fieldnames(ParametricNLPModelMeta)
  meth = Symbol("get_", field)
  @eval begin
    $meth(meta::AbstractParametricNLPModelMeta) = getproperty(meta, $(QuoteNode(field)))
    $meth(nlp::NLPModels.AbstractNLPModel) = _get_param_meta_field(nlp, Val($(QuoteNode(field))))
    export $meth
  end
end

let kws = [Expr(:kw, field, Expr(:call, Symbol("get_", field), :nlp)) for field in fieldnames(ParametricNLPModelMeta)]
  @eval function ParametricNLPModelMeta(nlp::NLPModels.AbstractNLPModel; kwargs...)
    return ParametricNLPModelMeta(; $(kws...), kwargs...)
  end
end
