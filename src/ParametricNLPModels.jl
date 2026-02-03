module ParametricNLPModels

import NLPModels: AbstractNLPModel, @lencheck, coo_prod!, @closure, LinearOperator

struct ParametricNLPModelMeta
    nparam::Int
    nnzj::Int      # ∇ₚ g
    nnzh::Int      # ∇ₚ (∇ₓ L)
    nnzjlcon::Int  # ∇ₚ lcon
    nnzjucon::Int  # ∇ₚ ucon
    nnzjlvar::Int  # ∇ₚ lvar
    nnzjuvar::Int  # ∇ₚ uvar
end

ParametricNLPModelMeta() = ParametricNLPModelMeta(0,0,0,0,0,0,0)
ParametricNLPModelMeta(nparam, nnzj, nnzh) = ParametricNLPModelMeta(nparam,nnzj,nnzh,0,0,0,0)

export grad_param, grad_param!,
    jac_param_structure, jac_param_structure!,
    jac_param_coord, jac_param_coord!,
    jpprod, jpprod!,
    hess_param_structure, hess_param_structure!,
    hess_param_coord, hess_param_coord!,
    hpprod, hpprod!,
    lcon_jac_param_structure, lcon_jac_param_structure!,
    lcon_jpprod, lcon_jpprod!,
    ucon_jac_param_structure, ucon_jac_param_structure!,
    ucon_jpprod, ucon_jpprod!,
    lvar_jac_param_structure, lvar_jac_param_structure!,
    lvar_jpprod, lvar_jpprod!,
    uvar_jac_param_structure, uvar_jac_param_structure!,
    uvar_jpprod, uvar_jpprod!


"""
    g = grad_param(nlp, x)

Evaluate `∇ₚf(x)`, the gradient of the objective function at `x` wrt parameters.
"""
function grad_param(nlp::AbstractNLPModel{T, S}, x::AbstractVector) where {T, S}
  @lencheck nlp.meta.nvar x
  g = S(undef, nlp.pmeta.nparam)
  return grad_param!(nlp, x, g)
end

"""
    g = grad_param!(nlp, x, g)

Evaluate `∇ₚf(x)`, the gradient of the objective function at `x` wrt parameters in place.
"""
function grad_param! end

"""
    (rows,cols) = jac_param_structure(nlp)

Return the structure of the constraints Jacobian wrt parameters in sparse coordinate format.
"""
function jac_param_structure(nlp::AbstractNLPModel)
  rows = Vector{Int}(undef, nlp.meta.nnzj)
  cols = Vector{Int}(undef, nlp.meta.nnzj)
  jac_param_structure!(nlp, rows, cols)
end

"""
    jac_param_structure!(nlp, rows, cols)

Return the structure of the constraints Jacobian wrt parameters in sparse coordinate format in place.
"""
function jac_param_structure! end

"""
    vals = jac_param_coord!(nlp, x, vals)

Evaluate ``Jₚ(x)``, the constraints Jacobian wrt parameters at `x` in sparse coordinate format,
rewriting `vals`.
"""
function jac_param_coord! end

"""
    vals = jac_param_coord(nlp, x)

Evaluate ``Jₚ(x)``, the constraints Jacobian wrt parameters at `x` in sparse coordinate format.
"""
function jac_param_coord(nlp::AbstractNLPModel{T, S}, x::AbstractVector) where {T, S}
  @lencheck nlp.meta.nvar x
  vals = S(undef, nlp.pmeta.nnzj)
  return jac_param_coord!(nlp, x, vals)
end

"""
    Jv = jpprod(nlp, x, v)

Evaluate ``Jₚ(x)v``, the parametric Jacobian-vector product at `x`.
"""
function jpprod(nlp::AbstractNLPModel{T, S}, x::AbstractVector, v::AbstractVector) where {T, S}
  @lencheck nlp.meta.nvar x
  @lencheck nlp.pmeta.nparam v
  Jv = S(undef, nlp.meta.ncon)
  return jpprod!(nlp, x, v, Jv)
end

"""
    Jv = jpprod!(nlp, x, v, Jv)

Evaluate ``Jₚ(x)v``, the parametric Jacobian-vector product at `x` in place.
"""
function jpprod! end

"""
    (rows,cols) = hess_param_structure(nlp)

Return the structure of the Lagrangian variable-parameter Hessian in sparse coordinate format.
"""
function hess_param_structure(nlp::AbstractNLPModel)
  rows = Vector{Int}(undef, nlp.pmeta.nnzh)
  cols = Vector{Int}(undef, nlp.pmeta.nnzh)
  hess_param_structure!(nlp, rows, cols)
end

"""
    hess_param_structure!(nlp, rows, cols)

Return the structure of the Lagrangian variable-parameter Hessian in sparse coordinate format in place.
"""
function hess_param_structure! end

"""
    vals = hess_param_coord!(nlp, x, vals; obj_weight=1.0)

Evaluate the variable-parameter objective Hessian at `x` in sparse coordinate format,
with objective function scaled by `obj_weight`, overwriting `vals`.
"""
function hess_param_coord!(
  nlp::AbstractNLPModel{T, S},
  x::AbstractVector{T},
  vals::AbstractVector;
  obj_weight::Real = one(T),
) where {T, S}
  @lencheck nlp.meta.nvar x
  @lencheck nlp.pmeta.nnzh vals
  y = fill!(S(undef, nlp.meta.ncon), 0)
  hess_param_coord!(nlp, x, y, vals, obj_weight = obj_weight)
end

"""
    vals = hess_param_coord!(nlp, x, y, vals; obj_weight=1.0)

Evaluate the Lagrangian variable-parameter Hessian at `(x,y)` in sparse coordinate format,
with objective function scaled by `obj_weight`, overwriting `vals`.
"""
function hess_param_coord! end

"""
    vals = hess_param_coord(nlp, x; obj_weight=1.0)

Evaluate the variable-parameter objective Hessian at `x` in sparse coordinate format,
with objective function scaled by `obj_weight`.
"""
function hess_param_coord(
  nlp::AbstractNLPModel{T, S},
  x::AbstractVector;
  obj_weight::Real = one(T),
) where {T, S}
  @lencheck nlp.meta.nvar x
  vals = S(undef, nlp.pmeta.nnzh)
  return hess_param_coord!(nlp, x, vals; obj_weight = obj_weight)
end

"""
    vals = hess_param_coord(nlp, x, y; obj_weight=1.0)

Evaluate the Lagrangian variable-parameter Hessian at `(x,y)` in sparse coordinate format,
with objective function scaled by `obj_weight`.
"""
function hess_param_coord(
  nlp::AbstractNLPModel{T, S},
  x::AbstractVector,
  y::AbstractVector;
  obj_weight::Real = one(T),
) where {T, S}
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.ncon y
  vals = S(undef, nlp.pmeta.nnzh)
  return hess_param_coord!(nlp, x, y, vals; obj_weight = obj_weight)
end

"""
    Hv = hpprod(nlp, x, v; obj_weight=1.0)

Evaluate the product of the objective variable-parameter Hessian at `x` with the vector `v`,
with objective function scaled by `obj_weight`.
"""
function hpprod(
  nlp::AbstractNLPModel{T, S},
  x::AbstractVector,
  v::AbstractVector;
  obj_weight::Real = one(T),
) where {T, S}
  @lencheck nlp.meta.nvar x
  @lencheck nlp.pmeta.nparam v
  Hv = S(undef, nlp.meta.nvar)
  return hpprod!(nlp, x, v, Hv; obj_weight = obj_weight)
end

"""
    Hv = hpprod(nlp, x, y, v; obj_weight=1.0)

Evaluate the product of the Lagrangian variable-parameter Hessian at `(x,y)` with the vector `v`,
with objective function scaled by `obj_weight`.
"""
function hpprod(
  nlp::AbstractNLPModel{T, S},
  x::AbstractVector,
  y::AbstractVector,
  v::AbstractVector;
  obj_weight::Real = one(T),
) where {T, S}
  @lencheck nlp.meta.nvar x
  @lencheck nlp.pmeta.nparam v
  @lencheck nlp.meta.ncon y
  Hv = S(undef, nlp.meta.nvar)
  return hpprod!(nlp, x, y, v, Hv; obj_weight = obj_weight)
end

"""
    Hv = hpprod!(nlp, x, v, Hv; obj_weight=1.0)

Evaluate the product of the objective variable-parameter Hessian at `x` with the vector `v` in
place, with objective function scaled by `obj_weight`.
"""
function hpprod!(
  nlp::AbstractNLPModel{T, S},
  x::AbstractVector,
  v::AbstractVector,
  Hv::AbstractVector;
  obj_weight::Real = one(T),
) where {T, S}
  @lencheck nlp.meta.nvar x Hv
  @lencheck nlp.pmeta.nparam v
  y = fill!(S(undef, nlp.meta.ncon), 0)
  hpprod!(nlp, x, y, v, Hv, obj_weight = obj_weight)
end

"""
    Hv = hpprod!(nlp, rows, cols, vals, v, Hv)

Evaluate the product of the objective or Lagrangian variable-parameter Hessian given by `(rows, cols, vals)` in
triplet format with the vector `v` in place.
"""
function hpprod!(
  nlp::AbstractNLPModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
  vals::AbstractVector,
  v::AbstractVector,
  Hv::AbstractVector,
)
  @lencheck nlp.meta.nnzh rows cols vals
  @lencheck nlp.meta.nvar Hv
  @lencheck nlp.pmeta.nparam v
#   increment!(nlp, :neval_hpprod)
  coo_prod!(cols, rows, vals, v, Hv)
end

"""
    Hv = hpprod!(nlp, x, y, v, Hv; obj_weight=1.0)

Evaluate the product of the Lagrangian variable-parameter Hessian at `(x,y)` with the vector `v` in
place, with objective function scaled by `obj_weight`.
"""
function hpprod! end

# NOTE: we implement zero defaults for lcon/ucon/lvar/uvar since in most NLPModels, they are assumed constant/non-parametric.

"""
    (rows,cols) = lcon_jac_param_structure(nlp)

Return the structure of the constraint lower bound Jacobian wrt parameters in sparse coordinate format.
"""
function lcon_jac_param_structure(nlp::AbstractNLPModel)
    rows = Vector{Int}(undef, nlp.pmeta.nnzjlcon)
    cols = Vector{Int}(undef, nlp.pmeta.nnzjlcon)
    lcon_jac_param_structure!(nlp, rows, cols)
    return rows, cols
end

function lcon_jac_param_structure! end

"""
    (rows,cols) = ucon_jac_param_structure(nlp)

Return the structure of the constraint upper bound Jacobian wrt parameters in sparse coordinate format.
"""
function ucon_jac_param_structure(nlp::AbstractNLPModel)
    rows = Vector{Int}(undef, nlp.pmeta.nnzjucon)
    cols = Vector{Int}(undef, nlp.pmeta.nnzjucon)
    ucon_jac_param_structure!(nlp, rows, cols)
    return rows, cols
end

function ucon_jac_param_structure! end

"""
    (rows,cols) = lvar_jac_param_structure(nlp)

Return the structure of the variable lower bound Jacobian wrt parameters in sparse coordinate format.
"""
function lvar_jac_param_structure(nlp::AbstractNLPModel)
    rows = Vector{Int}(undef, nlp.pmeta.nnzjlvar)
    cols = Vector{Int}(undef, nlp.pmeta.nnzjlvar)
    lvar_jac_param_structure!(nlp, rows, cols)
    return rows, cols
end

function lvar_jac_param_structure! end

"""
    (rows,cols) = uvar_jac_param_structure(nlp)

Return the structure of the variable upper bound Jacobian wrt parameters in sparse coordinate format.
"""
function uvar_jac_param_structure(nlp::AbstractNLPModel)
    rows = Vector{Int}(undef, nlp.pmeta.nnzjuvar)
    cols = Vector{Int}(undef, nlp.pmeta.nnzjuvar)
    uvar_jac_param_structure!(nlp, rows, cols)
    return rows, cols
end

function uvar_jac_param_structure! end

"""
    Jv = lcon_jpprod(nlp, v)

Evaluate `∇ₚlcon`, the gradient of the constraint lower bound wrt parameters.
"""
function lcon_jpprod(nlp::AbstractNLPModel{T, S}, v::AbstractVector) where {T, S}
    @lencheck nlp.pmeta.nparam v
    Jv = S(undef, nlp.meta.ncon)
    return lcon_jpprod!(nlp, v, Jv)
end

"""
    Jv = lcon_jpprod!(nlp, v, Jv)

Evaluate `∇ₚlcon`, the gradient of the constraint lower bound wrt parameters in place.
"""
function lcon_jpprod!(nlp, v, Jv)
    fill!(Jv, zero(eltype(Jv)))
    return Jv
end

"""
    Jv = ucon_jpprod(nlp, v)

Evaluate `∇ₚucon`, the gradient of the constraint upper bound wrt parameters.
"""
function ucon_jpprod(nlp::AbstractNLPModel{T, S}, v::AbstractVector) where {T, S}
    @lencheck nlp.pmeta.nparam v
    Jv = S(undef, nlp.meta.ncon)
    return ucon_jpprod!(nlp, v, Jv)
end

"""
    Jv = ucon_jpprod!(nlp, v, Jv)

Evaluate `∇ₚucon`, the gradient of the constraint upper bound wrt parameters in place.
"""
function ucon_jpprod!(nlp, v, Jv)
    fill!(Jv, zero(eltype(Jv)))
    return Jv
end

"""
    Jv = lvar_jpprod(nlp, v)

Evaluate `∇ₚlvar`, the gradient of the variable lower bound wrt parameters.
"""
function lvar_jpprod(nlp::AbstractNLPModel{T, S}, v::AbstractVector) where {T, S}
    @lencheck nlp.pmeta.nparam v
    Jv = S(undef, nlp.meta.nvar)
    return lvar_jpprod!(nlp, v, Jv)
end

"""
    Jv = lvar_jpprod!(nlp, v, Jv)

Evaluate `∇ₚlvar`, the gradient of the variable lower bound wrt parameters in place.
"""
function lvar_jpprod!(nlp, v, Jv)
    fill!(Jv, zero(eltype(Jv)))
    return Jv
end

"""
    Jv = uvar_jpprod(nlp, v)

Evaluate `∇ₚuvar`, the gradient of the variable upper bound wrt parameters.
"""
function uvar_jpprod(nlp::AbstractNLPModel{T, S}, v::AbstractVector) where {T, S}
    Jv = S(undef, nlp.meta.nvar)
    return uvar_jpprod!(nlp, v, Jv)
end

"""
    Jv = uvar_jpprod!(nlp, v, Jv)

Evaluate `∇ₚuvar`, the gradient of the variable upper bound wrt parameters in place.
"""
function uvar_jpprod!(nlp, v, Jv)
    @lencheck nlp.pmeta.nparam v
    fill!(Jv, zero(eltype(Jv)))
    return Jv
end

end # module ParametricNLPModels
