"""
    grad_param(nlp, x)

Evaluate ``∇ₚf(x, p)``, the gradient of the objective function with respect to the parameters, at `x`.
"""
function grad_param(nlp::NLPModels.AbstractNLPModel{T, S}, x::AbstractVector) where {T, S}
  @lencheck NLPModels.get_nvar(nlp) x
  g = S(undef, get_nparam(nlp))
  return grad_param!(nlp, x, g)
end

"""
    jac_param_structure(nlp)

Return the structure of ``Jₚ(x, p)``, the constraints Jacobian with respect to the parameters, in sparse coordinate format.
"""
function jac_param_structure(nlp::NLPModels.AbstractNLPModel)
  rows = Vector{Int}(undef, get_nnzjp(nlp))
  cols = Vector{Int}(undef, get_nnzjp(nlp))
  jac_param_structure!(nlp, rows, cols)
end

"""
    jac_param_coord(nlp, x)

Evaluate ``Jₚ(x, p)``, the constraints Jacobian with respect to the parameters, at `x` in sparse coordinate format.
"""
function jac_param_coord(nlp::NLPModels.AbstractNLPModel{T, S}, x::AbstractVector) where {T, S}
  @lencheck NLPModels.get_nvar(nlp) x
  vals = S(undef, get_nnzjp(nlp))
  return jac_param_coord!(nlp, x, vals)
end

"""
    jpprod(nlp, x, v)

Evaluate ``Jₚ(x, p)v``, the parameter-Jacobian-vector product at `x`.
"""
function jpprod(nlp::NLPModels.AbstractNLPModel{T, S}, x::AbstractVector, v::AbstractVector) where {T, S}
  @lencheck NLPModels.get_nvar(nlp) x
  @lencheck get_nparam(nlp) v
  Jv = S(undef, NLPModels.get_ncon(nlp))
  return jpprod!(nlp, x, v, Jv)
end

"""
    jptprod(nlp, x, v)

Evaluate ``Jₚ(x, p)ᵀv``, the transposed-parameter-Jacobian-vector product at `x`.
"""
function jptprod(nlp::NLPModels.AbstractNLPModel{T, S}, x::AbstractVector, v::AbstractVector) where {T, S}
  @lencheck NLPModels.get_nvar(nlp) x
  @lencheck NLPModels.get_ncon(nlp) v
  Jtv = S(undef, get_nparam(nlp))
  return jptprod!(nlp, x, v, Jtv)
end

"""
    hess_param_structure(nlp)

Return the structure of ``∇ₓₚL(x, y, p)``, the mixed block of the Lagrangian Hessian, in sparse coordinate format.
"""
function hess_param_structure(nlp::NLPModels.AbstractNLPModel)
  rows = Vector{Int}(undef, get_nnzhp(nlp))
  cols = Vector{Int}(undef, get_nnzhp(nlp))
  hess_param_structure!(nlp, rows, cols)
end

"""
    hess_param_coord!(nlp, x, vals; obj_weight = 1)

Evaluate ``∇ₓₚL(x, 0, p)``, the mixed block of the Lagrangian Hessian with zero constraint multipliers, in sparse coordinate format in place.
"""
function hess_param_coord!(
  nlp::NLPModels.AbstractNLPModel{T, S},
  x::AbstractVector{T},
  vals::AbstractVector;
  obj_weight::Real = one(T),
) where {T, S}
  @lencheck NLPModels.get_nvar(nlp) x
  @lencheck get_nnzhp(nlp) vals
  y = fill!(S(undef, NLPModels.get_ncon(nlp)), 0)
  hess_param_coord!(nlp, x, y, vals, obj_weight = obj_weight)
end

"""
    hess_param_coord(nlp, x; obj_weight = 1)

Evaluate ``∇ₓₚL(x, 0, p)``, the mixed block of the Lagrangian Hessian with zero constraint multipliers, in sparse coordinate format.
"""
function hess_param_coord(
  nlp::NLPModels.AbstractNLPModel{T, S},
  x::AbstractVector;
  obj_weight::Real = one(T),
) where {T, S}
  @lencheck NLPModels.get_nvar(nlp) x
  vals = S(undef, get_nnzhp(nlp))
  return hess_param_coord!(nlp, x, vals; obj_weight = obj_weight)
end

"""
    hess_param_coord(nlp, x, y; obj_weight = 1)

Evaluate ``∇ₓₚL(x, y, p)``, the mixed block of the Lagrangian Hessian at `(x, y)`, in sparse coordinate format.
"""
function hess_param_coord(
  nlp::NLPModels.AbstractNLPModel{T, S},
  x::AbstractVector,
  y::AbstractVector;
  obj_weight::Real = one(T),
) where {T, S}
  @lencheck NLPModels.get_nvar(nlp) x
  @lencheck NLPModels.get_ncon(nlp) y
  vals = S(undef, get_nnzhp(nlp))
  return hess_param_coord!(nlp, x, y, vals; obj_weight = obj_weight)
end

"""
    hpprod(nlp, x, v; obj_weight = 1)

Evaluate ``∇ₓₚL(x, 0, p)v``, the mixed-Hessian-vector product with zero constraint multipliers.
"""
function hpprod(
  nlp::NLPModels.AbstractNLPModel{T, S},
  x::AbstractVector,
  v::AbstractVector;
  obj_weight::Real = one(T),
) where {T, S}
  @lencheck NLPModels.get_nvar(nlp) x
  @lencheck get_nparam(nlp) v
  Hv = S(undef, NLPModels.get_nvar(nlp))
  return hpprod!(nlp, x, v, Hv; obj_weight = obj_weight)
end

"""
    hpprod(nlp, x, y, v; obj_weight = 1)

Evaluate ``∇ₓₚL(x, y, p)v``, the mixed-Hessian-vector product at `(x, y)`.
"""
function hpprod(
  nlp::NLPModels.AbstractNLPModel{T, S},
  x::AbstractVector,
  y::AbstractVector,
  v::AbstractVector;
  obj_weight::Real = one(T),
) where {T, S}
  @lencheck NLPModels.get_nvar(nlp) x
  @lencheck get_nparam(nlp) v
  @lencheck NLPModels.get_ncon(nlp) y
  Hv = S(undef, NLPModels.get_nvar(nlp))
  return hpprod!(nlp, x, y, v, Hv; obj_weight = obj_weight)
end

"""
    hpprod!(nlp, x, v, Hv; obj_weight = 1)

Evaluate ``∇ₓₚL(x, 0, p)v``, the mixed-Hessian-vector product with zero constraint multipliers, in place.
"""
function hpprod!(
  nlp::NLPModels.AbstractNLPModel{T, S},
  x::AbstractVector,
  v::AbstractVector,
  Hv::AbstractVector;
  obj_weight::Real = one(T),
) where {T, S}
  @lencheck NLPModels.get_nvar(nlp) x Hv
  @lencheck get_nparam(nlp) v
  y = fill!(S(undef, NLPModels.get_ncon(nlp)), 0)
  hpprod!(nlp, x, y, v, Hv, obj_weight = obj_weight)
end

"""
    hptprod(nlp, x, y, v; obj_weight = 1)

Evaluate ``∇ₓₚL(x, y, p)ᵀv``, the transposed-mixed-Hessian-vector product at `(x, y)`.
"""
function hptprod(
  nlp::NLPModels.AbstractNLPModel{T, S},
  x::AbstractVector,
  y::AbstractVector,
  v::AbstractVector;
  obj_weight::Real = one(T),
) where {T, S}
  @lencheck NLPModels.get_nvar(nlp) x v
  @lencheck NLPModels.get_ncon(nlp) y
  Htv = S(undef, get_nparam(nlp))
  return hptprod!(nlp, x, y, v, Htv; obj_weight = obj_weight)
end

"""
    lcon_jac_param_structure(nlp)

Return the structure of ``∂ℓᶜ(p) / ∂p`` in sparse coordinate format.
"""
function lcon_jac_param_structure(nlp::NLPModels.AbstractNLPModel)
  rows = Vector{Int}(undef, get_nnzjplcon(nlp))
  cols = Vector{Int}(undef, get_nnzjplcon(nlp))
  lcon_jac_param_structure!(nlp, rows, cols)
end

"""
    lcon_jac_param_coord(nlp)

Evaluate ``∂ℓᶜ(p) / ∂p`` in sparse coordinate format.
"""
function lcon_jac_param_coord(nlp::NLPModels.AbstractNLPModel{T, S}) where {T, S}
  vals = S(undef, get_nnzjplcon(nlp))
  return lcon_jac_param_coord!(nlp, vals)
end

"""
    lcon_jpprod(nlp, v)

Evaluate ``(∂ℓᶜ(p) / ∂p)v``.
"""
function lcon_jpprod(nlp::NLPModels.AbstractNLPModel{T, S}, v::AbstractVector) where {T, S}
  @lencheck get_nparam(nlp) v
  Jv = S(undef, NLPModels.get_ncon(nlp))
  return lcon_jpprod!(nlp, v, Jv)
end

"""
    lcon_jptprod(nlp, v)

Evaluate ``(∂ℓᶜ(p) / ∂p)ᵀv``.
"""
function lcon_jptprod(nlp::NLPModels.AbstractNLPModel{T, S}, v::AbstractVector) where {T, S}
  @lencheck NLPModels.get_ncon(nlp) v
  Jtv = S(undef, get_nparam(nlp))
  return lcon_jptprod!(nlp, v, Jtv)
end

"""
    ucon_jac_param_structure(nlp)

Return the structure of ``∂uᶜ(p) / ∂p`` in sparse coordinate format.
"""
function ucon_jac_param_structure(nlp::NLPModels.AbstractNLPModel)
  rows = Vector{Int}(undef, get_nnzjpucon(nlp))
  cols = Vector{Int}(undef, get_nnzjpucon(nlp))
  ucon_jac_param_structure!(nlp, rows, cols)
end

"""
    ucon_jac_param_coord(nlp)

Evaluate ``∂uᶜ(p) / ∂p`` in sparse coordinate format.
"""
function ucon_jac_param_coord(nlp::NLPModels.AbstractNLPModel{T, S}) where {T, S}
  vals = S(undef, get_nnzjpucon(nlp))
  return ucon_jac_param_coord!(nlp, vals)
end

"""
    ucon_jpprod(nlp, v)

Evaluate ``(∂uᶜ(p) / ∂p)v``.
"""
function ucon_jpprod(nlp::NLPModels.AbstractNLPModel{T, S}, v::AbstractVector) where {T, S}
  @lencheck get_nparam(nlp) v
  Jv = S(undef, NLPModels.get_ncon(nlp))
  return ucon_jpprod!(nlp, v, Jv)
end

"""
    ucon_jptprod(nlp, v)

Evaluate ``(∂uᶜ(p) / ∂p)ᵀv``.
"""
function ucon_jptprod(nlp::NLPModels.AbstractNLPModel{T, S}, v::AbstractVector) where {T, S}
  @lencheck NLPModels.get_ncon(nlp) v
  Jtv = S(undef, get_nparam(nlp))
  return ucon_jptprod!(nlp, v, Jtv)
end

"""
    lvar_jac_param_structure(nlp)

Return the structure of ``∂ℓˣ(p) / ∂p`` in sparse coordinate format.
"""
function lvar_jac_param_structure(nlp::NLPModels.AbstractNLPModel)
  rows = Vector{Int}(undef, get_nnzjplvar(nlp))
  cols = Vector{Int}(undef, get_nnzjplvar(nlp))
  lvar_jac_param_structure!(nlp, rows, cols)
end

"""
    lvar_jac_param_coord(nlp)

Evaluate ``∂ℓˣ(p) / ∂p`` in sparse coordinate format.
"""
function lvar_jac_param_coord(nlp::NLPModels.AbstractNLPModel{T, S}) where {T, S}
  vals = S(undef, get_nnzjplvar(nlp))
  return lvar_jac_param_coord!(nlp, vals)
end

"""
    lvar_jpprod(nlp, v)

Evaluate ``(∂ℓˣ(p) / ∂p)v``.
"""
function lvar_jpprod(nlp::NLPModels.AbstractNLPModel{T, S}, v::AbstractVector) where {T, S}
  @lencheck get_nparam(nlp) v
  Jv = S(undef, NLPModels.get_nvar(nlp))
  return lvar_jpprod!(nlp, v, Jv)
end

"""
    lvar_jptprod(nlp, v)

Evaluate ``(∂ℓˣ(p) / ∂p)ᵀv``.
"""
function lvar_jptprod(nlp::NLPModels.AbstractNLPModel{T, S}, v::AbstractVector) where {T, S}
  @lencheck NLPModels.get_nvar(nlp) v
  Jtv = S(undef, get_nparam(nlp))
  return lvar_jptprod!(nlp, v, Jtv)
end

"""
    uvar_jac_param_structure(nlp)

Return the structure of ``∂uˣ(p) / ∂p`` in sparse coordinate format.
"""
function uvar_jac_param_structure(nlp::NLPModels.AbstractNLPModel)
  rows = Vector{Int}(undef, get_nnzjpuvar(nlp))
  cols = Vector{Int}(undef, get_nnzjpuvar(nlp))
  uvar_jac_param_structure!(nlp, rows, cols)
end

"""
    uvar_jac_param_coord(nlp)

Evaluate ``∂uˣ(p) / ∂p`` in sparse coordinate format.
"""
function uvar_jac_param_coord(nlp::NLPModels.AbstractNLPModel{T, S}) where {T, S}
  vals = S(undef, get_nnzjpuvar(nlp))
  return uvar_jac_param_coord!(nlp, vals)
end

"""
    uvar_jpprod(nlp, v)

Evaluate ``(∂uˣ(p) / ∂p)v``.
"""
function uvar_jpprod(nlp::NLPModels.AbstractNLPModel{T, S}, v::AbstractVector) where {T, S}
  @lencheck get_nparam(nlp) v
  Jv = S(undef, NLPModels.get_nvar(nlp))
  return uvar_jpprod!(nlp, v, Jv)
end

"""
    uvar_jptprod(nlp, v)

Evaluate ``(∂uˣ(p) / ∂p)ᵀv``.
"""
function uvar_jptprod(nlp::NLPModels.AbstractNLPModel{T, S}, v::AbstractVector) where {T, S}
  @lencheck NLPModels.get_nvar(nlp) v
  Jtv = S(undef, get_nparam(nlp))
  return uvar_jptprod!(nlp, v, Jtv)
end
