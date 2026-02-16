
export grad_param, grad_param!,
    jac_param, jac_param!,
    jac_param_structure, jac_param_structure!,
    jac_param_coord, jac_param_coord!,
    jpprod, jpprod!,
    jptprod, jptprod!,
    hess_param, hess_param!,
    hess_param_structure, hess_param_structure!,
    hess_param_coord, hess_param_coord!,
    hpprod, hpprod!,
    hptprod, hptprod!,
    lcon_jac, lcon_jac!,
    lcon_jac_param_structure, lcon_jac_param_structure!,
    lcon_jpprod, lcon_jpprod!,
    lcon_jptprod, lcon_jptprod!,
    ucon_jac, ucon_jac!,
    ucon_jac_param_structure, ucon_jac_param_structure!,
    ucon_jpprod, ucon_jpprod!,
    ucon_jptprod, ucon_jptprod!,
    lvar_jac_param, lvar_jac!_param,
    lvar_jac_param_structure, lvar_jac_param_structure!,
    lvar_jpprod, lvar_jpprod!,
    lvar_jptprod, lvar_jptprod!,
    uvar_jac_param, uvar_jac!_param,
    uvar_jac_param_structure, uvar_jac_param_structure!,
    uvar_jpprod, uvar_jpprod!,
    uvar_jptprod, uvar_jptprod!


"""
    g = grad_param(nlp, x)

Evaluate `∇ₚf(x)`, the gradient of the objective function at `x` wrt parameters.
This function is only available if `nlp.pmeta.grad_param_available` is set to `true`.
"""
function grad_param(nlp::AbstractNLPModel{T, S}, x::AbstractVector) where {T, S}
  @lencheck nlp.meta.nvar x
  g = S(undef, nlp.pmeta.nparam)
  return grad_param!(nlp, x, g)
end

"""
    g = grad_param!(nlp, x, g)

Evaluate `∇ₚf(x)`, the gradient of the objective function at `x` wrt parameters in place.
This function is only available if `nlp.pmeta.grad_param_available` is set to `true`.
"""
function grad_param! end

"""
    (rows,cols) = jac_param_structure(nlp)

Return the structure of the constraints Jacobian wrt parameters in sparse coordinate format.
This function is only available if `nlp.pmeta.jac_param_available` is set to `true`.
"""
function jac_param_structure(nlp::AbstractNLPModel)
  rows = Vector{Int}(undef, nlp.pmeta.nnzj)
  cols = Vector{Int}(undef, nlp.pmeta.nnzj)
  jac_param_structure!(nlp, rows, cols)
end

"""
    jac_param_structure!(nlp, rows, cols)

Return the structure of the constraints Jacobian wrt parameters in sparse coordinate format in place.
This function is only available if `nlp.pmeta.jac_param_available` is set to `true`.
"""
function jac_param_structure! end

"""
    vals = jac_param_coord!(nlp, x, vals)

Evaluate ``Jₚ(x)``, the constraints Jacobian wrt parameters at `x` in sparse coordinate format, rewriting `vals`.
This function is only available if `nlp.pmeta.jac_param_available` is set to `true`.
"""
function jac_param_coord! end

"""
    vals = jac_param_coord(nlp, x)

Evaluate ``Jₚ(x)``, the constraints Jacobian wrt parameters at `x` in sparse coordinate format.
This function is only available if `nlp.pmeta.jac_param_available` is set to `true`.
"""
function jac_param_coord(nlp::AbstractNLPModel{T, S}, x::AbstractVector) where {T, S}
  @lencheck nlp.meta.nvar x
  vals = S(undef, nlp.pmeta.nnzj)
  return jac_param_coord!(nlp, x, vals)
end

"""
    J = jac_param(nlp, x)

Evaluate the constraints Jacobian wrt parameters ``Jₚ(x)`` as a sparse matrix.
This function is only available if `nlp.pmeta.jac_param_available` is set to `true`.
"""
function jac_param(nlp::AbstractNLPModel, x::AbstractVector)
  @lencheck nlp.meta.nvar x
  rows, cols = jac_param_structure(nlp)
  vals = jac_param_coord(nlp, x)
  sparse(rows, cols, vals, nlp.meta.ncon, nlp.pmeta.nparam)
end

"""
    J = jac_param!(nlp, x, J)

Evaluate the constraints Jacobian wrt parameters ``Jₚ(x)`` in place.
This function is only available if `nlp.pmeta.jac_param_available` is set to `true`.
"""
function jac_param! end

"""
    Jv = jpprod(nlp, x, v)

Evaluate ``Jₚ(x)v``, the parametric Jacobian-vector product at `x`.
This function is only available if `nlp.pmeta.jpprod_available` is set to `true`.
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
This function is only available if `nlp.pmeta.jptprod_available` is set to `true`.
"""
function jpprod! end

"""
    Jtv = jptprod(nlp, x, v)

Evaluate ``Jₚ(x)v``, the parametric Jacobian-vector product at `x`.
This function is only available if `nlp.pmeta.jptprod_available` is set to `true`.
"""
function jptprod(nlp::AbstractNLPModel{T, S}, x::AbstractVector, v::AbstractVector) where {T, S}
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.ncon v
  Jtv = S(undef, nlp.pmeta.nparam)
  return jptprod!(nlp, x, v, Jtv)
end

"""
    Jtv = jptprod!(nlp, x, v, Jtv)

Evaluate ``Jₚ(x)v``, the parametric Jacobian-vector product at `x` in place.
This function is only available if `nlp.pmeta.jptprod_available` is set to `true`.
"""
function jptprod! end

"""
    (rows,cols) = hess_param_structure(nlp)

Return the structure of the Lagrangian variable-parameter Hessian in sparse coordinate format.
This function is only available if `nlp.pmeta.hess_param_available` is set to `true`.
"""
function hess_param_structure(nlp::AbstractNLPModel)
  rows = Vector{Int}(undef, nlp.pmeta.nnzh)
  cols = Vector{Int}(undef, nlp.pmeta.nnzh)
  hess_param_structure!(nlp, rows, cols)
end

"""
    hess_param_structure!(nlp, rows, cols)

Return the structure of the Lagrangian variable-parameter Hessian in sparse coordinate format in place.
This function is only available if `nlp.pmeta.hess_param_available` is set to `true`.
"""
function hess_param_structure! end

"""
    vals = hess_param_coord!(nlp, x, vals; obj_weight=1.0)

Evaluate the variable-parameter objective Hessian at `x` in sparse coordinate format,
with objective function scaled by `obj_weight`, overwriting `vals`.
This function is only available if `nlp.pmeta.hess_param_available` is set to `true`.
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
This function is only available if `nlp.pmeta.hess_param_available` is set to `true`.
"""
function hess_param_coord! end

"""
    vals = hess_param_coord(nlp, x; obj_weight=1.0)

Evaluate the variable-parameter objective Hessian at `x` in sparse coordinate format,
with objective function scaled by `obj_weight`.
This function is only available if `nlp.pmeta.hess_param_available` is set to `true`.
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
This function is only available if `nlp.pmeta.hess_param_available` is set to `true`.
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
    H = hess_param(nlp, x; obj_weight=1.0)

Evaluate the objective variable-parameter Hessian at `x` as a sparse matrix.
This function is only available if `nlp.pmeta.hess_param_available` is set to `true`.
"""
function hess_param(
  nlp::AbstractNLPModel{T, S},
  x::AbstractVector;
  obj_weight::Real = one(T),
) where {T, S}
  @lencheck nlp.meta.nvar x
  rows, cols = hess_param_structure(nlp)
  vals = hess_param_coord(nlp, x, obj_weight = obj_weight)
  sparse(rows, cols, vals, nlp.meta.nvar, nlp.pmeta.nparam)
end

"""
    H = hess_param(nlp, x, y; obj_weight=1.0)

Evaluate the Lagrangian variable-parameter Hessian at `(x,y)` as a sparse matrix.
This function is only available if `nlp.pmeta.hess_param_available` is set to `true`.
"""
function hess_param(
  nlp::AbstractNLPModel{T, S},
  x::AbstractVector,
  y::AbstractVector;
  obj_weight::Real = one(T),
) where {T, S}
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.ncon y
  rows, cols = hess_param_structure(nlp)
  vals = hess_param_coord(nlp, x, y, obj_weight = obj_weight)
  sparse(rows, cols, vals, nlp.meta.nvar, nlp.pmeta.nparam)
end

"""
    H = hess_param!(nlp, x, y, H; obj_weight=1.0)

Evaluate the Lagrangian variable-parameter Hessian at `(x,y)` in place.
This function is only available if `nlp.pmeta.hess_param_available` is set to `true`.
"""
function hess_param! end

"""
    Hv = hpprod(nlp, x, v; obj_weight=1.0)

Evaluate the product of the objective variable-parameter Hessian at `x` with the vector `v`,
with objective function scaled by `obj_weight`.
This function is only available if `nlp.pmeta.hpprod_available` is set to `true`.
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
This function is only available if `nlp.pmeta.hpprod_available` is set to `true`.
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
This function is only available if `nlp.pmeta.hpprod_available` is set to `true`.
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
    Hv = hpprod!(nlp, x, y, v, Hv; obj_weight=1.0)

Evaluate the product of the Lagrangian variable-parameter Hessian at `(x,y)` with the vector `v` in
place, with objective function scaled by `obj_weight`.
This function is only available if `nlp.pmeta.hpprod_available` is set to `true`.
"""
function hpprod! end

"""
    Htv = hptprod(nlp, x, y, v; obj_weight=1.0)

Evaluate the product of the Lagrangian variable-parameter Hessian at `(x,y)` with the vector `v`,
with objective function scaled by `obj_weight`.
This function is only available if `nlp.pmeta.hptprod_available` is set to `true`.
"""
function hptprod(
  nlp::AbstractNLPModel{T, S},
  x::AbstractVector,
  y::AbstractVector,
  v::AbstractVector;
  obj_weight::Real = one(T),
) where {T, S}
  @lencheck nlp.meta.nvar x v
  @lencheck nlp.meta.ncon y
  Htv = S(undef, nlp.pmeta.nparam)
  return hptprod!(nlp, x, y, v, Htv; obj_weight = obj_weight)
end


"""
    Htv = hptprod!(nlp, x, y, v, Htv; obj_weight=1.0)

Evaluate the product of the Lagrangian variable-parameter Hessian at `(x,y)` with the vector `v` in
place, with objective function scaled by `obj_weight`.
This function is only available if `nlp.pmeta.hptprod_available` is set to `true`.
"""
function hptprod! end

"""
    (rows,cols) = lcon_jac_param_structure(nlp)

Return the structure of the constraint lower bound Jacobian wrt parameters in sparse coordinate format.
This function is only available if `nlp.pmeta.lcon_jac_available` is set to `true`.
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
This function is only available if `nlp.pmeta.ucon_jac_available` is set to `true`.
"""
function ucon_jac_param_structure(nlp::AbstractNLPModel)
    rows = Vector{Int}(undef, nlp.pmeta.nnzjucon)
    cols = Vector{Int}(undef, nlp.pmeta.nnzjucon)
    ucon_jac_param_structure!(nlp, rows, cols)
    return rows, cols
end

function ucon_jac_param_structure! end

"""
    Jv = lcon_jpprod(nlp, v)

Evaluate `∇ₚlcon`, the gradient of the constraint lower bound wrt parameters.
This function is only available if `nlp.pmeta.lcon_jpprod_available` is set to `true`.
"""
function lcon_jpprod(nlp::AbstractNLPModel{T, S}, v::AbstractVector) where {T, S}
    @lencheck nlp.pmeta.nparam v
    Jv = S(undef, nlp.meta.ncon)
    return lcon_jpprod!(nlp, v, Jv)
end

function lcon_jpprod! end

"""
    Jv = ucon_jpprod(nlp, v)

Evaluate `∇ₚucon`, the gradient of the constraint upper bound wrt parameters.
This function is only available if `nlp.pmeta.ucon_jpprod_available` is set to `true`.
"""
function ucon_jpprod(nlp::AbstractNLPModel{T, S}, v::AbstractVector) where {T, S}
    @lencheck nlp.pmeta.nparam v
    Jv = S(undef, nlp.meta.ncon)
    return ucon_jpprod!(nlp, v, Jv)
end

function ucon_jpprod! end

"""
    Jtv = lcon_jptprod(nlp, v)

Evaluate `∇ₚlcon`, the gradient of the constraint lower bound wrt parameters.
This function is only available if `nlp.pmeta.lcon_jptprod_available` is set to `true`.
"""
function lcon_jptprod(nlp::AbstractNLPModel{T, S}, v::AbstractVector) where {T, S}
    @lencheck nlp.meta.ncon v
    Jtv = S(undef, nlp.pmeta.nparam)
    return lcon_jptprod!(nlp, v, Jtv)
end

function lcon_jptprod! end

"""
    Jtv = ucon_jptprod(nlp, v)

Evaluate `∇ₚucon`, the gradient of the constraint upper bound wrt parameters.
This function is only available if `nlp.pmeta.ucon_jptprod_available` is set to `true`.
"""
function ucon_jptprod(nlp::AbstractNLPModel{T, S}, v::AbstractVector) where {T, S}
    @lencheck nlp.meta.ncon v
    Jtv = S(undef, nlp.pmeta.nparam)
    return ucon_jptprod!(nlp, v, Jtv)
end

function ucon_jptprod! end

"""
    J = lcon_jac_param(nlp)

Evaluate the Jacobian of lower constraint bounds wrt parameters.
This function is only available if `nlp.pmeta.lcon_jac_available` is set to `true`.
"""
function lcon_jac_param(nlp::AbstractNLPModel)
    J = similar(NLPModels.get_x0(nlp), nlp.meta.ncon, nlp.pmeta.nparam)
    return lcon_jac_param!(nlp, J)
end

function lcon_jac_param! end

"""
    J = ucon_jac_param(nlp)

Evaluate the Jacobian of upper constraint bounds wrt parameters.
This function is only available if `nlp.pmeta.ucon_jac_available` is set to `true`.
"""
function ucon_jac_param(nlp::AbstractNLPModel)
    J = similar(NLPModels.get_x0(nlp), nlp.meta.ncon, nlp.pmeta.nparam)
    return ucon_jac_param!(nlp, J)
end

function ucon_jac_param! end

"""
    (rows,cols) = lvar_jac_param_structure(nlp)

Return the structure of the variable lower bound Jacobian wrt parameters in sparse coordinate format.
This function is only available if `nlp.pmeta.lvar_jac_available` is set to `true`.
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
This function is only available if `nlp.pmeta.uvar_jac_available` is set to `true`.
"""
function uvar_jac_param_structure(nlp::AbstractNLPModel)
    rows = Vector{Int}(undef, nlp.pmeta.nnzjuvar)
    cols = Vector{Int}(undef, nlp.pmeta.nnzjuvar)
    uvar_jac_param_structure!(nlp, rows, cols)
    return rows, cols
end

function uvar_jac_param_structure! end

"""
    Jv = lvar_jpprod(nlp, v)

Evaluate `∇ₚlvar`, the gradient of the variable lower bound wrt parameters.
This function is only available if `nlp.pmeta.lvar_jpprod_available` is set to `true`.
"""
function lvar_jpprod(nlp::AbstractNLPModel{T, S}, v::AbstractVector) where {T, S}
    @lencheck nlp.pmeta.nparam v
    Jv = S(undef, nlp.meta.nvar)
    return lvar_jpprod!(nlp, v, Jv)
end

function lvar_jpprod! end

"""
    Jv = uvar_jpprod(nlp, v)

Evaluate `∇ₚuvar`, the gradient of the variable upper bound wrt parameters.
This function is only available if `nlp.pmeta.uvar_jpprod_available` is set to `true`.
"""
function uvar_jpprod(nlp::AbstractNLPModel{T, S}, v::AbstractVector) where {T, S}
    Jv = S(undef, nlp.meta.nvar)
    return uvar_jpprod!(nlp, v, Jv)
end

function uvar_jpprod! end

"""
    Jtv = lvar_jptprod(nlp, v)

Evaluate `∇ₚlvar`, the gradient of the variable lower bound wrt parameters.
This function is only available if `nlp.pmeta.lvar_jptprod_available` is set to `true`.
"""
function lvar_jptprod(nlp::AbstractNLPModel{T, S}, v::AbstractVector) where {T, S}
    @lencheck nlp.meta.nvar v
    Jtv = S(undef, nlp.pmeta.nparam)
    return lvar_jptprod!(nlp, v, Jtv)
end

function lvar_jptprod! end

"""
    Jtv = uvar_jptprod(nlp, v)

Evaluate `∇ₚuvar`, the gradient of the variable upper bound wrt parameters.
This function is only available if `nlp.pmeta.uvar_jptprod_available` is set to `true`.
"""
function uvar_jptprod(nlp::AbstractNLPModel{T, S}, v::AbstractVector) where {T, S}
    @lencheck nlp.meta.nvar v
    Jtv = S(undef, nlp.pmeta.nparam)
    return uvar_jptprod!(nlp, v, Jtv)
end

function uvar_jptprod! end

"""
    J = lvar_jac_param(nlp)

Evaluate the Jacobian of lower variable bounds wrt parameters.
This function is only available if `nlp.pmeta.lvar_jac_available` is set to `true`.
"""
function lvar_jac_param(nlp::AbstractNLPModel)
    J = similar(NLPModels.get_x0(nlp), nlp.meta.nvar, nlp.pmeta.nparam)
    return lvar_jac_param!(nlp, J)
end

function lvar_jac_param! end

"""
    J = uvar_jac_param(nlp)

Evaluate the Jacobian of upper variable bounds wrt parameters.
This function is only available if `nlp.pmeta.uvar_jac_available` is set to `true`.
"""
function uvar_jac_param(nlp::AbstractNLPModel)
    J = similar(NLPModels.get_x0(nlp), nlp.meta.nvar, nlp.pmeta.nparam)
    return uvar_jac_param!(nlp, J)
end

function uvar_jac_param! end
