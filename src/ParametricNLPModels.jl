module ParametricNLPModels

import NLPModels: AbstractNLPModel, @lencheck, coo_prod!, @closure, LinearOperator

struct ParametricNLPModelMeta
    nparam::Int
    nnzj::Int
    lin_nnzj::Int
    nln_nnzj::Int
    nnzh::Int
end

export grad_param, grad_param!,  # df/dp
    jac_param_structure, jac_param_structure!,  # dg/dp
    jac_param_lin_structure, jac_param_lin_structure!,
    jac_param_nln_structure, jac_param_nln_structure!,
    jac_param_coord, jac_param_coord!,
    jac_param_lin_coord, jac_param_lin_coord!,
    jac_param_nln_coord, jac_param_nln_coord!,
    jac_param, jac_param_lin, jac_param_nln,
    jpprod, jpprod!,
    jpprod_lin, jpprod_lin!,
    jpprod_nln, jpprod_nln!,
    jptprod, jptprod!,
    jptprod_lin, jptprod_lin!,
    jptprod_nln, jptprod_nln!,
    jac_param_op, jac_param_op!,
    jac_param_lin_op, jac_param_lin_op!,
    jac_param_nln_op, jac_param_nln_op!,
    hess_param_structure, hess_param_structure!,  # d²L/dxdp
    hess_param_coord, hess_param_coord!,
    hess_param,
    hpprod, hpprod!,
    hptprod, hptprod!,
    hess_param_op, hess_param_op!,
    lcon_grad_param, lcon_grad_param!,
    ucon_grad_param, ucon_grad_param!,
    lvar_grad_param, lvar_grad_param!,
    uvar_grad_param, uvar_grad_param!


"""
    g = grad_param(nlp, x)

Evaluate `∇ₚf(x)`, the gradient of the objective function at `x` wrt parameters.
"""
function grad_param end

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
function jac_param_structure!(
  nlp::AbstractNLPModel,
  rows::AbstractVector{T},
  cols::AbstractVector{T},
) where {T}
  @lencheck nlp.pmeta.nnzj rows cols
  lin_ind = 1:(nlp.pmeta.lin_nnzj)
  if nlp.meta.nlin > 0
    if nlp.meta.nnln == 0
      jac_param_lin_structure!(nlp, rows, cols)
    else
      jac_param_lin_structure!(nlp, view(rows, lin_ind), view(cols, lin_ind))
      for i in lin_ind
        rows[i] += count(x < nlp.meta.lin[rows[i]] for x in nlp.meta.nln)
      end
    end
  end
  if nlp.meta.nnln > 0
    if nlp.meta.nlin == 0
      jac_param_nln_structure!(nlp, rows, cols)
    else
      nln_ind = (nlp.pmeta.lin_nnzj + 1):(nlp.pmeta.lin_nnzj + nlp.pmeta.nln_nnzj)
      jac_param_nln_structure!(nlp, view(rows, nln_ind), view(cols, nln_ind))
      for i in nln_ind
        rows[i] += count(x < nlp.meta.nln[rows[i]] for x in nlp.meta.lin)
      end
    end
  end
  return rows, cols
end

"""
    (rows,cols) = jac_param_lin_structure(nlp)

Return the structure of the linear constraints Jacobian wrt parameters in sparse coordinate format.
"""
function jac_param_lin_structure(nlp::AbstractNLPModel)
  rows = Vector{Int}(undef, nlp.pmeta.lin_nnzj)
  cols = Vector{Int}(undef, nlp.pmeta.lin_nnzj)
  jac_param_lin_structure!(nlp, rows, cols)
end

"""
    jac_param_lin_structure!(nlp, rows, cols)

Return the structure of the linear constraints Jacobian wrt parameters in sparse coordinate format in place.
"""
function jac_param_lin_structure! end

"""
    (rows,cols) = jac_param_nln_structure(nlp)

Return the structure of the nonlinear constraints Jacobian wrt parameters in sparse coordinate format.
"""
function jac_param_nln_structure(nlp::AbstractNLPModel)
  rows = Vector{Int}(undef, nlp.pmeta.nln_nnzj)
  cols = Vector{Int}(undef, nlp.pmeta.nln_nnzj)
  jac_param_nln_structure!(nlp, rows, cols)
end

"""
    jac_param_nln_structure!(nlp, rows, cols)

Return the structure of the nonlinear constraints Jacobian wrt parameters in sparse coordinate format in place.
"""
function jac_param_nln_structure! end

"""
    vals = jac_param_coord!(nlp, x, vals)

Evaluate ``Jₚ(x)``, the constraints Jacobian wrt parameters at `x` in sparse coordinate format,
rewriting `vals`.
"""
function jac_param_coord!(nlp::AbstractNLPModel, x::AbstractVector, vals::AbstractVector)
  @lencheck nlp.meta.nvar x
  @lencheck nlp.pmeta.nnzj vals
#   increment!(nlp, :neval_jac_param)
  if nlp.meta.nlin > 0
    if nlp.meta.nnln == 0
      jac_param_lin_coord!(nlp, x, vals)
    else
      lin_ind = 1:(nlp.pmeta.lin_nnzj)
      jac_param_lin_coord!(nlp, x, view(vals, lin_ind))
    end
  end
  if nlp.meta.nnln > 0
    if nlp.meta.nlin == 0
      jac_param_nln_coord!(nlp, x, vals)
    else
      nln_ind = (nlp.pmeta.lin_nnzj + 1):(nlp.pmeta.lin_nnzj + nlp.pmeta.nln_nnzj)
      jac_param_nln_coord!(nlp, x, view(vals, nln_ind))
    end
  end
  return vals
end

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
    Jx = jac_param(nlp, x)

Evaluate ``Jₚ(x)``, the constraints Jacobian wrt parameters at `x` as a sparse matrix.
"""
function jac_param(nlp::AbstractNLPModel, x::AbstractVector)
  @lencheck nlp.meta.nvar x
  rows, cols = jac_param_structure(nlp)
  vals = jac_param_coord(nlp, x)
  sparse(rows, cols, vals, nlp.meta.ncon, nlp.pmeta.nparam)
end

"""
    vals = jac_param_lin_coord!(nlp, x, vals)

Evaluate ``J(x)``, the linear constraints Jacobian wrt parameters at `x` in sparse coordinate format,
overwriting `vals`.
"""
function jac_param_lin_coord! end

"""
    vals = jac_param_lin_coord(nlp, x)

Evaluate ``J(x)``, the linear constraints Jacobian wrt parameters at `x` in sparse coordinate format.
"""
function jac_param_lin_coord(nlp::AbstractNLPModel{T, S}, x::AbstractVector) where {T, S}
  @lencheck nlp.meta.nvar x
  vals = S(undef, nlp.pmeta.lin_nnzj)
  return jac_param_lin_coord!(nlp, x, vals)
end

"""
    Jx = jac_param_lin(nlp, x)

Evaluate ``Jₚ(x)``, the linear constraints Jacobian wrt parameters at `x` as a sparse matrix.
"""
function jac_param_lin(nlp::AbstractNLPModel, x::AbstractVector)
  @lencheck nlp.meta.nvar x
  rows, cols = jac_param_lin_structure(nlp)
  vals = jac_param_lin_coord(nlp, x)
  sparse(rows, cols, vals, nlp.meta.nlin, nlp.pmeta.nparam)
end

"""
    vals = jac_param_nln_coord!(nlp, x, vals)

Evaluate ``Jₚ(x)``, the nonlinear constraints Jacobian wrt parameters at `x` in sparse coordinate format,
overwriting `vals`.
"""
function jac_param_nln_coord! end

"""
    vals = jac_param_nln_coord(nlp, x)

Evaluate ``Jₚ(x)``, the nonlinear constraints Jacobian wrt parameters at `x` in sparse coordinate format.
"""
function jac_param_nln_coord(nlp::AbstractNLPModel{T, S}, x::AbstractVector) where {T, S}
  @lencheck nlp.meta.nvar x
  vals = S(undef, nlp.pmeta.nln_nnzj)
  return jac_param_nln_coord!(nlp, x, vals)
end

"""
    Jx = jac_param_nln(nlp, x)

Evaluate ``Jₚ(x)``, the nonlinear constraints Jacobian wrt parameters at `x` as a sparse matrix.
"""
function jac_param_nln(nlp::AbstractNLPModel, x::AbstractVector)
  @lencheck nlp.meta.nvar x
  rows, cols = jac_param_nln_structure(nlp)
  vals = jac_param_nln_coord(nlp, x)
  sparse(rows, cols, vals, nlp.meta.nnln, nlp.pmeta.nparam)
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
function jpprod!(nlp::AbstractNLPModel, x::AbstractVector, v::AbstractVector, Jv::AbstractVector)
  @lencheck nlp.meta.nvar x
  @lencheck nlp.pmeta.nparam v
  @lencheck nlp.meta.ncon Jv
#   increment!(nlp, :neval_jpprod)
  if nlp.meta.nlin > 0
    if nlp.meta.nnln == 0
      jpprod_lin!(nlp, x, v, Jv)
    else
      jpprod_lin!(nlp, x, v, view(Jv, nlp.meta.lin))
    end
  end
  if nlp.meta.nnln > 0
    if nlp.meta.nlin == 0
      jpprod_nln!(nlp, x, v, Jv)
    else
      jpprod_nln!(nlp, x, v, view(Jv, nlp.meta.nln))
    end
  end
  return Jv
end

"""
    Jv = jpprod!(nlp, rows, cols, vals, v, Jv)

Evaluate ``Jₚ(x)v``, the parametric Jacobian-vector product, where the Jacobian wrt parameters is given by
`(rows, cols, vals)` in triplet format.
"""
function jpprod!(
  nlp::AbstractNLPModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
  vals::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  @lencheck nlp.meta.nnzj rows cols vals
  @lencheck nlp.pmeta.nparam v
  @lencheck nlp.meta.ncon Jv
#   increment!(nlp, :neval_jpprod)
  coo_prod!(rows, cols, vals, v, Jv)
end

"""
    Jv = jpprod_lin(nlp, x, v)

Evaluate ``Jₚ(x)v``, the linear parametric Jacobian-vector product at `x`.
"""
function jpprod_lin(nlp::AbstractNLPModel{T, S}, x::AbstractVector, v::AbstractVector) where {T, S}
  @lencheck nlp.meta.nvar x
  @lencheck nlp.pmeta.nparam v
  Jv = S(undef, nlp.meta.nlin)
  return jpprod_lin!(nlp, x, v, Jv)
end

"""
    Jv = jpprod_lin!(nlp, x, v, Jv)

Evaluate ``Jₚ(x)v``, the linear parametric Jacobian-vector product at `x` in place.
"""
function jpprod_lin! end

"""
    Jv = jpprod_lin!(nlp, rows, cols, vals, v, Jv)

Evaluate ``Jₚ(x)v``, the linear parametric Jacobian-vector product, where the parametric Jacobian is given by
`(rows, cols, vals)` in triplet format.
"""
function jpprod_lin!(
  nlp::AbstractNLPModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
  vals::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  @lencheck nlp.meta.lin_nnzj rows cols vals
  @lencheck nlp.pmeta.nparam v
  @lencheck nlp.meta.nlin Jv
#   increment!(nlp, :neval_jpprod_lin)
  coo_prod!(rows, cols, vals, v, Jv)
end

"""
    Jv = jpprod_nln(nlp, x, v)

Evaluate ``Jₚ(x)v``, the nonlinear parametric Jacobian-vector product at `x`.
"""
function jpprod_nln(nlp::AbstractNLPModel{T, S}, x::AbstractVector, v::AbstractVector) where {T, S}
  @lencheck nlp.meta.nvar x
  @lencheck nlp.pmeta.nparam v
  Jv = S(undef, nlp.meta.nnln)
  return jpprod_nln!(nlp, x, v, Jv)
end

"""
    Jv = jpprod_nln!(nlp, x, v, Jv)

Evaluate ``Jₚ(x)v``, the nonlinear parametric Jacobian-vector product at `x` in place.
"""
function jpprod_nln! end

"""
    Jv = jpprod_nln!(nlp, rows, cols, vals, v, Jv)

Evaluate ``Jₚ(x)v``, the nonlinear parametric Jacobian-vector product, where the Jacobian wrt parameters is given by
`(rows, cols, vals)` in triplet format.
"""
function jpprod_nln!(
  nlp::AbstractNLPModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
  vals::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  @lencheck nlp.meta.nln_nnzj rows cols vals
  @lencheck nlp.pmeta.nparam v
  @lencheck nlp.meta.nnln Jv
#   increment!(nlp, :neval_jpprod_nln)
  coo_prod!(rows, cols, vals, v, Jv)
end

"""
    Jtv = jptprod(nlp, x, v)

Evaluate ``Jₚ(x)^Tv``, the parametric transposed-Jacobian-vector product at `x`.
"""
function jptprod(nlp::AbstractNLPModel{T, S}, x::AbstractVector, v::AbstractVector) where {T, S}
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.ncon v
  Jtv = S(undef, nlp.pmeta.nparam)
  return jptprod!(nlp, x, v, Jtv)
end

"""
    Jtv = jptprod!(nlp, x, v, Jtv)

Evaluate ``Jₚ(x)^Tv``, the parametric transposed-Jacobian-vector product at `x` in place.
If the problem has linear and nonlinear constraints, this function allocates.
"""
function jptprod!(nlp::AbstractNLPModel, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector)
  @lencheck nlp.meta.nvar x
  @lencheck nlp.pmeta.nparam Jtv
  @lencheck nlp.meta.ncon v
#   increment!(nlp, :neval_jptprod)
  if nlp.meta.nnln == 0
    (nlp.meta.nlin > 0) && jptprod_lin!(nlp, x, v, Jtv)
  elseif nlp.meta.nlin == 0
    (nlp.meta.nnln > 0) && jptprod_nln!(nlp, x, v, Jtv)
  elseif nlp.meta.nlin >= nlp.meta.nnln
    jptprod_lin!(nlp, x, view(v, nlp.meta.lin), Jtv)
    if nlp.meta.nnln > 0
      Jtv .+= jptprod_nln(nlp, x, view(v, nlp.meta.nln))
    end
  else
    jptprod_nln!(nlp, x, view(v, nlp.meta.nln), Jtv)
    if nlp.meta.nlin > 0
      Jtv .+= jptprod_lin(nlp, x, view(v, nlp.meta.lin))
    end
  end
  return Jtv
end

"""
    Jtv = jptprod!(nlp, rows, cols, vals, v, Jtv)

Evaluate ``J(x)^Tv``, the parametric transposed-Jacobian-vector product, where the
Jacobian is given by `(rows, cols, vals)` in triplet format.
"""
function jptprod!(
  nlp::AbstractNLPModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
  vals::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  @lencheck nlp.meta.nnzj rows cols vals
  @lencheck nlp.meta.ncon v
  @lencheck nlp.pmeta.nparam Jtv
#   increment!(nlp, :neval_jtprod)
  coo_prod!(cols, rows, vals, v, Jtv)
end

"""
    Jtv = jptprod_lin(nlp, x, v)

Evaluate ``J(x)^Tv``, the linear parametric transposed-Jacobian-vector product at `x`.
"""
function jptprod_lin(nlp::AbstractNLPModel{T, S}, x::AbstractVector, v::AbstractVector) where {T, S}
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.nlin v
  Jtv = S(undef, nlp.pmeta.nparam)
  return jptprod_lin!(nlp, x, v, Jtv)
end


"""
    Jtv = jptprod_lin!(nlp, x, v, Jtv)

Evaluate ``J(x)^Tv``, the linear parametric transposed-Jacobian-vector product at `x` in place.
"""
function jptprod_lin! end

"""
    Jtv = jptprod_lin!(nlp, rows, cols, vals, v, Jtv)

Evaluate ``J(x)^Tv``, the linear parametric transposed-Jacobian-vector product, where the
parametric Jacobian is given by `(rows, cols, vals)` in triplet format.
"""
function jptprod_lin!(
  nlp::AbstractNLPModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
  vals::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  @lencheck nlp.meta.lin_nnzj rows cols vals
  @lencheck nlp.meta.nlin v
  @lencheck nlp.pmeta.nparam Jtv
#   increment!(nlp, :neval_jtprod_lin)
  coo_prod!(cols, rows, vals, v, Jtv)
end


"""
    Jtv = jptprod_nln(nlp, x, v)

Evaluate ``J(x)^Tv``, the nonlinear parametric transposed-Jacobian-vector product at `x`.
"""
function jptprod_nln(nlp::AbstractNLPModel{T, S}, x::AbstractVector, v::AbstractVector) where {T, S}
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.nnln v
  Jtv = S(undef, nlp.pmeta.nparam)
  return jptprod_nln!(nlp, x, v, Jtv)
end

"""
    Jtv = jptprod_nln!(nlp, x, v, Jtv)

Evaluate ``J(x)^Tv``, the nonlinear parametric transposed-Jacobian-vector product at `x` in place.
"""
function jptprod_nln! end

"""
    Jtv = jptprod_nln!(nlp, rows, cols, vals, v, Jtv)

Evaluate ``J(x)^Tv``, the nonlinear parametric transposed-Jacobian-vector product, where the
parametric Jacobian is given by `(rows, cols, vals)` in triplet format.
"""
function jptprod_nln!(
  nlp::AbstractNLPModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
  vals::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  @lencheck nlp.meta.nln_nnzj rows cols vals
  @lencheck nlp.meta.nnln v
  @lencheck nlp.pmeta.nparam Jtv
#   increment!(nlp, :neval_jptprod_nln)
  coo_prod!(cols, rows, vals, v, Jtv)
end

"""
    J = jac_param_op(nlp, x)

Return the parametric Jacobian at `x` as a linear operator.
The resulting object may be used as if it were a matrix, e.g., `J * v` or
`J' * v`.
"""
function jac_param_op(nlp::AbstractNLPModel{T, S}, x::AbstractVector) where {T, S}
  @lencheck nlp.meta.nvar x
  Jv = S(undef, nlp.meta.ncon)
  Jtv = S(undef, nlp.pmeta.nparam)
  return jac_param_op!(nlp, x, Jv, Jtv)
end

"""
    J = jac_param_op!(nlp, x, Jv, Jtv)

Return the parametric Jacobian at `x` as a linear operator.
The resulting object may be used as if it were a matrix, e.g., `J * v` or
`J' * v`. The values `Jv` and `Jtv` are used as preallocated storage for the
operations.
"""
function jac_param_op!(
  nlp::AbstractNLPModel{T, S},
  x::AbstractVector{T},
  Jv::AbstractVector,
  Jtv::AbstractVector,
) where {T, S}
  @lencheck nlp.meta.nvar x
  @lencheck nlp.pmeta.nparam Jtv
  @lencheck nlp.meta.ncon Jv
  prod! = @closure (res, v, α, β) -> begin # res = α * J * v + β * res
    jpprod!(nlp, x, v, Jv)
    if β == 0
      res .= α .* Jv
    else
      res .= α .* Jv .+ β .* res
    end
    return res
  end
  ctprod! = @closure (res, v, α, β) -> begin
    jptprod!(nlp, x, v, Jtv)
    if β == 0
      res .= α .* Jtv
    else
      res .= α .* Jtv .+ β .* res
    end
    return res
  end
  return LinearOperator{T}(nlp.meta.ncon, nlp.pmeta.nparam, false, false, prod!, ctprod!, ctprod!)
end

"""
    J = jac_param_op!(nlp, rows, cols, vals, Jv, Jtv)

Return the parametric Jacobian given by `(rows, cols, vals)` as a linear operator.
The resulting object may be used as if it were a matrix, e.g., `J * v` or `J' * v`.
The values `Jv` and `Jtv` are used as preallocated storage for the operations.
"""
function jac_param_op!(
  nlp::AbstractNLPModel{T, S},
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
  vals::AbstractVector{T},
  Jv::AbstractVector,
  Jtv::AbstractVector,
) where {T, S}
  @lencheck nlp.meta.nnzj rows cols vals
  @lencheck nlp.meta.ncon Jv
  @lencheck nlp.pmeta.nparam Jtv
  prod! = @closure (res, v, α, β) -> begin # res = α * J * v + β * res
    jpprod!(nlp, rows, cols, vals, v, Jv)
    if β == 0
      res .= α .* Jv
    else
      res .= α .* Jv .+ β .* res
    end
    return res
  end
  ctprod! = @closure (res, v, α, β) -> begin
    jptprod!(nlp, rows, cols, vals, v, Jtv)
    if β == 0
      res .= α .* Jtv
    else
      res .= α .* Jtv .+ β .* res
    end
    return res
  end
  return LinearOperator{T}(nlp.meta.ncon, nlp.pmeta.nparam, false, false, prod!, ctprod!, ctprod!)
end

"""
    J = jac_param_lin_op(nlp, x)

Return the linear parametric Jacobian at `x` as a linear operator.
The resulting object may be used as if it were a matrix, e.g., `J * v` or
`J' * v`.
"""
function jac_param_lin_op(nlp::AbstractNLPModel{T, S}, x::AbstractVector) where {T, S}
  @lencheck nlp.meta.nvar x
  Jv = S(undef, nlp.meta.nlin)
  Jtv = S(undef, nlp.pmeta.nparam)
  return jac_param_lin_op!(nlp, x, Jv, Jtv)
end

"""
    J = jac_param_lin_op!(nlp, x, Jv, Jtv)

Return the linear parametric Jacobian at `x` as a linear operator.
The resulting object may be used as if it were a matrix, e.g., `J * v` or
`J' * v`. The values `Jv` and `Jtv` are used as preallocated storage for the
operations.
"""
function jac_param_lin_op!(
  nlp::AbstractNLPModel{T, S},
  x::AbstractVector{T},
  Jv::AbstractVector,
  Jtv::AbstractVector,
) where {T, S}
  @lencheck nlp.meta.nvar x
  @lencheck nlp.pmeta.nparam Jtv
  @lencheck nlp.meta.nlin Jv
  prod! = @closure (res, v, α, β) -> begin # res = α * J * v + β * res
    jpprod_lin!(nlp, x, v, Jv)
    if β == 0
      res .= α .* Jv
    else
      res .= α .* Jv .+ β .* res
    end
    return res
  end
  ctprod! = @closure (res, v, α, β) -> begin
    jptprod_lin!(nlp, x, v, Jtv)
    if β == 0
      res .= α .* Jtv
    else
      res .= α .* Jtv .+ β .* res
    end
    return res
  end
  return LinearOperator{T}(nlp.meta.nlin, nlp.pmeta.nparam, false, false, prod!, ctprod!, ctprod!)
end


"""
    J = jac_param_lin_op!(nlp, rows, cols, vals, Jv, Jtv)

Return the linear parametric Jacobian given by `(rows, cols, vals)` as a linear operator.
The resulting object may be used as if it were a matrix, e.g., `J * v` or `J' * v`.
The values `Jv` and `Jtv` are used as preallocated storage for the operations.
"""
function jac_param_lin_op!(
  nlp::AbstractNLPModel{T, S},
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
  vals::AbstractVector{T},
  Jv::AbstractVector,
  Jtv::AbstractVector,
) where {T, S}
  @lencheck nlp.meta.lin_nnzj rows cols vals
  @lencheck nlp.meta.nlin Jv
  @lencheck nlp.pmeta.nparam Jtv
  prod! = @closure (res, v, α, β) -> begin # res = α * J * v + β * res
    jpprod_lin!(nlp, rows, cols, vals, v, Jv)
    if β == 0
      res .= α .* Jv
    else
      res .= α .* Jv .+ β .* res
    end
    return res
  end
  ctprod! = @closure (res, v, α, β) -> begin
    jptprod_lin!(nlp, rows, cols, vals, v, Jtv)
    if β == 0
      res .= α .* Jtv
    else
      res .= α .* Jtv .+ β .* res
    end
    return res
  end
  return LinearOperator{T}(nlp.meta.nlin, nlp.pmeta.nparam, false, false, prod!, ctprod!, ctprod!)
end

"""
    J = jac_param_nln_op(nlp, x)

Return the nonlinear parametric Jacobian at `x` as a linear operator.
The resulting object may be used as if it were a matrix, e.g., `J * v` or
`J' * v`.
"""
function jac_param_nln_op(nlp::AbstractNLPModel{T, S}, x::AbstractVector) where {T, S}
  @lencheck nlp.meta.nvar x
  Jv = S(undef, nlp.meta.nnln)
  Jtv = S(undef, nlp.pmeta.nparam)
  return jac_param_nln_op!(nlp, x, Jv, Jtv)
end

"""
    J = jac_param_nln_op!(nlp, x, Jv, Jtv)

Return the nonlinear parametric Jacobian at `x` as a linear operator.
The resulting object may be used as if it were a matrix, e.g., `J * v` or
`J' * v`. The values `Jv` and `Jtv` are used as preallocated storage for the
operations.
"""
function jac_param_nln_op!(
  nlp::AbstractNLPModel{T, S},
  x::AbstractVector{T},
  Jv::AbstractVector,
  Jtv::AbstractVector,
) where {T, S}
  @lencheck nlp.meta.nvar x
  @lencheck nlp.pmeta.nparam Jtv
  @lencheck nlp.meta.nnln Jv
  prod! = @closure (res, v, α, β) -> begin # res = α * J * v + β * res
    jpprod_nln!(nlp, x, v, Jv)
    if β == 0
      res .= α .* Jv
    else
      res .= α .* Jv .+ β .* res
    end
    return res
  end
  ctprod! = @closure (res, v, α, β) -> begin
    jptprod_nln!(nlp, x, v, Jtv)
    if β == 0
      res .= α .* Jtv
    else
      res .= α .* Jtv .+ β .* res
    end
    return res
  end
  return LinearOperator{T}(nlp.meta.nnln, nlp.pmeta.nparam, false, false, prod!, ctprod!, ctprod!)
end

"""
    J = jac_param_nln_op!(nlp, rows, cols, vals, Jv, Jtv)

Return the nonlinear parametric Jacobian given by `(rows, cols, vals)` as a linear operator.
The resulting object may be used as if it were a matrix, e.g., `J * v` or `J' * v`.
The values `Jv` and `Jtv` are used as preallocated storage for the operations.
"""
function jac_param_nln_op!(
  nlp::AbstractNLPModel{T, S},
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
  vals::AbstractVector{T},
  Jv::AbstractVector,
  Jtv::AbstractVector,
) where {T, S}
  @lencheck nlp.meta.nln_nnzj rows cols vals
  @lencheck nlp.meta.nnln Jv
  @lencheck nlp.pmeta.nparam Jtv
  prod! = @closure (res, v, α, β) -> begin # res = α * J * v + β * res
    jpprod_nln!(nlp, rows, cols, vals, v, Jv)
    if β == 0
      res .= α .* Jv
    else
      res .= α .* Jv .+ β .* res
    end
    return res
  end
  ctprod! = @closure (res, v, α, β) -> begin
    jptprod_nln!(nlp, rows, cols, vals, v, Jtv)
    if β == 0
      res .= α .* Jtv
    else
      res .= α .* Jtv .+ β .* res
    end
    return res
  end
  return LinearOperator{T}(nlp.meta.nnln, nlp.pmeta.nparam, false, false, prod!, ctprod!, ctprod!)
end

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
    Hx = hess_param(nlp, x; obj_weight=1.0)

Evaluate the variable-parameter objective Hessian at `x` as a sparse matrix,
with objective function scaled by `obj_weight`.
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
    Hx = hess_param(nlp, x, y; obj_weight=1.0)

Evaluate the Lagrangian variable-parameter Hessian at `(x,y)` as a sparse matrix,
with objective function scaled by `obj_weight`.
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

"""
    Htv = hptprod(nlp, x, v; obj_weight=1.0)

Evaluate the product of the objective variable-parameter Hessian-transpose at `x` with the vector `v`,
with objective function scaled by `obj_weight`.
"""
function hptprod(
  nlp::AbstractNLPModel{T, S},
  x::AbstractVector,
  v::AbstractVector;
  obj_weight::Real = one(T),
) where {T, S}
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.nvar v
  Htv = S(undef, nlp.pmeta.nparam)
  return hptprod!(nlp, x, v, Htv; obj_weight = obj_weight)
end

"""
    Htv = hptprod(nlp, x, y, v; obj_weight=1.0)

Evaluate the product of the Lagrangian variable-parameter Hessian-transpose at `(x,y)` with the vector `v`,
with objective function scaled by `obj_weight`.
"""
function hptprod(
  nlp::AbstractNLPModel{T, S},
  x::AbstractVector,
  y::AbstractVector,
  v::AbstractVector;
  obj_weight::Real = one(T),
) where {T, S}
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.nvar v
  @lencheck nlp.meta.ncon y
  Htv = S(undef, nlp.pmeta.nparam)
  return hptprod!(nlp, x, y, v, Htv; obj_weight = obj_weight)
end

"""
    Htv = hptprod!(nlp, x, v, Htv; obj_weight=1.0)

Evaluate the product of the objective variable-parameter Hessian-transpose at `x` with the vector `v` in
place, with objective function scaled by `obj_weight`.
"""
function hptprod!(
  nlp::AbstractNLPModel{T, S},
  x::AbstractVector,
  v::AbstractVector,
  Htv::AbstractVector;
  obj_weight::Real = one(T),
) where {T, S}
  @lencheck nlp.meta.nvar x
  @lencheck nlp.pmeta.nparam Htv
  @lencheck nlp.meta.nvar v
  y = fill!(S(undef, nlp.meta.ncon), 0)
  hptprod!(nlp, x, y, v, Htv, obj_weight = obj_weight)
end

"""
    Htv = hptprod!(nlp, rows, cols, vals, v, Htv)

Evaluate the product of the objective or Lagrangian variable-parameter Hessian-transpose given by `(rows, cols, vals)` in
triplet format with the vector `v` in place.
"""
function hptprod!(
  nlp::AbstractNLPModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
  vals::AbstractVector,
  v::AbstractVector,
  Htv::AbstractVector,
)
  @lencheck nlp.meta.nnzh rows cols vals
  @lencheck nlp.pmeta.nparam Htv
  @lencheck nlp.meta.nvar v
#   increment!(nlp, :neval_hptprod)
  coo_prod!(cols, rows, vals, v, Htv)
end

"""
    Htv = hptprod!(nlp, x, y, v, Htv; obj_weight=1.0)

Evaluate the product of the Lagrangian variable-parameter Hessian-transpose at `(x,y)` with the vector `v` in
place, with objective function scaled by `obj_weight`.
"""
function hptprod! end

"""
    H = hess_param_op(nlp, x; obj_weight=1.0)

Return the objective variable-parameter Hessian at `x` with objective function scaled by
`obj_weight` as a linear operator. The resulting object may be used as if it were a
matrix, e.g., `H * v`.
"""
function hess_param_op(
  nlp::AbstractNLPModel{T, S},
  x::AbstractVector;
  obj_weight::Real = one(T),
) where {T, S}
  @lencheck nlp.meta.nvar x
  Hv = S(undef, nlp.meta.nvar)
  return hess_param_op!(nlp, x, Hv, obj_weight = obj_weight)
end

"""
    H = hess_param_op(nlp, x, y; obj_weight=1.0)

Return the Lagrangian variable-parameter Hessian at `(x,y)` with objective function scaled by
`obj_weight` as a linear operator. The resulting object may be used as if it were a
matrix, e.g., `H * v`.
"""
function hess_param_op(
  nlp::AbstractNLPModel{T, S},
  x::AbstractVector,
  y::AbstractVector;
  obj_weight::Real = one(T),
) where {T, S}
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.ncon y
  Hv = S(undef, nlp.meta.nvar)
  return hess_param_op!(nlp, x, y, Hv, obj_weight = obj_weight)
end

"""
    H = hess_param_op!(nlp, x, Hv; obj_weight=1.0)

Return the objective variable-parameter Hessian at `x` with objective function scaled by
`obj_weight` as a linear operator, and storing the result on `Hv`. The resulting
object may be used as if it were a matrix, e.g., `w = H * v`. The vector `Hv` is
used as preallocated storage for the operation.
"""
function hess_param_op!(
  nlp::AbstractNLPModel{T, S},
  x::AbstractVector,
  Hv::AbstractVector,
  Htv::AbstractVector;
  obj_weight::Real = one(T),
) where {T, S}
  @lencheck nlp.meta.nvar x Hv
  prod! = @closure (res, v, α, β) -> begin
    hpprod!(nlp, x, v, Hv; obj_weight = obj_weight)
    if β == 0
      res .= α .* Hv
    else
      res .= α .* Hv .+ β .* res
    end
    return res
  end
  ctprod! = @closure (res, v, α, β) -> begin
    hptprod!(nlp, x, v, Htv; obj_weight = obj_weight)
    if β == 0
      res .= α .* Htv
    else
      res .= α .* Htv .+ β .* res
    end
    return res
  end
  return LinearOperator{T}(nlp.meta.nvar, nlp.pmeta.nparam, false, false, prod!, ctprod!, ctprod!)
end

"""
    H = hess_param_op!(nlp, x, y, Hv, Htv; obj_weight=1.0)

Return the Lagrangian variable-parameter Hessian at `(x,y)` with objective function scaled by
`obj_weight` as a linear operator, and storing the result on `Hv`. The resulting
object may be used as if it were a matrix, e.g., `w = H * v`. The vectors `Hv` and `Htv` are
used as preallocated storage for the operations.
"""
function hess_param_op!(
  nlp::AbstractNLPModel{T, S},
  x::AbstractVector,
  y::AbstractVector,
  Hv::AbstractVector,
  Htv::AbstractVector;
  obj_weight::Real = one(T),
) where {T, S}
  @lencheck nlp.meta.nvar x Hv
  @lencheck nlp.meta.ncon y
  prod! = @closure (res, v, α, β) -> begin
    hpprod!(nlp, x, y, v, Hv; obj_weight = obj_weight)
    if β == 0
      res .= α .* Hv
    else
      res .= α .* Hv .+ β .* res
    end
    return res
  end
  ctprod! = @closure (res, v, α, β) -> begin
    hptprod!(nlp, x, y, v, Htv; obj_weight = obj_weight)
    if β == 0
      res .= α .* Htv
    else
      res .= α .* Htv .+ β .* res
    end
    return res
  end
  return LinearOperator{T}(nlp.meta.nvar, nlp.pmeta.nparam, false, false, prod!, ctprod!, ctprod!)
end

"""
    g = lcon_grad_param(nlp)

Evaluate `∇ₚlcon`, the gradient of the constraint lower bound wrt parameters.
"""
function lcon_grad_param(nlp::AbstractNLPModel{T, S}) where {T, S}
    g = S(undef, nlp.meta.ncon)
    return lcon_grad_param!(nlp, g)
end

"""
    g = lcon_grad_param!(nlp, g)

Evaluate `∇ₚlcon`, the gradient of the constraint lower bound wrt parameters in place.
"""
function lcon_grad_param!(nlp, g)
    fill!(g, zero(eltype(g)))
    return g
end

"""
    g = ucon_grad_param(nlp)

Evaluate `∇ₚucon`, the gradient of the constraint upper bound wrt parameters.
"""
function ucon_grad_param(nlp::AbstractNLPModel{T, S}) where {T, S}
    g = S(undef, nlp.meta.ncon)
    return ucon_grad_param!(nlp, g)
end

"""
    g = ucon_grad_param!(nlp, g)

Evaluate `∇ₚucon`, the gradient of the constraint upper bound wrt parameters in place.
"""
function ucon_grad_param!(nlp, g)
    fill!(g, zero(eltype(g)))
    return g
end

"""
    g = lvar_grad_param(nlp)

Evaluate `∇ₚlvar`, the gradient of the variable lower bound wrt parameters.
"""
function lvar_grad_param(nlp::AbstractNLPModel{T, S}) where {T, S}
    g = S(undef, nlp.meta.nvar)
    return lvar_grad_param!(nlp, g)
end

"""
    g = lvar_grad_param!(nlp, g)

Evaluate `∇ₚlvar`, the gradient of the variable lower bound wrt parameters in place.
"""
function lvar_grad_param!(nlp, g)
    fill!(g, zero(eltype(g)))
    return g
end

"""
    g = uvar_grad_param(nlp)

Evaluate `∇ₚuvar`, the gradient of the variable upper bound wrt parameters.
"""
function uvar_grad_param(nlp::AbstractNLPModel{T, S}) where {T, S}
    g = S(undef, nlp.meta.nvar)
    return uvar_grad_param!(nlp, g)
end

"""
    g = uvar_grad_param!(nlp, g)

Evaluate `∇ₚuvar`, the gradient of the variable upper bound wrt parameters in place.
"""
function uvar_grad_param!(nlp, g)
    fill!(g, zero(eltype(g)))
    return g
end

end # module ParametricNLPModels
