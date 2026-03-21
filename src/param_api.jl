export set_param_values!, get_param_values,
  grad_param, grad_param!,
  jac_param_structure, jac_param_structure!,
  jac_param_coord, jac_param_coord!,
  jpprod, jpprod!,
  jptprod, jptprod!,
  hess_param_structure, hess_param_structure!,
  hess_param_coord, hess_param_coord!,
  hpprod, hpprod!,
  hptprod, hptprod!,
  lcon_jac_param_structure, lcon_jac_param_structure!,
  lcon_jac_param_coord, lcon_jac_param_coord!,
  lcon_jpprod, lcon_jpprod!,
  lcon_jptprod, lcon_jptprod!,
  ucon_jac_param_structure, ucon_jac_param_structure!,
  ucon_jac_param_coord, ucon_jac_param_coord!,
  ucon_jpprod, ucon_jpprod!,
  ucon_jptprod, ucon_jptprod!,
  lvar_jac_param_structure, lvar_jac_param_structure!,
  lvar_jac_param_coord, lvar_jac_param_coord!,
  lvar_jpprod, lvar_jpprod!,
  lvar_jptprod, lvar_jptprod!,
  uvar_jac_param_structure, uvar_jac_param_structure!,
  uvar_jac_param_coord, uvar_jac_param_coord!,
  uvar_jpprod, uvar_jpprod!,
  uvar_jptprod, uvar_jptprod!

function get_param_values end
function set_param_values! end
function grad_param! end
function jac_param_structure! end
function jac_param_coord! end
function jpprod! end
function jptprod! end
function hess_param_structure! end
function hess_param_coord! end
function hpprod! end
function hptprod! end
function lcon_jac_param_structure! end
function lcon_jac_param_coord! end
function lcon_jpprod! end
function lcon_jptprod! end
function ucon_jac_param_structure! end
function ucon_jac_param_coord! end
function ucon_jpprod! end
function ucon_jptprod! end
function lvar_jac_param_structure! end
function lvar_jac_param_coord! end
function lvar_jpprod! end
function lvar_jptprod! end
function uvar_jac_param_structure! end
function uvar_jac_param_coord! end
function uvar_jpprod! end
function uvar_jptprod! end

function grad_param(nlp::NLPModels.AbstractNLPModel{T, S}, x::AbstractVector) where {T, S}
  @lencheck NLPModels.get_nvar(nlp) x
  g = S(undef, get_nparam(nlp))
  return grad_param!(nlp, x, g)
end

function jac_param_structure(nlp::NLPModels.AbstractNLPModel)
  rows = Vector{Int}(undef, get_nnzjp(nlp))
  cols = Vector{Int}(undef, get_nnzjp(nlp))
  jac_param_structure!(nlp, rows, cols)
end

function jac_param_coord(nlp::NLPModels.AbstractNLPModel{T, S}, x::AbstractVector) where {T, S}
  @lencheck NLPModels.get_nvar(nlp) x
  vals = S(undef, get_nnzjp(nlp))
  return jac_param_coord!(nlp, x, vals)
end

function jpprod(nlp::NLPModels.AbstractNLPModel{T, S}, x::AbstractVector, v::AbstractVector) where {T, S}
  @lencheck NLPModels.get_nvar(nlp) x
  @lencheck get_nparam(nlp) v
  Jv = S(undef, NLPModels.get_ncon(nlp))
  return jpprod!(nlp, x, v, Jv)
end

function jptprod(nlp::NLPModels.AbstractNLPModel{T, S}, x::AbstractVector, v::AbstractVector) where {T, S}
  @lencheck NLPModels.get_nvar(nlp) x
  @lencheck NLPModels.get_ncon(nlp) v
  Jtv = S(undef, get_nparam(nlp))
  return jptprod!(nlp, x, v, Jtv)
end

function hess_param_structure(nlp::NLPModels.AbstractNLPModel)
  rows = Vector{Int}(undef, get_nnzhp(nlp))
  cols = Vector{Int}(undef, get_nnzhp(nlp))
  hess_param_structure!(nlp, rows, cols)
end

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

function hess_param_coord(
  nlp::NLPModels.AbstractNLPModel{T, S},
  x::AbstractVector;
  obj_weight::Real = one(T),
) where {T, S}
  @lencheck NLPModels.get_nvar(nlp) x
  vals = S(undef, get_nnzhp(nlp))
  return hess_param_coord!(nlp, x, vals; obj_weight = obj_weight)
end

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

function lcon_jac_param_structure(nlp::NLPModels.AbstractNLPModel)
  rows = Vector{Int}(undef, get_nnzjplcon(nlp))
  cols = Vector{Int}(undef, get_nnzjplcon(nlp))
  lcon_jac_param_structure!(nlp, rows, cols)
end

function lcon_jac_param_coord(nlp::NLPModels.AbstractNLPModel{T, S}) where {T, S}
  vals = S(undef, get_nnzjplcon(nlp))
  return lcon_jac_param_coord!(nlp, vals)
end

function lcon_jpprod(nlp::NLPModels.AbstractNLPModel{T, S}, v::AbstractVector) where {T, S}
  @lencheck get_nparam(nlp) v
  Jv = S(undef, NLPModels.get_ncon(nlp))
  return lcon_jpprod!(nlp, v, Jv)
end

function lcon_jptprod(nlp::NLPModels.AbstractNLPModel{T, S}, v::AbstractVector) where {T, S}
  @lencheck NLPModels.get_ncon(nlp) v
  Jtv = S(undef, get_nparam(nlp))
  return lcon_jptprod!(nlp, v, Jtv)
end

function ucon_jac_param_structure(nlp::NLPModels.AbstractNLPModel)
  rows = Vector{Int}(undef, get_nnzjpucon(nlp))
  cols = Vector{Int}(undef, get_nnzjpucon(nlp))
  ucon_jac_param_structure!(nlp, rows, cols)
end

function ucon_jac_param_coord(nlp::NLPModels.AbstractNLPModel{T, S}) where {T, S}
  vals = S(undef, get_nnzjpucon(nlp))
  return ucon_jac_param_coord!(nlp, vals)
end

function ucon_jpprod(nlp::NLPModels.AbstractNLPModel{T, S}, v::AbstractVector) where {T, S}
  @lencheck get_nparam(nlp) v
  Jv = S(undef, NLPModels.get_ncon(nlp))
  return ucon_jpprod!(nlp, v, Jv)
end

function ucon_jptprod(nlp::NLPModels.AbstractNLPModel{T, S}, v::AbstractVector) where {T, S}
  @lencheck NLPModels.get_ncon(nlp) v
  Jtv = S(undef, get_nparam(nlp))
  return ucon_jptprod!(nlp, v, Jtv)
end

function lvar_jac_param_structure(nlp::NLPModels.AbstractNLPModel)
  rows = Vector{Int}(undef, get_nnzjplvar(nlp))
  cols = Vector{Int}(undef, get_nnzjplvar(nlp))
  lvar_jac_param_structure!(nlp, rows, cols)
end

function lvar_jac_param_coord(nlp::NLPModels.AbstractNLPModel{T, S}) where {T, S}
  vals = S(undef, get_nnzjplvar(nlp))
  return lvar_jac_param_coord!(nlp, vals)
end

function lvar_jpprod(nlp::NLPModels.AbstractNLPModel{T, S}, v::AbstractVector) where {T, S}
  @lencheck get_nparam(nlp) v
  Jv = S(undef, NLPModels.get_nvar(nlp))
  return lvar_jpprod!(nlp, v, Jv)
end

function lvar_jptprod(nlp::NLPModels.AbstractNLPModel{T, S}, v::AbstractVector) where {T, S}
  @lencheck NLPModels.get_nvar(nlp) v
  Jtv = S(undef, get_nparam(nlp))
  return lvar_jptprod!(nlp, v, Jtv)
end

function uvar_jac_param_structure(nlp::NLPModels.AbstractNLPModel)
  rows = Vector{Int}(undef, get_nnzjpuvar(nlp))
  cols = Vector{Int}(undef, get_nnzjpuvar(nlp))
  uvar_jac_param_structure!(nlp, rows, cols)
end

function uvar_jac_param_coord(nlp::NLPModels.AbstractNLPModel{T, S}) where {T, S}
  vals = S(undef, get_nnzjpuvar(nlp))
  return uvar_jac_param_coord!(nlp, vals)
end

function uvar_jpprod(nlp::NLPModels.AbstractNLPModel{T, S}, v::AbstractVector) where {T, S}
  @lencheck get_nparam(nlp) v
  Jv = S(undef, NLPModels.get_nvar(nlp))
  return uvar_jpprod!(nlp, v, Jv)
end

function uvar_jptprod(nlp::NLPModels.AbstractNLPModel{T, S}, v::AbstractVector) where {T, S}
  @lencheck NLPModels.get_nvar(nlp) v
  Jtv = S(undef, get_nparam(nlp))
  return uvar_jptprod!(nlp, v, Jtv)
end
