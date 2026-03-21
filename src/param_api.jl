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

"""
    get_param_values(nlp)

Return ``p``, the current parameter values stored by `nlp`.
"""
function get_param_values end
"""
    set_param_values!(nlp, p)

Overwrite the parameter values stored by `nlp` with ``p``.
"""
function set_param_values! end
"""
    grad_param!(nlp, x, g)

Evaluate ``∇ₚf(x, p)``, the gradient of the objective function with respect to the parameters, at `x` in place.
"""
function grad_param! end
"""
    jac_param_structure!(nlp, rows, cols)

Return the structure of ``Jₚ(x, p)``, the constraints Jacobian with respect to the parameters, in sparse coordinate format in place.
"""
function jac_param_structure! end
"""
    jac_param_coord!(nlp, x, vals)

Evaluate ``Jₚ(x, p)``, the constraints Jacobian with respect to the parameters, at `x` in sparse coordinate format in place.
"""
function jac_param_coord! end
"""
    jpprod!(nlp, x, v, Jv)

Evaluate ``Jₚ(x, p)v``, the parameter-Jacobian-vector product, at `x` in place.
"""
function jpprod! end
"""
    jptprod!(nlp, x, v, Jtv)

Evaluate ``Jₚ(x, p)ᵀv``, the transposed-parameter-Jacobian-vector product, at `x` in place.
"""
function jptprod! end
"""
    hess_param_structure!(nlp, rows, cols)

Return the structure of ``∇ₓₚL(x, y, p)``, the mixed block of the Lagrangian Hessian, in sparse coordinate format in place.
"""
function hess_param_structure! end
"""
    hess_param_coord!(nlp, x, y, vals; obj_weight = 1)

Evaluate ``∇ₓₚL(x, y, p)``, the mixed block of the Lagrangian Hessian, at `(x, y)` in sparse coordinate format in place.
"""
function hess_param_coord! end
"""
    hpprod!(nlp, x, y, v, Hv; obj_weight = 1)

Evaluate ``∇ₓₚL(x, y, p)v``, the mixed-Hessian-vector product, at `(x, y)` in place.
"""
function hpprod! end
"""
    hptprod!(nlp, x, y, v, Htv; obj_weight = 1)

Evaluate ``∇ₓₚL(x, y, p)ᵀv``, the transposed-mixed-Hessian-vector product, at `(x, y)` in place.
"""
function hptprod! end
"""
    lcon_jac_param_structure!(nlp, rows, cols)

Return the structure of ``∂ℓᶜ(p) / ∂p`` in sparse coordinate format in place.
"""
function lcon_jac_param_structure! end
"""
    lcon_jac_param_coord!(nlp, vals)

Evaluate ``∂ℓᶜ(p) / ∂p`` in sparse coordinate format in place.
"""
function lcon_jac_param_coord! end
"""
    lcon_jpprod!(nlp, v, Jv)

Evaluate ``(∂ℓᶜ(p) / ∂p) v`` in place.
"""
function lcon_jpprod! end
"""
    lcon_jptprod!(nlp, v, Jtv)

Evaluate ``(∂ℓᶜ(p) / ∂p)ᵀ v`` in place.
"""
function lcon_jptprod! end
"""
    ucon_jac_param_structure!(nlp, rows, cols)

Return the structure of ``∂uᶜ(p) / ∂p`` in sparse coordinate format in place.
"""
function ucon_jac_param_structure! end
"""
    ucon_jac_param_coord!(nlp, vals)

Evaluate ``∂uᶜ(p) / ∂p`` in sparse coordinate format in place.
"""
function ucon_jac_param_coord! end
"""
    ucon_jpprod!(nlp, v, Jv)

Evaluate ``(∂uᶜ(p) / ∂p) v`` in place.
"""
function ucon_jpprod! end
"""
    ucon_jptprod!(nlp, v, Jtv)

Evaluate ``(∂uᶜ(p) / ∂p)ᵀ v`` in place.
"""
function ucon_jptprod! end
"""
    lvar_jac_param_structure!(nlp, rows, cols)

Return the structure of ``∂ℓˣ(p) / ∂p`` in sparse coordinate format in place.
"""
function lvar_jac_param_structure! end
"""
    lvar_jac_param_coord!(nlp, vals)

Evaluate ``∂ℓˣ(p) / ∂p`` in sparse coordinate format in place.
"""
function lvar_jac_param_coord! end
"""
    lvar_jpprod!(nlp, v, Jv)

Evaluate ``(∂ℓˣ(p) / ∂p) v`` in place.
"""
function lvar_jpprod! end
"""
    lvar_jptprod!(nlp, v, Jtv)

Evaluate ``(∂ℓˣ(p) / ∂p)ᵀ v`` in place.
"""
function lvar_jptprod! end
"""
    uvar_jac_param_structure!(nlp, rows, cols)

Return the structure of ``∂uˣ(p) / ∂p`` in sparse coordinate format in place.
"""
function uvar_jac_param_structure! end
"""
    uvar_jac_param_coord!(nlp, vals)

Evaluate ``∂uˣ(p) / ∂p`` in sparse coordinate format in place.
"""
function uvar_jac_param_coord! end
"""
    uvar_jpprod!(nlp, v, Jv)

Evaluate ``(∂uˣ(p) / ∂p) v`` in place.
"""
function uvar_jpprod! end
"""
    uvar_jptprod!(nlp, v, Jtv)

Evaluate ``(∂uˣ(p) / ∂p)ᵀ v`` in place.
"""
function uvar_jptprod! end
