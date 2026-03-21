# ParametricNLPModels.jl

!!! warning
    This documentation is a work in progress.
    Please open an issue if content is missing or unclear.

ParametricNLPModels.jl extends `NLPModels.jl` with metadata and derivative interfaces for model parameters.

## Installation

```julia
import Pkg
Pkg.add(; url = "https://github.com/klamike/ParametricNLPModels.jl")
```

## Usage

To make your `NLPModel` model parametric:

1. Add a `param_meta::ParametricNLPModelMeta` field to the model.
2. Implement `get_param_values` and `set_param_values!`.
3. Implement the parameter derivative methods your model supports, such as `grad_param!`, `jac_param_coord!`, `jpprod!`, `hess_param_coord!`, `hpprod!`, and `hptprod!`.

## MadDiff integration

The table below lists the parameter methods used by each mode:

| Mode | Methods used |
| --- | --- |
| `jvp` | `hpprod!`, `jpprod!`, `lvar_jpprod!`, `uvar_jpprod!`, `lcon_jpprod!`, `ucon_jpprod!` |
| `jvp` objective term `dobj` | `grad_param!` |
| `vjp` | `hptprod!`, `jptprod!`, `lvar_jptprod!`, `uvar_jptprod!`, `lcon_jptprod!`, `ucon_jptprod!` |
| `vjp` objective seed `dobj` | `grad_param!` |
| `jacobian!` | `hess_param_structure`, `hess_param_coord!`, `jac_param_structure`, `jac_param_coord!`, `lvar_jac_param_structure`, `lvar_jac_param_coord!`, `uvar_jac_param_structure`, `uvar_jac_param_coord!`, `lcon_jac_param_structure`, `lcon_jac_param_coord!`, `ucon_jac_param_structure`, `ucon_jac_param_coord!` |
| `jacobian!` objective row `dobj` | `grad_param!` |
| `jacobian_transpose!` | `hess_param_structure`, `hess_param_coord!`, `jac_param_structure`, `jac_param_coord!`, `lvar_jac_param_structure`, `lvar_jac_param_coord!`, `uvar_jac_param_structure`, `uvar_jac_param_coord!`, `lcon_jac_param_structure`, `lcon_jac_param_coord!`, `ucon_jac_param_structure`, `ucon_jac_param_coord!` |
| `jacobian_transpose!` objective column `dobj` | `grad_param!` |

This means, e.g.

1. If you only want `vjp` or `jvp`, you do not need the coordinate-form Jacobian or Hessian methods.
2. If a part of the model is not parametric, then the corresponding methods can be omitted. For example, if the variable bounds do not depend on the parameters, you do not need `lvar_*` or `uvar_*`; if the constraint bounds do not depend on the parameters, you do not need `lcon_*` or `ucon_*`.

`MadDiff` decides whether to call these methods from the parameter metadata. It first checks the `ParametricNLPModelMeta` counts such as `nnzgp`, `nnzjp`, `nnzhp`, `nnzjplvar`, `nnzjpuvar`, `nnzjplcon`, and `nnzjpucon` which describe which derivatives are actually nonzero for your model. If that part of the model is not parametric, its `nnz*` count should be zero, and the corresponding methods do not need to be implemented since `MadDiff` skips the call altogether.
