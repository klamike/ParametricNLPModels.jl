@testset "Parametric API" begin
  nlp = SimpleParamNLPModel()
  @test eltype(nlp) == Float64

  n = NLPModels.get_nvar(nlp)
  m = NLPModels.get_ncon(nlp)
  np = get_nparam(nlp)
  p1, p2 = nlp.ps

  @test np == 2
  @test get_nnzjp(nlp) == 1
  @test get_nnzhp(nlp) == 3
  @test get_nnzgp(nlp) == 2
  @test get_nnzjplcon(nlp) == 1
  @test get_nnzjpucon(nlp) == 1
  @test get_nnzjplvar(nlp) == 2
  @test get_nnzjpuvar(nlp) == 2

  x = [1.0, 2.0]
  y = [0.5]
  v_p = [0.3, 0.7]
  v_x = [0.4, 0.6]
  v_c = [1.2]
  σ = 2.0

  @testset "grad_param" begin
    gp_exact = [-2 * (x[1] - p1); -2 * (x[2] - p2)]
    @test grad_param(nlp, x) ≈ gp_exact
    gp = zeros(np)
    @test grad_param!(nlp, x, gp) ≈ gp_exact
    @test gp ≈ gp_exact
  end

  @testset "jac_param" begin
    rows, cols = jac_param_structure(nlp)
    @test rows == [1]
    @test cols == [1]
    vals_exact = [x[2]]
    @test jac_param_coord(nlp, x) ≈ vals_exact
    vals2 = zeros(get_nnzjp(nlp))
    @test jac_param_coord!(nlp, x, vals2) ≈ vals_exact
    @test vals2 ≈ vals_exact
  end

  @testset "jpprod / jptprod" begin
    Jv_exact = [x[2] * v_p[1]]
    @test jpprod(nlp, x, v_p) ≈ Jv_exact
    Jv = zeros(m)
    @test jpprod!(nlp, x, v_p, Jv) ≈ Jv_exact
    @test Jv ≈ Jv_exact

    Jtv_exact = [x[2] * v_c[1]; 0.0]
    @test jptprod(nlp, x, v_c) ≈ Jtv_exact
    Jtv = zeros(np)
    @test jptprod!(nlp, x, v_c, Jtv) ≈ Jtv_exact
    @test Jtv ≈ Jtv_exact
  end

  @testset "hess_param" begin
    rows, cols = hess_param_structure(nlp)
    @test rows == [1, 2, 2]
    @test cols == [1, 1, 2]
    @test hess_param_coord(nlp, x) ≈ [-2.0, 0.0, -2.0]
    @test hess_param_coord(nlp, x, y) ≈ [-2.0, y[1], -2.0]
    @test hess_param_coord(nlp, x, y; obj_weight = σ) ≈ [-2σ, y[1], -2σ]
    vals2 = zeros(get_nnzhp(nlp))
    @test hess_param_coord!(nlp, x, y, vals2; obj_weight = σ) ≈ [-2σ, y[1], -2σ]
    @test vals2 ≈ [-2σ, y[1], -2σ]
  end

  @testset "hpprod / hptprod" begin
    Hv_exact = [-2σ * v_p[1]; y[1] * v_p[1] - 2σ * v_p[2]]
    @test hpprod(nlp, x, y, v_p; obj_weight = σ) ≈ Hv_exact
    Hv = zeros(n)
    @test hpprod!(nlp, x, y, v_p, Hv; obj_weight = σ) ≈ Hv_exact
    @test Hv ≈ Hv_exact
    @test hpprod(nlp, x, v_p) ≈ [-2 * v_p[1]; -2 * v_p[2]]

    Htv_exact = [-2σ * v_x[1] + y[1] * v_x[2]; -2σ * v_x[2]]
    @test hptprod(nlp, x, y, v_x; obj_weight = σ) ≈ Htv_exact
    Htv = zeros(np)
    @test hptprod!(nlp, x, y, v_x, Htv; obj_weight = σ) ≈ Htv_exact
    @test Htv ≈ Htv_exact
  end

  @testset "lcon_jac" begin
    rows, cols = lcon_jac_param_structure(nlp)
    @test rows == [1]
    @test cols == [2]
    @test lcon_jac_param_coord(nlp) ≈ [-1.0]
    vals2 = zeros(get_nnzjplcon(nlp))
    @test lcon_jac_param_coord!(nlp, vals2) ≈ [-1.0]
    Jv_exact = [-v_p[2]]
    @test lcon_jpprod(nlp, v_p) ≈ Jv_exact
    Jv = zeros(m)
    @test lcon_jpprod!(nlp, v_p, Jv) ≈ Jv_exact
    Jtv_exact = [0.0; -v_c[1]]
    @test lcon_jptprod(nlp, v_c) ≈ Jtv_exact
    Jtv = zeros(np)
    @test lcon_jptprod!(nlp, v_c, Jtv) ≈ Jtv_exact
  end

  @testset "ucon_jac" begin
    rows, cols = ucon_jac_param_structure(nlp)
    @test rows == [1]
    @test cols == [2]
    @test ucon_jac_param_coord(nlp) ≈ [1.0]
    vals2 = zeros(get_nnzjpucon(nlp))
    @test ucon_jac_param_coord!(nlp, vals2) ≈ [1.0]
    Jv_exact = [v_p[2]]
    @test ucon_jpprod(nlp, v_p) ≈ Jv_exact
    Jv = zeros(m)
    @test ucon_jpprod!(nlp, v_p, Jv) ≈ Jv_exact
    Jtv_exact = [0.0; v_c[1]]
    @test ucon_jptprod(nlp, v_c) ≈ Jtv_exact
    Jtv = zeros(np)
    @test ucon_jptprod!(nlp, v_c, Jtv) ≈ Jtv_exact
  end

  @testset "lvar_jac" begin
    rows, cols = lvar_jac_param_structure(nlp)
    @test rows == [1, 2]
    @test cols == [1, 2]
    @test lvar_jac_param_coord(nlp) ≈ [-1.0, -1.0]
    vals2 = zeros(get_nnzjplvar(nlp))
    @test lvar_jac_param_coord!(nlp, vals2) ≈ [-1.0, -1.0]
    @test lvar_jpprod(nlp, v_p) ≈ -v_p
    Jv = zeros(n)
    @test lvar_jpprod!(nlp, v_p, Jv) ≈ -v_p
    @test lvar_jptprod(nlp, v_x) ≈ -v_x
    Jtv = zeros(np)
    @test lvar_jptprod!(nlp, v_x, Jtv) ≈ -v_x
  end

  @testset "uvar_jac" begin
    rows, cols = uvar_jac_param_structure(nlp)
    @test rows == [1, 2]
    @test cols == [1, 2]
    @test uvar_jac_param_coord(nlp) ≈ [1.0, 1.0]
    vals2 = zeros(get_nnzjpuvar(nlp))
    @test uvar_jac_param_coord!(nlp, vals2) ≈ [1.0, 1.0]
    @test uvar_jpprod(nlp, v_p) ≈ v_p
    Jv = zeros(n)
    @test uvar_jpprod!(nlp, v_p, Jv) ≈ v_p
    @test uvar_jptprod(nlp, v_x) ≈ v_x
    Jtv = zeros(np)
    @test uvar_jptprod!(nlp, v_x, Jtv) ≈ v_x
  end

  @testset "non-parametric behavior" begin
    plain = PlainNLPModel()
    @test get_nparam(plain) == 0
    @test get_nnzjp(plain) == 0
    @test get_nnzhp(plain) == 0
    @test get_nnzgp(plain) == 0
    @test get_nnzjplcon(plain) == 0
    @test get_nnzjpucon(plain) == 0
    @test get_nnzjplvar(plain) == 0
    @test get_nnzjpuvar(plain) == 0
    @test !get_grad_param_available(plain)
    @test !get_jac_param_available(plain)
    @test !get_hess_param_available(plain)
  end
end
