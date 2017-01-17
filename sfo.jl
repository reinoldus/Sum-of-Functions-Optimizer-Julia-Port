
include("types.jl")
include("subfunctions.jl")
include("derivative_related.jl")
include("subspace.jl")
include("history.jl")

function SFO(
    f_df, theta, subfunction_references; args=nothing, kwargs=nothing, display=2,
    max_history_terms=10, hessian_init=1e5, init_subf=2, hess_max_dev = 1e8,
    hessian_algorithm="bfgs", subfunction_selection="distance both",
    max_gradient_noise=1., max_step_length_ratio=10., minimum_step_length=1e-8
  )
  # Just initialize all the parameters we have to keep
  # params = SFO_States.SFO_Parameters(
  #   f_df, theta, subfunction_references, args, kwargs, display,
  #   max_history_terms, hessian_init, init_subf, hess_max_dev, hessian_algorithm,
  #   subfunction_selection, max_gradient_noise, max_step_length_ratio,
  #   minimum_step_length
  # )
  N = length(subfunction_references)
  M = length(theta) # expected to be nx1
  K_min = min(2*N+2, M)
  K_max = convert(Int, min(M, ceil(K_min * 1.5)))
  K_current = 1

  old_theta = copy(theta)
  theta = vec(theta)
  print(old_theta == theta)
  println(size(theta), size(old_theta))

  consts = SFO_States.SFO_Constants(
    display, # verbosity
    f_df,
    N, # Number of subfunctions
    subfunction_references, # subfunction_references
    "random",
    M,
    K_min,
    K_max
  )

  # init theta, method doesn't exist in original
  P, K_current = init_subspace(theta, M, K_max, display, K_current)

  #projecte theta into the subspace!
  theta_proj = P' * theta
  active = convert(Array{Bool}, zeros(N))
  inds = rand(1:N)
  active[inds] = true
  min_eig_sub = zeros(N)
  max_eig_sub = zeros(N)
  min_eig_sub[inds] = hessian_init
  max_eig_sub[inds] = hessian_init

  vars = SFO_States.SFO_Vars(
    theta, # theta
    copy(theta), # theta_prior_step
    P, # P
    min_eig_sub,# min_eig_sub
    max_eig_sub,# max_eig_sub
    active, #  active | create a list for each active subfunction
    0, # total_distance |
    zeros(N), # number of function evaluations for each subfunction
    0, # eval_count_total | total evaluation count
    K_current,
    theta_proj,
    repmat(theta_proj, 1, N), # last_theta | Holds the last gradients
    zeros(K_max, N), # last_df | Holds the last gradients
    zeros(K_max, max_history_terms, N), # hist_deltatheta
    zeros(K_max, max_history_terms, N), # hist_deltadf
    ones(N, max_history_terms), # hist_f | history of evaluations of f
    # todo in python implementation this is complex -> not sure whether julia converts automagically...
    zeros(K_max, 2 * max_history_terms, N),# b | the approximate Hessian for each subfunction is stored as np.dot(self.b[:.:.index], self.b[:.:.inedx].T)
    zeros(K_max, K_max) # full_H | the full Hessian (sum over all the subfunctions)
  )

  optimization_step(vars, consts)
  # M = length(theta)
  # K_min = min(M, N*2+2)
  #
  #
  # sld_subspace = zeros(Float64, (M, K_max))
  #
  # SFO_Parameters(
  #   M,
  #   N,
  #   K_min,
  #   K_max
  # )
  println("worked")

end

function optimization_step(vars::SFO_States.SFO_Vars, consts::SFO_States.SFO_Constants)
  trgt_index = get_target_index(vars, consts)

  if consts.display > 2
    #TODO: add output
    println("Sorry No output yet... pure laziness")
  end

  # TODO: Add event stuf

  f, df_proj = f_df_wrapper(vars, consts, vars.theta, trgt_index)

  # TODO: IMPLEMENT HANDLE STEP FAILURE

  # add the change in theta and the change in gradient to the history for this subfunction
  update_history(vars, consts, trgt_index, vars.theta, f, df_proj)

  # increment the total distance traveled using the last update
  vars.total_distance += norm(vars.theta - vars.theta_prior_step, 2)

  # the current contribution from this subfunction to the total Hessian approximation
  H_pre_update = vars.b[:,:,trgt_index] *  vars.b[:,:,trgt_index]'

  update_hessian(vars, consts, trgt_index)
end

# Up next: Update subfunction_references, so we can do test f_df_wrapper
# Use sfo-demo.py
