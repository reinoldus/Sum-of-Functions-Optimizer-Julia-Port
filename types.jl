module SFO_States

  type History
    hist_delta_theta  # the history of theta changes for each subfunction
    hist_delta_df     # the history of gradient changes for each subfunction
    hist_f            # the history of function values for each subfunction
  end

  type SFO_Vars
    theta
    theta_prior_step
    # step_scale
    P
    min_eig_sub
    max_eig_sub
    # iter_since_active_growth
    active
    total_distance
    eval_count
    eval_count_total
    K_current
    theta_proj
    last_theta
    last_df
    hist_deltatheta
    hist_deltadf
    hist_f
    # hist_f_flat
    b
    full_H
    # events
    # cyclic_subfunction_index
  end

  immutable SFO_Constants
    display::Int64
    f_df::Function
    # args
    # kwargs
    # max_history
    # max_gradient_noise
    # hessian_init::Float64
    N::Int64 # Number of subfunctions
    subfunction_references::Array
    # hessian_algorithm::String
    subfunction_selection::String
    # max_step_length_ratio::Float64
    M::Int
    # minimum_step_length
    K_min::Int
    K_max::Int
    # eps::Float64
  end

  type SFO_Parameters
    f_df::Function
    theta::Int
    subfunction_references
    args
    kwargs
    display
    max_history_terms
    hessian_init::Float64
    init_subf::Int
    hess_max_dev::Float64
    hessian_algorithm::String
    subfunction_selection::String
    max_gradient_noise::Float64
    max_step_length_raio::Float64
    minimum_step_length::Float64
  end
end
