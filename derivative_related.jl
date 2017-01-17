include("types.jl")
include("subspace.jl")

function f_df_wrapper(
    vars,
    consts,
    theta_in, index; return_full=false
  )
  f, df = consts.f_df(theta_in, consts.subfunction_references[index]) #Todo: add kwargs args
  # add time tracking

  if return_full
    return f, df
  end

  # update_subspace(vars, consts, df) -> have to fix update_subspace

  df_proj = vars.P' * df # project into subspace
#  print(size(vars.hist_f))
  #push!(vars.hist_f, f) probably not necessary
  vars.eval_count[index] += 1
  vars.eval_count_total += 1

  return f, df_proj
end


function update_hessian(vars, consts, indx)
  gd = find(sum(vars.hist_deltatheta[:,:,indx].^2, 1).>0)
  num_gd = length(gd)
  if num_gd == 0
    if consts.display > 0
      println("no history")
    end

    vars.b[:,:,indx] = 0.
    H = get_full_H_with_diagonal(vars, consts)
    U, V = eig(H)
    vars.min_eig_sub[indx] = median(U) / sum(vars.active)
    vars.max_eig_sub[indx] = vars.min_eig_sub[indx]

    if vars.eval_count > 2
      if consts.display > 2 | sum(vars.eval_count) < 5
        println("Subfunction evaluated ", self.eval_count[indx], " times, but has no stored history.")
      end
      if sum(vars.eval_count) <
        print("You probably need to initialize SFO with a smaller hessian_init value.  Scaling down the Hessian to try to recover.  You're better off correcting the hessian_init value though!")
        self.min_eig_sub[indx] /= 10.
      end
    end

    return
  end
end

function get_full_H_with_diagonal(vars, consts)
  full_H_combined = vars.full_H + eye(consts.K_max) .* sum(vars.min_eig_sub[vars.active])
  return full_H_combined
end
