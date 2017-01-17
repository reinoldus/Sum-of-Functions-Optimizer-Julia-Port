include("types.jl")


function update_history(
    vars,
    consts,
    index, theta_proj, f, df_proj; skip_delta=false
  )

  if vars.eval_count[index] > 1 && !skip_delta
    # differences in gradient and position
    ddf = df_proj - vars.last_df[:, [index]] # TODO: Check whether this is the same as: self.last_df[:,[indx]]
    ddt = theta_proj - vars.last_theta[:,[index]] # TODO: Same as above

    # length of gradient and position change vectors
    lddt = norm(ddt)
    lddf = norm(ddf)

    # No idea what this is telling me, coding it anyway :O
    corr_ddf_ddt = (ddf' * ddt)[1,1] / (lddt*lddf)

    if consts.display > 3 && corr_ddf_ddt < 0
      println("Warning!  Negative dgradient dtheta inner product.  Adding it anyway.")
    end

    if lddt < consts.epsilon
      if consts.display > 2
        print("Largest change in theta too small (", lddt ,"). Not adding to history.")
      end
    elseif lddf < consts.epsilon
      if consts.display > 2
        print("Largest change in gradient too small  (", lddf ,"). Not adding to history.")
      end
    elseif abs(corr_ddf_ddt) < consts.epsilon
      if consts.display > 2
        print("Inner product between dgradient and dtheta too smalll (", corr_ddf_ddt ,"). Not adding to history.")
      end
    else
      if consts.display > 3
        # TODO: The original paper does: np.sum(ddt*ddf)/(lddt*lddf) for corr(ddf, dtheta) -> WEIRD!
        print("subf ||dtheta|| ", lddt, ", subf ||ddf|| ", lddf, ", corr(ddf, dtheta) ", corr_ddf_ddt)
      end
      print("?")
      # This should be equivalent to the python code
      # shift the history by one timestep;
      vars.hist_deltatheta[:, 2:end, index] = vars.hist_deltatheta[:,1:(end-1), index]
      # store the difference in theta since the subfunction was last evaluated;
      vars.hist_deltatheta[:, 1, index] = ddt

      # do the same thing for the change in gradient;
      vars.hist_deltadf[:, 2:end, index] = vars.hist_deltadf[:,1:(end-1), index]
      vars.hist_deltadf[:, 1, index] = ddt
    end
  end

  vars.last_theta[:, index] = theta_proj
  vars.last_df[:, index] = df_proj
  vars.hist_f[index, 2:end] = vars.hist_f[index, 1:(end-1)]
  vars.hist_f[index, 1] = f
end
