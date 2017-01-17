include("types.jl")

print("been executed")

function get_target_index(vars, consts)
  # if an active subfunction has one evaluation, get a second
  # so we can have a Hessian estimate
  gd = find(vars.eval_count .== 1 & vars.active .== true)
  if length(gd) > 0
    return gd[1]
  end
  # If an active subfunction has less than two observations, then
  # evaluate it.  We want to get to two evaluations per subfunction
  # as quickly as possibly so that it's possible to estimate a Hessian
  # for it
  gd = find(vars.eval_count .< 2 & vars.active .== true)
  if length(gd) > 0
    rand(gd) # Pick a value at random
  end

  #TODO: Implement other subfunction selections

  if consts.subfunction_selection == "random"
    # choose an index to update at random
    return rand(find(vars.active .== true))
  end
end
