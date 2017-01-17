include("types.jl")


function update_subspace(v, P, K_current)
  if K_current >= sfo_const.M
    # no need to update the subspace if it spans the full space
    return
  end

  # TODO: bad vector check missing

  v_length = norm(v, 2)

  if v_length < sfo_const.eps
    # if the new vector is too short, nothing to do
    return
  end

  # Unit length for vector v
  v_norm = v / v_length

  # Find the component of x pointing out of the existing subspace.
  # We need to do this multiple times for numerical stability.
  for i in 1:3
    println(v_norm)
    println((P' * v_norm))
    println(P * (P' * v_norm))
    v_norm -= P * (P' * v_norm)
    ss = norm(v_norm)

    if ss < sfo_const.eps
      # it barely points out of the existing subspace
      # no need to add a new direction to the subspace
      return
    end

    # make it unit length
    v_norm = v_norm / ss

    # if it was already largely orthogonal then numerical
    # stability will be good enough
    # TODO replace this with a more principled test
    if ss > 0.1
      break
    end
  end

  P[:,K_current] = v_norm
  K_current += 1

  if K_current >= sfo_const.K_max
    # the subspace has exceeded its maximum allowed size -- collapse it
    #TODO: SOME EVENT STUFF
    # xl may not be in the history yet, so we pass it in explicitly to make
    # sure it's used
    vl = P * v
    # TODO implement collapse subspace
  end
end


function init_subspace(theta, M, K_max, display, K_current)
  P = zeros(M, K_max)
  theta_norm = norm(theta, 2)
  if theta_norm > 0
    print(theta)
    P[:, 1] = theta / theta_norm
  else
    rand_init_vec = rand(M) # Maybe we have to take the shape of theta here.
    P[:, 1] = rand_init_vec / norm(rand_init_vec, 2)
  end

  if M == K_max
    if display > 1
      println("Subspace is full span: M:", M, " K_max:", K_max)
    end
    P = eye(M)
    K_current = M+1
  end

  return P, K_current
end
