include("sfo.jl")

using MAT

function log_reg(theta, X, y, lambda)
  number_of_examples = size(X)[2]
  log_reg_result = 0

  log_reg_result = lambda/2 * norm(theta)^2
  for i in 1:number_of_examples
    log_reg_result += log(1 + exp(-1*y[i] * dot(theta, X[:,i])))
  end

  log_reg_result
end

function d_log_reg(theta, X, y, lambda)
  number_of_examples = size(X)[2]

  d_log_reg_result = lambda * theta
  for i in 1:number_of_examples
    d_log_reg_result += -1*y[i]*X[:,i]* (exp(-1*y[i] * dot(theta, X[:,i]))/(1 + exp(-1*y[i] * dot(theta, X[:,i]))))
  end

  d_log_reg_result
end

function f_df(theta, i)
  vars = matread("mnist67.scale.1k.mat")
  X = vars["X"]
  realX = zeros(1000,748)
  for i in 1:1000
    for j in 1:748
      realX[i, j] = X[i, j]
    end
  end
  X = transpose(realX)
  y = vars["y"]
  y = vec(y)
  number_of_features = size(X)[1]
  number_of_examples = size(X)[2]
  N = convert(Int, floor(sqrt(number_of_examples)/10.))
  lambda = 0

  sub_refs = []
  #println(number_of_examples)
  for i in 1:number_of_examples
    # extract a single minibatch of training data.
    #append!(sub_refs, X[:,i:N:end])
    #println(length(sub_refs))
    push!(sub_refs, X[:,i])
  end
  return log_reg(theta, X, y, lambda), d_log_reg(theta, reshape(X[:,i], number_of_features, 1), y[i], lambda)
end

vars = matread("mnist67.scale.1k.mat")
X = vars["X"]
realX = zeros(1000,748)
for i in 1:1000
  for j in 1:748
    realX[i, j] = X[i, j]
  end
end
X = transpose(realX)
# y = vars["y"]
# y = vec(y)
number_of_features = size(X)[1]
number_of_examples = size(X)[2]
# N = convert(Int, floor(sqrt(number_of_examples)/10.))
# lambda = 0

sub_refs = []
#println(number_of_examples)
for i in 1:number_of_examples
  # extract a single minibatch of training data.
  #append!(sub_refs, X[:,i:N:end])
  #println(length(sub_refs))
  push!(sub_refs, X[:,i])
end
theta = fill(0.0, number_of_features)
#a = f_df(theta, X, y, lambda)
#print(length(sub_refs))
SFO(f_df,theta,sub_refs)
