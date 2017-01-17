include("sfo.jl")

# define an objective function and gradient
function f_df_autoencoder(theta, v)
    # [f, dfdtheta] = f_df_autoencoder(theta, v)
    #     Calculate L2 reconstruction error and gradient for an autoencoder
    #     with sigmoid nonlinearity.
    #     Parameters:
    #         theta - A cell array containing
    #              {[weight matrix], [hidden bias], [visible bias]}.
    #         v - A [# visible, # datapoints] matrix containing training data.
    #              v will be different for each subfunction.
    #     Returns:
    #         f - The L2 reconstruction error for data v and parameters theta.
    #         df - A cell array containing the gradient of f with each of the
    #              parameters in theta.

    W = theta[1];
    b_h = theta[2];
    b_v = theta[3];
    #h = 1./(1. + np.exp(-(np.dot(theta['W'], v) + theta['b_h'])))
    calculation = W * v
    calculation += repmat(b_h, 1, size(calculation)[2])
    h = 1./(1 + exp(-calculation))

    calculation = W' * h
    v_hat = calculation + repmat(b_v, 1, size(calculation)[2]) # np.dot(theta['W'].T, h) + theta['b_v']
    ##f = np.sum((v_hat - v)**2) / v.shape[1]
    f = sum(sum((v_hat - v) * (v_hat - v)')) / size(v, 2)
    dv_hat = 2*(v_hat - v) / size(v, 2);
    db_v = sum(dv_hat, 2);
    dW = h * dv_hat';
    dh = W * dv_hat;
    db_h = sum(dh.*h.*(1-h), 2);
    dW = dW + dh.*h.*(1-h) * v';
    # give the gradients the same order as the parameters
    dfdtheta = [dW, db_h, db_v];

    return f, dfdtheta
end


M = 20
J = 10
D = 100000
N = convert(Int, floor(sqrt(D)/10.))
v = randn(M,D)
theta_init = [randn(J,M), randn(J,1), randn(M,1)];
print(N)
# create the array of subfunction specific arguments
sub_refs = []
for i in 1:N
  # extract a single minibatch of training data.
  append!(sub_refs, v[:,i:N:end])
end


SFO(f_df_autoencoder,theta_init,sub_refs)
