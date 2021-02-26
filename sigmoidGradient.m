function g = sigmoidGradient(z)
% Returns sigmoid function gradient evaluated at z. If  z is
% a vector or matrix, returns the gradient of each elementof z

    g = sigmoid(z).*(1-sigmoid(z));
end
