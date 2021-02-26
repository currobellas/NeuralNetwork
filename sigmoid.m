function g = sigmoid(z)
% Implements sigmoid function
    g = 1.0 ./ (1.0 + exp(-z));
end
