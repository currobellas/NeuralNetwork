function W = randInitializeWeights(L_in, L_out)
% Init NN layer's weights with L_in input connections
% and L_out output connections.
% W:  L_out x (L_in + 1) array to get bias terms from de NN.

    W = zeros(L_out, 1 + L_in);

    epsilon_init = 0.12;
    W = rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init;
end
