function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
X = [ones(m, 1) X];
z2 = Theta1 * X';
a2 = sigmoid(z2);

a2 = [ones(1, m); a2];

z3 = Theta2 * a2;
h_theta = sigmoid(z3);
h_theta_transpose = h_theta';

new_y = [];
for c = 1:num_labels
    new_y = [new_y (y == c)];
end
temp = 0;
for b = 1:m
    temp2 = 0;
    for c = 1:num_labels
%    	temp2 = temp2 + ((-1 * y(b, c) * log(h_theta'(b,c))) - (1 - y(b, c) * log(1- h_theta'(b,c))));


	temp2 = temp2 + ( (-1 * new_y(b,c) * log(h_theta_transpose(b,c))) - ((1 - new_y(b,c)) * log(1-h_theta_transpose(b,c))));
    end
    
    temp = temp + temp2;
end
J = temp/m;


Theta1_without_bias = Theta1(:,2:end);
Theta2_without_bias = Theta2(:,2:end);

Theta1_without_bias = Theta1_without_bias.^2;
Theta2_without_bias = Theta2_without_bias.^2;

regularization_factor = (sum(Theta1_without_bias(:)) + sum(Theta2_without_bias(:))) * (lambda/(2*m));
size(regularization_factor)
J = J + regularization_factor;

% Initialize array with elements from 1 to 10, used for converting y to 10 output classes
a = 1:num_labels;
big_delta_1 = zeros(size(Theta1));
big_delta_2 = zeros(size(Theta2));
for t = 1:m
    a_1 = X(t,:);
    z_2 = Theta1 * a_1';
    a_2 = sigmoid(z_2);
    a_2 = [1 ; a_2];
    z_3 = Theta2 * a_2;
    a_3 = sigmoid(z_3);
    delta_3 = a_3 - (a == y(t))';
    delta_2 = (Theta2' * delta_3) .* sigmoidGradient([1;z_2]);
    delta_2 = delta_2(2:end);
    big_delta_1 = big_delta_1 + delta_2 * a_1;
    big_delta_2 = big_delta_2 + delta_3 * a_2';
end

Theta1_grad(:, 1) = big_delta_1(:, 1) ./ m;
Theta1_grad(:, 2:end) = big_delta_1(:, 2:end) ./m + ((lambda/m) * Theta1(:, 2:end))

Theta2_grad(:, 1) = big_delta_2(:, 1) ./ m;
Theta2_grad(:, 2:end) = big_delta_2(:, 2:end) ./m + ((lambda/m) * Theta2(:, 2:end))
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
