function [J, grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
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
% add ones (bias) to the first column of X
X = [ones(m, 1), X];
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% =========================================================================
% ======= LAYER 2 =======
% sigmoid input (can be vector or matrix - depending on Theta1)
z2 = X*Theta1';
% activation unit (this can be matrix or vector, depending on output of sigmoid)
a2 = sigmoid(z2);
% add ones to first column of a2
a2 = [ones(m, 1), a2];
% ======= LAYER 3 =======
% sigmoid input (can be vector or matrix - depending on Theta2)
z3 = a2*Theta2';
% activation unit (this can be matrix or vector, depending on output of sigmoid)
a3 = sigmoid(z3);

% Matrix of all classes
Y_k  = eye(num_labels);
temp_cost = 0;

% iterate through all samples and compute cost of each output
for i = 1:m
   h_theta = a3(i, :);
   temp_cost = temp_cost + (log(h_theta)*Y_k(:,y(i)) + log(1-h_theta)*(1-Y_k(:,y(i))));
end
J = (-1/m)*temp_cost;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
