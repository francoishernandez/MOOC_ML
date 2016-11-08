function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

H = sigmoid(theta' * X');

J = 1/size(X,1) * ((-y' .* log(H)) - ((1-y') .* log(1-H))) * ones(size(X,1),1) + lambda/(2*size(X,1))*(theta.^2)'*[0;ones(size(theta,1)-1,1)];

grad = (1/size(X,1) * ((H - y')*X)') + (lambda/m * theta .* ([0;ones(size(theta,1)-1,1)]));




% =============================================================

end
