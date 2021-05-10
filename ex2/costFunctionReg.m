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
%               derivatives of the cost w.r.t. each parameter in

% Cost function
regu = 0;
for i = 2:size(theta,1)
    regu = regu + theta(i)^2;
end
regu = (lambda/(2*m))*regu;

J = -y'*log(sigmoid(X*theta)) - (1-y')*log(1-sigmoid(X*theta));
J = J/m + regu;

% Another way

% for i = 1:m
%     h = sigmoid(X(i,:) * theta);
%     J = J + (-y(i)*log(h) - (1 - y(i))*log(1-h));
% end

% J = -y'*log(sigmoid(X*theta)) - (1-y')*log(1-sigmoid(X*theta));
% J = J / m + norm(theta(2:size(theta)))^2 * lambda / (2*m);

% Gradient
n = size(theta);
s = (sigmoid(X*theta)-y)/m;
grad(1)= X(:,1)'*s;
grad(2:n) = X(:,2:n)' * s + lambda * theta(2:n) / m;



% =============================================================

end
