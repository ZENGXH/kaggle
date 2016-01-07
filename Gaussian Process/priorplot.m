clearvars;

% X is our sampling points for x
X = linspace(-10,10,500)';

kernelType = 'linear';
% init hyper para
ko = 1;
k1 = 1;
L = 1;
X1 = X;
X2 = X;
% -- COMPUTE KERNEL MATRIX K(X,X) ---
% size of K(X1, X2) = #num of rows of X1 x #num of rows of X2S
if strcmp(kernelType,'RBF')
    K = RBFKernel(X1, X2, ko, k1, L);
elseif strcmp(kernelType,'linear')
    K = linearKernel(X1, X2, ko, k1);
else
    K = constantKernel(X1, X2, ko); % constant kernel
end

% -- SET MU (MEAN OF THE PRIOR). TRY ZEROes INITIALLY --
mu = zeros(size(X1, 1),1);

% do different draws of f(X)
for i=1:100
    clf;
    
    % sample from multivariate gaussian prior
    y = mvnrnd(mu, K, 1);
    
    % plot 95% confidence interval (independent of draw)
    uncert = 2*sqrt(diag(K));  % +/- 2 sigma => 95% confidence interval
    jbfill(X, mu + uncert, ...
              mu - uncert, [0.8,0.8,0.8]); hold on;
%     fprintf('%f', y);
    % plot draw from the prior
    hold on;
    plot(X,y,'k','LineWidth',2);
    
    axis([-10 10 -5 5]);  % fix axis
    drawnow();
    pause(0.01);    % small delay
end