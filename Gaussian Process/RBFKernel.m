function K = RBFKernel(X1, X2, ko, k1, L)
% L has effect on the smoothness of the prior

if(size(X1, 1)==1 && size(X2, 1)==1) % K is scala
    K = ko^2 + k1^2*exp(-norm(X1 - X2)/L^2);
else
    K = zeros(size(X1, 1), size(X2, 1));
     
    for n = 1 : size(K, 1)
        for m = 1 : size(K, 2)
            K(n, m) = ko^2 + k1^2 * exp(-norm(X1(n, :) - X2(m, :))/L^2);
        end
    end
end