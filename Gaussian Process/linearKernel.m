function K = linearKernel(X1, X2, ko, k1)


if(size(X1, 1)==1 && size(X2, 1)==1) % K is scala
    K = ko^2 + k1^2*(X1' * X2);
else
    K = zeros(size(X1, 1), size(X2, 1));
   
    for n = 1 : size(X1, 1)
        for m = 1 : size(X2, 1)
            K(n, m) = ko^2 + k1^2*(X1(n, :) * X2(m, :)');
        end
    end
end