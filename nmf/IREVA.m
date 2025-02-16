function [W, r] = IREVA(A, n) 

x = ones(n, 1);
w = x'*A*x;
V = 1/sqrt(w) *A*x;
B = A - V*V';
L = cholcov(B);
L = L';
r = size(L,2) + 1;
V = [V L];
Vtilde = V(any(V,2),:);
Vtilde = normr(Vtilde);
R = []; 

for i = 1:r 

    [~, minind] = min(Vtilde(:,1));
    vi = Vtilde(minind,:)';
    c = Vtilde*vi;
    Z = find(c < 1e-13);
    Vtilde = Vtilde(Z,:);
    R = [R vi];

end
W = V*R;
W(W < 1e-13) = 0;