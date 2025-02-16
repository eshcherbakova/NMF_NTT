function [Ur, Sr, Vhr]=HMT(X, rank, p, k)
% Shcherbakova Elena M., Matveev Sergey A., 
% Smirnov Alexander P., Tyrtyshnikov Eugene E. 
% Study of performance of low-rank nonnegative tensor factorization methods //
% Russian Journal of Numerical Analysis and Mathematical Modelling.
% --2023. -- V. 38, ? 4. -- P. 231-239.


n = size(X, 2);
Psi = TestMatrix(n, k);
Z1 = X * Psi;
[Q,~] = qr(Z1, 0);
for i = 1:p
    Z2 = Q'*X;
    [Q,~] = qr(Z2', 0);
    Z1 = X * Q;
    [Q,~] = qr(Z1, 0);
end
Z2 = Q'*X;
[Ur, Sr, Vhr] = svds(Z2, rank);
Ur = Q * Ur;
end
% def HMT(X, rank, p, k, distr='rademacher', rho=None):
%     n = X.shape[1]
% 
%     Psi = TestMatrix(n, k, distr, rho)
%     Z1 = X @ Psi
%     Q, _ = np.linalg.qr(Z1)
%     for _ in range(p):
%         Z2 = Q.T @ X
%         Q, _ = np.linalg.qr(Z2.T)
%         Z1 = X @ Q
%         Q, _ = np.linalg.qr(Z1)
%     Z2 = Q.T @ X
%     Ur, Sr, Vhr = svdr(Z2, rank)
%     Ur = Q @ Ur
%     
%     return Ur, Sr, Vhr
%     
    
    