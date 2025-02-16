function [Ur, Sr, Vhr]=Tropp(X, rank, k, l)
% Shcherbakova Elena M., Matveev Sergey A., 
% Smirnov Alexander P., Tyrtyshnikov Eugene E. 
% Study of performance of low-rank nonnegative tensor factorization methods //
% Russian Journal of Numerical Analysis and Mathematical Modelling.
% --2023. -- V. 38, ? 4. -- P. 231-239.



m = size(X, 1);
n = size(X, 2);

Psi = TestMatrix(n, k);
Phi = TestMatrix(l, m);

Z = X * Psi;

[Q, R] = qr(Z, 0);
W = Phi * Q;
[P, T] = qr(W, 0);
G = inv(T) * P' * Phi * X;
[Ur, Sr, Vhr] = svds(G, rank);
Ur = Q * Ur;

end



% def Tropp(X, rank, k, l, distr='rademacher', rho=None):
%     m, n = X.shape
%         
%     Psi = TestMatrix(n, k, distr, rho)
%     Phi = TestMatrix(l, m, distr, rho)
%     Z = X @ Psi
%     Q, R = np.linalg.qr(Z)
%     W = Phi @ Q
%     P, T = np.linalg.qr(W)
%     G = np.linalg.inv(T) @ P.T @ Phi @ X
%     Ur, Sr, Vhr = svdr(G, rank)
%     Ur = Q @ Ur
%     
%     return Ur, Sr, Vhr