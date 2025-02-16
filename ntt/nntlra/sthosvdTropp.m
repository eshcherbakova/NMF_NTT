function [S,Q,R] = sthosvdTropp(X,r_fixed, par1, par2)
% Shcherbakova Elena M., Matveev Sergey A., 
% Smirnov Alexander P., Tyrtyshnikov Eugene E. 
% Study of performance of low-rank nonnegative tensor factorization methods //
% Russian Journal of Numerical Analysis and Mathematical Modelling.
% --2023. -- V. 38, ? 4. -- P. 231-239.


S = X;
N = ndims(X);
X_size = size(X);
Q = cell(1, N);
for n = 1:N
      [Q{n}, ~, ~] = Tropp(double(tenmat(S , n)), r_fixed(n), par1, par2);
      S = ttm(S,Q{n}.',n);

end
R = size(S);