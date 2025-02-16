% Sequentially Truncated HOSVD
% Shcherbakova Elena M., Matveev Sergey A., 
% Smirnov Alexander P., Tyrtyshnikov Eugene E. 
% Study of performance of low-rank nonnegative tensor factorization methods //
% Russian Journal of Numerical Analysis and Mathematical Modelling.
% --2023. -- V. 38, ? 4. -- P. 231-239.

% input:
%   X:       a data tensor of size I_1 x I_2 x ... x I_N
%   epsilon: tolerance
% ouput:
%   S: a core tensor of size R_1 x R_2 x ... x R_N
%   Q: a cell containing N factor matrices, factor Q{n} has size I_n x R_n
%   R: a vector containg multilinear ranks of the core tensor, note that
%      R_n <= I_n for all n=1, 2, ... , N.
function [S,Q,R] = sthosvd(X,r_fixed)
S = X;
N = ndims(X);
X_size = size(X);
Q = cell(1, N);
for n = 1:N
%     [Q{n},~,~] = svdsketch(double(tenmat(S , n)),epsilon/N);
    [Q{n}, Sr, Vhr] = svds(double(tenmat(S , n)), r_fixed(n));
    S = ttm(S,Q{n}.',n);
%       X_size(n) = r_fixed(n);
%       S  = tensor((Sr * Vhr')', X_size);
end
R = size(S);