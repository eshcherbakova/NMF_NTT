function [J] = lowrankSepNMF(u, v) 
% lowrankSepNMF - method for separable NMF based on lowrank approximation
% See E. Tyrtyshnikov and E. Shcherbakova, 
% Methods for Nonnegative Matrix Factorization Based on Low-Rank Cross Approximations,  
% Computational Mathematics and Mathematical Physics, 59 (8), pp. 1251-1266, 2019.
%
% J = lowrankSepNMF(u,v)
%
% ****** Input ******
% uv = M = WH + N : M is a noisy separable matrix, W is full rank,
%              H = [I,H']P where I is the identity matrix, H'>= 0 and its
%              columns sum to at most one, P is a permutation matrix, and
%              N is sufficiently small. Matrices u, W have r columns and
%              matrices v, H have r rows.
%
% ****** Output ******
% J        : index set of the extracted columns.

r = size(u, 2);
Duvec = sum(u, 1);
RV = v;
elemD = Duvec*RV;
elemD(elemD == 0) = 1;
Drv = repmat(elemD, [r, 1]);
RV = RV ./ Drv;

[J,~,~] = FastSepNMF(RV,r,0);

end
