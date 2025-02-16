function [xt, yt, zt, wt, vt, cmp, eps1, eps2, eps3, eps4, eps5] = cmpMethodTimeRank2(m,n,seed) 
r = 2;
rng(seed)
W = abs(randn(m, r));
rW = rank(W);    
W = W(:, 1:rW);
rng(seed)
H = abs(randn(rW, n));
H = H./(ones(size(H, 1), 1)*sum(H));
H = [eye(rW) H];
rng(seed)
shuffle = randsample(1:n,n);
H = H(:,shuffle);

M = W*H;

tic
[u, v] = cross2d_mat(M, 1e-10);

Duvec = sum(u, 1);
% Get sum of columns and replicate vertically.
% Du = repmat(Duvec, [m, 1]);
% Do the division.
% u = u ./ Du;

RV = v';
RV = spdiags((Duvec)', 0, rW, rW)*RV;

Drv = repmat(sum(RV, 1), [r, 1]);
% Do the division.
RV = RV ./ Drv;

[~, J11] = min(RV(2, :));
[~, J12] = max(RV(2, :));

J1 = [J11 J12];

W = M(:, J1);

k = find(W(:, 1), 1);

for i = 1:m
    if (W(k, 1) * W(i, 2) - W(i, 1) * W(k, 2) ~= 0)
        break
    end
end

G = [W(i, 2) -W(k, 2); -W(i, 1) W(k, 1)];
G = G .* 1/(W(k, 1) * W(i, 2) - W(i, 1) * W(k, 2));

xt = toc;

eps1 = norm(M - W*G*M([k i], :), 'fro');
% eps1 = 0;

tic
[J2,~,~] = FastSepNMF(M,rW,1);
W = M(:, J1);

k = find(W(:, 1), 1);

for i = 1:m
    if (W(k, 1) * W(i, 2) - W(i, 1) * W(k, 2) ~= 0)
        break
    end
end

G = [W(i, 2) -W(k, 2); -W(i, 1) W(k, 1)];
G = G .* 1/(W(k, 1) * W(i, 2) - W(i, 1) * W(k, 2));

yt = toc;

eps2 = norm(M - W*G*M([k i], :), 'fro');
% eps2 = 0;

cmp = isequal(sort(J1), sort(J2));

% tic
% rng(seed)
% W = rand(m, 2);
% rng(seed)
% H = rand(2, n);
% [W, H] = nmfsh_comb_rank2(M, W, H);
% zt = toc;
% 
% eps3 = norm(M - W*H, 'fro');
% 
zt = 0;
eps3 = 0;

% tic
% [U,V,~] = rank2nmf(M);
% wt = toc;
% 
% eps4 = norm(M - U*V, 'fro');
wt = 0;
eps4 = 0;

% tic
% [U,V,~] = rank2nmfUPG(M);
% vt = toc;
% 
% eps5 = norm(M - U*V, 'fro');
vt = 0;
eps5 = 0;

end
