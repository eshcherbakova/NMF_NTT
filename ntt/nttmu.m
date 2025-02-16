function [t, erAr, time]=nttmu(ta, t, setsize, setd, setr, maxIter, als_r)

% Shcherbakova, E. Nonnegative Tensor Train Factorization with DMRG Technique //
% Lobachevskii Journal of Mathematics. -- 2019. -- V. 40, ? 11. -- P. 1863-1872.


% function [t, erAr, time]=nttmu(ta, setsize, setd, setr, maxIter, als_r)
global tmArray;

% ta = tt_tensor(ta, 1e-6);


% t0 = tt_rand(setsize,setd,setr);
% 
% % t = t0;
% t = tt_als(ta, t0, als_r);

% t = ta;
% 
% t = tt_tensor(tmArray, 0.01, setsize, 1, 1, setr(2:setd));
% 
% for i = 1:setd
%     t{i} = max(t{i}, 1e-8);
% end

% t = tt_rand_pos(setsize',setd,setr(2:setd)');

% t = ta;

d = setd;

iter = 1;
% eps = 0.0775;
eps = 0.01;
erAr = [];

tic
while 1
    
for i = 1:d
    
    if i > 1
       qlessi = dot(chunk(t,1, i-1), chunk(t,1, i-1));
    else
       qlessi = 1;
    end
    
    %dlessi = diag(1./diag(qlessi));
    
    
    if i < d
       qbigi = dot(chunk(t, i + 1, d), chunk(t, i + 1, d));
    else
       qbigi = 1;
    end
    
    %dbigi = diag(1./diag(qbigi));
    
    if i < d
       vht1 = dot(chunk(ta, i + 1, d), chunk(t, i + 1, d));
    else 
       vht1 = 1;
    end
    
    if i > 1
        vht2 = dot(chunk(ta,1, i-1), chunk(t,1, i-1));
    else 
        vht2 = 1;
    end
    
    numer = kron(vht1, vht2);
    
    gta = permute(ta{i}, [2 1 3]);
    
    gta = reshape(gta, ta.n(i), ta.r(i)*ta.r(i+1));
    
%     numer = gta*numer;
    
    numer = max(gta*numer, 1e-9);

    denom = kron(qbigi, qlessi);
    
    gt = permute(t{i}, [2 1 3]);
    
    gt = reshape(gt, t.n(i), t.r(i)*t.r(i+1));
    
    denom = gt*denom;
    
%     res = max(gt .* (numer ./ denom), 1e-9);
    
    res = (gt .* (numer ./ denom));

    res = reshape(res, t.n(i), t.r(i), t.r(i + 1));
    t{i} = permute(res, [2 1 3]);
    
                       
end

iter = iter + 1;

% disp(iter);

erAr = [erAr (norm(minus(ta, t)))/norm(ta)];
er_diff = 1;
if length(erAr) > 1
    er_diff = abs(erAr(end)-erAr(end-1));
end
if norm(minus(ta, t))/norm(ta) < eps || iter > maxIter
    break;
end
% erAr = [erAr (norm(minus(ta, t)))];
% if norm(minus(ta, t)) < eps || iter > maxIter
%     break;
% end

end

time = toc;
