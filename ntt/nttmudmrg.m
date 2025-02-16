function [erAr2, time, t]=nttmudmrg(ta_orig, r_f, setsize, setd, setr, maxIter, max_iter)
%with DMRG

% Shcherbakova, E. Nonnegative Tensor Train Factorization with DMRG Technique //
% Lobachevskii Journal of Mathematics. -- 2019. -- V. 40, ? 11. -- P. 1863-1872.





%uncompress = full(ta_orig, ta_orig.n');
% uncompress = ta_orig;

tic

% ta = tt_tensor(uncompress, 1.5e-2);

ta = ta_orig;

t0 = tt_rand(setsize,setd,setr);

t = tt_als(ta, t0, 10);

for i = 1:setd
    t{i} = abs(t{i});
end


d = setd;

iter = 1;
eps = 1.6;

erAr2 = [];


while 1
    
for i = 1:(d - 1)
    
    if i > 1
       qlessi = dot(chunk(t,1, i-1), chunk(t,1, i-1));
    else
       qlessi = 1;
    end
    
    %dlessi = diag(1./diag(qlessi));
    
    
    if i < d - 1
       qbigi = dot(chunk(t, i + 2, d), chunk(t, i + 2, d));
    else
       qbigi = 1;
    end
    
    %dbigi = diag(1./diag(qbigi));
    
    if i < d - 1
       vht1 = dot(chunk(ta, i + 2, d), chunk(t, i + 2, d));
    else 
       vht1 = 1;
    end
    
    if i > 1
        vht2 = dot(chunk(ta,1, i-1), chunk(t,1, i-1));
    else 
        vht2 = 1;
    end
    
    numer = kron(vht1, vht2);
    
    
    gta = full(chunk(ta, i, i + 1));

    if i > 1
    gta = permute(gta, [2 1 3]);
    end
    
    gta = reshape(gta, ta.n(i)*ta.n(i+1), ta.r(i)*ta.r(i+2));
       
%     numer = gta*numer;
    
    numer = max(gta*numer, 1e-9);

    denom = kron(qbigi, qlessi);
    
    gt = full(chunk(t, i, i + 1));
    
    if i > 1
    gt = permute(gt, [2 1 3]);
    end
    
    gt = reshape(gt, t.n(i)*t.n(i+1), t.r(i)*t.r(i+2));
    
    denom = gt*denom;
    
%     res = max(gt .* (numer ./ denom), 1e-9);
    
    res = (gt .* (numer ./ denom));
    
    res = reshape(res, t.n(i)*t.n(i+1), t.r(i), t.r(i+2));

    res = permute(res, [2 1 3]);

    
    res = reshape(res, t.n(i)*t.r(i), t.n(i+1)*t.r(i + 2));
    
%     opts=struct('eps', 4.1e-2, 'alg','hals');
      opts=struct('eps', 1e-3, 'alg','hals', 'r', r_f(i+1));
%     [W, H, U, V] = lraNMF_cross(res,opts);
%     k = size(W, 2);    
%     k = min(fix(1.5*k), min(size(res)));
%     opts=struct('eps', 0.001, 'alg','hals', 'r', k, 'x', U, 'y', V);
    [W,H, ~, ~] = lraNMF_cross(res,opts);
%         
%         [W,H, ~, ~] = lraNMF_cross(res,opts);
    
%      opts=struct('eps', 0.01, 'alg','hals');
%     [W, H, U, V] = lraNMF_cross(res,opts);
%     k = size(W, 2);
%     iterin = 1;
%     
% %     disp(norm(c-W*H,'fro')/norm(c, 'fro'));
%     
%     while (norm(res-W*H,'fro')/norm(res, 'fro') > eps && iterin <= max_iter && k < min(size(res)) && k > 1)
%         if (rem(iterin, 10) == 1)
% %             k = min(2*k, fix((min(size(c)) + k)/2));
%             k = fix(k/2);
%         else
%             k = k - 1;
%         end
%         opts=struct('eps', 0.001, 'alg','hals', 'r', k, 'x', U, 'y', V);
%         
%         [W,H, ~, ~] = lraNMF_cross(res,opts);
% %         disp(norm(c-W*H,'fro')/norm(c, 'fro'))
%         
%         iterin = iterin + 1;
% %          k = k + 1;
%     end
        
    k = size(W, 2);
    W = reshape(W, t.r(i), t.n(i), k);  
    H = reshape(H, k, t.n(i+1), t.r(i+2));
    t{i} = W;
    t{i+1} = H;
    t.r(i+1) = k;
                       
end

iter = iter + 1;
erAr2 = [erAr2 ((norm(minus(ta, t)))/norm(ta))*100];
if erAr2(end) < eps || iter > maxIter
    break;
end

end

time = toc;