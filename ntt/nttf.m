function [t, resSVD, wtime, relError1, nr, relError3, nrSVD, wtimeSVD]=nttf(uncompress)

% Shcherbakova E., Tyrtyshnikov E. Nonnegative Tensor Train Factorizations and Some Applications // 
% Lecture Notes in Computer Science. -- 2020. -- V. 11958 -- P. 156-164.

% clearvars -except uncompress
tic

alg = {'mm', 'cjlin', 'als', 'alsobs', 'prob'};
eps = 0.5;
max_iter = 1;
n=size(uncompress); n=n(:); d=numel(n); r=ones(d+1,1);
d=numel(n);
c=uncompress;
core=[];
pos=1;
% ep=eps/sqrt(d-1);
n=n(:); r=r(:);
t=tt_tensor;

% resSVD = tt_tensor(uncompress, eps);

for i=1:d-1
    m=n(i)*r(i); c=reshape(c,[m,numel(c)/m]);
    %   [u,s,v]=svd(c,'econ');
    %   s=diag(s); r1=my_chop2(s,ep*norm(s));
    %   u=u(:,1:r1); s=s(1:r1);
    %   r(i+1)=r1;
    %   core(pos:pos+r(i)*n(i)*r(i+1)-1)=u(:);
    %   v=v(:,1:r1);
    %   v=v*diag(s); c=v';
    
    opts=struct('eps', 0.01, 'alg','hals');
    [W, H, U, V] = lraNMF_cross(c,opts);
    k = size(W, 2);
    iter = 1;
    disp(i);
%     disp(norm(c-W*H,'fro')/norm(c, 'fro'));
    
    while (norm(c-W*H,'fro')/norm(c, 'fro') > eps && iter <= max_iter && k < min(size(c)) && k > 1)
%         if (rem(iter, 10) == 1)
% %             k = min(2*k, fix((min(size(c)) + k)/2));
%             k = fix(k/2);
%         else
%             k = k - 1;
%         end
        disp(k);
        k = k + 1;
        opts=struct('eps', 0.01, 'alg','hals', 'r', k, 'x', U, 'y', V);
        
        [W,H, ~, ~] = lraNMF_cross(c,opts);
%         disp(norm(c-W*H,'fro')/norm(c, 'fro'))
        disp(iter);
        iter = iter + 1;
    end
    
    if (norm(c-W*H,'fro')/norm(c, 'fro') > eps)
        k = min(size(c));
        if (size(c, 1) > size(c, 2))
            W = c;
            H = eye(k);
        else
            W = eye(k);
            H = c;
        end
    end
    
    H = H * norm(W,'fro');
    W = W / norm(W,'fro');
    
    r(i+1)= k;
    core(pos:pos+r(i)*n(i)*r(i+1)-1)= W(:);
    c = H(:);
    pos=pos+r(i)*n(i)*r(i+1);
end
core(pos:pos+r(d)*n(d)*r(d+1)-1)=c(:);
core=core(:);
ps=cumsum([1;n.*r(1:d).*r(2:d+1)]);
t.d=d;
t.n=n;
t.r=r;
t.ps=ps;
t.core=core;

wtime = toc;
%% 
tic
resSVD = tt_tensor(uncompress, eps);
wtimeSVD = toc;
res = reshape(full(t), 1, numel(full(t)));
source = reshape(uncompress, 1, numel(uncompress));
relError1 = norm(res - source)/ norm(source) * 100;
relError2 = norm(resSVD - t) / norm(resSVD) * 100;
resSVDunc = reshape(full(resSVD), 1, numel(full(resSVD)));
relError3 = norm(resSVDunc - source) / norm(source) * 100;

nr = t.r;
nrSVD = resSVD.r;

end