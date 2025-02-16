function [ev]=tt_powerm(cur_tt, t0, max_iter, eps)
% Shcherbakova, E., Tyrtyshnikov E. 
% Fast Nonnegative Tensor Factorizations with Tensor Train Model // 
% Lobachevskii Journal of Mathematics. -- 2022. -- V. 43, ? 4. -- P. 882?894.
ev = t0;
 
for i = 1:max_iter
    ev = cur_tt .* ev;
    ev = round(ev, eps, 5);
    ev_nrm = norm(ev);
    if ev_nrm ~= 0
       ev = ev / ev_nrm;
    else
        ev = ev * 0.0;
    end
end