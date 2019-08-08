function out = vl_nntanh(x,dzdy)


y =  (exp(x)- exp(-x))./ ( exp(x) + exp(-x));

if nargin <= 1 || isempty(dzdy)
  out = y ;
else
  out = dzdy .* (1 - y.^2) ;  %should add .
end