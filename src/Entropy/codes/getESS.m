function [ Ess ] = getESS( w_unnormlized )

w = w_unnormlized / sum(w_unnormlized);
Ess = sum(1./sum(w.^2));

end

