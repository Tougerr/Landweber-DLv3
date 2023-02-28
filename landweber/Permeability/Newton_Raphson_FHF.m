function [F,E] = Newton_Raphson_FHF(W,P,k)

    [m,n]=size(W);
    f=zeros(n,1);
    alpha = 0.1;
    I =  eye(n);
for j=1:k
    E= W*f - P;
    normr2= E'*E;
    
    if (normr2<1e-12)
      break;  
    end
    
    f = f - inv(( W' * W + alpha * I)) * W' * (W * f - P);
end

    F=f;
end