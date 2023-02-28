function[F]=Tikhonov(W,P)
[m,n]=size(W);
B=W'*W+0.01*eye(n);
D=W'*P;
F=B\D;