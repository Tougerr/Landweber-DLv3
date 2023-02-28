function [F, E,alpha] = LandWeber2(W,P,k)
[m,c] = size(W);    % n表示W的列数
f = zeros(c,1);
S = W'*W;% W':780x28 W:28x780
[t,s]=size(S);
I=eye(t,s);
[x,y]= eig(S);
n=diag(y);
i=length(n);
u1=max(n);
n(n<=1e-5)=inf;
um=min(n);
b=(u1+um)*0.5;
uk=abs(b-u1);
us=uk*(u1+um-uk)+u1*um;
alpha=2/us;
V=(2*b*I-S);
    for j = 1:k         %定义一个循环，迭代k次
        E = P - W*f;    %定义误差
        normr2 = E'*E;  %误差向量的2范数的平方

        if (normr2 < 1e-12)   %添加一个判断，如果误差小于1e-12，就停止迭代了
            break;
        end
        f = f + alpha*V*W'*E;
    end
    %%
F = f;
end


