function [F, E,alpha] = LandWeber2(W,P,k)
[m,c] = size(W);    % n��ʾW������
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
    for j = 1:k         %����һ��ѭ��������k��
        E = P - W*f;    %�������
        normr2 = E'*E;  %���������2������ƽ��

        if (normr2 < 1e-12)   %���һ���жϣ�������С��1e-12����ֹͣ������
            break;
        end
        f = f + alpha*V*W'*E;
    end
    %%
F = f;
end


