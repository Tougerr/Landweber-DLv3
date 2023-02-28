clc;
clear;
load('S.mat');
for d=1:381
name=strcat('8du/c',num2str(d),'.xlsx');
B1 = xlsread(name);
B_empty = xlsread('BemptyA.xlsx'); 
P1 = ((B1 -B_empty)./B_empty)';
%% ���ȣ����ڲ����Ÿ�Ӧǿ�ȵ������˵����Ҫ���ﳡ������P1�������Ⱦ�����Sen
k = 200;
W = Sen;
% [F, E] = Newton_Raphson_FHF(W,P1,k);%uΪk=50
% [F] =Tikhonov(W,P1);
[F,E,alpha] = LandWeber2(W,P1,k);%k=200
% [F]=ART(W,P1,k);
%% �˴���MatrixReshape��������Ϊ����489��ɨ���˳��д�Ļ����
map1 = MatrixReshape(F);
map = normalize(map1);
%% griddata
x = -50:100/24:50;
y = -50:100/24:50;
xlin = linspace(min(x),max(x),500);
ylin = linspace(min(y),max(y),500);
[X,Y] = ndgrid(xlin,ylin);
Z = griddata(x,y,map,X,Y,'cubic');  %����map'ʹ�û�ͼ��ʵ�ʵĳ��������е�����λ��һ��
maxz = max(max(Z));
minz = min(min(Z));
%% ����һ��Բ�ε�ͼ
ydia = length(X);
xdia = length(Y);
yrad = ydia/2;
xrad = xdia/2;
for f = 1:ydia
    for g = 1:xdia
       if Z(f,g) < 0.1
           Z(f,g) = 0;
        end
    end
end
for f = 1:ydia
    for g = 1:xdia
        if (((f - yrad)^2) + ((g - xrad)^2) > (yrad * xrad))
            Z(f,g) =NaN;
        end
    end
end
%% ��ͼ����
surf(X,-Y,Z)
axis tight
view(0,90);
axis square
shading flat
h=colorbar;
% set(get(h,'Title'),'string','permeativity');
colormap jet
% caxis([0.1 0.95])
axis off;
name2=strcat('8du/rc/rc',num2str(d),'.xlsx');
xlswrite(name2,Z);
end