clc;
clear;
load('S.mat');
for d=1:381
name=strcat('8du/c',num2str(d),'.xlsx');
B1 = xlsread(name);
B_empty = xlsread('BemptyA.xlsx'); 
P1 = ((B1 -B_empty)./B_empty)';
%% 首先，对于测量磁感应强度的情况来说，需要的物场数据是P1，灵敏度矩阵是Sen
k = 200;
W = Sen;
% [F, E] = Newton_Raphson_FHF(W,P1,k);%u为k=50
% [F] =Tikhonov(W,P1);
[F,E,alpha] = LandWeber2(W,P1,k);%k=200
% [F]=ART(W,P1,k);
%% 此处，MatrixReshape（）函数为按照489次扫描的顺序写的回填函数
map1 = MatrixReshape(F);
map = normalize(map1);
%% griddata
x = -50:100/24:50;
y = -50:100/24:50;
xlin = linspace(min(x),max(x),500);
ylin = linspace(min(y),max(y),500);
[X,Y] = ndgrid(xlin,ylin);
Z = griddata(x,y,map,X,Y,'cubic');  %这里map'使得绘图跟实际的成像区域中的物体位置一样
maxz = max(max(Z));
minz = min(min(Z));
%% 创建一个圆形的图
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
%% 绘图设置
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