load('SA.mat');
for d=1:381
name=strcat('8dc/c',num2str(d),'.xlsx');
[num1,txt1,raw1] = xlsread(name);
L1 = str2double(raw1);
c1 = abs(L1);
OBJ_data = c1;
[num2,txt2,raw2] = xlsread('MemptyA.xlsx');
L2 = str2double(raw2);
c2= abs(L2);
EMP_data=c2;
%% 根据迭代算法，设置三个传递参数，分别是灵敏度矩阵W(56 X 489)，测量值P（56 X 1, 此处为仿真值），迭代次数k
W = -Sen;
P1 = ((OBJ_data - EMP_data)./EMP_data)';
P2 = (EMP_data)';
k =500;
% [F1]=Tikhonov(W,P1);
[F1,E,alpha] = LandWeber2(W,P1,k);
% [F1,E,alpha] = LandWeber_FHF(W,P1,k);%Landweber和ART
% [F1] = ART(W,P1,k);
% [F1,E] = Newton_Raphson_FHF(W,P1,k);   %进行Newton-Raphson算法的迭代，迭代次数较少，100以内
%% 此处，MatrixReshape（）函数为按照489次扫描的顺序写的回填函数
map1 = MatrixReshape(F1);
Z1 = normalize(map1);
map = Z1;
%% griddata
x = [-50:100/24:50];
y = [-50:100/24:50];
xlin = linspace(min(x),max(x),500);
ylin = linspace(min(y),max(y),500);
[X,Y] = ndgrid(xlin,ylin);
Z = griddata(x,y,map,X,Y,'cubic');  %这里map'使得绘图跟实际的成像区域中的物体位置一样
maxz = max(max(Z));
%% 创建一个圆形的图
ydia = length(X);
xdia = length(Y);
yrad = ydia/2;
xrad = xdia/2;
for f = 1:ydia
    for g = 1:xdia
        if (((f - yrad)^2) + ((g - xrad)^2) > (yrad * xrad))
            Z(f,g) = NaN;
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
% set(get(h,'Title'),'string','conductivity');
colormap jet
axis off;
name2=strcat('8dc/rc/rc',num2str(d),'.xlsx');
xlswrite(name2,Z1);
end