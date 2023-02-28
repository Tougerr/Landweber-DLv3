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
%% ���ݵ����㷨�������������ݲ������ֱ��������Ⱦ���W(56 X 489)������ֵP��56 X 1, �˴�Ϊ����ֵ������������k
W = -Sen;
P1 = ((OBJ_data - EMP_data)./EMP_data)';
P2 = (EMP_data)';
k =500;
% [F1]=Tikhonov(W,P1);
[F1,E,alpha] = LandWeber2(W,P1,k);
% [F1,E,alpha] = LandWeber_FHF(W,P1,k);%Landweber��ART
% [F1] = ART(W,P1,k);
% [F1,E] = Newton_Raphson_FHF(W,P1,k);   %����Newton-Raphson�㷨�ĵ����������������٣�100����
%% �˴���MatrixReshape��������Ϊ����489��ɨ���˳��д�Ļ����
map1 = MatrixReshape(F1);
Z1 = normalize(map1);
map = Z1;
%% griddata
x = [-50:100/24:50];
y = [-50:100/24:50];
xlin = linspace(min(x),max(x),500);
ylin = linspace(min(y),max(y),500);
[X,Y] = ndgrid(xlin,ylin);
Z = griddata(x,y,map,X,Y,'cubic');  %����map'ʹ�û�ͼ��ʵ�ʵĳ��������е�����λ��һ��
maxz = max(max(Z));
%% ����һ��Բ�ε�ͼ
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
%% ��ͼ����
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