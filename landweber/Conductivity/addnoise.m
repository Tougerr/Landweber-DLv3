clc;
clear;
for d=1
name=strcat('F:\8线圈仿真\matlab成像程序_数据集\数据集（磁导率）\250_250_landweber\rc-8du\rc',num2str(d),'.xlsx');
B = xlsread(name);
% [num1,txt1,raw1] = xlsread(name);
% B1 = str2double(raw1);
% B=imag(B1);
map1 = awgn(B,60,'measured');
name1=strcat('F:\8线圈仿真\matlab成像程序_数据集\数据集（磁导率）\250_250_landweber\rc-noise-8du\60/c',num2str(d),'.xlsx');
xlswrite(name1,map1);
map2 = awgn(B,50,'measured');
name2=strcat('F:\8线圈仿真\matlab成像程序_数据集\数据集（磁导率）\250_250_landweber\rc-noise-8du\50/c',num2str(d),'.xlsx');
xlswrite(name2,map2);
map3 = awgn(B,40,'measured');
name3=strcat('F:\8线圈仿真\matlab成像程序_数据集\数据集（磁导率）\250_250_landweber\rc-noise-8du\40/c',num2str(d),'.xlsx');
xlswrite(name3,map3);
map4 = awgn(B,30,'measured');
name4=strcat('F:\8线圈仿真\matlab成像程序_数据集\数据集（磁导率）\250_250_landweber\rc-noise-8du\30/c',num2str(d),'.xlsx');
xlswrite(name4,map4);
map5 = awgn(B,20,'measured');
name5=strcat('F:\8线圈仿真\matlab成像程序_数据集\数据集（磁导率）\250_250_landweber\rc-noise-8du\20/c',num2str(d),'.xlsx');
xlswrite(name5,map5);
end