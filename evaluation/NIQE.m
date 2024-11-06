clc
clear
% 设置文件夹路径
folderPath = 'D:\Users\16590\Desktop\FLW-Net-main\result-new\Test\low';

% 获取文件夹中所有图片文件
imageFiles = dir(fullfile(folderPath, '*.png')); % 这里假设你的图片是PNG格式，你可以根据需要修改

% 初始化存储NIQE值的数组
niqeValues = zeros(1, numel(imageFiles));

% 循环处理每张图片
for i = 1:numel(imageFiles)
    % 读取图片
    imagePath = fullfile(folderPath, imageFiles(i).name);
    img = imread(imagePath);
    
    % 计算NIQE值
    niqeValues(i) = niqe(img);
end

% 计算平均NIQE值
averageNIQE = mean(niqeValues);

% 显示结果
disp(['平均NIQE值: ', num2str(averageNIQE)]);
disp('每张图片的NIQE值:');
disp(niqeValues);