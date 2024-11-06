function averageLOE = calculateAverageLOE()

    inputFolder = 'D:\Users\16590\Desktop\FLW-Net-main\data\Test\low';
    enhancedFolder = 'D:\Users\16590\Desktop\result-new\Test\low';

    % 获取输入文件夹中的所有图片文件
    inputFiles = dir(fullfile(inputFolder, '*.png'));

    totalLOE = 0; % 用于累积 LOE 值的总和

    for fileIdx = 1:length(inputFiles)
        % 构建文件路径
        inputFilePath = fullfile(inputFolder, inputFiles(fileIdx).name);
        enhancedFilePath = fullfile(enhancedFolder, strrep(inputFiles(fileIdx).name, '.png', '.png'));

        % 计算 LOE
        currentLOE = LOE(inputFilePath, enhancedFilePath);

        % 累积 LOE 值
        totalLOE = totalLOE + currentLOE;
    end

    % 计算平均 LOE
    averageLOE = totalLOE / length(inputFiles);

end

function LOE_value = LOE(inputFilePath, enhancedFilePath)
    % 读取输入图像和增强图像
    ipic = imread(inputFilePath);
    ipic = double(ipic);
    epic = imread(enhancedFilePath);
    epic = double(epic);

    [m, n, ~] = size(ipic);

    % 获取本地最大值
    win = 7;
    imax = round(max(max(ipic(:,:,1), ipic(:,:,2)), ipic(:,:,3)));
    imax = getlocalmax(imax, win);

    emax = round(max(max(epic(:,:,1), epic(:,:,2)), epic(:,:,3)));
    emax = getlocalmax(emax, win);

    % 获取下采样图像
    blkwin = 50;
    mind = min(m, n);
    step = floor(mind / blkwin);
    blkm = floor(m / step);
    blkn = floor(n / step);
    ipic_ds = zeros(blkm, blkn);
    epic_ds = zeros(blkm, blkn);
    LOE_matrix = zeros(blkm, blkn);

    for i = 1:blkm
        for j = 1:blkn
            ipic_ds(i,j) = imax(i * step, j * step);
            epic_ds(i,j) = emax(i * step, j * step);
        end
    end

    for i = 1:blkm
        for j = 1:blkn
            flag1 = ipic_ds >= ipic_ds(i,j);
            flag2 = epic_ds >= epic_ds(i,j);
            flag = (flag1 ~= flag2);
            LOE_matrix(i,j) = sum(flag(:));
        end
    end

    LOE_value = mean(LOE_matrix(:));
end

% 其余两个辅助函数保持不变
    function output=getlocalmax(pic,win)
        [m,n]=size(pic);
        extpic=getextpic(pic,win);
        output=zeros(m,n);
        for i=1+win:m+win
            for j=1+win:n+win
                modual=extpic(i-win:i+win,j-win:j+win);
                output(i-win,j-win)=max(modual(:));
            end
        end
    end
    
    function output=getextpic(im,win_size)
        [h,w,c]=size(im);
        extpic=zeros(h+2*win_size,w+2*win_size,c);
        extpic(win_size+1:win_size+h,win_size+1:win_size+w,:)=im;
        for i=1:win_size%extense row
            extpic(win_size+1-i,win_size+1:win_size+w,:)=extpic(win_size+1+i,win_size+1:win_size+w,:);%top edge
            extpic(h+win_size+i,win_size+1:win_size+w,:)=extpic(h+win_size-i,win_size+1:win_size+w,:);%botom edge
        end
        for i=1:win_size%extense column
            extpic(:,win_size+1-i,:)=extpic(:,win_size+1+i,:);%left edge
            extpic(:,win_size+w+i,:)=extpic(:,win_size+w-i,:);%right edge
        end
        output=extpic;
    end
