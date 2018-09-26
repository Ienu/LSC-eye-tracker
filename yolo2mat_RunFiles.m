clear;clc;
foldersName = importdata('foldersName.txt');
[r,c]=size(foldersName);
for k= 1 : r
   foldersName = importdata('foldersName.txt');
   
   cd (foldersName{k});
   disp(pwd);
   
   %run('yolo2mat.m');
   %==========================================
   
    face = importdata('face.po');
    gaze = importdata('gazeData.txt');
    name = importdata('name');
    dir = './jpg/';
    width = 1920;
    height = 1080;
    faceData = zeros(50000, 200, 200, 3);
    eyeTrackData = zeros(50000, 4);
    for i = 0 : 49999
        fileName = sprintf('%s%d.jpg', dir, i);
        disp(fileName);
        imageSrc = imread(fileName);

        xc = face(i + 1, 2) * width;
        yc = face(i + 1, 3) * height;
        hfw = floor(face(i + 1, 4) * width / 2);
        hfh = floor(face(i + 1, 5) * height / 2);

        faceSrc = imageSrc(yc + hfh - 2 * hfw : yc + hfh, xc - hfw : xc + hfw, :);
        image = imresize(faceSrc, [200 200]);
        faceData(i + 1, :, :, :) = image;
        eyeTrackData(i + 1, :) = gaze(i + 1, 2 : 5);

        %imshow(image);
        %pause(0.01);
    end
    %toc;
    file = sprintf('data_yolo_%s.mat',name{1});
    save(file, 'eyeTrackData', 'faceData', '-v7.3');
    %toc;
   
   %==========================================
   cd ..;   
end
