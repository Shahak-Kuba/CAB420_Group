clear all; close all; clc

% 0 = no tumor, 1 = glioma, 2 = meningioma, 3 = pituitary
data_label = [0,1,2,3];

targetX = 64
targetY = 64

%% for training data
glioma_images_tr = dir("Training\glioma_tumor\*.jpg");
meningioma_images_tr = dir("Training\meningioma_tumor\*.jpg");
pituitary_images_tr = dir("Training\pituitary_tumor\*.jpg");
no_images_tr = dir( "Training\no_tumor\*.jpg");

size(glioma_images_tr)

all_tr = {no_images_tr, glioma_images_tr, meningioma_images_tr, pituitary_images_tr};

tr_idx = 1;

for i = 1:length(all_tr)
    label = i;
    files = all_tr{i};
    for j = 1:length(files)
        labels_tr(tr_idx) = label;
        images_tr(:, :, :, tr_idx) = imresize(imread(strcat(files(j).folder, "/", files(j).name)), [targetX, targetY]);
        tr_idx = tr_idx + 1;
    end
end

figure;
imshow(images_tr(:,:,:,1))


%% for testing data
glioma_images_te = dir("Testing\glioma_tumor\*.jpg");
meningioma_images_te = dir("Testing\meningioma_tumor\*.jpg");
pituitary_images_te = dir("Testing\pituitary_tumor\*.jpg");
no_images_te = dir("Testing\no_tumor\*.jpg");

all_te = {no_images_te, glioma_images_te, meningioma_images_te, pituitary_images_te};

te_idx = 1;

for i = 1:length(all_te)
    label = i;
    files = all_te{i};
    for j = 1:length(files)
        labels_te(te_idx) = label;
        images_te(:, :, :, te_idx) = imresize(imread(strcat(files(j).folder, "/", files(j).name)), [targetX, targetY]);
        te_idx = te_idx + 1;
    end
end

figure;
imshow(images_te(:,:,:,1))


%%
save("data.mat", "labels_tr", "images_tr", "labels_te", "images_te");


