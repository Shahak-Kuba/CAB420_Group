clear all; close all; clc

% 0 = no tumor, 1 = glioma, 2 = meningioma, 3 = pituitary
data_label = [0,1,2,3];

targetX = 64;
targetY = targetX; % square image

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
        labels_tr(tr_idx) = label - 1;
        images_training(:, :, :, tr_idx) = imresize(imread(strcat(files(j).folder, "/", files(j).name)), [targetX, targetY]);
        images_train(:,:, tr_idx) = mat2gray(images_training(:,:,1,tr_idx));
        tr_idx = tr_idx + 1;
    end
end

figure;
imshow(images_train(:,:,1))


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
        labels_te(te_idx) = label - 1;
        images_testing(:, :, :, te_idx) = imresize(imread(strcat(files(j).folder, "/", files(j).name)), [targetX, targetY]);
        images_test(:,:,te_idx) = mat2gray(images_testing(:,:,1,te_idx));
        te_idx = te_idx + 1;
    end
end

figure;
imshow(images_test(:,:,1))


%% Saving the data
save("tumor_training.mat", "labels_tr", "images_train");
save("tumor_testing.mat", "labels_te", "images_test");


