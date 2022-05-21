clear all; close all; clc

% 0 = no tumor, 1 = glioma, 2 = meningioma, 3 = pituitary
data_label = [0,1,2,3];

targetX = 32;
targetY = targetX; % square image

%% load in all data

glioma_input_tr = dir("Training\glioma_tumor\*.jpg");
meningioma_input_tr = dir("Training\meningioma_tumor\*.jpg");
pituitary_input_tr = dir("Training\pituitary_tumor\*.jpg");
no_input_tr = dir( "Training\no_tumor\*.jpg");

glioma_input_te = dir("Testing\glioma_tumor\*.jpg");
meningioma_input_te = dir("Testing\meningioma_tumor\*.jpg");
pituitary_input_te = dir("Testing\pituitary_tumor\*.jpg");
no_input_te = dir("Testing\no_tumor\*.jpg");

%% concatenate training and test for split

glioma_input_all = cat(1, glioma_input_tr, glioma_input_te);
meningioma_input_all = cat(1, meningioma_input_tr, meningioma_input_te);
pituitary_input_all = cat(1, pituitary_input_tr, pituitary_input_te);
no_input_all = cat(1, no_input_tr, no_input_te);

size(glioma_input_all)
size(meningioma_input_all)
size(pituitary_input_all)
size(no_input_all)

%% remove excess above 900

% top limit to capture data
%limit = 900;

%glioma_input_all = glioma_input_all(1:limit);
%meningioma_input_all = meningioma_input_all(1:limit);
%pituitary_input_all = pituitary_input_all(1:limit);
%no_input_all = no_input_all(1:limit);

%% randomise before splitting set

glioma_rand = glioma_input_all(randperm(length(glioma_input_all)));
meningioma_rand = meningioma_input_all(randperm(length(meningioma_input_all)));
pituitary_rand = pituitary_input_all(randperm(length(pituitary_input_all)));
no_rand = no_input_all(randperm(length(no_input_all)));


%% split into sets

training_glioma = glioma_rand(1:700);
validation_glioma = glioma_rand(701:850);
testing_glioma = glioma_rand(851:926);

training_meningioma = meningioma_rand(1:700);
validation_meningioma = meningioma_rand(701:850);
testing_meningioma = meningioma_rand(851:937);

training_pituitary = pituitary_rand(1:700);
validation_pituitary = pituitary_rand(701:850);
testing_pituitary = pituitary_rand(851:901);

training_no = pituitary_rand(1:400);
validation_no = pituitary_rand(401:450);
testing_no = pituitary_rand(451:500);

%% gather training data

data_training = {training_no, training_glioma, training_meningioma, training_pituitary};

idx_tr = 1;

for label = 1:length(data_training)
    
    files = data_training{label};
    
    for j = 1:length(files)
        
        labels_training(idx_tr) = label - 1;
        images_training(:, :, :, idx_tr) = imresize(imread(strcat(files(j).folder, "/", files(j).name)), [targetX, targetY]);
        final_images_training(:,:, idx_tr) = mat2gray(images_training(:,:,1,idx_tr));
        idx_tr = idx_tr + 1;
    end
end
final_labels_training = labels_training';

%% gather validation data

data_validation = {validation_no, validation_glioma, validation_meningioma, validation_pituitary};

idx_v = 1;

for label = 1:length(data_validation)
    
    files = data_validation{label};
    
    for j = 1:length(files)
        
        labels_validation(idx_v) = label - 1;
        images_validation(:, :, :, idx_v) = imresize(imread(strcat(files(j).folder, "/", files(j).name)), [targetX, targetY]);
        final_images_validation(:,:, idx_v) = mat2gray(images_validation(:,:,1,idx_v));
        idx_v = idx_v + 1;
    end
end
final_labels_validation = labels_validation';


%% gather testing data

data_testing = {testing_no, testing_glioma, testing_meningioma, testing_pituitary};

idx_t = 1;

for label = 1:length(data_testing)
    
    files = data_testing{label};
    
    for j = 1:length(files)
        
        labels_testing(idx_t) = label - 1;
        images_testing(:, :, :, idx_t) = imresize(imread(strcat(files(j).folder, "/", files(j).name)), [targetX, targetY]);
        final_images_testing(:,:, idx_t) = mat2gray(images_testing(:,:,1,idx_t));
        idx_t = idx_t + 1;
    end
end
final_labels_testing = labels_testing';
%% randomise data for output - mix labels

training_randperm = randperm(2500);
validation_randperm = randperm(500);
testing_randperm = randperm(264);

img_train = final_images_training(:, :, training_randperm);
labels_train = final_labels_training(training_randperm);

img_val = final_images_validation(:, :, validation_randperm);
labels_val = final_labels_validation(validation_randperm);

img_test = final_images_testing(:, :, testing_randperm);
labels_test = final_labels_testing(testing_randperm);

%% Saving the data
save("tumor_train_data_"+targetX+".mat", "labels_train", "img_train");
save("tumor_val_data_"+targetX+".mat", "labels_val", "img_val");
save("tumor_test_data_"+targetX+".mat", "labels_test", "img_test");



