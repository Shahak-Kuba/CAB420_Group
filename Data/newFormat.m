clear all; close all; clc

% 0 = no tumor, 1 = glioma, 2 = meningioma, 3 = pituitary
data_label = [0,1,2,3];

targetX = 256;
targetY = targetX; % square image

%% load in all data


glioma_input_tr = readImgs('Training\glioma_tumor\', targetX);
meningioma_input_tr = readImgs('Training\meningioma_tumor\', targetX);
pituitary_input_tr = readImgs('Training\pituitary_tumor\', targetX);
no_input_tr = readImgs('Training\no_tumor\', targetX);

glioma_input_te = readImgs('Testing\glioma_tumor\', targetX);
meningioma_input_te = readImgs('Testing\meningioma_tumor\', targetX);
pituitary_input_te = readImgs('Testing\pituitary_tumor\', targetX);
no_input_te = readImgs('Testing\no_tumor\', targetX);


%% concatenating

glioma_images_all = cat(3, glioma_input_te, glioma_input_tr); %902
meningioma_images_all = cat(3, meningioma_input_te, meningioma_input_tr); %927
pituitary_images_all = cat(3, pituitary_input_te, pituitary_input_tr); %846
no_images_all = cat(3, no_input_te, no_input_tr); %408

%% randomise for split

initial_no_randperm = randperm(408);
initial_no_rand = no_images_all(:, :, initial_no_randperm);

initial_glioma_randperm = randperm(902);
initial_glioma_rand = glioma_images_all(:, :, initial_glioma_randperm);

initial_meningioma_randperm = randperm(927);
initial_meningioma_rand = meningioma_images_all(:, :, initial_meningioma_randperm);

initial_pituitary_randperm = randperm(846);
initial_pituitary_rand = pituitary_images_all(:, :, initial_pituitary_randperm);


%% Training, val and test split

training_no = initial_no_rand(:,:,1:350);
validation_no = initial_no_rand(:,:,351:400);
testing_no = initial_no_rand(:,:,401:408);

training_glioma = initial_glioma_rand(:,:,1:700);
validation_glioma = initial_glioma_rand(:,:,701:800);
testing_glioma = initial_glioma_rand(:,:,801:902);

training_meningioma = initial_meningioma_rand(:,:,1:700);
validation_meningioma = initial_meningioma_rand(:,:,701:800);
testing_meningioma = initial_meningioma_rand(:,:,801:927);

training_pituitary = initial_pituitary_rand(:,:,1:700);
validation_pituitary = initial_pituitary_rand(:,:,701:800);
testing_pituitary = initial_pituitary_rand(:,:,801:846);


%% concatenate all

training_temp1 = cat(3, training_no, training_glioma); 
training_temp2 = cat(3, training_meningioma, training_pituitary);
training_all = cat(3, training_temp1, training_temp2);

validation_temp1 = cat(3, validation_no, validation_glioma); 
validation_temp2 = cat(3, validation_meningioma, validation_pituitary);
validation_all = cat(3, validation_temp1, validation_temp2);

testing_temp1 = cat(3, testing_no, testing_glioma); 
testing_temp2 = cat(3, testing_meningioma, testing_pituitary);
testing_all = cat(3, testing_temp1, testing_temp2);

%% Labels

no_lbls_training = 0.* ones(350,1);
glioma_lbls_training = 1.* ones(700,1);
meningioma_lbls_training = 2.* ones(700,1);
pituitary_lbls_training = 3.* ones(700,1);
lbls_training = cat(1, cat(1, no_lbls_training, glioma_lbls_training), cat(1,meningioma_lbls_training,pituitary_lbls_training));

no_lbls_validation = 0.* ones(50,1);
glioma_lbls_validation = 1.* ones(100,1);
meningioma_lbls_validation = 2.* ones(100,1);
pituitary_lbls_validation = 3.* ones(100,1);
lbls_validation = cat(1, cat(1, no_lbls_validation, glioma_lbls_validation), cat(1,meningioma_lbls_validation,pituitary_lbls_validation));

no_lbls_testing = 0.* ones(8,1);
glioma_lbls_testing = 1.* ones(102,1);
meningioma_lbls_testing = 2.* ones(127,1);
pituitary_lbls_testing = 3.* ones(46,1);
lbls_testing = cat(1, cat(1, no_lbls_testing, glioma_lbls_testing), cat(1,meningioma_lbls_testing,pituitary_lbls_testing));

%% randomise data for output - mix labels

training_randperm = randperm(2450);
validation_randperm = randperm(350);
testing_randperm = randperm(283);

img_train = training_all(:, :, training_randperm);
labels_train = lbls_training(training_randperm);

img_val = validation_all(:, :, validation_randperm);
labels_val = lbls_validation(validation_randperm);

img_test = testing_all(:, :, testing_randperm);
labels_test = lbls_testing(testing_randperm);

%% Saving the data
save("tumor_train_data_"+targetX+".mat", "labels_train", "img_train");
save("tumor_val_data_"+targetX+".mat", "labels_val", "img_val");
save("tumor_test_data_"+targetX+".mat", "labels_test", "img_test");

%% functions

function allImgs = readImgs(directory, targetSize)

    targetX = targetSize;
    targetY = targetSize;
    imagefiles = dir(strcat(directory,'*.jpg'));      
    nfiles = length(imagefiles);    % Number of files found
    
    for i = 1:nfiles
        currentfilename = imagefiles(i).name;
        tempdir = strcat(directory, currentfilename);
        images_training(:, :, :, i) = imresize(imread(tempdir), [targetX, targetY]);
        tempImgs(:,:, i) = mat2gray(images_training(:,:,1,i));
    end
    
    sizeImgs = size(tempImgs);
    
    theRows = reshape(tempImgs,[],sizeImgs(3))';
    
    uniqueImgs = unique(theRows,'stable','rows');
    
    allImgs = reshape(uniqueImgs' ,sizeImgs(1),sizeImgs(2),[]);
end