import os
# disable tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import random
# for consistency
random.seed(4)

# Tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorboard import notebook
from tensorflow.keras import layers
from tensorflow.keras import backend as K

# Sklearn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
# function to compute class weights
from sklearn.utils import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn import decomposition
from sklearn import discriminant_analysis
from sklearn import datasets
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

#Matplotlib
import matplotlib.pyplot as plt

# Data management libraries
import scipy.io as scio
import numpy as np
import numpy
# Visualising Packages
import visualkeras as vk
from PIL import ImageFont
from collections import defaultdict


######### External Variables ###############

cmp = defaultdict(dict)
cmp[layers.Conv2D]['fill'] = 'orange'
cmp[layers.MaxPooling2D]['filal'] = 'red'
cmp[layers.Dense]['fill'] = 'black'
cmp[layers.Flatten]['fill'] = 'teal'
cmp[layers.Dropout]['fill'] = 'purple'


######### General Functions ################

def load_data(train_dir, val_dir, test_dir):

    data_train = scio.loadmat(train_dir)
    data_val = scio.loadmat(val_dir)
    data_test = scio.loadmat(test_dir)
    
    return data_train, data_val, data_test


## PCA Functions

def plot_cumulative_sum(cs, top90, top95, top99):
    fig = plt.figure(figsize=[20, 8])
    plt.plot(cs)
    plt.title('PCA components to recreate data')
    plt.xlabel('Num components')
    plt.ylabel('Accuracy % explained ')
    plt.scatter(top90, cs[top90], c='#ff0000')
    plt.scatter(top95, cs[top95], c='#9e01e3')
    plt.scatter(top99, cs[top99], c='#0f1234')
    plt.legend(["Num Components vs Accuracy", "90% recovered", "95% recovered", "99% recovered"])
    print('top90: ' + str(top90) + ',', 'top95: ' + str(top95) + ',', 'top99: ' + str(top99))

def search_hyperparams(model, params, X_train, Y_train, X_validate, Y_validate):
    # Create list of all possible combinations
    param_list = list(ParameterGrid(params))
    
    # Initialising 
    best_result = 0.00;
    best_params = param_list[0];
    worst_result = 1.00;
    worst_params = param_list[0];
    # looping through all parameters in parameter list
    for params in param_list:
        # creating model with set parameters
        model = model.set_params(**params)
        # training the model
        model.fit(X_train, Y_train)
        # retrieving model score
        result = model.score(X_validate, Y_validate)
        # checking if model score is better, then allocating best parameters
        if result > best_result:
            best_result = result
            best_params = params
        if result < worst_result:
            worst_result = result
            worst_params = params

    
    # Return the best
    print(best_params)
    print("Validation Accuracy " + str(best_result))
    print(worst_params)
    print("Validation Accuracy " + str(worst_result))
    return best_params

def eval_model_pca(model, X_train, Y_train, X_test, Y_test):
    labels = ['No Tumour', 'Glioma', 'Meningioma', 'Pituitary']
    fig = plt.figure(figsize=[25, 8])
    ax = fig.add_subplot(1, 2, 1)
    conf = ConfusionMatrixDisplay.from_estimator(model, X_train, Y_train, normalize='true', ax=ax, display_labels=labels)
    #conf.ax_.set_title('Training Set Performance: %1.3f' % (sum(model.predict(X_train) == Y_train)/len(Y_train)));
    ax = fig.add_subplot(1, 2, 2)
    conf = ConfusionMatrixDisplay.from_estimator(model, X_test, Y_test, normalize='true', ax=ax, display_labels=labels)
    #conf.ax_.set_title('Testing Set Performance: %1.3f' % (sum(model.predict(X_test) == Y_test)/len(Y_test)));
    print(classification_report(Y_test, model.predict(X_test)))

############# CMC PLOT FUNCTIONS #######################
def get_ranked_histogram_l1_distance(gallery_feat, gallery_Y, probe_feat, probe_Y, verbose = False):
    
    # storage for ranked histogram
    # length equal to number of unique subjects in the gallery
    ranked_histogram = np.zeros(len(np.unique(gallery_Y)))

    # loop over all samples in the probe set
    for i in range(len(probe_Y)):
        # get the true ID of this sample
        true_ID = probe_Y[i]
        if verbose:
            print('Searching for ID %d' % (true_ID))

        # get the distance between the current probe and the whole gallery, L1 distance here. Note that L1
        # may not always be the best choice, so consider your distance metric given your problem
        dist = np.linalg.norm(gallery_feat - probe_feat[i,:], axis=1, ord=1)
        if verbose:
            print(dist)

        # get the sorted order of the distances
        a = np.argsort(dist)
        # apply the order to the gallery IDs, such that the first ID in the list is the closest, the second
        # ID is the second closest, and so on
        ranked = gallery_Y[a]
        if verbose:
            print('Ranked IDs for query:')
            print(a)

        # find the location of the True Match in the ranked list
        ranked_result = np.where(ranked == true_ID)[0][0]
        if verbose:
            print(ranked_result)

        # store the ranking result in the histogram
        ranked_histogram[ranked_result] += 1
        if verbose:
            print('')
    
    if verbose:
        print(ranked_histogram)
    
    return ranked_histogram

def ranked_hist_to_CMC(ranked_hist):

    cmc = np.zeros(len(ranked_hist))
    for i in range(len(ranked_hist)):
        cmc[i] = np.sum(ranked_hist[:(i + 1)])
    
    return (cmc / len(ranked_hist))

def plot_cmc(cmc):
    fig = plt.figure(figsize=[10, 8])
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range(1, len(cmc)+1), cmc)
    ax.set_xlabel('Rank')
    ax.set_ylabel('Count')
    ax.set_ylim([0, 1.0])
    ax.set_title('CMC Curve')    