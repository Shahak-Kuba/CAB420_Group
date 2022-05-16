import os
# disable tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

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

#Matplotlib
import matplotlib.pyplot as plt

# Data management libraries
import scipy.io as scio
import numpy as np

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