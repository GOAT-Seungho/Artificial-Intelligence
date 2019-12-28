import os
import sys
import math
import time
import struct
import cupy as cp
import numpy as np
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin

# BinarayClassifier
class BinaryClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, batch_size=16, max_iter=50, learning_rate=0.01, random_state=1, C=100):
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.C = C
        self.rgen = np.random.RandomState(self.random_state)
        
    def fit(self, X, y):
        # Exception Handling
        if self.C < 0:
            raise ValueError("The C value of %r must be positive" % self.C)
        if ((self.learning_rate < 0) or (self.learning_rate > 1)):
            raise ValueError("The learning_rate value of %r is invalid." % self.learning_rate,
                             "Set the learning_rate value between 0.0 and 1.0.")        
            
        n_batches = math.ceil(len(X) / self.batch_size)
        self.rest_batch_size = X.shape[0] - (n_batches-1) * self.batch_size
        
        self.w_ = cp.array(self.rgen.normal(loc=0.0, scale=0.01, size=X.shape[1]))
        self.b_ = 0.
        
        for epoch in range(self.max_iter):
            X, y = self.shuffle(X, y)
            
            [self.calculateGradientAndUpdate(X, y, j, rest_exist = False) for j in range(n_batches - 1)]
            self.calculateGradientAndUpdate(X, y, n_batches - 1, rest_exist = True)
            
        return self
    
    def hypothesis(self, X):
        return cp.dot(cp.array(X), self.w_) + self.b_
    
    def shuffle(self, X, y):
        shuffle_index = np.arange(X.shape[0])
        np.random.shuffle(shuffle_index)
        return X[shuffle_index], y[shuffle_index]

    def set_batch(self, X, y, n_batch, rest_exist):
        if (rest_exist == True):
            X_mini = X[n_batch*self.batch_size : n_batch*self.batch_size + self.rest_batch_size]
            y_mini = y[n_batch*self.batch_size : n_batch*self.batch_size + self.rest_batch_size]
            batch_size = self.rest_batch_size
        else:
            X_mini = X[n_batch*self.batch_size : (n_batch+1)*self.batch_size]
            y_mini = y[n_batch*self.batch_size : (n_batch+1)*self.batch_size]
            batch_size = self.batch_size
        
        return cp.array(X_mini), cp.array(y_mini), batch_size
    
    def calculateGradientAndUpdate(self, X, y, n_batch, rest_exist):
        X_mini, y_mini, batch_size = self.set_batch(X, y, n_batch, rest_exist)

        grad_w = cp.zeros(X.shape[1])
        grad_b = 0
        mask = cp.less_equal(cp.multiply(y_mini, self.hypothesis(X_mini)), 1)
        
        Xy = cp.multiply(X_mini.T, y_mini)
        masked_Xy = cp.multiply(Xy, mask)
        grad_w = cp.sum(-masked_Xy, axis=1)
        grad_w /= batch_size
        grad_w += self.w_/self.C
        self.w_ -= self.learning_rate * grad_w
        
        masked_y = cp.multiply(y_mini, mask)
        grad_b = cp.sum(-masked_y, axis=0)
        grad_b = grad_b / batch_size
        self.b_ -= self.learning_rate * grad_b
        return grad_w, grad_b

# Multiclass Classifier
class MulticlassClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, batch_size=16, max_iter=50, learning_rate=0.01, random_state=1, C=100):
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.C = C
        
    def fit(self, X, y):
        self.labels = np.unique(y) # 0 ~ 9
        # self.labels = cp.unique(y) # 0 ~ 9
        self.outputs_ = []
        for label in range(len(self.labels)):
            y_binary = np.where(np.array(y==label),1,-1)
            b_c = BinaryClassifier(self.batch_size, self.max_iter, self.learning_rate, self.random_state, self.C)
            b_c.fit(X, y_binary)
            self.outputs_.append(b_c)
        return self
        
    def predict(self, X):
        prediction = []
        for o in self.outputs_:
            prediction.append(o.hypothesis(X))
        return self.labels[np.argmax(prediction, axis=0)]

# Read Functions
def read(images, labels):
    with open(labels, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

def read_no_label(images):
    with open(images, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(60000, 784)
    return images

# MNIST Data Read
X_train, y_train = read(sys.argv[1], sys.argv[2])
X_test = read_no_label(sys.argv[3])

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Preprocessing
## Convolution
def zero_padding(img, n=1):

    m = img.shape[0]
    w = img.shape[1]
    h = img.shape[2]
    
    padded_img = np.ones((m, w + 2 * n, h + 2 * n))
    
    padded_img[:, n : padded_img.shape[1] - n, n : padded_img.shape[2] - n] = img
    
    return padded_img
    
def horizontal_filter(img):
    h_filter = np.array([
        [ 0, 0, 0],
        [ 0, 1, 0],
        [-1, 0, 0]
    ])
    h_filter2 = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ])
    h_filter.reshape(1,3,3)
    h_filter2.reshape(1,3,3)
    m = img.shape[0]
    w = img.shape[1]
    h = img.shape[2]
    horizontal_grad = np.zeros((m, w - 2, h - 2))
    horizontal_grad2 = np.zeros((m, w - 2, h - 2))

    for i in range(1, w - 1):
        for j in range(1, h - 1):
            images_slice = img[:, i - 1 : i + 2, j - 1 : j + 2]
            horizontal_grad[:, i - 1, j - 1] = np.sum(np.multiply(images_slice, h_filter), axis=(1, 2))
            horizontal_grad2[:, i - 1, j - 1] = np.sum(np.multiply(images_slice, h_filter2), axis=(1, 2))
            
    return horizontal_grad, horizontal_grad2
    
def MaxPooling(img):

    m = img.shape[0]
    w = img.shape[1]
    h = img.shape[2]
    
    img = zero_padding(img, 1)
    img2, img3 = horizontal_filter(img)
    
    pooling_grad = np.zeros((m, w//2, h//2))
    pooling_grad2 = np.zeros((m, w//2, h//2))

    for i in range(0, w//2):
        for j in range(0, h//2):
            pooling_grad[: , i , j ] = np.max(img2[: , 2*i : 2*i + 2, 2*j : 2*j + 2])
            pooling_grad2[: , i , j ] = np.max(img3[: , 2*i : 2*i + 2, 2*j : 2*j + 2])

    pooling_grad = pooling_grad.reshape(m,-1)
    pooling_grad2 = pooling_grad.reshape(m,-1)    
            
    return pooling_grad, pooling_grad2

X_train_reshaped = X_train.reshape(X_train.shape[0], 28, 28)
X_test_reshaped = X_test.reshape(X_test.shape[0], 28, 28)

X_train_conv1, X_train_conv2 = MaxPooling(X_train_reshaped)
X_test_conv1, X_test_conv2 = MaxPooling(X_test_reshaped)

## PCA
pca = PCA(n_components=50)
X_train_pca = pca.fit_transform(X_train) 
X_test_pca = pca.transform(X_test)

## Polynomial
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=True, order='F')
X_train_pca_poly = poly.fit_transform(X_train_pca)
X_test_pca_poly = poly.transform(X_test_pca)

# Merge
X_train_pca_poly_conv12 = np.hstack((X_train_pca_poly,X_train_conv1,X_train_conv2))
X_test_pca_poly_conv12 = np.hstack((X_test_pca_poly, X_test_conv1, X_test_conv2))

# Check shapes of data
print("X_train : ", X_train.shape)
print("X_train_conv1 : ", X_train_conv1.shape)
print("X_train_conv2 : ", X_train_conv2.shape)
print("X_train_pca : ", X_train_pca.shape)
print("X_train_pca_poly : ", X_train_pca_poly.shape)
print("최종 데이터 shape : ", X_train_pca_poly_conv12.shape)

# Training & Testing
MC=MulticlassClassifier(max_iter=10, batch_size=256, learning_rate=0.01, C=1000)


print("Start Time : ", datetime.now())
start = time.time()

MC.fit(X_train_pca_poly_conv12, y_train)
y_pred = MC.predict(X_test_pca_poly_conv12)
# score = accuracy_score(y_test, y_pred)

print("Learning Time : ", (int)(time.time() - start), "초")
print("End Time : ", datetime.now())
# print(score)



# Making prediction.txt
f = open("./prediction.txt", 'w')
for i in range(len(y_pred)):
    f.write('%d|\n' % y_pred[i])
f.close()










# Grid Search CV
# param_grid = [{
#     'C' : [10, 100, 1000, 10000],
#     'learning_rate' : [0.1, 0.01, 0.001, 0.0001],
#     'batch_size' : [32, 64, 128, 256, 512],
#     'max_iter' : [5]
# }]

# grid_search = GridSearchCV(MulticlassClassifier(), 
#                             param_grid=param_grid, 
#                             cv=5, scoring='accuracy',
#                             verbose=1, n_jobs=-1)

# grid_search.fit(X_train_pca_poly_conv12, y_train)
# print(grid_search.best_params_)
# print(grid_search.best_score_)