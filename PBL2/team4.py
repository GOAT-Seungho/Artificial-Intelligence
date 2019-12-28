import os
import sys
import math
import time
import struct
import itertools
import numpy as np
import matplotlib.pyplot as pyplot
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV


# BinaryClassifier
class BinaryClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, batch_size=32, max_iter=100, learning_rate=0.1, random_state=1, C=100):
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.C = C
        self.rgen = np.random.RandomState(self.random_state)
        
    def fit(self, X, y):
        
        # 예외 처리
        if self.C < 0:
            raise ValueError("The C value of %r must be positive" % self.C)
        if ((self.learning_rate < 0) or (self.learning_rate > 1)):
            raise ValueError("The learning_rate value of %r is invalid. Set the learning_rate value between 0.0 and 1.0." % self.learning_rate)
            
        # 배치 개수 설정 : 데이터의 총 개수 / 배치 사이즈
        n_batches = math.ceil(len(X) / self.batch_size)
        n_rest = X.shape[0] - (n_batches-1) * self.batch_size # 80000 - (313-1)*256 = 128
        
        # w, b 값 초기화
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = 0.
        
        for epoch in range(self.max_iter):
            
            # 매 에포크마다 데이터 셔플
            X, y = self.shuffle(X, y)
            
            for j in range(n_batches - 1):
                # 입력한 배치 사이즈만큼 배치를 설정해준다. (defalut : 32 이므로 0~31, 32~63, ...)
                X_mini = X[j*self.batch_size : (j+1)*self.batch_size]
                y_mini = y[j*self.batch_size : (j+1)*self.batch_size]
                
                F_prime_w_i_sum = np.zeros(X.shape[1])
                F_prime_b_i_sum = 0.
                
                # 각 배치의 사이즈 만큼 (defalut : 32)
                for i in range(self.batch_size):
                    if ( y_mini[i] * self.hypothesis(X_mini[i]) < 1):
                        F_prime_w_i_sum += (-1)*y_mini[i]*X_mini[i]
                        F_prime_b_i_sum += (-1)*y_mini[i]

                # 가중치, 편향 업데이트
                self.w_ -= self.learning_rate * ((1 / self.batch_size) * F_prime_w_i_sum + (1/self.C)*self.w_)
                self.b_ -= self.learning_rate * ((1 / self.batch_size) * F_prime_b_i_sum)
            
            # 마지막 배치
            X_mini_rest = X[j*n_rest : (j+1)*n_rest]
            y_mini_rest = y[j*n_rest : (j+1)*n_rest]
            F_prime_w_i_sum = np.zeros(X.shape[1])
            F_prime_b_i_sum = 0.
            
            for i in range(n_rest):
                if ( y_mini_rest[i] * self.hypothesis(X_mini_rest[i]) < 1):
                    F_prime_w_i_sum += (-1)*y_mini_rest[i]*X_mini_rest[i]
                    F_prime_b_i_sum += (-1)*y_mini_rest[i]
                    
            self.w_ -= self.learning_rate * ((1 / n_rest) * F_prime_w_i_sum + (1/self.C)*self.w_)
            self.b_ -= self.learning_rate * ((1 / n_rest) * F_prime_b_i_sum)
            
        return self
    
    def predict(self, X):
        return np.where(self.hypothesis(X) >= 1, 1, -1)
    
    def hypothesis(self, X):
        return np.dot(X, self.w_) + self.b_
    
    def shuffle(self, X, y):
        shuffle_index = np.arange(X.shape[0])
        np.random.shuffle(shuffle_index)
        return X[shuffle_index], y[shuffle_index]


# Multiclass Classifier
class MulticlassClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, batch_size=32, max_iter=100, learning_rate=0.1, random_state=1, C=100):
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.C = C
        
    def fit(self, X, y):
        self.labels = np.unique(y) # 0 ~ 9
        self.outputs_ = []
        for label in range(len(self.labels)):
            y_binary = np.where(y == label, 1, -1)
            b_c = BinaryClassifier(self.batch_size, self.max_iter, self.learning_rate, self.random_state, self.C)
            b_c.fit(X, y_binary)
            self.outputs_.append(b_c)
        return self
        
    def predict(self, X):
        prediction = []
        for o in self.outputs_:
            prediction.append(o.hypothesis(X))
        return self.labels[np.argmax(prediction, axis=0)]


# Read Function
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
X_test_no_label = read_no_label(sys.argv[3])


# Data Preprocessing : Standard Scaling
scaler = StandardScaler()
StandardScaler(copy=True, with_mean=True, with_std=True)
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test_no_label)

# Data Preprocessing : PCA
pca = PCA(n_components=0.95)
X_train_temp = X_train.reshape(-1, 28*28)
X_train_reduced = pca.fit_transform(X_train_temp) 
X_test_reduced = pca.transform(X_test_no_label)

# Data Preprocessing : both Standard Scaling and PCA
X_train_scaled_temp = X_train_scaled.reshape(-1, 28*28)
X_train_scaled_reduced = pca.fit_transform(X_train_scaled_temp)
X_test_scaled_reduced = pca.transform(X_test_no_label)

# Training
start = time.time()
m = MulticlassClassifier(C=1000, learning_rate=0.01, batch_size=256)
m.fit(X_train_scaled_reduced, y_train)

# Prediction
y_pred = m.predict(X_test_scaled_reduced)

# txt
f = open("prediction.txt", 'w')
for i in range(len(y_pred)):
    f.write('%d|\n' % y_pred[i])
f.close()


# GridSearch (주석)
# param_grid = [{
#     'C' : [10, 100, 1000],
#     'learning_rate' : [0.1, 0.01, 0.001],
#     'batch_size' : [8, 16, 32, 64, 256]
# }]

# grid_search = GridSearchCV(MulticlassClassifier(), 
#                             param_grid=param_grid, 
#                             cv=3, scoring='accuracy',
#                             n_jobs=-1)

# grid_search.fit(X_train_scaled_reduced, y_train)
# print(grid_search.best_params_)
# print(grid_search.best_score_)