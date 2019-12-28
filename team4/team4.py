import os
import struct
import itertools
import numpy as np
import matplotlib.pyplot as pyplot

def read(dataset = "training", path = "."):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """

    if dataset is "training_D1_60k":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is "testing_D2_10k":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    elif dataset is "testing_D3":
        fname_img = os.path.join(path, 'testall-images-idx3-ubyte')
    elif dataset is "training_new_10k":
        fname_img = os.path.join(path, 'new1k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'new1k-labels-idx1-ubyte')
    elif dataset is "training_D1_D2_new_10k":
        fname_img = os.path.join(path, 'newtrain-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'newtrain-labels-idx1-ubyte')
    else:
        raise Exception("dataset must be 'testing' or 'training'")

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    get_img = lambda idx: (lbl[idx], img[idx])

    # Create an iterator which returns each image in turn
    for i in range(len(lbl)):
        yield get_img(i)

def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
pyplot.show()

def plot_confusion_matrix(cm, classes, 
                          normalize=False, 
                          title='Confusion matrix', 
                          cmap=pyplot.cm.Blues):
#     ***
#     This function prints and plots the confusion matrix.
#     Normalizaton can be applied by setting 'normalize=True'.
#     ***
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    
    pyplot.imshow(cm, interpolation = 'nearest', cmap = cmap)
    pyplot.title(title)
    pyplot.colorbar()
    tick_marks = np.arange(len(classes))
    pyplot.xticks(tick_marks, classes, rotation=45)
    pyplot.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        pyplot.text(j, i, format(cm[i,j], fmt),
                 horizontalalignment = "center",
                 color="white" if cm[i,j] > thresh else "black")
        
    pyplot.tight_layout()
    pyplot.ylabel('True label')
    pyplot.xlabel('Predicted label')

# D1 : train 60k
train_D1 = list(read("training_D1_60k", "./data"))
X_train_D1, y_train_D1 = [], []
for i in range(len(train_D1)):
    X_train_D1.append(np.ravel(train_D1[i][1]))
    y_train_D1.append(train_D1[i][0])

# new_10k : train 10k
train_new_10k = list(read("training_new_10k", "./data"))
X_train_new_10k, y_train_new_10k = [], []
for i in range(len(train_new_10k)):
    X_train_new_10k.append(np.ravel(train_new_10k[i][1]))
    y_train_new_10k.append(train_new_10k[i][0])
    
# total_train : D1 + D2 + new_10k -> We will use for training data (80k)
total_train = list(read("training_D1_D2_new_10k", "./data"))
X_train_total, y_train_total = [], []
for i in range(len(total_train)):
    X_train_total.append(np.ravel(total_train[i][1]))
    y_train_total.append(total_train[i][0])

    
# test_D2 : test 10k
test_D2 = list(read("testing_D2_10k", "./data"))
X_test_D2, y_test_D2 = [], []
for i in range(len(test_D2)):
    X_test_D2.append(np.ravel(test_D2[i][1]))
    y_test_D2.append(test_D2[i][0])

# # test_D3 : D2 + new 50k -> We will use for testing data (60k)
# test_D3 = list(read("testing_D3", "./data"))
# X_test_D3, y_test_D3 = [], []
# for i in range(len(test_D3)):
#     X_test_D3.append(np.ravel(test_D3[i][1]))


class MBSGDClassifier(object):
    """ Mini-Batch Stochastic Gradient Descent
    
    매개변수
    -------
    learning_rate : float (defalut = 0.1)
        학습률 (0.0 과 1.0 사이)
    max_iter : int (default = 100)
        훈련 데이터셋 최대 반복 횟수
    batch_size : int (defalut = 32)
        Mini-Batch 크기
    C : float (defalue = 1.0)
        규제 파라미터 (1/lambda)
        반드시 양수여야 한다. (Must be positive)
    random_state : int (defalut = 1)
        가중치 무작위 초기화를 위한 난수 생성기 시드
        
        
    속성
    ------
    w_ : 1d-array
        학습된 가중치
    b_ : float
        학습된 편향(bias)
    """
    
    def __init__(self, batch_size=32, max_iter=100, learning_rate=0.1, random_state=1, C=1.0):
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.C = C
        
        
    def fit(self, X, y):
        
        # 예외 처리
        if self.C < 0:
            raise ValueError("The C value of %r must be positive" % self.C)
        if ((learning_rate < 0) or (learning_rate > 1)):
            raise ValueError("The learning_rate value of %r is invalid. Set the learning_rate value between 0.0 and 1.0." % learning_rate)
            
        # w, b 값 초기화
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=len(X))
        self.b_ = 0
        
        # 입력한 최대 실행 횟수 만큼 실행
        for e in range(self.max_iter):
            
            # F_prime_w : w에 대한 F' (w에 대한 F 미분)
            # 최종 F'_w, F'_b 계산에 필요한 F'_w_i, F'_b_i의 합 
            F_prime_w_i_sum = 0
            F_prime_b_i_sum = 0
            
            # 데이터 랜덤 셔플
            X_y_zip = [[x,y] for x, y in zip(X, y)]
            rgen.shuffle(X_y_zip)
            
            # 배치 개수 설정 : 데이터의 총 개수 / 배치 사이즈  ->  # mnist_train : 60000 / 32 = 1875
                # 질문 : 데이터의 개수 가 배치 사이즈로 안나눠떨어질 경우는 어떻게 해야하나?
            n_batches = int(len(X_y_zip) / self.batch_size)
            
            # 배치의 개수 만큼 -> mnist_train : 1875
            for j in range(n_batches):
                # 설정한 배치 사이즈만큼 배치를 설정해준다. (defalut : 32 이므로 0~31, 32~63, ...)
                X_y_zip_mini = X_y_zip[j*self.batch_size : (j+1)*self.batch_size]
                X_mini = [n[0] for n in X_y_zip_mini]
                y_mini = [n[1] for n in X_y_zip_mini]
                
                # 각 배치의 사이즈 만큼 (defalut : 32)
                for k in range(self.batch_size):
                    X_mini_k = X_mini[k]
                    y_mini_k = y_mini[k]
                    # 식의 이해를 돕기 위해 & 변수 명을 간단히 하기 위해, X_i, y_i 사용
                    y_i = y_mini_k # y_i는 한 데이터 (784개로 이루어진) 마다 모두 동일
                    
                    # X는 784개의 원소로 이루어져 있으므로 각각 X_i를 계산해주기 위해 for문을 한 번 더 사용
                    for i in range(len(X_mini)): 
                        X_i = X_mini_k[i]
                        hyperplane = np.dot(X_i, self.w_) + self.b_
                        
                        """
                        에러 발생
                        ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
                        """
                        if ((y_i*hyperplane).all() < 1):
                            F_prime_w_i = (-1)*y_i*X_i + (1 / self.C)*self.w_
                            F_prime_b_i = (-1)*y_i
                        else:
                            F_prime_w_i = (1 / self.C) * self.w_
                            F_prime_b_i = 0
                
                        F_prime_w_i_sum += F_prime_w_i
                        F_prime_b_i_sum += F_prime_b_i
                
                
            F_prime_w = (1 / self.batch_size) * F_prime_w_i_sum
            F_prime_b = (1 / self.batch_size) * F_prime_b_i_sum
                
            # 최종 가중치, 편향 업데이트
            self.w_ = self.w_ - self.learning_rate * F_prime_w
            self.b_ = self.b_ - self.learning_rate * F_prime_b
            
        return self
    
    def predict(self, X):
        Z = np.zeros((np.shape(X)[0]))
        for i in enumerate(X):
            Z[i] = np.dot(X[i], self.w_) + self.b_
        return Z


learning_rate = 0.1
mbsgd=MBSGDClassifier(learning_rate=0.1)
mbsgd.fit(X_train_new_10k, y_train_new_10k)