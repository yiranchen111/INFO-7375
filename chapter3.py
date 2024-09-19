import numpy as np
import os
from PIL import Image


#sigmoid function

def sigmoid(z):
    return 1/(1 + np.exp(-z))

#loss function 

def loss_function(A,Y):
    m = Y.shape[0]
    A = np.clip(A, 1e-10, 1 - 1e-10)
    loss = -1/m * np.sum(Y*np.log(A) + (1-Y)*np.log(1 - A))
    return loss

def forward_propagation (X,W,b):
    Z  = np.dot(X,W)+b
    A = sigmoid(Z)
    return A

def backward_propagation(X,A,Y):
    m = X.shape[0]
    dZ = A - Y.reshape(-1,1)
    dW = np.dot(X.T,dZ)/m
    db = np.sum(dZ)/m
    return dW,db


def update_parameters(W,b,dW,db,rate):
    W = W - rate * dW
    b = b - rate * db
    return W,b

def initialize_parameters(size):
    W = np.random.randn(size,1)*0.01
    b = 0
    return W,b

def train (X_train,Y_train,number,rate):
    size = X_train.shape[1]
    W,b = initialize_parameters(size)

    for i in range(number):
        A = forward_propagation(X_train , W , b)
        loss = loss_function(A,Y_train)
        dW,db = backward_propagation(X_train , A , Y_train)
        W,b = update_parameters(W,b,dW,db,rate)

        if i % 10 == 0:
            print(f" {i} ，loss functions：{loss}")
            print(f" W: {W.flatten()[:5]}， b: {b}")

    return W,b

def test (X_test , W , b):
    A_test = forward_propagation(X_test , W ,b)
    predictions = (A_test > 0.5).astype(int)
    return predictions

def load_images_from_folder(folder_path, img_size=(20, 20), labeled=True):
    images = []
    labels = []
    
    if labeled: 
        for label in range(10):  
            digit_folder = os.path.join(folder_path, str(label))
            for img_filename in os.listdir(digit_folder):  
                if img_filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    img_path = os.path.join(digit_folder, img_filename)
                    img = Image.open(img_path).convert('L')  
                    img_resized = img.resize(img_size)  
                    img_array = np.array(img_resized) / 255.0  
                    images.append(img_array.flatten())  
                    labels.append(label)
        return np.array(images), np.array(labels)
    else:  
        for img_filename in os.listdir(folder_path):
            if img_filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                img_path = os.path.join(folder_path, img_filename)
                img = Image.open(img_path).convert('L') 
                img_resized = img.resize(img_size)  
                img_array = np.array(img_resized) / 255.0  
                images.append(img_array.flatten())  
        return np.array(images)

def one_vs_all_train(X_train, Y_train, num_classes, num_iterations, learning_rate):
    W_all = []
    b_all = []
    
    # 对每个类别分别训练感知器
    for class_label in range(num_classes):
        W, b = initialize_parameters(X_train.shape[1])
        Y_binary = np.where(Y_train == class_label, 1, 0)  # 将标签二值化，当前类别为1，其余为0
        
        # 对每个类别的感知器进行训练
        for i in range(num_iterations):
            A = forward_propagation(X_train, W, b)
            loss = loss_function(A, Y_binary)
            dW, db = backward_propagation(X_train, A, Y_binary)
            W, b = update_parameters(W, b, dW, db, learning_rate)
            
            # 每100次迭代打印一次损失
            if i % 100 == 0:
                print(f"Class {class_label}, Iteration {i}, Loss: {loss}")
        
        # 保存每个类别的权重和偏置
        W_all.append(W)
        b_all.append(b)
    
    return W_all, b_all

# One-vs-All 的预测方法
def one_vs_all_predict(X_test, W_all, b_all):
    scores = []
    
    # 对每个类别的感知器计算得分
    for W, b in zip(W_all, b_all):
        A = forward_propagation(X_test, W, b)
        scores.append(A)
    
    scores = np.hstack(scores)  # 将每个类别的得分拼接成一个矩阵
    predictions = np.argmax(scores, axis=1)  # 返回得分最高的类别作为预测结果
    return predictions


if __name__ == '__main__':
 
    train_image_folder = './digits/'  
    test_image_folder = './test/'  
    

    X_train, Y_train = load_images_from_folder(train_image_folder, labeled=True)
    X_test = load_images_from_folder(test_image_folder, labeled=False)

    num_classes = 10  # 0-9 共10个类别
    num_iterations = 1000  # 每个感知器的训练迭代次数
    learning_rate = 0.01  # 学习率

    # 训练
    W_all, b_all = one_vs_all_train(X_train, Y_train, num_classes, num_iterations, learning_rate)

    # 预测
    predictions = one_vs_all_predict(X_test, W_all, b_all)
    print("测试结果：", predictions)
    

