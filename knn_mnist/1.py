import operator
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms
batch_size=100
# MNIST dataset
train_dataset=dsets.MNIST(root="./pymnist",
                          train=True,
                          transform=None,
                          download=True)
test_dataset=dsets.MNIST(root="./pymnist",
                         train=False,
                         transform=None,
                         download=True)
# 加载数据
train_loader=torch.utils.data.DataLoader(dataset=train_dataset,
                                         batch_size=batch_size,
                                         shuffle=True)
test_loader=torch.utils.data.DataLoader(dataset=test_dataset,
                                        batch_size=batch_size,
                                        shuffle=True)

def KNN_classify(k,dis,X_train,x_train,Y_test):
    assert dis=='E' or dis=='M','dis must E or M, E代表欧氏距离, M代表曼哈顿距离'
    num_test=Y_test.shape[0]
    label_list=[]
    # 使用欧拉公式作为距离度量
    if(dis=='E'):
        for i in range(num_test):
            # 实现欧式距离公式
            distance=np.sqrt(np.sum(((X_train-np.tile(Y_test[i],(X_train.shape[0],1)))**2),axis=1))
            nearest_k=np.argsort(distance)
            topK=nearest_k[:k]
            classCount={}
            for i in topK:
                classCount[x_train[i]]=classCount.get(x_train[i],0)+1
            sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
            label_list.append(sortedClassCount[0][0])
        return np.array(label_list)
    # 使用曼哈顿公式作为距离度量
    if(dis=='M'):
        for i in range(num_test):
            # 实现曼哈顿公式作为距离度量
            distance=np.sum(np.abs(X_train-np.tile(Y_test[i],(X_train.shape[0],1))),axis=1)
            nearest_k = np.argsort(distance)
            topK = nearest_k[:k]
            classCount = {}
            for i in topK:
                classCount[x_train[i]] = classCount.get(x_train[i], 0) + 1
            sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
            label_list.append(sortedClassCount[0][0])
        return np.array(label_list)

if __name__=='__main__':
    X_train=train_loader.dataset.data.numpy()
    X_train=X_train.reshape(X_train.shape[0],28*28)
    y_train=train_loader.dataset.targets.numpy()
    X_test=test_loader.dataset.data[:1000].numpy()
    X_test=X_test.reshape(X_test.shape[0],28*28)
    y_test=test_loader.dataset.targets[:1000].numpy()
    num_test=y_test.shape[0]
    y_test_pred=KNN_classify(5,'M',X_train,y_train,X_test)
    num_correct=np.sum(y_test_pred==y_test)
    accuracy=float(num_correct)/num_test
    print('Got %d / %d correct => accuracy: %f' % (num_correct,num_test,accuracy))