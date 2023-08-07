import numpy as np
import operator
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms
batch_size=100
# Cifar10 dataset
train_dataset=dsets.CIFAR10(root="./pycifar",     #选择数据根目录
                          train=True,           #选择训练集
                          transform=None,
                          download=True)        #从网络上下载图片
test_dataset=dsets.CIFAR10(root="./pycifar",
                         train=False,
                         transform=None,
                         download=True)
# 加载数据
train_loader=torch.utils.data.DataLoader(dataset=train_dataset,
                                         batch_size=batch_size,
                                         shuffle=True)   #将数据打乱
test_loader=torch.utils.data.DataLoader(dataset=test_dataset,
                                        batch_size=batch_size,
                                        shuffle=True)

def getXmean(X_train):
    X_train=np.reshape(X_train,(X_train.shape[0],-1))  # 将图片从二位展开为一维
    mean_image = np.mean(X_train, axis=0)              # 求出训练集中所有图片每个像素位置上的均值
    return mean_image

def centralized(X_test, mean_image):
    X_test=np.reshape(X_test,(X_test.shape[0],-1))     # 将图片从二维展开为一维
    X_test=X_test.astype(float)
    X_test-=mean_image                                 # 对输入数据集X进行中心化处理，减去均值图像，实现零均值化
    return X_test

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

if __name__ =='__main__':
    X_train=train_loader.dataset.data
    mean_image=getXmean(X_train)
    X_train=centralized(X_train,mean_image)
    y_train=train_loader.dataset.targets
    X_test=test_loader.dataset.data[:100]
    X_test=centralized(X_test,mean_image)
    y_test=test_loader.dataset.targets[:100]
    num_test=len(y_test)
    y_test_pred=KNN_classify(6,'M',X_train,y_train,X_test)
    num_correct=np.sum(y_test_pred==y_test)
    accuracy=float(num_correct)/num_test
    print('Got %d / %d correct => accuracy: %f' % (num_correct,num_test,accuracy))