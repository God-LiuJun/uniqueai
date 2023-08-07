import numpy as np
import operator
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

def getXmean(X_train):
    X_train=np.reshape(X_train,(X_train.shape[0],-1))  # 将图片从二位展开为一维
    mean_image = np.mean(X_train, axis=0)              # 求出训练集中所有图片每个像素位置上的均值
    return mean_image

def centralized(X_test, mean_image):
    X_test=np.reshape(X_test,(X_test.shape[0],-1))     # 将图片从二维展开为一维
    X_test=X_test.astype(float)
    X_test-=mean_image                                 # 对输入数据集X进行中心化处理，减去均值图像，实现零均值化
    return X_test

class Knn:

    def __init__(self):
        pass

    def fit(self,X_train,y_train):
        self.Xtr=X_train
        self.ytr=y_train

    def predit(self,k,dis,X_test):
        assert dis=='E' or dis=='M', 'dis muat be E or M'
        num_test = X_test.shape[0]
        label_list = []
        # 使用欧拉公式作为距离度量
        if (dis == 'E'):
            for i in range(num_test):
                # 实现欧式距离公式
                distance = np.sqrt(np.sum(((self.Xtr - np.tile(X_test[i], (self.Xtr.shape[0], 1))) ** 2), axis=1))
                nearest_k = np.argsort(distance)
                topK = nearest_k[:k]
                classCount = {}
                for i in topK:
                    classCount[self.ytr[i]] = classCount.get(self.ytr[i], 0) + 1
                sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
                label_list.append(sortedClassCount[0][0])
            return np.array(label_list)
        # 使用曼哈顿公式作为距离度量
        if (dis == 'M'):
            for i in range(num_test):
                # 实现曼哈顿公式
                # 按照列的方向相加，其实就是行的相加
                distance = np.sum(np.abs(self.Xtr - np.tile(X_test[i], (self.Xtr.shape[0], 1))), axis=1)
                nearest_k = np.argsort(distance)
                topK = nearest_k[:k]
                classCount = {}
                for i in topK:
                    classCount[self.ytr[i]] = classCount.get(self.ytr[i], 0) + 1
                sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
                label_list.append(sortedClassCount[0][0])
            return np.array(label_list)

if __name__ =='__main__':
    X_train=train_loader.dataset.data.numpy()
    mean_image=getXmean(X_train)
    X_train=centralized(X_train,mean_image)
    X_train=X_train.reshape(X_train.shape[0],28*28)
    y_train=train_loader.dataset.targets.numpy()
    X_test=test_loader.dataset.data[:1000].numpy()
    X_test=X_test.reshape(X_test.shape[0],28*28)
    y_test=test_loader.dataset.targets[:1000].numpy()
    num_test=y_test.shape[0]
    k=Knn()
    k.fit(X_train,y_train)
    y_test_pred=k.predit(5,'M',X_test)
    num_correct=np.sum(y_test_pred==y_test)
    accuracy=float(num_correct)/num_test
    print('Got %d / %d correct => accuracy: %f' % (num_correct,num_test,accuracy))