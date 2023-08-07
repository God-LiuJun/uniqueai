import numpy as np
import operator
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

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
    X_train=X_train.reshape(X_train.shape[0],-1)
    mean_image=getXmean(X_train)
    X_train=centralized(X_train,mean_image)
    y_train=train_loader.dataset.targets.numpy()
    y_train=np.array(y_train)
    X_test=test_loader.dataset.data.numpy()
    X_test=X_test.reshape(X_test.shape[0],-1)
    X_test=centralized(X_test,mean_image)
    y_test=test_loader.dataset.targets.numpy()
    y_test=np.array(y_test)

    """X_train=train_loader.dataset.data.numpy()
    mean_image=getXmean(X_train)
    X_train=centralized(X_train,mean_image)
    X_train=X_train.reshape(X_train.shape[0],28*28)
    y_train=train_loader.dataset.targets.numpy()
    X_test=test_loader.dataset.data[:1000].numpy()
    X_test=X_test.reshape(X_test.shape[0],28*28)
    y_test=test_loader.dataset.targets[:1000].numpy()"""

    num_folds=5
    k_choices=[1,3,5,8,10,12,15,20]
    num_training=X_train.shape[0]
    X_train_folds=[]
    y_train_folds=[]
    X_indices=np.array_split(X_train,indices_or_sections=num_folds)
    y_indices=np.array_split(y_train,indices_or_sections=num_folds)
    for i in range(num_folds):
        X_train_folds.append(X_indices[i])
        y_train_folds.append(y_indices[i])
    k_to_accuracies={}
    for k in k_choices:
        print(k)
        # 进行交叉验证
        acc=[]
        for i in range(num_folds):
            print(i)
            x=X_train_folds[0:i]+X_train_folds[i+1:]   # 训练集不包含验证集
            x=np.concatenate(x,axis=0)                 # 将4个训练集合并在一起
            y = y_train_folds[0:i] + y_train_folds[i + 1:]
            y = np.concatenate(y)

            test_x=X_train_folds[i]
            test_y=y_train_folds[i]

            classifier=Knn()
            classifier.fit(x,y)

            y_pred=classifier.predit(k,'M',test_x)
            accuracy=np.sum(y_pred==test_y)
            acc.append(accuracy)
            print(accuracy)
        k_to_accuracies[k]=acc                         # 计算交叉验证的平均准确率
        # 输出准确度
        for k in sorted(k_to_accuracies):
            for accuracy in k_to_accuracies[k]:
                print("k = %d, accuracy = %f"%(k,accuracy))

    for k in k_choices:
        accuracies=k_to_accuracies[k]
        plt.scatter([k]*len(accuracies),accuracies)

    accuracies_mean=np.array([np.mean(v) for (k,v) in sorted(k_to_accuracies.items())])
    accuracies_std = np.array([np.std(v) for (k, v) in sorted(k_to_accuracies.items())])

    plt.errorbar(k_choices,accuracies_mean,yeer=accuracies_std)
    plt.title('Cross-validation on k')
    plt.xlabel('k')
    plt.ylabel('Cross-validation accuracy')
    plt.show()