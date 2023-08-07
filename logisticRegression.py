import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.gofplots import ProbPlot
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource


def f(x):
    return 1/(1+np.exp(-x))

class lg():
    def __init__(self,X_train,y_train):
        self.Xtr=X_train
        self.ytr=y_train
        self.x=[]
        self.y=[]
        self.z=[]
        if os.path.isfile("wb.csv"):
            wb=pd.read_csv("wb.csv")
            self.weight=np.array(list(wb["w"])).reshape(1,X_train.shape[1])
            self.b=float(wb.loc[0,"b"])

        else:
            self.weight=np.ones(X_train.shape[1])
            self.b=0
        self.epoch=100
        self.lr=1e-4
        self.threshold=1e-5

    def fit(self):
        epoch=0
        a=0
        while epoch<self.epoch:
            self.x.append(epoch)
            loss=0
            grad=0
            #print(self.weight)
            for i in range(self.Xtr.shape[0]):
                #print(np.dot(self.weight,self.Xtr[i].reshape(26,1)))
                """loss=loss+np.log(f(np.dot(self.weight,self.Xtr[i].reshape(26,1))+self.b))*self.ytr[i]+np.log(1-f(np.dot(self.weight,self.Xtr[i].reshape(26,1))+self.b))*(1-self.ytr[i])
                grad=grad+(self.ytr[i]-np.log(f(np.dot(self.weight,self.Xtr[i].reshape(26,1))+self.b)))*self.Xtr[i]"""
                loss = loss + (-self.ytr[i] * np.log(f(np.dot(self.weight, self.Xtr[i]) + self.b)) - (
                            1 - self.ytr[i]) * np.log(1 - f(np.dot(self.weight, self.Xtr[i]) + self.b)))

                grad = grad + (f(np.dot(self.weight, self.Xtr[i]) + self.b) - self.ytr[i]) * self.Xtr[i]
                #print(grad)
                #print(loss)
            if abs(loss-a)<self.threshold:
                break

            a=loss
            self.z.append(loss)
            self.weight-=grad*self.lr
            #print(grad)
            d={"w":self.weight,"b":self.b}
            df=pd.DataFrame(d)
            df.to_csv("wb.csv",index=False)
            self.rate(self.predit(self.Xtr),self.ytr)
            epoch+=1
        plt.plot(self.x,self.y)
        plt.plot(self.x,self.z)
        plt.show()
        """source = ColumnDataSource(data=dict(x=self.x, y=self.y, z=self.z))
        p = figure(plot_width=1200, plot_height=600, tools='pan,box_zoom,reset', title='正确率与损失值折线图')
        p.line('x', 'y', source=source, line_width=2, color='green', legend_label='accuracy')
        p.line('x', 'z', source=source, line_width=2, color='red', legend_label='loss')
        show(p)"""


    def predit(self,X_test):
        y_pred=[]
        for i in X_test:
            y_pred.append(int(np.dot(self.weight,i.T)>0))
        return np.array(y_pred)

    def rate(self,y_pred,y_test):
        num_test=y_test.shape[0]
        num_correct = np.sum(y_pred.reshape(891,1) == y_test)
        accuracy = float(num_correct) / num_test
        print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))
        self.y.append(accuracy)
        #print(self.y)


data=pd.read_csv("data.csv")
print(data.keys())
X_train=np.array(data[['Pclass', 'Name_ Capt', 'Name_ Col', 'Name_ Don', 'Name_ Dr',
       'Name_ Jonkheer', 'Name_ Lady', 'Name_ Major', 'Name_ Master',
       'Name_ Miss', 'Name_ Mlle', 'Name_ Mme', 'Name_ Mr', 'Name_ Mrs',
       'Name_ Ms', 'Name_ Rev', 'Name_ Sir', 'Name_ the Countess',
       'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Sex', 'Age',
       'SibSp', 'Parch', 'Fare']])
y_train=np.array(data[['Survived']])
lg=lg(X_train,y_train)
lg.fit()
