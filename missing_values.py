import pandas as pd

def fill_average(data,key):
    """
    均值填充缺失位
    :param data: dataframe
    :param key: head
    :return: dataframe
    """
    average=sum(data[key].dropna())/len(data[key])
    for i in data[data[key].isnull()].index.tolist():
        data.loc[i,key]=average
    return data

def fill_mode(data,key):
    """
    众数填充缺失位
    :param data: dataframe
    :param key: head
    :return: dataframe
    """
    mode=data[key].dropna().mode()[0]
    for i in data[data[key].isnull()].index.tolist():
        data.loc[i,key]=mode
    return data

def fill_data(data,key,fill):
    """
    用自定义数据填充缺失位
    :param data: dataframe
    :param key: head
    :param fill: 你想用来填充空缺位的数据
    :return: dataframe
    """
    for i in data[data[key].isnull()].index.tolist():
        data.loc[i,key]=fill
    return data

def fill_knn(data,key,k):
    """
    knn算法填补空缺值
    :param data: dataframe
    :param key: head，限Age、Cabin、Embarked
    :param k: k值
    :return: dataframe
    """
    for index in data[data[key].isnull()].index.tolist():
        #print(index)
        distance={}
        for i in range(len(data)):
            if i==index:
                continue
            for j in ["Survived","Pclass","Sex","SibSp","Parch","Ticket","Fare"]:
                if data.loc[i,j]!=data.loc[index,j]:
                    distance[i]=distance.get(i,0)+1
        distance = dict(sorted(distance.items(), key=lambda x: x[1]))
        d = list(distance.keys())[:k]
        if key=="Age":
            sum=0
            for i in d:
                sum+=data.loc[i,"Age"]
            data.loc[index,"Age"]=sum/k
        else:
            dd={}
            for i in d:
                dd[data.loc[i,key]]=dd.get(data.loc[i,key],0)+1
            dd = dict(sorted(dd.items(), key=lambda x: x[1]))
            data.loc[index,key]=list(dd.keys())[-1]
    return data


if __name__=="__main__":
    data=pd.read_csv("ttnkh.csv")
    print(fill_knn(data,"Cabin",3))
    print(data.isnull().sum())

