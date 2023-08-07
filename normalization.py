import pandas as pd
import missing_values

def min_max(data,key):
    MAX=max(data[key])
    MIN=min(data[key])
    for i in range(len(data[key])):
        data.loc[i,key]=(data.loc[i,key]-MIN)/MAX
    return data

def z_score(data,key):
    average=sum(data[key])/len(data[key])
    s=0
    for i in range(len(data[key])):
        s+=(data.loc[i,key]-average)**2
    s=(s/(len(data[key]-1)))**(1/2)
    for i in range(len(data[key])):
        data.loc[i,key]=(data.loc[i,key]-average)/s
    return data

if __name__=="__main__":
    data=pd.read_csv("ttnkh.csv")
    data=missing_values.fill_average(data,"Age")
    print(z_score(data,"Age")["Age"])