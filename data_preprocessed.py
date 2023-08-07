import pandas as pd
import re
import missing_values
import normalization

if __name__=="__main__":
    data=pd.read_csv("ttnkh.csv")
    data=missing_values.fill_knn(data, "Cabin", 5)
    data=missing_values.fill_knn(data, "Embarked", 5)
    data=missing_values.fill_average(data, "Age")
    for i in range(len(data["Sex"])):
        if data.loc[i,"Sex"]=="male":
            data.loc[i,"Sex"]=1
        else:
            data.loc[i,"Sex"]=0


    for i in range(len(data["Name"])):
            data.loc[i,"Name"]=re.split(r"[,.]",data.loc[i,"Name"])[1]

    new_data=data[["Name",'Pclass',"Embarked"]].copy()
    new_data=pd.get_dummies(new_data)
    new_data=pd.concat([new_data,data[['Survived','Sex','Age','SibSp','Parch','Fare']]],axis=1)

    new_data=normalization.min_max(new_data,"Age")
    new_data=normalization.min_max(new_data,"Fare")
    new_data=normalization.min_max(new_data,"SibSp")
    new_data=normalization.min_max(new_data,"Parch")


    new_data.to_csv("data.csv",index=False)

    print(new_data)