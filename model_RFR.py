import numpy as np
import pandas as pd
from sklearn.metrics import r2_score 
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
#梯度提升樹、決策樹、RFC
#XGBOOST 弱評估booster
file_path = 'originalData.csv'
data = pd.read_csv(file_path)

X = data.drop(['Global_Sales'], axis=1)
y = data['Global_Sales']

x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=90)

#初始隨機森林回歸
rfr = RFR(n_estimators=100,random_state=90,min_samples_leaf=1,max_depth=7)
#rfc = RFR(n_estimators=100,random_state=90)
rfr.fit(x_train,y_train)
prediction = rfr.predict(x_test)
#print(prediction)
score_pre = cross_val_score(rfr,x_train,y_train,cv=10).mean()
print(f"平方交叉score: {score_pre}")
# print(f"train r2:{r2_score(y_train,rfc.predict(x_train))}")
# print(f"test  r2:{r2_score(y_test,prediction)}")

# #調參 ,"max_depth":range(5,100)
# rfr=RFR(n_estimators=100,random_state=90)
# param = {"min_samples_leaf": range(1,20)}     #要調整的參數
# gs = GridSearchCV(estimator=rfc,param_grid=param,cv=5)
# gs.fit(x_train,y_train)

# #調參成績
# best_score=gs.best_score_
# best_params=gs.best_params_
# print(best_score,best_params,end='\n')



answerData = pd.read_csv("newPredictGlobal.csv")
#answerData = answerData.drop(columns=['Unnamed: 10'])
myPredict = rfr.predict(answerData)
answerData["Global_Sales"] = myPredict
answerData.to_csv("40871223H_answer.csv",index=False)