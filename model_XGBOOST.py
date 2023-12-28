from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split

file_path = 'originalData.csv'
data = pd.read_csv(file_path)

X = data.drop(['Global_Sales'], axis=1)
y = data['Global_Sales']
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=90)

model = XGBRegressor(objective ='reg:squarederror')
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(rmse)

score_pre = cross_val_score(model,x_train,y_train,cv=10).mean()
print(f"平方交叉score: {score_pre}")

answerData = pd.read_csv("newPredictGlobal.csv")
myPredict = model.predict(answerData)
answerData["Global_Sales"] = myPredict
answerData.to_csv("40871223H_answer.csv",index=False)