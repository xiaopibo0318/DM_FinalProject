from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split




#梯度提升樹、決策樹、RFC
#XGBOOST 弱評估booster
file_path = 'originalData.csv'
data = pd.read_csv(file_path)

X = data.drop(['Global_Sales'], axis=1)
y = data['Global_Sales']

x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=90)

# Training the XGBoost model
svm_model = make_pipeline(StandardScaler(), SVR())
svm_model.fit(x_train, y_train)

score_pre = cross_val_score(svm_model,x_train,y_train,cv=10).mean()
print(f"平方交叉score: {score_pre}")

# Preparing the new data for prediction
new_data_prepared = X.copy()

# Predicting the Global Sales for the new data
new_predictions = svm_model.predict(new_data_prepared)

answerData = pd.read_csv("newPredictGlobal.csv")
myPredict = svm_model.predict(answerData)
answerData["Global_Sales"] = myPredict
answerData.to_csv("40871223H_answer.csv",index=False)