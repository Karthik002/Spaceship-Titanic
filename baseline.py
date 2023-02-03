import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

# read data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# data processing
features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa','VRDeck']
y = train_data['Transported']
X = train_data.copy().loc[:, features]
X = X.fillna(0)

# logistic regression
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
lr = LogisticRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict_proba(x_test)
print("Logistic Regression Model Accuracy: ")
print(log_loss(y_test, y_pred))