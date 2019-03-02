
import pandas as pd
import numpy as np

train = pd.read_csv('weatherHistory.csv')   # Read file

train = train.dropna()  # Drop null values

# Convert categorical values to numerical values
from sklearn.preprocessing import LabelEncoder
train = train.apply(LabelEncoder().fit_transform)

##Null values
nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'


X = train.drop(['Temperature (C)'], axis=1)
y = train['Temperature (C)']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, random_state=42, test_size=.33)
from sklearn import linear_model
lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)

# Evaluate the performance and visualize results
# Evaluate R^2 and RMSE
print("R^2 is: \n", model.score(X_test, y_test))
predictions = model.predict(X_test)
from sklearn.metrics import mean_squared_error
print('RMSE is: \n', mean_squared_error(y_test, predictions))
