
"""
Created on Sat Jun 23 14:40:59 2018

@author: gates
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

train_dataset = pd.read_csv('train.csv')
test_dataset = pd.read_csv('test.csv')

#get all the features
features = list(train_dataset.columns.values)
#get rid of the id and sales price
features = features[1:-1]
numerical_features = train_dataset._get_numeric_data().columns.values
#get rid of the id and sales price
numerical_features = numerical_features[1:-1]

#uses only numerical features
'''
used_features = []
for i in range(0,len(numerical_features)):
    for j in range(i, len(numerical_features)):
        used_features.append(numerical_features[j])
        X = train_dataset[used_features].iloc[:,:].values
        y = train_dataset.iloc[:, -1].values

        #get rid of NA values
        imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
        imputer = imputer.fit(X)
        X = imputer.transform(X)

        #use polynomial features
        from sklearn.preprocessing import PolynomialFeatures
        poly_reg = PolynomialFeatures(degree = 2)
        X = poly_reg.fit_transform(X)
        
        #split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.7)

        #train the model
        regressor = LinearRegression()
        regressor = regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)

        #test accuracy
        mse = mean_squared_error(y_test, y_pred)
        print(mse)
        if i==0 and j==0:
            min_mse = mse
            min_features = used_features
        elif mse < min_mse:
            min_mse = mse
            min_features = used_features
    used_features = []

print(min_features)
'''

used_features = []
used_categorical_features = []
categorical_features_indicies = []
min_mse = 0
#for all sets of features
for i in range(0,len(features)):
    for j in range(i, len(features)):
        print(str(i) + "   " + str(j))
        if features[j] not in numerical_features:#if it is categorical
            if not train_dataset[features[j]].isnull().values.any():#if there are no NA values
                used_categorical_features.append(features[j])
                categorical_features_indicies.append(len(used_features))#this is necessary for label encoding
                used_features.append(features[j])
            else:
                continue
        else:
            used_features.append(features[j])#numerical features always get added in

        X = train_dataset[used_features].iloc[:,:].values
        y = train_dataset.iloc[:, -1].values
        print(X.shape)
        #label encode all categorical features
        for k in categorical_features_indicies:
            labelencoder = LabelEncoder()
            X[:, k] = labelencoder.fit_transform(X[:, k])
            
        imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
        imputer = imputer.fit(X)
        X = imputer.transform(X)
        
        #one hot encoding
        onehotencoder = OneHotEncoder(categorical_features = categorical_features_indicies)
        X = onehotencoder.fit_transform(X)

        #split dataset into testset and cross validation set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.7)
        
        #train the model
        regressor = DecisionTreeRegressor()
        regressor = regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        #test accuracy
        mse = mean_squared_error(y_test, y_pred)
        #print(mse)
        if i==0 and j==0:
            min_mse = mse
            min_features = used_features
        elif mse < min_mse:
            min_mse = mse
            min_features = used_features
    #reset
    used_features = []
    used_categorical_features = []
    categorical_features_indicies = []
print(min_features)





