import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import Imputer

train_dataset = pd.read_csv('train.csv')
test_dataset = pd.read_csv('test.csv')

y_train = train_dataset.iloc[:, -1].values

used_features = ['LotFrontage', 'LotArea', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'BldgType', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'Exterior2nd', 'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'HeatingQC', 'CentralAir', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'SaleCondition']
X_train = train_dataset[used_features].iloc[:,:].values
X_test = test_dataset[used_features].iloc[:,:].values

#encode all of the categorical variables
numerical_features = list(train_dataset[used_features]._get_numeric_data().columns.values)
features = list(train_dataset[used_features].columns.values)

categorical_features = []
categorical_features_indicies = []
for feature in features:
    if feature not in numerical_features:
        categorical_features.append(feature)
        categorical_features_indicies.append(features.index(feature))

#begin labelencoding
for k in categorical_features_indicies:
    labelencoder1 = LabelEncoder()
    labelencoder2 = LabelEncoder()
    X_train[:, k] = labelencoder1.fit_transform(X_train[:, k].astype(str))
    X_test[:, k] = labelencoder2.fit_transform(X_test[:, k].astype(str))

#apparently there are NA values
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X_train)
X_train = imputer.transform(X_train)
X_test = imputer.transform(X_test)
'''
#check each column and make sure that the values are the same
train_df = pd.DataFrame(data=X_train, index = train_dataset[['Id']].values, columns=features)
test_df = pd.DataFrame(data=X_test, index = test_dataset[['Id']].values, columns=features)

new_features = []

for feature in features:
    if feature in categorical_features:
        train_feature = set(train_df[[feature]].values.flatten())
        test_feature = set(test_df[[feature]].values.flatten())
        #if the two sets are equal
        if train_feature == test_feature:
            new_features.append(feature)
    else:
        new_features.append(feature)

print(new_features)
'''
#do onehotencoding
onehotencoder1 = OneHotEncoder(categorical_features = categorical_features_indicies)
onehotencoder2 = OneHotEncoder(categorical_features = categorical_features_indicies)
X_train = onehotencoder1.fit_transform(X_train).toarray()
X_test = onehotencoder2.fit_transform(X_test).toarray()

#avoid dummy variable trap
X_train = X_train[:, 1:]
X_test = X_test[:, 1:]

'''
#train the model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
#only use polynomial features for certain features
X_train[0] = poly_reg.fit_transform(X_train[0].reshape(-1,1))
X_train[1] = poly_reg.fit_transform(X_train[1].reshape(-1,1))
X_test[0] = poly_reg.fit_transform(X_test[0].reshape(-1,1))
X_test[1] = poly_reg.fit_transform(X_test[1].reshape(-1,1))
'''

regressor = DecisionTreeRegressor()
regressor = regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

#need to prep data for submission, add id 
ids = test_dataset[['Id']]
preds = pd.DataFrame(y_pred)
df = pd.merge(ids, preds, left_index=True, right_index=True)
df.columns = ['Id', 'SalePrice']
df.to_csv("y_predict.csv", index=False)
print("done")

