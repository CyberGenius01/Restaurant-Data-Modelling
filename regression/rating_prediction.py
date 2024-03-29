""" RESTAURANT RATING PREDICTION MODEL """

# Importing Basic Libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Dataset.csv')
dataset = dataset.iloc[:,[7,8,10,12,13,14,15,16,17]]
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# Handling the missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=0, strategy='mean')
numeric_cols = [0,1,2,7]
X[:,numeric_cols] = imputer.fit_transform(X[:,numeric_cols])


# Encoding the categorical variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:,3] = le.fit_transform(X[:,3])
for i in [4,5,6]:
    X[:,i] = le.transform(X[:,i])

"""np.savetxt('X.txt', X)"""
    
# Splitting the dataset into test and training set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# Trainig the dataset using SVR
from sklearn.svm import SVR
regressor_sv = SVR(kernel='rbf')
regressor_sv.fit(X_train, y_train)

X_opt_test = X_test

# Trainig the dataset using Backward elimination
import statsmodels.api as sm
def __backwardElimination(dataset, sl):
  for _ in range(1,len(dataset[0])):
    regressor_OLS = sm.OLS(endog = y_train, exog = dataset).fit()
    maxPval = max(regressor_OLS.pvalues[1:])
    if maxPval > sl:
      j = [k for k in range(1,len(dataset[0])) if regressor_OLS.pvalues[k] == maxPval]
      dataset = np.delete(dataset, j, 1)
      global X_opt_test
      X_opt_test = np.delete(X_opt_test, j, 1)
  return dataset, regressor_OLS

SL = 0.02
X_opt, regressor_ols = __backwardElimination(X_train.astype(float), SL)


# Trainig the dataset using Random Forest
from sklearn.ensemble import RandomForestRegressor
regressor_rf = RandomForestRegressor(n_estimators=1000)
regressor_rf.fit(X_train, y_train)


# Trainig the dataset using Decision Tree
from sklearn.tree import DecisionTreeRegressor
regressor_dt = DecisionTreeRegressor()
regressor_dt.fit(X_train, y_train)

# Comparing result
sv_pred = regressor_sv.predict(X_test)
ols_pred = regressor_ols.predict(X_opt_test)
dt_pred = regressor_dt.predict(X_test)
rf_pred = regressor_rf.predict(X_test)

from sklearn.metrics import r2_score
print((r2_score(y_test, sv_pred), r2_score(y_test, ols_pred), r2_score(y_test, dt_pred), r2_score(y_test, rf_pred)))

# Choosing best model and then implementning single valued predictions
test = np.array([[-84.206944,31.622412,10,'No','No','No','No', 1]])
def predict_rating(test):
    for i in [3,4,5,6]:
        test[:,i] = le.transform(test[:,i])
    testVal = regressor_rf.predict(test)
    return testVal

print(predict_rating(test)[0])
# Actual value = 3.4
# Result value = 3.49


