# model.py

# 2020 T2 COMP9417 Group Project

# Group Member

"""

Shu Yang (z5172181)  
Yue Qi (z5219951)  
Tim Luo (z5115679) 
Yixiao Zhan (z5210796)

"""

# Models
# The following is a stacked model. This constists of:
# LEVEL 1: Catboost, Random Forests, Linear Regressor, KNN, Ridge Regression, SVM, ElasticNet, XGMBoost
# LEVEL 2: Linear Regression (As the meta learning model) which uses the StackingRegressor to find the best model.

from catboost import CatBoostRegressor
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn import preprocessing
from numpy import mean
import xgboost as xgb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import copy

# Suppress Warnings
def warn(*args, **kwargs):
    pass 

import warnings 
warnings.warn = warn

# Load data
train = pd.read_csv('final_train.csv')
test = pd.read_csv('final_test.csv')

# Fix differing features
train_objs_num = len(train)
dataset = pd.concat(objs=[train, test], axis=0, sort=True)
dataset = pd.get_dummies(dataset)
train = copy.copy(dataset[:train_objs_num])
test = copy.copy(dataset[train_objs_num:])

# Fill remaining NA's with 0 and negatives with 0
train.fillna(0, inplace=True)
test.fillna(0, inplace=True)
train[train < 0] = 0
test[test < 0] = 0

# Drop ID Column
train.drop(['id'], axis=1, inplace=True)
test.drop(['id'], axis=1, inplace=True)

y = train.revenue
X = train.drop('revenue', axis=1)

z = train.sort_values('revenue', ascending=False)
z['revenue']

# Select Top 50 Best Features
number_of_features = 50
best_features = SelectKBest(score_func=chi2, k=number_of_features)
y = y.astype('int')
fit = best_features.fit(X, y)
df_scores = pd.DataFrame(fit.scores_)
df_columns = pd.DataFrame(X.columns)
feature_scores = pd.concat([df_columns, df_scores], axis=1)
feature_scores.columns = ['Specs', 'Score']
print(feature_scores.nlargest(number_of_features, 'Score'))

selected_features = feature_scores.nlargest(number_of_features, 'Score')['Specs'].tolist()

X = X[selected_features]
X.describe()

test = test[selected_features]

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=.2, random_state=13)

k = 5

print("\n")

# Prep to determine best alpha for some models


# Find the alpha with best value (here we choose 0.001 for ridge regression)
alphas = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1,0.5, 1]

for a in alphas:
    model = Ridge(alpha=a, normalize=True).fit(X,y) 
    score = model.score(X, y)
    pred_y = model.predict(X)
    mse = mean_squared_error(y, pred_y) 
    print("Alpha:{0:.6f}, R2:{1:.3f}, MSE:{2:.2f}, RMSE:{3:.2f}".format(a, score, mse, np.sqrt(mse)))

# Best alpha for elasticNet is 0.001
for a in alphas:
    model = ElasticNet(alpha=a, normalize=True).fit(X,y) 
    score = model.score(X, y)
    pred_y = model.predict(X)
    mse = mean_squared_error(y, pred_y) 
    print("Alpha:{0:.6f}, R2:{1:.3f}, MSE:{2:.2f}, RMSE:{3:.2f}".format(a, score, mse, np.sqrt(mse)))

# Prepare the stack
def get_stack():
    layer1 = list()
    layer1.append(('cat_boost', CatBoostRegressor(loss_function='RMSE', logging_level='Silent', depth = 9, early_stopping_rounds = 200, iterations = 1000, eval_metric='RMSE', learning_rate = 0.01)))
    layer1.append(('random_forests', RandomForestRegressor(n_estimators = 3000, max_depth = 9, criterion='mse')))
    layer1.append(('linear_reg', LinearRegression()))
    layer1.append(('knn', KNeighborsRegressor(n_neighbors=10, weights='distance', p=5)))
    layer1.append(('ridge_reg', Ridge(alpha=a, normalize=True)))
    layer1.append(('svr', SVR(kernel='rbf',C=2.0, epsilon=0.2, gamma='auto')))
    layer1.append(('elastic_net', ElasticNet(alpha=a, normalize=True)))
    layer1.append(('xgm', xgb.XGBRegressor()))
    layer2 = list()
    layer2.append(('random_forests2', RandomForestRegressor(n_estimators = 3000, max_depth = 9, criterion='mse')))
    layer2.append(('decision_tree', DecisionTreeRegressor(min_samples_leaf=5, criterion='mse', max_depth=9)))
    layer3 = StackingRegressor(estimators = layer2, final_estimator=LinearRegression(), cv=k)
    model = StackingRegressor(estimators=layer1, final_estimator=layer3, cv=k)
    return model

# Compare with just models themselves without stacking
def get_models():
    models = dict()
    models['cat_boost'] = CatBoostRegressor(loss_function='RMSE', logging_level='Silent', depth = 9, early_stopping_rounds = 200, iterations = 1000, eval_metric='RMSE', learning_rate = 0.01)
    models['random_forests'] = RandomForestRegressor(n_estimators = 3000, max_depth = 9, criterion='mse')
    models['linear_reg'] = LinearRegression()
    models['knn'] = KNeighborsRegressor(n_neighbors=10, weights='distance', p=5)
    models['ridge_reg'] = Ridge(alpha=0.001, normalize=True)
    models['svr'] = SVR(kernel='rbf',C=2.0, epsilon=0.2, gamma='auto')
    models['elastic_net'] = ElasticNet(alpha=0.001, normalize=True)
    models['xgm'] = xgb.XGBRegressor()
    models['stacked'] = get_stack()
    return models

# Cross Validation
def evaluate_model(model):
    cv = KFold(n_splits = k, random_state = 10, shuffle=True)
    scores = cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=1, error_score='raise')
    return scores

# Workaround to bug in CatBoostRegressor not having attribute n_features_in_
class CatBoostRegressor(CatBoostRegressor):
    def n_features_in_(self):
        return self.get_feature_count()

# Workaround to bug in StackingRegressor not having attribute final_estimator_
class StackingRegressor(StackingRegressor):
    def final_estimator_(self):
        return self.final_estimator_

print("\n")

# Show which model is better
models = get_models()
results, names = list(), list()
for name, model in models.items():
    scores = evaluate_model(model)
    results.append(-scores)
    names.append(name)
    print("SCORE OF {} === {}".format(name, -mean(scores)))

# Plots of all models
# plt.figure(figsize=(15,15))
# plt.boxplot(results, labels=names, showmeans=True)
# plt.xlabel('Models', fontsize=12)
# plt.ylabel('RMSE', fontsize=12)
# plt.suptitle('Performance Of Different Models', fontsize=14)
# plt.show()

# Train and predict stacked model
stack = get_stack()
stack.fit(X, y)
predictions = stack.predict(test)

# Submission
submission = pd.read_csv('Data/sample_submission.csv')
submission['revenue'] = np.expm1(predictions)
submission.to_csv('submission.csv', index = False)
