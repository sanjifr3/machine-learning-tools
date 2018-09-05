#!/usr/bin/python2.7

import random
import itertools
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot

# Preprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Metrics
from sklearn.metrics import confusion_matrix

class MLTools(object):
  models = {
    'knn': KNeighborsClassifier(),
    'log': LogisticRegression(),
    'svm': SVC(),
    'dt':  DecisionTreeClassifier(),
    'rf': RandomForestClassifier(n_estimators=20)
  }
  
  param_grids = {}
  param_grids['knn'] = {"n_neighbors": np.arange(1,31,2),
                        "metric": ['euclidean','cityblock','minkowski']}
  param_grids['log'] = {'C': [0.1,1, 10,100,1000],
                        'solver':['newton-cg','lbfgs','liblinear','sag']}
  param_grids['svm'] = {'C': [0.1,1,10,100], 'gamma':[1,0.1,0.01,0.001],
                        'kernel': ['rbf','linear','poly','sigmoid'],
                        'degree': [3,4,5,6]}                      
  param_grids['dt'] = {'criterion': ['gini','entropy'],
                       'max_depth': np.arange(1,6,1),
                       'min_samples_split': np.arange(2,100,2),
                       'min_samples_leaf': np.arange(1,100,2),
                       'max_features': np.arange(1,6,1)}
  param_grids['rf'] =  {"max_depth": [1,2,3],
                        "max_features": [1, 2, 3],
                        "min_samples_split": [2, 3],
                        "min_samples_leaf": [2, 3],
                        "bootstrap": [True, False],
                        "criterion": ["gini", "entropy"]}
  
  percentiles=[0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95]
  
  random_state=101
  scaler = StandardScaler()

  best_model = None
  best_scaler = None

  def __init__(self):
    print 'Machine Learning Tools Ready'\

  def getXY(self, df, target_feature):
    X = df.drop(target_feature, axis=1)
    y = df[target_feature]
    return X, y
  
  def splitData(self, df, target_feature, test_size=0.3, stratify=True):
    X,y = self.getXY(df, target_feature)
    
    if stratify:
      return train_test_split(X,y,test_size=test_size, random_state=self.random_state, stratify=y)
   
    else:
      return train_test_split(X,y,test_size=test_size, random_state=self.random_state)

  def getKFolds(self, df, target_feature, k=10, shuffle=True):
    X, y = self.getXY(df, target_feature)
    folds = list(StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=self.random_state).split(X,y))
    return X, y, folds

  def generateFeatureSets(self, df, insignificant_features):
    possible_features = list(df.columns.values)
    possible_features = [feature for feature in possible_features if feature not in insignificant_features]
    feature_sets = []

    for i in range(1,len(possible_features)+1):
      feature_sets += list(itertools.combinations(possible_features, i))

    return feature_sets

  def scale(self, X_train, X_test):
    X_train = self.scaler.fit_transform(X_train)
    X_test = self.scaler.transform(X_test)

    return X_train, X_test

  def trainTestSplitEvaluate(self, df, target_feature, model_name='log', params=None, scale_features=True, test_size=0.3):
    assert model_name in self.models.keys()

    X_train, X_test, y_train, y_test = self.splitData(df, target_feature, test_size)

    if scale_features:
      X_train, X_test = self.scale(X_train,X_test)

    model = self.models[model_name]

    if params is not None:
      model.set_params(**params)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    TN = confusion_matrix(y_test, predictions)[0][0]
    FP = confusion_matrix(y_test, predictions)[0][1]
    FN = confusion_matrix(y_test, predictions)[1][0]
    TP = confusion_matrix(y_test, predictions)[1][1]
    total = TN + FP + FN + TP
    ACC = (TP + TN) / float(total)

    if scale_features:
      print ("The {} model got an accuracy of {}% on the testing set w/ scaling".format(model_name, round(ACC*100,2)))
    else:
      print ("The {} model got an accuracy of {}% on the testing set w/o scaling".format(model_name, round(ACC*100,2)))


  def kFoldEvaluate(self, df, target_feature, model_name='log', params=None, scale_features=True, folds=10):
    assert model_name in self.models.keys()

    X, y, folds = self.getKFolds(df, target_feature, k=folds)

    acc = np.zeros(10)

    for fold_id, (train_idx, test_idx) in enumerate(folds):
      #print '\nFolds:',fold_id

      X_train, y_train = X.values[train_idx], y.values[train_idx]
      X_test, y_test = X.values[test_idx], y.values[test_idx]

      if scale_features:
        X_train, X_test = self.scale(X_train, X_test)

      model = self.models[model_name]

      if params is not None:
        model.set_params(**params)

      model.fit(X_train, y_train)

      predictions = model.predict(X_test)

      TN = confusion_matrix(y_test, predictions)[0][0]
      FP = confusion_matrix(y_test, predictions)[0][1]
      FN = confusion_matrix(y_test, predictions)[1][0]
      TP = confusion_matrix(y_test, predictions)[1][1]
      total = TN + FP + FN + TP
      fold_acc = (TP + TN) / float(total) * 100.0

      acc[fold_id] = fold_acc

      print ("Fold {}: Accuracy: {}%".format(fold_id, round(fold_acc,2)))

    print ("{} Model Average Score: {}% ({}%)".format(model_name, round(np.mean(acc),3),round(np.std(acc),3)))


  def gsParamsTrainTestSplitEvaluate (self, df, target_feature, model_name='log', scale_features=True, test_size=0.3, trials=10):
    assert model_name in self.models.keys()

    X_train, X_test, y_train, y_test = self.splitData(df, target_feature, test_size)

    current_param_grid = self.param_grids[model_name]
    best_acc = 0.0
    best_params = {}

    if scale_features:
      X_train, X_test = self.scale(X_train,X_test)

    start_time = time.time()

    for n_trials in range(0, trials):
      
      if n_trials % 10 == 0:
        print 'Progress (',n_trials,'/',trials,'): ',np.float64(n_trials)/trials*100.0,'%'
        print ' RunTime: ',round((time.time() - start_time)/60,2), 'mins'
      
      model = self.models[model_name]
      
      params = {}
      for k, v in current_param_grid.items():
        params[k] = v[random.randrange(len(v))]

      model.set_params(**params)

      model.fit(X_train, y_train)
      predictions = model.predict(X_test)

      TN = confusion_matrix(y_test, predictions)[0][0]
      FP = confusion_matrix(y_test, predictions)[0][1]
      FN = confusion_matrix(y_test, predictions)[1][0]
      TP = confusion_matrix(y_test, predictions)[1][1]
      total = TN + FP + FN + TP
      ACC = (TP + TN) / float(total)*100.0

      if ACC > best_acc:
        best_params = params
        best_acc = ACC

    if scale_features:
      print ("The {} model w/ the following hyperparams: {} achieved the highest accuracy of {}% on the testing set w scaling".format(model_name, str(best_params), round(best_acc,2)))
    else:
      print ("The {} model w/ the following hyperparams: {} achieved the highest accuracy of {}% on the testing set w/o scaling".format(model_name, str(best_params), round(best_acc,2)))

  def gsParamskFoldEvaluate(self, df, target_feature, model_name='log', scale_features=True, folds=10, trials=10):
    assert model_name in self.models.keys()

    X, y, folds = self.getKFolds(df, target_feature, k=folds)
    current_param_grid = self.param_grids[model_name]

    best_acc = 0.0
    best_params = {}

    start_time = time.time()

    for n_trial in range(trials):

      if n_trials % 10 == 0:
        print 'Progress (',n_trials,'/',trials,'): ',np.float64(n_trials)/trials*100.0,'%'
        print ' RunTime: ',round((time.time() - start_time)/60,2), 'mins'

      model = self.models[model_name]

      params = {}
      for k, v in current_param_grid.items():
        params[k] = v[random.randrange(len(v))]

      model.set_params(**params)

      acc = np.zeros(len(folds))

      for fold_id, (train_idx, test_idx) in enumerate(folds):
        
        X_train, y_train = X.values[train_idx], y.values[train_idx]
        X_test, y_test = X.values[test_idx], y.values[test_idx]

        if scale_features:
          X_train, X_test = self.scale(X_train, X_test)

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        TN = confusion_matrix(y_test, predictions)[0][0]
        FP = confusion_matrix(y_test, predictions)[0][1]
        FN = confusion_matrix(y_test, predictions)[1][0]
        TP = confusion_matrix(y_test, predictions)[1][1]
        total = TN + FP + FN + TP
        fold_acc = (TP + TN) / float(total) * 100.0

        acc[fold_id] = fold_acc

      if np.mean(acc) > best_acc:
        best_params = params
        best_acc = np.mean(acc)

    if scale_features:
      print ("The {} model w/ the following hyperparams: {} achieved the highest cv accuracy of {}% on the testing set w scaling".format(model_name, str(best_params), round(best_acc,2)))
    else:
      print ("The {} model w/ the following hyperparams: {} achieved the highest cv accuracy of {}% on the testing set w/o scaling".format(model_name, str(best_params), round(best_acc,2)))

    return best_params


  def gsModelsParamskFoldEvaluate(self, df, target_feature, scale_features=True, folds=10, trials=10):
    X, y, folds = self.getKFolds(df, target_feature, k=folds)

    best_acc = 0
    best_model = 0
    best_params = 0

    start_time = time.time()
    
    for n_trials in range(trials):

      if n_trials % 10 == 0:
        print 'Progress (',n_trials,'/',trials,'): ',np.float64(n_trials)/trials*100.0,'%'
        print ' RunTime: ',round((time.time() - start_time)/60,2), 'mins'      

      model_name = self.models.keys()[random.randrange(len(self.models))]
      
      model = self.models[model_name]
      current_param_grid = self.param_grids[model_name]

      params = {}
      for k, v in current_param_grid.items():
        params[k] = v[random.randrange(len(v))]

      model.set_params(**params)

      acc = np.zeros(len(folds))

      for fold_id, (train_idx, test_idx) in enumerate(folds):
        X_train, y_train = X.values[train_idx], y.values[train_idx]
        X_test, y_test = X.values[test_idx], y.values[test_idx]

        if scale_features:
          X_train, X_test = self.scale(X_train, X_test)

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        TN = confusion_matrix(y_test, predictions)[0][0]
        FP = confusion_matrix(y_test, predictions)[0][1]
        FN = confusion_matrix(y_test, predictions)[1][0]
        TP = confusion_matrix(y_test, predictions)[1][1]
        total = TN + FP + FN + TP
        fold_acc = (TP + TN) / float(total) * 100.0

        acc[fold_id] = fold_acc

      if np.mean(acc) > best_acc:
        model = model_name
        best_params = params
        best_acc = np.mean(acc)

    if scale_features:
      print ("The {} model w/ the following hyperparams: {} achieved the highest cv accuracy of {}% on the testing set w scaling".format(best_model, str(best_params), round(best_acc,2)))
    else:
      print ("The {} model w/ the following hyperparams: {} achieved the highest cv accuracy of {}% on the testing set w/o scaling".format(best_model, str(best_params), round(best_acc,2)))

    return best_model, best_params

  def gsFeaturesModelsParamskFoldEvaluate(self, df, target_feature, scale_features=True, folds=3, trials=10):
    X, y, folds = self.getKFolds(df, target_feature, k=folds)
    all_feature_sets = self.generateFeatureSets(df, target_feature)

    best_acc = 0.0
    best_model = ''
    best_params = ''
    best_features = ''

    start_time = time.time()

    for n_trials in range(trials):

      if n_trials % 10 == 0:
        print 'Progress (',n_trials,'/',trials,'): ',np.float64(n_trials)/trials*100.0,'%'
        print ' RunTime: ',round((time.time() - start_time)/60,2), 'mins'

      model_name = self.models.keys()[random.randrange(len(self.models))]
      feature_set = all_feature_sets[random.randrange(len(all_feature_sets))]

      current_param_grid = self.param_grids[model_name]

      model = self.models[model_name]

      params = {}
      for k, v in current_param_grid.items():
        params[k] = v[random.randrange(len(v))]

      model.set_params(**params)
      acc = np.zeros(len(folds))

      X_temp = X[list(feature_set)]

      for fold_id, (train_idx, test_idx) in enumerate(folds):
        X_train, y_train = X_temp.values[train_idx], y.values[train_idx]
        X_test, y_test = X_temp.values[test_idx], y.values[test_idx]

        if scale_features:
          X_train, X_test = self.scale(X_train, X_test)

        try:
          model.fit(X_train, y_train)
        except ValueError as e:
          print 'Fit Error: ', e
          continue

        predictions = model.predict(X_test)

        TN = confusion_matrix(y_test, predictions)[0][0]
        FP = confusion_matrix(y_test, predictions)[0][1]
        FN = confusion_matrix(y_test, predictions)[1][0]
        TP = confusion_matrix(y_test, predictions)[1][1]
        total = TN + FP + FN + TP
        fold_acc = (TP + TN) / float(total) * 100.0

        acc[fold_id] = fold_acc

      if np.mean(acc) > best_acc:
        best_model = model_name
        best_params = params
        best_features = feature_set
        best_acc = np.mean(acc)
        print 'New best model accuracy: ',round(np.mean(acc),2),'%'

    if scale_features:
      print ("The {} model w/ the following hyperparams: {} and features: {} achieved the highest cv accuracy of {}% on the testing set w scaling".format(best_model, str(best_params), str(best_features), round(best_acc,2)))
    else:
      print ("The {} model w/ the following hyperparams: {} and features: {} achieved the highest cv accuracy of {}% on the testing set w/o scaling".format(best_model, str(best_params), str(best_features), round(best_acc,2)))

    return best_model, best_params, best_features

  def generateModel(self, df, target_feature, model, scale_features=True, params=None, features=None):
    X, y = self.getXY(df, target_feature)

    model = self.models[model]

    if params is not None:
      model.set_params(**params)

    if features is not None:
      X = X[list(features)]

    if scale_features:
      X = self.scaler.fit_transform(X)
      self.best_scaler = self.scaler    

    model.fit(X,y)

    self.best_model = model

  def multiPredict(self, df, target_feature, features=None):

    X_test = df
    y_test = None
    
    if isinstance(df, pd.DataFrame) and target_feature in df.columns.values:
      X_test, y_test = self.getXY(df, target_feature)
    elif isinstance(df, pd.Series) and target_feature in df.index.values:
      X_test = df.drop(target_feature)
      y_test = df[target_feature]

    if features is not None:
      X_test = X_test[list(features)]

    if self.best_scaler is not None:
      try:
        X_test = self.best_scaler.transform(X_test)
      except ValueError as e:
        X_test = X_test.values.reshape(1,-1)
        X_test = self.best_scaler.transform(X_test)

    predictions = self.best_model.predict(X_test)
    
    if y_test is not None:
      TN = confusion_matrix(y_test, predictions)[0][0]
      FP = confusion_matrix(y_test, predictions)[0][1]
      FN = confusion_matrix(y_test, predictions)[1][0]
      TP = confusion_matrix(y_test, predictions)[1][1]
      
      total = TN + FP + FN + TP
      acc = (TP + TN) / float(total) * 100.0

      print acc

    if isinstance(df, pd.DataFrame)
      df['predicted_' + target_feature] = pd.Series(predictions)
    else:
      df.

    return df

  def predict(self, series, target_feature, features=None):
    X_test = series
    y_test = None

    if target_feature in series.index.values:
      X_test = series.drop(target_feature)
      y_test = series[target_feature]

    if features is not None:
      X_test = X_test[list(features)]

    X_test = X_test.values.reshape(1,-1)

    if self.best_scaler is not None:
      X_test = self.best_scaler.transform(X_test)
      
    prediction = self.best_model.predict(X_test)

    print series, ':', target_feature, ':', prediction[0]







