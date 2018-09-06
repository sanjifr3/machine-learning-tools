#!/usr/bin/python

import pandas as pd
from random import shuffle

from libraries.DataScienceTools import DSTools
from libraries.MachineLearningTools import MLTools

num_trials=10
#data_set = 'abalone'
data_set = 'titanic'

titleDict  = {
    "Capt":         "Officer",
    "Col":          "Officer",
    "Major":        "Officer",
    "Jonkheer":     "Nobel",
    "Don":          "Nobel",
    "Sir" :         "Nobel",
    "Dr":           "Officer",
    "Rev":          "Officer",
    "the Countess": "Nobel",
    "Dona":         "Nobel",
    "Mme":          "Mrs",
    "Mlle":         "Miss",
    "Ms":           "Mrs",
    "Mr" :          "Mr",
    "Mrs" :         "Mrs",
    "Miss" :        "Miss",
    "Master" :      "Master",
    "Lady" :        "Nobel"
}    

def fixNaNAge(age, pClass):
  if age == age:
    return age
  if pClass == 1:
    return 38
  elif pClass == 2:
    return 30
  else:
    return 25

def getPerson(passenger):
  age, sex = passenger
  return 'child' if age < 16 else sex

def binTarget(x):
    if x <=  3: return 0
    if x <=  6: return 1 
    if x <=  9: return 2
    if x <= 12: return 3
    if x <= 15: return 4
    if x <= 18: return 5
    if x <= 21: return 6
    if x <= 24: return 7
    if x <= 27: return 8
    if x <= 30: return 9  

if __name__ == "__main__":

  trainDF = ''
  testDF = ''
  target_feature = ''

  assert(data_set == 'titanic' or data_set == 'abalone')

  dsTools = DSTools()
  mlTools = MLTools()

  if data_set == 'titanic':
    trainDF = pd.read_csv('data/Titanic_train.csv',sep=',')
    testDF = pd.read_csv('data/Titanic_test.csv',sep=',')

    ''' 
    # Data Cleaning #
    Replace Embarked nulls with most common city
    Replace Fare nulls with most common fare
    Replace Age nulls with most common age based on passenger class
    
    # Feature Engineering #
    Create feature containing people titles
    Create family size feature
    
    # Drop Redundant Features #
    Remove features used in feature engineering, and features that don't supply any
    useful information for machine learning
    '''
    for df in [trainDF, testDF]:
      # Data Cleaning
      dsTools.checkForNaNs(df)
      dsTools.fillNaNwValue(df, 'Embarked', 'C')
      dsTools.fillNaNwValue(df, 'Fare', 8.05)
      dsTools.dropFeature(df, 'Cabin')
      df['Age'] = df.apply(lambda row: fixNaNAge(row['Age'], row['Pclass']), axis=1)
      dsTools.checkForNaNs(df)

      # Feature Engineering
      df['Title'] = df['Name'].apply(lambda x: titleDict[x.split(',')[1].split('.')[0].strip()])
      df['FamilySize'] = df['SibSp'] + df['Parch']
      df['Person'] = df[['Age','Sex']].apply(getPerson, axis=1)

      # Drop redundant features
      features_to_drop = ['PassengerId','Name','Sex','Ticket','SibSp','Parch']
      dsTools.dropFeature(df, features_to_drop)

    # Convert categorical features
    trainDF = dsTools.hotEncode(trainDF, 'Person')
    trainDF = dsTools.hotEncode(trainDF, 'Embarked')
    trainDF = dsTools.hotEncode(trainDF, 'Title')

    testDF = dsTools.hotEncode(testDF, 'Person')
    testDF = dsTools.hotEncode(testDF, 'Embarked')
    testDF = dsTools.hotEncode(testDF, 'Title')

    target_feature = 'Survived'

  elif data_set == 'abalone':
    trainDF = pd.read_csv('data/abalone.data',sep=',', 
                    names=['sex','length','diameter','height',
                           'wholeWeight','shuckedWeight',
                           'visceraWeight','shellWeight','rings'])

    # Bin target variable 'rings' into bins to turn it into a classificaiton problem
    trainDF['target'] = trainDF['rings'].apply(binTarget)
    dsTools.dropFeature(trainDF, 'rings')

    # Check for missing values
    dsTools.checkForNaNs(trainDF)

    # Encode categorical variables
    trainDF = dsTools.hotEncode(trainDF, 'sex')

    # Create validation set by taking 10% of the data and storing it in a validation set
    trainDF = trainDF.sample(frac=1).reset_index(drop=True)

    val_size = .10
    val_samples = int(len(trainDF)*val_size)

    testDF = trainDF.ix[len(trainDF)-val_samples:].reset_index(drop=True)
    trainDF = trainDF.ix[:len(trainDF)-val_samples].reset_index(drop=True)

    target_feature = 'target'


  print '\nLoad model'
  mlTools.loadModel(data_set)

  print '\nExecute grid search of features, params, and model types with cross validation ' \
      + 'to generate optimal model on training set'
  model, params, features = mlTools.gsFeaturesModelsParamskFoldEvaluate(trainDF, target_feature, trials=num_trials)

  print '\nUse found params, features, and model_type to generate optimal model trained on full ' \
    + 'training set'
  mlTools.generateModel(trainDF, target_feature, model, params=params, features=features)

  print '\nMake prediction on single sample'
  series = testDF.ix[0].copy()
  mlTools.predict(series, target_feature)
  #mlTools.multiPredict(series, target_feature, features=features)

  print '\nMake predictions on entire validation set'
  mlTools.multiPredict(testDF, target_feature)

  print '\nSave updated model'
  mlTools.saveModel(data_set)
