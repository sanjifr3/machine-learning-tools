#!/usr/bin/python

import pandas as pd

from libraries.DataScienceTools import DSTools
from libraries.MachineLearningTools import MLTools
from libraries.DeepLearningTools import DLTools

num_trials=10

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

if __name__ == "__main__":

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

  dsTools = DSTools()
  mlTools = MLTools()

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

  # D
  #mlTools.trainTestSplitEvaluate(trainDF, 'Survived')
  #mlTools.trainTestSplitEvaluate(trainDF, 'Survived', 'knn')
  #mlTools.trainTestSplitEvaluate(trainDF, 'Survived', 'knn', scale_features=False)
  #mlTools.trainTestSplitEvaluate(trainDF, 'Survived', 'knn', scale_features=False, params={'n_neighbors':7})

  #mlTools.kFoldEvaluate(trainDF, 'Survived')
  #mlTools.kFoldEvaluate(trainDF, 'Survived', 'knn')

  #mlTools.gridSearchSplitEvaluate(trainDF, 'Survived')
  #mlTools.gridSearchkFoldEvaluate(trainDF, 'Survived')

  

  mlTools.fullGS(trainDF, 'Survived', trials=50)






  