#!/usr/bin/python2.7

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

class DSTools(object):

  scaler = StandardScaler()

  def __init__(self):
    print 'Data Science Tools Ready'  

  def fillNaNwValue(self, df, feature, value):
    return df.set_value(df[feature].isnull(), feature, value)

  def dropFeature(self, df, feature):
    return df.drop(feature, axis=1, inplace=True)

  def checkForNaNs(self, df):

    nanDF = pd.DataFrame(columns=df.columns.values)
    
    for column in df.columns.values:
      nanDF = pd.concat([nanDF,df[df[column].isnull()]], axis=0)

    print "NaN Columns: ", len(nanDF)

  def hotEncode(self, df, feature):
    dummies = pd.get_dummies(df[feature], prefix=feature)
    df = pd.concat([df, dummies], axis=1)
    return df.drop([feature, dummies.columns.values[0]], axis=1)