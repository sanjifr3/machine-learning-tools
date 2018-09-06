#!/usr/bin/python

import pandas as pd
from random import shuffle

from libraries.DataScienceTools import DSTools
from libraries.DeepLearningTools import DLTools

num_trials=10

if __name__ == "__main__":

  trainDF = ''
  testDF = ''
  target_feature = ''

  dsTools = DSTools()
  dlTools = DLTools('tv')

  print '\nLoad model'
  dlTools.loadModel(k=3)

  # Train model
  print '\nTrain to find better model'
  dlTools.train(epochs=3, trials=3, k=3)

  # Load model and classify image from file
  print '\n Classify images'
  dlTools.classifyFromFile('/home/sanjif/Database/objects/tv/tv2622.jpg')
  dlTools.classifyFromFile('/home/sanjif/Database/objects/tv/tv2626.jpg')

  # Label new images for classification 
  #dlTools.labelImgs()