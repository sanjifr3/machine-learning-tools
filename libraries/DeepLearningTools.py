#!/usr/bin/python2.7

import os
import numpy as np
import pandas as pd
import cv2
import time
from numpy import random

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential, Model, load_model
from keras.layers import Input, GlobalAveragePooling2D, BatchNormalization
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Conv2D
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import SGD, Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10, mnist
from keras.utils import np_utils
from keras import backend

class DLTools(object):


  val_size = 0.1
  bias = 1.0
  default_input_size = (128,128)
  num_classes = 1

  obj_path = os.environ['HOME'] + '/Database/objects'
  model_path = 'models/keras'

  random_state = 101
  verbose = True
  use_preset_vals = False
  
  possible_classes = ['head','microwave','tv','laptop','sink','dish','light','lamp',
                        'newspaper','food','oven','book']

  options = {
    'input_size':[40,50,75,90,128],
    'batch_size':[32,64,96],
    'dense_units':[32,64,128,256,512],
    'fourth_layer':[True, False],
    'fifth_layer':[True,False],
    'optimizer':['adam','sgd','rmsprop'],
    'optimizer_params':{
      'lr':[0.001,0.01,0.1],
      'rho':[0.7,0.8,0.9],
      'epsilon':[None, 1e-08, 1e-07, 1e-06, 1e-05, 1e-04, 1e-03],
      'decay':[0, 0.0001, 0.00025, 0.0005, 1e-6, 1e-5, 1e-4],
      'beta_1':[0.85,0.9,0.95],
      'beta_2':[0.951, 0.96, 0.97, 0.98, 0.99],
      'momentum':[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
      'nesterov':[True,False],
      'amsgrad':[True,False]
    }
  }

  result_keys = options.keys()
  result_keys.remove('optimizer_params')
  result_keys += ['num_folds']
  result_keys += options['optimizer_params'].keys()
  result_keys += ['cv_acc_mean','cv_acc_std','cv_loss_mean','cv_loss_std',
                  'valid_acc','valid_loss']


  min_size = 3000 # for labelling images

  def __init__(self, classes, channels=3, dtype=None):

    self.channels = channels

    if classes is not None:
      self.classes = []

      for class_name in classes.split(','):
        if class_name in self.possible_classes:
          self.classes.append(class_name)
        else:
          print 'ERROR:', class_name, 'not in', self.possible_classes

    self.folder_name = '_'.join(sorted(self.classes))

    if dtype is not None:
      self.dtype = dtype
    elif channels != 3:
      self.dtype = 'uint16'
    else:
      self.dtype = 'uint8'

    print 'Deep Learning Tools Ready'

  def loadData(self):
    df = pd.DataFrame()

    for class_name in self.classes:
      try:
        tempDF = pd.read_csv('%s/%s.csv' % (self.obj_path, self.folder_name))
        tempDF['path'] = self.obj_path + '/' + class_name + '/' + tempDF['image']
        df = pd.concat([df,tempDF], axis=0)
      except IOError:
        print "Error: %s/%s.csv doesn't exist" % (self.obj_path, class_name)

    return df

  def splitData(self, df):
    X = []
    y = []

    sets = {label:[] for label in list('0123456')}

    for i, row in df.iterrows():
      sets[str(row['state'])].append(row['path'])

    if self.bias != 0.0:
      min_images = 1000000
      for k,st in sets.items():
        if len(st) > 0 and len(st) < min_images:
          min_images = len(st)

      for k,st in sets.items():
        if len(st) > 0:
          random.shuffle(st)
          if len(st) != min_images:
            sets[k] = st[0:int(min_images*self.bias)]
        
    for k,st in sets.items():
      print k, len(st)
      for im_file in st:
        if self.dtype != 'uint8': im = cv2.imread(im_file,-1)
        elif self.channels == 1: im = cv2.imread(im_file, 0)
        else: im = cv2.imread(im_file)

        im = cv2.resize(im, (self.default_input_size[0],self.default_input_size[1]))
        X.append(im)
        y.append(int(k))

    X = np.array(X)
    self.dtype = X.dtype
    X = X.astype('float32')
    if len(X.shape) == 3:
      X = X.reshape(X.shape[0],X.shape[1],X.shape[2],1)
    y = np.array(y).reshape(X.shape[0],1)

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=self.val_size, random_state=self.random_state, stratify=y)

    if self.num_classes > 1:
      Y_train = np_utils.to_categorical(Y_train, nb_classes)
      Y_test = np_utils.to_categorical(Y_test, nb_classes)

    return X_train, X_test, y_train, y_test

  def getRandomParams(self):
    params = {}

    for param,val in self.options.items():
      if param != 'optimizer_params':
        params[param] = np.random.choice(val)

    if self.use_preset_vals:
      params['dense_units'] = 32
      params['optimizer'] = 'rmsprop'
      params['batch_size'] = 64
      params['fourth_layer'] = False
      params['input_size'] = 128

    params['optimizer_params'] = {}
    for param, val in self.options['optimizer_params'].items():
      if ((param in ['lr','decay']) or 
         (params['optimizer'] == 'adam' and param in ['beta_1','beta_2','epsilon','amsgrad']) or
         (params['optimizer'] == 'sgd' and param in ['momentum','nesterov']) or
         (params['optimizer'] == 'rmsprop' and param in ['rho','epsilon'])):

          params['optimizer_params'][param] = np.random.choice(val)

          if params['optimizer'] == 'adam' and param == 'lr':
            while params['optimizer_params'][param] >= 0.1:
              params['optimizer_params'][param] = np.random.choice(val)

    ## Random params
    params['dropout_01'] = random.uniform(0,1)
    params['dropout_02'] = random.uniform(0,1)
    params['dropout_03'] = random.uniform(0,1)
    params['dropout_04'] = random.uniform(0,1)
    params['dropout_05'] = random.uniform(0,1)

    if not params['fourth_layer']:
      params['fifth_layer'] = False

    return params

  def getOptimizer(self,params):
    optim = None
    p_ = params['optimizer_params']
    if params['optimizer'] == 'adam':
      optim = Adam(lr=p_['lr'], beta_1=p_['beta_1'], beta_2=p_['beta_2'], 
        epsilon=p_['epsilon'], decay=p_['decay'], amsgrad=p_['amsgrad'])
    elif params['optimizer'] == 'sgd':
      optim = SGD(lr=p_['lr'], decay=p_['decay'], momentum=p_['momentum'], nesterov=p_['nesterov'])
    elif params['optimizer'] == 'rmsprop':
      optim = RMSprop(lr=p_['lr'], decay=p_['decay'], rho=p_['rho'], epsilon=p_['epsilon'])

    return optim

  def getModel(self, params, optim):
    model = Sequential()
    model.add(BatchNormalization(axis=3, input_shape=(params['input_size'], params['input_size'], self.channels)))
    model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(params['dropout_01']))

    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(params['dropout_02']))

    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(params['dropout_03']))

    if params['fourth_layer']:
      model.add(BatchNormalization(axis=3))
      model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
      model.add(MaxPooling2D(pool_size=(2, 2)))
      model.add(Dropout(params['dropout_04']))

      if params['fifth_layer']:
        model.add(BatchNormalization(axis=3))
        model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(params['dropout_05']))

    model.add(Flatten())
    model.add(Dense(params['dense_units'], activation='relu'))
    model.add(Dropout(0.5))
    if self.num_classes == 1:
      model.add(Dense(self.num_classes, activation='sigmoid'))
      model.compile(loss='binary_crossentropy', metrics=['accuracy'],optimizer=optim)
    else:
      model.add(Dense(self.num_classes, activation='softmax'))
      model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer=optim)

    return model

  def train(self, epochs=50, trials=10, k=None):
    df = self.loadData()
    full_X_train, full_X_valid, Y_train, Y_valid = self.splitData(df)

    print 'Training set shape:', full_X_train.shape
    print 'Labels shape:', Y_train.shape
    print full_X_train.shape[0], 'training samples'
    print full_X_valid.shape[0], 'validation samples'

    datagen = ImageDataGenerator(
      shear_range=0.3,
      zoom_range=0.3,
      rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
      width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
      height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
      horizontal_flip=True,  # randomly flip images
      vertical_flip=True  # randomly flip images
    )

    if k is not None:
      folds = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=101).split(full_X_train,Y_train))
    else:
      folds = []

    start_time = time.time()
    best_acc = 0.0
    best_loss = 0.0

    model_name = self.folder_name
    if k is not None:
      model_name += '_' + str(k) + 'fold'
    
    try:
      resultsDF = pd.read_csv('%s/%s.csv' % (self.model_path, model_name), sep=',')
    except IOError:
      resultsDF = pd.DataFrame(columns=self.result_keys)
    idx = len(resultsDF)

    for n_trial in range(trials):
      print '\nTrial:', n_trial

      # Generate random params
      params = self.getRandomParams()

      # Generate optimizer
      optim = self.getOptimizer(params)

      # Reshape Inputs
      X_train = []
      X_valid = []
      for i in range(full_X_train.shape[0]):
        X_train.append(cv2.resize(full_X_train[i], (params['input_size'],params['input_size'])))

      for i in range(full_X_valid.shape[0]):
        X_valid.append(cv2.resize(full_X_valid[i], (params['input_size'],params['input_size'])))

      # Normalize by max pixel value
      if self.dtype == 'uint8': max_val = 2**8-1
      elif self.dtype == 'uint16': max_val = 2**16-1
      else:
        print ('ERROR: Unknown datatype',dtype)
        exit(1)

      X_train = np.array(X_train) / max_val
      X_valid = np.array(X_valid) / max_val

      cv_acc = 0.0
      cv_loss = 0.0
      
      if k is not None:

        cv_acc = np.zeros(k)
        cv_loss = np.zeros(k)

        # K-Fold CV
        for fold_id, (train_idx, test_idx) in enumerate(folds):
          print '\nFolds:',fold_id

          x_train = X_train[train_idx]
          y_train = Y_train[train_idx]

          x_test = X_train[test_idx]
          y_test = Y_train[test_idx]

          datagen.fit(x_train)

          model = self.getModel(params, optim)

          model.fit_generator(
            datagen.flow(x_train, y_train,batch_size=params['batch_size']),
            steps_per_epoch=x_train.shape[0]/params['batch_size'],
            epochs=epochs,
            validation_data=(x_test, y_test),
            validation_steps=x_test.shape[0]/params['batch_size'],
            verbose=self.verbose
          )

          cv_loss[fold_id], cv_acc[fold_id] = model.evaluate(x_test, y_test, verbose=self.verbose)

      # Track history
      history = LossHistory()

      # Train on entire dataset
      datagen.fit(X_train)
    
      model = self.getModel(params, optim)

      model.fit_generator(
        datagen.flow(X_train, Y_train,batch_size=params['batch_size']),
        steps_per_epoch=X_train.shape[0]/params['batch_size'],
        epochs=epochs,
        validation_data=(X_valid, Y_valid),
        validation_steps=X_valid.shape[0]/params['batch_size'],
        callbacks=[history],
        verbose=self.verbose
      )

      valid_loss, valid_acc = model.evaluate(X_valid, Y_valid, verbose=self.verbose)

      if k is None:
        cv_loss, cv_acc = model.evaluate(X_train, Y_train, verbose=self.verbose)
        print 'Training acc: %.2f%%' % (cv_acc)
        print 'Training loss: %.2f' % (cv_loss)
      else:
        print '%d fold CV acc: %.2f%% (%.2f)' % (k, cv_acc.mean(), cv_acc.std())
        print '%d fold CV loss: %.2f (%.2f)' % (k, cv_loss.mean(), cv_loss.std())
      print 'Validation acc: %.2f%%' % (valid_acc)
      print 'Validation loss: %.2f' % (valid_loss)

      if len(resultsDF) > 0:
        if k is not None:
          best_acc = resultsDF[resultsDF['num_folds'] == k]['cv_acc_mean'].max()
        else:
          best_acc = resultsDF[resultsDF['num_folds'] == 0]['cv_acc_mean'].max()

      update_model = False      
      if k is not None and cv_acc.mean() > best_acc:
        best_acc = cv_acc.mean()
        best_loss = cv_loss.mean()
        update_model = True
        print 'New best %s %d fold CV acc: %.2f%%' % (self.folder_name, k, best_acc)
        
      elif k is None and cv_acc > best_acc:
        best_acc = cv_acc
        best_loss = cv_loss
        update_model = True
        print 'New best %s training acc: %.2f%%' % (self.folder_name, best_acc)

      if update_model: 
        self.best_model = model
        self.default_input_size = (params['input_size'],params['input_size'])

        model.save('%s/%s.h5' % (self.model_path, model_name))
        with open('%s/%s.log' % (self.model_path, model_name), 'w+') as f:
          f.write(history.losses)

      for param, val in params.items():
        if param == 'optimizer_params':
          for param, val in params['optimizer_params'].items():
            resultsDF.loc[idx,param] = params['optimizer_params'][param]
        else:
          resultsDF.loc[idx,param] = params[param] 

      if k is not None:
        resultsDF.loc[idx,'num_folds'] = k
        resultsDF.loc[idx,'cv_acc_mean'] = cv_acc.mean()
        resultsDF.loc[idx,'cv_acc_std'] = cv_acc.std()
        resultsDF.loc[idx,'cv_loss_mean'] = cv_loss.mean()
        resultsDF.loc[idx,'cv_loss_std'] = cv_loss.std()
      else:
        resultsDF.loc[idx, 'num_folds'] = 0
        resultsDF.loc[idx,'cv_acc_mean'] = cv_acc
        resultsDF.loc[idx,'cv_acc_std'] = 0
        resultsDF.loc[idx,'cv_loss_mean'] = cv_loss
        resultsDF.loc[idx,'cv_loss_std'] = 0
    
      resultsDF.loc[idx,'valid_acc'] = valid_acc
      resultsDF.loc[idx,'valid_loss'] = valid_loss

      resultsDF.to_csv('%s/%s.csv' % (self.model_path, model_name), sep=',', index=False)

      idx+=1

  def loadModel(self, k=None):

    model_name = self.folder_name
    if k is not None:
      model_name += '_' + str(k) + 'fold'

    try:
      self.best_model = load_model('%s/%s.h5' %(self.model_path, model_name))
    except:
      print 'Could not load model'
      return

    df = pd.read_csv('%s/%s.csv' %(self.model_path, model_name), sep=',')
    self.default_input_size = int(df[df['cv_acc_mean'] == df['cv_acc_mean'].max()]['input_size'].values[0])
    self.default_input_size = (self.default_input_size, self.default_input_size)

    print 'Model loaded!'

  def classify(self, im):
    im = cv2.resize(im, self.default_input_size)
    im = np.array(im).astype('float32')

    if len(im.shape) == 3:
      im = im.reshape(1,im.shape[0],im.shape[1],im.shape[2]) / (2**8-1)
    else:
      im = im.reshape(1,im.shape[0],im.shape[1],1) / (2**16-1)

    prediction = self.best_model.predict(im)

    print 'Prediction:', prediction

  def classifyFromFile(self, im_path):
    if self.dtype == 'uint16':
      im = cv2.imread(im_path, -1)
    else:
      im = cv2.imread(im_path)

    self.classify(im)

  def labelImgs(self):
    images = os.listdir('%s/%s' % (self.obj_path, self.folder_name))
    random.shuffle(images)

    df = pd.DataFrame()

    try:
      df = pd.read_csv('%s/%s.csv' % (self.obj_path, self.folder_name), sep=',')
    except IOError as e:
      df = pd.DataFrame(columns=['image','state'])

    print 'press 0 for apply state 0'
    print 'press 1 for apply state 1'
    print 'press 2 for apply state 2'
    print 'press z or 7 to delete last applied label'
    print 'press . to skip image'
    print 'press q or 9 to quit'

    i = 0
    while i < len(images):
      im_file = images[i]
      if im_file not in df['image']:
        print '%s/%s/%s' % (self.obj_path, self.folder_name, im_file)
        im = cv2.imread('%s/%s/%s' % (self.obj_path, self.folder_name, im_file))

        if im is None:
          print 'Invalid image file: ', im_file
          i += 1
          continue

        im_size = im.shape[0] * im.shape[1]

        if im_size < self.min_size:
          print 'Invalid image size: ', im_size , '<', self.min_size
          i += 1
          continue

        cv2.imshow('im',im)

        key = cv2.waitKey(0) & 0xFF
        state = -1

        if key == ord('q') or key == ord('9'):
          break
        elif key == ord('0'):
          state = 0
        elif key == ord('1'):
          state = 1
        elif key == ord('2'):
          state = 2
        elif key == ord('.'):
          i+=1
          print 'skip'
          continue
        elif key == ord('z') or key == ord('7'):
          if len(df) > 0:
            df.drop(df.index[len(df)-1],inplace=True)
            print 'delete last label'
            i-=1
          continue
        else:
          continue

        print im_file, state
        print 'Labelled:', len(df), '/', len(images), 'images'
        df = df.append({'image':im_file,'state':state}, ignore_index=True)
        df.sort_values('image',inplace=True)
        df.to_csv('%s/%s.csv'%(self.obj_path,self.folder_name),sep=',',index=False)
        i+=1

  def getClasses(self, file):
    classes = []
    with open(file, 'r') as f:
      for line in f:
        classes.append(line.strip('\n').split())

class LossHistory(Callback):
  def __init__(self):
    #super().__init__()
    self.epoch_id = 0
    self.losses = ''
 
  def on_epoch_end(self, epoch, logs={}):
    self.losses += "Epoch {}: accuracy -> {:.4f}, val_accuracy -> {:.4f}\n"\
        .format(str(self.epoch_id), logs.get('acc'), logs.get('val_acc'))
    self.epoch_id += 1
 
  def on_train_begin(self, logs={}):
    self.losses += 'Training begins...\n'