#!/usr/bin/env python
# coding: utf-8

# In[86]:


# Predict next day's rainfall (yes/no) from historical measures

import os, numpy as np
import pandas as pd
homedir = os.getenv('HOME')


# read .csv: Data from a single station in county Kilkenny.

datapath = os.path.join(homedir, 'Dokumente','python-apps','tensorflow', 'irish_meteorological_service')
datafile = 'dailyrain.csv'

indatapath = os.path.join(datapath,datafile)


# In[87]:


df = pd.read_csv(indatapath, header=9, sep=',',usecols=[0,2], names=['date', 'rain_mm'],                dtype={'date': "string", 'rain_mm': np.float64})
df = df.loc[df['rain_mm'] != np.nan]
df2 = df.iloc[0:7875,:].copy()

fmtnum = lambda x: 0 if x == 0 else 1
df2['labelnum'] = df2['rain_mm'].map(fmtnum).astype(np.int)
df2['rain_mm'] = df2['rain_mm']/100
df3 = df2.drop(columns=['date'])
labels = df3["labelnum"].copy()
df3.drop(columns="labelnum", inplace=True)

a = df3.to_numpy(copy=True)
l = labels.to_numpy(copy=True)
a.resize(1125, 7, 1)
l.resize(1125, 7, 1)
labelL = []
for row in l:
    labelL.append(row[6])
labelsA = np.array(labelL)
labelsA.flatten()

print(a.shape)


# In[88]:


train = a[0:900, :]
test = a[900:, :]
trainl = labelsA[0:900, :]
testl  = labelsA[900:, :]


# In[89]:


'''
conv1d:

When using this layer as the first layer in a model, provide an input_shape argument 
(tuple of integers or None, e.g. (10, 128) for sequences of 10 vectors of 128-dimensional
vectors, or (None, 128) for variable-length sequences of 128-dimensional vectors

3D tensor with shape: (samples, steps, features)

'''
# Expected: 3D tensor with shape: (samples, time points, features) for conv1d

from tensorflow import keras

model =keras.models.Sequential()
# input shape: (None for variable sequence-length)
model.add(keras.layers.Conv1D(20, 1, activation='elu',input_shape=(7, 1)))
model.add(keras.layers.MaxPooling1D(1))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(25, activation='elu'))
model.add(keras.layers.Dense(1, activation = 'sigmoid'))
#model.summary()

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(train, trainl, epochs=4, batch_size=1)
test_loss, test_acc = model.evaluate(test, testl)
print('Accuracy of CONV1D:\n')
print(test_acc)


'''
GRU layer:
3D tensor with shape: (samples, steps, features)

'''

model =keras.models.Sequential()
# input shape: (None for variable sequence-length)
model.add(keras.layers.GRU(50, input_shape=(train.shape[1], train.shape[2])))
model.add(keras.layers.Dense(25, activation='elu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train, trainl, epochs=4, batch_size=1)
test_loss, test_acc = model.evaluate(test, testl)
print('Accuracy of GRU:\n')
print(test_acc)
