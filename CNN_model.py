
# coding: utf-8

import numpy as np
import pandas as pd 
import matplotlib.image as mpimg
from PIL import Image
import os
from os import listdir
from PIL import Image
from sklearn.model_selection import train_test_split


from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,AveragePooling2D,Flatten,Dense,Dropout,Activation
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adam, SGD

from keras.models import Model


with np.load('xtest_mod.npz') as data1:
    xtest = data1['arr_0']
    
with np.load('xtrain_mod.npz') as data2:
    xtrain = data2['arr_0']


y1 = pd.read_csv('ytrain.csv',header = None)
# y_train = y1.values.tolist()

y2 = pd.read_csv('ytest.csv',header = None)
# y_test = y2.values.tolist()


# In[5]:


y_train = pd.get_dummies(y1[1])
y_test = pd.get_dummies(y2[1])

ytrain = y_train.values
ytest = y_test.values

# print(type(ytrain))
# print(type(ytest))


# In[8]:


activation = 'relu'
kernel_regularizer = l2(1e-5)
dropout_conv = 0.25 #Fraction of units to DROP, i.e. set to 0. for no dropout
dropout_mlp = 0.5 #Fraction of units to DROP, i.e. set to 0. for no dropout

layer_names = ['conv1','pool1','conv2','pool2']
i = 0
model = Sequential()

model.add(Conv2D(64, (3, 3), padding='same', input_shape=(350,350,1)))
# model.add(BatchNormalization(axis=3)) 
model.add(Activation(activation, name=layer_names[i]))
i+=1

model.add(MaxPooling2D(pool_size=(2, 2), name=layer_names[i]))
i+=1
# model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), padding='same'))
# model.add(BatchNormalization(axis=3))
model.add(Activation(activation, name=layer_names[i]))
i+=1

model.add(MaxPooling2D(pool_size=(2, 2), name=layer_names[i]))
i+=1
# model.add(Dropout(0.25))

# model.add(Conv2D(64, (3, 3), padding='same'))
# model.add(BatchNormalization(axis=3))
# model.add(Activation(activation, name=layer_names[i]))
# i+=1

# model.add(Conv2D(64, (3, 3), padding='same'))
# model.add(BatchNormalization(axis=3))
# model.add(Activation(activation, name=layer_names[i]))
# i+=1

# model.add(MaxPooling2D(pool_size=(2, 2), name=layer_names[i]))
# i+=1
# model.add(Dropout(dropout_conv))

model.add(Flatten())
model.add(Dense(200, activation=activation, kernel_regularizer=kernel_regularizer))
model.add(Dropout(0.5))
model.add(Dense(8, activation='softmax', kernel_regularizer=kernel_regularizer))

print(model.summary())
# plot_model(model, to_file='emotion_model.png', show_shapes=True, show_layer_names=False)
# SVG(model_to_dot(model).create(prog='dot', format='svg'))


# In[9]:


# Train and test
optimizer = Adam()
loss = 'categorical_crossentropy'
metrics = ['accuracy']
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

batch_size = 32
epochs = 10
history = model.fit(xtrain,ytrain, batch_size=batch_size, epochs=epochs)

score = model.evaluate(xtest,ytest, batch_size=batch_size)
print('Test accuracy = {0}'.format(100*score[1]))

model.save('emotion_model_trained10ep.h5')

