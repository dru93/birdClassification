'''
- Sequential neural network model
'''

import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adadelta

# keep original directory
originalDir = os.getcwd()

# extract features from wav files
if not os.path.exists(originalDir + '/Results/'):
    import featureExtraction

# Set working directory
path = 'Results/'
os.chdir(path)

# load features and labels
X = np.load('features10k.npy')
y = np.load('labels10k.npy')

lb = LabelEncoder()
y = np_utils.to_categorical(lb.fit_transform(y))

# split to train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 7)


num_labels = y.shape[1]

# build model
model = Sequential()

model.add(Dense(256, input_shape = (193,)))
model.add(Activation('softmax'))
model.add(Dropout(0.25))

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(num_labels))
model.add(Activation('softmax'))

model.compile(loss = 'categorical_crossentropy', metrics = ['accuracy'], optimizer = 'Adadelta')

# model training
model.fit(X_train, y_train, batch_size = 32, epochs = 20, validation_data = (X_test, y_test))

# return to original directory
os.chdir(originalDir)