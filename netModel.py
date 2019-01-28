import matplotlib
matplotlib.use('Agg')
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
import matplotlib.pyplot as plt
from keras.utils import plot_model

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
history = model.fit(X, y, validation_split = 0.3, epochs = 200, batch_size = 128, verbose = 1)

# save and plot model
model.save('model')
plot_model(model, to_file='model.png')
# Plot training & validation accuracy values
plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('accuracy.png')

# Plot training & validation loss values
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('Model-loss.png')

# return to original directory
os.chdir(originalDir)