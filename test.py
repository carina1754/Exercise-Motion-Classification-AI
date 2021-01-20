import pandas as pd
import numpy as np
import math
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
train=pd.read_csv('train_features.csv')
train_labels=pd.read_csv('train_labels.csv')
test=pd.read_csv('test_features.csv')
submission=pd.read_csv('sample_submission.csv')

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM

X=tf.reshape(np.array(train.iloc[:,2:]),[-1, 600, 6])
y = tf.keras.utils.to_categorical(train_labels['label']) 
#가벼운 모델 생성
model = Sequential()
model.add(LSTM(32, input_shape=(600,6)))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(61, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X,y, epochs=100, batch_size=128, validation_split=0.2)

fig, ax = plt.subplots()
ax.plot(history.history['loss'], 'b', label = 'loss')
ax.plot(history.history['val_loss'], 'r', label='val_loss')
ax.set_xlabel('epoch')
ax.set_ylabel('loss')
ax.legend(loc='upper left')

plt.show()

test_X=tf.reshape(np.array(test.iloc[:,2:]),[-1, 600, 6])
test_X.shape

prediction=model.predict(test_X)

submission.iloc[:,1:]=prediction

submission.to_csv('submission.csv', index=False)