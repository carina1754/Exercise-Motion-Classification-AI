
import pandas as pd
import numpy as np
from time import time
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob

scaler = StandardScaler()

early_stopping = EarlyStopping()

train_features=pd.read_csv('train_features.csv')
train_labels=pd.read_csv('train_labels.csv')
test=pd.read_csv('test_features.csv')
submission=pd.read_csv('sample_submission.csv')

start = time()

x_train = []

for uid in tqdm(train_features['id'].unique()):
    temp = np.array(train_features[train_features['id'] == uid].iloc[:,2:], np.float32).T
    x_train.append(temp)

x_train = np.array(x_train, np.float32)
x_train = x_train[:,:,:,np.newaxis]

x_test = []

for uid in tqdm(test['id'].unique()):
    temp = np.array(test[test['id'] == uid].iloc[:,2:], np.float32).T
    x_test.append(temp)

x_test = np.array(x_test, np.float32)
x_test = x_test[:,:,:,np.newaxis]
def aug(data, uid, shift = 0):
    shift_data = np.roll(data, shift, axis=2)
    
for _ in range(10):
    aug(x_train, 0, int(random.random()*600))

y = tf.keras.utils.to_categorical(train_labels['label'])

#sgd = tf.keras.optimizers.SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)

from tensorflow.keras.layers import Dropout,Flatten
model = Sequential()
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(61, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train,y, epochs=100, batch_size=128, validation_split=0.2)

print(f'\nWhen hidden layers are 2, Elapse training time : {time() - start} seconds\n')
start = time()
loss_and_metrics = model.evaluate(x_train, y)
print(f'\nLoss : {loss_and_metrics[0]:.6}')
print(f'Accuracy : {loss_and_metrics[1]*100:.6}%')
print(f'When hidden layers are 2, Elapse test time : {time() - start} seconds')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['train_loss', 'val_loss'])
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train_accuracy', 'val_accuracy'])
plt.show()

prediction=model.predict(x_test)
submission.iloc[:,1:]=prediction
submission.to_csv('submission.csv', index=False)