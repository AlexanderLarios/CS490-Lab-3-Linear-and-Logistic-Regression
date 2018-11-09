from __future__ import print_function
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import metrics
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam, RMSprop, Nadam
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.utils import plot_model
from keras.models import load_model
from sklearn.model_selection import train_test_split

df = pd.read_csv('OceanTempSalinity.csv')
oceanData = pd.DataFrame(df, columns=[
        'T_degC','Salnty'])
label_col = 'T_degC'
print(oceanData.describe())


x_train, x_valid, y_train, y_valid = train_test_split(oceanData.iloc[:,0:1], oceanData.iloc[:,1],test_size=0.3, random_state=87)
np.random.seed(816)

def the_model():
    model = Sequential()
    model.add(Dense(1, input_dim=1, init='normal', activation='relu'))
    model.add(Dense(1, init='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

model = the_model()
model.summary()
epochs = 200
batch_size =100
history = model.fit(x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    shuffle=True,
    verbose=2, # Change it to 2, if wished to observe execution
    validation_data=(x_valid, y_valid),)
score = model.evaluate(x_valid, y_valid)
print("test accuracy", score[1])

# Get training and test loss histories
training_loss = history.history['loss']
test_loss = history.history['val_loss']

# Create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)

# Visualize loss history
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, test_loss, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show();
