from __future__ import print_function

from time import time
import numpy as np
import pandas as pd
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

# read in data
auto_data = pd.read_csv('auto-mpg.csv')
# clean up and fill in missing data points
auto_data.horsepower = auto_data.horsepower.fillna(value=auto_data.horsepower.mean())
# create the tensorboard framework
tensor_board = TensorBoard(log_dir="linear_logs/{}".format(time()))
# split up the data
x_train, x_valid, y_train, y_valid = train_test_split(auto_data.iloc[:,0:5], auto_data.iloc[:,5],test_size=0.2, random_state=87)
np.random.seed(816)
# create the model
model = Sequential()
model.add(Dense(1, input_dim=5, init='normal', activation="linear"))
model.add(Dense(1, init='normal'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=["accuracy"])
# train and test the model
model.summary()
epochs = 6000
batch_size =10
history = model.fit(x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    shuffle=True,
    verbose=2,
    validation_data=(x_valid, y_valid), callbacks=[tensor_board])
score = model.evaluate(x_valid, y_valid)
print("test accuracy", score[1])

