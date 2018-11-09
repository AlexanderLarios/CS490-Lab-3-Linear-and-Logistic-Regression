from __future__ import print_function

from time import time
import numpy as np
import pandas as pd
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

auto_data = pd.read_csv('auto-mpg.csv')

print(auto_data.describe())
print(auto_data.columns[auto_data.isnull().any()])
auto_data.horsepower = auto_data.horsepower.fillna(value=auto_data.horsepower.mean())
print(auto_data.columns[auto_data.isnull().any()])

tensor_board = TensorBoard(log_dir="linear_logs/{}".format(time()))
x_train, x_valid, y_train, y_valid = train_test_split(auto_data.iloc[:,0:5], auto_data.iloc[:,5],test_size=0.2, random_state=87)
np.random.seed(816)
model = Sequential()
model.add(Dense(1, input_dim=5, init='normal'))
model.add(Dense(1, init='normal'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=["accuracy"])

model.summary()
epochs = 6000
batch_size =10
history = model.fit(x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    shuffle=True,
    verbose=2, # Change it to 2, if wished to observe execution
    validation_data=(x_valid, y_valid), callbacks=[tensor_board])
score = model.evaluate(x_valid, y_valid)
print("test accuracy", score[1])

