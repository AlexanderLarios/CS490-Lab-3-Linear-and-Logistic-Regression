import pandas
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import tensorflow
import tensorboard

titanic_data = pandas.read_excel('titanic3.xls', 'titanic3')
x_train, x_test, y_train, y_test = train_test_split(titanic_data.iloc[:,0:6], titanic_data.iloc[:,6], test_size=0.3, random_state=87)

sex_label = LabelEncoder()

# clean up the training data
filler_age = x_train.age.median()
x_train.age = x_train.age.fillna(value=filler_age)
x_train.sex = sex_label.fit_transform(x_train.sex)
x_train.fare = x_train.fare.fillna(value=x_train.fare.median())

# clean up testing data
filler_age_test = x_test.age.median()
x_test.age = x_test.age.fillna(value=filler_age_test)
filler_fare_test = x_test.fare.median()
x_test.sex = sex_label.fit_transform(x_test.sex)

kModel = Sequential()
kModel.add(Dense(1, input_dim=6, activation='sigmoid'))
kModel.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy"])
print(kModel.summary())
fit_results = kModel.fit(x_train, y_train, batch_size=10, nb_epoch=500, validation_data=(x_test, y_test))
score=kModel.evaluate(x_test, y_test)

print("test accuracy", score[1])


