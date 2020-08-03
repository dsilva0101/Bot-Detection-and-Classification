#RNA to classify twitter accounts as bot or not bot. 
#By Gabrieli Silva
import matplotlib.pyplot as plt
import tensorflow as tf 
import pandas as pd 
import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split, KFold 
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn import metrics
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D,AveragePooling2D
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.layers import Activation, Dense, Dropout
from keras.models import Sequential

# Importing the dataset
dataset = pd.read_csv('dataset_Bot.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 10].values

# split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
x=y_test

#Normalization of training data
scaler = MinMaxScaler()
# fit scaler on data
scaler.fit(X_train)
# apply transform
normalized = scaler.transform(X_train)
linha,coluna=normalized.shape
saida=1
input_img = Input(shape=(coluna,))
output_img=Input(shape=(saida,))
# Normalization of test data
scaler = MinMaxScaler()
# fit scaler on data
scaler.fit(X_test)
# apply transform
normalized_validacao = scaler.transform(X_test)

#Implementing the ANN model
dropout=0.4
modelo = Sequential()
modelo.add(Dense(10, activation='relu',input_dim=coluna))
modelo.add(Dropout(dropout))
modelo.add(Dense(8, activation='relu'))
modelo.add(Dropout(dropout))
modelo.add(Dense(3, activation='relu'))
modelo.add(Dropout(dropout))
modelo.add(Dense(1, activation='sigmoid'))


modelo.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy'])


#ANN training and testing.
modelo_cifra = modelo.fit(normalized, y_train,
                epochs=200,
                batch_size=10,
                shuffle=True, validation_data=(normalized_validacao, y_test))

#The pred variable stores the classification result
pred=modelo.predict_classes(normalized_validacao)

#The score variable is a two-position vector that stores RNA accuracy and error
score = modelo.evaluate(normalized_validacao, y_test) 


#accuracy
modelo.predict_classes
print('Test accuracy:', score[1])

#Drawing the error and accuracy graph
plt.subplot(211)

plt.plot(modelo_cifra.history['loss'], label='Treinamento')
plt.plot(modelo_cifra.history['val_loss'], label='Teste')
plt.xlabel('Número de épocas')
plt.ylabel('Erro')
plt.legend()
plt.show()
plt.subplot(212)

plt.plot(modelo_cifra.history['accuracy'], label='Treinamento')
plt.plot(modelo_cifra.history['val_accuracy'], label='Teste')
plt.xlabel('Número de épocas')
plt.ylabel('Acurácia')

plt.legend()
plt.show()

#Generating the confusion matrix
labels = ['Class 0', 'Class 1']
matriz = metrics.confusion_matrix(x,pred)
print(matriz)

#Network performance metrics
print(classification_report(x, pred, target_names=labels))

# To see the number of examples for each class.
dataset.groupby('bot').size()

