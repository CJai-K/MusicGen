import tensorflow as tf
from tensorflow import keras
import pickle
import numpy as np

seq_len = 10000 #make
def build_nn(in_shape,out_shape):
    model = keras.Sequential()
    #model.add(keras.layers.Embedding())
    #model.add(keras.layers.LSTM(100,))
    model.add(keras.layers.Dense(100,input_shape=in_shape))
    model.add(keras.layers.Dense(2))
    #model.add(keras.layers.Reshape(out_shape))
    model.compile(optimizer='Adam',loss='L2')

    print(model.summary())
    return model


if __name__ == '__main__':
    with open('data_file.p','rb') as data_f:
        x_train,x_test,y_train,y_test = pickle.load(data_f)
        print(x_train[0])
        input_shape = (x_train[0].get_shape()[0],x_train[0].get_shape()[1])
        output_shape = (y_train[0].get_shape()[0],y_train[0].get_shape()[1])
        model = build_nn(input_shape,output_shape)
        model.fit(x=x_train,y=y_train,validation_data=(x_test,y_test))


