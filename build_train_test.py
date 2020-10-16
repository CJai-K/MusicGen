#!/usr/bin/env python3
import tensorflow as tf
from tensorflow import audio
from tensorflow import keras
import sys
import pickle
import os
import numpy as np

EPOCHS = 3
BATCH_SIZE = 20
seq_len = 1000 #global variable for chunk size
tensor_path = 'tensors/' #path to tensors
BUFFER_SIZE = 1000

def load_data(tensor_path):
    data = [] #list of data chunks
    dataset = None #to be filled with chunks of songs
    for f in os.listdir(tensor_path):
        if 'rate' in f:
            continue
        with open(tensor_path+f,'rb') as p_file:
            song = pickle.load(p_file)
            song_dataset = tf.data.Dataset.from_tensor_slices(song)
            sequences = song_dataset.batch(seq_len, drop_remainder=True)
            data.append(sequences)
            #lengths = calculate_chunk_size(song.get_shape()[0],seq_len)
            #for i in range(ta.size()):
            #    data.append(ta.read(i))

    dataset = data[0]
    for i in range(1,len(data)):
        dataset = dataset.concatenate(data[i])

    dataset = dataset.map(split_input_target)
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE,drop_remainder=True)


    return dataset

#helper method for TensorArray.Split(); requires 1-d array of lengths to be sum of total tensor length
def calculate_chunk_size(tensor_len,chunk_len):
    while tensor_len%chunk_len != 0:
        chunk_len -= 1
    num_chunks = int(tensor_len/chunk_len)

    lengths = []

    for _ in range(num_chunks):
        lengths.append(chunk_len)

    return lengths

# Offesets chunks by 1; uses (0 to n-1) as input and (1 to n) as target
def split_input_target(chunk):

    x = chunk[:-1]
    y = chunk[1:]

    return x,y

def loss(labels,logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels,logits.set_shape(labels.get_shape()),from_logits=True)

def build_nn(in_shape):
    model = keras.Sequential()
    #model.add(keras.layers.Embedding())
    model.add(keras.layers.InputLayer(input_shape=in_shape,batch_size=BATCH_SIZE))
    model.add(keras.layers.LSTM(100))
    model.add(keras.layers.Dense(100))
    model.add(keras.layers.Dense(1998))
    model.add(keras.layers.Reshape(in_shape))
    model.compile(optimizer='Adam',loss='categorical_crossentropy')

    print(model.summary())
    return model


def split_train_test(x,y):
    print(len(x),len(y))
    x_train = x[:int(len(x)*.8)]
    x_test = x[int(len(x)*.8):]
    y_train = y[:int(len(x)*.8)]
    y_test = y[int(len(x)*.8):]

    return x_train,x_test,y_train,y_test

if __name__ == '__main__':
    data = load_data(tensor_path)
    #with open('data_file.p','wb') as f:
       # pickle.dump(data,f)





    data_shape = [x[0] for x in data.take(1).as_numpy_iterator()][0].shape
    print(data_shape)
    model = build_nn(tuple(list(data_shape)[1:]))


    # Directory where the checkpoints will be saved
    checkpoint_dir = './training_checkpoints'
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,save_weights_only=True)

    history = model.fit(data, epochs=EPOCHS, callbacks=[checkpoint_callback])

