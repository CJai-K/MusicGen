#!/usr/bin/env python3
import tensorflow as tf
from tensorflow import audio
import sys
import pickle
import os
import numpy as np

seq_len = 1000 #global variable for chunk size
tensor_path = 'tensors/' #path to tensors

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

def split_train_test(x,y):
    print(len(x),len(y))
    x_train = x[:int(len(x)*.8)]
    x_test = x[int(len(x)*.8):]
    y_train = y[:int(len(x)*.8)]
    y_test = y[int(len(x)*.8):]

    return x_train,x_test,y_train,y_test

if __name__ == '__main__':
    data = load_data(tensor_path)
    print(data.cardinality())
    #x,y = split_input_target(data)
    #x_train,x_test,y_train,y_test = split_train_test(x,y)

    #with open('data_file.p','wb') as f:
     #   pickle.dump([x_train,x_test,y_train,y_test],f)
