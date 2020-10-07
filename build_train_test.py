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

    for f in os.listdir(tensor_path):
        if 'rate' in f:
            continue
        with open(tensor_path+f,'rb') as p_file:
            song = pickle.load(p_file)
            ta = tf.TensorArray(tf.float32,size=0,dynamic_size=True)

            lengths = calculate_chunk_size(song.get_shape()[0],seq_len)
            ta = ta.split(value=song,lengths=lengths)
            for i in range(ta.size()):
                data.append(ta.read(i))
    np.random.shuffle(data)

    return data

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
def split_input_target(data):

    x = []
    y = []

    for chunk in data:
        x.append(chunk[:-1])
        y.append(chunk[1:])

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
    x,y = split_input_target(data)
    x_train,x_test,y_train,y_test = split_train_test(x,y)

    with open('data_file.p','wb') as f:
        pickle.dump([x_train,x_test,y_train,y_test],f)
