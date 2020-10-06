#!/usr/bin/env python3
import tensorflow as tf
from tensorflow import audio
import sys
import pickle
import os
import numpy as np

seq_len = 10000 #global variable for chunk size
tensor_path = 'tensors/' #path to tensors

def load_data(tensor_path):
    data = [] #list of data chunks

    for f in os.listdir(tensor_path):
        if 'rate' in f:
            continue
        with open(tensor_path+f,'rb') as p_file:
            song = pickle.load(p_file)
            data.append(song.batch(seq_len))
    np.shuffle(data)

    return data

# Offesets chunks by 1; uses (0 to n-1) as input and (1 to n) as target
def split_input_target(data):

    x = []
    y = []

    for chunk in data:
        x = chunk[:-1]
        y = chunk[1:]

    return x,y

def split_train_test(x,y):

    x_train = x[:len(x)*.8]
    x_test = x[len(x)*.8:]
    y_train = y[:len(y)*.8]
    y_test = y[len(y)*.8:]

    return x_train,x_test,y_train,y_test

if __name__ == '__main__':
    data = load_data(tensor_path)
    x,y = split_input_target(data)
    x_train,x_test,y_train,y_test = split_train_test(x,y)
    print(x_train[0])
