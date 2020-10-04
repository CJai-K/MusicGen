import tensorflow as tf
from tensorflow import audio
import sys
import pickle
import os

seq_len = 10000 #global variable for chunk size

def load_data(tensor_path):
    data = [] #list of data chunks

    for f in os.listdir(tensor_path):
        if 'rate' is in f:
            continue
        song = load(tensor_path+f)
        data.append(song.batch(seq_len))

def split_input_target(data):

