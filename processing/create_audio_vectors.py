import tensorflow as tf
from tensorflow import audio
from tensorflow import io
import sys
import os
import pickle

def song_vectors(song_dir):

    dir_name = '../tensors/'
    with open('song_vectors.txt','w') as f:
        f_index = 0
        for file_name in os.listdir(song_dir):
            raw_audio = io.read_file(song_dir+file_name)
            song_vector,sample_rate = audio.decode_wav(raw_audio, desired_samples=100000)

            song_pickle = open(dir_name+'song_tensor'+str(f_index),'wb')
            rate_pickle = open(dir_name+'rate_tensor'+str(f_index),'wb')
            pickle.dump(song_vector,song_pickle)
            pickle.dump(sample_rate,rate_pickle)

            song_pickle.close()
            rate_pickle.close()

            f.write(str(sample_rate)+':')
            for tensor in song_vector:
                f.write(str(tensor))
            f.write('\n')
            f_index+=1

if __name__ == "__main__":
    song_vectors(sys.argv[1])




