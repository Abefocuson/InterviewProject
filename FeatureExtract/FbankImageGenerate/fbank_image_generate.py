
import os
from glob import glob
import pickle
import itertools
import numpy as np
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image
import librosa
import IPython
import random
from IPython.display import Audio
from python_speech_features.base import fbank
import matplotlib.pyplot as pl



def fbak_feature_extract(y,sr=16000):
    
    fbank_feature = fbank(y,samplerate=16000,winlen=0.025,winstep=0.001,
          nfilt=128,nfft=512,lowfreq=0,highfreq=None,preemph=0.97) 
    return fbank_feature[0].T


def mel_spectrogram(y, sr=16000, n_fft=512, win_length=256, hop_length=128, window='hamming', n_mels=128, fmax=4000):
    
    # Compute spectogram
    mel_spect = np.abs(librosa.stft(y, n_fft=n_fft, window=window, win_length=win_length, hop_length=hop_length)) ** 2

    # Compute mel spectrogram
    mel_spect = librosa.feature.melspectrogram(S=mel_spect, sr=sr, n_mels=n_mels, fmax=fmax)
    
    # Compute log-mel spectrogram
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    
    return mel_spect

def extract_fbank(y, sr, size=3, n_fft = 512, hop_length=.01, n_mels=40):
    """
    extract log mel spectrogram feature
    :param y: the input signal (audio time series)
    :param sr: sample rate of 'y'
    :param size: the length (seconds) of random crop from original audio, default as 3 seconds
    :param n_fft:
    :param hop_length: hop time in second, default 10ms
    :param n_nmels: 
    :return: log-mel spectrogram feature
    """
    # normalization
    y = y.astype(np.float32)

    # Pre-Emphasis
    pre_emphasis = 0.97
    emphasized_signal = np.append(y[0], y[1:] - pre_emphasis * y[:-1])
    
    signal_size = librosa.samples_to_time(len(emphasized_signal), sr)
    
    if(size < signal_size): 
        # random crop
        start = random.randint(0, len(y) - size * sr)
        #y = y[start: start + size * sr]
        emphasized_signal = emphasized_signal[0:int(size*sr)]

    # extract log mel spectrogram #####
    melspectrogram = librosa.feature.melspectrogram(y=emphasized_signal, sr=sr, n_fft=n_fft, hop_length=int(hop_length*sr), n_mels=n_mels)
    logmelspec = librosa.power_to_db(melspectrogram)

    return logmelspec
# Audio file path and names
from_dir = "C:\\Users\\admin\\Desktop\\Caugh\\audio\\audio_p\\train\\"
file_names = os.listdir(from_dir)
# print(file_names)
# exit()

signal = []
labels = []

sample_rate = 22050     
max_pad_len = 160000

for audio_index, audio_file in enumerate(file_names):
  
    # Read audio file
    if (os.path.splitext(audio_file)[1] in ['.wav']):
        print(audio_file)
        y, sr = librosa.load(from_dir + audio_file, sr=sample_rate, offset=0)       
        S = fbak_feature_extract(y,sr)  

        fig = pl.figure(figsize=(2.24, 2.24), dpi=100)    
        ax = fig.add_subplot(111)
        plt.pcolormesh(np.abs(S.T))
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0, 0)  
        # plt.show()
        # exit(0) 
        pl.savefig(from_dir + audio_file+"FBAK.jpg") 
        pl.close('all')
       
     




if __name__ == '__main__':
    pass