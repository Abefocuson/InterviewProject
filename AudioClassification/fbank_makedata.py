
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

# Label dict
label_dict_cough = {'not_sick': '0', 'sick':'1'}

def set_label(cough_type):
    label = label_dict_cough.get(cough_type)   
    return label


def fbak_feature_extract(y,sr=16000):
    
    # Compute fbank spectrogram
    fbank_feature = fbank(y,samplerate=16000,winlen=0.025,winstep=0.01,
          nfilt=128,nfft=512,lowfreq=0,highfreq=None,preemph=0.97) 
    return fbank_feature[0].T

# Split spectrogram into frames
def frame(x, win_step=128, win_size=64):
    nb_frames = 1 + int((x.shape[2] - win_size) / win_step)
    frames = np.zeros((x.shape[0], nb_frames, x.shape[1], win_size)).astype(np.float32)
    for t in range(nb_frames):
        frames[:,t,:,:] = np.copy(x[:,:,(t * win_step):(t * win_step + win_size)]).astype(np.float32)
    return frames

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
file_path = 'D:\\InterView\\Data\\audio\\audio_lite\\train\\'
file_names = os.listdir(file_path)
signal = []
labels = []

sample_rate = 16000     
max_pad_len = 16000*6
for home, dirs, files in os.walk(file_path):    
    for filename in files:
        fullname = os.path.join(home, filename)      
        y, sr = librosa.load(fullname, sr=sample_rate, offset=0)
        print(len(y), sr)
        y = zscore(y)
        if len(y) < max_pad_len:    
            y_padded = np.zeros(max_pad_len)
            y_padded[:len(y)] = y
            y = y_padded
        elif len(y) > max_pad_len:
            y = np.asarray(y[:max_pad_len])
        signal.append(y)
        labels.append(set_label(fullname.split('\\')[-2]))
     

labels = np.asarray(labels).ravel()
print("Import Data: END \n")
fbank_spect = np.asarray(list(map(fbak_feature_extract, signal)))
print("================================================")
print('all data:', fbank_spect.shape, labels)

X_train =fbank_spect
y_train = labels

win_ts = 128
hop_ts = 64

# Frame Split for TimeDistributed model
X_train = frame(X_train, hop_ts, win_ts)
print('trian:', X_train.shape, y_train)

pickle.dump(X_train.astype(np.float16), open('./Pickle/[Cough][FBANK][X_train].p', 'wb'))
pickle.dump(y_train, open('./Pickle/[Cough][FBANK][y_train].p', 'wb'))

if __name__ == '__main__':
    pass