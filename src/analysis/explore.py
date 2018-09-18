from os import listdir
from os.path import isfile, join

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

import streamlit as st

st.title('Audio exploration')

# Directory where mp3 are stored.
AUDIO_DIR = '/home/ubuntu/DeepBass/data/raw/000/'

filenum = 10
filename = [f for f in listdir(AUDIO_DIR) if isfile(join(AUDIO_DIR, f))]
filename = filename[filenum]
print('File: {}'.format(filename))

x, sr = librosa.load(AUDIO_DIR + filename, sr=None, mono=True)
print('Duration: {:.2f}s, {} samples'.format(x.shape[-1] / sr, x.size))
time = np.arange(0,len(x),1)/sr

snippet_time = 5 # seconds
snippet_length = snippet_time * sr # number of samples

# Calculate Mel Spectrogram
nfft = 2048
hop_length = 512
stft = np.abs(librosa.stft(x[-snippet_length:], n_fft=nfft, 
                           hop_length=hop_length))
mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
log_mel = librosa.amplitude_to_db(mel)

# Plot truncated waveform and its spectrogram
plt.figure(1)
plt.plot(time[-snippet_length:], x[-snippet_length:])
plt.ylim((-1, 1))
st.pyplot()

plt.figure(2)
librosa.display.specshow(log_mel, sr=sr, y_axis='mel', fmax=8000, 
                         x_axis='time', hop_length = hop_length)
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
st.pyplot()

# Test out saving to .wav file
output_name = 'trimmed_5s.wav'
librosa.output.write_wav('/home/ubuntu/DeepBass/data/preprocessed/' + \
                         output_name, x[-snippet_length:], sr)