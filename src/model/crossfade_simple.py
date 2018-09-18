from os import listdir
from os.path import isfile, join

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

import streamlit as st

st.title('Simple cross fading')

# Directory where mp3 are stored.
AUDIO_DIR = '/home/ubuntu/DeepBass/data/raw/000/'
filenames = [f for f in listdir(AUDIO_DIR) if isfile(join(AUDIO_DIR, f))]

# Randomly sample two songs from the folder
selected_files = np.random.choice(filenames,size=2)
print('Files: {}'.format(selected_files))

# Load both songs
x1, sr1 = librosa.load(AUDIO_DIR + selected_files[0], sr=None, mono=True)
print('Song 1 Duration: {:.2f}s, {} samples'.format(x1.shape[-1] / sr1, 
      x1.size))
time1 = np.arange(0,len(x1),1)/sr1

x2, sr2 = librosa.load(AUDIO_DIR + selected_files[1], sr=None, mono=True)
print('Song 2 Duration: {:.2f}s, {} samples'.format(x2.shape[-1] / sr2, 
      x2.size))
time2 = np.arange(0,len(x2),1)/sr2

# Trim the songs to the beginning and end
if sr1 != sr2:
    raise ValueError('Sampling rates between songs must be identical')

snippet_time = 10 # seconds
snippet_length = snippet_time * sr1 # number of samples

x1_trim =x1[-snippet_length:]
t1_trim = time1[-snippet_length:]
x2_trim =x2[0:snippet_length]
t2_trim = time2[0:snippet_length]

# Plot amplitudes for the beginning and end
fig, ax = plt.subplots(2)
ax[0].plot(t1_trim, x1_trim, 'r-')
ax[1].plot(t2_trim, x2_trim, 'b-')
st.pyplot()

# Cross Fading Choice
Ramp = 'Linear'

if Ramp == 'Linear':
    x1_trim_faded = x1_trim*np.linspace(1,0,len(x1_trim))
    x2_trim_faded = x2_trim*np.linspace(0,1,len(x2_trim))
elif Ramp == 'Sigmoid':
    temp = np.linspace(-6,6,len(x1_trim))
    x1_trim_faded = x1_trim*(1/(1+np.exp(temp)))
    x2_trim_faded = x2_trim*(1/(1+np.exp(-temp)))

fig, ax = plt.subplots(2)
ax[0].plot(t1_trim, x1_trim_faded, 'r-')
ax[1].plot(t2_trim, x2_trim_faded, 'b-')
st.pyplot()

# Mixed audio
fade = x1_trim_faded + x2_trim_faded
fig, ax = plt.subplots()
ax.plot(t1_trim, fade, 'g-')
st.pyplot()

# Create a single wav file incorporating both songs and the cross fade
mix = np.concatenate((x1[:-snippet_length], fade, x2[snippet_length:]), axis=0)
st.write('Total time steps for the mixed song ', mix.shape)
fig, ax = plt.subplots()
ax.plot(mix, 'k-')
st.pyplot()

output_name = Ramp + '_CrossFaded_' + str(snippet_time) + '.wav'
librosa.output.write_wav('/home/ubuntu/DeepBass/data/processed/' + \
                         output_name, mix, sr1)

