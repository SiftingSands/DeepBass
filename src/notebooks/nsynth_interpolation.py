from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
from magenta.models.nsynth.wavenet import fastgen
import sys
# Change path back to /src to load other modules
sys.path.insert(0, '/home/ubuntu/DeepBass/src')
from ingestion.IO_utils import Load, Save
import streamlit as st
import time

###############################################################################

def LinearFade(length):
    fadein = np.linspace(0, 1, length).reshape(1, -1, 1)
    return fadein

###############################################################################
    
def HannFade(length):
    fadein = (0.5 * (1.0 - np.cos(3.1415 * np.arange(length) / 
                                  float(length)))).reshape(1, -1, 1)
    return fadein

###############################################################################

def fade(encoding, fade_type, mode='in'):
    length = encoding.shape[1]
    method = globals().copy().get(fade_type)
    if not method:
         raise NotImplementedError("Fade %s not implemented" % fade_type)
    fadein = method(length)
    if mode == 'in':
        return fadein * encoding
    else:
        return (1.0 - fadein) * encoding

###############################################################################

def crossfade(encoding1, encoding2, fade_type):
    return fade(encoding1, fade_type, 'out') + fade(encoding2, fade_type, 'in')

###############################################################################

"""Demo of cross fading in the NSynth embedding space

User Parameters:
    tlen (float): Amount of time for reconstruction
    silence_len1 (float) : Skip this many seconds of the ending that is silent
    silence_len2 (float) : Skip this many seconds of the beginning that is silent
    AUDIO_DIR (str) : Directory of the audio files
    output_dir (str) : Directory to save the reconstruction
    model_dir (str) : Directory of the pretrained model (tf checkpoint)

Returns:
    Streamlit notebook
    Crossfaded audio in the form of a wav file
    
Notes:
    sr must be 16 kHz per the model architecture
    
"""

# Directory where mp3 are stored.
AUDIO_DIR = '/home/ubuntu/DeepBass/data/raw/EDM_Test'
filenames = [f for f in listdir(AUDIO_DIR) if isfile(join(AUDIO_DIR, f))]

sr = 16000
# magenta also uses librosa for loading
x1, _ = Load(AUDIO_DIR, filenames[0], sr=sr)
x2, _ = Load(AUDIO_DIR, filenames[1], sr=sr)

# Take the last n seconds
t_len = 4
silence_len1 = 11 # Skip the near silent part of the ending
x1 = x1[:-silence_len1*sr]
x1 = x1[-sr*t_len:]

silence_len2 = 1
x2 = x2[silence_len2*sr:]
x2 = x2[:t_len*sr]

sample_length = x1.shape[0]

# Plot PCM of both snippets
fig, axs = plt.subplots(2, 1, figsize=(10, 5))
axs[0].plot(x1)
axs[0].set_title('First Song')
axs[1].plot(x2)
axs[1].set_title('Second Song')
st.pyplot()

# Save original snippets
output_dir = '/home/ubuntu/DeepBass/src/notebooks/'
output_name1 = 'original_' + filenames[0] + '.wav'
Save(output_dir, output_name1, x1, sr)
output_name2 = 'original_' + filenames[1] + '.wav'
Save(output_dir, output_name2, x2, sr)

model_dir = '/home/ubuntu/DeepBass/src/notebooks/wavenet-ckpt/model.ckpt-200000'

# Create encodings
start = time.time()
enc1 = fastgen.encode(x1, model_dir, sample_length)
enc2 = fastgen.encode(x2, model_dir, sample_length)
end = time.time()
st.write('Encoding took ' + str((end-start)) + ' seconds')

# Create cross fading in the latent space
fade_type = 'LinearFade'
xfade_encoding = crossfade(enc1, enc2, fade_type)

fig, axs = plt.subplots(3, 1, figsize=(10, 7))
axs[0].plot(enc1[0])
axs[0].set_title('Encoding 1')
axs[1].plot(enc2[0])
axs[1].set_title('Encoding 2')
axs[2].plot(xfade_encoding[0])
axs[2].set_title('Crossfade')
st.pyplot()r

start = time.time()
@st.cache
def synth():
    fastgen.synthesize(xfade_encoding, checkpoint_path = model_dir, 
                       save_paths=['enc_' + fade_type + '_' + filenames[0] + \
                                   filenames[1]], 
                       samples_per_save=sample_length)
    return None
synth()
end = time.time()
st.write('Decoding took ' + str((end-start)) + ' seconds')

xfade_audio, _ = Load(output_dir, 'enc_' + fade_type + '_' + filenames[0] + \
                               filenames[1], sr=sr)
fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(xfade_audio)
ax.set_title('Crossfaded audio')
st.pyplot()