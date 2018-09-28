from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
from magenta.models.nsynth.wavenet import fastgen
import sys
# Change path back to /src to load other modules
sys.path.insert(0, '/home/ubuntu/DeepBass/src')
from ingestion.IO_utils import Load, Save
from preprocess.SilenceRemoval import SR
import streamlit as st
import time
import math

###############################################################################

def LinearFade(length):
    fadein = np.linspace(0, 1, length).reshape(1, -1, 1)
    return fadein

###############################################################################
    
def HannFade(length):
    fadein = (0.5 * (1.0 - np.cos(math.pi * np.arange(length) / 
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
AUDIO_DIR = '/home/ubuntu/test'
filenames = [f for f in listdir(AUDIO_DIR) if isfile(join(AUDIO_DIR, f))]

sr = 16000
# magenta also uses librosa for loading
FirstSong_fname = filenames[1]
SecondSong_fname = filenames[0]
FirstSong, _ = Load(AUDIO_DIR, FirstSong_fname , sr=sr)
SecondSong, _ = Load(AUDIO_DIR, SecondSong_fname, sr=sr)

# Remove any silence at the end of the first song
# and the beginning of the second song
t_snip = 30 # interrogation length in seconds
end_index = SR(FirstSong, 'end', t_snip=t_snip)
end_index = int(t_snip*sr - end_index) # change index reference frame
start_index = SR(SecondSong, 'begin', t_snip=t_snip)
FirstSong = FirstSong[:-end_index]
SecondSong = SecondSong[start_index:]

# Trim to t_len seconds
t_len = 5
sample_length = t_len*sr
FirstSong_end = FirstSong[-sample_length:]
SecondSong_begin = SecondSong[0:sample_length]

# Plot PCM of both snippets
fig, axs = plt.subplots(2, 1, figsize=(10, 5))
axs[0].plot(FirstSong_end)
axs[0].set_title('First Song')
axs[1].plot(SecondSong_begin)
axs[1].set_title('Second Song')
st.pyplot()

# Save original snippets
output_dir = '/home/ubuntu/DeepBass/src/notebooks/'
output_name1 = 'originalend_' + FirstSong_fname + '.wav'
Save(output_dir, output_name1, FirstSong_end, sr)
output_name2 = 'originalbegin_' + SecondSong_fname + '.wav'
Save(output_dir, output_name2, SecondSong_begin, sr)

model_dir = '/home/ubuntu/DeepBass/src/notebooks/wavenet-ckpt/model.ckpt-200000'

# Create encodings
start = time.time()
enc1 = fastgen.encode(FirstSong_end, model_dir, sample_length)
enc2 = fastgen.encode(SecondSong_begin, model_dir, sample_length)
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
st.pyplot()

start = time.time()
@st.cache
def synth():
    fastgen.synthesize(xfade_encoding, checkpoint_path = model_dir, 
                       save_paths=['enc_' + fade_type + '_' + FirstSong_fname + \
                                   SecondSong_fname], 
                       samples_per_save=sample_length)
    return None
synth()
end = time.time()
st.write('Decoding took ' + str((end-start)) + ' seconds')

xfade_audio, _ = Load(output_dir, 'enc_' + fade_type + '_' + FirstSong_fname + \
                      SecondSong_fname, sr=sr)
fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(xfade_audio)
ax.set_title('Crossfaded audio')
st.pyplot()