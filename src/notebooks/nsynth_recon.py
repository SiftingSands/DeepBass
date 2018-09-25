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

"""Demo of NSynth embedding of audio and its reconstruction

Args:
    tlen (float): Amount of time for reconstruction
    silence_len (float) : Skip this many seconds of the ending that is silent
    output_dir (str) : Directory to save the reconstruction
    model_dir (str) : Directory of the pretrained model (tf checkpoint)

    sr must be 16 kHz per the model architecture
    
Returns:
    Streamlit notebook
    Reconstructed audio in the form of a wav file
"""

# Directory where mp3 are stored.
AUDIO_DIR = '/home/ubuntu/DeepBass/data/raw/EDM_Test'
filenames = [f for f in listdir(AUDIO_DIR) if isfile(join(AUDIO_DIR, f))]

sr = 16000
# magenta also uses librosa for loading
x1, _ = Load(AUDIO_DIR, filenames[0], sr=sr)

# Take the last four seconds
t_len = 1
silence_len = 7 
x1 = x1[:silence_len*sr]
x1 = x1[-sr*t_len:]
sample_length = x1.shape[0]
output_dir = '/home/ubuntu/DeepBass/src/notebooks/'
output_name = 'original_' + filenames[0] + '.wav'
Save(output_dir, output_name, x1, sr)

model_dir = '/home/ubuntu/DeepBass/src/notebooks/wavenet-ckpt/model.ckpt-200000'

# Create encoding
start = time.time()
encoding = fastgen.encode(x1, model_dir, sample_length)
end = time.time()

st.write('Encoding took ' + str((end-start)) + ' seconds')
st.write('Encoding shape ' + str(encoding.shape))

# Save encoding
np.save(filenames[0] + '.npy', encoding)

# Plot PCM and
fig, axs = plt.subplots(2, 1, figsize=(10, 5))
axs[0].plot(x1)
axs[0].set_title('Audio Signal')
axs[1].plot(encoding[0])
axs[1].set_title('NSynth Encoding')
st.pyplot()

# Decoding
start = time.time()
fastgen.synthesize(encoding, checkpoint_path = model_dir, 
                   save_paths=['gen_' + filenames[0]], 
                   samples_per_save=sample_length)
end = time.time()
st.write('Decoding took ' + str((end-start)) + ' seconds')

# Evaluate reconstruction
x1_gen, _ = Load(output_dir, 'gen_' + filenames[0], sr=sr)
fig, ax = plt.subplots(figsize=(10, 5))
axs.plot(x1_gen)
axs.set_title('Reconstructed Audio Signal')
st.pyplot()
