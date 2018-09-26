from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
import sys
# Change path back to /src to load other modules
sys.path.insert(0, '/home/ubuntu/DeepBass/src')
from ingestion.IO_utils import Load
import streamlit as st
import json

@st.cache
def load_data(AUDIO_DIR, filename, sr):
    audio, _ = Load(AUDIO_DIR, filename, sr=sr)
    return audio

def window(a, w = 4, o = 2, copy = False):
    sh = (a.size - w + 1, w)
    st = a.strides * 2
    view = np.lib.stride_tricks.as_strided(a, strides = st, shape = sh)[0::o]
    if copy:
        return view.copy()
    else:
        return view
    
"""Demo of silence detection based on the amplitude RMS

User Parameters:
    t_snip (float): Time in seconds for interrogation of the beginning or end
    AUDIO_DIR (str) : Directory of the audio files
    style (str) : Select detection for the beginning or the end of the audio
    window_size (float) : Sliding window size in seconds
    window_step (float) : Sliding window step size in seconds
    Threshold (float) : Percentage of overall RMS to cut off silence
Returns:
    Streamlit notebook
    target_index is the cutoff for silence
    
Notes:
    sr must be 16 kHz per the NSynth architecture
    
"""

# Directory where mp3 are stored.
AUDIO_DIR = '/home/ubuntu/DeepBass/data/raw/EDM_Test'
filenames = [f for f in listdir(AUDIO_DIR) if isfile(join(AUDIO_DIR, f))]

sr = 16000
st.write('Loading data...')
audio = load_data(AUDIO_DIR, filenames[1], sr)
st.write('Done! (using st.cache)')

# Crop to either the beginning or the ending
t_snip = 30
style = 'begin'
if style == 'begin':
    snippet = audio[:int(t_snip*sr)]
else:
    snippet = audio[-int(t_snip*sr):]
audio_RMS_overall = np.sqrt(np.mean(audio**2))
st.write(audio_RMS_overall)

# Create sliding windows
window_size = 0.25
window_step = window_size/2
Windows = window(snippet, w=int(window_size*sr), o=int(window_step*sr), 
                 copy=True)
st.write(Windows.shape)

Threshold = 15 
RMS_Threshold = Threshold*audio_RMS_overall/100

# Loop over windows from beginning to end until the RMS is below the threshold
Detected = True
i = 0
target_index = 0
while Detected:
    RMS = np.sqrt(np.mean(Windows[i,:]**2))
    if style == 'begin':
        if RMS >= RMS_Threshold:
            Detected = False
        else:
            i += 1
            target_index += window_step*sr
    else:
        if RMS <= RMS_Threshold:
            Detected = False
        else:
            i += 1
            target_index += window_step*sr

# Plot audio with a red line denoting the silence cutoff
fig, ax = plt.subplots()
ax.plot(snippet)
ax.axvline(target_index,linewidth=2, color='r')
st.pyplot()
st.write(target_index)

st.json(json.dumps({k:v for (k,v) in locals().items() if not k.startswith('_')}, 
                    default=lambda x: 'Cannot serialize'))
