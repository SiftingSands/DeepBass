import numpy as np
    
"""Silence detection based on the amplitude RMS

User Parameters:
    audio (numpy array) : Audio data in the form of PCM (amplitude)
    
    t_snip (float): Time in seconds for interrogation of the beginning or end
    
    mode (str) : Select detection for the beginning or the end of the audio
    
    window_size (float) : Sliding window size in seconds
    
    window_step (float) : Sliding window step size in seconds
    
    Threshold (float) : Percentage of overall RMS to cut off silence
    
    sr (int) : Sampling rate - Must be 16kHz for the NSynth architecture
    
Returns:
    target_index is the cutoff for silence
    
"""
# Function to create sliding windows with overlap
def window(a, w = 4, o = 2, copy = False):
    sh = (a.size - w + 1, w)
    st = a.strides * 2
    view = np.lib.stride_tricks.as_strided(a, strides = st, shape = sh)[0::o]
    if copy:
        return view.copy()
    else:
        return view
    
def SR(audio, mode, t_snip = 30, window_size = 0.25, window_step = 0.125, 
       Threshold = 15, sr = 16000):

    # Crop to either the beginning or the ending
    if mode == 'begin':
        snippet = audio[:int(t_snip*sr)]
    elif mode == 'end':
        snippet = audio[-int(t_snip*sr):]
    else:
        raise ValueError('An invalid silence removal mode was entered!')
        
    # Get the RMS threshold
    audio_RMS_overall = np.sqrt(np.mean(audio**2))
    RMS_Threshold = Threshold*audio_RMS_overall/100
    
    # Create sliding windows
    Windows = window(snippet, w=int(window_size*sr), o=int(window_step*sr), 
                     copy=True)
    
    # Loop over windows from beginning to end until the RMS is below the threshold
    Detected = True
    i = 0
    target_index = 0
    while Detected:
        RMS = np.sqrt(np.mean(Windows[i,:]**2))
        
        if mode == 'begin':
            if RMS >= RMS_Threshold: # Assume start quiet and end loud
                Detected = False
            elif i == len(Windows[:,0])-1:
                Warning('The entirety of the beginning is silent. Clipping \
                        by ' + "{:10.2f}".format(t_snip) + ' seconds')
                target_index = int(t_snip*sr)
                Detected = False
            else:
                i += 1
                target_index += int(window_step*sr)

        elif mode == 'end': # Invalid modes were already detected upstream
            if RMS <= RMS_Threshold: # Assume that we start loud and end quiet
                Detected = False
            elif i == len(Windows[:,0])-1:
                Warning('No silence detected at the ending. No clipping.')
                target_index = int(t_snip*sr)
                Detected = False
            else:
                i += 1
                target_index += int(window_step*sr)

    # Trim the silence
    if mode == 'begin':
        audio_trim = audio[target_index:]
    elif mode == 'end':
        # change index reference frame 
        end_index = int(t_snip*sr - target_index) 
        audio_trim = audio[:-end_index-1]
        
    return audio_trim