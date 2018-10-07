from mutagen.mp3 import MP3
import os

"""Silence detection based on the amplitude RMS

Parameters:
    fname (str) : File name for the audio file
    
    fname (str) : File name for the audio file
    
Returns:
    Duration of the audio file in seconds (float)

"""

def get_time(load_dir, fname):
    filepath = os.path.join(load_dir, fname)
    try:
        audio = MP3(filepath)
        return audio.info.length
    except:
        return False