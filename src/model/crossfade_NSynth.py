import numpy as np
from magenta.models.nsynth.wavenet import fastgen
import time
import os
import librosa
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
    
#def DilateLinearFade()

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

def mu_law(x, mu=255):
    """ Mu-Law encoding in numpy
    Args:
        x: The audio samples to encode.
        mu: The Mu to use in our Mu-Law.
    Returns:
        out: The Mu-Law encoded data.
    """
    out = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
    return out.astype(np.uint8)

###############################################################################
    
def Interp(enc1, enc2, weight):
    interp_enc = enc1*weight + enc2*(1-weight)
    return interp_enc

###############################################################################
    
def NSynth(FirstSong, SecondSong, fade_type, fade_length, model_dir, save_dir,
           savename, repeats=3):
    """Create snippet of audio with cross fading in the embedding space
    Resulting audio is the same length as the input songs
    
    Args:
        FirstSong (numpy array, float): First Song
        
        SecondSong (numpy array, float) : Second Song
        
        fade_type (str) : Type of cross fading or create repeats (dilation)
        
        fade_length (int) : Time steps taken from the beginning of the first 
        and the end of the second song for mixing
        
        model_dir (str) : Directory of the pretrained model
        
        save_dir (str) : Directory to store mixed audio
        
        savename (str) : Filename to be used for saving the mixed audio

    Returns:
        Mix (numpy array, float) : Mixed audio stream
        
        x1_trim_faded (numpy array, float) : Faded first song ending
        
        x2_trim_faded (numpy array, float) : Faded second song beginning
        
        enc1 (numpy array, float) : encoding array for the first song ending
        
        enc2 (numpy array, float) : encoding array for the second song beginning
    """
    
    x1_trim = FirstSong[-fade_length:]
    x2_trim = SecondSong[0:fade_length]
    
    start = time.time()
    enc1 = fastgen.encode(x1_trim, model_dir, fade_length)
    enc2 = fastgen.encode(x2_trim, model_dir, fade_length)
    end = time.time()
    print('*** Encoding took ' + str((end-start)) + ' seconds ***')
    
    os.chdir(save_dir)
    if fade_type == 'Dilation':
        xfade_audio = []
        # Create linear interpolation
        end = 1.0/(repeats+1)
        start = 1 - end
        weights = np.linspace(start, end, num=repeats)
        for n in range(repeats):
            interp_enc = Interp(enc1,enc2,weights[n])
            fastgen.synthesize(interp_enc, checkpoint_path = model_dir, 
                           save_paths=[savename + '_' + str(n) + '.wav'], 
                           samples_per_save=fade_length)
            
            temp, _ = librosa.load(savename + '_' + str(n) + '.wav')
            xfade_audio.append(temp)
        
    else:
        xfade_encoding = crossfade(enc1, enc2, fade_type)
        start = time.time()
        fastgen.synthesize(xfade_encoding, checkpoint_path = model_dir, 
                           save_paths=[savename + '.wav'], 
                           samples_per_save=fade_length)
        end = time.time()
        print('*** Decoding took ' + str((end-start)) + ' seconds ***')
    
        # Load the generated audio and the mu-law encodings for comparison
        xfade_audio, _ = librosa.load(savename + '.wav')
        
    x1_trim_mu = mu_law(x1_trim)
    x2_trim_mu = mu_law(x2_trim)
    
    return xfade_audio, x1_trim_mu, x2_trim_mu, enc1, enc2