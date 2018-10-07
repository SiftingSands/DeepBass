import numpy as np

def Crossfade_Simple(FirstSong, SecondSong, Ramp, fade_length):
    """Create snippet of audio with simple cross faded transition between 
    two songs
    
    Args:
        FirstSong (numpy array, float): First Song
        
        SecondSong (numpy array, float) : Second Song
        
        Ramp (str) : Type of ramping function to use
        
        fade_length (int) : Time steps taken from the beginning of the first 
        and the end of the second song for mixing

    Returns:
        Mix (numpy array, float) : Mixed audio stream
        
        x1_trim_faded (numpy array, float) : Faded first song ending
        
        x2_trim_faded (numpy array, float) : Faded second song beginning
    """
    
    x1_trim =FirstSong[-fade_length:]
    x2_trim =SecondSong[0:fade_length]

    if Ramp == 'Linear':
        x1_trim_faded = x1_trim*np.linspace(1, 0, len(x1_trim))
        x2_trim_faded = x2_trim*np.linspace(0, 1, len(x2_trim))
    elif Ramp == 'Sigmoid':
        temp = np.linspace(-6,6,len(x1_trim))
        x1_trim_faded = x1_trim*(1/(1+np.exp(temp)))
        x2_trim_faded = x2_trim*(1/(1+np.exp(-temp)))
    elif Ramp == 'Random_Linear':
        # Randomly sample from both songs but with linearly ramped probability
        prob = np.random.uniform(0, 1, size=len(x1_trim))
        prob_threshold = np.linspace(0, 1, len(x1_trim))
        FirstSong_Switch = np.greater(prob, prob_threshold)
        x1_trim_faded = x1_trim*(1*FirstSong_Switch) # convert from bool to int
        x2_trim_faded = x2_trim*(1*np.invert(FirstSong_Switch))
        # Fade both trims again with a linear ramp
        x1_trim_faded = x1_trim_faded*np.linspace(1, 0, len(x1_trim))
        x2_trim_faded = x2_trim_faded*np.linspace(0, 1, len(x2_trim))
    else:
        TypeError('Invalid ramp function specified.')
        
    # Mixed audio
    mix = x1_trim_faded + x2_trim_faded
    
    return mix, x1_trim_faded, x2_trim_faded, x1_trim, x2_trim
