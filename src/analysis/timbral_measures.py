import importlib

def timbral_measures(fname, measure):
    """ Given a audio file, calculate various timbral characteristics.
    We typically only ask for roughness to evaluate if the decoder performed
    well. Other measures have various parameters that are not asked for in 
    this function such as fft window size.
    
    Args:
        fname (str) : full path directory of the audio file
        measure (str) : Must be one of the following exactly
                        Timbral_Hardness
                        Timbral_Depth
                        Timbral_Brightness
                        Timbral_Roughness
                        Timbral_Warmth
                        Timbral_Sharpness
                        Timbral_Booming
                        
    Return:
        output (float)
    """
    
    model = importlib.import_module('.' + measure, package='timbral_models')
    function = getattr(model, measure.lower())
    output = function(fname)
    
    return output