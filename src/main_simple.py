import os
import argparse
from ingestion.IO_utils import Load, Save
from model.crossfade_simple import Crossfade_Simple
import errno
import configparser
import sys

###############################################################################
# Directory fault checking
class FullPaths(argparse.Action):
    """Expand user- and relative-paths"""
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, os.path.abspath(os.path.expanduser(values)))

def is_dir(dirname):
    """Checks if a path is an actual directory"""
    if not os.path.isdir(dirname):
        msg = "{0} is not a directory".format(dirname)
        raise argparse.ArgumentTypeError(msg)
    else:
        return dirname
###############################################################################
"""Main script to perform cross fading between two songs using ramp functions
applied in the raw audio (PCM) domain.

See CreateConfig.py for input details.

"""

config = configparser.ConfigParser()
config_path = '../configs/config.ini'
# Create config if it does not exist
if not os.path.exists(config_path):
    sys.path.insert(0, '../') # Navigate to config folder
    from configs.CreateConfig import createconfig
    createconfig(config_path)

# Load config file and make namespace less bulky
config.read(config_path)
sr = config.getint('DEFAULT','samplingrate')
fade_style = config['Simple XFade Settings']['Style']
fade_time = config.getfloat('Simple XFade Settings','Time')
t_snip = config.getfloat('Preprocess','SR window duration')
FirstSong = config['IO']['FirstSong']
SecondSong = config['IO']['SecondSong']
load_dir = config['IO']['Load Directory']
save_dir = config['IO']['Save Directory']
savename = config['IO']['Save Name']

# Number of samples for fade
fade_length = int(fade_time * sr) 

# Load Songs
FirstSong,_ = Load(load_dir, FirstSong, sr)
SecondSong,_ = Load(load_dir, SecondSong, sr)

# Create the save folder if it does not exist
if not os.path.exists(save_dir):
    try:
        os.makedirs(save_dir)
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

# Create cross fading
fade, x1_trim_faded, x2_trim_faded, x1_trim, x2_trim = Crossfade_Simple( \
                                                       FirstSong, 
                                                       SecondSong, 
                                                       fade_style,
                                                       fade_length)

# Save faded and trimmed segments for reference
Save(save_dir, 'end_faded.wav', x1_trim_faded, sr)
Save(save_dir, 'begin_faded.wav', x2_trim_faded, sr)
Save(save_dir, 'end_trim.wav', x1_trim, sr)
Save(save_dir, 'begin_trim.wav', x2_trim, sr)

# Save combined cross faded audio
Save(save_dir, savename+'_simple.wav', fade, sr)