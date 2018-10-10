import os
import argparse
from ingestion.IO_utils import Load, Save
from model.crossfade_NSynth import NSynth
from preprocess.SilenceRemoval import SR
from analysis.timbral_measures import timbral_measures
import sys
import numpy as np
import configparser
import errno

"""Main script to perform cross fading between two songs using low-dimensional
   embeddings created from the NSynth Wavenet autoencoder.
   
Give the config file name as the input or let it be generated from the 
CreateConfig.py script. See CreateConfig.py for input descriptions.

Ex : python main_simple.py config.ini

"""

parser = argparse.ArgumentParser(description='Load Audio Files')
parser.add_argument('-config_fname', default='config.ini',
                    help='Config file name. Must be in /configs/', 
                    type=str)
args = parser.parse_args()

config = configparser.ConfigParser()
config_path = os.path.join('../configs', args.config_fname)
# Create config if it does not exist
if not os.path.exists(config_path):
    sys.path.insert(0, '../') # Navigate to config folder
    from configs.CreateConfig import createconfig
    createconfig(config_path)

# Load config file and make namespace less bulky
config.read(config_path)
sr = config.getint('DEFAULT','samplingrate')
fade_style = config['NSynth XFade Settings']['Style']
fade_time = config.getfloat('NSynth XFade Settings','Time')
t_snip = config.getfloat('Preprocess','SR window duration')
FirstSong = config['IO']['FirstSong']
SecondSong = config['IO']['SecondSong']
load_dir = config['IO']['Load Directory']
save_dir = config['IO']['Save Directory']
savename = config['IO']['Save Name']
modelweights_path = config['IO']['Model Weights']

# number of samples for fade
fade_length = int(fade_time * sr)
# Load Songs
FirstSong, _ = Load(load_dir, FirstSong, sr)
SecondSong, _ = Load(load_dir, SecondSong, sr)

# Remove any silence at the end of the first song
# and the beginning of the second song
FirstSong_trim = SR(FirstSong, 'end', t_snip=t_snip)
SecondSong_trim = SR(SecondSong, 'begin', t_snip=t_snip)

# Create the save folder if it does not exist
if not os.path.exists(save_dir):
    try:
        os.makedirs(save_dir)
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

# Create transition and save the result
xfade_audio, x1_trim, x2_trim, enc1, enc2 = NSynth(FirstSong,
                                                   SecondSong,
                                                   fade_style,
                                                   fade_length,
                                                   modelweights_path,
                                                   save_dir,
                                                   savename+'_NSynth')

# Save encodings of the audio
np.save('begin_enc' + '.npy', enc1)
np.save('end_enc' + '.npy', enc2)

# Save trimmed segments for reference
Save(save_dir, 'end.wav', x1_trim, sr)
Save(save_dir, 'begin.wav', x2_trim, sr)

# Evaluate roughness of the cross fade and save to text file
roughness = timbral_measures(os.path.join(save_dir,savename+'_NSynth.wav'),\
                             'Timbral_Roughness')
textfilepath = os.path.join(save_dir, savename+'_nsynth_roughness.txt')
with open(textfilepath, 'w') as f:
    f.write('%f' % roughness)