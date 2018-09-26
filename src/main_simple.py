import os
import argparse
from ingestion.IO_utils import Load, Save
from model.crossfade_simple import Crossfade_Simple
import random
import numpy as np

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
"""Example

python main.py ~/DeepBass/data/raw/House_Top20 Linear 10 test 44100

"""

# Directory where mp3s are stored
parser = argparse.ArgumentParser(description='Load Audio Files')
parser.add_argument('playlist_dir', help='Directory of playlist with audio \
                    files', action=FullPaths, type=is_dir)
parser.add_argument('fade_style', help='Method for cross fading', 
                    choices=['Linear', 'Sigmoid', 'Random_Linear'])
parser.add_argument('fade_time', help='Specify the cross fading duration',
                    type=float)
parser.add_argument('savename', help='Specify the prefix for saving the playlist',
                    type=str)
parser.add_argument('sr', help='Specify sampling rate for song mix', type=int)
args = parser.parse_args()

# Randomly make playlist order
playlist_order = os.listdir(args.playlist_dir)
random.shuffle(playlist_order) # shuffle is an in-place operation

fade_length = int(args.fade_time * args.sr) # number of samples for fade

for n in range(len(playlist_order)-1):
    # Load Songs
    if n==0:
        FirstSong,_ = Load(args.playlist_dir, playlist_order[n], args.sr)
        SecondSong,_ = Load(args.playlist_dir, playlist_order[n+1], args.sr)
    else:
        FirstSong = SecondSong # Remove redundant load operation
        SecondSong,_ = Load(args.playlist_dir, playlist_order[n+1], args.sr)
    
    # Create transition
    fade, x1_trim_faded, x2_trim_faded = Crossfade_Simple(FirstSong, 
                                                          SecondSong, 
                                                          args.fade_style,
                                                          fade_length)
    
    if n==0:
        mix = np.concatenate((FirstSong[:-fade_length], fade, 
                              SecondSong[fade_length:-fade_length]), 
                              axis=0)
    elif n==(len(playlist_order)-2):
        # Use the end of the last song
        mix = np.concatenate((mix, fade, SecondSong[fade_length:]), axis=0)
    else:
        mix = np.concatenate((mix, fade, 
                              SecondSong[fade_length:-fade_length]), 
                              axis=0)

output_name = args.savename + '_' + str(args.fade_time)+ '_' + \
              str(args.fade_style) + '.wav'
output_dir = '/home/ubuntu/DeepBass/data/processed/'
Save(output_dir, output_name, mix, args.sr)