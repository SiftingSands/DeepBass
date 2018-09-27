import os
import argparse
from ingestion.IO_utils import Load, Save
from model.crossfade_NSynth import NSynth
from preprocess.SilenceRemoval import SR
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

python main_NSynth.py ~/DeepBass/data/raw/EDM_Test ~/DeepBass/src/notebooks/wavenet-ckpt/model.ckpt-200000 Dilation 1 /home/ubuntu/DeepBass/data/processed test -rep=3

"""

# Directory where mp3s are stored
parser = argparse.ArgumentParser(description='Load Audio Files')
parser.add_argument('playlist_dir', help='Directory of playlist with audio \
                    files', action=FullPaths, type=is_dir)
parser.add_argument('model_dir', help='Directory of the pretrained NSynth model',
                    type=str)
parser.add_argument('fade_style', help='Method for cross fading', 
                    choices=['HannFade', 'LinearFade', 'Dilation'])
parser.add_argument('fade_time', help='Specify the cross fading duration',
                    type=float)
parser.add_argument('save_dir', help='Directory to save the audio files', 
                    action=FullPaths, type=is_dir)
parser.add_argument('savename', help='Specify the prefix for saving the playlist',
                    type=str)
parser.add_argument('-rep', help='Number of repeats for dilation option', 
                    default = 3, type=int)
parser.add_argument('-sr', help='Audio sampling rate, must be 16kHz for NSynth', 
                    default = 16000, type=int)
args = parser.parse_args()

# Alphabetical order for now
playlist_order = os.listdir(args.playlist_dir)

fade_length = int(args.fade_time * args.sr) # number of samples for fade

for n in range(len(playlist_order)-1):
    # Load Songs
    if n==0:
        FirstSong,_ = Load(args.playlist_dir, playlist_order[n], args.sr)
        SecondSong,_ = Load(args.playlist_dir, playlist_order[n+1], args.sr)
        
    else:
        FirstSong = SecondSong # Remove redundant load operation
        SecondSong,_ = Load(args.playlist_dir, playlist_order[n+1], args.sr)

    # Remove any silence at the end of the first song
    # and the beginning of the second song
    end_index = SR(FirstSong, 'end')
    start_index = SR(SecondSong, 'begin')
    FirstSong = FirstSong[:end_index]
    SecondSong = SecondSong[start_index:]
    
    # Create transition
    xfade_audio, x1_trim_mu, x2_trim_mu, enc1, enc2 = NSynth(FirstSong, 
                                                             SecondSong, 
                                                             args.fade_style,
                                                             fade_length,
                                                             args.model_dir,
                                                             args.save_dir,
                                                             args.savename,
                                                             args.rep)
    
    # Save encodings of the audio
    np.save(str(n) + '_begin_enc' + '.npy', enc1)
    np.save(str(n) + '_end_enc' + '.npy', enc2)
    
    # Save mu encoded trim segments for reference
#    Save(args.save_dir, 'end.wav', x1_trim_mu, args.sr)
#    Save(args.save_dir, 'begin.wav', x2_trim_mu, args.sr)
    
    # Create the mix between the two songs
    if args.fade_style == 'Dilation':
        transition = np.concatenate(xfade_audio)
        Mix = np.concatenate((FirstSong, transition, SecondSong), axis=0)
        Save(args.save_dir, args.savename + 'Full', Mix, args.sr)
    
    print(str(n))
    