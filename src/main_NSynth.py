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

python main_NSynth.py ~/DeepBass/data/raw/Exp1 Ladonna.mp3 IMRemix.mp3 ~/DeepBass/src/notebooks/wavenet-ckpt/model.ckpt-200000 HannFade 7 /home/ubuntu/DeepBass/data/processed Exp1

"""

# Directory where mp3s are stored
parser = argparse.ArgumentParser(description='Load Audio Files')
parser.add_argument('playlist_dir', help='Directory of playlist with audio \
                    files', action=FullPaths, type=is_dir)
parser.add_argument('FirstSong', help='Filename of for the first song')
parser.add_argument('SecondSong', help='Filename of for the second song')
parser.add_argument('model_dir', help='Directory of the pretrained NSynth model',
                    type=str)
parser.add_argument('fade_style', help='Method for cross fading', 
                    choices=['HannFade', 'LinearFade', 'Extend'])
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

# Alphabetical order for now and ignore subdirectories
playlist_order = [f for f in os.listdir(args.playlist_dir) if \
                  os.path.isfile(os.path.join(args.playlist_dir, f))]
fade_length = int(args.fade_time * args.sr) # number of samples for fade

# Load Songs
FirstSong, _ = Load(args.playlist_dir, args.FirstSong, args.sr)
SecondSong, _ = Load(args.playlist_dir, args.SecondSong, args.sr)

# Remove any silence at the end of the first song
# and the beginning of the second song
t_snip = 30 # interrogation length in seconds
end_index = SR(FirstSong, 'end', t_snip=t_snip)
end_index = int(t_snip*args.sr - end_index) # change index reference frame 
start_index = SR(SecondSong, 'begin', t_snip=t_snip)
FirstSong = FirstSong[:-end_index]
SecondSong = SecondSong[start_index:]

# Create transition
xfade_audio, x1_trim, x2_trim, enc1, enc2 = NSynth(FirstSong,
                                                   SecondSong,
                                                   args.fade_style,
                                                   fade_length,
                                                   args.model_dir,
                                                   args.save_dir,
                                                   args.savename,
                                                   args.rep)

# Save encodings of the audio
np.save('begin_enc' + '.npy', enc1)
np.save('end_enc' + '.npy', enc2)

# Save mu encoded trim segments for reference
Save(args.save_dir, 'end.wav', x1_trim, args.sr)
Save(args.save_dir, 'begin.wav', x2_trim, args.sr)

# Create the mix between the two songs
if args.fade_style == 'Dilation':
    transition = np.concatenate(xfade_audio)
    Mix = np.concatenate((FirstSong, transition, SecondSong), axis=0)
    Save(args.save_dir, args.savename + 'Full', Mix, args.sr)
