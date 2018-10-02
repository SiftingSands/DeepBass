import os
import argparse
from ingestion.IO_utils import Load
import numpy as np
import errno
from preprocess.SilenceRemoval import SR
from preprocess.np_to_tfrecords import np_to_tfrecords
from os import listdir
from os.path import isfile, join

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
"""Script to load in training/validation/test data, preprocess, and convert to
tfrecords format for training the NSynth model.

Example : 
python DataPrep.py ~/DeepBass/data/raw/EDM_Test 4 \
~/DeepBass/data/preprocessed/EDM_Test example

"""

parser = argparse.ArgumentParser(description='Load Audio Files')
parser.add_argument('load_dir', help='Directory of audio files', 
                    action=FullPaths, type=is_dir)
parser.add_argument('time', help='Specify the amount of time to crop to',
                    type=float)
parser.add_argument('save_dir', help='Directory to save processed audio files',
                    type=str)
parser.add_argument('savename', help='Specify the name of the tfrecords file',
                    type=str)
parser.add_argument('-crop_style', help='Method for temporal cropping', 
                    choices=['BegEnd', 'Middle'], default='BegEnd')
parser.add_argument('-sr', default=16000, help='Specify sampling rate for audio',
                    type=int)
args = parser.parse_args()

filenames = [f for f in listdir(args.load_dir) if isfile(join(args.load_dir,
             f))]

# Number of samples to export
sample_length = int(args.time * args.sr)
Data = []
for fname in filenames:
    # Load Audio
    audio, _ = Load(args.load_dir, fname, args.sr)
    if args.crop_style == 'BegEnd':
        begin_audio = SR(audio, 'begin')[0:sample_length]
        end_audio = SR(audio, 'end')[-sample_length:]
        Data.append(begin_audio)
        Data.append(end_audio)
        
# Create the save folder if it does not exist
if not os.path.exists(args.save_dir):
    try:
        os.makedirs(args.save_dir)
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

os.chdir(args.save_dir) # Move directory for saving
np_to_tfrecords(np.array(Data), None, args.savename, verbose=True)