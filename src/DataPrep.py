import os
import argparse
from ingestion.IO_utils import Load
import numpy as np
import errno
from preprocess.SilenceRemoval import SR
from preprocess.np_to_tfrecords import np_to_tfrecords
from preprocess.get_time import get_time
from os import listdir
from os.path import isfile, join
from joblib import Parallel, delayed

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
def Prep(fname, load_dir, sr, duration_thresh, sample_length, crop_style):
    # Remove very long, short, or corrupt audio files.
    # Don't want remixes of a lot of songs (typically long audio files).
    duration = get_time(load_dir, fname)
    if duration >= duration_thresh or duration == False:
        return None
    elif duration < sample_length*3/sr: # give ourselves some buffer with x3
        return None
    else:
        # Load audio only for valid files
        Data = []
        audio, _ = Load(load_dir, fname, sr, verbose=False)
        if crop_style == 'BegEnd':
            begin_audio = SR(audio, 'begin')[0:sample_length]
            end_audio = SR(audio, 'end')[-sample_length:]
            Data.append(begin_audio)
            Data.append(end_audio)
        return (Data)
###############################################################################
"""Script to load in training/validation/test data, preprocess, and convert to
tfrecords format for training the NSynth model.

Example : 
python DataPrep.py ~/Data/EDM/ 4 ~/DeepBass/data/preprocessed/EDM/ EDM -n_cpu=72

"""

parser = argparse.ArgumentParser(description='Load Audio Files')
parser.add_argument('-load_dir', help='Directory of audio files', 
                    action=FullPaths, type=is_dir, required=True)
parser.add_argument('-time', help='Specify the amount of time to crop to',
                    type=float, required=True)
parser.add_argument('-save_dir', help='Directory to save processed audio files',
                    type=str, required=True)
parser.add_argument('-savename', help='Specify the name of the tfrecords file',
                    type=str, required=True)
parser.add_argument('-crop_style', help='Method for temporal cropping', 
                    choices=['BegEnd'], default='BegEnd')
parser.add_argument('-sr', default=16000, help='Specify sampling rate for audio',
                    type=int)
parser.add_argument('-duration_thresh', default=1000, help='Maximum number of \
                    seconds per audio file.', type=float)
parser.add_argument('-n_cpu', default=1, help='Number of CPU threads to use.',
                    type=int)
args = parser.parse_args()

# Create the save folder if it does not exist
if not os.path.exists(args.save_dir):
    try:
        os.makedirs(args.save_dir)
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

filenames = [f for f in listdir(args.load_dir) if isfile(join(args.load_dir,
             f))]

# Number of samples to export
sample_length = int(args.time * args.sr)
Data = []
Data = Parallel(n_jobs=args.n_cpu)(delayed(Prep)(fname, 
                                                 args.load_dir, 
                                                 args.sr,
                                                 args.duration_thresh, 
                                                 sample_length, 
                                                 args.crop_style) \
                                                 for fname in filenames)
# Remove audio snippets that returned None
Data = [x for x in Data if x is not None]
# Merge everything into one list from a list of lists
Data = [item for sublist in Data for item in sublist]
# Remove empty lists
Data = [x for x in Data if x != []]

os.chdir(args.save_dir) # Move directory for saving
np_to_tfrecords(np.stack(Data), None, args.savename, verbose=True)