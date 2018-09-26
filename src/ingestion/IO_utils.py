import librosa
import os

def Load(file_dir, file_name, sr):
    # Import as non-stereo audio and at a target sampling rate, if desired
    x, sr = librosa.load(os.path.join(file_dir, file_name), sr=sr, mono=True)
    print('Duration: {:.2f}s, {} samples'.format(x.shape[-1] / sr, x.size))
    return x, sr

def Save(file_dir, file_name, audio, sr):
    librosa.output.write_wav(os.path.join(file_dir, file_name), audio, sr)
