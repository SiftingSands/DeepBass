# Deep Bass
Automatic content driven cross fading between subsequent songs using a Wavenet autoencoder (NSynth).

This currently works on two audio files at a time. The ending of the first song and the beginning of the second song is isolated.

[Trim](static/Audio_trim.pdf)

For simple cross fading, the audio is multiplied by a ramp function and summed to generate the mixed audio.

[Ramped](static/xfade.pdf)

[Crossfaded](static/mixed.pdf)

[Play Audio](https://www.youtube.com/watch?v=uJoLrR6eXBQ)

For the cross fading with NSynth, lower dimensional embeddings are created. Cross fading is done on the embeddings instead of the raw audio, and this modified embedding is fed through the decoder to generate the resulting audio.

[Encodings](static/NSynth_enc.pdf)

[NSynth cross fade (click for audio)](static/NSynth_xfade.pdf)

[Play Audio](https://www.youtube.com/watch?v=pmEGEVNAf4g)

See this [presentation](http://bit.ly/2DZyzni) for further details.

Table of contents
=================

<!--ts-->
   * [Table of contents](#table-of-contents)
   * [Setup](#setup)
   * [Usage](#usage)
      * [Config File](#example-config-file)
      * [Simple Cross Fading](#run-simple-cross-fading)
      * [NSynth Cross Fading](#run-nsynth-cross-fading)
      * [Training](#train-the-model-from-a-checkpoint)
   * [Background](#model-background)
<!--te-->

Setup
=====
Clone repository and update python path
```
repo_name=DeepBass 
username=SiftingSands
git clone https://github.com/$username/$repo_name
cd $repo_name
echo "export $repo_name=${PWD}" >> ~/.bash_profile
echo "export PYTHONPATH=$repo_name/src:${PYTHONPATH}" >> ~/.bash_profile
source ~/.bash_profile
```
Create virtual environment (commands below are for Anaconda, otherwise follow https://docs.python.org/3/tutorial/venv.html). Install dependent packages from requirements.txt. "timbral_models" needed to be installed from source.
```
conda create --name DeepBass
pip install -r /<path>/DeepBass/build/requirements.txt
git clone https://github.com/AudioCommons/timbral_models.git
cd timbral_models
pip install .
```

## Required Packages
- 'numpy'
- 'librosa'
- 'streamlit'
- 'matplotlib'
- 'tensorflow-gpu'
- 'timbral_models'
- 'soundfile'
- 'scipy'
- 'sklearn'
- 'essentia'
- 'joblib'

Usage
=====

Work flow is typically as follows:
1. Setup the config file with the data inputs (songs) and cross fading parameters.
2. Run the cross fading method.
3. Evaluate the output audio. 
4. An empirical 'roughness' measure is also calculated and output to a text file. See Section 1.5 of [timbral_models documentation](https://www.audiocommons.org/assets/files/AC-WP5-SURREY-D5.2%20First%20prototype%20of%20timbral%20characterisation%20tools%20for%20semantically%20annotating%20non-musical%20content.pdf) for details on roughness.
5. Alternatively, a similar measure called Sethares dissonance can be calculated over the peak frequencies over time. [Example Python script.](https://gist.github.com/endolith/3066664)

Example config file
-----
```
[DEFAULT]
samplingrate = 16000

[NSynth XFade Settings]
Style = LinearFade
time = 4

[Simple XFade Settings]
Style = Linear
time = 4

[Preprocess]
SR window duration = 30

[IO]
firstsong = Song1.mp3
secondsong = Song2.mp3
load directory = /home/ubuntu/DeepBass/data/raw/Exp6
save directory = /home/ubuntu/DeepBass/data/processed/Exp6_Retrained
save name = Exp6
model weights = /home/ubuntu/nsynth_train/model.ckpt-320000

[Plot]
Flag = True
```

Run Simple Cross Fading
-----
- Create a 'config.ini' either manually (place in /configs/) or by editing and running '/configs/CreateConfig.py'
1. Loads the first and second songs per the 'load directory', 'firstsong', and 'secondsong' in the config.ini
2. Detects if the ending and beginning has silence within a 'SR window duration' in seconds
3. Trims the audio to 'time' length (seconds) snippets under 'Simple XFade Settings'
4. Applys a linear crossfade between the beginning and ending snippets (different crossfade ramping functions are available)
5. Saves the crossfaded audio file per 'save name' and 'save directory'
```
cd ~/DeepBass/src
python main_simple.py -config_fname=<your-config-file-name>
```

Run NSynth Cross Fading
-----
- Create a 'config.ini' either manually (place in /configs/) or by editing and running '/configs/CreateConfig.py'
1. Loads the first and second songs per the 'load directory', 'firstsong', and 'secondsong' in the config.ini
2. Detects if the ending and beginning has silence within a 'SR window duration' in seconds
3. Trims the audio to 'time' length (seconds) snippets under 'NSynth XFade Settings'
4. Load the weights for the neural network from 'model weights'
5. Calculate encoding from the audio snippets
6. Combine both encodings
7. Synthesize mixed audio from encoding
8. Saves the crossfaded audio file per 'save name' and 'save directory'
```
cd ~/DeepBass/src
python main_NSynth.py -config_fname=<your-config-file-name>
```

Train the model from a checkpoint
-----
1. Create a folder with all of the audio examples that you want to train on
2. Download previous checkpoint for the model (for example http://download.magenta.tensorflow.org/models/nsynth/wavenet-ckpt.tar)
3. Trim the beginning and endings of the songs and convert the data to the 'tfrecords' format
4. Run the training script with the path to the training data and model checkpoint. Currently no multi-GPU capability, see https://github.com/tensorflow/magenta/issues/625
```
cd ~/DeepBass/src
python DataPrep.py -load_dir=<path-to-audio-files> -time=4 -save_dir=<path-typically-in-/DeepBass/Data/preprocessed> -savename=<name-for-tfrecords-file> -n_cpu=<number-of-cpu-threads>
python train.py --train_path=<path-to-tfrecords> --total_batch_size=6 --logdir=<path-to-checkpoint>
```

Model Background
=====
The NSynth model from Google's Magenta team was used to create the audio mixes (https://magenta.tensorflow.org/nsynth). Starting from their published checkpoint, the model was further trained on 4 second snippets from the beginning and endings of over 500 songs in the EDM genre. Therefore, the 16 kHz downsampling and 8-bit Mu encoding is still present. Thanks to the Magenta team for making their work open source.
