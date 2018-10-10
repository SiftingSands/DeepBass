# Deep Bass
Automatic content driven cross fading between subsequent songs using a Wavenet autoencoder following the Magenta NSynth model (https://magenta.tensorflow.org/nsynth).

Table of contents
=================

<!--ts-->
   * [Table of contents](#table-of-contents)
   * [Setup](#setup)
   * [Requisites](#requisites)
   * [Usage](#usage)
      * [Config File](#example-config-file)
      * [Simple Cross Fading](#run-simple-cross-fading)
      * [NSynth Cross Fading](#run-nsynth-cross-fading)
      * [Training](#train-the-model-from-a-checkpoint)
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
Install dependent packages. "timbral_models" needed to be installed from source.
```
pip install -r /<path>/DeepBass/build/requirements.txt
git clone https://github.com/AudioCommons/timbral_models.git
cd timbral_models
pip install .
```

Requisites
=====
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
python main_simple.py
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
python main_NSynth.py
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
