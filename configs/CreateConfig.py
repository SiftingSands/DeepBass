import configparser

def createconfig(config_path):
    config = configparser.ConfigParser()
    config['DEFAULT'] = {'SamplingRate': '16000'} # Needs to be 16k for NSynth
    
    config['NSynth XFade Settings'] = {}
    config['NSynth XFade Settings']['Style'] = 'LinearFade' #['HannFade', 'LinearFade', 'Extend'])
    config['NSynth XFade Settings']['Time'] = '4' # seconds to cross fade
    
    config['Simple XFade Settings'] = {}
    config['Simple XFade Settings']['Style'] = 'Linear' #['Linear', 'Sigmoid', 'Random_Linear'])
    config['Simple XFade Settings']['Time'] = '4' # seconds to cross fade
    
    config['Preprocess'] = {}
    config['Preprocess']['SR window duration'] = 30 # seconds for silence detection
    
    config['IO'] = {}
    config['IO']['FirstSong'] = 'Song1.mp3'
    config['IO']['SecondSong'] = 'Song2.mp3'
    config['IO']['Load Directory'] = '~/DeepBass/data/raw/Exp1'
    config['IO']['Save Directory'] = '~/home/ubuntu/DeepBass/data/processed/Exp1'
    config['IO']['Save Name'] = 'Exp1'
    config['IO']['Model Weights'] = '~/DeepBass/src/notebooks/wavenet-ckpt/model.ckpt-200000'
    
    config['Plot'] = {}
    config['Plot']['Flag'] = 'True'
    
    with open(config_path, 'w') as configfile:
        config.write(configfile)
        
    return None
