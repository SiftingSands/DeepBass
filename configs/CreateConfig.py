import configparser

def createconfig(config_path):
    config = configparser.ConfigParser()
    config['DEFAULT'] = {'SamplingRate': '16000'}
    
    config['XFade Settings'] = {}
    config['XFade Settings']['Style'] = 'LinearFade' #['HannFade', 'LinearFade', 'Extend'])
    config['XFade Settings']['Time'] = '5'
    
    config['Preprocess'] = {}
    config['Preprocess']['SR window duration'] = 30 # seconds for silence detection
    
    config['IO'] = {}
    config['IO']['FirstSong'] = 'SmallTownBoy.mp3'
    config['IO']['SecondSong'] = 'IMRemix.mp3'
    config['IO']['Load Directory'] = '~/DeepBass/data/raw/Exp1'
    config['IO']['Save Directory'] = '~/home/ubuntu/DeepBass/data/processed/Exp1'
    config['IO']['Save Name'] = 'Exp1'
    config['IO']['Model Weights'] = '~/DeepBass/src/notebooks/wavenet-ckpt/model.ckpt-200000'
    
    with open(config_path, 'w') as configfile:
        config.write(configfile)
        
    return None
