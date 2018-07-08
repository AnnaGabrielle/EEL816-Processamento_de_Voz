import seaborn as sb 
import matplotlib.pyplot as plt 
import librosa as lb
import python_speech_features as psf
import scipy.io.wavfile as wav
import scipy.signal as sgn
import numpy as np
import math
import operator


class SoundObject:
    word = ''
    mfcc = []
    d_mfcc = []
    d2_mfcc = []
    logFilterBank = []
    audioSignal = []
    audioFiltered = []
    audioRate = 0

    def __init__(self,file,word=None):
        self.word = word
        (self.audioSignal, self.audioRate) = lb.load(file, sr=16000)
        self.remove_noise()
        self.extract_features()
    
    def remove_noise(self):
        self.audioFiltered = sgn.wiener(self.audioSignal) 

    def pre_empashis(self):
        self.audioSignal =  psf.sigproc.preemphasis(self.audioSignal)

    def extract_features(self):
        
        #extract mfcc and normalize
        self.mfcc = lb.feature.mfcc(self.audioFiltered,sr=16000,n_mfcc=13)
        self.d_mfcc = lb.feature.delta(self.mfcc, order=1)
        self.d2_mfcc = lb.feature.delta(self.mfcc, order=2)

        self.mfcc -= (np.mean(self.mfcc, axis=0) + 1e-8)
        self.d_mfcc -= (np.mean(self.d_mfcc, axis=0) + 1e-8)
        self.d2_mfcc -= (np.mean(self.d2_mfcc, axis=0) + 1e-8)        

        #extract logfilterbaks and normalize
        self.logFilterBank = np.transpose(psf.logfbank(self.audioFiltered))
        self.logFilterBank -= (np.mean( self.logFilterBank, axis=0) + 1e-8)
    
    def features(self):
        return [self.mfcc, self.d_mfcc, self.d2_mfcc, self.logFilterBank]

def maxValue(dic):
    return  max(dic.items(), key=operator.itemgetter(1))[0]

def compare(base, audio):
    best = {
        'mfcc': [None,math.inf],
        'd_mfcc': [None,math.inf],
        'd2_mfcc': [None,math.inf],
        'logFilterBank': [None,math.inf]
    }

    words = ['zero', 'um', 'dois', 'tres', 'quatro', 'cinco', 'seis', 'sete', 'oito', 'nove', 'mais', 'menos', 'dividido', 'vezes']
    features = ['mfcc', 'd_mfcc', 'd2_mfcc', 'logFilterBank']
    for word in words:
        for idx, feature in enumerate(features):
            for data in base[word][feature]:
                dist, path = lb.core.dtw(data, audio.features()[idx])
                if(dist[-1][-1]<best[feature][1]):
                    best[feature] = [word,dist[-1][-1]]
    return best

if __name__ == "__main__":
    audios = {
        'zero':[],
        'um':[],
        'dois':[],
        'tres':[],
        'quatro':[],
        'cinco':[],
        'seis':[],
        'sete':[],
        'oito':[],
        'nove':[],
        'mais':[],
        'menos':[],
        'dividido':[],
        'vezes':[]
    }

    audiosFeatures = {
        'zero':{
            'mfcc':[],
            'd_mfcc':[],
            'd2_mfcc':[],
            'logFilterBank':[]
        },
        'um':{
            'mfcc':[],
            'd_mfcc':[],
            'd2_mfcc':[],
            'logFilterBank':[]
        },
        'dois':{
            'mfcc':[],
            'd_mfcc':[],
            'd2_mfcc':[],
            'logFilterBank':[]
        },
        'tres':{
            'mfcc':[],
            'd_mfcc':[],
            'd2_mfcc':[],
            'logFilterBank':[]
        },
        'quatro':{
            'mfcc':[],
            'd_mfcc':[],
            'd2_mfcc':[],
            'logFilterBank':[]            
        },
        'cinco':{
            'mfcc':[],
            'd_mfcc':[],
            'd2_mfcc':[],
            'logFilterBank':[]
        },
        'seis':{
            'mfcc':[],
            'd_mfcc':[],
            'd2_mfcc':[],
            'logFilterBank':[]            
        },
        'sete':{
            'mfcc':[],
            'd_mfcc':[],
            'd2_mfcc':[],
            'logFilterBank':[]            
        },
        'oito':{
            'mfcc':[],
            'd_mfcc':[],
            'd2_mfcc':[],
            'logFilterBank':[]            
        },
        'nove':{
            'mfcc':[],
            'd_mfcc':[],
            'd2_mfcc':[],
            'logFilterBank':[]            
        },
        'mais':{
            'mfcc':[],
            'd_mfcc':[],
            'd2_mfcc':[],
            'logFilterBank':[]            
        },
        'menos':{
            'mfcc':[],
            'd_mfcc':[],
            'd2_mfcc':[],
            'logFilterBank':[]            
        },
        'dividido':{
            'mfcc':[],
            'd_mfcc':[],
            'd2_mfcc':[],
            'logFilterBank':[]            
        },
        'vezes':{
            'mfcc':[],
            'd_mfcc':[],
            'd2_mfcc':[],
            'logFilterBank':[]            
        }
    }

    #inputFile = input("Dataset: ")
    inputFile = 'input.txt'
    with open(inputFile) as file:
        for line in file:
            soundPath, word = line.split()
            audios[word].append(SoundObject(soundPath,word))
            audio = audios[word][-1]
            audiosFeatures[word]['mfcc'].append(np.array(audio.mfcc))
            audiosFeatures[word]['d_mfcc'].append(np.array(audio.d_mfcc))
            audiosFeatures[word]['d2_mfcc'].append(np.array(audio.d2_mfcc))
            audiosFeatures[word]['logFilterBank'].append(np.array(audio.logFilterBank))
    
    audioP = './data/dois4.wav'
    testSound = SoundObject(audioP)

    match = compare(audiosFeatures, testSound)
    print(audioP)
    print(match)