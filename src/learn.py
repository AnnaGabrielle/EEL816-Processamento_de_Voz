import matplotlib.pyplot as plt 
import librosa as lb
import python_speech_features as psf
import scipy.io.wavfile as wav
import scipy.signal as sgn
import numpy as np
import math
import operator
import pyaudio
from array import array
import sys
import wave
import os
from casting import *
import time as t

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


class Recoder:    
    

    def record(self):
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        CHUNK = 128
        RECORD_SECONDS = 1.5
        WAVE_OUTPUT_FILENAME = "temp.wav"
         
        audio = pyaudio.PyAudio()
        os.system('cls' if os.name == 'nt' else 'clear')
        _ = input("Precione Enter para começar a gravação de ~ 1.5s")

        # start Recording
        stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
        print("recording...")
        frames = []
         
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        print("finished recording")
         
        # stop Recording
        stream.stop_stream()
        stream.close()
        audio.terminate()
         
        waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()

def maxValue(dic):
    return  max(dic.items(), key=operator.itemgetter(1))[0]

def voteResult(dic):
    res = {}
    result = [None,0]
    for word, votes in dic.items():

        if(votes[0] in res):
            res[votes[0]]+=1
        else:
            res[votes[0]]=1
    for word, votes in res.items():
        if(votes>=result[1]):
            result = [word,votes]
    return result

def compare(base, audio):
    best = {
        'mfcc': [None,math.inf],
        'd_mfcc': [None,math.inf],
        'd2_mfcc': [None,math.inf],
        'logFilterBank': [None,math.inf]
    }

    words = ['zero', 'um', 'dois', 'tres', 'quatro', 'cinco', 'seis', 'sete', 'oito', 'nove', 'mais', 'menos', 'dividido', 'vezes','igual']
    features = ['mfcc', 'd_mfcc', 'd2_mfcc', 'logFilterBank']
    for word in words:
        for idx, feature in enumerate(features):
            for data in base[word][feature]:
                dist, path = lb.core.dtw(data, audio.features()[idx])
                if(dist[-1][-1]<best[feature][1]):
                    best[feature] = [word,dist[-1][-1]]
    return voteResult(best)

def mainFunction(audiosFeatures):
    
    rec = Recoder()
    play = True
    equation = []#por extenso.
    while(play):
        
        rec.record()
        
        audio = SoundObject('temp.wav')
        word = compare(audiosFeatures, audio)
        os.remove('temp.wav')

        print("Palavra reconhecida", word)
        t.sleep(1)

        if(word[0] != "igual"):
            equation.append(word[0])#passar como lista direto
        else:
            play = False

    print("Equação", equation)
    c = TextCalculator()
    (op,x,y) = c.data_extract(equation)#passar como lista direto
    print(c.calculator(op,x,y))



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
        'vezes':[],
        'igual':[]
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
        },
        'igual':{
            'mfcc':[],
            'd_mfcc':[],
            'd2_mfcc':[],
            'logFilterBank':[]            
        }
    }

    #inputFile = input("Dataset: ")
    inputFile = 'train.txt'
    with open(inputFile) as file:
        for line in file:
            soundPath, word = line.split()
            audios[word].append(SoundObject(soundPath,word))
            audio = audios[word][-1]
            audiosFeatures[word]['mfcc'].append(np.array(audio.mfcc))
            audiosFeatures[word]['d_mfcc'].append(np.array(audio.d_mfcc))
            audiosFeatures[word]['d2_mfcc'].append(np.array(audio.d2_mfcc))
            audiosFeatures[word]['logFilterBank'].append(np.array(audio.logFilterBank))
    
    #a = SoundObject('./data/test/cinco/cinco18.wav')
    #r = Recoder()
    #r.record()
    #a = SoundObject('./data/test/igual/igual20.wav')
    #print(compare(audiosFeatures,a)[0])
    mainFunction(audiosFeatures)
