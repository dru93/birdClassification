'''
- Import data
- Feature extraction(train/test): mfccs, chroma, mel, contrast, tonnetz
'''

import os
import sys
import random
import librosa
import numpy as np
from tqdm import tqdm

# keep original directory
originalDir = os.getcwd()

#  high pass all wav files
if not os.path.exists(originalDir + '/Data/HP/'):
    import highPassFilter

# Set working directory
path = 'Data/'
os.chdir(path)

# Import truth csv
my_data = np.matrix(np.genfromtxt('warblrb10k_public_metadata.csv', delimiter=',' , dtype=str , skip_header=1 ))

# Set directory for wav files
d = 'HP/'
os.chdir(d)

# Import wavs
raw = []
name = []
dir_length=len(os.listdir(os.getcwd()))
print('Importing wav files...')
for filename in tqdm(os.listdir(os.getcwd())):
    x, sr = librosa.load(filename)
    raw.append(x)
    name.append(filename.split('.')[0])

# reshape raw to a np.array
raw = np.array(raw)

# labeled groundtruth of wav files
ind = np.where(my_data[:,0] == name)[0] # find indexes of our wav in my_data
truth = np.matrix(my_data[ind]) # make truth only with these indexes
# Fix the order of truth
tmp = np.argsort(np.where(truth[:,0] == name)[1]) # index the mapping 
truth = truth[tmp] # set the actural order

# Feature extraction with librosa
def extract_feature(sound):
    stft = np.abs(librosa.stft(sound))
    mfccs = np.mean(librosa.feature.mfcc(y = sound, sr = sr, n_mfcc = 40).T, axis = 0)
    chroma = np.mean(librosa.feature.chroma_stft(S = stft, sr = sr).T, axis = 0)
    mel = np.mean(librosa.feature.melspectrogram(sound, sr = sr).T, axis = 0)
    contrast = np.mean(librosa.feature.spectral_contrast(S = stft, sr = sr).T, axis = 0)
    tonnetz = np.mean(librosa.feature.tonnetz(y = librosa.effects.harmonic(sound), sr = sr).T, axis = 0)
    return mfccs, chroma, mel, contrast, tonnetz

# Concatenate all names , features and labels for each wav file
def concatenate(sound , groundTruth):
    features, labels , names = np.empty((0,193)), np.empty(0) , np.empty(0)
    print('Concatenating: names, features and labels for each wav file....')
    pbar = tqdm(total=len(sound)) # Specify the progressBar    
    for fn,name,lab in zip(sound,groundTruth[:,0],groundTruth[:,1]):        
        mfccs, chroma, mel, contrast, tonnetz = extract_feature(fn)
        ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
        features = np.vstack([features,ext_features])
        labels = np.append(labels, lab)
        names = np.append(names, name)
        pbar.update()
    pbar.close()    
    return np.array(names, dtype =str), np.array(features), np.array(labels, dtype = np.int)

names, features, label = concatenate(raw, truth)

# create Results directory and save labels and features there
resultsDir = originalDir + '/Results'
if not os.path.exists(resultsDir):
    os.makedirs(resultsDir)

np.save(resultsDir + '/features', features)
np.save(resultsDir + '/labels', label)

# return to original directory
os.chdir(originalDir)