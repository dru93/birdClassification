''' 
- Visualization of soundwaves and spectograms
'''

import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
from tqdm import tqdm
import numpy as np

# keep original directory
originalDir = os.getcwd()

# Set directory of wav files to plot
d = 'Data/wavPlots/'
os.chdir(d)

# Import wavs
raw = []
dir_length = len(os.listdir(os.getcwd()))
print('Importing wav files...')
#pbar = tqdm(total=dir_length) # Specify the progressBar
for filename in tqdm(os.listdir(os.getcwd())):    
    x, sr = librosa.load(filename)
    raw.append(x)

# reshape raw to a np.array
raw = np.array(raw)

# Function for plotting waves    
def ploting_wave(sound):    
    length = len(sound)
    i = 1
    plt.figure()
    for freq in sound:        
        plt.subplot(length, 1, i)
        librosa.display.waveplot(freq, sr = 22050)    
        i += 1
    plt.show()
    
# Function for spectograms        
def plot_specgram(sound):
    i = 1
    length = len(sound)
    plt.figure()
    for f in sound:
        plt.subplot(length, 1, i)
        specgram(f, Fs = 22050)
        i += 1
    plt.show()    

#PLOTTING
# plot the first n waves
flag1 = input('Show soundwaves? [y/n]        >_ ')
if flag1 == 'Y' or flag1 == 'y' or flag1 == 'yes':    
    f1 = input('Specify how many soundwaves to visualize: (number)  >_')
    ploting_wave(raw[0:int(f1)])

# Plot the five n spectograms
flag2 = input('Show spectograms? [y/n]       >_ ')
if flag2 == 'Y' or flag2 == 'y' or flag2 == 'yes':
    f2 = input('Specify how many spectograms to visualize: (number)  >_')
    plot_specgram(raw[0:int(f2)])

resultsDir = originalDir + '/Results'
if not os.path.exists(resultsDir):
    os.makedirs(resultsDir)

# return to original directory
os.chdir(originalDir)