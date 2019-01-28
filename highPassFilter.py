'''
- High passes a .wav file at 2kHz
'''

import os
import glob
import numpy as np
from tqdm import tqdm
from scipy import signal
from scipy.io import wavfile

# keep original directory
originalDir = os.getcwd()

# Set working directory
path = 'Data/smallset/'
os.chdir(path)

# create directory that high passed wavs will be saved
newDir =  originalDir + '/Data/HP/'
if not os.path.exists(newDir):
    os.makedirs(newDir)
else: # if directiory already exists, delete all files in it
	for root, dirs, files in os.walk(newDir):
		for f in files:
			os.unlink(os.path.join(root, f))

# Import wavs
dir_length=len(os.listdir(os.getcwd()))
print('\nHigh passing wav files...')
for filename in tqdm(os.listdir(os.getcwd())):
	sr,x =  wavfile.read(filename)
	b = signal.firwin(101, cutoff = 2000, nyq = sr, pass_zero = False)
	x = signal.lfilter(b, [1.0], x)
	wavfile.write(newDir + filename, sr, x.astype(np.int16))

# return to original directory
os.chdir(originalDir)