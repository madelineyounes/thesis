from hashlib import new
from scipy.io import wavfile
from scipy.signal import butter, lfilter
from pydub import AudioSegment
import numpy as np
import noisereduce as nr
import sounddevice



lo, hi = 300, 3400
sr, y = wavfile.read('/Users/myounes/Documents/Code/thesis/testfiles/--YRssvbUts_000027-000603.wav')
b, a = butter(N=6, Wn=[2*lo/sr, 2*hi/sr], btype='band')
x = lfilter(b, a, y)
# Convert to normalized 32 bit floating point
normalized_x = x / np.abs(x).max()
wavfile.write('off_plus_noise_filtered.wav', sr,
              normalized_x.astype(np.float32))
