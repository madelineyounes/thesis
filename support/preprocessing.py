"""
filter_module.py 

Madeline Younes Arabic DID Thesis 
File contains a functions for filtering and normalising noisy audio files. 
"""
from hashlib import new
from scipy.io import wavfile
from scipy.signal import butter, sosfilt
import matplotlib.pyplot as plt
from pydub import AudioSegment
import numpy as np
import noisereduce as nr
import matplotlib.pyplot as plt


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter(order, [lowcut, highcut], fs=fs, btype='band', output='sos')
    y = sosfilt(sos, data)
    samples = len(y)
    return y, samples

def noise_filter(rate, data):
    # perform noise reduction
    reduced_noise = nr.reduce_noise(y=data, sr=rate)
    return reduced_noise; 

def band_pass_filter(rate, data): 
    nyq = 0.5 * rate 
    lowcut = 300 
    highcut = 2000
    order = 5
    return butter_bandpass_filter(data, lowcut, highcut, rate, order)

def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

def export_file(filename):
    new_filename = filename.removesuffix('.wav') + 'myfilter_preprocessed.wav'
    rate, data = wavfile.read(filename)
    reduced_noise = noise_filter(rate, data)
    filtered, samples = band_pass_filter(rate, reduced_noise)
    wavfile.write(new_filename, rate, np.int16(filtered))
    reduced_noise_segment = AudioSegment.from_wav(new_filename)
    normalised_sound = match_target_amplitude(reduced_noise_segment, -20.0)
    normalised_sound.export(new_filename, format="wav")

    r, normal = wavfile.read(new_filename)
    # plot relevent data
    plt.figure(1)
    p1 = plt.plot(data, label='Orignal Audio Signal')
    p2 = plt.plot(reduced_noise, label='Reduce Noise function filtered signal')
    plt.ylabel('Magnitude (db)')
    plt.xlabel('Samples')
    plt.title('Filtered using bandpass filter')
    plt.legend(handles=[p1[0], p2[0]])


    plt.figure(2)
    p1 = plt.plot(reduced_noise, label='Reduce Noise function filtered signal')
    p2 = plt.plot(filtered, label='Bandpassed filtered signal')
    plt.ylabel('Magnitude (db)')
    plt.xlabel('Samples')
    plt.title ('Filtered using reduce noise')
    plt.legend(handles=[p1[0], p2[0]])
    plt.show()

    return new_filename;


def test_preprocessing(): 
    export_file('../testfiles/--YRssvbUts_000027-000603.wav')
    export_file('../testfiles/--zhvFBThyM_000027-001885.wav')

test_preprocessing()
