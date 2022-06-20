"""
filter_module.py 

Madeline Younes Arabic DID Thesis 
File contains a functions for filtering and normalising noisy audio files. 
"""
from hashlib import new
from scipy.io import wavfile
from scipy.signal import butter, lfilter
from pydub import AudioSegment
import noisereduce as nr


def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def noise_filter(rate, data):
    # perform noise reduction
    reduced_noise = nr.reduce_noise(y=data, sr=rate)
    return reduced_noise; 

def band_pass_filter(rate, data): 
    lowcut = 50
    highcut = 8000
    order = 10
    butter_bandpass_filter(data, lowcut, highcut, rate, order)

def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

def export_file(filename):
    new_filename = filename.removesuffix('.wav') + 'myfilter_preprocessed.wav'
    rate, data = wavfile.read(filename)
    filtered = band_pass_filter(rate, data)
    reduced_noise = noise_filter(rate, data)
    wavfile.write(new_filename, rate, filtered)
    reduced_noise_segment = AudioSegment.from_wav(new_filename)

    normalised_sound = match_target_amplitude(reduced_noise_segment, -25.0)
    normalised_sound.export(new_filename, format="wav")
    return new_filename;


def test_preprocessing(): 
    export_file('--YRssvbUts_000027-000603.wav')
    export_file('--zhvFBThyM_000027-001885.wav')

test_preprocessing()
