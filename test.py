from torchvision import transforms
import customTransform as T
from torch.utils.data import Dataset
import pickle
import pandas as pd
import torchaudio
import torch
import fnmatch
import os


dir_path = "/Users/myounes/Documents/Code/thesis_files/dev_segments/_FBO2f3kW5Q_000136-000568.wav"
target_sampling_rate = 16000
speech_array, sampling_rate = torchaudio.load(dir_path, format="wav")
resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
speech = resampler(speech_array).squeeze().numpy()
print(speech)
