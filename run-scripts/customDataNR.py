import torch
import torchaudio
import pandas as pd
import pickle
from torch.utils.data import Dataset
import customTransform as T
from torchvision import transforms
from transformers.models.wav2vec2.modeling_wav2vec2 import (Wav2Vec2FeatureExtractor)
import noisereduce as nr

label_list = ['NOR', 'EGY', 'GLF', 'LEV']
label2id, id2label = dict(), dict()
for i, label in enumerate(label_list):
    label2id[label] = str(i)
    id2label[str(i)] = label

def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

def speech_file_to_array_fn(path, target_sampling_rate, norm):
    speech_array, sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    if norm is True:
        speech = match_target_amplitude(speech, -20.0)
    speech = nr.reduce_noise(y=speech, sr=sampling_rate)
    return speech
class CustomDataset(Dataset):
    def __init__(self, csv_fp, data_fp, labels, transform=None, sampling_rate=16000, model_name="facebook/wav2vec2-base", max_length=0.1, norm=False):
        """
        Args:
        csv_fp (string): Path to csv with audio file ids and labels.
        data_fp (string): Path to audio files
        transform: Transform to be performed on audio file
        """
        if transform is None:
            self.transform = transforms.Compose([T.Extractor(model_name, sampling_rate, max_length)])
        else:
            self.transform = transform

        self.data_frame = pd.read_csv(csv_fp, delimiter=',')
        self.data_fp = data_fp
        self.norm = norm
        self.sampling_rate = sampling_rate
        self.labels = labels

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audiopath = self.data_fp + self.data_frame.iloc[idx, 0] + ".wav"
        speech = speech_file_to_array_fn(audiopath, self.sampling_rate, self.norm)
        
        if self.transform:
            speech_features = self.transform(speech)[0]
            speech_mask = self.transform(speech)[1]

        label = int(label2id[self.data_frame.iloc[idx, 1]])

        speech_features = speech_features.float()
        speech_mask = speech_mask.long()
        item = {}
        item['input_values'] = speech_features
        item['attention_mask'] = speech_mask
        item['labels'] = torch.tensor(label)
        return item
