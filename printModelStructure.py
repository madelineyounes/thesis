from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import torch
import librosa
from datasets import load_dataset
import torch.nn as nn

# models view their structure: 
# elgeish/wav2vec2-large-xlsr-53-arabic
# facebook/wav2vec2-large-xlsr-53
# facebook/wav2vec2-base
# facebook/wav2vec2-large-960h

# print("XLSR")
# model = Wav2Vec2ForSequenceClassification.from_pretrained(
#     "facebook/wav2vec2-large-xlsr-53")
# print(model)

model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "wav2vec2-large-xlsr-53-arabic")
print(model)


model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-large-xlsr-53")
print(model)


print("WAV2VEC2 LARGE")
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-large-960h")
print(model)


print("WAV2VEC2 LARGE")
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "log0/wav2vec2-base-lang-id")
print(model)