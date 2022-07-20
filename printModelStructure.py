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

print("XLSR")
try:
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        "facebook/wav2vec2-large-xlsr-53")
    print(model)
except:
    print("Cant print XLSR")

print("XLSR ARABIC")
try:
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        "elgeish/wav2vec2-large-xlsr-53-arabic")
    print(model)
except:
    print("Cant print XLSR arabic")

print("WAV2VEC2 LARGE")
try:
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        "facebook/wav2vec2-large-960h")
    print(model)
except: 
    print("Cant print wav2vec large")

print("WAV2VEC2 LID")
try:
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        "log0/wav2vec2-base-lang-id")
    print(model)
except:
    print("Cant print wav2vec lid")
