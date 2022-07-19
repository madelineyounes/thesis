import torch
import librosa
from datasets import load_dataset
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import torch.nn as nn


def map_to_array(example):
    speech, _ = librosa.load(example, sr=16000, mono=True)
    return speech

# load a demo dataset and read audio files

# data1, sr1 = torchaudio.load("eng1.wav")

audio = map_to_array("english.wav")

model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-sid")

"facebook/wav2vec2-base-960h"

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-sid")


inputs = feature_extractor(audio, sampling_rate=16000, padding='max_length', return_tensors="pt", max_length=20*16000, truncation = True, return_attention_mask = True)
print(inputs)

model.classifier = nn.Linear(256,107)

# for params in model.wav2vec2:

logits = model(**inputs).logits

print(logits.shape)
predicted_ids = torch.argmax(logits, dim=-1)


from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
import torch

# load model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
print(processor)
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")


def map_to_array(example):
    speech, _ = librosa.load(example, sr=16000, mono=True)
    return speech


activation_dict = {}


def get_activation(layer_name):

    def hook(model, input, output):
        activation_dict[layer_name + gpu_id] = output.detach()
    return hook


# load a demo dataset and read audio files

# data1, sr1 = torchaudio.load("eng1.wav")

audio1 = map_to_array("/Users/myounes/Documents/Code/thesis/testfiles/--YRssvbUts_000027-000603.wav")
audio2 = map_to_array("/Users/myounes/Documents/Code/thesis/testfiles/--zhvFBThyM_000027-001885.wav")


audio_list = [audio1, audio2]
# load dummy dataset and read soundfiles
#  ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")

# tokenize
input_values = processor(audio_list, sampling_rate=16000, padding='max_length', return_tensors="pt",
                         max_length=1*16000, truncation=True, return_attention_mask=True)  # Batch size 1
print(input_values)

input_values['input_values'] = input_values['input_values']
input_values['attention_mask'] = input_values['attention_mask']

# input_values.cuda().contiguous()
# model.dropout = nn.Identity()
model.lm_head = nn.Sequnetial(
    nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 3))

model.wav2vec2.encoder.layers[22].final_layer_norm.register_forward_hook(
    get_activation('layer_22_out'))
model.wav2vec2.encoder.layers[10].final_layer_norm.register_forward_hook(
    get_activation('layer_10_out'))

print(model)

# retrieve logits
print(activation_dict)
logits = model(**input_values).logits
print(activation_dict)

activation_dict.layer_10_out


logits = model(**input_values).logits
print(activation_dict)


# print(logits.shape)
# take argmax and decode
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)
