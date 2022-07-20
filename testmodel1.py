from transformers import Wav2Vec2ForSequenceClassification
model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-sid")
print(model)
