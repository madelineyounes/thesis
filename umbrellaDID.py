from datasets import load_dataset, Audio
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from tqdm.auto import tqdm
import tensorflow as tf


# check how to load in adi17
dataset = load_dataset("adi17", name="adi17", split="train")
dataset = dataset.cast_column("audio", Audio(
    sampling_rate=speech_recognizer.feature_extractor.sampling_rate))


model_name = 'facebook/wav2vec2-xls-r-1b'
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
classifier = pipeline("audio-classification", model=model,
                      tokenizer=tokenizer)  # select the type of classifier
pt_batch # data preposing 


pt_outputs = model(**pt_batch)
tf_predictions = tf.nn.softmax(tf_outputs.logits, axis=-1)
