from transformers import Wav2Vec2FeatureExtractor

class Transform:
    def __init__(self):
        pass

    def __call__(self, x):
        self.set_state()
        x = self.transform(x)
        return x

    def set_state(self):
        pass

    def transform(self, x):
        x = self.do_transform(x)
        return x

    def do_transform(self, x):
        raise NotImplementedError

class Extractor(Transform):
    def __init__(self, base_transformer, sampling_rate, segment_length):
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            base_transformer)
        self.sampling_rate = sampling_rate
        self.max_length = int(self.sampling_rate * (segment_length))

    def do_transform(self, x):
        features = self.feature_extractor(x, sampling_rate=self.sampling_rate, padding='max_length', do_normalize=True,
                                          max_length=self.max_length, return_tensors="pt", return_attention_mask=True, truncation=True)
        print("features extracted")
        return features['input_values'].reshape((-1)), features['attention_mask'].reshape((-1))
