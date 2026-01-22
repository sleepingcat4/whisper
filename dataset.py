from preprocess import preprocess_dataset
from imports import *

model_id = "openai/whisper-medium"
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id)
tokenizer = WhisperTokenizer.from_pretrained(model_id, language="Hebrew", task="transcribe")

dataset_dir = "/home/ammar/hebrew/train/data"
output_dir = "/home/ammar/hebrew/train/dataset"

ds = preprocess_dataset(
    dataset_dir=dataset_dir,
    feature_extractor=feature_extractor,
    tokenizer=tokenizer,
    sampling_rate=16000
)


hebrew_dataset = DatasetDict(ds)

hebrew_dataset.save_to_disk(output_dir)
print(f"Processed dataset saved to {output_dir}")
