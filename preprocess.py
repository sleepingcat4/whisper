import json
from pathlib import Path
from datasets import Dataset, Audio
from tqdm import tqdm

def _prepare_dataset(batch, feature_extractor, tokenizer):
    audio = batch['audio']
    batch['input_features'] = feature_extractor(
        audio['array'], sampling_rate=audio['sampling_rate']
    ).input_features[0]
    batch['labels'] = tokenizer(batch['text']).input_ids
    return batch

def preprocess_dataset(
    dataset_dir,
    splits=("train", "val"),
    feature_extractor=None,
    tokenizer=None,
    sampling_rate=16000,
):
    dataset_dir = Path(dataset_dir)
    ds = {}

    for split in splits:
        jsonl_file = dataset_dir / f"{split}.jsonl"
        data_items = []

        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Loading {split} JSONL"):
                item = json.loads(line)
                data_items.append({
                    "uuid": item["uuid"],
                    "text": item["text"],
                    "audio": str(dataset_dir / item["path"])
                })

        ds[split] = Dataset.from_list(data_items)
        ds[split] = ds[split].cast_column("audio", Audio(sampling_rate=sampling_rate))
        ds[split] = ds[split].map(
            lambda batch: _prepare_dataset(batch, feature_extractor, tokenizer),
            remove_columns=ds[split].column_names,
            desc=f"Processing {split} dataset"
        )

    return ds
