from datasets import load_dataset, DatasetDict, load_from_disk, Audio
from transformers import (
    WhisperTokenizer,
    WhisperProcessor,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from dataclasses import dataclass
from typing import Any, Dict, List, Union 
import torch
import evaluate
from tqdm import tqdm
import warnings
import logging
from transformers import logging as hf_logging
