from datasets import load_dataset, DatasetDict
from transformers import (
    WhisperTokenizer,
    WhisperProcessor,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from datasets import Audio
from dataclasses import dataclass
from typing import Any, Dict, List, Union
 
import torch
import evaluate
