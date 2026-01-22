from imports import *
from collator import DataCollatorSpeechSeq2SeqWithPadding
from metrics import compute_metrics

warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)


model_id = 'openai/whisper-medium'
out_dir = 'whisper_medium_hebrew'
epochs = 10
batch_size = 16

feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id)
tokenizer = WhisperTokenizer.from_pretrained(model_id, language='Hebrew', task='transcribe')
processor = WhisperProcessor.from_pretrained(model_id, language='Hebrew', task='transcribe')

# Load preprocessed dataset from disk
hebrew_dataset = load_from_disk("/home/ammar/hebrew/train/dataset")

model = WhisperForConditionalGeneration.from_pretrained(model_id)
model.generation_config.task = 'transcribe'
model.generation_config.forced_decoder_ids = None

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

training_args = Seq2SeqTrainingArguments(
    output_dir=out_dir,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    warmup_steps=1000,
    bf16=True,
    fp16=False,
    num_train_epochs=epochs,
    eval_strategy='epoch',
    logging_strategy='epoch',
    save_strategy='epoch',
    predict_with_generate=True,
    generation_max_length=225,
    report_to=['tensorboard'],
    load_best_model_at_end=True,
    metric_for_best_model='wer',
    greater_is_better=False,
    dataloader_num_workers=8,
    save_total_limit=2,
    lr_scheduler_type='constant',
    seed=42,
    data_seed=42
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=hebrew_dataset["train"],
    eval_dataset=hebrew_dataset["val"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

trainer.train()
