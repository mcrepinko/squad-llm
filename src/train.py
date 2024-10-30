import numpy as np
import importlib

from functools import partial
from transformers import (AutoTokenizer, DataCollatorForSeq2Seq,
    AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, pipeline,
    set_seed, T5ForConditionalGeneration, T5Tokenizer, Trainer,
    TrainingArguments
)
from transformers.optimization import Adafactor, AdafactorSchedule
from utils import load_dataset_from_disk, tokenize_dataset, get_reduced_dataset
import torch
import evaluate

import config
importlib.reload(config)
from config import DATA, MODEL, SEED, TRAINER_ARGUMENTS

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

USE_GPU = MODEL["use_gpu"]
DEVICE = "cuda:0" if USE_GPU and torch.cuda.is_available() else "cpu"
set_seed(SEED)
metric = evaluate.load("accuracy")

dataset = load_dataset_from_disk(
    DATA["dataset_path"],
    DATA["train_file_path"],
    DATA["test_file_path"],
    DATA["validation_file_path"],
)
dataset = get_reduced_dataset(dataset, 100)
tokenizer = T5Tokenizer.from_pretrained(MODEL["model_name"], legacy=False)
model = T5ForConditionalGeneration.from_pretrained(
    MODEL["model_name"], cache_dir=MODEL["model_cache_path"]
)
model.to(DEVICE)

dataset = tokenize_dataset(
    dataset,
    max_inpt_len=MODEL["hyperparams"]["max_inpt_len"],
    max_label_len=MODEL["hyperparams"]["max_label_len"],
    tokenizer=tokenizer
)

dataset = dataset.remove_columns([
    "context", "question", "answer", "input_text"
])

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=MODEL['model_name']
)

training_args = Seq2SeqTrainingArguments(
    **TRAINER_ARGUMENTS
)

model.eval()
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    compute_metrics=compute_metrics,
)


res = trainer.evaluate()
