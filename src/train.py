import numpy as np
import importlib

from functools import partial
from transformers import (AutoTokenizer, DataCollatorForSeq2Seq,
    AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, pipeline,
    set_seed, T5ForConditionalGeneration, T5Tokenizer, Trainer,
    TrainingArguments, GenerationConfig
)
from transformers.optimization import Adafactor, AdafactorSchedule
from utils import load_dataset_from_disk, tokenize_dataset, get_reduced_dataset
import torch
import evaluate
import config
importlib.reload(config)
from config import DATA, MODEL, SEED, TRAINER_ARGUMENTS

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(pred.strip() for pred in decoded_preds)]
    decoded_labels = ["\n".join(label.strip() for label in decoded_labels)]

    # Calculate ROUGE scores
    result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    
    # Extract and format individual ROUGE metrics
    result = {key: value * 100 for key, value in result.items()}
    return result


USE_GPU = MODEL["use_gpu"]
DEVICE = "cuda:0" if USE_GPU and torch.cuda.is_available() else "cpu"
rouge_metric = evaluate.load("rouge")
set_seed(SEED)

dataset = load_dataset_from_disk(
    DATA["dataset_path"],
    DATA["train_file_path"],
    DATA["test_file_path"],
    DATA["validation_file_path"],
)
dataset = get_reduced_dataset(dataset, 50)

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
    **TRAINER_ARGUMENTS,
    generation_config=GenerationConfig(
        max_length=MODEL["hyperparams"]["max_label_len"],
        pad_token_id=tokenizer.pad_token_id,
        decoder_start_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,

        # repetition_penalty=MODEL["generation_config"]["repetition_penalty"],
        # num_beams=MODEL["generation_config"]["num_beams"],
        # # early_stopping=MODEL["generation_config"]["early_stopping"],
        # top_k=MODEL["generation_config"]["top_k"],
        # # # top_p=MODEL["generation_config"]["top_p"],
        # # do_sample=MODEL["generation_config"]["do_sample"],
    ),
)
print(training_args.generation_config)

trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics,
)

print("-------- EVALUATION BEFORE TRAINING -------- ")
model.eval()
metric = trainer.evaluate()
print(metric)


print("-------- TRAINING... -------- ")
model.train()
trainer.eval_dataset=dataset["validation"]
trainer.train()

print("-------- EVALUATION AFTER TRAINING -------- ")
model.eval()
trainer.eval_dataset=dataset["test"]
metric = trainer.evaluate()
print(metric)
