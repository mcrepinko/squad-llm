"""Config file.
"""
SEED = 420

DATA = {
    "dataset_name": "rajpurkar/squad",
    "dataset_path": "../data/squad",
    "train_file_path": "train/data-00000-of-00001.arrow",
    "test_file_path": "test/data-00000-of-00001.arrow",
    "validation_file_path": "validation/data-00000-of-00001.arrow",
    "test_size": 0.12,
    "remove_columns": ["id", "title", "answers"]
}

MODEL = {
    "use_gpu": True,
    "model_name": "google-t5/t5-small",
    "model_path": "../model",
    "model_cache_path": "../model/cache",
    "hyperparams": {
        "max_inpt_len": 256,
        "max_label_len": 32,
    },
    "generation_config": {
        "num_beams": 4,
        "early_stopping": True,
        "top_k": 50,
        "top_p": 0.92,
        "do_sample": True,
        "repetition_penalty": 1.2
    }
}

TRAINER_ARGUMENTS = {
    "output_dir": "../model/t5_abs_qa",
    "eval_strategy":"epoch",
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 2,
    "weight_decay": 0.01,
    "save_total_limit": 3,
    "optim": "adafactor",
    "learning_rate": 1e-3,
    "num_train_epochs": 6,
    "predict_with_generate": True,
    "fp16": True,
    "predict_with_generate": True,
    "metric_for_best_model": "rouge1",
    "report_to": "tensorboard",
    "gradient_accumulation_steps": 4,
    "gradient_checkpointing": True,
}