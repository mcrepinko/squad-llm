from transformers import IntervalStrategy

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
        "max_label_len": 256,
    }
}

TRAINER_ARGUMENTS = {
    "output_dir": "../model/t5_abs_qa",
    "eval_strategy": IntervalStrategy.STEPS,
    # "generation_num_beams": 4,
    # "generation_max_length": 150,
    # "evaluation_strategy": "epoch",
    "eval_steps": 250,
    "metric_for_best_model": "rougeL",
    "load_best_model_at_end": True,
    # "learning_rate": 1e-3,
    # "learning_rate": 1.5e-4,
    "learning_rate": 2e-4,
    # "learning_rate": 3e-4,
    # "learning_rate": 2e-5,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "weight_decay": 0.02,
    # "weight_decay": 0.1,
    "save_total_limit": 3,
    "num_train_epochs": 4,
    # "predict_with_generate": True,
    # "fp16": True,
    "fp16": False,
    "push_to_hub": False,
    "logging_steps": 50,
    # "use_cpu": True, # should be included in latest version, but isn't
    "no_cuda": False # will be depricated
}