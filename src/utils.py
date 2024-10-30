from datasets import load_dataset, DatasetDict, Dataset


def form_inputs(example):
    example["input_text"] = f"question: {example['question']} context: {example['context']}"
    return example


def preprocess_examples(example_batch):
    """Removes unnecesarry data from the answers, multiplies contexts
    and questions for the number of answers.
    """
    no_answers = [
        len(example["text"]) for example in example_batch["answers"]
    ]
    contexts = [
        c for context, no_answer in zip(example_batch["context"], no_answers)
        for c in [context]*no_answer
    ]
    questions = [
        q for question, no_answer in zip(example_batch["question"], no_answers)
        for q in [question]*no_answer
    ]
    answers = [a for answer in example_batch["answers"] for a in answer["text"]]
    # texts = [
    #     f"question: {question} context: {context}" for
    #     question, context in zip(questions, contexts)
    # ]

    return {
        "context": contexts,
        "question": questions,
        # "text": texts,
        "answer": answers
    }


def fetch_dataset(
    dataset_name: str,
    dataset_path_on_disk: str,
    test_size: float,
    remove_columns: list[str]
):
    """Downloads the dataset from HF, 
    splits the train and test, calls preporcessing and saves it to disk.

    Args:
        dataset_name (str): Name of the HF dataset
        test_size (float): Ratio of the test split
        dataset_path_on_disk (str): Local path where to save the dataset
        remove_columns (list[str]): Columns to be removed from dataset
    """
    dataset = load_dataset(dataset_name)
    train_test_ds = dataset["train"].train_test_split(test_size=test_size)
    dataset = DatasetDict({
        "train": train_test_ds["train"],
        "test": train_test_ds["test"],
        "validation": dataset["validation"]
    })
    dataset = dataset.map(
        preprocess_examples,
        batched=True,
        remove_columns=remove_columns
    )
    dataset["train"] = dataset["train"].map(lambda x: form_inputs(x))
    dataset["test"] = dataset["test"].map(lambda x: form_inputs(x))
    dataset["validation"] = dataset["validation"].map(lambda x: form_inputs(x))
    dataset.save_to_disk(dataset_path_on_disk)


def load_dataset_from_disk(
    dataset_path: str,
    train_file_path: str,
    test_file_path: str,
    validation_file_path: str,
) -> Dataset:
    """Loades the dataset from disk.

    Args:
        dataset_path (str): Local path where the dataset is saved
        train_file_path (str): Path to train file inside `dataset_path`
        test_file_path (str): Path to test file inside `dataset_path`
        validation_file_path (str): Path to validaiton file inside `dataset_path`

    Returns:
        Dataset: Loaded dataset.
    """
    return load_dataset(
        dataset_path,
        data_files={
            "train": train_file_path,
            "test": test_file_path,
            "validation": validation_file_path,
        }
    )


def tokenize_dataset(
    dataset: Dataset,
    max_inpt_len: int,
    max_label_len: int,
    tokenizer
):
    def _tokenize(example):
        model_inputs = tokenizer(
            example['input_text'],
            max_length=max_inpt_len,
            padding="max_length",
            truncation=True,
        )
        labels = tokenizer(
            text_target=example["answer"],
            max_length=max_label_len,
            padding="max_length",
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    return dataset.map(_tokenize, batched=True, num_proc=3)


def get_reduced_dataset(
    dataset: DatasetDict, size: int
) -> DatasetDict:
    for key in dataset.keys():
        dataset[key] = dataset[key].select(range(0, size))
    return dataset
