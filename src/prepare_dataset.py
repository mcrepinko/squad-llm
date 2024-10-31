from utils import fetch_dataset, load_dataset_from_disk
from config import DATA

if __name__ == "__main__":
    fetch_dataset(
        DATA["dataset_name"], DATA["dataset_path"],
        DATA["test_size"], DATA["remove_columns"]
    )
    # just to test
    dataset = load_dataset_from_disk(
        DATA["dataset_path"],
        DATA["train_file_path"],
        DATA["test_file_path"],
        DATA["validation_file_path"],
    )
    print(dataset)
    print("example [train]:", dataset["train"][0])
    print("example [test]:", dataset["test"][0])
    print("example [validation]:", dataset["validation"][0])

