from datasets import load_dataset

class LoadDatasets:
    def __init__(self):
        self.dataset = load_dataset("json", data_files="data/raw/data.json")

    def load_dataset(self):
        return self.datasets