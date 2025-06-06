from datasets import load_dataset

def load_coco_datasets():
    val, test = load_dataset("lmms-lab/COCO-Caption", split=["val", "test"])
    return val, test