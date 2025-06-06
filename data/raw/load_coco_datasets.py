from datasets import load_dataset

def load_coco_datasets():
    '''
    Loads the COCO-Caption dataset from the Hugging Face Hub.
    Returns:
        val: The validation set.
        test: The test set.
    '''
    val, test = load_dataset("lmms-lab/COCO-Caption", split=["val", "test"])
    return val, test