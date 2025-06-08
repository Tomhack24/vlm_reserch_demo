from datasets import load_dataset

def load_vqav2_val():
    dataset = load_dataset("HuggingFaceM4/VQAv2", split="validation[:10%]", trust_remote_code=True)
    return dataset