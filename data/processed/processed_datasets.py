from ..raw.load_coco_datasets import load_coco_datasets
from transformers import BlipProcessor
import torch
from typing import Dict, List, Tuple

def process_coco_for_blip(
    batch_size: int = 32,
    max_length: int = 30
) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    """
    COCOデータセットをBLIPモデル用に加工する関数
    
    Args:
        batch_size (int): バッチサイズ
        max_length (int): キャプションの最大長
        
    Returns:
        Tuple[Dict[str, torch.Tensor], List[str]]: 
            - 加工された画像とテキストのバッチ
            - 元のキャプションのリスト
    """
    # COCOデータセットの読み込み
    coco_dataset = load_coco_datasets()
    
    # BLIPプロセッサーの初期化
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    
    # データの加工
    processed_batch = {
        "pixel_values": [],
        "input_ids": [],
        "attention_mask": []
    }
    original_captions = []
    
    # データセットからバッチサイズ分のデータを取得
    for i in range(min(batch_size, len(coco_dataset))):
        sample = coco_dataset[i]
        
        # 画像の加工
        image = sample["image"]
        text = sample["caption"]
        
        # プロセッサーで画像とテキストを処理
        inputs = processor(
            images=image,
            text=text,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
            truncation=True
        )
        
        # バッチに追加
        processed_batch["pixel_values"].append(inputs["pixel_values"])
        processed_batch["input_ids"].append(inputs["input_ids"])
        processed_batch["attention_mask"].append(inputs["attention_mask"])
        original_captions.append(text)
    
    # テンソルに変換
    processed_batch = {
        k: torch.cat(v, dim=0) for k, v in processed_batch.items()
    }
    
    return processed_batch, original_captions






