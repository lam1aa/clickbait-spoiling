# scripts/data_loader.py
import json
import logging
from typing import Tuple
import pandas as pd

logger = logging.getLogger(__name__)

def load_jsonl(file_path: str) -> pd.DataFrame:
    """Load JSONL file into pandas DataFrame"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

def load_datasets(train_path: str, val_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load both training and validation datasets"""
    logger.info("Loading datasets...")
    train_df = load_jsonl(train_path)
    val_df = load_jsonl(val_path)
    logger.info(f"Loaded {len(train_df)} training and {len(val_df)} validation samples")
    return train_df, val_df