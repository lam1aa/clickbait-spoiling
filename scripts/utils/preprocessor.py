# scripts/preprocessor.py
import re
import pandas as pd
from typing import List
import logging

logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    if not isinstance(text, str):
        return ""
    
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s]', ' ', text)
    text = ' '.join(text.split())
    
    return text

def combine_paragraphs(paragraphs: List[str]) -> str:
    """Combine multiple paragraphs into single text"""
    if not paragraphs:
        return ""
    
    return ' '.join([clean_text(p) for p in paragraphs])

def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare dataset with all preprocessing steps"""
    logger.info("Preprocessing dataset...")
    
    # Clean text fields
    df['clean_title'] = df['postText'].apply(
        lambda x: clean_text(x[0] if isinstance(x, list) else x)
    )
    df['clean_paragraphs'] = df['targetParagraphs'].apply(combine_paragraphs)
    
    # Convert tags to numerical labels
    tag_to_idx = {'phrase': 0, 'passage': 1, 'multi': 2}
    df['label'] = df['tags'].apply(lambda x: tag_to_idx[x[0]])
    
    # Print statistics
    logger.info("Label distribution:")
    for tag, idx in tag_to_idx.items():
        count = len(df[df['label'] == idx])
        logger.info(f"{tag}: {count}")
    
    return df