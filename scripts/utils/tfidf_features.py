# scripts/tfidf_features.py
import numpy as np
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

logger = logging.getLogger(__name__)

def create_tfidf_features(train_df, val_df, max_features=10000):
    """
    Create TF-IDF features from title and paragraphs
    
    Args:
        train_df: DataFrame with clean_title and clean_paragraphs
        val_df: DataFrame with clean_title and clean_paragraphs
        max_features: Maximum number of features to extract
        
    Returns:
        X_train, X_val: Feature matrices for training and validation sets
    """
    logger.info(f"Creating TF-IDF features with max_features={max_features}")
    
    # Combine title and paragraphs with different weights
    # Title is given more weight by repeating it 3 times
    train_texts = [" ".join([row['clean_title']] * 3 + [row['clean_paragraphs']]) 
                  for _, row in train_df.iterrows()]
    
    val_texts = [" ".join([row['clean_title']] * 3 + [row['clean_paragraphs']]) 
                for _, row in val_df.iterrows()]
    
    # Initialize and fit TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=3,           # Minimum document frequency
        max_df=0.9,         # Maximum document frequency (ignore very common terms)
        ngram_range=(1, 2), # Include unigrams and bigrams
        sublinear_tf=True   # Apply sublinear tf scaling (1 + log(tf))
    )
    
    # Fit on training data and transform both train and validation
    logger.info("Fitting TF-IDF vectorizer on training data...")
    X_train = vectorizer.fit_transform(train_texts)
    logger.info(f"Training features shape: {X_train.shape}")
    
    # Transform validation data
    logger.info("Transforming validation data...")
    X_val = vectorizer.transform(val_texts)
    logger.info(f"Validation features shape: {X_val.shape}")
    
    # Save the vectorizer for future use
    os.makedirs('models/tfidf', exist_ok=True)
    joblib.dump(vectorizer, 'models/tfidf/tfidf_vectorizer.pkl')
    logger.info("TF-IDF vectorizer saved to models/tfidf/tfidf_vectorizer.pkl")
    
    return X_train, X_val

def create_separate_tfidf_features(train_df, val_df, max_features=5000):
    """
    Create separate TF-IDF features for title and paragraphs, then combine them
    
    Args:
        train_df: DataFrame with clean_title and clean_paragraphs
        val_df: DataFrame with clean_title and clean_paragraphs
        max_features: Maximum number of features for each vectorizer
        
    Returns:
        X_train, X_val: Feature matrices for training and validation sets
    """
    logger.info(f"Creating separate TF-IDF features for title and paragraphs with max_features={max_features}")
    
    # Extract titles and paragraphs
    train_titles = train_df['clean_title'].tolist()
    train_paragraphs = train_df['clean_paragraphs'].tolist()
    val_titles = val_df['clean_title'].tolist()
    val_paragraphs = val_df['clean_paragraphs'].tolist()
    
    # Initialize and fit TF-IDF vectorizers
    title_vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=2,
        max_df=0.9,
        ngram_range=(1, 2)
    )
    
    para_vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=3,
        max_df=0.9,
        ngram_range=(1, 2)
    )
    
    # Fit and transform title features
    logger.info("Creating title features...")
    X_train_title = title_vectorizer.fit_transform(train_titles)
    X_val_title = title_vectorizer.transform(val_titles)
    
    # Fit and transform paragraph features
    logger.info("Creating paragraph features...")
    X_train_para = para_vectorizer.fit_transform(train_paragraphs)
    X_val_para = para_vectorizer.transform(val_paragraphs)
    
    # Save the vectorizers
    os.makedirs('models/tfidf', exist_ok=True)
    joblib.dump(title_vectorizer, 'models/tfidf/title_vectorizer.pkl')
    joblib.dump(para_vectorizer, 'models/tfidf/para_vectorizer.pkl')
    
    # Horizontally stack features with scipy's hstack
    from scipy.sparse import hstack
    X_train = hstack([X_train_title, X_train_para])
    X_val = hstack([X_val_title, X_val_para])
    
    logger.info(f"Combined training features shape: {X_train.shape}")
    logger.info(f"Combined validation features shape: {X_val.shape}")
    
    return X_train, X_val