#!/usr/bin/env python3
"""
Clickbait Classification System - Main Script

This script provides a command-line interface for training different types of 
clickbait classification models, including logistic regression, BiLSTM, and ModernBERT.
The script handles data loading, preprocessing, feature extraction, model training,
and evaluation.
"""
import os
import logging
import joblib
import argparse
import torch

from scripts.utils.data_loader import load_datasets
from scripts.utils.preprocessor import prepare_dataset
from scripts.utils.tfidf_features import create_tfidf_features, create_separate_tfidf_features
from scripts.utils.ngrams_features import create_ngram_tfidf_features
from scripts.train_logreg import train_evaluate_logreg
from scripts.train_bilstm import train_evaluate_bilstm
from scripts.train_transformer import train_evaluate_modernbert

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_logreg(train_df, val_df, args):
    """
    Train and evaluate a logistic regression model for clickbait classification.
    
    Args:
        train_df: DataFrame containing training data
        val_df: DataFrame containing validation data
        args: Command line arguments
        
    Returns:
        Trained logistic regression model
    """
    logger.info("=== Training Logistic Regression Model ===")
    
    # Create features based on selected feature type
    logger.info(f"Creating features: {args.logreg_features} with max_features={args.logreg_max_features}")
    
    if args.logreg_features == 'ngrams':
        X_train, X_val, vectorizer = create_ngram_tfidf_features(
            train_df, val_df, max_features=args.logreg_max_features
        )
        model_path = f'models/classifier/logreg_ngrams_{args.logreg_max_features}.pkl'
    elif args.logreg_features == 'separate':
        X_train, X_val = create_separate_tfidf_features(
            train_df, val_df, max_features=args.logreg_max_features
        )
        model_path = f'models/classifier/logreg_separate_{args.logreg_max_features}.pkl'
    else:  # default tfidf
        X_train, X_val = create_tfidf_features(
            train_df, val_df, max_features=args.logreg_max_features
        )
        model_path = f'models/classifier/logreg_tfidf_{args.logreg_max_features}.pkl'
    
    # Get labels
    y_train = train_df['label']
    y_val = val_df['label']
    
    # Train and evaluate
    logger.info("Training logistic regression...")
    model = train_evaluate_logreg(X_train, y_train, X_val, y_val)
    
    # Save model
    os.makedirs('models/classifier', exist_ok=True)
    logger.info(f"Saving model to {model_path}...")
    joblib.dump(model, model_path)
    
    return model


def train_bilstm(train_df, val_df, args):
    """
    Train and evaluate a BiLSTM model for clickbait classification.
    
    Args:
        train_df: DataFrame containing training data
        val_df: DataFrame containing validation data
        args: Command line arguments
        
    Returns:
        Trained BiLSTM model and word-to-index mapping
    """
    logger.info("\n=== Training BiLSTM Model ===")
    
    model_dir = 'models/bilstm'
    os.makedirs(model_dir, exist_ok=True)
    
    # Use simpler parameters if quick mode is enabled
    if args.quick:
        bilstm_params = {
            'max_features': 10000,
            'max_len': 50,
            'embed_dim': 300,
            'lstm_units': 64,
            'batch_size': 64,
            'epochs': 5,
            'model_dir': model_dir
        }
        logger.info("Using quick training settings")
    else:
        # Use parameters from args
        bilstm_params = {
            'max_features': args.bilstm_max_features,
            'max_len': args.bilstm_max_len,
            'embed_dim': args.bilstm_embed_dim,
            'lstm_units': args.bilstm_lstm_units,
            'batch_size': args.bilstm_batch_size,
            'epochs': args.bilstm_epochs,
            'model_dir': model_dir
        }
    
    # Train and evaluate BiLSTM
    model = train_evaluate_bilstm(
        train_df=train_df,
        val_df=val_df,
        **bilstm_params
    )
    
    return model


def train_modernbert(train_df, val_df, args):
    """
    Train and evaluate a ModernBERT model for clickbait classification.
    
    Args:
        train_df: DataFrame containing training data
        val_df: DataFrame containing validation data
        args: Command line arguments
        
    Returns:
        Trained ModernBERT model
    """
    logger.info("\n=== Training ModernBERT Model ===")
    
    model_dir = 'models/modernbert'
    os.makedirs(model_dir, exist_ok=True)
    
    # Configure ModernBERT parameters
    modernbert_params = {
        'model_name': args.modernbert_model,
        'output_dir': model_dir,
        'max_length': args.modernbert_max_length,
        'batch_size': args.modernbert_batch_size,
        'learning_rate': args.modernbert_learning_rate,
        'num_epochs': args.modernbert_epochs,
        'from_checkpoint': args.modernbert_from_checkpoint,
        'weight_decay': 0.01,
        'warmup_ratio': 0.05,
        'fp16': torch.cuda.is_available(),
        'gradient_accumulation_steps': 2,  # If memory issues occur
        'logging_steps': 25,
        'eval_steps': 100,
        'save_steps': 100,
        'save_total_limit': 2,
        'quick_mode': args.quick  # Use quick mode if specified
    }
    
    # Train and evaluate ModernBERT
    model = train_evaluate_modernbert(
        train_df=train_df,
        val_df=val_df,
        **modernbert_params
    )
    
    return model


def parse_arguments():
    """
    Parse command line arguments with clear grouping by model type.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Train clickbait classifier',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Main selection argument
    parser.add_argument(
        '--model', 
        type=str, 
        default='logreg', 
        choices=['logreg', 'bilstm', 'modernbert', 'all'],
        help='Model to train'
    )
    
    # Training speed option (note: currently only affects BiLSTM and ModernBERT)
    parser.add_argument(
        '--quick', 
        action='store_true',
        help='Use quick training settings for faster results (only affects BiLSTM and ModernBERT)'
    )
    
    # Create argument groups for each model type
    logreg_group = parser.add_argument_group('Logistic Regression options')
    bilstm_group = parser.add_argument_group('BiLSTM options')
    modernbert_group = parser.add_argument_group('ModernBERT options')
    
    # Logistic Regression options
    logreg_group.add_argument(
        '--logreg-features', 
        type=str, 
        default='tfidf',
        choices=['tfidf', 'separate', 'ngrams'],
        help='Feature type for LogReg'
    )
    
    logreg_group.add_argument(
        '--logreg-max-features', 
        type=int, 
        default=10000,
        help='Maximum features to extract for LogReg'
    )
    
    # BiLSTM options
    bilstm_group.add_argument(
        '--bilstm-max-features', 
        type=int, 
        default=20000,
        help='Vocabulary size for BiLSTM (reduced when --quick is used)'
    )
    
    bilstm_group.add_argument(
        '--bilstm-max-len', 
        type=int, 
        default=100,
        help='Maximum sequence length for BiLSTM (reduced when --quick is used)'
    )
    
    bilstm_group.add_argument(
        '--bilstm-embed-dim', 
        type=int, 
        default=300,
        help='Embedding dimension for BiLSTM'
    )
    
    bilstm_group.add_argument(
        '--bilstm-lstm-units', 
        type=int, 
        default=128,
        help='Number of LSTM units (reduced when --quick is used)'
    )
    
    bilstm_group.add_argument(
        '--bilstm-batch-size', 
        type=int, 
        default=32,
        help='Batch size for BiLSTM training (increased when --quick is used)'
    )
    
    bilstm_group.add_argument(
        '--bilstm-epochs', 
        type=int, 
        default=15,
        help='Number of epochs for BiLSTM training (reduced when --quick is used)'
    )
    
    # ModernBERT options
    modernbert_group.add_argument(
        '--modernbert-model', 
        type=str, 
        default='answerdotai/ModernBERT-base',
        help='ModernBERT model to use'
    )
    
    modernbert_group.add_argument(
        '--modernbert-max-length', 
        type=int, 
        default=128,
        help='Maximum sequence length for ModernBERT'
    )
    
    modernbert_group.add_argument(
        '--modernbert-learning-rate', 
        type=float, 
        default=5e-5,
        help='Learning rate for ModernBERT'
    )
    
    modernbert_group.add_argument(
        '--modernbert-epochs', 
        type=int, 
        default=3,
        help='Number of epochs for ModernBERT (may be reduced when --quick is used)'
    )
    
    modernbert_group.add_argument(
        '--modernbert-batch-size', 
        type=int, 
        default=16,
        help='Batch size for ModernBERT training'
    )

    #TODO: test if it generates the same output
    modernbert_group.add_argument(
        '--modernbert-from-checkpoint', 
        type=str,
        default=None,
        help='Path to load ModernBERT model checkpoint from'
    )
    
    return parser.parse_args()

def main():
    """Main function to orchestrate the training process."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Define paths
    train_path = "data/train.jsonl"
    val_path = "data/validation.jsonl"
    
    # Load and preprocess datasets
    logger.info("Loading and preprocessing datasets...")
    train_df, val_df = load_datasets(train_path, val_path)
    train_df = prepare_dataset(train_df)
    val_df = prepare_dataset(val_df)
    
    models_trained = []
    
    # Train selected models
    if args.model in ['logreg', 'all']:
        logreg_model = train_logreg(train_df, val_df, args)
        models_trained.append('Logistic Regression')
    
    if args.model in ['bilstm', 'all']:
        bilstm_model = train_bilstm(train_df, val_df, args)
        models_trained.append('BiLSTM')
    
    if args.model in ['modernbert', 'all']:
        modernbert_model = train_modernbert(train_df, val_df, args)
        models_trained.append('ModernBERT')
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("Training completed successfully!")
    logger.info(f"Models trained: {', '.join(models_trained)}")
    logger.info("="*50)


if __name__ == "__main__":
    main()