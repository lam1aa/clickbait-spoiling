"""
This module implements a Logistic Regression model for clickbait classification.
It provides functionality for training, evaluating, and saving Logistic Regression models
with support for pre-trained word embeddings as features.
"""
import os
import json
import numpy as np
import joblib
import logging
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report, precision_recall_fscore_support

logger = logging.getLogger(__name__)

def create_weighted_features(df, embeddings: dict, embedding_size: int = 300) -> np.ndarray:
    """
    Create weighted features from both title and article text.
    
    Args:
        df: DataFrame with clean_title and clean_paragraphs
        embeddings: Dictionary of word embeddings
        embedding_size: Size of word vectors (default 300 for Google News)
        
    Returns:
        np.ndarray: Feature vectors for each document
    """
    logger.info(f"Creating features with embedding size: {embedding_size}")
    
    feature_vectors = []
    missing_words = 0
    total_words = 0
    
    for _, row in df.iterrows():
        # Get title embedding
        title_words = row['clean_title'].split()
        total_words += len(title_words)
        title_vectors = []
        
        for w in title_words:
            if w in embeddings:
                title_vectors.append(embeddings[w])
            else:
                missing_words += 1
                
        title_vec = np.mean(title_vectors, axis=0) if title_vectors else np.zeros(embedding_size)
        
        # Get article embedding
        article_words = row['clean_paragraphs'].split()
        total_words += len(article_words)
        article_vectors = []
        
        for w in article_words:
            if w in embeddings:
                article_vectors.append(embeddings[w])
            else:
                missing_words += 1
                
        article_vec = np.mean(article_vectors, axis=0) if article_vectors else np.zeros(embedding_size)
        
        # Combine with weights (0.7 for title, 0.3 for article)
        combined_vec = (0.7 * title_vec + 0.3 * article_vec)
        feature_vectors.append(combined_vec)
    
    # Report vocabulary coverage
    coverage = (1 - missing_words / total_words) * 100 if total_words > 0 else 0
    logger.info(f"Embeddings coverage: {coverage:.2f}% ({total_words - missing_words}/{total_words} words found)")
    
    return np.vstack(feature_vectors)


def calculate_metrics(true_labels, predictions):
    """
    Calculate various evaluation metrics.
    
    Args:
        true_labels (list): True labels
        predictions (list): Predicted labels
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Calculate precision, recall, and F1 score (macro average)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='macro'
    )
    
    # Calculate balanced accuracy
    balanced_acc = balanced_accuracy_score(true_labels, predictions)
    
    # Calculate accuracy for each class
    class_report = classification_report(true_labels, predictions, output_dict=True)
    
    metrics = {
        "balanced_accuracy": balanced_acc,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "class_metrics": {
            str(cls): {
                "precision": class_report[str(cls)]["precision"],
                "recall": class_report[str(cls)]["recall"],
                "f1-score": class_report[str(cls)]["f1-score"],
                "support": int(class_report[str(cls)]["support"])
            } for cls in sorted(set(true_labels))
        }
    }
    
    return metrics


def save_results(model_name, params, metrics, model_dir='models/logreg'):
    """
    Save training parameters and evaluation results to a JSON file.
    
    Args:
        model_name (str): Name of the model
        params (dict): Dictionary of model parameters
        metrics (dict): Dictionary of evaluation metrics
        model_dir (str): Directory to save results
        
    Returns:
        str: Path to the saved results file
    """
    # Create results directory
    results_dir = 'results/logreg'
    os.makedirs(results_dir, exist_ok=True)
    
    # Create results dictionary
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results = {
        "timestamp": timestamp,
        "model_name": model_name,
        "parameters": params,
        "evaluation_results": metrics
    }
    
    # Generate filename with timestamp
    results_filename = f"{model_name}_results_{timestamp}.json"
    results_path = os.path.join(results_dir, results_filename)
    
    # Save results to file
    logger.info(f"Saving evaluation results and parameters to {results_path}")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
        


def train_evaluate_logreg(X_train, y_train, X_val, y_val, model_dir='models/logreg', model_name='logreg'):
    """
    Train and evaluate logistic regression with balanced metrics.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        model_dir (str): Directory to save model and results
        model_name (str): Name for the model
        
    Returns:
        LogisticRegression: Trained model
    """
    # Ensure model directory exists
    os.makedirs(model_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Initialize model with parameters
    logger.info("Initializing logistic regression...")
    model = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,     # number of max iterations
        C=0.1,             # Add regularization
        solver='lbfgs',    # Specify solver
        random_state=42,
        n_jobs=-1  # Use all available cores
    )
    
    # Train the model
    logger.info("Training logistic regression...")
    model.fit(X_train, y_train)
    
    # Predict on train and validation sets
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    
    # Calculate detailed metrics
    train_metrics = calculate_metrics(y_train, y_pred_train)
    val_metrics = calculate_metrics(y_val, y_pred_val)
    
    # Print detailed performance metrics
    logger.info("\n" + "=" * 50)
    logger.info("TRAINING SET PERFORMANCE:")
    logger.info("=" * 50)
    logger.info(f"Balanced Accuracy: {train_metrics['balanced_accuracy']:.4f}")
    logger.info(f"Macro F1 Score: {train_metrics['macro_f1']:.4f}")
    
    logger.info("\n" + "=" * 50)
    logger.info("VALIDATION SET PERFORMANCE:")
    logger.info("=" * 50)
    logger.info(f"Balanced Accuracy: {val_metrics['balanced_accuracy']:.4f}")
    logger.info(f"Macro Precision: {val_metrics['macro_precision']:.4f}")
    logger.info(f"Macro Recall: {val_metrics['macro_recall']:.4f}")
    logger.info(f"Macro F1: {val_metrics['macro_f1']:.4f}")
    
    # Print classification report in a formatted way
    logger.info("\nCLASSIFICATION REPORT:")
    logger.info("-" * 70)
    logger.info(f"{'Class':<8} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support'}")
    logger.info("-" * 70)
    
    for cls, cls_metrics in val_metrics['class_metrics'].items():
        logger.info(f"{cls:<8} {cls_metrics['precision']:<12.4f} {cls_metrics['recall']:<12.4f} "
                   f"{cls_metrics['f1-score']:<12.4f} {cls_metrics['support']}")
    
    logger.info("-" * 70)
    
    # Save model
    model_file = os.path.join(model_dir, f"{model_name}_model.pkl")
    joblib.dump(model, model_file)
    logger.info(f"Model saved to {model_file}")
    
    # Create parameter dictionary for saving results
    params = {
        "class_weight": "balanced",
        "max_iter": 1000,
        "C": 0.1,
        "solver": "lbfgs",
        "feature_weights": {
            "title_weight": 0.7,
            "article_weight": 0.3
        }
    }
    
    # Add full metrics for saving
    all_metrics = {
        "training": train_metrics,
        "validation": val_metrics
    }
    
    # Save results to JSON
    save_results(model_name, params, all_metrics, model_dir)
    
    return model


def load_logreg_model(model_path):
    """
    Load a trained Logistic Regression model from disk.
    
    Args:
        model_path (str): Path to the saved model file
            
    Returns:
        LogisticRegression: Loaded model
    """
    logger.info(f"Loading logistic regression model from {model_path}")
    model = joblib.load(model_path)
    return model