# scripts/train_transformer.py
"""
This module provides functionality for training and evaluating ModernBERT
models on clickbait classification tasks.
"""
import os
import logging
import json
from datetime import datetime

import torch
from torch.utils.data import Dataset
from sklearn.metrics import classification_report, balanced_accuracy_score
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)

logger = logging.getLogger(__name__)


class ClickbaitDataset(Dataset):
    """Dataset for fine-tuning ModernBERT on clickbait classification.
    
    This dataset class handles the preparation of input data for the ModernBERT model,
    including tokenization and formatting of the combined title and text.
    """
    
    def __init__(self, texts, titles, labels, tokenizer, max_length=128):
        """Initialize the dataset with clickbait data.
        
        Args:
            texts (list): List of content text strings
            titles (list): List of title text strings
            labels (list): List of classification labels
            tokenizer: HuggingFace tokenizer instance
            max_length (int, optional): Maximum sequence length for tokenization. Defaults to 128.
        """
        self.texts = texts
        self.titles = titles
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.labels)
    
    def __getitem__(self, idx):
        """Get a single sample from the dataset.
        
        The method combines the title (repeated three times for emphasis) with the text,
        tokenizes the combined content, and returns the tokenized representation.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            dict: Dictionary containing input_ids, attention_mask, and labels
        """
        # Combine title and text with special emphasis on the title (repeated 3 times)
        combined_text = f"{self.titles[idx]} {self.titles[idx]} {self.titles[idx]} [SEP] {self.texts[idx]}"
        
        # Tokenize the combined text
        encoding = self.tokenizer(
            combined_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Convert to dict and remove batch dimension added by tokenizer
        item = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }
        
        return item


def compute_metrics(pred):
    """Compute evaluation metrics for the model.
    
    Args:
        pred: Prediction object containing predictions and label_ids
        
    Returns:
        dict: Dictionary of evaluation metrics including balanced accuracy,
              precision, recall, F1 score, and per-class F1 scores
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    # Calculate balanced accuracy
    balanced_acc = balanced_accuracy_score(labels, preds)
    
    # Get full classification report as dict
    report = classification_report(labels, preds, output_dict=True)
    
    # Extract metrics we want to track
    metrics = {
        'balanced_accuracy': balanced_acc,
        'precision_weighted': report['weighted avg']['precision'],
        'recall_weighted': report['weighted avg']['recall'],
        'f1_weighted': report['weighted avg']['f1-score']
    }
    
    # Add per-class metrics
    for label, metrics_dict in report.items():
        if isinstance(metrics_dict, dict) and label not in ['micro avg', 'macro avg', 'weighted avg']:
            metrics[f'f1_class_{label}'] = metrics_dict['f1-score']
    
    return metrics


def prepare_modernbert_datasets(train_df, val_df, model_name="answerdotai/ModernBERT-base", max_length=128):
    """Prepare datasets for ModernBERT fine-tuning.
    
    Args:
        train_df (DataFrame): Training data with clean_paragraphs, clean_title, and label columns
        val_df (DataFrame): Validation data with clean_paragraphs, clean_title, and label columns
        model_name (str, optional): Name of the ModernBERT model to use. Defaults to "answerdotai/ModernBERT-base".
        max_length (int, optional): Maximum sequence length. Defaults to 128.
        
    Returns:
        tuple: Tuple containing (train_dataset, val_dataset, tokenizer)
    """
    logger.info(f"Loading ModernBERT tokenizer from {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Prepare training data
    train_texts = train_df['clean_paragraphs'].tolist()
    train_titles = train_df['clean_title'].tolist()
    train_labels = train_df['label'].tolist()
    
    # Prepare validation data
    val_texts = val_df['clean_paragraphs'].tolist()
    val_titles = val_df['clean_title'].tolist()
    val_labels = val_df['label'].tolist()
    
    # Create datasets
    train_dataset = ClickbaitDataset(
        texts=train_texts,
        titles=train_titles,
        labels=train_labels,
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    val_dataset = ClickbaitDataset(
        texts=val_texts,
        titles=val_titles,
        labels=val_labels,
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    return train_dataset, val_dataset, tokenizer


def save_results(model_name, max_length, num_epochs, batch_size, learning_rate, weight_decay, warmup_ratio, eval_results):
    """Save training parameters and evaluation results to a JSON file.
    
    Args:
        model_name (str): Name of the model used
        max_length (int): Maximum sequence length used
        num_epochs (int): Number of training epochs
        batch_size (int): Training batch size
        learning_rate (float): Learning rate used
        weight_decay (float): Weight decay parameter
        warmup_ratio (float): Warmup ratio parameter
        eval_results (dict): Dictionary of evaluation results
    """
    # Create a results directory if it doesn't exist
    results_dir = 'results/modernbert'
    os.makedirs(results_dir, exist_ok=True)
    
    # Create results dictionary
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "parameters": {
            "model_name": model_name,
            "max_length": max_length,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio
        },
        "evaluation_results": {k: float(v) for k, v in eval_results.items()}
    }
    
    # Generate filename with timestamp
    timestamp = results["timestamp"]
    results_filename = f"eval_results_{timestamp}.json"
    results_path = os.path.join(results_dir, results_filename)
    
    # Save results to file
    logger.info(f"Saving evaluation results and parameters to {results_path}")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)


def train_evaluate_modernbert(train_df, val_df, **kwargs):
    """Train and evaluate ModernBERT for clickbait classification.
    
    Args:
        train_df (DataFrame): Training data with clean_paragraphs, clean_title, and label columns
        val_df (DataFrame): Validation data with clean_paragraphs, clean_title, and label columns
        **kwargs: Additional parameters including:
            - model_name (str): Name of the ModernBERT model to use
            - output_dir (str): Directory to save model outputs
            - max_length (int): Maximum sequence length
            - batch_size (int): Training batch size
            - learning_rate (float): Learning rate
            - num_epochs (int): Number of training epochs
            - weight_decay (float): Weight decay parameter
            - warmup_ratio (float): Warmup ratio parameter
            - fp16 (bool): Whether to use mixed precision training
            - from_checkpoint (str): Path to checkpoint to resume from
            - gradient_accumulation_steps (int): Number of steps for gradient accumulation
            - max_grad_norm (float): Maximum gradient norm for clipping
            - quick_mode (bool): Whether to use quick mode with reduced settings
    
    Returns:
        model: Trained ModernBERT model
    """
    # Extract parameters with defaults
    model_name = kwargs.get('model_name', 'answerdotai/ModernBERT-large')
    output_dir = kwargs.get('output_dir', 'models/modernbert')
    max_length = kwargs.get('max_length', 128)
    batch_size = kwargs.get('batch_size', 16)
    learning_rate = kwargs.get('learning_rate', 5e-5)
    num_epochs = kwargs.get('num_epochs', 3)
    weight_decay = kwargs.get('weight_decay', 0.01)
    warmup_ratio = kwargs.get('warmup_ratio', 0.1)
    logging_steps = kwargs.get('logging_steps', 50)
    eval_steps = kwargs.get('eval_steps', 100)
    save_steps = kwargs.get('save_steps', 100)
    save_total_limit = kwargs.get('save_total_limit', 2)
    fp16 = kwargs.get('fp16', False)
    from_checkpoint = kwargs.get('from_checkpoint', None)
    gradient_accumulation_steps = kwargs.get('gradient_accumulation_steps', 1)
    max_grad_norm = kwargs.get('max_grad_norm', 1.0)
    quick_mode = kwargs.get('quick_mode', False)
    
    # Check for CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # If in quick mode, use shorter training
    if quick_mode:
        logger.info("Using quick training mode with reduced settings")
        num_epochs = 1
        max_length = 64
        batch_size = min(batch_size * 2, 32)
        logging_steps = 10
        eval_steps = 50
        save_steps = 50
    
    # Prepare datasets
    train_dataset, val_dataset, tokenizer = prepare_modernbert_datasets(
        train_df=train_df, 
        val_df=val_df,
        model_name=model_name,
        max_length=max_length
    )
    
    # Get number of labels
    num_labels = len(train_df['label'].unique())
    
    # Create model directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load pre-trained model with classification head
    if from_checkpoint:
        logger.info(f"Loading model from checkpoint: {from_checkpoint}")
        model = AutoModelForSequenceClassification.from_pretrained(
            from_checkpoint,
            num_labels=num_labels,
            problem_type="single_label_classification"
        )
    else:
        logger.info(f"Loading ModernBERT model from {model_name} with {num_labels} labels")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            problem_type="single_label_classification"
        )
    
    # Create Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        logging_dir=f"{output_dir}/logs",
        logging_steps=logging_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="balanced_accuracy",
        greater_is_better=True,
        fp16=fp16,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=max_grad_norm,
        report_to="none",  # Disable wandb, tensorboard, etc.
    )
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train the model
    logger.info("Starting ModernBERT training...")
    trainer.train()
    
    # Evaluate the model
    logger.info("Evaluating ModernBERT model...")
    eval_results = trainer.evaluate()
    
    # Log results
    logger.info("Evaluation results:")
    for key, value in eval_results.items():
        logger.info(f"  {key}: {value:.4f}")
    
    # Save model and tokenizer
    model_save_path = f"{output_dir}/best"
    logger.info(f"Saving model to {model_save_path}")
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    # Save evaluation results and parameters to a file
    save_results(
        model_name=model_name, 
        max_length=max_length, 
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        eval_results=eval_results
    )
    
    return model