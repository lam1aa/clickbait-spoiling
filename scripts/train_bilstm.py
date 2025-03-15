# scripts/train_bilstm.py
"""
This module implements a Bidirectional LSTM model for clickbait classification.
It provides functionality for training, evaluating, and saving BiLSTM models
with support for pre-trained word embeddings.
"""
import numpy as np
import logging
import os
import json
import joblib
from datetime import datetime
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import classification_report, balanced_accuracy_score, precision_recall_fscore_support

# Import embedding utilities
from scripts.utils.embedding import load_generic_embeddings

logger = logging.getLogger(__name__)


class ClickbaitDataset(Dataset):
    """
    PyTorch Dataset for clickbait classification.
    
    Attributes:
        texts (list): List of preprocessed text samples
        labels (list): List of corresponding labels
        word_to_idx (dict): Mapping from words to indices
        max_len (int): Maximum sequence length
    """
    
    def __init__(self, texts, labels, word_to_idx, max_len=100):
        """
        Initialize the dataset with texts, labels and vocabulary mapping.
        
        Args:
            texts (list): List of preprocessed text samples
            labels (list): List of corresponding labels
            word_to_idx (dict): Mapping from words to indices
            max_len (int): Maximum sequence length to consider
        """
        self.texts = texts
        self.labels = labels
        self.word_to_idx = word_to_idx
        self.max_len = max_len
        
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.texts)
    
    def __getitem__(self, idx):
        """
        Convert a text sample to a tensor of word indices.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            tuple: (tokenized_text, label) as tensors
        """
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Convert words to indices, truncate if too long
        tokens = [self.word_to_idx.get(word, self.word_to_idx['<UNK>']) 
                 for word in text.split()][:self.max_len]
        
        # Convert to tensor
        tokens = torch.tensor(tokens, dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)
        
        return tokens, label


def collate_fn(batch):
    """
    Custom collate function to handle variable length sequences in a batch.
    
    Args:
        batch (list): List of (text, label) tuples
        
    Returns:
        tuple: (padded_texts, labels) as tensors
    """
    texts, labels = zip(*batch)
    
    # Pad sequences to the length of the longest sequence
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    
    return texts_padded, labels


class BiLSTM(nn.Module):
    """
    Bidirectional LSTM model for text classification with pre-trained embeddings support.
    
    Features:
    - Optional pre-trained embeddings
    - Bidirectional LSTM
    - Combined max and average pooling
    - Dropout for regularization
    """
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, 
                 embedding_matrix=None, freeze_embeddings=True, dropout=0.65):
        """
        Initialize the BiLSTM model.
        
        Args:
            vocab_size (int): Size of the vocabulary
            embed_dim (int): Dimension of word embeddings
            hidden_dim (int): Dimension of LSTM hidden state
            num_classes (int): Number of output classes
            embedding_matrix (ndarray, optional): Pre-trained word vectors
            freeze_embeddings (bool): Whether to freeze embeddings during training
            dropout (float): Dropout probability
        """
        super(BiLSTM, self).__init__()
        
        # Initialize embedding layer (with pre-trained embeddings if provided)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
            if freeze_embeddings:
                self.embedding.weight.requires_grad = False
                logger.info("Embeddings are frozen during training")
            else:
                logger.info("Embeddings will be fine-tuned during training")
        
        self.embedding_dropout = nn.Dropout(dropout - 0.1)
        
        # BiLSTM layer
        self.lstm = nn.LSTM(
            embed_dim, 
            hidden_dim, 
            batch_first=True, 
            bidirectional=True,
            num_layers=1
        )
        
        # Dropout after LSTM
        self.lstm_dropout = nn.Dropout(dropout)
        
        # Pooling functions
        self.global_max_pool = lambda x: torch.max(x, dim=1)[0]
        self.global_avg_pool = lambda x: torch.mean(x, dim=1)
        
        # Classifier
        self.fc = nn.Linear(hidden_dim * 4, num_classes)  # *4 for bidirectional + two pooling types
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (tensor): Input tensor of token indices [batch_size, seq_len]
            
        Returns:
            tensor: Output logits [batch_size, num_classes]
        """
        # Create mask for padding (1 for real tokens, 0 for padding)
        mask = (x != 0).float().unsqueeze(-1)
        
        # Embedding with dropout
        embedded = self.embedding(x)
        embedded = self.embedding_dropout(embedded)
        
        # LSTM
        lstm_out, _ = self.lstm(embedded)
        
        # Apply mask to LSTM outputs to ignore padding
        lstm_out = lstm_out * mask
        
        # Apply dropout
        lstm_out = self.lstm_dropout(lstm_out)
        
        # Combine max pooling and average pooling
        max_pooled = self.global_max_pool(lstm_out)
        avg_pooled = self.global_avg_pool(lstm_out)
        
        # Concatenate the pooled representations
        pooled = torch.cat([max_pooled, avg_pooled], dim=1)
        
        # Output layer
        out = self.fc(pooled)
        
        return out


def build_vocabulary(texts, min_freq=5, max_vocab=15000):
    """
    Build vocabulary from training texts.
    
    Args:
        texts (list): List of text samples
        min_freq (int): Minimum frequency for a word to be included
        max_vocab (int): Maximum vocabulary size
        
    Returns:
        tuple: (word_to_idx, idx_to_word) dictionaries
    """
    logger.info("Building vocabulary...")
    
    word_counts = Counter()
    for text in texts:
        word_counts.update(text.split())
    
    # Filter by frequency
    word_counts = {word: count for word, count in word_counts.items() 
                   if count >= min_freq}
    
    # Sort by frequency
    words = sorted(word_counts.keys(), key=lambda x: word_counts[x], reverse=True)
    
    # Limit vocabulary size
    if len(words) > max_vocab - 2:  # leave room for PAD and UNK
        words = words[:max_vocab - 2]
    
    # Create word-to-index mapping
    word_to_idx = {'<PAD>': 0, '<UNK>': 1}
    for i, word in enumerate(words):
        word_to_idx[word] = i + 2
    
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    
    logger.info(f"Vocabulary size: {len(word_to_idx)}")
    
    return word_to_idx, idx_to_word


def create_embedding_matrix(word_to_idx, pretrained_embeddings, embed_dim=300):
    """
    Create embedding matrix from pre-trained embeddings.
    
    Args:
        word_to_idx (dict): Mapping from words to indices
        pretrained_embeddings (dict): Pre-trained word vectors
        embed_dim (int): Dimension of embeddings
        
    Returns:
        ndarray: Embedding matrix of shape [vocab_size, embed_dim]
    """
    logger.info("Creating embedding matrix from pre-trained embeddings...")
    
    # Initialize embedding matrix with random values
    vocab_size = len(word_to_idx)
    embedding_matrix = np.random.uniform(-0.25, 0.25, (vocab_size, embed_dim))
    
    # Set padding token embedding to zeros
    embedding_matrix[0] = np.zeros(embed_dim)
    
    # Count how many words are found in pre-trained embeddings
    found_count = 0
    
    # Fill embedding matrix with pre-trained embeddings when available
    for word, idx in word_to_idx.items():
        if word in pretrained_embeddings:
            embedding_matrix[idx] = pretrained_embeddings[word]
            found_count += 1
    
    coverage = found_count / vocab_size * 100
    logger.info(f"Found {found_count} words in pre-trained embeddings out of {vocab_size} ({coverage:.2f}%)")
    
    return embedding_matrix


def preprocess_text_for_bilstm(train_df, val_df, max_vocab=15000, max_len=100, use_pretrained=True):
    """
    Preprocess text data for BiLSTM model with pre-trained embeddings option.
    
    Args:
        train_df (DataFrame): Training data with clean_title and clean_paragraphs columns
        val_df (DataFrame): Validation data with clean_title and clean_paragraphs columns
        max_vocab (int): Maximum vocabulary size
        max_len (int): Maximum sequence length
        use_pretrained (bool): Whether to use pre-trained embeddings
        
    Returns:
        tuple: (train_dataset, val_dataset, word_to_idx, idx_to_word, 
                embedding_matrix, class_weights)
    """
    logger.info("Preprocessing text data for BiLSTM...")
    
    # Weight title more by repeating it twice
    train_texts = [f"{row['clean_title']} {row['clean_title']} [SEP] {row['clean_paragraphs'][:500]}" 
                  for _, row in train_df.iterrows()]
    val_texts = [f"{row['clean_title']} {row['clean_title']} [SEP] {row['clean_paragraphs'][:500]}" 
                for _, row in val_df.iterrows()]
    
    # Build vocabulary
    word_to_idx, idx_to_word = build_vocabulary(train_texts, min_freq=5, max_vocab=max_vocab)
    
    # Load pre-trained embeddings if requested
    embedding_matrix = None
    if use_pretrained:
        try:
            logger.info("Loading pre-trained word embeddings...")
            pretrained_embeddings = load_generic_embeddings()
            embedding_matrix = create_embedding_matrix(word_to_idx, pretrained_embeddings)
        except Exception as e:
            logger.warning(f"Failed to load pre-trained embeddings: {e}")
            logger.warning("Continuing with random embeddings")
    
    # Get labels
    y_train = train_df['label'].values
    y_val = val_df['label'].values
    
    # Create PyTorch datasets
    train_dataset = ClickbaitDataset(train_texts, y_train, word_to_idx, max_len)
    val_dataset = ClickbaitDataset(val_texts, y_val, word_to_idx, max_len)
    
    # Calculate class weights for handling imbalance
    class_counts = Counter(y_train)
    num_samples = len(y_train)
    class_weights = torch.tensor(
        [num_samples / (len(class_counts) * count) for _, count in sorted(class_counts.items())],
        dtype=torch.float
    )
    
    return train_dataset, val_dataset, word_to_idx, idx_to_word, embedding_matrix, class_weights


def train_epoch(model, dataloader, criterion, optimizer, device, clip_value=0.5):
    """
    Train model for one epoch with gradient clipping.
    
    Args:
        model (nn.Module): The BiLSTM model
        dataloader (DataLoader): Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on (cuda/cpu)
        clip_value (float): Gradient clipping value
        
    Returns:
        tuple: (average_loss, accuracy) for the epoch
    """
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    
    for texts, labels in dataloader:
        texts, labels = texts.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(texts)
        
        # Compute loss
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        
        optimizer.step()
        
        # Track metrics
        epoch_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return epoch_loss / len(dataloader), correct / total


def evaluate(model, dataloader, criterion, device):
    """
    Evaluate model on validation data.
    
    Args:
        model (nn.Module): The BiLSTM model
        dataloader (DataLoader): Validation data loader
        criterion: Loss function
        device: Device to evaluate on (cuda/cpu)
        
    Returns:
        tuple: (average_loss, balanced_accuracy, predictions, true_labels)
    """
    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for texts, labels in dataloader:
            texts, labels = texts.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(texts)
            
            # Compute loss
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            # Get predictions
            _, predicted = torch.max(outputs, 1)
            
            # Store for metrics
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate balanced accuracy (better for imbalanced datasets)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    
    return val_loss / len(dataloader), balanced_acc, all_preds, all_labels


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


def save_results(model_name, params, metrics, model_dir):
    """
    Save training parameters and evaluation results to a JSON file.
    
    Args:
        model_name (str): Name of the model
        params (dict): Dictionary of model parameters
        metrics (dict): Dictionary of evaluation metrics
        model_dir (str): Directory to save results
    """
    # Create results directory
    results_dir = 'results/bilstm'
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
    results_filename = f"bilstm_results_{timestamp}.json"
    results_path = os.path.join(results_dir, results_filename)
    
    # Save results to file
    logger.info(f"Saving evaluation results and parameters to {results_path}")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)


def train_evaluate_bilstm(train_df, val_df, **kwargs):
    """
    Train and evaluate BiLSTM model for clickbait classification with optional pre-trained embeddings.
    
    Args:
        train_df (DataFrame): Training data with clean_title, clean_paragraphs, and label columns
        val_df (DataFrame): Validation data with clean_title, clean_paragraphs, and label columns
        **kwargs: Additional parameters including:
            - max_features (int): Maximum vocabulary size
            - max_len (int): Maximum sequence length
            - embed_dim (int): Dimension of word embeddings
            - lstm_units (int): Dimension of LSTM hidden state
            - dropout_rate (float): Dropout probability
            - batch_size (int): Training batch size
            - epochs (int): Number of training epochs
            - learning_rate (float): Learning rate
            - model_dir (str): Directory to save model
            - patience (int): Early stopping patience
            - weight_decay (float): L2 regularization parameter
            - use_pretrained (bool): Whether to use pre-trained embeddings
            - freeze_embeddings (bool): Whether to freeze embeddings during training
            - model_name (str): Name for the model
        
    Returns:
        tuple: (model, word_to_idx, metrics) - Trained model, vocabulary mapping, and evaluation metrics
    """
    # Extract parameters with defaults
    max_vocab = kwargs.get('max_features', 15000)
    max_len = kwargs.get('max_len', 100)
    embed_dim = kwargs.get('embed_dim', 300)
    hidden_dim = kwargs.get('lstm_units', 64)
    dropout_rate = kwargs.get('dropout_rate', 0.65)
    batch_size = kwargs.get('batch_size', 32)
    epochs = kwargs.get('epochs', 20)
    learning_rate = kwargs.get('learning_rate', 0.0003)
    model_dir = kwargs.get('model_dir', 'models/bilstm')
    patience = kwargs.get('patience', 6)
    weight_decay = kwargs.get('weight_decay', 8e-4)
    use_pretrained = kwargs.get('use_pretrained', True)
    freeze_embeddings = kwargs.get('freeze_embeddings', True)
    model_name = kwargs.get('model_name', 'bilstm')
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Create model directory
    os.makedirs(model_dir, exist_ok=True)
    
    # Set device (GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Preprocess data
    train_dataset, val_dataset, word_to_idx, idx_to_word, embedding_matrix, class_weights = preprocess_text_for_bilstm(
        train_df, val_df, max_vocab=max_vocab, max_len=max_len, use_pretrained=use_pretrained
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    
    # Initialize model
    num_classes = len(Counter(train_df['label']))
    vocab_size = len(word_to_idx)
    
    model = BiLSTM(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        embedding_matrix=embedding_matrix,
        freeze_embeddings=freeze_embeddings,
        dropout=dropout_rate
    )
    model = model.to(device)
    
    # Print model summary
    logger.info(f"Model architecture:\n{model}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Loss function with class weights for imbalance
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer with weight decay for regularization
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler to reduce LR when performance plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True, min_lr=1e-6
    )
    
    # Training loop with early stopping
    best_val_acc = 0
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    # Store training history
    training_history = []
    
    logger.info(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Train for one epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, clip_value=0.5
        )
        train_losses.append(train_loss)
        
        # Evaluate on validation set
        val_loss, val_acc, val_preds, val_labels = evaluate(
            model, val_loader, criterion, device
        )
        val_losses.append(val_loss)
        
        # Update learning rate based on validation performance
        scheduler.step(val_acc)
        
        logger.info(f"Epoch {epoch+1}/{epochs} - "
                   f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                   f"Val Loss: {val_loss:.4f}, Val Balanced Acc: {val_acc:.4f}")
        
        # Store epoch results
        epoch_results = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        }
        training_history.append(epoch_results)
        
        # Check if model improved
        improved = False
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            improved = True
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            improved = True
        
        if improved:
            best_epoch = epoch
            patience_counter = 0
            best_val_preds = val_preds
            best_val_labels = val_labels
            
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'word_to_idx': word_to_idx,
                'idx_to_word': idx_to_word,
                'params': {
                    'vocab_size': vocab_size,
                    'embed_dim': embed_dim,
                    'hidden_dim': hidden_dim,
                    'num_classes': num_classes,
                    'max_len': max_len,
                    'dropout': dropout_rate
                }
            }, os.path.join(model_dir, 'bilstm_model.pt'))
            
            logger.info(f"Saved best model at epoch {epoch+1}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Additional early stopping for overfitting detection
        if epoch > 5 and (train_losses[-1] < 0.3 and val_losses[-1] > 1.5):
            logger.info(f"Stopping due to diverging train/validation loss (overfitting)")
            break
    
    # Final evaluation with best model
    logger.info(f"Training completed. Best validation balanced accuracy: {best_val_acc:.4f} at epoch {best_epoch+1}")
    
    # Load best model
    checkpoint = torch.load(os.path.join(model_dir, 'bilstm_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Calculate detailed metrics
    metrics = calculate_metrics(best_val_labels, best_val_preds)
    
    # Print detailed performance metrics in a more readable format
    logger.info("\n" + "=" * 50)
    logger.info("FINAL VALIDATION PERFORMANCE:")
    logger.info(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    logger.info(f"Macro Precision: {metrics['macro_precision']:.4f}")
    logger.info(f"Macro Recall: {metrics['macro_recall']:.4f}")
    logger.info(f"Macro F1: {metrics['macro_f1']:.4f}")
    
    
    # Print classification report in a formatted way
    logger.info("-" * 70)
    logger.info(f"{'Class':<8} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support'}")
    logger.info("-" * 70)
    
    for cls, cls_metrics in metrics['class_metrics'].items():
        logger.info(f"{cls:<8} {cls_metrics['precision']:<12.4f} {cls_metrics['recall']:<12.4f} "
                   f"{cls_metrics['f1-score']:<12.4f} {cls_metrics['support']}")
    
    logger.info("-" * 70)
    
    # Save vocabulary for later use
    joblib.dump(word_to_idx, os.path.join(model_dir, 'word_to_idx.pkl'))
    
    # Create parameter dictionary for saving results
    params = {
        "vocab_size": vocab_size,
        "max_len": max_len,
        "embed_dim": embed_dim,
        "hidden_dim": hidden_dim,
        "dropout_rate": dropout_rate,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "epochs_trained": best_epoch + 1,
        "use_pretrained": use_pretrained,
        "freeze_embeddings": freeze_embeddings
    }
    
    # Add training history to metrics
    metrics["training_history"] = training_history
    
    # Save results to JSON
    save_results(model_name, params, metrics, model_dir)
    
    return model


def load_bilstm_model(model_path, word_to_idx_path=None):
    """
    Load a trained BiLSTM model.
    
    Args:
        model_path (str): Path to the saved model file
        word_to_idx_path (str, optional): Path to vocabulary file
            
    Returns:
        tuple: (model, word_to_idx) - Loaded model and vocabulary
    """
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    params = checkpoint['params']
    
    # Initialize model with saved parameters
    model = BiLSTM(
        vocab_size=params['vocab_size'],
        embed_dim=params['embed_dim'],
        hidden_dim=params['hidden_dim'],
        num_classes=params['num_classes'],
        dropout=params['dropout']
    )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Get word_to_idx mapping
    if 'word_to_idx' in checkpoint:
        word_to_idx = checkpoint['word_to_idx']
    elif word_to_idx_path:
        word_to_idx = joblib.load(word_to_idx_path)
    else:
        raise ValueError("word_to_idx not found in model checkpoint and no path provided")
    
    return model, word_to_idx