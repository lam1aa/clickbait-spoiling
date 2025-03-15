Example Commands:

```bash
# Train LogReg with default TF-IDF (default max-features is 10000)
python main.py

# Train LogReg with N-grams features
python main.py --features ngrams

# Train LogReg with N-grams features and max-features
python main.py --features ngrams --max-features 15000

# Train LogReg with separate TF-IDF features
python main.py --features separate

# Train LogReg with separate TF-IDF features and max-features
python main.py --features separate --max-features 8000

# Train BiLSTM with quick settings (change the quick parameters from main if you want)
python main.py --model bilstm --quick

# Train both models with ngrams features for LogReg
python main.py --model both --features ngrams
```


# ModernBERT Integration Guide

This guide explains how to use the newly added ModernBERT model for clickbait classification in this project.

## Overview

ModernBERT is a refresh of the traditional encoder architecture (like BERT and RoBERTa) with modern architectural improvements including:

- **Rotary Positional Embeddings** supporting sequences up to 8192 tokens
- **Unpadding** to ensure no compute is wasted on padding tokens
- **GeGLU** layers replacing original MLP layers
- **Alternating Attention** with sliding windows
- **Flash Attention** for faster processing

This model is well-suited for text classification tasks like clickbait detection and should provide better accuracy than the traditional LogReg and BiLSTM models.

## Training ModernBERT

### Basic Training

To train the ModernBERT model with default settings:

```bash
python main.py --model modernbert
```

### Advanced Options

The following parameters can be customized for ModernBERT training:

```bash
python main.py --model modernbert \
  --modernbert-model answerdotai/ModernBERT-base \  # Pre-trained model to use
  --max-length 128 \                              # Maximum sequence length
  --learning-rate 5e-5 \                          # Learning rate
  --num-epochs 3                                  # Number of training epochs
```

### Quick Training Mode

For faster training (useful for testing):

```bash
python main.py --model modernbert --quick
```

### Training All Models

To train all models (LogReg, BiLSTM, and ModernBERT):

```bash
python main.py --model all
```

## Model Files

After training, the ModernBERT model will be saved to:

```
models/modernbert/best/
```

This directory contains:
- Model weights and configuration
- Tokenizer files
- Training logs

## Performance Comparison

ModernBERT generally outperforms traditional models in accuracy and F1 score, especially for complex classification tasks, at the cost of slightly slower inference time and higher resource usage.

| Model      | Training Time | Inference Time |
|------------|---------------|----------------|
| LogReg     | Fast          | Very Fast      |
| BiLSTM     | Medium        | Fast           |
| ModernBERT | Slow          | Medium         |

## Requirements

The ModernBERT integration requires the following additional packages:

- transformers>=4.38.0
- torch>=2.0.0

Make sure to install these dependencies before using ModernBERT.

## Troubleshooting

If training is too slow:
- Use the `--quick` flag
- Reduce `--num-epochs`
- Try a smaller model variant like "answerdotai/ModernBERT-small" if available