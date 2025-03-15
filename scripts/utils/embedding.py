# scripts/embedding_trainer.py
import numpy as np
from gensim.models import Word2Vec, KeyedVectors
import gensim.downloader as api
from sklearn.cross_decomposition import CCA
from collections import Counter
from typing import List, Dict
import logging
import os
from pathlib import Path
from scipy.linalg import orthogonal_procrustes
import shutil

logger = logging.getLogger(__name__)

def load_generic_embeddings() -> KeyedVectors:
    """Load Word2Vec embeddings with explicit download handling"""
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    embeddings_dir = project_root / "models" / "embeddings"
    cache_path = embeddings_dir / "google_news.bin"

    if cache_path.exists():
        logger.info(f"Loading cached embeddings from project: {cache_path}")
        return KeyedVectors.load(str(cache_path))

    # First time setup
    logger.info("First time setup - downloading embeddings...")
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    # Download using gensim (will use default location temporarily)
    logger.info("Downloading Word2Vec embeddings...")
    model = api.load('word2vec-google-news-300')
    
    # Move to project directory
    logger.info(f"Saving embeddings to project: {cache_path}")
    model.save(str(cache_path))

    # Cleanup gensim download (optional)
    gensim_data = Path.home() / "gensim-data"
    if gensim_data.exists():
        logger.info("Cleaning up temporary download...")
        shutil.rmtree(gensim_data, ignore_errors=True)

    return model

def train_domain_embeddings(texts: List[str], vector_size: int = 300) -> Word2Vec:
    """Train Word2Vec on clickbait corpus"""
    logger.info("Training domain-specific embeddings...")
    
    # Tokenize texts
    tokenized_texts = [text.split() for text in texts]
    
    # Count word frequencies
    word_freq = Counter(word for text in tokenized_texts for word in text)
    
    # Train Word2Vec with higher min_count to reduce vocabulary
    model = Word2Vec(
        sentences=tokenized_texts,
        vector_size=vector_size,
        window=5,
        min_count=5,  # Increased from 1 to 5
        workers=4
    )
    
    return model


def align_and_combine_embeddings(generic_model: KeyedVectors, 
                               domain_model: Word2Vec,
                               max_vocab_size: int = 50000) -> Dict[str, np.ndarray]:
    """Align embeddings using Orthogonal Procrustes"""
    logger.info("Aligning and combining embeddings...")
    
    # Get common vocabulary
    generic_vocab = set(generic_model.key_to_index.keys())
    domain_vocab = set(domain_model.wv.key_to_index.keys())
    common_words = list(generic_vocab.intersection(domain_vocab))
    
    # Use most frequent words
    if len(common_words) > max_vocab_size:
        logger.info(f"Using {max_vocab_size} most frequent common words")
        common_words = common_words[:max_vocab_size]
    
    # Get embedding matrices
    generic_vecs = np.vstack([generic_model[word] for word in common_words])
    domain_vecs = np.vstack([domain_model.wv[word] for word in common_words])
    
    # Compute alignment matrix
    logger.info("Computing orthogonal transformation...")
    R, _ = orthogonal_procrustes(domain_vecs, generic_vecs)
    
    # Align domain embeddings
    domain_aligned = domain_vecs @ R
    
    # Combine embeddings with weighted average
    alpha = 0.5  # Weight parameter
    logger.info(f"Combining embeddings with alpha={alpha}")
    
    combined_embeddings = {}
    for i, word in enumerate(common_words):
        combined_embeddings[word] = (
            alpha * generic_vecs[i] + (1 - alpha) * domain_aligned[i]
        )
    
    return combined_embeddings

def align_and_combine_cca_embeddings(generic_model: KeyedVectors, 
                               domain_model: Word2Vec,
                               max_vocab_size: int = 50000) -> Dict[str, np.ndarray]:
    """Align and combine generic and domain-specific embeddings with optimizations"""
    logger.info("Aligning and combining embeddings...")
    
    # Get common vocabulary (limit size)
    generic_vocab = set(generic_model.key_to_index.keys())
    domain_vocab = set(domain_model.wv.key_to_index.keys())
    common_words = list(generic_vocab.intersection(domain_vocab))
    
    # Filter to most frequent words in domain model
    word_frequencies = {word: domain_model.wv.get_vecattr(word, 'count') 
                       for word in common_words}
    common_words = sorted(word_frequencies.keys(), 
                         key=lambda x: word_frequencies[x], 
                         reverse=True)[:max_vocab_size]
    
    logger.info(f"Using {len(common_words)} most frequent common words")
    
    # Get embeddings for common words
    generic_vecs = np.array([generic_model[word] for word in common_words])
    domain_vecs = np.array([domain_model.wv[word] for word in common_words])
    
    # Apply CCA with reduced components
    n_components = min(300, len(common_words) - 1)
    cca = CCA(
        n_components=n_components,
        max_iter=1000,  # Increased from default 500
        tol=1e-6       # Adjusted tolerance
    )
    
    logger.info("Performing CCA alignment...")
    try:
        generic_aligned, domain_aligned = cca.fit_transform(generic_vecs, domain_vecs)
        logger.info(f"CCA completed with {cca.n_iter_} iterations")
    except Exception as e:
        logger.warning(f"CCA encountered issues: {e}")
        # Fallback to simpler alignment
        generic_aligned = generic_vecs
        domain_aligned = domain_vecs
    
    # Use simple weighted average instead of optimization
    alpha = 0.5  # Equal weight to both embeddings
    
    # Create combined embeddings
    logger.info("Creating combined embeddings...")
    combined_embeddings = {}
    for i, word in enumerate(common_words):
        combined_embeddings[word] = (
            alpha * generic_aligned[i] + (1 - alpha) * domain_aligned[i]
        )
        
        # Add original generic embeddings for words not in common vocabulary
        if len(combined_embeddings) < max_vocab_size:
            for word in generic_vocab:
                if word not in combined_embeddings:
                    combined_embeddings[word] = generic_model[word]
                    if len(combined_embeddings) >= max_vocab_size:
                        break
    
    return combined_embeddings

def save_embeddings(embeddings: Dict[str, np.ndarray], output_dir: str):
    """Save embeddings in binary format only for efficiency"""
    logger.info("Saving embeddings...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as Word2Vec binary format
    wv = KeyedVectors(vector_size=300)
    wv.add_vectors(list(embeddings.keys()), list(embeddings.values()))
    wv.save(os.path.join(output_dir, "combined_embeddings.bin"))
    
    # Save vocabulary size info
    with open(os.path.join(output_dir, "vocab_info.txt"), 'w') as f:
        f.write(f"Vocabulary size: {len(embeddings)}\n")
    
    logger.info(f"Embeddings saved in {output_dir}")