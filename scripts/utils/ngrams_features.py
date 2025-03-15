import logging
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

def create_ngram_tfidf_features(train_df, val_df, max_features=10000, ngram_range=(1,3)):
    """
    Extracts TF-IDF features with N-grams (unigrams, bigrams, trigrams).

    Args:
        train_df (DataFrame): Training dataset.
        val_df (DataFrame): Validation dataset.
        max_features (int): Maximum number of TF-IDF features.
        ngram_range (tuple): Range of N-grams to include.

    Returns:
        X_train (sparse matrix): TF-IDF features for training data.
        X_val (sparse matrix): TF-IDF features for validation data.
        vectorizer (TfidfVectorizer): Fitted TF-IDF vectorizer.
    """

    logger.info("Extracting N-gram TF-IDF features...")

    # Convert lists to strings (handles cases where postText or targetParagraphs are stored as lists)
    train_df["postText"] = train_df["postText"].apply(lambda x: " ".join(x) if isinstance(x, list) else str(x))
    train_df["targetParagraphs"] = train_df["targetParagraphs"].apply(lambda x: " ".join(x) if isinstance(x, list) else str(x))

    val_df["postText"] = val_df["postText"].apply(lambda x: " ".join(x) if isinstance(x, list) else str(x))
    val_df["targetParagraphs"] = val_df["targetParagraphs"].apply(lambda x: " ".join(x) if isinstance(x, list) else str(x))

    # Define TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        stop_words="english"
    )

    # Fit TF-IDF on training data and transform both datasets
    X_train = vectorizer.fit_transform(train_df["postText"] + " " + train_df["targetParagraphs"])
    X_val = vectorizer.transform(val_df["postText"] + " " + val_df["targetParagraphs"])

    logger.info(f"N-gram TF-IDF feature extraction completed with {max_features} features.")

    return X_train, X_val, vectorizer
