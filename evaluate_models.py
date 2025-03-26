import os
import joblib
import pandas as pd
from sklearn import logger
from sklearn.metrics import classification_report, accuracy_score
from scripts.utils.data_loader import load_datasets
from scripts.utils.preprocessor import prepare_dataset
from scripts.utils.tfidf_features import create_tfidf_features, create_separate_tfidf_features
from scripts.utils.ngrams_features import create_ngram_tfidf_features
from scripts.train_bilstm import evaluate_bilstm
from scripts.train_transformer import evaluate_modernbert

def evaluate_logreg(test_df, args):
    """Evaluate Logistic Regression model"""
    model_path = f'models/classifier/logreg_{args.logreg_features}_{args.logreg_max_features}.pkl'
    model = joblib.load(model_path)
    
    if args.logreg_features == 'ngrams':
        X_test, _, _ = create_ngram_tfidf_features(test_df, test_df, max_features=args.logreg_max_features)
    elif args.logreg_features == 'separate':
        X_test, _ = create_separate_tfidf_features(test_df, test_df, max_features=args.logreg_max_features)
    else:  # default tfidf
        X_test, _ = create_tfidf_features(test_df, test_df, max_features=args.logreg_max_features)
    
    y_test = test_df['label']
    y_pred = model.predict(X_test)
    
    print("Logistic Regression Evaluation:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

def evaluate_models(args):
    """Evaluate all models"""
    # Define paths
    test_path = "data/test.jsonl"
    
    # Load and preprocess datasets
    logger.info("Loading and preprocessing test dataset...")
    test_df = load_datasets(test_path)
    test_df = prepare_dataset(test_df)
    
    # Evaluate selected models
    if args.model in ['logreg', 'all']:
        evaluate_logreg(test_df, args)
    
    if args.model in ['bilstm', 'all']:
        evaluate_bilstm(test_df, args)
    
    if args.model in ['modernbert', 'all']:
        evaluate_modernbert(test_df, args)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate clickbait classifier models')
    parser.add_argument('--model', type=str, default='all', choices=['logreg', 'bilstm', 'modernbert', 'all'], help='Model to evaluate')
    parser.add_argument('--logreg-features', type=str, default='tfidf', choices=['tfidf', 'separate', 'ngrams'], help='Feature type for LogReg')
    parser.add_argument('--logreg-max-features', type=int, default=10000, help='Maximum features to extract for LogReg')
    args = parser.parse_args()
    
    evaluate_models(args)