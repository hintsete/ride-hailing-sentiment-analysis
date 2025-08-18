import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import os
import sys

# Ensure output is not buffered
def print_flush(message):
    print(message, flush=True)

# Download required NLTK data
print_flush("Downloading NLTK data...")
nltk.download('punkt_tab', quiet=False)
nltk.download('stopwords', quiet=False)
print_flush("NLTK data download complete")

def clean_text(text):
    """
    Clean the review text: lowercase, remove punctuation, remove English stopwords.
    Preserves Amharic text for multilingual analysis.
    """
    if not isinstance(text, str):
        return ""
    # Lowercase and remove punctuation
    text = re.sub(r'[^\w\s]', '', text.lower())
    # Tokenize and remove English stop words
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    cleaned = [token for token in tokens if token not in stop_words]
    return " ".join(cleaned)

def preprocess_reviews(app_name):
    """
    Load raw reviews, clean, and save processed data.
    """
    raw_path = os.path.join('data', 'raw', f'{app_name}_reviews.json').replace('\\', '/')
    
    # Check current working directory
    print_flush(f"Current working directory: {os.getcwd()}")
    
    # Check if file exists
    if not os.path.exists(raw_path):
        print_flush(f"Error: File not found at {raw_path}")
        return None
    
    print_flush(f"Found file: {raw_path}")
    
    # Check file size
    file_size = os.path.getsize(raw_path)
    print_flush(f"File size: {file_size} bytes")
    if file_size == 0:
        print_flush(f"Error: {raw_path} is empty")
        return None
    
    try:
        # Load JSON
        print_flush(f"Loading raw reviews for {app_name}...")
        df = pd.read_json(raw_path, encoding='utf-8')
        print_flush(f"Loaded {len(df)} raw reviews for {app_name}")
        
        # Check if DataFrame is empty
        if df.empty:
            print_flush(f"Warning: No reviews found in {raw_path}")
            return None
        
        # Check for 'content' column
        if 'content' not in df.columns:
            print_flush(f"Error: 'content' column missing in {raw_path}")
            return None
        
        # Drop missing and duplicate content
        print_flush("Dropping missing and duplicate content...")
        initial_len = len(df)
        df = df.dropna(subset=['content'])
        df = df.drop_duplicates(subset=['content'])
        print_flush(f"After cleaning: {len(df)} reviews (removed {initial_len - len(df)})")
        
        # Check if any reviews remain
        if df.empty:
            print_flush(f"Warning: No reviews remain after cleaning for {app_name}")
            return None
        
        # Clean review content
        print_flush("Cleaning review content...")
        df['cleaned_content'] = df['content'].apply(clean_text)
        
        # Save processed CSV
        processed_path = os.path.join('data', 'processed', f'{app_name}_cleaned.csv').replace('\\', '/')
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        df.to_csv(processed_path, index=False, encoding='utf-8')
        print_flush(f"Saved processed data to {processed_path}")
        return df
    
    except ValueError as ve:
        print_flush(f"JSON parsing error for {app_name}: {ve}")
        return None
    except Exception as e:
        print_flush(f"Unexpected error preprocessing {app_name}: {e}")
        return None

if __name__ == "__main__":
    print_flush("Starting preprocessing script...")
    for app in ['ride', 'feres']:
        print_flush(f"\nProcessing {app}...")
        result = preprocess_reviews(app)
        if result is not None:
            print_flush(f"Successfully processed {app} with {len(result)} reviews")
        else:
            print_flush(f"Failed to process {app}")
        print_flush("-" * 50)