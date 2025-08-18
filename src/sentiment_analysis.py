import pandas as pd
from transformers import pipeline
import os
import sys
from tqdm import tqdm  # For progress bar

# Ensure output is not buffered
def print_flush(message):
    print(message, flush=True)

# Initialize sentiment classifier once
print_flush("Initializing sentiment classifier...")
try:
    classifier = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')
    print_flush("Classifier initialized successfully")
except Exception as e:
    print_flush(f"Error initializing classifier: {e}")
    sys.exit(1)

def classify_sentiment(text):
    """
    Classify sentiment using Hugging Face multilingual model.
    """
    if not isinstance(text, str) or not text.strip():
        return 'neutral'
    try:
        result = classifier(text[:512])[0]  # Truncate to 512 tokens
        score = int(result['label'].split()[0])  # Extract star rating (1-5)
        if score >= 4:
            return 'positive'
        elif score == 3:
            return 'neutral'
        return 'negative'
    except Exception as e:
        print_flush(f"Error classifying text '{text[:50]}...': {e}")
        return 'neutral'

def analyze_sentiment(app_name, sample_size=None):
    """
    Apply sentiment analysis to cleaned reviews.
    """
    processed_path = os.path.join('data', 'processed', f'{app_name}_cleaned.csv').replace('\\', '/')
    if not os.path.exists(processed_path):
        print_flush(f"Error: Processed file not found at {processed_path}")
        return None
    
    try:
        df = pd.read_csv(processed_path, encoding='utf-8')
        print_flush(f"Loaded {len(df)} reviews for sentiment analysis of {app_name}")
        
        if df.empty:
            print_flush(f"Warning: No reviews to analyze for {app_name}")
            return None
        
        # Sample reviews for testing (optional)
        # if sample_size is not None:
        #     df = df.sample(n=min(sample_size, len(df)), random_state=42)
        #     print_flush(f"Sampled {len(df)} reviews for {app_name}")
        
        # Apply sentiment analysis with progress bar
        print_flush(f"Classifying sentiments for {app_name}...")
        df['sentiment'] = [classify_sentiment(text) for text in tqdm(df['cleaned_content'], desc=f"Processing {app_name}")]
        
        # Save results
        df.to_csv(processed_path, index=False, encoding='utf-8')
        print_flush(f"Saved sentiment results for {app_name} to {processed_path}")
        return df
    except Exception as e:
        print_flush(f"Error analyzing {app_name}: {e}")
        return None

if __name__ == "__main__":
    print_flush("Starting sentiment analysis...")
    for app in ['ride', 'feres']:
        print_flush(f"\nAnalyzing {app}...")
        result = analyze_sentiment(app, sample_size=100)  # Sample 100 reviews for testing
        if result is not None:
            print_flush(f"Completed sentiment analysis for {app} with {len(result)} reviews")
        else:
            print_flush(f"Failed to process {app}")
        print_flush("-" * 50)

