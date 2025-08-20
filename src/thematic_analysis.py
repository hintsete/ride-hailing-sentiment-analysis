import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import os

def print_flush(message):
    print(message, flush=True)

def extract_keywords(df, sentiment_type, top_n=20):
    """
    Extract top keywords/n-grams using TF-IDF.
    """
    texts = df[df['sentiment'] == sentiment_type]['cleaned_content'].dropna()
    if texts.empty:
        print_flush(f"No {sentiment_type} reviews for keyword extraction")
        return []
    vectorizer = TfidfVectorizer(max_features=top_n, ngram_range=(1, 2), lowercase=False)
    tfidf_matrix = vectorizer.fit_transform(texts)
    return vectorizer.get_feature_names_out().tolist()

def group_themes(keywords, app_name, sentiment):
    """
    Group keywords into themes.
    """
    themes = {
        'Pricing': ['price', 'cost', 'expensive', 'cheap', 'fare', 'costly', 'affordable', 'ዋጋ'],
        'Safety': ['safe', 'safety', 'dangerous', 'secure', 'risk', 'unsafe'],
        'Usability': ['app', 'interface', 'easy', 'user', 'navigate', 'friendly', 'bug','crash', 'update','buggy','trash'],
        'Service': ['driver', 'service', 'customer', 'support', 'staff', 'ride'],
        'Reliability': ['reliable', 'delay', 'wait', 'time', 'late', 'prompt']
    }
    grouped = {theme: [kw for kw in keywords if any(t in kw.lower() for t in words)] for theme, words in themes.items()}
    print_flush(f"\n{app_name.capitalize()} {sentiment.capitalize()} Themes:")
    for theme, kws in grouped.items():
        if kws:
            print_flush(f"{theme}: {kws}")
    return grouped

def thematic_analysis():
    """
    Perform thematic analysis for both apps.
    """
    print_flush("Starting thematic analysis...")
    for app in ['ride', 'feres']:
        processed_path = os.path.join('data', 'processed', f'{app}_cleaned.csv').replace('\\', '/')
        if not os.path.exists(processed_path):
            print_flush(f"Error: Processed file not found at {processed_path}")
            continue
        df = pd.read_csv(processed_path, encoding='utf-8')
        print_flush(f"\nAnalyzing themes for {app}...")
        for sentiment in ['positive', 'negative']:
            keywords = extract_keywords(df, sentiment)
            if keywords:
                group_themes(keywords, app, sentiment)
        print_flush("-" * 50)

if __name__ == "__main__":
    thematic_analysis()