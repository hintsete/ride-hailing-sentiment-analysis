import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
        print_flush(f"No {sentiment_type} reviews for keyword extraction in {df['app'].iloc[0] if 'app' in df else 'unknown'}")
        return []
    vectorizer = TfidfVectorizer(max_features=top_n, ngram_range=(1, 2), lowercase=False)
    tfidf_matrix = vectorizer.fit_transform(texts)
    return vectorizer.get_feature_names_out().tolist()

def group_themes(keywords):
    """
    Group keywords into themes, including Amharic equivalents.
    """
    themes = {
        'Pricing': ['price', 'cost', 'expensive', 'cheap', 'fare', 'costly', 'affordable', 'ዋጋ', 'ወጪ', 'ውድ', 'ርካሽ'],
        'Safety': ['safe', 'safety', 'dangerous', 'secure', 'risk', 'unsafe', 'ደህንነት', 'አደገኛ', 'አስተማማኝ'],
        'Usability': ['app', 'interface', 'easy', 'user', 'navigate', 'friendly', 'bug', 'መተግበሪያ', 'ቀላል', 'ተጠቃሚ'],
        'Service': ['driver', 'service', 'customer', 'support', 'staff', 'ride', 'ነዳይ', 'አገልግሎት', 'ደንበኛ', 'ድጋፍ'],
        'Reliability': ['reliable', 'delay', 'wait', 'time', 'late', 'prompt', 'ታማኝ', 'መዘግየት', 'ጠብቅ', 'ጊዜ']
    }
    grouped = {theme: [kw for kw in keywords if any(t in kw.lower() for t in words)] for theme, words in themes.items()}
    return grouped

def generate_visuals():
    """
    Generate sentiment and thematic visualizations for both apps.
    """
    print_flush("Starting visualization...")
    os.makedirs('outputs', exist_ok=True)
    
    # Load data and add 'app' column
    ride_df = pd.read_csv(os.path.join('data', 'processed', 'ride_cleaned.csv').replace('\\', '/'), encoding='utf-8')
    feres_df = pd.read_csv(os.path.join('data', 'processed', 'feres_cleaned.csv').replace('\\', '/'), encoding='utf-8')
    ride_df['app'] = 'RIDE'
    feres_df['app'] = 'Feres'
    
    # Filter rows with sentiment labels
    ride_df = ride_df[ride_df['sentiment'].notna()]
    feres_df = feres_df[feres_df['sentiment'].notna()]
    
    # Print sentiment distribution summary
    print_flush("\nSentiment Distribution Summary:")
    for app, df in [('RIDE', ride_df), ('Feres', feres_df)]:
        total = len(df)
        if total == 0:
            print_flush(f"No sentiment data for {app}")
            continue
        sentiment_counts = df['sentiment'].value_counts(normalize=True) * 100
        print_flush(f"{app}:")
        for sentiment, percentage in sentiment_counts.items():
            print_flush(f"  {sentiment.capitalize()}: {percentage:.2f}%")
    
    # Pie charts for each app
    for app, df in [('RIDE', ride_df), ('Feres', feres_df)]:
        if df.empty:
            print_flush(f"Skipping pie chart for {app} (no data)")
            continue
        sentiment_counts = df['sentiment'].value_counts()
        plt.figure(figsize=(6, 6))
        plt.pie(
            sentiment_counts,
            labels=sentiment_counts.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=['#66b3ff', '#ff9999', '#99ff99']  # Blue for neutral, red for negative, green for positive
        )
        plt.title(f'{app} Sentiment Distribution')
        output_path = os.path.join('outputs', f'{app.lower()}_sentiment_pie.png').replace('\\', '/')
        plt.savefig(output_path)
        plt.close()
        print_flush(f"Saved pie chart for {app} to {output_path}")
    
    # Bar plot for sentiment comparison
    combined = pd.concat([ride_df, feres_df])
    plt.figure(figsize=(8, 6))
    sns.countplot(data=combined, x='sentiment', hue='app', palette='Set2')
    plt.title('Sentiment Comparison: RIDE vs Feres')
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Reviews')
    output_path = os.path.join('outputs', 'sentiment_comparison.png').replace('\\', '/')
    plt.savefig(output_path)
    plt.close()
    print_flush(f"Saved sentiment comparison plot to {output_path}")
    
    # Thematic analysis and side-by-side bar plots for theme frequencies
    print_flush("\nGenerating thematic visualizations...")
    theme_data = []
    for app, df in [('RIDE', ride_df), ('Feres', feres_df)]:
        for sentiment in ['positive', 'negative']:
            keywords = extract_keywords(df, sentiment)
            if not keywords:
                continue
            grouped = group_themes(keywords)
            
            # Print grouped themes (for reference)
            print_flush(f"\n{app} {sentiment.capitalize()} Themes:")
            for theme, kws in grouped.items():
                if kws:
                    print_flush(f"{theme}: {kws}")
            
            # Collect theme frequencies for plotting
            for theme, kws in grouped.items():
                if kws:
                    theme_data.append({
                        'App': app,
                        'Sentiment': sentiment,
                        'Theme': theme,
                        'KeywordCount': len(kws)
                    })
    
    # Compare where RIDE beats Feres and vice versa
    print_flush("\nThematic Comparison: Where Each App Outperforms")
    theme_df = pd.DataFrame(theme_data)
    for sentiment in ['positive', 'negative']:
        print_flush(f"\n{sentiment.capitalize()} Sentiment:")
        sentiment_df = theme_df[theme_df['Sentiment'] == sentiment]
        if sentiment_df.empty:
            print_flush(f"No themes for {sentiment} sentiment")
            continue
        for theme in sentiment_df['Theme'].unique():
            theme_subset = sentiment_df[sentiment_df['Theme'] == theme]
            ride_count = theme_subset[theme_subset['App'] == 'RIDE']['KeywordCount'].sum()
            feres_count = theme_subset[theme_subset['App'] == 'Feres']['KeywordCount'].sum()
            if ride_count > feres_count:
                print_flush(f"  {theme}: RIDE outperforms Feres ({ride_count} vs {feres_count} keywords)")
            elif feres_count > ride_count:
                print_flush(f"  {theme}: Feres outperforms RIDE ({feres_count} vs {ride_count} keywords)")
            else:
                print_flush(f"  {theme}: RIDE and Feres are equal ({ride_count} keywords)")
    
    # Create side-by-side bar plots for theme frequencies
    if theme_data:
        for sentiment in ['positive', 'negative']:
            sentiment_df = theme_df[theme_df['Sentiment'] == sentiment]
            if sentiment_df.empty:
                print_flush(f"No themes for {sentiment} sentiment")
                continue
            plt.figure(figsize=(10, 6))
            sns.barplot(data=sentiment_df, x='Theme', y='KeywordCount', hue='App', palette='Set2')
            plt.title(f'Theme Comparison: RIDE vs Feres ({sentiment.capitalize()} Sentiment)')
            plt.xlabel('Theme')
            plt.ylabel('Number of Keywords')
            plt.xticks(rotation=45)
            output_path = os.path.join('outputs', f'theme_comparison_{sentiment}.png').replace('\\', '/')
            plt.savefig(output_path)
            plt.close()
            print_flush(f"Saved theme comparison bar plot for {sentiment} sentiment to {output_path}")

if __name__ == "__main__":
    generate_visuals()