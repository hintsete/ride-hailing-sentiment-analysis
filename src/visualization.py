import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def print_flush(message):
    print(message, flush=True)

def generate_visuals():
    """
    Generate pie charts for each app and a bar plot comparing sentiments.
    """
    print_flush("Starting visualization...")
    os.makedirs('outputs', exist_ok=True)
    
    # Load data
    ride_df = pd.read_csv(os.path.join('data', 'processed', 'ride_cleaned.csv').replace('\\', '/'), encoding='utf-8')
    feres_df = pd.read_csv(os.path.join('data', 'processed', 'feres_cleaned.csv').replace('\\', '/'), encoding='utf-8')
    
    # Filter rows with sentiment labels (since only 100 reviews per app have labels)
    ride_df = ride_df[ride_df['sentiment'].notna()]
    feres_df = feres_df[feres_df['sentiment'].notna()]
    
    # Print summary table of sentiment percentages
    print_flush("\nSentiment Distribution Summary:")
    for app, df in [('RIDE', ride_df), ('Feres', feres_df)]:
        total = len(df)
        if total == 0:
            print_flush(f"No sentiment data for {app}")
            continue
        sentiment_counts = df['sentiment'].value_counts(normalize=True) * 100  # Percentages
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
            colors=['#66b3ff', '#ff9999', '#99ff99']  # Blue, red, green for neutral, negative, positive
        )
        plt.title(f'{app} Sentiment Distribution')
        output_path = os.path.join('outputs', f'{app.lower()}_sentiment_pie.png').replace('\\', '/')
        plt.savefig(output_path)
        plt.close()
        print_flush(f"Saved pie chart for {app} to {output_path}")
    
    # Bar plot for comparison
    combined = pd.concat([ride_df.assign(app='RIDE'), feres_df.assign(app='Feres')])
    plt.figure(figsize=(8, 6))
    sns.countplot(data=combined, x='sentiment', hue='app', palette='Set2')
    plt.title('Sentiment Comparison: RIDE vs Feres')
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Reviews')
    output_path = os.path.join('outputs', 'sentiment_comparison.png').replace('\\', '/')
    plt.savefig(output_path)
    plt.close()
    print_flush(f"Saved sentiment comparison plot to {output_path}")

if __name__ == "__main__":
    generate_visuals()