import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
import io
from transformers import pipeline

# Initialize sentiment classifier
@st.cache_resource
def load_classifier():
    try:
        return pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')
    except Exception as e:
        st.error(f"Failed to load classifier: {e}")
        st.stop()

classifier = load_classifier()

# Functions
def extract_keywords(texts, top_n=20):
    """
    Extract top keywords/n-grams using TF-IDF.
    """
    if texts.empty:  # Check if Series is empty
        return []
    vectorizer = TfidfVectorizer(max_features=top_n, ngram_range=(1, 2), lowercase=False)
    vectorizer.fit_transform(texts)
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

def classify_sentiment(text):
    """
    Classify sentiment for a single review.
    """
    if not isinstance(text, str) or not text.strip():
        return 'neutral'
    try:
        result = classifier(text[:512])[0]
        score = int(result['label'].split()[0])
        if score >= 4:
            return 'positive'
        elif score == 3:
            return 'neutral'
        return 'negative'
    except:
        return 'neutral'

def plot_to_bytes(fig):
    """
    Convert matplotlib figure to bytes for Streamlit.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return buf.getvalue()

# Streamlit app
st.title('RIDE vs Feres Sentiment & Theme Analyzer')

# Load data
try:
    ride_df = pd.read_csv('data/processed/ride_cleaned.csv', encoding='utf-8')
    feres_df = pd.read_csv('data/processed/feres_cleaned.csv', encoding='utf-8')
    ride_df['app'] = 'RIDE'
    feres_df['app'] = 'Feres'
    combined_df = pd.concat([ride_df, feres_df])
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Sidebar for filtering
st.sidebar.header('Filter Reviews')
app_choice = st.sidebar.selectbox('Select App', ['Both', 'RIDE', 'Feres'])
sentiment_choice = st.sidebar.selectbox('Select Sentiment', ['All', 'positive', 'negative', 'neutral'])
theme_choice = st.sidebar.selectbox('Select Theme', ['All', 'Pricing', 'Safety', 'Usability', 'Service', 'Reliability'])
num_reviews = st.sidebar.slider('Number of Reviews to Display', 1, 50, 10)

# Summary stats
st.sidebar.subheader('Summary Stats')
st.sidebar.write(f"Total Reviews: {len(combined_df)}")
for sentiment in ['positive', 'negative', 'neutral']:
    count = len(combined_df[combined_df['sentiment'] == sentiment])
    st.sidebar.write(f"{sentiment.capitalize()}: {count}")

# Filter data
filtered_df = combined_df
if app_choice != 'Both':
    filtered_df = filtered_df[filtered_df['app'] == app_choice]
if sentiment_choice != 'All':
    filtered_df = filtered_df[filtered_df['sentiment'] == sentiment_choice]
if theme_choice != 'All':
    theme_keywords = {
        'Pricing': ['price', 'cost', 'expensive', 'cheap', 'fare', 'costly', 'affordable', 'ዋጋ', 'ወጪ', 'ውድ', 'ርካሽ'],
        'Safety': ['safe', 'safety', 'dangerous', 'secure', 'risk', 'unsafe', 'ደህንነት', 'አደገኛ', 'አስተማማኝ'],
        'Usability': ['app', 'interface', 'easy', 'user', 'navigate', 'friendly', 'bug', 'መተግበሪያ', 'ቀላል', 'ተጠቃሚ'],
        'Service': ['driver', 'service', 'customer', 'support', 'staff', 'ride', 'ነዳይ', 'አገልግሎት', 'ደንበኛ', 'ድጋፍ'],
        'Reliability': ['reliable', 'delay', 'wait', 'time', 'late', 'prompt', 'ታማኝ', 'መዘግየት', 'ጠብቅ', 'ጊዜ']
    }
    filtered_df = filtered_df[filtered_df['cleaned_content'].apply(lambda x: any(kw in str(x).lower() for kw in theme_keywords[theme_choice]) if isinstance(x, str) else False)]

# Display filtered reviews with keyword highlighting
st.header('Filtered Reviews')
st.write(f'Showing {len(filtered_df)} reviews')
if not filtered_df.empty:
    for index, row in filtered_df[['app', 'content', 'sentiment']].head(num_reviews).iterrows():
        content = row['content']
        if theme_choice != 'All':
            for kw in theme_keywords[theme_choice]:
                content = content.replace(kw, f"**{kw}**")
        st.markdown(f"**{row['app']}**: {content} ({row['sentiment']})")

# Download button for filtered reviews
if not filtered_df.empty:
    csv = filtered_df[['app', 'content', 'sentiment']].to_csv(index=False)
    st.download_button("Download Filtered Reviews", csv, "filtered_reviews.csv", "text/csv")

# Sentiment visualizations
st.header('Sentiment Analysis')
if app_choice == 'Both':
    for app in ['RIDE', 'Feres']:
        df = filtered_df[filtered_df['app'] == app]
        if not df.empty:
            sentiment_counts = df['sentiment'].value_counts()
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#ff9999', '#99ff99'])
            ax.set_title(f'{app} Sentiment Distribution')
            st.image(plot_to_bytes(fig))
            plt.close()
else:
    if not filtered_df.empty:
        sentiment_counts = filtered_df['sentiment'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#ff9999', '#99ff99'])
        ax.set_title(f'{app_choice} Sentiment Distribution')
        st.image(plot_to_bytes(fig))
        plt.close()

# Thematic comparison
st.header('Thematic Comparison')
theme_data = []
for app, df in [('RIDE', ride_df), ('Feres', feres_df)]:
    for sentiment in ['positive', 'negative']:
        texts = df[df['sentiment'] == sentiment]['cleaned_content'].dropna()
        keywords = extract_keywords(texts)
        if keywords:
            grouped = group_themes(keywords)
            for theme, kws in grouped.items():
                if kws:
                    theme_data.append({
                        'App': app,
                        'Sentiment': sentiment,
                        'Theme': theme,
                        'KeywordCount': len(kws)
                    })

if theme_data:
    theme_df = pd.DataFrame(theme_data)
    for sentiment in ['positive', 'negative']:
        sentiment_df = theme_df[theme_df['Sentiment'] == sentiment]
        if not sentiment_df.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=sentiment_df, x='Theme', y='KeywordCount', hue='App', palette='Set2', ax=ax)
            ax.set_title(f'Theme Comparison: RIDE vs Feres ({sentiment.capitalize()} Sentiment)')
            ax.set_xlabel('Theme')
            ax.set_ylabel('Number of Keywords')
            plt.xticks(rotation=45)
            st.image(plot_to_bytes(fig))
            plt.close()

    # Print where each app outperforms
    st.subheader('Where Each App Outperforms')
    for sentiment in ['positive', 'negative']:
        st.write(f"{sentiment.capitalize()} Sentiment:")
        sentiment_df = theme_df[theme_df['Sentiment'] == sentiment]
        if sentiment_df.empty:
            st.write(f"No themes for {sentiment} sentiment")
            continue
        for theme in sentiment_df['Theme'].unique():
            theme_subset = sentiment_df[sentiment_df['Theme'] == theme]
            ride_count = theme_subset[theme_subset['App'] == 'RIDE']['KeywordCount'].sum()
            feres_count = theme_subset[theme_subset['App'] == 'Feres']['KeywordCount'].sum()
            if ride_count > feres_count:
                st.write(f"  {theme}: RIDE outperforms Feres ({ride_count} vs {feres_count} keywords)")
            elif feres_count > ride_count:
                st.write(f"  {theme}: Feres outperforms RIDE ({feres_count} vs {ride_count} keywords)")
            else:
                st.write(f"  {theme}: RIDE and Feres are equal ({ride_count} keywords)")

# Input new review
st.header('Analyze a New Review')
new_review = st.text_area('Enter a review (English or Amharic):')
if st.button('Analyze'):
    if new_review:
        sentiment = classify_sentiment(new_review)
        keywords = extract_keywords([new_review])  # Wrap in list for single review
        themes = group_themes(keywords)
        st.write(f'Sentiment: {sentiment}')
        st.write('Themes:', {k: v for k, v in themes.items() if v})

# Instructions
st.sidebar.markdown("""
### Instructions
- Select an app, sentiment, or theme to filter reviews.
- View sentiment pie charts and thematic bar plots.
- Download filtered reviews as CSV.
- Enter a new review to analyze it in real-time.
""")