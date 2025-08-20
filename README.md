# RIDE vs. Feres Sentiment & Theme Analyzer

A **Streamlit** web app to analyze RIDE and Feres reviews, providing **sentiment** and **thematic insights** with support for **English** and **Amharic**.  
Built with **Python**, it uses **BERT** for sentiment analysis and **TF-IDF** for theme extraction, offering an interactive UI for filtering, visualizing, and analyzing reviews.

## 🚀 Features

- **Sentiment Analysis** – Classifies reviews as Positive, Negative, or Neutral.  
- **Thematic Analysis** – Groups keywords into themes (**Pricing, Safety, Usability, Service, Reliability**), including Amharic.  
- **Interactive Filters** – Filter by app, sentiment, or theme with highlighted keywords.  
- **Visualizations** – Pie charts for sentiment distribution; bar plots for theme comparison.    
- **Export** – Download filtered reviews as CSV.  

---

## ⚙️ Setup

### 1. Clone Repository:
git clone https://github.com/hintsete/ride-hailing-sentiment-analysis
cd ride-hailing-sentiment-analysis


### 2. Set Up Virtual Environment:
python -m venv venv
source venv/Scripts/activate   Windows
or source venv/bin/activate   Linux/Mac


### 3. Install Dependencies:
pip install -r requirements.txt


### 4. Run App:
streamlit run app.py

Opens at http://localhost:8501.

📊 Usage

Filter Reviews – Select app (RIDE / Feres / Both), sentiment, or theme (e.g., Pricing) to view reviews with bolded keywords.

View Charts – Explore sentiment pie charts and theme comparison bar plots.

Analyze New Review – Enter a review to instantly view sentiment and theme results.

Download – Export filtered reviews as CSV.


<img width="1873" height="850" alt="image" src="https://github.com/user-attachments/assets/b41c05ab-a755-4456-a6ed-31e7cb0821f8" />
                



License
MIT License
