import requests
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Pre-trained sentiment analysis pipeline from Hugging Face
sentiment_pipeline = pipeline("sentiment-analysis")

# VADER for financial text sentiment (optional)
vader = SentimentIntensityAnalyzer()

# News API setup
NEWS_API_KEY = "YOUR_NEWSAPI_KEY"
NEWS_API_URL = "https://newsapi.org/v2/everything"

# Fetch financial news related to the stock
def fetch_news(stock_name):
    try:
        params = {
            "q": stock_name,
            "apiKey": "91320d29db69404193ce5597fa9e9479",
            "sortBy": "relevancy",
            "language": "en",
            "pageSize": 10
        }
        response = requests.get(NEWS_API_URL, params=params)
        response.raise_for_status()
        articles = response.json().get("articles", [])
        return [article["title"] + " " + article.get("description", "") for article in articles if article["title"]]
    except Exception as e:
        print(f"Error fetching news: {e}")
        return []

# Analyze sentiment using Hugging Face’s model
def analyze_sentiment(texts):
    try:
        results = sentiment_pipeline(texts)
        sentiments = [1 if result['label'] == 'POSITIVE' else -1 for result in results]
        return sum(sentiments) / len(sentiments) if sentiments else 0
    except Exception as e:
        print(f"Error analyzing sentiment: {e}")
        return 0

# Optional: Combine VADER sentiment
def vader_sentiment(texts):
    try:
        scores = [vader.polarity_scores(text)['compound'] for text in texts]
        return sum(scores) / len(scores) if scores else 0
    except Exception as e:
        print(f"Error with VADER sentiment: {e}")
        return 0

# Overall sentiment analysis
def get_stock_sentiment(stock_name):
    news_texts = fetch_news(stock_name)
    hf_score = analyze_sentiment(news_texts)
    vader_score = vader_sentiment(news_texts)
    combined_score = (hf_score + vader_score) / 2
    sentiment = "Positive" if combined_score > 0.2 else "Negative" if combined_score < -0.2 else "Neutral"
    return sentiment, combined_score
