import numpy as np
import pandas as pd

# Placeholder function for getting predictions — replace with real LSTM and sentiment outputs
def get_stock_data(ticker, start_date, end_date):
    # Simulated data for now
    predicted_price = np.random.randint(1000, 3000)
    sentiment_score = round(np.random.uniform(-1, 1), 2)
    recommendation = "Buy" if sentiment_score > 0.2 else "Hold" if sentiment_score > -0.2 else "Sell"

    return {
        "ticker": ticker,
        "predicted_price": predicted_price,
        "sentiment_score": sentiment_score,
        "recommendation": recommendation
    }

def compare_stocks(selected_stock, start_date, end_date):
    stocks = ["AAPL", "AMZN", "GOOGL", "TSLA", "MSFT"]
    comparison_data = []

    for stock in stocks:
        stock_data = get_stock_data(stock, start_date, end_date)
        comparison_data.append(stock_data)

    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(comparison_data)

    # Highlight the best-performing stock
    best_stock = df.loc[df['predicted_price'].idxmax()]
    insights = f"{best_stock['ticker']} shows the highest predicted growth potential with a price of {best_stock['predicted_price']}."

    return df, insights
