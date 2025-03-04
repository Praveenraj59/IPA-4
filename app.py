from flask import Flask, render_template, request
import yfinance as yf
from lstm_model import predict_stock_price
from sentiment_analysis import get_stock_sentiment
import plotly.graph_objs as go
import json
from plotly.utils import PlotlyJSONEncoder

app = Flask(__name__)

# Available stocks for dropdown
STOCKS = {
    "AAPL": "Apple",
    "AMZN": "Amazon",
    "GOOGL": "Google",
    "TSLA": "Tesla",
    "MSFT": "Microsoft"
}

# Home page
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/planning_to_invest", methods=["GET", "POST"])
def planning_to_invest():
    if request.method == "POST":
        try:
            ticker = request.form["ticker"]
            stock_name = STOCKS.get(ticker)
            investment_amount = float(request.form["investment_amount"])
            review_frequency = request.form["review_frequency"]
            risk_tolerance = request.form["risk_tolerance"]

            if not stock_name:
                return render_template("error.html", message="Invalid stock selection.")

            # Predict stock price for the selected stock
            predicted_price, actual_prices, predicted_prices = predict_stock_price(ticker)

            if actual_prices is None or actual_prices.empty:
                return render_template("error.html", message=f"No data available for {stock_name}. Please try again later.")

            current_price = float(actual_prices.iloc[-1].iloc[0])

            if predicted_prices is None or len(predicted_prices) == 0:
                return render_template("error.html", message="Prediction data is not available. Please try again later.")

            # Get sentiment analysis
            sentiment, sentiment_score = get_stock_sentiment(stock_name)

            sentiment_message = ""
            if sentiment_score > 0.2:
                sentiment_message = f"The market outlook for {stock_name} is optimistic, suggesting potential growth ahead."
            elif sentiment_score < -0.2:
                sentiment_message = f"The market sentiment for {stock_name} is cautious, indicating possible risks."
            else:
                sentiment_message = f"The market sentiment for {stock_name} is stable, with no strong signals in either direction."

            # Predicted price and potential return calculation
            predicted_price = round(predicted_price, 2)
            potential_return = round(investment_amount * (predicted_price / current_price), 2)

            if predicted_price > current_price:
                price_message = f"{stock_name}’s predicted price is ₹{predicted_price}, indicating a potential upward trend."
            elif predicted_price < current_price:
                price_message = f"{stock_name}’s predicted price is ₹{predicted_price}, suggesting a possible decline."
            else:
                price_message = f"{stock_name}’s predicted price is ₹{predicted_price}, showing stability."

            # 🚨 NEW RECOMMENDATION LOGIC 🚨
            recommendation = "Wait and Observe"
            if predicted_price > current_price:
                if risk_tolerance == "High":
                    recommendation = "Strong Buy"
                elif risk_tolerance == "Medium":
                    recommendation = "Buy"
                elif risk_tolerance == "Low":
                    recommendation = "Cautious Buy"

            # Comparative analysis for other stocks
            comparison_data = []
            for ticker_symbol, name in STOCKS.items():
                try:
                    pred_price, act_prices, pred_prices = predict_stock_price(ticker_symbol)
                    if act_prices is not None and not act_prices.empty and pred_prices is not None and len(pred_prices) > 0:
                        current = float(act_prices.iloc[-1].iloc[0])
                        growth_potential = round((pred_price - current) / current * 100, 2)
                        comparison_data.append((name, growth_potential))
                except Exception as e:
                    print(f"Could not fetch data for {name}: {str(e)}")

            # Find the best-performing stock
            best_stock = max(comparison_data, key=lambda x: x[1], default=None)

            # If another stock shows higher growth, recommend it
            if best_stock and best_stock[0] != stock_name and best_stock[1] > (predicted_price - current_price) / current_price * 100:
                recommendation = f"Consider investing in {best_stock[0]} instead — it shows better growth potential ({best_stock[1]}%)"

            # Insights combining all factors
            insights = f"{sentiment_message} {price_message} Based on your risk tolerance ({risk_tolerance}) and review frequency ({review_frequency}), we suggest: {recommendation}. Estimated return on ₹{investment_amount} investment: ₹{potential_return}."

            # Visualization
            graph = go.Figure()
            graph.add_trace(go.Scatter(x=actual_prices.index, y=actual_prices.iloc[:, 0], mode='lines', name='Actual Price'))
            graph.add_trace(go.Scatter(x=actual_prices.index, y=predicted_prices, mode='lines', name='Predicted Price'))
            graph.update_layout(title=f"{stock_name} Price Prediction", xaxis_title="Date", yaxis_title="Price (INR)")

            graph_json = json.dumps(graph, cls=PlotlyJSONEncoder)

            # Comparative analysis graph
            comparison_graph = go.Figure()
            comparison_graph.add_trace(go.Bar(
                x=[item[0] for item in comparison_data],
                y=[item[1] for item in comparison_data],
                marker=dict(color='blue')
            ))
            comparison_graph.update_layout(
                title="📊 Growth Potential Comparison of Selected Stocks",
                xaxis_title="Stock",
                yaxis_title="Growth Potential (%)",
                yaxis=dict(tickformat=".2f"),
                plot_bgcolor='rgba(240,240,240,0.9)'
            )
            comparison_graph_json = json.dumps(comparison_graph, cls=PlotlyJSONEncoder)

            return render_template("result.html", stock_name=stock_name, ticker=ticker,
                                   predicted_price=predicted_price,
                                   recommendation=recommendation,
                                   insights=insights,
                                   graph_json=graph_json,
                                   comparison_graph=comparison_graph_json)

        except Exception as e:
            return render_template("error.html", message=f"An error occurred: {str(e)}")

    return render_template("planning_to_invest.html", stocks=STOCKS)

import plotly.graph_objects as go
import json

@app.route("/already_invested", methods=["GET", "POST"])
def already_invested():
    if request.method == "POST":
        try:
            # Get form data
            ticker = request.form["ticker"]
            stock_name = STOCKS.get(ticker)
            investment_amount = float(request.form["investment_amount"])
            purchase_price = float(request.form["purchase_price"])
            risk_tolerance = request.form["risk_tolerance"]

            if not stock_name:
                return render_template("error.html", message="Invalid stock selection.")

            # Calculate number of shares (volume)
            volume = investment_amount / purchase_price

            # Predict stock price
            predicted_price, actual_prices, predicted_prices = predict_stock_price(ticker)

            # Check if actual_prices is empty
            if actual_prices is None or actual_prices.empty:
                return render_template("error.html", message=f"No data available for {stock_name}. Please try again later.")

            # Get the most recent actual price
            current_price = float(actual_prices.iloc[-1].iloc[0])

            # Check if predicted_prices array is empty
            if predicted_prices is None or len(predicted_prices) == 0:
                return render_template("error.html", message="Prediction data is not available. Please try again later.")

            # Get sentiment analysis
            sentiment, sentiment_score = get_stock_sentiment(stock_name)

            # User-friendly sentiment message
            if sentiment_score > 0.2:
                sentiment_message = f"The market outlook for {stock_name} is optimistic, suggesting potential growth ahead."
            elif sentiment_score < -0.2:
                sentiment_message = f"The market sentiment for {stock_name} is cautious, indicating possible risks."
            else:
                sentiment_message = f"The market sentiment for {stock_name} is stable, with no strong signals in either direction."

            # Current portfolio value and gain/loss calculation
            current_value = round(volume * current_price, 2)
            gain_loss = round(current_value - investment_amount, 2)

            # Price message
            predicted_price = round(predicted_price, 2)
            if predicted_price > current_price:
                price_message = f"{stock_name}’s predicted price is ₹{predicted_price}, indicating potential growth."
            elif predicted_price < current_price:
                price_message = f"{stock_name}’s predicted price is ₹{predicted_price}, suggesting a possible decline."
            else:
                price_message = f"{stock_name}’s predicted price is ₹{predicted_price}, showing stability."

            # Improved price change percentage calculation
            price_change_percentage = (predicted_price - current_price) / current_price * 100

            # Refined risk-based recommendation logic
            if abs(price_change_percentage) <= 2:  # Prices are very close
                recommendation = "Hold"
            elif predicted_price > current_price:
                if risk_tolerance == "High":
                    recommendation = "Strong Buy"
                elif risk_tolerance == "Medium":
                    recommendation = "Consider Buying More"
                else:
                    recommendation = "Hold"
            else:  # predicted_price < current_price
                if price_change_percentage < -20 and risk_tolerance == "Low":
                    recommendation = "Strongly Consider Reducing Exposure"
                elif price_change_percentage < -15 and risk_tolerance == "Medium":
                    recommendation = "Consider Selling"
                elif price_change_percentage < -10 and risk_tolerance == "High":
                    recommendation = "Reassess and Monitor Closely"
                else:
                    recommendation = "Hold and Review"

            # More meaningful insights based on recommendation
            if "Buy" in recommendation:
                insights = f"{sentiment_message} {price_message} The current market conditions and projected growth suggest adding to your position could be beneficial. Current portfolio value: ₹{current_value} (Gain/Loss: ₹{gain_loss})."
            elif "Sell" in recommendation or "Reducing Exposure" in recommendation:
                insights = f"{sentiment_message} {price_message} Given the projected decline, it might be wise to reconsider your position. Current portfolio value: ₹{current_value} (Gain/Loss: ₹{gain_loss})."
            else:
                insights = f"{sentiment_message} {price_message} A cautious approach is advised. Hold and monitor the market. Current portfolio value: ₹{current_value} (Gain/Loss: ₹{gain_loss})."

            # Price prediction line graph
            graph = go.Figure()
            graph.add_trace(go.Scatter(x=actual_prices.index, y=actual_prices.iloc[:, 0], mode='lines', name='Actual Price'))
            graph.add_trace(go.Scatter(x=actual_prices.index, y=predicted_prices, mode='lines', name='Predicted Price'))
            graph.update_layout(title=f"{stock_name} Price Analysis", xaxis_title="Date", yaxis_title="Price (INR)")

            graph_json = json.dumps(graph, cls=PlotlyJSONEncoder)

           # Gain/Loss bar chart (only if data exists)
            bar_chart_json = None
            if investment_amount and current_value:
                try:
                    bar_chart = go.Figure()
                    bar_chart.add_trace(go.Bar(x=['Investment Amount', 'Current Value'], 
                                            y=[investment_amount, current_value], 
                                            marker_color=['blue', 'green']))
                    bar_chart.update_layout(title='Investment Performance: Gain/Loss', 
                                            yaxis_title='Amount (INR)', 
                                            xaxis_title='Metrics')

                    bar_chart_json = json.dumps(bar_chart, cls=PlotlyJSONEncoder)
                except Exception as e:
                    print(f"Error creating bar chart: {str(e)}")


            return render_template("result.html", stock_name=stock_name, ticker=ticker,
                                   predicted_price=predicted_price,
                                   recommendation=recommendation,
                                   insights=insights,
                                   graph_json=graph_json,
                                   bar_chart_json=bar_chart_json)

        except Exception as e:
            return render_template("error.html", message=f"An error occurred: {str(e)}")

    return render_template("already_invested.html", stocks=STOCKS)

# Error page
@app.errorhandler(500)
def internal_error(e):
    return render_template("error.html", message="Something went wrong. Please try again later.")

if __name__ == "__main__":
    app.run(debug=True)
