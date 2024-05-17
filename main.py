from data_loader import get_stock_data
from preprocess_data import preprocess_data
from model import create_model
from train import train_model
from predict import predict_stock_price
from evaluate import evaluate_model
import time

def real_time_learning(ticker, interval='1d', epochs=1, batch_size=1):
    while True:
        data = get_stock_data(ticker, '2020-01-01', '2021-01-01')
        scaled_data, scaler = preprocess_data(data)
        model = create_model()
        train_model(model, scaled_data, epochs, batch_size)
        predicted_price = predict_stock_price(model, data, scaler)
        evaluate_model(data['Close'].values, predicted_price)
        time.sleep(3600)  # Update every hour

if __name__ == "__main__":
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'FB']  # Example tickers
    for ticker in tickers:
        print(f"Training model for {ticker}...")
        real_time_learning(ticker)
