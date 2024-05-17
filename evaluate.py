import matplotlib.pyplot as plt

def evaluate_model(actual_price, predicted_price):
    plt.plot(actual_price, color='black', label='Actual Stock Price')
    plt.plot(predicted_price, color='green', label='Predicted Stock Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()
