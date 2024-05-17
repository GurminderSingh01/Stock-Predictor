import numpy as np

def predict_stock_price(model, data, scaler):
    model_inputs = data[len(data) - len(data) - 60:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)

    x_test = []
    for i in range(60, len(model_inputs)):
        x_test.append(model_inputs[i-60:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predicted_price = model.predict(x_test)
    predicted_price = scaler.inverse_transform(predicted_price)
    return predicted_price
