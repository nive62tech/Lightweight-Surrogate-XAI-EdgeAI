import tensorflow as tf
import numpy as np
import pandas as pd

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="models/cnn_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load dataset
data = np.load("data/mnist_processed.npz")
x_val = data["x_test"]
y_val = data["y_test"]

cnn_preds = []

for i in range(len(x_val)):
    input_data = np.expand_dims(x_val[i], axis=0).astype(input_details[0]['dtype'])
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    pred = np.argmax(output)
    cnn_preds.append(pred)

df = pd.DataFrame({"cnn_prediction": cnn_preds})
df.to_csv("results/cnn_predictions.csv", index=False)

print("CNN predictions saved.")