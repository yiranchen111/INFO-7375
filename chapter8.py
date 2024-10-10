import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np


(x_train_full, y_train_full), (x_test_full, y_test_full) = tf.keras.datasets.cifar10.load_data()


x_data = np.concatenate((x_train_full[:800], x_test_full[:200]))
y_data = np.concatenate((y_train_full[:800], y_test_full[:200]))


x_data = x_data.astype('float32') / 255.0


x_train_proto, x_temp, y_train_proto, y_temp = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
x_val_proto, x_test_proto, y_val_proto, y_test_proto = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)


print(f"Training set size: {len(x_train_proto)}")
print(f"Validation set size: {len(x_val_proto)}")
print(f"Test set size: {len(x_test_proto)}")


unique_train, counts_train = np.unique(y_train_proto, return_counts=True)
unique_val, counts_val = np.unique(y_val_proto, return_counts=True)
unique_test, counts_test = np.unique(y_test_proto, return_counts=True)

print("Training set label distribution:", dict(zip(unique_train.flatten(), counts_train)))
print("Validation set label distribution:", dict(zip(unique_val.flatten(), counts_val)))
print("Test set label distribution:", dict(zip(unique_test.flatten(), counts_test)))